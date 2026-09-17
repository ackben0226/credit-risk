"""
Bureau aggregation: bureau + bureau_balance → per-applicant features.

Home Credit's bureau tables describe the applicant's external credit
history. Structure:

    bureau                (one row per SK_ID_BUREAU per SK_ID_CURR)
        │
        └── bureau_balance (one row per SK_ID_BUREAU per month)

This module performs a two-level aggregation:

    Level 1: bureau_balance → per SK_ID_BUREAU
             (max DPD status, counts of each status, etc.)

    Level 2: bureau + bureau_balance aggregates → per SK_ID_CURR
             (counts, sums, means, maxes of credit amounts, overdue
              amounts, DPD history)

The output is one row per applicant, with a stable schema and a
presence indicator (has_bureau_history) so downstream stages can
distinguish "no bureau records" from "bureau records exist but all
values are null".

Null handling
-------------
- Bureau records that exist but have no bureau_balance rows produce
  null bureau_balance-derived features for that applicant.
- Applicants with no bureau records produce null for all bureau_* and
  bb_* features except the presence and count indicators (which are 0).
- No imputation is performed here. Nulls are informative and preserved.

Reference
---------
Data Quality Report §6.2 and §6.5 — bureau_balance coverage of bureau
is ~45%, and orphan rates across child tables cluster in the 37K–48K
range. This module preserves those characteristics rather than
flattening them.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


logger = logging.getLogger("credit_risk.features.aggregations.bureau")


# ---------------------------------------------------------------------------
# bureau_balance STATUS ordinal encoding
# ---------------------------------------------------------------------------
#
# STATUS is a categorical column with values:
#   0    — no overdue
#   1    — 1-30 days past due
#   2    — 31-60 days past due
#   3    — 61-90 days past due
#   4    — 91-120 days past due
#   5    — 120+ days past due
#   C    — closed
#   X    — unknown
#
# For numeric aggregation (max, mean), we map:
#   "0".."5" → 0..5
#   "C"      → NaN  (closed is not a delinquency state)
#   "X"      → NaN  (unknown)
#
# A separate count of closed months is preserved as a feature.
#
STATUS_TO_ORDINAL: dict[str, float] = {
    "0": 0.0,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BureauAggregationConfig:
    """Resolved configuration for a bureau aggregation run."""

    project_root: Path
    interim_dir: Path
    reports_dir: Path

    bureau_file: str = "bureau.parquet"
    bureau_balance_file: str = "bureau_balance.parquet"
    output_file: str = "bureau_aggregated.parquet"
    feature_catalogue_file: str = "bureau_aggregated_features.json"

    def bureau_path(self) -> Path:
        return self.interim_dir / self.bureau_file

    def bureau_balance_path(self) -> Path:
        return self.interim_dir / self.bureau_balance_file

    def output_path(self) -> Path:
        return self.interim_dir / self.output_file

    def catalogue_path(self) -> Path:
        return self.reports_dir / self.feature_catalogue_file


# ---------------------------------------------------------------------------
# Feature catalogue entry
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """Metadata for one output feature."""

    name: str
    source: str
    aggregation: str
    description: str
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "aggregation": self.aggregation,
            "description": self.description,
            "dtype": self.dtype,
        }


@dataclass
class AggregationReport:
    """Outcome of a bureau aggregation run."""

    generated_at: str
    input_bureau_rows: int
    input_bureau_balance_rows: int
    output_rows: int
    n_features: int
    duration_seconds: float
    features: list[FeatureSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "input_bureau_rows": self.input_bureau_rows,
            "input_bureau_balance_rows": self.input_bureau_balance_rows,
            "output_rows": self.output_rows,
            "n_features": self.n_features,
            "duration_seconds": round(self.duration_seconds, 3),
            "features": [f.to_dict() for f in self.features],
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class BureauAggregator:
    """
    Aggregates bureau + bureau_balance to one row per SK_ID_CURR.

    Two-stage pipeline:
        1. bureau_balance → SK_ID_BUREAU level features
        2. bureau + per-bureau features → SK_ID_CURR level features
    """

    config: BureauAggregationConfig
    report: AggregationReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> AggregationReport:
        start = time.monotonic()
        self.config.interim_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        bureau = self._load(self.config.bureau_path(), "bureau")
        bureau_balance = self._load(self.config.bureau_balance_path(), "bureau_balance")

        logger.info(
            "Loaded bureau: %d rows, bureau_balance: %d rows",
            len(bureau), len(bureau_balance),
        )

        bb_per_bureau = self._aggregate_bureau_balance(bureau_balance)
        logger.info(
            "Aggregated bureau_balance to %d unique SK_ID_BUREAU",
            len(bb_per_bureau),
        )

        bureau_enriched = self._enrich_bureau(bureau, bb_per_bureau)

        result = self._aggregate_to_applicant(bureau_enriched)
        logger.info("Aggregated bureau to %d unique SK_ID_CURR", len(result))

        result = self._add_presence_indicator(result, bureau)

        self._write_output(result)

        duration = time.monotonic() - start
        self.report = self._build_report(
            bureau_rows=len(bureau),
            bureau_balance_rows=len(bureau_balance),
            result=result,
            duration=duration,
        )
        self._write_report(self.report)
        self._log_summary(self.report)
        return self.report

    # ---- input loading ---------------------------------------------------

    def _load(self, path: Path, name: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Interim Parquet for '{name}' not found: {path}. "
                "Run scripts/run_ingestion.py first."
            )
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("  loaded %s: %d rows, %d cols", name, len(df), len(df.columns))
        return df

    # ---- level 1: bureau_balance → per SK_ID_BUREAU ---------------------

    def _aggregate_bureau_balance(self, bb: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce bureau_balance (monthly rows) to one row per SK_ID_BUREAU.

        Produces:
            bb_max_status             max ordinal delinquency status ever
            bb_mean_status            mean ordinal status (across non-null)
            bb_count_months           count of monthly observations
            bb_count_status_0..5      count of months at each status
            bb_count_status_c         count of months marked "C"
            bb_count_status_x         count of months marked "X"
            bb_months_at_1_plus       count of months with status >= 1
            bb_months_at_3_plus       count of months with status >= 3
            bb_ever_closed            any month marked "C"
            bb_ever_unknown           any month marked "X"
        """
        logger.info("Aggregating bureau_balance to SK_ID_BUREAU level")

        bb = bb.copy()
        bb["STATUS"] = bb["STATUS"].astype(str).str.strip().str.upper()
        bb["STATUS_ORD"] = bb["STATUS"].map(STATUS_TO_ORDINAL)

        group = bb.groupby("SK_ID_BUREAU", sort=False)

        aggregated = pd.DataFrame(index=group.size().index)
        aggregated.index.name = "SK_ID_BUREAU"

        aggregated["bb_max_status"] = group["STATUS_ORD"].max()
        aggregated["bb_mean_status"] = group["STATUS_ORD"].mean()
        aggregated["bb_count_months"] = group.size()

        # ---- Counts by status category ---------------------------------
        status_counts = (
            bb.assign(_one=1)
            .pivot_table(
                index="SK_ID_BUREAU",
                columns="STATUS",
                values="_one",
                aggfunc="sum",
                fill_value=0,
            )
        )
        for status_value in ["0", "1", "2", "3", "4", "5", "C", "X"]:
            col_name = f"bb_count_status_{status_value.lower()}"
            if status_value in status_counts.columns:
                aggregated[col_name] = (
                    status_counts[status_value]
                    .reindex(aggregated.index)
                    .fillna(0)
                    .astype("int64")
                )
            else:
                aggregated[col_name] = pd.Series(0, index=aggregated.index, dtype="int64")

        # ---- Composite counts ------------------------------------------
        aggregated["bb_months_at_1_plus"] = (
            aggregated[[f"bb_count_status_{s}" for s in ["1", "2", "3", "4", "5"]]]
            .sum(axis=1)
            .astype("int64")
        )
        aggregated["bb_months_at_3_plus"] = (
            aggregated[[f"bb_count_status_{s}" for s in ["3", "4", "5"]]]
            .sum(axis=1)
            .astype("int64")
        )
        aggregated["bb_ever_closed"] = (
            (aggregated["bb_count_status_c"] > 0).astype("int8")
        )
        aggregated["bb_ever_unknown"] = (
            (aggregated["bb_count_status_x"] > 0).astype("int8")
        )

        return aggregated.reset_index()

    # ---- level 2 prep: enrich bureau with per-bureau features ----------

    def _enrich_bureau(
        self,
        bureau: pd.DataFrame,
        bb_per_bureau: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Left-join per-bureau aggregates onto bureau.

        Applicants with no bureau_balance history get NaN for bb_* columns.
        """
        logger.info("Joining bureau_balance aggregates onto bureau")
        enriched = bureau.merge(
            bb_per_bureau,
            on="SK_ID_BUREAU",
            how="left",
            validate="one_to_one",
        )
        logger.info(
            "  enriched bureau: %d rows, %d cols",
            len(enriched), len(enriched.columns),
        )
        return enriched

    # ---- level 2: bureau → per SK_ID_CURR --------------------------------

    def _aggregate_to_applicant(self, bureau_enriched: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce enriched bureau to one row per SK_ID_CURR.

        Produces a mix of count / sum / mean / max aggregates for
        credit amounts, overdue amounts, and DPD history.
        """
        logger.info("Aggregating bureau to SK_ID_CURR level")
        group = bureau_enriched.groupby("SK_ID_CURR", sort=False)

        out = pd.DataFrame(index=group.size().index)
        out.index.name = "SK_ID_CURR"

        # ---- Counts ----------------------------------------------------
        out["bureau_count_credits"] = group.size().astype("int64")

        out["bureau_count_active"] = self._conditional_count(
            bureau_enriched, "CREDIT_ACTIVE", "Active", out.index
        )
        out["bureau_count_closed"] = self._conditional_count(
            bureau_enriched, "CREDIT_ACTIVE", "Closed", out.index
        )
        out["bureau_count_bad_debt"] = self._conditional_count(
            bureau_enriched, "CREDIT_ACTIVE", "Bad debt", out.index
        )
        out["bureau_count_sold"] = self._conditional_count(
            bureau_enriched, "CREDIT_ACTIVE", "Sold", out.index
        )

        # ---- Credit amounts -------------------------------------------
        out["bureau_credit_sum_total"] = group["AMT_CREDIT_SUM"].sum(min_count=1)
        out["bureau_credit_sum_mean"] = group["AMT_CREDIT_SUM"].mean()
        out["bureau_credit_sum_max"] = group["AMT_CREDIT_SUM"].max()
        out["bureau_credit_sum_min"] = group["AMT_CREDIT_SUM"].min()
        out["bureau_credit_sum_std"] = group["AMT_CREDIT_SUM"].std()

        out["bureau_credit_sum_debt_total"] = group["AMT_CREDIT_SUM_DEBT"].sum(min_count=1)
        out["bureau_credit_sum_debt_mean"] = group["AMT_CREDIT_SUM_DEBT"].mean()
        out["bureau_credit_sum_debt_max"] = group["AMT_CREDIT_SUM_DEBT"].max()

        out["bureau_credit_sum_limit_total"] = group["AMT_CREDIT_SUM_LIMIT"].sum(min_count=1)
        out["bureau_credit_sum_limit_mean"] = group["AMT_CREDIT_SUM_LIMIT"].mean()

        # ---- Overdue amounts ------------------------------------------
        out["bureau_credit_sum_overdue_total"] = group["AMT_CREDIT_SUM_OVERDUE"].sum(min_count=1)
        out["bureau_credit_sum_overdue_max"] = group["AMT_CREDIT_SUM_OVERDUE"].max()
        out["bureau_credit_max_overdue_max"] = group["AMT_CREDIT_MAX_OVERDUE"].max()

        # ---- Prolongation ---------------------------------------------
        out["bureau_cnt_prolong_sum"] = group["CNT_CREDIT_PROLONG"].sum(min_count=1)
        out["bureau_cnt_prolong_max"] = group["CNT_CREDIT_PROLONG"].max()

        # ---- Credit duration ------------------------------------------
        out["bureau_days_credit_min"] = group["DAYS_CREDIT"].min()
        out["bureau_days_credit_max"] = group["DAYS_CREDIT"].max()
        out["bureau_days_credit_mean"] = group["DAYS_CREDIT"].mean()

        out["bureau_days_credit_enddate_min"] = group["DAYS_CREDIT_ENDDATE"].min()
        out["bureau_days_credit_enddate_max"] = group["DAYS_CREDIT_ENDDATE"].max()

        out["bureau_days_enddate_min"] = group["DAYS_ENDDATE_FACT"].min()
        out["bureau_days_enddate_max"] = group["DAYS_ENDDATE_FACT"].max()

        # ---- Recency --------------------------------------------------
        out["bureau_days_credit_update_min"] = group["DAYS_CREDIT_UPDATE"].min()
        out["bureau_days_credit_update_max"] = group["DAYS_CREDIT_UPDATE"].max()

        # ---- bureau_balance-derived (bb_*) ----------------------------
        out["bb_max_status_max"] = group["bb_max_status"].max()
        out["bb_max_status_mean"] = group["bb_max_status"].mean()
        out["bb_mean_status_mean"] = group["bb_mean_status"].mean()
        out["bb_count_months_total"] = group["bb_count_months"].sum(min_count=1)
        out["bb_months_at_1_plus_total"] = group["bb_months_at_1_plus"].sum(min_count=1)
        out["bb_months_at_3_plus_total"] = group["bb_months_at_3_plus"].sum(min_count=1)
        out["bb_ever_closed_sum"] = group["bb_ever_closed"].sum(min_count=1)
        out["bb_ever_unknown_sum"] = group["bb_ever_unknown"].sum(min_count=1)

        return out.reset_index()

    def _conditional_count(
        self,
        df: pd.DataFrame,
        column: str,
        value: str,
        index: pd.Index,
    ) -> pd.Series:
        """
        Count rows per SK_ID_CURR where df[column] == value.

        Returns a Series aligned to `index`, with 0 for applicants that
        have no matching rows. "No bad-debt records" means 0 bad-debt
        records, not NaN.
        """
        mask = df[column] == value
        counts = df.loc[mask].groupby("SK_ID_CURR", sort=False).size()
        return counts.reindex(index).fillna(0).astype("int64")

    # ---- presence indicator ---------------------------------------------

    def _add_presence_indicator(
        self,
        result: pd.DataFrame,
        bureau: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add has_bureau_history flag.

        In this aggregated table, every applicant has bureau records by
        construction, so the flag will be 1 for all rows. The column
        becomes meaningful when this table is left-joined to
        application_train during assembly, where applicants with no
        bureau records receive 0.
        """
        present = set(bureau["SK_ID_CURR"].unique())
        result["has_bureau_history"] = (
            result["SK_ID_CURR"].isin(present).astype("int8")
        )
        return result

    # ---- output writing --------------------------------------------------

    def _write_output(self, result: pd.DataFrame) -> Path:
        # Reorder: SK_ID_CURR first, then alphabetical for reviewability
        cols = ["SK_ID_CURR"] + sorted([c for c in result.columns if c != "SK_ID_CURR"])
        result = result[cols]

        out_path = self.config.output_path()
        result.to_parquet(
            out_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        logger.info(
            "Wrote %s: %d rows, %d cols",
            out_path.name, len(result), len(result.columns),
        )
        return out_path

    # ---- reporting -------------------------------------------------------

    def _build_report(
        self,
        bureau_rows: int,
        bureau_balance_rows: int,
        result: pd.DataFrame,
        duration: float,
    ) -> AggregationReport:
        feature_specs = self._build_feature_specs(result)
        return AggregationReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            input_bureau_rows=bureau_rows,
            input_bureau_balance_rows=bureau_balance_rows,
            output_rows=len(result),
            n_features=len(result.columns) - 1,  # exclude SK_ID_CURR
            duration_seconds=duration,
            features=feature_specs,
        )

    def _build_feature_specs(self, result: pd.DataFrame) -> list[FeatureSpec]:
        """Emit a feature catalogue entry for each output column."""
        specs: list[FeatureSpec] = []
        for col in sorted(result.columns):
            if col == "SK_ID_CURR":
                continue
            specs.append(self._describe_feature(col, str(result[col].dtype)))
        return specs

    def _describe_feature(self, name: str, dtype: str) -> FeatureSpec:
        """Human-readable provenance and description for a feature."""
        if name == "has_bureau_history":
            return FeatureSpec(
                name=name,
                source="bureau",
                aggregation="presence",
                description="1 if applicant has at least one bureau record",
                dtype=dtype,
            )

        source = (
            "bureau_balance (rolled up via bureau)"
            if name.startswith("bb_")
            else "bureau"
        )
        aggregation = self._infer_aggregation(name)
        description = self._infer_description(name)

        return FeatureSpec(
            name=name,
            source=source,
            aggregation=aggregation,
            description=description,
            dtype=dtype,
        )

    def _infer_aggregation(self, name: str) -> str:
        """Infer the aggregation operation from the feature name suffix."""
        for token in ("total", "count", "sum", "mean", "max", "min", "std"):
            if name.endswith(f"_{token}"):
                return token
        for token in ("months", "ever"):
            if f"_{token}" in name:
                return token
        return "aggregate"

    def _infer_description(self, name: str) -> str:
        """
        Descriptions are filled in Stage C when the modelling feature
        set is finalised. Until then, return empty rather than guess.
        """
        return ""

    def _write_report(self, report: AggregationReport) -> Path:
        out_path = self.config.catalogue_path()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Feature catalogue written: %s", out_path.name)
        return out_path

    def _log_summary(self, report: AggregationReport) -> None:
        logger.info("=" * 70)
        logger.info("Bureau aggregation complete")
        logger.info("  input bureau rows:          %d", report.input_bureau_rows)
        logger.info("  input bureau_balance rows:  %d", report.input_bureau_balance_rows)
        logger.info("  output rows:                %d", report.output_rows)
        logger.info("  features produced:          %d", report.n_features)
        logger.info("  duration:                   %.2fs", report.duration_seconds)
        logger.info("=" * 70)