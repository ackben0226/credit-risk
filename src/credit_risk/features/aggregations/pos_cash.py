"""
POS_CASH_balance aggregation → per-applicant features.

Home Credit's POS_CASH_balance table contains monthly snapshots of point-
of-sale and cash loan performance. Structure:

    POS_CASH_balance (one row per SK_ID_PREV per MONTHS_BALANCE)
        │
        └── keyed by SK_ID_PREV → previous_application.SK_ID_PREV
                                  → previous_application.SK_ID_CURR

This module performs a two-level aggregation:

    Level 1: POS_CASH_balance → per SK_ID_PREV
             (max DPD, counts of each contract status, installment counts)

    Level 2: per SK_ID_PREV → per SK_ID_CURR
             (via previous_application; count/sum/mean/max across the
              applicant's POS/CASH loan history)

The output is one row per applicant, with a stable schema and a presence
indicator (has_pos_cash_history).

Null handling
-------------
- Applicants with no POS/CASH records are absent from the output. The
  assembly stage left-joins this table onto application_train; applicants
  absent here receive NaN for pos_* features.
- Applicants with POS/CASH records but no non-null SK_DPD values get
  NaN for DPD aggregates. "No data" is distinct from "zero DPD."
- No imputation is performed here.

Reference
---------
Data Quality Report §6.3 and §6.4 — POS/CASH coverage of previous_application
is ~55%. Roughly 45% of previous applications have no POS/CASH records.
This module preserves that characteristic. The presence indicator and
count-based features carry the signal that would otherwise be lost to
nulls.

Notes
-----
- POS_CASH_balance contains both SK_ID_PREV and SK_ID_CURR. This module
  deliberately joins through SK_ID_PREV → previous_application to enforce
  parent-child consistency. Relying on the child's own SK_ID_CURR is
  riskier because orphan records exist (inspection found 37,422 orphan
  SK_ID_PREV values).
- NAME_CONTRACT_STATUS has 7 known values. The module emits per-status
  counts; any unrecognized values are captured by a residual count.
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


logger = logging.getLogger("credit_risk.features.aggregations.pos_cash")


# ---------------------------------------------------------------------------
# Known NAME_CONTRACT_STATUS values
# ---------------------------------------------------------------------------
# Emitting explicit counts for known statuses ensures the feature catalogue
# is stable even if a status is absent from the data on a given run.
# Unrecognized statuses are captured by pos_count_status_other.
#
KNOWN_CONTRACT_STATUSES: tuple[str, ...] = (
    "Active",
    "Completed",
    "Signed",
    "Demand",
    "Returned to the store",
    "Approved",
    "Amortized debt",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PosCashAggregationConfig:
    """Resolved configuration for a POS/CASH aggregation run."""

    project_root: Path
    interim_dir: Path
    reports_dir: Path

    pos_cash_file: str = "pos_cash_balance.parquet"
    previous_application_file: str = "previous_application.parquet"
    output_file: str = "pos_cash_aggregated.parquet"
    feature_catalogue_file: str = "pos_cash_aggregated_features.json"

    def pos_cash_path(self) -> Path:
        return self.interim_dir / self.pos_cash_file

    def previous_application_path(self) -> Path:
        return self.interim_dir / self.previous_application_file

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
    """Outcome of a POS/CASH aggregation run."""

    generated_at: str
    input_pos_cash_rows: int
    input_previous_application_rows: int
    output_rows: int
    n_features: int
    duration_seconds: float
    features: list[FeatureSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "input_pos_cash_rows": self.input_pos_cash_rows,
            "input_previous_application_rows": self.input_previous_application_rows,
            "output_rows": self.output_rows,
            "n_features": self.n_features,
            "duration_seconds": round(self.duration_seconds, 3),
            "features": [f.to_dict() for f in self.features],
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class PosCashAggregator:
    """
    Aggregates POS_CASH_balance to one row per SK_ID_CURR.

    Two-stage pipeline:
        1. POS_CASH_balance → SK_ID_PREV level features
        2. per SK_ID_PREV + previous_application → SK_ID_CURR level features
    """

    config: PosCashAggregationConfig
    report: AggregationReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> AggregationReport:
        start = time.monotonic()
        self.config.interim_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        pos_cash = self._load(self.config.pos_cash_path(), "pos_cash_balance")
        prev_app = self._load(
            self.config.previous_application_path(), "previous_application"
        )

        logger.info(
            "Loaded pos_cash_balance: %d rows, previous_application: %d rows",
            len(pos_cash), len(prev_app),
        )

        per_prev = self._aggregate_to_prev(pos_cash)
        logger.info(
            "Aggregated pos_cash_balance to %d unique SK_ID_PREV", len(per_prev)
        )

        per_curr = self._aggregate_to_applicant(per_prev, prev_app)
        logger.info("Aggregated POS/CASH to %d unique SK_ID_CURR", len(per_curr))

        per_curr = self._add_presence_indicator(per_curr)

        self._write_output(per_curr)

        duration = time.monotonic() - start
        self.report = self._build_report(
            pos_cash_rows=len(pos_cash),
            prev_app_rows=len(prev_app),
            result=per_curr,
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

    # ---- level 1: pos_cash_balance → per SK_ID_PREV ---------------------

    def _aggregate_to_prev(self, pos_cash: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce POS_CASH_balance (monthly rows) to one row per SK_ID_PREV.

        Produces:
            pos_prev_sk_dpd_max              max days past due ever
            pos_prev_sk_dpd_mean             mean days past due
            pos_prev_sk_dpd_sum              sum days past due
            pos_prev_sk_dpd_positive_months  months with any DPD
            pos_prev_sk_dpd_def_max          max defined-DPD
            pos_prev_sk_dpd_def_mean         mean defined-DPD
            pos_prev_sk_dpd_def_positive_months
            pos_prev_count_months            number of monthly snapshots
            pos_prev_cnt_instalment_max      max total installments planned
            pos_prev_cnt_instalment_future_max
            pos_prev_cnt_instalment_future_last
            pos_prev_months_balance_min      oldest snapshot (most negative)
            pos_prev_months_balance_max      most recent snapshot
            pos_prev_status_<value>          months at each contract status
            pos_prev_ever_status_<value>     any month at each contract status
        """
        logger.info("Aggregating pos_cash_balance to SK_ID_PREV level")

        pc = pos_cash.copy()

        # Normalize status: strip whitespace; do not uppercase since some
        # values are multi-word with capitalisation (e.g. "Returned to the store")
        pc["NAME_CONTRACT_STATUS"] = (
            pc["NAME_CONTRACT_STATUS"].astype(str).str.strip()
        )

        # Compute DPD positivity flags per row (before grouping)
        pc["_sk_dpd_positive"] = (pc["SK_DPD"] > 0).astype("int8")
        pc["_sk_dpd_def_positive"] = (pc["SK_DPD_DEF"] > 0).astype("int8")

        group = pc.groupby("SK_ID_PREV", sort=False)

        aggregated = pd.DataFrame(index=group.size().index)
        aggregated.index.name = "SK_ID_PREV"

        aggregated["pos_prev_count_months"] = group.size().astype("int64")

        # ---- DPD aggregates --------------------------------------------
        aggregated["pos_prev_sk_dpd_max"] = group["SK_DPD"].max()
        aggregated["pos_prev_sk_dpd_mean"] = group["SK_DPD"].mean()
        aggregated["pos_prev_sk_dpd_sum"] = group["SK_DPD"].sum(min_count=1)
        aggregated["pos_prev_sk_dpd_positive_months"] = (
            group["_sk_dpd_positive"].sum().astype("int64")
        )

        aggregated["pos_prev_sk_dpd_def_max"] = group["SK_DPD_DEF"].max()
        aggregated["pos_prev_sk_dpd_def_mean"] = group["SK_DPD_DEF"].mean()
        aggregated["pos_prev_sk_dpd_def_sum"] = group["SK_DPD_DEF"].sum(min_count=1)
        aggregated["pos_prev_sk_dpd_def_positive_months"] = (
            group["_sk_dpd_def_positive"].sum().astype("int64")
        )

        # ---- Installment counts ----------------------------------------
        aggregated["pos_prev_cnt_instalment_max"] = group["CNT_INSTALMENT"].max()
        aggregated["pos_prev_cnt_instalment_mean"] = group["CNT_INSTALMENT"].mean()

        aggregated["pos_prev_cnt_instalment_future_max"] = (
            group["CNT_INSTALMENT_FUTURE"].max()
        )
        aggregated["pos_prev_cnt_instalment_future_mean"] = (
            group["CNT_INSTALMENT_FUTURE"].mean()
        )

        # Most recent snapshot's future-installment count
        # MONTHS_BALANCE is negative going back in time; max = most recent
        aggregated["pos_prev_cnt_instalment_future_last"] = (
            pc.sort_values("MONTHS_BALANCE")
            .groupby("SK_ID_PREV", sort=False)["CNT_INSTALMENT_FUTURE"]
            .last()
        )

        # ---- Snapshot recency ------------------------------------------
        aggregated["pos_prev_months_balance_min"] = group["MONTHS_BALANCE"].min()
        aggregated["pos_prev_months_balance_max"] = group["MONTHS_BALANCE"].max()

        # ---- Contract status counts ------------------------------------
        for status in KNOWN_CONTRACT_STATUSES:
            col_name = f"pos_prev_status_{self._slug(status)}"
            mask = pc["NAME_CONTRACT_STATUS"] == status
            counts = pc.loc[mask].groupby("SK_ID_PREV", sort=False).size()
            aggregated[col_name] = (
                counts.reindex(aggregated.index).fillna(0).astype("int64")
            )

        # Residual status count for any unrecognized values
        known_mask = pc["NAME_CONTRACT_STATUS"].isin(KNOWN_CONTRACT_STATUSES)
        other_counts = pc.loc[~known_mask].groupby("SK_ID_PREV", sort=False).size()
        aggregated["pos_prev_status_other"] = (
            other_counts.reindex(aggregated.index).fillna(0).astype("int64")
        )

        # ---- Ever-status flags -----------------------------------------
        for status in KNOWN_CONTRACT_STATUSES:
            col_name = f"pos_prev_ever_{self._slug(status)}"
            count_col = f"pos_prev_status_{self._slug(status)}"
            aggregated[col_name] = (aggregated[count_col] > 0).astype("int8")

        return aggregated.reset_index()

    # ---- level 2: per SK_ID_PREV → per SK_ID_CURR -----------------------

    def _aggregate_to_applicant(
        self,
        per_prev: pd.DataFrame,
        prev_app: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate per-SK_ID_PREV features to SK_ID_CURR via
        previous_application.

        Inner join on SK_ID_PREV — rows in pos_cash_balance whose
        SK_ID_PREV does not appear in previous_application are orphans
        and are dropped by the join. This is deliberate: without a valid
        parent, we cannot attribute the record to an applicant.
        """
        logger.info("Joining POS/CASH per-SK_ID_PREV features to previous_application")

        # Only need the join keys from previous_application
        prev_keys = prev_app[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates(
            subset=["SK_ID_PREV"]
        )

        joined = prev_keys.merge(
            per_prev,
            on="SK_ID_PREV",
            how="inner",
            validate="one_to_many",
        )
        logger.info(
            "  joined: %d rows across %d unique SK_ID_CURR",
            len(joined), joined["SK_ID_CURR"].nunique(),
        )

        # Aggregate to SK_ID_CURR
        group = joined.groupby("SK_ID_CURR", sort=False)
        out = pd.DataFrame(index=group.size().index)
        out.index.name = "SK_ID_CURR"

        # ---- Counts at applicant level ---------------------------------
        out["pos_count_prev_loans"] = group.size().astype("int64")
        out["pos_count_total_months"] = (
            group["pos_prev_count_months"].sum(min_count=1)
        )

        # ---- DPD aggregates across all POS/CASH loans ------------------
        out["pos_sk_dpd_max"] = group["pos_prev_sk_dpd_max"].max()
        out["pos_sk_dpd_mean"] = group["pos_prev_sk_dpd_mean"].mean()
        out["pos_sk_dpd_sum"] = group["pos_prev_sk_dpd_sum"].sum(min_count=1)
        out["pos_sk_dpd_positive_months_total"] = (
            group["pos_prev_sk_dpd_positive_months"].sum(min_count=1)
        )

        out["pos_sk_dpd_def_max"] = group["pos_prev_sk_dpd_def_max"].max()
        out["pos_sk_dpd_def_mean"] = group["pos_prev_sk_dpd_def_mean"].mean()
        out["pos_sk_dpd_def_sum"] = group["pos_prev_sk_dpd_def_sum"].sum(min_count=1)
        out["pos_sk_dpd_def_positive_months_total"] = (
            group["pos_prev_sk_dpd_def_positive_months"].sum(min_count=1)
        )

        # ---- Installment aggregates ------------------------------------
        out["pos_cnt_instalment_max"] = group["pos_prev_cnt_instalment_max"].max()
        out["pos_cnt_instalment_mean"] = group["pos_prev_cnt_instalment_mean"].mean()

        out["pos_cnt_instalment_future_max"] = (
            group["pos_prev_cnt_instalment_future_max"].max()
        )
        out["pos_cnt_instalment_future_mean"] = (
            group["pos_prev_cnt_instalment_future_mean"].mean()
        )

        out["pos_cnt_instalment_future_current"] = (
            group["pos_prev_cnt_instalment_future_last"].sum(min_count=1)
        )

        # ---- Snapshot recency ------------------------------------------
        out["pos_months_balance_min"] = group["pos_prev_months_balance_min"].min()
        out["pos_months_balance_max"] = group["pos_prev_months_balance_max"].max()

        # ---- Contract status counts at applicant level -----------------
        for status in KNOWN_CONTRACT_STATUSES:
            slug = self._slug(status)
            out[f"pos_count_status_{slug}"] = (
                group[f"pos_prev_status_{slug}"].sum(min_count=1).astype("int64")
            )
            out[f"pos_count_ever_{slug}"] = (
                group[f"pos_prev_ever_{slug}"].sum(min_count=1).astype("int64")
            )

        out["pos_count_status_other"] = (
            group["pos_prev_status_other"].sum(min_count=1).astype("int64")
        )

        # ---- Derived rates ---------------------------------------------
        total_months = out["pos_count_total_months"].replace(0, pd.NA)

        out["pos_rate_dpd_positive"] = (
            out["pos_sk_dpd_positive_months_total"] / total_months
        )
        out["pos_rate_dpd_def_positive"] = (
            out["pos_sk_dpd_def_positive_months_total"] / total_months
        )

        return out.reset_index()

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _slug(value: str) -> str:
        """Normalize a category value into a feature-name-friendly slug."""
        return (
            value.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    def _add_presence_indicator(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Add has_pos_cash_history flag.

        In this aggregated table, every applicant has POS/CASH records by
        construction. The flag becomes meaningful during assembly when
        this table is left-joined onto application_train.
        """
        result["has_pos_cash_history"] = 1
        result["has_pos_cash_history"] = result["has_pos_cash_history"].astype("int8")
        return result

    # ---- output writing --------------------------------------------------

    def _write_output(self, result: pd.DataFrame) -> Path:
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
        pos_cash_rows: int,
        prev_app_rows: int,
        result: pd.DataFrame,
        duration: float,
    ) -> AggregationReport:
        feature_specs = self._build_feature_specs(result)
        return AggregationReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            input_pos_cash_rows=pos_cash_rows,
            input_previous_application_rows=prev_app_rows,
            output_rows=len(result),
            n_features=len(result.columns) - 1,
            duration_seconds=duration,
            features=feature_specs,
        )

    def _build_feature_specs(self, result: pd.DataFrame) -> list[FeatureSpec]:
        specs: list[FeatureSpec] = []
        for col in sorted(result.columns):
            if col == "SK_ID_CURR":
                continue
            specs.append(self._describe_feature(col, str(result[col].dtype)))
        return specs

    def _describe_feature(self, name: str, dtype: str) -> FeatureSpec:
        if name == "has_pos_cash_history":
            return FeatureSpec(
                name=name,
                source="POS_CASH_balance",
                aggregation="presence",
                description="1 if applicant has any POS/CASH loan records",
                dtype=dtype,
            )

        return FeatureSpec(
            name=name,
            source="POS_CASH_balance (rolled up via previous_application)",
            aggregation=self._infer_aggregation(name),
            description="",
            dtype=dtype,
        )

    def _infer_aggregation(self, name: str) -> str:
        for token in ("total", "count", "sum", "mean", "max", "min",
                      "last", "current", "rate", "months", "ever"):
            if name.endswith(f"_{token}") or f"_{token}_" in name:
                return token
        return "aggregate"

    def _write_report(self, report: AggregationReport) -> Path:
        out_path = self.config.catalogue_path()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Feature catalogue written: %s", out_path.name)
        return out_path

    def _log_summary(self, report: AggregationReport) -> None:
        logger.info("=" * 70)
        logger.info("POS/CASH aggregation complete")
        logger.info("  input pos_cash_balance rows:      %d", report.input_pos_cash_rows)
        logger.info("  input previous_application rows:  %d", report.input_previous_application_rows)
        logger.info("  output rows:                      %d", report.output_rows)
        logger.info("  features produced:                %d", report.n_features)
        logger.info("  duration:                         %.2fs", report.duration_seconds)
        logger.info("=" * 70)