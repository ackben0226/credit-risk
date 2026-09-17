"""
Previous application aggregation: previous_application → per-applicant features.

Home Credit's previous_application table records the applicant's prior
loan applications with Home Credit itself (as opposed to external
credit, which is in the bureau tables). Structure:

    previous_application  (one row per SK_ID_PREV per SK_ID_CURR)

Unlike bureau, previous_application has no grandchildren in this module.
The three child tables (POS_CASH_balance, installments_payments,
credit_card_balance) are aggregated in their own modules and joined in
a later stage.

This module reduces previous_application to one row per SK_ID_CURR.

Feature families
----------------
1. Counts           — total applications, counts by status, by type
2. Rates            — approval rate, refusal rate
3. Numeric aggregates — sums, means, maxes of amounts and rates
4. Date aggregates  — min, max, mean of negative day counts
5. Recency (prev_last_*) — values from the most recent application

The recency family is particularly important in this dataset: what
happened in the applicant's most recent interaction with Home Credit
carries more signal than the average across all applications.

Null handling
-------------
- Applicants with no previous applications produce null for all prev_*
  features except the presence indicator and count features (which are 0).
- No imputation is performed here.

Naming convention
-----------------
All features are prefixed with `prev_` to identify their source table.
The `prev_last_*` prefix denotes features taken from the most recent
application by DAYS_DECISION.
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


logger = logging.getLogger(
    "credit_risk.features.aggregations.previous_application"
)


# ---------------------------------------------------------------------------
# Ordinal mappings for categorical columns used in prev_last_* features
# ---------------------------------------------------------------------------
#
# The recency family needs numeric encodings for categorical columns.
# These ordinals are ordered by strength of signal, not alphabetical.
# They are used only for prev_last_* features; the full categorical
# distributions are captured as counts elsewhere.
#
CONTRACT_STATUS_ORDINAL: dict[str, int] = {
    "Approved": 0,
    "Refused": 1,
    "Canceled": 2,
    "Unused offer": 3,
}

CLIENT_TYPE_ORDINAL: dict[str, int] = {
    "New": 0,
    "Repeater": 1,
    "Refreshed": 2,
}

CONTRACT_TYPE_ORDINAL: dict[str, int] = {
    "Cash loans": 0,
    "Consumer loans": 1,
    "Revolving loans": 2,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreviousApplicationAggregationConfig:
    """Resolved configuration for a previous_application aggregation run."""

    project_root: Path
    interim_dir: Path
    reports_dir: Path

    input_file: str = "previous_application.parquet"
    output_file: str = "previous_application_aggregated.parquet"
    feature_catalogue_file: str = "previous_application_aggregated_features.json"

    def input_path(self) -> Path:
        return self.interim_dir / self.input_file

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
    """Outcome of a previous_application aggregation run."""

    generated_at: str
    input_rows: int
    output_rows: int
    n_features: int
    duration_seconds: float
    features: list[FeatureSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "n_features": self.n_features,
            "duration_seconds": round(self.duration_seconds, 3),
            "features": [f.to_dict() for f in self.features],
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class PreviousApplicationAggregator:
    """
    Aggregates previous_application to one row per SK_ID_CURR.

    Single-stage: groupby SK_ID_CURR and produce aggregates.
    Includes a recency family (prev_last_*) drawn from the application
    with the maximum DAYS_DECISION (i.e. the most recent).
    """

    config: PreviousApplicationAggregationConfig
    report: AggregationReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> AggregationReport:
        start = time.monotonic()
        self.config.interim_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        prev = self._load(self.config.input_path(), "previous_application")
        logger.info(
            "Loaded previous_application: %d rows, %d cols",
            len(prev), len(prev.columns),
        )

        result = self._aggregate(prev)
        logger.info("Aggregated to %d unique SK_ID_CURR", len(result))

        result = self._add_presence_indicator(result, prev)

        self._write_output(result)

        duration = time.monotonic() - start
        self.report = self._build_report(
            input_rows=len(prev),
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

    # ---- aggregation -----------------------------------------------------

    def _aggregate(self, prev: pd.DataFrame) -> pd.DataFrame:
        """Reduce previous_application to one row per SK_ID_CURR."""
        logger.info("Aggregating previous_application to SK_ID_CURR level")

        group = prev.groupby("SK_ID_CURR", sort=False)
        out = pd.DataFrame(index=group.size().index)
        out.index.name = "SK_ID_CURR"

        # ---- Counts ----------------------------------------------------
        out["prev_count_total"] = group.size().astype("int64")

        out["prev_count_approved"] = self._conditional_count(
            prev, "NAME_CONTRACT_STATUS", "Approved", out.index
        )
        out["prev_count_refused"] = self._conditional_count(
            prev, "NAME_CONTRACT_STATUS", "Refused", out.index
        )
        out["prev_count_canceled"] = self._conditional_count(
            prev, "NAME_CONTRACT_STATUS", "Canceled", out.index
        )
        out["prev_count_unused_offer"] = self._conditional_count(
            prev, "NAME_CONTRACT_STATUS", "Unused offer", out.index
        )

        out["prev_count_cash_loans"] = self._conditional_count(
            prev, "NAME_CONTRACT_TYPE", "Cash loans", out.index
        )
        out["prev_count_consumer_loans"] = self._conditional_count(
            prev, "NAME_CONTRACT_TYPE", "Consumer loans", out.index
        )
        out["prev_count_revolving_loans"] = self._conditional_count(
            prev, "NAME_CONTRACT_TYPE", "Revolving loans", out.index
        )

        # Residual count: rows whose NAME_CONTRACT_TYPE is not one of the
        # three known categories (or is null). Ensures the contract-type
        # categories sum exactly to prev_count_total.
        out["prev_count_other_contract_type"] = (
            out["prev_count_total"]
            - out["prev_count_cash_loans"]
            - out["prev_count_consumer_loans"]
            - out["prev_count_revolving_loans"]
        ).astype("int64")

        out["prev_count_client_new"] = self._conditional_count(
            prev, "NAME_CLIENT_TYPE", "New", out.index
        )
        out["prev_count_client_repeater"] = self._conditional_count(
            prev, "NAME_CLIENT_TYPE", "Repeater", out.index
        )
        out["prev_count_client_refreshed"] = self._conditional_count(
            prev, "NAME_CLIENT_TYPE", "Refreshed", out.index
        )

        # ---- Rates -----------------------------------------------------
        total = out["prev_count_total"].astype("float64")
        out["prev_approval_rate"] = (out["prev_count_approved"] / total).astype("float64")
        out["prev_refusal_rate"] = (out["prev_count_refused"] / total).astype("float64")

        # ---- Numeric aggregates ----------------------------------------
        for col, prefix in [
            ("AMT_CREDIT", "prev_amt_credit"),
            ("AMT_APPLICATION", "prev_amt_application"),
            ("AMT_ANNUITY", "prev_amt_annuity"),
            ("AMT_GOODS_PRICE", "prev_amt_goods_price"),
            ("AMT_DOWN_PAYMENT", "prev_amt_down_payment"),
            ("CNT_PAYMENT", "prev_cnt_payment"),
            ("RATE_DOWN_PAYMENT", "prev_rate_down_payment"),
        ]:
            if col not in prev.columns:
                continue
            out[f"{prefix}_sum"] = group[col].sum(min_count=1)
            out[f"{prefix}_mean"] = group[col].mean()
            out[f"{prefix}_max"] = group[col].max()
            out[f"{prefix}_min"] = group[col].min()

        for col, prefix in [
            ("RATE_INTEREST_PRIMARY", "prev_rate_interest_primary"),
            ("RATE_INTEREST_PRIVILEGED", "prev_rate_interest_privileged"),
        ]:
            if col not in prev.columns:
                continue
            out[f"{prefix}_mean"] = group[col].mean()
            out[f"{prefix}_max"] = group[col].max()

        # ---- Date aggregates (negative day counts) ---------------------
        for col, prefix in [
            ("DAYS_DECISION", "prev_days_decision"),
            ("DAYS_FIRST_DRAWING", "prev_days_first_drawing"),
            ("DAYS_FIRST_DUE", "prev_days_first_due"),
            ("DAYS_LAST_DUE_1ST_VERSION", "prev_days_last_due_1st"),
            ("DAYS_LAST_DUE", "prev_days_last_due"),
            ("DAYS_TERMINATION", "prev_days_termination"),
        ]:
            if col not in prev.columns:
                continue
            out[f"{prefix}_min"] = group[col].min()
            out[f"{prefix}_max"] = group[col].max()
            out[f"{prefix}_mean"] = group[col].mean()

        # ---- Recency family (prev_last_*) ------------------------------
        out = self._add_recency_features(prev, out)

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
        have no matching rows.
        """
        mask = df[column] == value
        counts = df.loc[mask].groupby("SK_ID_CURR", sort=False).size()
        return counts.reindex(index).fillna(0).astype("int64")

    def _add_recency_features(
        self,
        prev: pd.DataFrame,
        out: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add prev_last_* features drawn from the most recent application.

        "Most recent" is defined as the row with the maximum DAYS_DECISION
        (values are negative, so max is closest to zero = most recent).
        """
        logger.info("Computing prev_last_* features from most recent application")

        # Locate the most recent row per SK_ID_CURR
        idx = prev.groupby("SK_ID_CURR", sort=False)["DAYS_DECISION"].idxmax()
        last = prev.loc[idx].set_index("SK_ID_CURR")
        last = last.reindex(out.index)

        # Numeric recency features
        numeric_last_map = [
            ("AMT_CREDIT", "prev_last_amt_credit"),
            ("AMT_ANNUITY", "prev_last_amt_annuity"),
            ("AMT_APPLICATION", "prev_last_amt_application"),
            ("DAYS_DECISION", "prev_last_days_decision"),
            ("DAYS_FIRST_DRAWING", "prev_last_days_first_drawing"),
            ("DAYS_LAST_DUE", "prev_last_days_last_due"),
            ("CNT_PAYMENT", "prev_last_cnt_payment"),
            ("RATE_DOWN_PAYMENT", "prev_last_rate_down_payment"),
        ]
        for col, out_name in numeric_last_map:
            if col in last.columns:
                out[out_name] = last[col]

        # Categorical recency features (ordinal-encoded)
        ordinal_map = [
            ("NAME_CONTRACT_STATUS", "prev_last_name_contract_status", CONTRACT_STATUS_ORDINAL),
            ("NAME_CLIENT_TYPE", "prev_last_name_client_type", CLIENT_TYPE_ORDINAL),
            ("NAME_CONTRACT_TYPE", "prev_last_name_contract_type", CONTRACT_TYPE_ORDINAL),
        ]
        for col, out_name, mapping in ordinal_map:
            if col in last.columns:
                out[out_name] = last[col].map(mapping).astype("float64")

        return out

    # ---- presence indicator ---------------------------------------------

    def _add_presence_indicator(
        self,
        result: pd.DataFrame,
        prev: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add has_previous_application flag.

        In this aggregated table, every applicant has previous applications
        by construction, so the flag will be 1 for all rows. The column
        becomes meaningful when this table is left-joined to
        application_train during assembly.
        """
        present = set(prev["SK_ID_CURR"].unique())
        result["has_previous_application"] = (
            result["SK_ID_CURR"].isin(present).astype("int8")
        )
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
        input_rows: int,
        result: pd.DataFrame,
        duration: float,
    ) -> AggregationReport:
        feature_specs = self._build_feature_specs(result)
        return AggregationReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            input_rows=input_rows,
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
        if name == "has_previous_application":
            return FeatureSpec(
                name=name,
                source="previous_application",
                aggregation="presence",
                description="1 if applicant has at least one previous application",
                dtype=dtype,
            )

        aggregation = self._infer_aggregation(name)
        return FeatureSpec(
            name=name,
            source="previous_application",
            aggregation=aggregation,
            description="",
            dtype=dtype,
        )

    def _infer_aggregation(self, name: str) -> str:
        """Infer the aggregation operation from the feature name."""
        if name.startswith("prev_last_"):
            return "last"
        if "approval_rate" in name or "refusal_rate" in name:
            return "rate"
        if "_count_" in name or name.endswith("_count"):
            return "count"
        for token in ("total", "sum", "mean", "max", "min", "std"):
            if name.endswith(f"_{token}"):
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
        logger.info("Previous application aggregation complete")
        logger.info("  input rows:        %d", report.input_rows)
        logger.info("  output rows:       %d", report.output_rows)
        logger.info("  features produced: %d", report.n_features)
        logger.info("  duration:          %.2fs", report.duration_seconds)
        logger.info("=" * 70)