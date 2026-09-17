"""
installments_payments aggregation → per-applicant features.

Home Credit's installments_payments table records each scheduled and
actual installment payment for prior loans. It is the most direct
behavioral signal in the dataset: it captures whether the applicant
paid on time, in full, or late/under.

Structure:

    installments_payments (one row per installment per prior loan)
        │
        └── keyed by SK_ID_PREV → previous_application.SK_ID_PREV
                                  → previous_application.SK_ID_CURR

This module performs a two-level aggregation:

    Level 1: installments_payments → per SK_ID_PREV
             (payment delay, underpayment, payment ratio, counts,
              recency)

    Level 2: per SK_ID_PREV → per SK_ID_CURR
             (via previous_application; aggregated across the
              applicant's full installment history)

The output is one row per applicant, with a stable schema and a presence
indicator (has_installment_history).

Derived quantities
------------------
For each installment row, three quantities are computed:

    delay        = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
                   positive = late, negative = early
    underpayment = AMT_INSTALMENT - AMT_PAYMENT
                   positive = paid less than scheduled
    payment_ratio = AMT_PAYMENT / AMT_INSTALMENT
                   <1 = underpaid, 1 = exact, >1 = overpaid

Null handling
-------------
- DAYS_ENTRY_PAYMENT and AMT_PAYMENT each have ~0.02% nulls in the raw
  data. These propagate into the derived quantities — a null payment
  date means an unknown delay, which is correct.
- Applicants with no installment records are absent from the output.
  The assembly stage left-joins onto application_train.
- No imputation is performed here.

Orphan handling
---------------
Rows in installments_payments whose SK_ID_PREV is not present in
previous_application are dropped by the inner join. Inspection found
38,847 orphan SK_ID_PREV values. The report tracks this at the
unique-SK_ID_PREV level:

    unique_prev_before    — unique SK_ID_PREV values in per_prev
    unique_prev_after     — unique SK_ID_PREV values that matched
    orphan_prev_keys      — the difference

Row-level orphan counts are not reported here because they require a
separate scan of the raw table; the join operates on the already-
aggregated per-SK_ID_PREV table.

Reference
---------
Data Quality Report §6.1 and §6.5 — installments_payments has 13.6M rows
and 38,847 orphan SK_ID_PREV values. Nulls in DAYS_ENTRY_PAYMENT and
AMT_PAYMENT are ~0.02%.
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


logger = logging.getLogger("credit_risk.features.aggregations.installments")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstallmentsAggregationConfig:
    """Resolved configuration for an installments aggregation run."""

    project_root: Path
    interim_dir: Path
    reports_dir: Path

    installments_file: str = "installments_payments.parquet"
    previous_application_file: str = "previous_application.parquet"
    output_file: str = "installments_aggregated.parquet"
    feature_catalogue_file: str = "installments_aggregated_features.json"

    def installments_path(self) -> Path:
        return self.interim_dir / self.installments_file

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
    """Outcome of an installments aggregation run."""

    generated_at: str
    input_installments_rows: int
    input_previous_application_rows: int
    unique_prev_before: int
    unique_prev_after: int
    orphan_prev_keys: int
    output_rows: int
    n_features: int
    duration_seconds: float
    features: list[FeatureSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "input_installments_rows": self.input_installments_rows,
            "input_previous_application_rows": self.input_previous_application_rows,
            "unique_prev_before": self.unique_prev_before,
            "unique_prev_after": self.unique_prev_after,
            "orphan_prev_keys": self.orphan_prev_keys,
            "output_rows": self.output_rows,
            "n_features": self.n_features,
            "duration_seconds": round(self.duration_seconds, 3),
            "features": [f.to_dict() for f in self.features],
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class InstallmentsAggregator:
    """
    Aggregates installments_payments to one row per SK_ID_CURR.

    Two-stage pipeline:
        1. installments_payments → SK_ID_PREV level features
        2. per SK_ID_PREV + previous_application → SK_ID_CURR level features
    """

    config: InstallmentsAggregationConfig
    report: AggregationReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> AggregationReport:
        start = time.monotonic()
        self.config.interim_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        installments = self._load(
            self.config.installments_path(), "installments_payments"
        )
        prev_app = self._load(
            self.config.previous_application_path(), "previous_application"
        )

        logger.info(
            "Loaded installments_payments: %d rows, previous_application: %d rows",
            len(installments), len(prev_app),
        )

        per_prev = self._aggregate_to_prev(installments)
        logger.info(
            "Aggregated installments_payments to %d unique SK_ID_PREV",
            len(per_prev),
        )

        per_curr, join_diag = self._aggregate_to_applicant(per_prev, prev_app)
        logger.info(
            "Aggregated installments to %d unique SK_ID_CURR", len(per_curr)
        )

        per_curr = self._add_presence_indicator(per_curr)

        self._write_output(per_curr)

        duration = time.monotonic() - start
        self.report = self._build_report(
            installments_rows=len(installments),
            prev_app_rows=len(prev_app),
            join_diag=join_diag,
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

    # ---- level 1: installments_payments → per SK_ID_PREV ----------------

    def _aggregate_to_prev(self, installments: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce installments_payments to one row per SK_ID_PREV.

        Produces delay / underpayment / ratio / count / recency features
        for each prior loan.
        """
        logger.info("Aggregating installments_payments to SK_ID_PREV level")

        inst = installments.copy()

        # ---- Derived quantities per installment row --------------------
        inst["_delay"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
        inst["_underpayment"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]

        safe_denominator = inst["AMT_INSTALMENT"].where(inst["AMT_INSTALMENT"] > 0)
        inst["_payment_ratio"] = inst["AMT_PAYMENT"] / safe_denominator

        inst["_delay_positive"] = (inst["_delay"] > 0).astype("int8")
        inst["_underpayment_positive"] = (inst["_underpayment"] > 0).astype("int8")
        inst["_underpaid"] = (inst["_payment_ratio"] < 1).astype("int8")

        group = inst.groupby("SK_ID_PREV", sort=False)

        aggregated = pd.DataFrame(index=group.size().index)
        aggregated.index.name = "SK_ID_PREV"

        # ---- Counts ----------------------------------------------------
        aggregated["inst_prev_count_installments"] = group.size().astype("int64")
        aggregated["inst_prev_count_distinct_versions"] = (
            group["NUM_INSTALMENT_VERSION"].nunique().astype("int64")
        )
        aggregated["inst_prev_num_instalment_max"] = (
            group["NUM_INSTALMENT_NUMBER"].max()
        )

        # ---- Payment delay ---------------------------------------------
        aggregated["inst_prev_delay_max"] = group["_delay"].max()
        aggregated["inst_prev_delay_mean"] = group["_delay"].mean()
        aggregated["inst_prev_delay_sum"] = group["_delay"].sum(min_count=1)
        aggregated["inst_prev_delay_std"] = group["_delay"].std()
        aggregated["inst_prev_delay_positive_count"] = (
            group["_delay_positive"].sum().astype("int64")
        )

        # ---- Underpayment ----------------------------------------------
        aggregated["inst_prev_underpayment_max"] = group["_underpayment"].max()
        aggregated["inst_prev_underpayment_mean"] = group["_underpayment"].mean()
        aggregated["inst_prev_underpayment_sum"] = (
            group["_underpayment"].sum(min_count=1)
        )
        aggregated["inst_prev_underpayment_positive_count"] = (
            group["_underpayment_positive"].sum().astype("int64")
        )

        # ---- Payment ratio ---------------------------------------------
        aggregated["inst_prev_payment_ratio_mean"] = group["_payment_ratio"].mean()
        aggregated["inst_prev_payment_ratio_min"] = group["_payment_ratio"].min()
        aggregated["inst_prev_payment_ratio_max"] = group["_payment_ratio"].max()
        aggregated["inst_prev_underpaid_count"] = (
            group["_underpaid"].sum().astype("int64")
        )

        # ---- Amounts ---------------------------------------------------
        aggregated["inst_prev_amt_instalment_sum"] = (
            group["AMT_INSTALMENT"].sum(min_count=1)
        )
        aggregated["inst_prev_amt_instalment_mean"] = group["AMT_INSTALMENT"].mean()
        aggregated["inst_prev_amt_payment_sum"] = (
            group["AMT_PAYMENT"].sum(min_count=1)
        )
        aggregated["inst_prev_amt_payment_mean"] = group["AMT_PAYMENT"].mean()
        aggregated["inst_prev_amt_payment_max"] = group["AMT_PAYMENT"].max()

        # ---- Recency ---------------------------------------------------
        aggregated["inst_prev_days_instalment_min"] = group["DAYS_INSTALMENT"].min()
        aggregated["inst_prev_days_instalment_max"] = group["DAYS_INSTALMENT"].max()
        aggregated["inst_prev_days_entry_min"] = group["DAYS_ENTRY_PAYMENT"].min()
        aggregated["inst_prev_days_entry_max"] = group["DAYS_ENTRY_PAYMENT"].max()

        # ---- Most recent payment's delay and ratio ---------------------
        sorted_inst = inst.sort_values(
            ["SK_ID_PREV", "DAYS_ENTRY_PAYMENT"], kind="stable"
        )
        last_per_prev = (
            sorted_inst.groupby("SK_ID_PREV", sort=False)[
                ["_delay", "_payment_ratio", "AMT_PAYMENT"]
            ]
            .last()
        )
        last_per_prev = last_per_prev.rename(
            columns={
                "_delay": "inst_prev_delay_last",
                "_payment_ratio": "inst_prev_payment_ratio_last",
                "AMT_PAYMENT": "inst_prev_amt_payment_last",
            }
        )
        aggregated = aggregated.join(last_per_prev, how="left")

        return aggregated.reset_index()

    # ---- level 2: per SK_ID_PREV → per SK_ID_CURR -----------------------

    def _aggregate_to_applicant(
        self,
        per_prev: pd.DataFrame,
        prev_app: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """
        Aggregate per-SK_ID_PREV features to SK_ID_CURR via
        previous_application.

        Inner join on SK_ID_PREV. Rows whose SK_ID_PREV is absent from
        previous_application are orphans and are dropped. This is
        deliberate — without a valid parent we cannot attribute the
        record to an applicant reliably.

        Returns the aggregated DataFrame and a dict of join diagnostics
        measured at the unique-SK_ID_PREV level.
        """
        logger.info(
            "Joining installments per-SK_ID_PREV features to previous_application"
        )

        unique_prev_before = int(per_prev["SK_ID_PREV"].nunique())

        prev_keys = prev_app[["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates(
            subset=["SK_ID_PREV"]
        )

        joined = prev_keys.merge(
            per_prev,
            on="SK_ID_PREV",
            how="inner",
            validate="one_to_one",
        )

        unique_prev_after = int(joined["SK_ID_PREV"].nunique())
        orphan_prev_keys = unique_prev_before - unique_prev_after

        logger.info(
            "  join: %d unique SK_ID_PREV before, %d matched, %d orphan keys",
            unique_prev_before, unique_prev_after, orphan_prev_keys,
        )

        join_diag = {
            "unique_prev_before": unique_prev_before,
            "unique_prev_after": unique_prev_after,
            "orphan_prev_keys": orphan_prev_keys,
        }

        group = joined.groupby("SK_ID_CURR", sort=False)
        out = pd.DataFrame(index=group.size().index)
        out.index.name = "SK_ID_CURR"

        # ---- Counts at applicant level ---------------------------------
        out["inst_count_prev_loans"] = group.size().astype("int64")
        out["inst_count_installments_total"] = (
            group["inst_prev_count_installments"].sum(min_count=1)
        )
        out["inst_count_distinct_versions_total"] = (
            group["inst_prev_count_distinct_versions"].sum(min_count=1)
        )
        out["inst_num_instalment_max"] = group["inst_prev_num_instalment_max"].max()

        # ---- Payment delay ---------------------------------------------
        out["inst_delay_max"] = group["inst_prev_delay_max"].max()
        out["inst_delay_mean"] = group["inst_prev_delay_mean"].mean()
        out["inst_delay_sum"] = group["inst_prev_delay_sum"].sum(min_count=1)
        out["inst_delay_positive_count"] = (
            group["inst_prev_delay_positive_count"].sum(min_count=1)
        )
        out["inst_delay_positive_rate"] = (
            out["inst_delay_positive_count"] / out["inst_count_installments_total"]
        )

        # ---- Underpayment ----------------------------------------------
        out["inst_underpayment_max"] = group["inst_prev_underpayment_max"].max()
        out["inst_underpayment_mean"] = group["inst_prev_underpayment_mean"].mean()
        out["inst_underpayment_sum"] = (
            group["inst_prev_underpayment_sum"].sum(min_count=1)
        )
        out["inst_underpayment_positive_count"] = (
            group["inst_prev_underpayment_positive_count"].sum(min_count=1)
        )

        # ---- Payment ratio ---------------------------------------------
        out["inst_payment_ratio_mean"] = group["inst_prev_payment_ratio_mean"].mean()
        out["inst_payment_ratio_min"] = group["inst_prev_payment_ratio_min"].min()
        out["inst_underpaid_count"] = (
            group["inst_prev_underpaid_count"].sum(min_count=1)
        )
        out["inst_underpaid_rate"] = (
            out["inst_underpaid_count"] / out["inst_count_installments_total"]
        )

        # ---- Amounts ---------------------------------------------------
        out["inst_amt_instalment_sum"] = (
            group["inst_prev_amt_instalment_sum"].sum(min_count=1)
        )
        out["inst_amt_instalment_mean"] = group["inst_prev_amt_instalment_mean"].mean()
        out["inst_amt_payment_sum"] = (
            group["inst_prev_amt_payment_sum"].sum(min_count=1)
        )
        out["inst_amt_payment_mean"] = group["inst_prev_amt_payment_mean"].mean()
        out["inst_amt_payment_max"] = group["inst_prev_amt_payment_max"].max()

        # ---- Recency ---------------------------------------------------
        out["inst_days_instalment_min"] = group["inst_prev_days_instalment_min"].min()
        out["inst_days_instalment_max"] = group["inst_prev_days_instalment_max"].max()
        out["inst_days_entry_min"] = group["inst_prev_days_entry_min"].min()
        out["inst_days_entry_max"] = group["inst_prev_days_entry_max"].max()

        # ---- Most recent payment (across applicant's full history) -----
        sorted_prev_last = joined.sort_values(
            ["SK_ID_CURR", "inst_prev_days_entry_max"], kind="stable"
        )
        most_recent = (
            sorted_prev_last.groupby("SK_ID_CURR", sort=False)[
                ["inst_prev_delay_last",
                 "inst_prev_payment_ratio_last",
                 "inst_prev_amt_payment_last"]
            ]
            .last()
        )
        most_recent = most_recent.rename(
            columns={
                "inst_prev_delay_last": "inst_delay_last",
                "inst_prev_payment_ratio_last": "inst_payment_ratio_last",
                "inst_prev_amt_payment_last": "inst_amt_payment_last",
            }
        )
        out = out.join(most_recent, how="left")

        return out.reset_index(), join_diag

    # ---- helpers ---------------------------------------------------------

    def _add_presence_indicator(self, result: pd.DataFrame) -> pd.DataFrame:
        result["has_installment_history"] = 1
        result["has_installment_history"] = result["has_installment_history"].astype("int8")
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
        installments_rows: int,
        prev_app_rows: int,
        join_diag: dict[str, int],
        result: pd.DataFrame,
        duration: float,
    ) -> AggregationReport:
        feature_specs = self._build_feature_specs(result)
        return AggregationReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            input_installments_rows=installments_rows,
            input_previous_application_rows=prev_app_rows,
            unique_prev_before=join_diag["unique_prev_before"],
            unique_prev_after=join_diag["unique_prev_after"],
            orphan_prev_keys=join_diag["orphan_prev_keys"],
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
        if name == "has_installment_history":
            return FeatureSpec(
                name=name,
                source="installments_payments",
                aggregation="presence",
                description="1 if applicant has any installment history",
                dtype=dtype,
            )
        return FeatureSpec(
            name=name,
            source="installments_payments (rolled up via previous_application)",
            aggregation=self._infer_aggregation(name),
            description="",
            dtype=dtype,
        )

    def _infer_aggregation(self, name: str) -> str:
        for token in ("total", "count", "sum", "mean", "max", "min",
                      "std", "last", "rate"):
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
        logger.info("Installments aggregation complete")
        logger.info("  input installments rows:          %d", report.input_installments_rows)
        logger.info("  input previous_application rows:  %d", report.input_previous_application_rows)
        logger.info("  unique SK_ID_PREV before join:    %d", report.unique_prev_before)
        logger.info("  unique SK_ID_PREV matched:        %d", report.unique_prev_after)
        logger.info("  orphan SK_ID_PREV keys:           %d", report.orphan_prev_keys)
        logger.info("  output rows:                      %d", report.output_rows)
        logger.info("  features produced:                %d", report.n_features)
        logger.info("  duration:                         %.2fs", report.duration_seconds)
        logger.info("=" * 70)