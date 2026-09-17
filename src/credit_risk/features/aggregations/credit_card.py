"""
credit_card_balance aggregation → per-applicant features.

Home Credit's credit_card_balance table contains monthly snapshots of
credit-card accounts associated with prior loans. It is the sparsest
of the child tables: only ~5.6% of prior applications are credit cards.

Structure:

    credit_card_balance (one row per SK_ID_PREV per MONTHS_BALANCE)
        │
        └── keyed by SK_ID_PREV → previous_application.SK_ID_PREV
                                  → previous_application.SK_ID_CURR

This module performs a two-level aggregation:

    Level 1: credit_card_balance → per SK_ID_PREV
             (balances, utilization, drawings, payments, DPD, status)

    Level 2: per SK_ID_PREV → per SK_ID_CURR
             (via previous_application; aggregated across the
              applicant's full credit-card history)

The output is one row per applicant, with a stable schema and a presence
indicator (has_credit_card_history).

Coverage
--------
Only ~15-20% of applicants will appear in this table. During assembly,
the remaining ~80% of applicants will receive NaN for every cc_* feature.
This is expected and preserved — the absence of credit-card history is
itself informative, and this family is treated as a specialist feature
set in Stage C (feature transformation).

Derived quantities
------------------
Three quantities are computed per snapshot row:

    utilization          = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL
                           high values indicate credit-utilisation stress
    payment_to_min_ratio = AMT_PAYMENT_CURRENT / AMT_INST_MIN_REGULARITY
                           values near 1 indicate minimum-only payments
    net_drawing          = AMT_DRAWINGS_CURRENT - AMT_PAYMENT_CURRENT
                           positive values indicate growing balance

All ratios are guarded against zero or null denominators.

Null handling
-------------
- Many credit_card_balance columns are nullable, especially drawing and
  payment columns for months with no activity. Nulls propagate into
  derived quantities and aggregates. No imputation here.
- Applicants with no credit-card records are absent from the output.
  The assembly stage left-joins onto application_train.
- min_count=1 on sums preserves NaN for all-null groups.

Orphan handling
---------------
Rows whose SK_ID_PREV is absent from previous_application are dropped
by the inner join. Inspection found 11,372 orphan SK_ID_PREV values.
The report tracks orphans at the unique-SK_ID_PREV level.

Reference
---------
Data Quality Report §6.3 and §6.5 — credit_card_balance coverage of
previous_application is 5.6%, with 11,372 orphan SK_ID_PREV values.
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


logger = logging.getLogger("credit_risk.features.aggregations.credit_card")


# ---------------------------------------------------------------------------
# Known NAME_CONTRACT_STATUS values in credit_card_balance
# ---------------------------------------------------------------------------
# Observed during inspection. Unrecognized values fall into a residual
# "other" bucket, ensuring no rows are silently dropped if Home Credit
# adds a new status in a future refresh.
#
KNOWN_CONTRACT_STATUSES: tuple[str, ...] = (
    "Active",
    "Completed",
    "Demand",
    "Sent proposal",
    "Signed",
    "Approved",
    "Refused",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CreditCardAggregationConfig:
    """Resolved configuration for a credit-card aggregation run."""

    project_root: Path
    interim_dir: Path
    reports_dir: Path

    credit_card_file: str = "credit_card_balance.parquet"
    previous_application_file: str = "previous_application.parquet"
    output_file: str = "credit_card_aggregated.parquet"
    feature_catalogue_file: str = "credit_card_aggregated_features.json"

    def credit_card_path(self) -> Path:
        return self.interim_dir / self.credit_card_file

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
    """Outcome of a credit-card aggregation run."""

    generated_at: str
    input_credit_card_rows: int
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
            "input_credit_card_rows": self.input_credit_card_rows,
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
class CreditCardAggregator:
    """
    Aggregates credit_card_balance to one row per SK_ID_CURR.

    Two-stage pipeline:
        1. credit_card_balance → SK_ID_PREV level features
        2. per SK_ID_PREV + previous_application → SK_ID_CURR level features
    """

    config: CreditCardAggregationConfig
    report: AggregationReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> AggregationReport:
        start = time.monotonic()
        self.config.interim_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        credit_card = self._load(
            self.config.credit_card_path(), "credit_card_balance"
        )
        prev_app = self._load(
            self.config.previous_application_path(), "previous_application"
        )

        logger.info(
            "Loaded credit_card_balance: %d rows, previous_application: %d rows",
            len(credit_card), len(prev_app),
        )

        per_prev = self._aggregate_to_prev(credit_card)
        logger.info(
            "Aggregated credit_card_balance to %d unique SK_ID_PREV",
            len(per_prev),
        )

        per_curr, join_diag = self._aggregate_to_applicant(per_prev, prev_app)
        logger.info(
            "Aggregated credit-card to %d unique SK_ID_CURR", len(per_curr)
        )

        per_curr = self._add_presence_indicator(per_curr)

        self._write_output(per_curr)

        duration = time.monotonic() - start
        self.report = self._build_report(
            credit_card_rows=len(credit_card),
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

    # ---- level 1: credit_card_balance → per SK_ID_PREV ------------------

    def _aggregate_to_prev(self, credit_card: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce credit_card_balance (monthly rows) to one row per SK_ID_PREV.
        """
        logger.info("Aggregating credit_card_balance to SK_ID_PREV level")

        cc = credit_card.copy()

        # Normalize status
        cc["NAME_CONTRACT_STATUS"] = (
            cc["NAME_CONTRACT_STATUS"].astype(str).str.strip()
        )

        # ---- Derived quantities per snapshot row -----------------------
        # Utilization ratio (guarded against zero limit)
        safe_limit = cc["AMT_CREDIT_LIMIT_ACTUAL"].where(
            cc["AMT_CREDIT_LIMIT_ACTUAL"] > 0
        )
        cc["_utilization"] = cc["AMT_BALANCE"] / safe_limit

        # Payment-to-minimum ratio (guarded against zero minimum)
        safe_min = cc["AMT_INST_MIN_REGULARITY"].where(
            cc["AMT_INST_MIN_REGULARITY"] > 0
        )
        cc["_payment_to_min_ratio"] = cc["AMT_PAYMENT_CURRENT"] / safe_min

        # Net drawing this month
        cc["_net_drawing"] = cc["AMT_DRAWINGS_CURRENT"] - cc["AMT_PAYMENT_CURRENT"]

        # Positive DPD flag
        cc["_sk_dpd_positive"] = (cc["SK_DPD"] > 0).astype("int8")
        cc["_sk_dpd_def_positive"] = (cc["SK_DPD_DEF"] > 0).astype("int8")

        group = cc.groupby("SK_ID_PREV", sort=False)

        aggregated = pd.DataFrame(index=group.size().index)
        aggregated.index.name = "SK_ID_PREV"

        # ---- Counts ----------------------------------------------------
        aggregated["cc_prev_count_months"] = group.size().astype("int64")
        aggregated["cc_prev_months_balance_min"] = group["MONTHS_BALANCE"].min()
        aggregated["cc_prev_months_balance_max"] = group["MONTHS_BALANCE"].max()

        # ---- Balance ---------------------------------------------------
        aggregated["cc_prev_balance_max"] = group["AMT_BALANCE"].max()
        aggregated["cc_prev_balance_mean"] = group["AMT_BALANCE"].mean()
        aggregated["cc_prev_balance_min"] = group["AMT_BALANCE"].min()

        # ---- Credit limit ----------------------------------------------
        aggregated["cc_prev_credit_limit_max"] = group["AMT_CREDIT_LIMIT_ACTUAL"].max()
        aggregated["cc_prev_credit_limit_mean"] = group["AMT_CREDIT_LIMIT_ACTUAL"].mean()
        aggregated["cc_prev_credit_limit_min"] = group["AMT_CREDIT_LIMIT_ACTUAL"].min()

        # ---- Utilization -----------------------------------------------
        aggregated["cc_prev_utilization_max"] = group["_utilization"].max()
        aggregated["cc_prev_utilization_mean"] = group["_utilization"].mean()
        aggregated["cc_prev_utilization_min"] = group["_utilization"].min()

        # ---- Drawings --------------------------------------------------
        aggregated["cc_prev_drawings_total_sum"] = (
            group["AMT_DRAWINGS_CURRENT"].sum(min_count=1)
        )
        aggregated["cc_prev_drawings_total_max"] = group["AMT_DRAWINGS_CURRENT"].max()
        aggregated["cc_prev_drawings_atm_sum"] = (
            group["AMT_DRAWINGS_ATM_CURRENT"].sum(min_count=1)
        )
        aggregated["cc_prev_drawings_pos_sum"] = (
            group["AMT_DRAWINGS_POS_CURRENT"].sum(min_count=1)
        )
        aggregated["cc_prev_drawings_other_sum"] = (
            group["AMT_DRAWINGS_OTHER_CURRENT"].sum(min_count=1)
        )

        aggregated["cc_prev_cnt_drawings_total_sum"] = (
            group["CNT_DRAWINGS_CURRENT"].sum(min_count=1)
        )
        aggregated["cc_prev_cnt_drawings_atm_sum"] = (
            group["CNT_DRAWINGS_ATM_CURRENT"].sum(min_count=1)
        )
        aggregated["cc_prev_cnt_drawings_pos_sum"] = (
            group["CNT_DRAWINGS_POS_CURRENT"].sum(min_count=1)
        )

        # ---- Payments --------------------------------------------------
        aggregated["cc_prev_payment_total_sum"] = (
            group["AMT_PAYMENT_TOTAL_CURRENT"].sum(min_count=1)
        )
        aggregated["cc_prev_payment_total_mean"] = (
            group["AMT_PAYMENT_TOTAL_CURRENT"].mean()
        )
        aggregated["cc_prev_payment_current_sum"] = (
            group["AMT_PAYMENT_CURRENT"].sum(min_count=1)
        )
        aggregated["cc_prev_inst_min_regularity_mean"] = (
            group["AMT_INST_MIN_REGULARITY"].mean()
        )
        aggregated["cc_prev_inst_min_regularity_max"] = (
            group["AMT_INST_MIN_REGULARITY"].max()
        )

        aggregated["cc_prev_payment_to_min_ratio_mean"] = (
            group["_payment_to_min_ratio"].mean()
        )
        aggregated["cc_prev_payment_to_min_ratio_min"] = (
            group["_payment_to_min_ratio"].min()
        )

        aggregated["cc_prev_net_drawing_sum"] = (
            group["_net_drawing"].sum(min_count=1)
        )
        aggregated["cc_prev_net_drawing_mean"] = group["_net_drawing"].mean()

        # ---- Receivable ------------------------------------------------
        aggregated["cc_prev_receivable_principal_max"] = (
            group["AMT_RECEIVABLE_PRINCIPAL"].max()
        )
        aggregated["cc_prev_receivable_principal_mean"] = (
            group["AMT_RECEIVABLE_PRINCIPAL"].mean()
        )
        aggregated["cc_prev_receivable_total_max"] = (
            group["AMT_TOTAL_RECEIVABLE"].max()
        )
        aggregated["cc_prev_receivable_total_mean"] = (
            group["AMT_TOTAL_RECEIVABLE"].mean()
        )

        # ---- Installment count -----------------------------------------
        aggregated["cc_prev_cnt_instalment_mature_max"] = (
            group["CNT_INSTALMENT_MATURE_CUM"].max()
        )

        # ---- DPD -------------------------------------------------------
        aggregated["cc_prev_sk_dpd_max"] = group["SK_DPD"].max()
        aggregated["cc_prev_sk_dpd_mean"] = group["SK_DPD"].mean()
        aggregated["cc_prev_sk_dpd_positive_months"] = (
            group["_sk_dpd_positive"].sum().astype("int64")
        )
        aggregated["cc_prev_sk_dpd_def_max"] = group["SK_DPD_DEF"].max()
        aggregated["cc_prev_sk_dpd_def_mean"] = group["SK_DPD_DEF"].mean()
        aggregated["cc_prev_sk_dpd_def_positive_months"] = (
            group["_sk_dpd_def_positive"].sum().astype("int64")
        )

        # ---- Status counts per month -----------------------------------
        for status in KNOWN_CONTRACT_STATUSES:
            col_name = f"cc_prev_status_{self._slug(status)}"
            mask = cc["NAME_CONTRACT_STATUS"] == status
            counts = cc.loc[mask].groupby("SK_ID_PREV", sort=False).size()
            aggregated[col_name] = (
                counts.reindex(aggregated.index).fillna(0).astype("int64")
            )

        known_mask = cc["NAME_CONTRACT_STATUS"].isin(KNOWN_CONTRACT_STATUSES)
        other_counts = cc.loc[~known_mask].groupby("SK_ID_PREV", sort=False).size()
        aggregated["cc_prev_status_other"] = (
            other_counts.reindex(aggregated.index).fillna(0).astype("int64")
        )

        # ---- Most recent month's snapshot values -----------------------
        sorted_cc = cc.sort_values(
            ["SK_ID_PREV", "MONTHS_BALANCE"], kind="stable"
        )
        last_snapshot = (
            sorted_cc.groupby("SK_ID_PREV", sort=False)[
                ["AMT_BALANCE",
                 "AMT_CREDIT_LIMIT_ACTUAL",
                 "_utilization",
                 "SK_DPD",
                 "AMT_PAYMENT_CURRENT",
                 "CNT_INSTALMENT_MATURE_CUM"]
            ]
            .last()
        )
        last_snapshot = last_snapshot.rename(columns={
            "AMT_BALANCE": "cc_prev_balance_last",
            "AMT_CREDIT_LIMIT_ACTUAL": "cc_prev_credit_limit_last",
            "_utilization": "cc_prev_utilization_last",
            "SK_DPD": "cc_prev_sk_dpd_last",
            "AMT_PAYMENT_CURRENT": "cc_prev_payment_current_last",
            "CNT_INSTALMENT_MATURE_CUM": "cc_prev_cnt_instalment_mature_last",
        })
        aggregated = aggregated.join(last_snapshot, how="left")

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

        Inner join on SK_ID_PREV. Orphan rows are dropped.
        """
        logger.info(
            "Joining credit-card per-SK_ID_PREV features to previous_application"
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

        # ---- Counts ----------------------------------------------------
        out["cc_count_prev_loans"] = group.size().astype("int64")
        out["cc_count_total_months"] = (
            group["cc_prev_count_months"].sum(min_count=1)
        )
        out["cc_months_balance_min"] = group["cc_prev_months_balance_min"].min()
        out["cc_months_balance_max"] = group["cc_prev_months_balance_max"].max()

        # ---- Balance ---------------------------------------------------
        out["cc_balance_max"] = group["cc_prev_balance_max"].max()
        out["cc_balance_mean"] = group["cc_prev_balance_mean"].mean()
        out["cc_balance_min"] = group["cc_prev_balance_min"].min()

        # ---- Credit limit ----------------------------------------------
        out["cc_credit_limit_max"] = group["cc_prev_credit_limit_max"].max()
        out["cc_credit_limit_mean"] = group["cc_prev_credit_limit_mean"].mean()
        out["cc_credit_limit_min"] = group["cc_prev_credit_limit_min"].min()

        # ---- Utilization -----------------------------------------------
        out["cc_utilization_max"] = group["cc_prev_utilization_max"].max()
        out["cc_utilization_mean"] = group["cc_prev_utilization_mean"].mean()
        out["cc_utilization_min"] = group["cc_prev_utilization_min"].min()

        # ---- Drawings --------------------------------------------------
        out["cc_drawings_total_sum"] = (
            group["cc_prev_drawings_total_sum"].sum(min_count=1)
        )
        out["cc_drawings_total_max"] = group["cc_prev_drawings_total_max"].max()
        out["cc_drawings_atm_sum"] = (
            group["cc_prev_drawings_atm_sum"].sum(min_count=1)
        )
        out["cc_drawings_pos_sum"] = (
            group["cc_prev_drawings_pos_sum"].sum(min_count=1)
        )
        out["cc_drawings_other_sum"] = (
            group["cc_prev_drawings_other_sum"].sum(min_count=1)
        )
        out["cc_cnt_drawings_total_sum"] = (
            group["cc_prev_cnt_drawings_total_sum"].sum(min_count=1)
        )
        out["cc_cnt_drawings_atm_sum"] = (
            group["cc_prev_cnt_drawings_atm_sum"].sum(min_count=1)
        )
        out["cc_cnt_drawings_pos_sum"] = (
            group["cc_prev_cnt_drawings_pos_sum"].sum(min_count=1)
        )

        # ---- Payments --------------------------------------------------
        out["cc_payment_total_sum"] = (
            group["cc_prev_payment_total_sum"].sum(min_count=1)
        )
        out["cc_payment_total_mean"] = group["cc_prev_payment_total_mean"].mean()
        out["cc_payment_current_sum"] = (
            group["cc_prev_payment_current_sum"].sum(min_count=1)
        )
        out["cc_inst_min_regularity_mean"] = (
            group["cc_prev_inst_min_regularity_mean"].mean()
        )
        out["cc_inst_min_regularity_max"] = (
            group["cc_prev_inst_min_regularity_max"].max()
        )
        out["cc_payment_to_min_ratio_mean"] = (
            group["cc_prev_payment_to_min_ratio_mean"].mean()
        )
        out["cc_payment_to_min_ratio_min"] = (
            group["cc_prev_payment_to_min_ratio_min"].min()
        )
        out["cc_net_drawing_sum"] = (
            group["cc_prev_net_drawing_sum"].sum(min_count=1)
        )
        out["cc_net_drawing_mean"] = group["cc_prev_net_drawing_mean"].mean()

        # ---- Receivable ------------------------------------------------
        out["cc_receivable_principal_max"] = (
            group["cc_prev_receivable_principal_max"].max()
        )
        out["cc_receivable_principal_mean"] = (
            group["cc_prev_receivable_principal_mean"].mean()
        )
        out["cc_receivable_total_max"] = (
            group["cc_prev_receivable_total_max"].max()
        )
        out["cc_receivable_total_mean"] = (
            group["cc_prev_receivable_total_mean"].mean()
        )

        # ---- Installment count -----------------------------------------
        out["cc_cnt_instalment_mature_max"] = (
            group["cc_prev_cnt_instalment_mature_max"].max()
        )

        # ---- DPD -------------------------------------------------------
        out["cc_sk_dpd_max"] = group["cc_prev_sk_dpd_max"].max()
        out["cc_sk_dpd_mean"] = group["cc_prev_sk_dpd_mean"].mean()
        out["cc_sk_dpd_positive_months_total"] = (
            group["cc_prev_sk_dpd_positive_months"].sum(min_count=1)
        )
        out["cc_sk_dpd_def_max"] = group["cc_prev_sk_dpd_def_max"].max()
        out["cc_sk_dpd_def_mean"] = group["cc_prev_sk_dpd_def_mean"].mean()
        out["cc_sk_dpd_def_positive_months_total"] = (
            group["cc_prev_sk_dpd_def_positive_months"].sum(min_count=1)
        )

        # ---- Status counts at applicant level --------------------------
        for status in KNOWN_CONTRACT_STATUSES:
            slug = self._slug(status)
            out[f"cc_count_status_{slug}"] = (
                group[f"cc_prev_status_{slug}"].sum(min_count=1).astype("int64")
            )
        out["cc_count_status_other"] = (
            group["cc_prev_status_other"].sum(min_count=1).astype("int64")
        )

        # ---- Derived rates ---------------------------------------------
        total_months = out["cc_count_total_months"].replace(0, pd.NA)

        out["cc_rate_dpd_positive"] = (
            out["cc_sk_dpd_positive_months_total"] / total_months
        )
        out["cc_rate_dpd_def_positive"] = (
            out["cc_sk_dpd_def_positive_months_total"] / total_months
        )

        # ---- Most recent month's snapshot (across the applicant) -------
        sorted_prev_last = joined.sort_values(
            ["SK_ID_CURR", "cc_prev_months_balance_max"], kind="stable"
        )
        most_recent = (
            sorted_prev_last.groupby("SK_ID_CURR", sort=False)[
                ["cc_prev_balance_last",
                 "cc_prev_credit_limit_last",
                 "cc_prev_utilization_last",
                 "cc_prev_sk_dpd_last",
                 "cc_prev_payment_current_last",
                 "cc_prev_cnt_instalment_mature_last"]
            ]
            .last()
        )
        most_recent = most_recent.rename(columns={
            "cc_prev_balance_last": "cc_balance_last",
            "cc_prev_credit_limit_last": "cc_credit_limit_last",
            "cc_prev_utilization_last": "cc_utilization_last",
            "cc_prev_sk_dpd_last": "cc_sk_dpd_last",
            "cc_prev_payment_current_last": "cc_payment_current_last",
            "cc_prev_cnt_instalment_mature_last": "cc_cnt_instalment_mature_last",
        })
        out = out.join(most_recent, how="left")

        return out.reset_index(), join_diag

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _slug(value: str) -> str:
        """Normalize a category value into a feature-name-friendly slug."""
        return value.lower().replace(" ", "_").replace("-", "_")

    def _add_presence_indicator(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Add has_credit_card_history flag.

        Constant 1 in this table by construction. Becomes meaningful
        during assembly when left-joined onto application_train.
        """
        result["has_credit_card_history"] = 1
        result["has_credit_card_history"] = result["has_credit_card_history"].astype("int8")
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
        credit_card_rows: int,
        prev_app_rows: int,
        join_diag: dict[str, int],
        result: pd.DataFrame,
        duration: float,
    ) -> AggregationReport:
        feature_specs = self._build_feature_specs(result)
        return AggregationReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            input_credit_card_rows=credit_card_rows,
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
        if name == "has_credit_card_history":
            return FeatureSpec(
                name=name,
                source="credit_card_balance",
                aggregation="presence",
                description="1 if applicant has any credit-card history",
                dtype=dtype,
            )
        return FeatureSpec(
            name=name,
            source="credit_card_balance (rolled up via previous_application)",
            aggregation=self._infer_aggregation(name),
            description="",
            dtype=dtype,
        )

    def _infer_aggregation(self, name: str) -> str:
        for token in ("total", "count", "sum", "mean", "max", "min",
                      "std", "last", "rate", "months"):
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
        logger.info("Credit-card aggregation complete")
        logger.info("  input credit_card rows:            %d", report.input_credit_card_rows)
        logger.info("  input previous_application rows:   %d", report.input_previous_application_rows)
        logger.info("  unique SK_ID_PREV before join:     %d", report.unique_prev_before)
        logger.info("  unique SK_ID_PREV matched:         %d", report.unique_prev_after)
        logger.info("  orphan SK_ID_PREV keys:            %d", report.orphan_prev_keys)
        logger.info("  output rows:                       %d", report.output_rows)
        logger.info("  features produced:                 %d", report.n_features)
        logger.info("  duration:                          %.2fs", report.duration_seconds)
        logger.info("=" * 70)