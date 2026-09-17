"""
WoE binning, IV computation, and feature selection for the champion scorecard.

This module is the analytical core of Stage C. It:

    1. Fits a WoE binning per feature, on the DEV split only.
    2. Computes Information Value (IV) for each feature.
    3. Selects features based on IV thresholds and correlation pruning.
    4. Applies the DEV-derived binning to VAL and HOLDOUT splits.
    5. Emits WoE-encoded matrices for the champion scorecard.
    6. Persists per-feature binning tables for reproducibility at serving time.

Design notes
------------
- All fitting happens on the DEV split. VAL and HOLDOUT are transformed
  using DEV-derived parameters only. This is the leakage discipline that
  makes the resulting scorecard valid on unseen data.
- Monotonicity constraints are applied per feature via a whitelist. Only
  features with an economically justified direction are constrained.
  Forcing monotonicity on genuinely non-monotonic features produces a
  wrong model.
- Nulls are treated as their own bin. This resolves the sparsity problem
  structurally — no separate missing indicators are needed.
- IV thresholds drive feature selection. Features below IV_MIN are
  dropped. Features above IV_MAX are flagged for investigation.
- Correlation pruning removes redundant features. Correlation is computed
  on WoE-encoded values, which is what the model will see, not on raw
  values. This is the correct choice for scorecard feature selection.
- Feature names appear in reports as either "skipped" (constant or
  all-null in DEV — benign) or "failed" (fit raised an exception).

Persistence
-----------
Per-feature binning tables are written to artifacts/binning/<feature>.json.
Each table contains the bin boundaries, WoE values, IV, and null handling
required to reproduce the transformation at serving time.

Outputs
-------
- artifacts/binning/<feature>.json           (one file per fitted feature)
- artifacts/reports/binning_summary.json     (IV per feature, decisions)
- artifacts/reports/feature_selection.json   (kept vs dropped)
- data/processed/champion_train.parquet      (WoE-encoded)
- data/processed/champion_val.parquet
- data/processed/champion_holdout.parquet
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger("credit_risk.features.binning")


# ---------------------------------------------------------------------------
# Monotonicity whitelist
# ---------------------------------------------------------------------------
# -1 = higher feature value → lower risk (WoE decreases)
# +1 = higher feature value → higher risk (WoE increases)
# Absence from this dict = no monotonicity constraint
#
MONOTONICITY_WHITELIST: dict[str, int] = {
    # Base features
    "DAYS_BIRTH": -1,
    "AMT_INCOME_TOTAL": -1,
    "AMT_CREDIT": +1,
    "AMT_ANNUITY": +1,
    "EXT_SOURCE_1": -1,
    "EXT_SOURCE_2": -1,
    "EXT_SOURCE_3": -1,
    # Bureau
    "bureau_count_credits": +1,
    "bureau_count_active": +1,
    "bureau_count_bad_debt": +1,
    "bureau_credit_sum_overdue_max": +1,
    "bureau_credit_max_overdue_max": +1,
    # Installments
    "inst_delay_positive_rate": +1,
    "inst_delay_max": +1,
    "inst_underpaid_rate": +1,
    # POS/CASH
    "pos_sk_dpd_max": +1,
    "pos_sk_dpd_positive_months_total": +1,
    # Credit card
    "cc_sk_dpd_max": +1,
    "cc_utilization_last": +1,
    # bureau_balance
    "bb_max_status_max": +1,
    "bb_months_at_3_plus_total": +1,
}


# Feature selection thresholds
IV_MIN = 0.01
IV_MAX = 0.50
CORRELATION_MAX = 0.90

# Binning parameters
MAX_N_BINS = 6
MIN_BIN_SIZE = 0.05
MIN_BIN_N_EVENT = 100


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BinningConfig:
    """Resolved configuration for a binning run."""

    project_root: Path
    processed_dir: Path
    splits_dir: Path
    binning_dir: Path
    reports_dir: Path

    input_file: str = "application_train_features.parquet"
    splits_file: str = "splits.json"

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    # Output file names
    train_output: str = "champion_train.parquet"
    val_output: str = "champion_val.parquet"
    holdout_output: str = "champion_holdout.parquet"
    summary_output: str = "binning_summary.json"
    selection_output: str = "feature_selection.json"

    # Binning parameters
    max_n_bins: int = MAX_N_BINS
    min_bin_size: float = MIN_BIN_SIZE
    min_bin_n_event: int = MIN_BIN_N_EVENT
    iv_min: float = IV_MIN
    iv_max: float = IV_MAX
    correlation_max: float = CORRELATION_MAX

    # Feature exclusion — identifiers and target are never binned
    excluded_features: tuple[str, ...] = (
        "SK_ID_CURR",
        "TARGET",
    )

    def input_path(self) -> Path:
        return self.processed_dir / self.input_file

    def splits_path(self) -> Path:
        return self.splits_dir / self.splits_file

    def train_output_path(self) -> Path:
        return self.processed_dir / self.train_output

    def val_output_path(self) -> Path:
        return self.processed_dir / self.val_output

    def holdout_output_path(self) -> Path:
        return self.processed_dir / self.holdout_output

    def summary_path(self) -> Path:
        return self.reports_dir / self.summary_output

    def selection_path(self) -> Path:
        return self.reports_dir / self.selection_output

    def binning_feature_path(self, feature: str) -> Path:
        return self.binning_dir / f"{feature}.json"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FeatureBinning:
    """Binning result for one feature."""

    feature: str
    dtype: str
    monotonic_direction: int
    iv: float
    n_bins_total: int
    n_bins_regular: int
    has_null_bin: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "dtype": self.dtype,
            "monotonic_direction": self.monotonic_direction,
            "iv": round(self.iv, 6),
            "n_bins_total": self.n_bins_total,
            "n_bins_regular": self.n_bins_regular,
            "has_null_bin": self.has_null_bin,
        }


@dataclass
class FeatureDecision:
    """Feature selection decision for one feature."""

    feature: str
    iv: float
    kept: bool
    reason: str
    correlated_with: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "iv": round(self.iv, 6),
            "kept": self.kept,
            "reason": self.reason,
            "correlated_with": self.correlated_with,
        }


@dataclass
class BinningReport:
    """Summary of a binning run."""

    generated_at: str
    n_features_total: int
    n_features_binned: int
    n_features_skipped: int
    n_features_failed: int
    n_features_kept: int
    n_features_dropped_low_iv: int
    n_features_dropped_high_iv: int
    n_features_dropped_correlated: int
    duration_seconds: float
    train_shape: tuple[int, int]
    val_shape: tuple[int, int]
    holdout_shape: tuple[int, int]
    skipped_features: list[str] = field(default_factory=list)
    failed_features: list[str] = field(default_factory=list)
    binnings: list[FeatureBinning] = field(default_factory=list)
    decisions: list[FeatureDecision] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "n_features_total": self.n_features_total,
            "n_features_binned": self.n_features_binned,
            "n_features_skipped": self.n_features_skipped,
            "n_features_failed": self.n_features_failed,
            "n_features_kept": self.n_features_kept,
            "n_features_dropped_low_iv": self.n_features_dropped_low_iv,
            "n_features_dropped_high_iv": self.n_features_dropped_high_iv,
            "n_features_dropped_correlated": self.n_features_dropped_correlated,
            "duration_seconds": round(self.duration_seconds, 3),
            "train_shape": list(self.train_shape),
            "val_shape": list(self.val_shape),
            "holdout_shape": list(self.holdout_shape),
            "skipped_features": self.skipped_features,
            "failed_features": self.failed_features,
            "binnings": [b.to_dict() for b in self.binnings],
            "decisions": [d.to_dict() for d in self.decisions],
        }


# ---------------------------------------------------------------------------
# WoE Binner
# ---------------------------------------------------------------------------

@dataclass
class WoEBinner:
    """
    Fits WoE binning on a feature, produces IV, and serialises the binning
    for reuse.
    """

    config: BinningConfig
    binnings: dict[str, FeatureBinning] = field(default_factory=dict)

    # ---- public API ------------------------------------------------------

    def fit_and_transform(
        self,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        X_val: pd.DataFrame,
        X_holdout: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[FeatureBinning], list[str], list[str]]:
        """
        Fit binning on DEV, transform DEV/VAL/HOLDOUT.

        Returns:
            (dev_transformed, val_transformed, holdout_transformed,
             binnings, skipped_features, failed_features)
        """
        try:
            from optbinning import OptimalBinning  # noqa: F401
        except ImportError:
            raise ImportError(
                "optbinning is not installed. Install with: pip install optbinning"
            )

        feature_cols = [
            c for c in X_dev.columns if c not in self.config.excluded_features
        ]
        logger.info("Fitting WoE binning for %d features", len(feature_cols))

        dev_transformed: dict[str, pd.Series] = {}
        val_transformed: dict[str, pd.Series] = {}
        holdout_transformed: dict[str, pd.Series] = {}
        binnings: list[FeatureBinning] = []
        skipped: list[str] = []
        failed: list[str] = []

        for i, feat in enumerate(feature_cols, 1):
            try:
                b, s_dev, s_val, s_holdout, skip_reason = self._fit_single_feature(
                    feat=feat,
                    X_dev=X_dev,
                    y_dev=y_dev,
                    X_val=X_val,
                    X_holdout=X_holdout,
                )
                if b is None:
                    # Skipped (constant or all-null in DEV)
                    skipped.append(feat)
                    continue

                binnings.append(b)
                dev_transformed[feat] = s_dev
                val_transformed[feat] = s_val
                holdout_transformed[feat] = s_holdout
                self.binnings[feat] = b

            except Exception as exc:
                logger.warning("  failed %s: %s", feat, exc)
                failed.append(feat)

            if i % 25 == 0:
                logger.info("  progress: %d/%d features", i, len(feature_cols))

        logger.info(
            "Binning complete: %d succeeded, %d skipped, %d failed",
            len(binnings), len(skipped), len(failed),
        )
        if skipped:
            logger.info("  skipped (constant or all-null): %s", skipped)
        if failed:
            logger.warning("  failed: %s", failed)

        # Build transformed DataFrames
        dev_df = pd.DataFrame(dev_transformed, index=X_dev.index)
        val_df = pd.DataFrame(val_transformed, index=X_val.index)
        holdout_df = pd.DataFrame(holdout_transformed, index=X_holdout.index)

        # Preserve the key column
        for df, source in [
            (dev_df, X_dev),
            (val_df, X_val),
            (holdout_df, X_holdout),
        ]:
            df.insert(0, self.config.key_column, source[self.config.key_column].values)

        return dev_df, val_df, holdout_df, binnings, skipped, failed

    # ---- per-feature fit -------------------------------------------------

    def _fit_single_feature(
        self,
        feat: str,
        X_dev: pd.DataFrame,
        y_dev: pd.Series,
        X_val: pd.DataFrame,
        X_holdout: pd.DataFrame,
    ) -> tuple[FeatureBinning | None, pd.Series | None, pd.Series | None, pd.Series | None, str]:
        """
        Fit binning for one feature.

        Returns (FeatureBinning, s_dev, s_val, s_holdout, skip_reason).
        If FeatureBinning is None, the feature was skipped and skip_reason
        explains why.
        """
        from optbinning import OptimalBinning

        dev_col = X_dev[feat]
        dtype = self._classify_dtype(dev_col)

        if dtype == "constant":
            return None, None, None, None, "constant in DEV"

        monotonic = MONOTONICITY_WHITELIST.get(feat, 0)
        monotonic_trend = self._monotonic_trend_arg(monotonic)

        binner = OptimalBinning(
            name=feat,
            dtype="categorical" if dtype == "categorical" else "numerical",
            max_n_bins=self.config.max_n_bins,
            min_bin_size=self.config.min_bin_size,
            min_bin_n_event=self.config.min_bin_n_event,
            monotonic_trend=monotonic_trend,
        )

        # Fit on DEV
        binner.fit(dev_col, y_dev)

        # Extract everything from the binning table in one pass
        extract = self._extract_binning_table(binner.binning_table.build())

        if extract.iv is None:
            raise ValueError(
                f"Could not extract IV for '{feat}'; binning table malformed"
            )

        # Transform all three splits
        s_dev = pd.Series(
            binner.transform(dev_col, metric="woe"),
            index=dev_col.index,
            name=feat,
        )
        s_val = pd.Series(
            binner.transform(X_val[feat], metric="woe"),
            index=X_val.index,
            name=feat,
        )
        s_holdout = pd.Series(
            binner.transform(X_holdout[feat], metric="woe"),
            index=X_holdout.index,
            name=feat,
        )

        # Persist binning artifact
        self._persist_binning(
            feat=feat,
            binner=binner,
            dtype=dtype,
            monotonic=monotonic,
            extract=extract,
        )

        return (
            FeatureBinning(
                feature=feat,
                dtype=dtype,
                monotonic_direction=monotonic,
                iv=extract.iv,
                n_bins_total=extract.n_bins_total,
                n_bins_regular=extract.n_bins_regular,
                has_null_bin=extract.has_null_bin,
            ),
            s_dev,
            s_val,
            s_holdout,
            "",
        )

    # ---- binning table extraction ----------------------------------------

    @dataclass
    class _BinningExtract:
        """Parsed binning table — records, IV, bin counts, null-bin flag."""

        records: list[dict[str, Any]]
        iv: float | None
        n_bins_total: int
        n_bins_regular: int
        has_null_bin: bool

    def _extract_binning_table(self, binning_table: pd.DataFrame) -> "_BinningExtract":
        """
        Parse an optbinning binning table into serialisable records.

        The last row of the table is always the Totals row — identified by
        position, not by label, since the label format is version-dependent.

        Regular bins are those without "Special" or "Missing" in their label.
        """
        records: list[dict[str, Any]] = []
        has_null_bin = False
        n_bins_regular = 0
        iv_value: float | None = None

        n_rows = len(binning_table)

        for idx in range(n_rows):
            row = binning_table.iloc[idx]

            # Last row is Totals — capture IV from it, then skip
            if idx == n_rows - 1:
                iv_value = self._safe_float(row["IV"])
                continue

            bin_str = self._coerce_bin_label(row["Bin"])

            is_special = ("Special" in bin_str) or ("Missing" in bin_str)
            if is_special:
                has_null_bin = True
            else:
                n_bins_regular += 1

            records.append({
                "bin": bin_str,
                "count": self._safe_int(row["Count"]),
                "count_rate": self._safe_float(row["Count (%)"]),
                "non_event": self._safe_int(row["Non-event"]),
                "event": self._safe_int(row["Event"]),
                "event_rate": self._safe_float(row["Event rate"]),
                "woe": self._safe_float(row["WoE"]),
                "iv": self._safe_float(row["IV"]),
                "is_special": is_special,
            })

        return WoEBinner._BinningExtract(
            records=records,
            iv=iv_value,
            n_bins_total=len(records),
            n_bins_regular=n_bins_regular,
            has_null_bin=has_null_bin,
        )

    # ---- persistence -----------------------------------------------------

    def _persist_binning(
        self,
        feat: str,
        binner: Any,
        dtype: str,
        monotonic: int,
        extract: "_BinningExtract",
    ) -> None:
        """Persist the fitted binner to disk as a JSON artifact."""
        payload = {
            "feature": feat,
            "dtype": dtype,
            "monotonic_direction": monotonic,
            "iv": round(extract.iv, 6) if extract.iv is not None else None,
            "splits": self._coerce_splits(binner.splits),
            "n_bins_total": extract.n_bins_total,
            "n_bins_regular": extract.n_bins_regular,
            "has_null_bin": extract.has_null_bin,
            "binning_table": extract.records,
        }

        out_path = self.config.binning_feature_path(feat)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    # ---- type classification ---------------------------------------------

    @staticmethod
    def _classify_dtype(series: pd.Series) -> str:
        """
        Classify a feature as numeric, categorical, or constant.

        - constant: <= 1 distinct non-null value in the input sample
        - categorical: object dtype, or integer with <= 20 distinct values
        - numerical: everything else numeric
        """
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            return "constant"
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series) and n_unique <= 20:
                return "categorical"
            return "numerical"
        return "categorical"

    @staticmethod
    def _monotonic_trend_arg(direction: int) -> str:
        """Convert our integer direction to optbinning's string argument."""
        if direction == 0:
            return "auto"
        if direction == +1:
            return "ascending"
        return "descending"

    # ---- coercion helpers ------------------------------------------------

    @staticmethod
    def _safe_float(x: Any) -> float | None:
        """Convert to float, returning None for NaN, empty strings, or non-numerics."""
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        try:
            f = float(x)
        except (TypeError, ValueError):
            return None
        if np.isnan(f) or np.isinf(f):
            return None
        return f

    @staticmethod
    def _safe_int(x: Any) -> int | None:
        """Convert to int, returning None for NaN, empty strings, or non-numerics."""
        if x is None:
            return None
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_bin_label(value: Any) -> str:
        """
        Coerce a Bin column value to a scalar string safely.

        Optbinning sometimes stores tuples or numpy arrays in the Bin
        column for special bins. Any non-scalar value would break
        equality comparisons.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return str(value)
        except Exception:
            return repr(value)

    @staticmethod
    def _coerce_splits(splits: Any) -> list[float | str | None]:
        """
        Coerce optbinning splits to a JSON-safe list of floats.

        Splits are typically numpy floats including -inf and +inf. We
        convert infinities to None (JSON has no infinity).
        """
        if splits is None:
            return []

        try:
            raw = list(splits)
        except TypeError:
            return []

        result: list[float | str | None] = []
        for s in raw:
            # numpy scalars have .item()
            if hasattr(s, "item"):
                try:
                    s = s.item()
                except Exception:
                    pass

            if isinstance(s, (int, float)):
                if np.isnan(s) or np.isinf(s):
                    result.append(None)
                else:
                    result.append(float(s))
            else:
                result.append(str(s))

        return result


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

@dataclass
class FeatureSelector:
    """Applies IV thresholds and correlation pruning."""

    config: BinningConfig

    def select(
        self,
        binnings: list[FeatureBinning],
        transformed_dev: pd.DataFrame,
    ) -> list[FeatureDecision]:
        """Produce a keep/drop decision per feature."""
        logger.info("Applying feature selection")

        decisions: dict[str, FeatureDecision] = {}

        # ---- Step 1: IV threshold -----------------------------------------
        for b in binnings:
            if b.iv < self.config.iv_min:
                decisions[b.feature] = FeatureDecision(
                    feature=b.feature,
                    iv=b.iv,
                    kept=False,
                    reason=f"IV {b.iv:.4f} < threshold {self.config.iv_min}",
                )
            elif b.iv > self.config.iv_max:
                decisions[b.feature] = FeatureDecision(
                    feature=b.feature,
                    iv=b.iv,
                    kept=False,
                    reason=f"IV {b.iv:.4f} > threshold {self.config.iv_max} (suspicious)",
                )
            else:
                decisions[b.feature] = FeatureDecision(
                    feature=b.feature,
                    iv=b.iv,
                    kept=True,
                    reason=f"IV {b.iv:.4f} within range",
                )

        # ---- Step 2: Correlation pruning ----------------------------------
        kept_features = [f for f, d in decisions.items() if d.kept]
        if len(kept_features) > 1:
            logger.info(
                "  correlation pruning among %d kept features",
                len(kept_features),
            )
            # Correlation on WoE-encoded values — what the model sees
            corr_matrix = transformed_dev[kept_features].corr().abs()

            # Higher IV wins; sort descending
            kept_sorted = sorted(
                kept_features,
                key=lambda f: -decisions[f].iv,
            )

            dropped: set[str] = set()
            for i, f_a in enumerate(kept_sorted):
                if f_a in dropped:
                    continue
                for f_b in kept_sorted[i + 1:]:
                    if f_b in dropped:
                        continue
                    corr = corr_matrix.loc[f_a, f_b]
                    if corr > self.config.correlation_max:
                        decisions[f_b] = FeatureDecision(
                            feature=f_b,
                            iv=decisions[f_b].iv,
                            kept=False,
                            reason=(
                                f"correlated with {f_a} (|r|={corr:.3f} > "
                                f"{self.config.correlation_max})"
                            ),
                            correlated_with=f_a,
                        )
                        dropped.add(f_b)

            logger.info("  dropped %d features by correlation", len(dropped))

        return list(decisions.values())


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

@dataclass
class BinningPipeline:
    """
    Coordinates fitting, transforming, and selecting features.

    Runs the full Stage C binning workflow:
        1. Load assembled matrix and split indices
        2. Fit WoE binning per feature on DEV
        3. Compute IV per feature
        4. Select features by IV and correlation
        5. Emit WoE-encoded matrices for DEV, VAL, HOLDOUT
        6. Persist binnings and reports
    """

    config: BinningConfig
    report: BinningReport | None = None

    def run(self) -> BinningReport:
        start = time.monotonic()

        self.config.binning_dir.mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        df = self._load_input()
        splits = self._load_splits()
        X_dev, y_dev, X_val, y_val, X_holdout, y_holdout = self._apply_splits(df, splits)

        logger.info(
            "Splits applied: dev=%d, val=%d, holdout=%d",
            len(X_dev), len(X_val), len(X_holdout),
        )

        # Fit binning and transform all splits
        binner = WoEBinner(config=self.config)
        dev_t, val_t, holdout_t, binnings, skipped, failed = binner.fit_and_transform(
            X_dev=X_dev,
            y_dev=y_dev,
            X_val=X_val,
            X_holdout=X_holdout,
        )

        # Feature selection
        selector = FeatureSelector(config=self.config)
        decisions = selector.select(binnings=binnings, transformed_dev=dev_t)

        kept = [d.feature for d in decisions if d.kept]
        logger.info(
            "Feature selection: %d kept, %d dropped",
            len(kept), len(decisions) - len(kept),
        )

        # Restrict transformed matrices to kept features + key
        key = self.config.key_column
        keep_cols = [key] + sorted(kept)
        dev_final = dev_t[keep_cols].copy()
        val_final = val_t[keep_cols].copy()
        holdout_final = holdout_t[keep_cols].copy()

        # Attach target
        dev_final[self.config.target_column] = y_dev.values
        val_final[self.config.target_column] = y_val.values
        holdout_final[self.config.target_column] = y_holdout.values

        # Write outputs
        self._write_split(dev_final, self.config.train_output_path())
        self._write_split(val_final, self.config.val_output_path())
        self._write_split(holdout_final, self.config.holdout_output_path())

        # Report
        duration = time.monotonic() - start
        self.report = self._build_report(
            binnings=binnings,
            decisions=decisions,
            skipped=skipped,
            failed=failed,
            dev_shape=dev_final.shape,
            val_shape=val_final.shape,
            holdout_shape=holdout_final.shape,
            n_features_total=len(X_dev.columns),
            duration=duration,
        )
        self._write_reports(self.report)
        self._log_summary(self.report)
        return self.report

    def _write_reports(self, report: BinningReport) -> None:
        # Full summary
        with self.config.summary_path().open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # Selection manifest — lighter, easier to scan
        selection = {
            "generated_at": report.generated_at,
            "n_kept": report.n_features_kept,
            "n_dropped_low_iv": report.n_features_dropped_low_iv,
            "n_dropped_high_iv": report.n_features_dropped_high_iv,
            "n_dropped_correlated": report.n_features_dropped_correlated,
            "n_skipped": report.n_features_skipped,
            "n_failed": report.n_features_failed,
            "kept_features": sorted(
                d.feature for d in report.decisions if d.kept
            ),
            "dropped_features": sorted(
                d.feature for d in report.decisions if not d.kept
            ),
            "skipped_features": report.skipped_features,
            "failed_features": report.failed_features,
            "decisions": [d.to_dict() for d in report.decisions],
        }
        with self.config.selection_path().open("w", encoding="utf-8") as f:
            json.dump(selection, f, indent=2, default=str)

        logger.info("Binning summary written: %s", self.config.summary_path().name)
        logger.info("Feature selection written: %s", self.config.selection_path().name)

    def _log_summary(self, report: BinningReport) -> None:
        logger.info("=" * 70)
        logger.info("Binning and feature selection complete")
        logger.info("  features total:             %d", report.n_features_total)
        logger.info("  features binned:            %d", report.n_features_binned)
        logger.info("  features skipped:           %d", report.n_features_skipped)
        logger.info("  features failed:            %d", report.n_features_failed)
        logger.info("  features kept:              %d", report.n_features_kept)
        logger.info("  dropped (IV below min):     %d", report.n_features_dropped_low_iv)
        logger.info("  dropped (IV above max):     %d", report.n_features_dropped_high_iv)
        logger.info("  dropped (correlated):       %d", report.n_features_dropped_correlated)
        logger.info("  ---")
        logger.info("  train shape:                %d × %d", *report.train_shape)
        logger.info("  val shape:                  %d × %d", *report.val_shape)
        logger.info("  holdout shape:              %d × %d", *report.holdout_shape)
        logger.info("  duration:                   %.2fs", report.duration_seconds)

        if report.skipped_features:
            logger.info("  skipped features: %s", report.skipped_features)
        if report.failed_features:
            logger.warning("  failed features: %s", report.failed_features)

        # Top 15 features by IV
        top = sorted(report.binnings, key=lambda b: -b.iv)[:15]
        logger.info("  ---")
        logger.info("  Top 15 features by IV:")
        for i, b in enumerate(top, 1):
            monotonic = {0: "—", +1: "↑", -1: "↓"}[b.monotonic_direction]
            logger.info(
                "    #%2d  %-45s  IV=%.4f  bins=%2d (reg %2d)  mono=%s",
                i, b.feature, b.iv, b.n_bins_total, b.n_bins_regular, monotonic,
            )
        logger.info("=" * 70)

    # ---- input loading ---------------------------------------------------

    def _load_input(self) -> pd.DataFrame:
        path = self.config.input_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Input not found: {path}. Run assembly first."
            )
        logger.info("Loading %s", path.name)
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("  loaded: %d rows, %d cols", len(df), len(df.columns))
        return df

    def _load_splits(self) -> dict[str, Any]:
        path = self.config.splits_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Splits not found: {path}. Run splits first."
            )
        with path.open(encoding="utf-8") as f:
            splits = json.load(f)
        logger.info(
            "Splits loaded: dev=%d, val=%d, test=%d (seed=%d)",
            splits["n_dev"], splits["n_val"], splits["n_test"],
            splits["random_seed"],
        )
        return splits

    def _apply_splits(
        self,
        df: pd.DataFrame,
        splits: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Filter the assembled matrix to each split."""
        key = self.config.key_column
        target = self.config.target_column

        dev_ids = set(splits["dev_ids"])
        val_ids = set(splits["val_ids"])
        test_ids = set(splits["test_ids"])

        # Sanity: splits must be disjoint
        if dev_ids & val_ids:
            raise RuntimeError(f"dev and val splits overlap by {len(dev_ids & val_ids)} IDs")
        if dev_ids & test_ids:
            raise RuntimeError(f"dev and test splits overlap by {len(dev_ids & test_ids)} IDs")
        if val_ids & test_ids:
            raise RuntimeError(f"val and test splits overlap by {len(val_ids & test_ids)} IDs")

        df = df.set_index(key)

        total = len(dev_ids) + len(val_ids) + len(test_ids)
        if total != len(df):
            raise RuntimeError(
                f"Split total ({total}) != input rows ({len(df)}). "
                "Splits don't match the current input matrix."
            )

        dev = df.loc[df.index.isin(dev_ids)]
        val = df.loc[df.index.isin(val_ids)]
        holdout = df.loc[df.index.isin(test_ids)]

        y_dev = dev[target].astype(int)
        y_val = val[target].astype(int)
        y_holdout = holdout[target].astype(int)

        X_dev = dev.drop(columns=[target]).reset_index()
        X_val = val.drop(columns=[target]).reset_index()
        X_holdout = holdout.drop(columns=[target]).reset_index()

        y_dev.index = X_dev.index
        y_val.index = X_val.index
        y_holdout.index = X_holdout.index

        return X_dev, y_dev, X_val, y_val, X_holdout, y_holdout

    # ---- output writing --------------------------------------------------

    def _write_split(self, df: pd.DataFrame, path: Path) -> None:
        df.to_parquet(
            path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        logger.info("  wrote %s: %d rows, %d cols", path.name, len(df), len(df.columns))

    def _build_report(
        self,
        binnings: list[FeatureBinning],
        decisions: list[FeatureDecision],
        skipped: list[str],
        failed: list[str],
        dev_shape: tuple[int, int],
        val_shape: tuple[int, int],
        holdout_shape: tuple[int, int],
        n_features_total: int,
        duration: float,
    ) -> BinningReport:
        kept = sum(1 for d in decisions if d.kept)
        dropped_low = sum(
            1 for d in decisions
            if not d.kept and "<" in d.reason and "IV" in d.reason
        )
        dropped_high = sum(
            1 for d in decisions
            if not d.kept and ">" in d.reason and "suspicious" in d.reason
        )
        dropped_corr = sum(
            1 for d in decisions
            if not d.kept and "correlated" in d.reason
        )

        return BinningReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            n_features_total=n_features_total,
            n_features_binned=len(binnings),
            n_features_skipped=len(skipped),
            n_features_failed=len(failed),
            n_features_kept=kept,
            n_features_dropped_low_iv=dropped_low,
            n_features_dropped_high_iv=dropped_high,
            n_features_dropped_correlated=dropped_corr,
            duration_seconds=duration,
            train_shape=dev_shape,
            val_shape=val_shape,
            holdout_shape=holdout_shape,
            skipped_features=sorted(skipped),
            failed_features=sorted(failed),
            binnings=binnings,
            decisions=decisions,
        )

    def _write_reports(self, report: BinningReport) -> None:
        # Full summary
        with self.config.summary_path().open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # Selection