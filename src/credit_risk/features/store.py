"""
Challenger feature store: raw assembled features + categorical encoding +
explicit missing indicators.

The champion scorecard consumes WoE-encoded features. The challenger
(gradient boosting) consumes raw features instead:

    - Nulls are preserved. LightGBM routes them natively during tree
      growth, learning the missingness signal implicitly.
    - Categorical strings are factorized to integer codes, fitted on
      the DEV split and applied to VAL and HOLDOUT. Unseen categories
      map to -1.
    - A small set of explicit missing indicators is added for the
      sparsest features, so the missingness signal is exposed as its
      own feature rather than being absorbed into tree splits. This is
      optional for the model, but valuable for feature importance and
      explainability.

Why no imputation
-----------------
Imputation is not part of this pipeline.

    - The champion (logistic scorecard) does not need it: WoE encoding
      gives nulls their own bin with their own numeric value.
    - The challenger (LightGBM) does not need it: native null handling
      is more expressive than any single imputed value, and imputation
      would destroy the missingness signal.

The only preprocessing required for the challenger is categorical
encoding.

Design notes
------------
- Categorical mappings are fitted on the DEV split only. VAL and
  HOLDOUT are transformed with the same mapping. Unseen categories in
  VAL or HOLDOUT map to -1.
- Missing indicators are computed on the raw values, before any other
  transformation. They are binary int8 columns.
- Column order is deterministic: SK_ID_CURR, features alphabetically,
  TARGET. This makes files diffable and metrics comparable across runs.

Outputs
-------
- data/processed/challenger_train.parquet
- data/processed/challenger_val.parquet
- data/processed/challenger_holdout.parquet
- artifacts/reports/challenger_feature_catalogue.json
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


logger = logging.getLogger("credit_risk.features.store")


# ---------------------------------------------------------------------------
# Missing indicators
# ---------------------------------------------------------------------------
# Explicit binary columns exposing the missingness signal as its own
# feature. Only added for the sparsest features where missingness is
# likely informative. The naming convention is `<feature>_is_null`.
#
MISSING_INDICATOR_FEATURES: tuple[str, ...] = (
    "cc_utilization_last",
    "cc_balance_last",
    "bb_max_status_max",
    "bb_months_at_3_plus_total",
    "bureau_credit_sum_overdue_max",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StoreConfig:
    """Resolved configuration for building challenger feature stores."""

    project_root: Path
    processed_dir: Path
    splits_dir: Path
    reports_dir: Path

    input_file: str = "application_train_features.parquet"
    splits_file: str = "splits.json"

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    train_output: str = "challenger_train.parquet"
    val_output: str = "challenger_val.parquet"
    holdout_output: str = "challenger_holdout.parquet"
    catalogue_output: str = "challenger_feature_catalogue.json"

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

    def catalogue_path(self) -> Path:
        return self.reports_dir / self.catalogue_output


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class CategoricalMapping:
    """Fitted mapping for one categorical column."""

    column: str
    categories: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "n_categories": len(self.categories),
            "categories": [str(c) for c in self.categories],
        }


@dataclass
class StoreReport:
    """Outcome of a challenger store build."""

    generated_at: str
    n_features_input: int
    n_categoricals: int
    n_missing_indicators: int
    n_features_output: int
    train_shape: tuple[int, int]
    val_shape: tuple[int, int]
    holdout_shape: tuple[int, int]
    duration_seconds: float
    categorical_mappings: list[CategoricalMapping] = field(default_factory=list)
    missing_indicators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "n_features_input": self.n_features_input,
            "n_categoricals": self.n_categoricals,
            "n_missing_indicators": self.n_missing_indicators,
            "n_features_output": self.n_features_output,
            "train_shape": list(self.train_shape),
            "val_shape": list(self.val_shape),
            "holdout_shape": list(self.holdout_shape),
            "duration_seconds": round(self.duration_seconds, 3),
            "categorical_mappings": [m.to_dict() for m in self.categorical_mappings],
            "missing_indicators": self.missing_indicators,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

@dataclass
class ChallengerStoreBuilder:
    """Builds challenger train/val/holdout matrices from assembled features."""

    config: StoreConfig
    report: StoreReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> StoreReport:
        start = time.monotonic()
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        df = self._load_input()
        splits = self._load_splits()
        train, val, holdout = self._apply_splits(df, splits)

        logger.info(
            "Splits: train=%d, val=%d, holdout=%d",
            len(train), len(val), len(holdout),
        )

        # Add missing indicators BEFORE factorization — they compute on
        # the raw values.
        train, val, holdout, indicators_added = self._add_missing_indicators(
            train, val, holdout
        )

        # Factorize categoricals
        train, val, holdout, mappings = self._factorize_categoricals(
            train, val, holdout
        )

        # Reorder columns
        train = self._reorder(train, has_target=True)
        val = self._reorder(val, has_target=True)
        holdout = self._reorder(holdout, has_target=True)

        # Write outputs
        self._write(train, self.config.train_output_path())
        self._write(val, self.config.val_output_path())
        self._write(holdout, self.config.holdout_output_path())

        duration = time.monotonic() - start
        self.report = self._build_report(
            n_features_input=len(df.columns) - 2,  # exclude key + target
            train=train,
            val=val,
            holdout=holdout,
            mappings=mappings,
            indicators_added=indicators_added,
            duration=duration,
        )
        self._write_catalogue(self.report)
        self._log_summary(self.report)
        return self.report

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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter the assembled matrix to each split and return as DataFrames."""
        key = self.config.key_column

        dev_ids = set(splits["dev_ids"])
        val_ids = set(splits["val_ids"])
        test_ids = set(splits["test_ids"])

        # Disjointness sanity
        if dev_ids & val_ids or dev_ids & test_ids or val_ids & test_ids:
            raise RuntimeError("Split ID sets overlap")

        df = df.set_index(key)
        total = len(dev_ids) + len(val_ids) + len(test_ids)
        if total != len(df):
            raise RuntimeError(
                f"Split total ({total}) != input rows ({len(df)})"
            )

        train = df.loc[df.index.isin(dev_ids)].reset_index()
        val = df.loc[df.index.isin(val_ids)].reset_index()
        holdout = df.loc[df.index.isin(test_ids)].reset_index()

        return train, val, holdout

    # ---- missing indicators ----------------------------------------------

    def _add_missing_indicators(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        holdout: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        """Add `<feature>_is_null` binary columns for the sparsest features."""
        added: list[str] = []

        for feature in MISSING_INDICATOR_FEATURES:
            if feature not in train.columns:
                logger.warning(
                    "  missing indicator skipped (feature absent): %s", feature
                )
                continue

            indicator_name = f"{feature}_is_null"

            train[indicator_name] = train[feature].isna().astype("int8")
            val[indicator_name] = val[feature].isna().astype("int8")
            holdout[indicator_name] = holdout[feature].isna().astype("int8")

            added.append(indicator_name)

        logger.info(
            "Added %d missing indicators: %s", len(added), added
        )
        return train, val, holdout, added

    # ---- categorical factorization ---------------------------------------

    def _factorize_categoricals(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        holdout: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[CategoricalMapping]]:
        """
        Factorize categorical columns.

        Fitted on train; applied to val and holdout. Unseen categories in
        val or holdout map to -1.
        """
        cat_cols = [
            c for c in train.columns
            if train[c].dtype == "object" or pd.api.types.is_object_dtype(train[c])
        ]
        logger.info("Factorizing %d categorical columns", len(cat_cols))

        mappings: list[CategoricalMapping] = []

        for col in cat_cols:
            # Fit on train
            codes, uniques = pd.factorize(train[col], sort=True)
            train[col] = codes.astype("int32")

            # Apply to val / holdout — unseen categories → -1
            val[col] = pd.Categorical(
                val[col], categories=uniques
            ).codes.astype("int32")
            holdout[col] = pd.Categorical(
                holdout[col], categories=uniques
            ).codes.astype("int32")

            mappings.append(
                CategoricalMapping(column=col, categories=uniques.tolist())
            )

        return train, val, holdout, mappings

    # ---- column ordering -------------------------------------------------

    def _reorder(self, df: pd.DataFrame, *, has_target: bool) -> pd.DataFrame:
        """Deterministic column order: SK_ID_CURR, features alphabetically, TARGET."""
        key = self.config.key_column
        target = self.config.target_column

        feature_cols = sorted(
            c for c in df.columns
            if c != key and (not has_target or c != target)
        )

        ordered = [key] + feature_cols
        if has_target:
            ordered.append(target)

        # Sanity
        if set(ordered) != set(df.columns):
            missing = set(df.columns) - set(ordered)
            extra = set(ordered) - set(df.columns)
            raise RuntimeError(
                f"Column reordering mismatch. Missing: {missing}. Extra: {extra}."
            )

        return df[ordered]

    # ---- writing ---------------------------------------------------------

    def _write(self, df: pd.DataFrame, path: Path) -> None:
        df.to_parquet(
            path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        logger.info(
            "  wrote %s: %d rows, %d cols",
            path.name, len(df), len(df.columns),
        )

    # ---- reporting -------------------------------------------------------

    def _build_report(
        self,
        n_features_input: int,
        train: pd.DataFrame,
        val: pd.DataFrame,
        holdout: pd.DataFrame,
        mappings: list[CategoricalMapping],
        indicators_added: list[str],
        duration: float,
    ) -> StoreReport:
        # Features output = all non-key, non-target columns
        feature_cols = [
            c for c in train.columns
            if c != self.config.key_column and c != self.config.target_column
        ]
        return StoreReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            n_features_input=n_features_input,
            n_categoricals=len(mappings),
            n_missing_indicators=len(indicators_added),
            n_features_output=len(feature_cols),
            train_shape=(len(train), len(train.columns)),
            val_shape=(len(val), len(val.columns)),
            holdout_shape=(len(holdout), len(holdout.columns)),
            duration_seconds=duration,
            categorical_mappings=mappings,
            missing_indicators=indicators_added,
        )

    def _write_catalogue(self, report: StoreReport) -> Path:
        out_path = self.config.catalogue_path()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Feature catalogue written: %s", out_path.name)
        return out_path

    def _log_summary(self, report: StoreReport) -> None:
        logger.info("=" * 70)
        logger.info("Challenger feature store built")
        logger.info("  input features:             %d", report.n_features_input)
        logger.info("  categoricals encoded:       %d", report.n_categoricals)
        logger.info("  missing indicators added:   %d", report.n_missing_indicators)
        logger.info("  output features:            %d", report.n_features_output)
        logger.info("  ---")
        logger.info("  train shape:                %d × %d", *report.train_shape)
        logger.info("  val shape:                  %d × %d", *report.val_shape)
        logger.info("  holdout shape:              %d × %d", *report.holdout_shape)
        logger.info("  duration:                   %.2fs", report.duration_seconds)
        logger.info("  ---")
        logger.info("  missing indicators added:")
        for name in report.missing_indicators:
            logger.info("    %s", name)
        logger.info("=" * 70)