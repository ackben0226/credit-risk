"""
Deterministic train/validation/holdout splits for the training population.

This module produces and persists the split indices that every downstream
stage relies on. Splits are stratified on TARGET and expressed as sets of
SK_ID_CURR values — stable identifiers, not row positions.

Design notes
------------
- Splits are defined once, persisted, and reused. Every stage that fits
  a transformation or evaluates a model reads the same split file.
- Stratification preserves the ~8.07% positive rate across all three
  splits. Without stratification, small splits may skew materially on
  a rare target.
- Splits are stored as SK_ID_CURR values, not row indices. This makes
  them robust to DataFrame reordering and readable in a diff.
- 70/15/15 is standard for credit risk. It provides enough data for
  fitting (train), tuning (validation), and unbiased evaluation
  (holdout).
- The holdout is touched exactly once — at final evaluation. No fitting,
  no tuning, no selection. Every metric on holdout is a clean estimate
  of generalization.

Outputs
-------
- artifacts/splits/splits.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger("credit_risk.features.splits")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitsConfig:
    """Resolved configuration for a split generation run."""

    project_root: Path
    processed_dir: Path
    splits_dir: Path

    input_file: str = "application_train_features.parquet"
    output_file: str = "splits.json"

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    # 70 / 15 / 15 split
    dev_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    random_seed: int = 42

    def input_path(self) -> Path:
        return self.processed_dir / self.input_file

    def output_path(self) -> Path:
        return self.splits_dir / self.output_file


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SplitsReport:
    """Summary of the produced splits."""

    generated_at: str
    input_path: str
    random_seed: int
    n_total: int
    n_dev: int
    n_val: int
    n_test: int
    dev_positive_rate: float
    val_positive_rate: float
    test_positive_rate: float
    duration_seconds: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "input_path": self.input_path,
            "random_seed": self.random_seed,
            "n_total": self.n_total,
            "n_dev": self.n_dev,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "dev_positive_rate": round(self.dev_positive_rate, 6),
            "val_positive_rate": round(self.val_positive_rate, 6),
            "test_positive_rate": round(self.test_positive_rate, 6),
            "duration_seconds": round(self.duration_seconds, 3),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------

@dataclass
class Splitter:
    """
    Produces deterministic train/validation/holdout splits from the
    assembled training matrix.
    """

    config: SplitsConfig
    report: SplitsReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> SplitsReport:
        import time
        start = time.monotonic()

        self.config.splits_dir.mkdir(parents=True, exist_ok=True)

        df = self._load()
        self._validate_config()
        self._validate_target(df)

        dev_ids, val_ids, test_ids = self._split(df)
        self._validate_splits(dev_ids, val_ids, test_ids, df)
        self._write(dev_ids, val_ids, test_ids, df)

        duration = time.monotonic() - start
        self.report = self._build_report(dev_ids, val_ids, test_ids, df, duration)
        self._log_summary(self.report)
        return self.report

    # ---- preconditions ---------------------------------------------------

    def _load(self) -> pd.DataFrame:
        path = self.config.input_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Input not found: {path}. Run assembly first."
            )
        logger.info("Loading %s", path.name)
        df = pd.read_parquet(
            path,
            engine="pyarrow",
            columns=[self.config.key_column, self.config.target_column],
        )
        logger.info(
            "  loaded: %d rows, positive rate = %.4f",
            len(df), df[self.config.target_column].mean(),
        )
        return df

    def _validate_config(self) -> None:
        total = (
            self.config.dev_fraction
            + self.config.val_fraction
            + self.config.test_fraction
        )
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(
                f"Split fractions must sum to 1.0 (got {total}). "
                f"dev={self.config.dev_fraction}, "
                f"val={self.config.val_fraction}, "
                f"test={self.config.test_fraction}"
            )

    def _validate_target(self, df: pd.DataFrame) -> None:
        if self.config.target_column not in df.columns:
            raise KeyError(
                f"Target '{self.config.target_column}' not found in input."
            )
        if self.config.key_column not in df.columns:
            raise KeyError(
                f"Key '{self.config.key_column}' not found in input."
            )
        if df[self.config.key_column].duplicated().any():
            raise ValueError(
                f"Duplicate values in key column '{self.config.key_column}'."
            )
        if df[self.config.target_column].isna().any():
            raise ValueError("Target has null values.")
        unique = set(df[self.config.target_column].unique().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Target has unexpected values: {sorted(unique)}."
            )

    # ---- splitting -------------------------------------------------------

    def _split(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce three disjoint SK_ID_CURR sets via a two-step stratified split.

        Step 1: split off the holdout (test)
        Step 2: split the remainder into dev and val

        Each step is stratified on TARGET.
        """
        from sklearn.model_selection import train_test_split

        key = df[self.config.key_column].to_numpy()
        target = df[self.config.target_column].to_numpy()

        # Step 1: hold out test
        dev_val_ids, test_ids = train_test_split(
            key,
            test_size=self.config.test_fraction,
            stratify=target,
            random_state=self.config.random_seed,
        )

        # Step 2: split the remaining into dev and val
        # Adjust the ratio since dev + val = 1 - test_fraction
        remaining = self.config.dev_fraction + self.config.val_fraction
        val_size_of_remaining = self.config.val_fraction / remaining

        dev_ids, val_ids = train_test_split(
            dev_val_ids,
            test_size=val_size_of_remaining,
            stratify=self._target_for_keys(df, dev_val_ids),
            random_state=self.config.random_seed,
        )

        logger.info(
            "Splits produced: dev=%d, val=%d, test=%d",
            len(dev_ids), len(val_ids), len(test_ids),
        )
        return dev_ids, val_ids, test_ids

    @staticmethod
    def _target_for_keys(df: pd.DataFrame, keys: np.ndarray) -> np.ndarray:
        """Look up the target value for a subset of keys, preserving order."""
        keyed = df.set_index("SK_ID_CURR")["TARGET"]
        return keyed.loc[keys].to_numpy()

    # ---- validation ------------------------------------------------------

    def _validate_splits(
        self,
        dev_ids: np.ndarray,
        val_ids: np.ndarray,
        test_ids: np.ndarray,
        df: pd.DataFrame,
    ) -> None:
        """
        Assert splits are disjoint, complete, and correctly stratified.
        """
        dev_set = set(dev_ids.tolist())
        val_set = set(val_ids.tolist())
        test_set = set(test_ids.tolist())

        # Disjoint
        if dev_set & val_set:
            raise RuntimeError(
                f"dev ∩ val = {len(dev_set & val_set)} rows — splits overlap"
            )
        if dev_set & test_set:
            raise RuntimeError(
                f"dev ∩ test = {len(dev_set & test_set)} rows — splits overlap"
            )
        if val_set & test_set:
            raise RuntimeError(
                f"val ∩ test = {len(val_set & test_set)} rows — splits overlap"
            )

        # Complete
        all_ids = set(df[self.config.key_column].tolist())
        union = dev_set | val_set | test_set
        if union != all_ids:
            missing = all_ids - union
            extra = union - all_ids
            raise RuntimeError(
                f"Split union does not match input. "
                f"Missing: {len(missing)}. Extra: {len(extra)}."
            )

        # Sizes
        n_total = len(df)
        expected_dev = round(self.config.dev_fraction * n_total)
        expected_val = round(self.config.val_fraction * n_total)
        expected_test = round(self.config.test_fraction * n_total)

        for name, actual, expected in [
            ("dev", len(dev_ids), expected_dev),
            ("val", len(val_ids), expected_val),
            ("test", len(test_ids), expected_test),
        ]:
            # Allow ±1 due to rounding
            if abs(actual - expected) > 1:
                raise RuntimeError(
                    f"{name} split size {actual} differs from expected "
                    f"{expected} by more than 1"
                )

        # Stratification: positive rates must be close to overall
        overall_rate = df[self.config.target_column].mean()
        tolerance = 0.005  # 0.5 percentage points

        keyed_target = df.set_index(self.config.key_column)[self.config.target_column]

        for name, ids in [("dev", dev_ids), ("val", val_ids), ("test", test_ids)]:
            rate = keyed_target.loc[ids].mean()
            delta = abs(rate - overall_rate)
            if delta > tolerance:
                raise RuntimeError(
                    f"{name} positive rate {rate:.4f} differs from overall "
                    f"{overall_rate:.4f} by more than {tolerance}"
                )

    # ---- output ----------------------------------------------------------

    def _write(
        self,
        dev_ids: np.ndarray,
        val_ids: np.ndarray,
        test_ids: np.ndarray,
        df: pd.DataFrame,
    ) -> Path:
        keyed_target = df.set_index(self.config.key_column)[self.config.target_column]

        output = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "input_path": str(
                self.config.input_path().relative_to(self.config.project_root)
            ),
            "random_seed": self.config.random_seed,
            "ratios": {
                "dev": self.config.dev_fraction,
                "val": self.config.val_fraction,
                "test": self.config.test_fraction,
            },
            "n_total": len(df),
            "n_dev": len(dev_ids),
            "n_val": len(val_ids),
            "n_test": len(test_ids),
            "positive_rate_dev": round(float(keyed_target.loc[dev_ids].mean()), 6),
            "positive_rate_val": round(float(keyed_target.loc[val_ids].mean()), 6),
            "positive_rate_test": round(float(keyed_target.loc[test_ids].mean()), 6),
            "dev_ids": sorted(int(x) for x in dev_ids.tolist()),
            "val_ids": sorted(int(x) for x in val_ids.tolist()),
            "test_ids": sorted(int(x) for x in test_ids.tolist()),
        }

        out_path = self.config.output_path()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        logger.info("Splits written: %s", out_path.name)
        return out_path

    # ---- reporting -------------------------------------------------------

    def _build_report(
        self,
        dev_ids: np.ndarray,
        val_ids: np.ndarray,
        test_ids: np.ndarray,
        df: pd.DataFrame,
        duration: float,
    ) -> SplitsReport:
        keyed_target = df.set_index(self.config.key_column)[self.config.target_column]
        notes: list[str] = []

        # Sanity: check stratification is tight
        overall = float(df[self.config.target_column].mean())
        for name, ids in [("dev", dev_ids), ("val", val_ids), ("test", test_ids)]:
            rate = float(keyed_target.loc[ids].mean())
            if abs(rate - overall) > 0.002:
                notes.append(
                    f"{name} positive rate {rate:.4f} differs from overall "
                    f"{overall:.4f} by {abs(rate - overall):.4f}"
                )

        return SplitsReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            input_path=str(
                self.config.input_path().relative_to(self.config.project_root)
            ),
            random_seed=self.config.random_seed,
            n_total=len(df),
            n_dev=len(dev_ids),
            n_val=len(val_ids),
            n_test=len(test_ids),
            dev_positive_rate=float(keyed_target.loc[dev_ids].mean()),
            val_positive_rate=float(keyed_target.loc[val_ids].mean()),
            test_positive_rate=float(keyed_target.loc[test_ids].mean()),
            duration_seconds=duration,
            notes=notes,
        )

    def _log_summary(self, report: SplitsReport) -> None:
        logger.info("=" * 70)
        logger.info("Splits generated")
        logger.info("  input:              %s", report.input_path)
        logger.info("  random seed:        %d", report.random_seed)
        logger.info("  total rows:         %d", report.n_total)
        logger.info("  ---")
        logger.info(
            "  dev:                %d (%.2f%%)   positive rate %.4f",
            report.n_dev, 100 * report.n_dev / report.n_total,
            report.dev_positive_rate,
        )
        logger.info(
            "  val:                %d (%.2f%%)   positive rate %.4f",
            report.n_val, 100 * report.n_val / report.n_total,
            report.val_positive_rate,
        )
        logger.info(
            "  test:               %d (%.2f%%)   positive rate %.4f",
            report.n_test, 100 * report.n_test / report.n_total,
            report.test_positive_rate,
        )
        for note in report.notes:
            logger.info("  Note: %s", note)
        logger.info("  duration:           %.2fs", report.duration_seconds)
        logger.info("=" * 70)