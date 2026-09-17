"""
Baseline smoke test: train a quick LightGBM on raw assembled features.

Purpose
-------
Confirm that the assembled feature matrix carries predictive signal
before investing in the transformation infrastructure (WoE binning,
imputation, feature selection). If the raw features produce AUC in the
expected range, the pipeline is working end-to-end.

This script is a smoke test, not a deliverable. It will be superseded
by the champion/challenger pipeline in Stage C/D. It exists to catch
gross errors (leakage, corrupted features, wrong joins) before they
propagate.

Inputs
------
data/processed/application_train_features.parquet

Outputs
-------
- Console log (structured, timestamped)
- artifacts/reports/baseline_smoke_test.json

Exit codes
----------
0 — AUC within expected range
1 — AUC below expected range (something wrong)
2 — AUC above expected range (likely leakage)
3 — setup error (missing input, missing dependency)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger("credit_risk.baseline_smoke_test")


# ---------------------------------------------------------------------------
# Expected ranges (from published Home Credit baselines)
# ---------------------------------------------------------------------------

EXPECTED_AUC_MIN = 0.75
EXPECTED_AUC_MAX = 0.82
LEAKAGE_AUC_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmokeTestConfig:
    """Resolved configuration for the smoke test."""

    project_root: Path
    processed_dir: Path
    reports_dir: Path

    input_file: str = "application_train_features.parquet"
    output_file: str = "baseline_smoke_test.json"

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    holdout_fraction: float = 0.20
    random_seed: int = 42

    # LightGBM hyperparameters — deliberately modest
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0

    def input_path(self) -> Path:
        return self.processed_dir / self.input_file

    def output_path(self) -> Path:
        return self.reports_dir / self.output_file


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class SmokeTestResult:
    """Metrics and diagnostics from the smoke test."""

    generated_at: str
    input_path: str
    n_rows: int
    n_features: int
    n_train: int
    n_holdout: int
    positive_rate: float

    auc_train: float
    auc_holdout: float
    gini_holdout: float
    ks_holdout: float
    brier_holdout: float

    duration_seconds: float
    top_features: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "input_path": self.input_path,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "n_train": self.n_train,
            "n_holdout": self.n_holdout,
            "positive_rate": round(self.positive_rate, 6),
            "metrics": {
                "auc_train": round(self.auc_train, 6),
                "auc_holdout": round(self.auc_holdout, 6),
                "gini_holdout": round(self.gini_holdout, 6),
                "ks_holdout": round(self.ks_holdout, 6),
                "brier_holdout": round(self.brier_holdout, 6),
            },
            "duration_seconds": round(self.duration_seconds, 3),
            "top_features": self.top_features,
            "verdict": self.verdict,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn to keep the smoke test dependency-light."""
    # Use sklearn if available; it's the standard and correct implementation
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation of CDFs for positive/negative."""
    order = np.argsort(y_score)
    y_sorted = y_true[order]

    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    cum_pos = np.cumsum(y_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_sorted) / n_neg

    return float(np.max(np.abs(cum_pos - cum_neg)))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error of predicted probabilities."""
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

@dataclass
class BaselineSmokeTest:
    """Trains a quick LightGBM and reports discrimination and calibration metrics."""

    config: SmokeTestConfig
    result: SmokeTestResult | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> SmokeTestResult:
        start = time.monotonic()
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        df = self._load_data()
        X_train, X_holdout, y_train, y_holdout = self._split(df)
        model = self._train(X_train, y_train, X_holdout, y_holdout)
        result = self._evaluate(
            model=model,
            X_train=X_train, y_train=y_train,
            X_holdout=X_holdout, y_holdout=y_holdout,
            df=df,
            duration=time.monotonic() - start,
        )

        self.result = result
        self._write_report(result)
        self._log_summary(result)
        return result

    # ---- data loading ----------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        path = self.config.input_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Input not found: {path}. Run the earlier pipeline stages first."
            )
        logger.info("Loading %s", path.name)
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("  loaded: %d rows, %d cols", len(df), len(df.columns))

        # Sanity checks
        if self.config.target_column not in df.columns:
            raise KeyError(
                f"Target '{self.config.target_column}' not present in "
                f"{path.name}. This is the training matrix."
            )
        if self.config.key_column not in df.columns:
            raise KeyError(
                f"Key '{self.config.key_column}' not present in {path.name}."
            )

        # Basic target integrity
        y = df[self.config.target_column]
        if y.isna().any():
            raise ValueError(
                f"Target has {int(y.isna().sum())} nulls. Cannot train."
            )
        unique = set(y.unique().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"Target has unexpected values: {sorted(unique)}. Expected {{0, 1}}."
            )

        return df

    def _encode_categoricals(
        self,
        X_train: pd.DataFrame,
        X_holdout: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode object (string) columns to integer codes.

        This is a smoke-test-only transformation. The production pipeline
        (Stage C) will use proper WoE binning for the champion scorecard
        and target encoding for the challenger.

        Encoding strategy:
        - Categoricals are encoded with pandas' factorize, which maps each
          unique value to an integer code.
        - Categories are learned from the training data and applied to the
          holdout. Unseen categories in the holdout are mapped to -1.
        - Integer codes are then cast to int32 so LightGBM accepts them.
        """
        object_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
        if not object_cols:
            return X_train, X_holdout

        logger.info(
            "Encoding %d object columns to integer codes: %s",
            len(object_cols), object_cols,
        )

        X_train = X_train.copy()
        X_holdout = X_holdout.copy()

        for col in object_cols:
            codes, uniques = pd.factorize(X_train[col], sort=True)
            X_train[col] = codes.astype("int32")

            X_holdout[col] = pd.Categorical(
                X_holdout[col], categories=uniques
            ).codes.astype("int32")

        return X_train, X_holdout

    def _split(
        self, df: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Stratified train/holdout split with categorical encoding."""
        from sklearn.model_selection import train_test_split

        # Feature matrix: drop key and target
        drop_cols = [self.config.key_column, self.config.target_column]
        X = df.drop(columns=drop_cols)
        y = df[self.config.target_column].astype(int)

        logger.info(
            "Feature matrix: %d columns, target positive rate = %.4f",
            X.shape[1], y.mean(),
        )

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y,
            test_size=self.config.holdout_fraction,
            stratify=y,
            random_state=self.config.random_seed,
        )

        X_train, X_holdout = self._encode_categoricals(X_train, X_holdout)

        logger.info(
            "Split: train %d × %d, holdout %d × %d",
            X_train.shape[0], X_train.shape[1],
            X_holdout.shape[0], X_holdout.shape[1],
        )
        return X_train, X_holdout, y_train, y_holdout
    
    # ---- training --------------------------------------------------------

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
    ):
        """Train a LightGBM classifier with early stopping on the holdout."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )

        logger.info("Training LightGBM baseline")
        model = lgb.LGBMClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            max_depth=self.config.max_depth,
            min_child_samples=self.config.min_child_samples,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            objective="binary",
            random_state=self.config.random_seed,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_holdout, y_holdout)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        logger.info("  trained: best iteration = %d", model.best_iteration_)
        return model

    # ---- evaluation ------------------------------------------------------

    def _evaluate(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series,
        df: pd.DataFrame,
        duration: float,
    ) -> SmokeTestResult:
        logger.info("Evaluating on train and holdout")

        p_train = model.predict_proba(X_train)[:, 1]
        p_holdout = model.predict_proba(X_holdout)[:, 1]

        y_train_arr = np.asarray(y_train)
        y_holdout_arr = np.asarray(y_holdout)

        auc_train = compute_auc(y_train_arr, p_train)
        auc_holdout = compute_auc(y_holdout_arr, p_holdout)
        gini_holdout = 2 * auc_holdout - 1
        ks_holdout = compute_ks(y_holdout_arr, p_holdout)
        brier_holdout = compute_brier(y_holdout_arr, p_holdout)

        # Feature importances
        top_features = self._top_importances(model, X_train.columns, k=20)

        # Verdict + notes
        verdict, notes = self._verdict(
            auc_holdout=auc_holdout,
            auc_train=auc_train,
        )

        return SmokeTestResult(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            input_path=str(self.config.input_path().relative_to(self.config.project_root)),
            n_rows=len(df),
            n_features=X_train.shape[1],
            n_train=len(X_train),
            n_holdout=len(X_holdout),
            positive_rate=float(df[self.config.target_column].mean()),
            auc_train=auc_train,
            auc_holdout=auc_holdout,
            gini_holdout=gini_holdout,
            ks_holdout=ks_holdout,
            brier_holdout=brier_holdout,
            duration_seconds=duration,
            top_features=top_features,
            verdict=verdict,
            notes=notes,
        )

    def _top_importances(
        self, model: Any, columns: pd.Index, k: int = 20
    ) -> list[dict[str, Any]]:
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1][:k]
        return [
            {
                "rank": i + 1,
                "feature": str(columns[idx]),
                "importance": int(importances[idx]),
            }
            for i, idx in enumerate(order)
        ]

    def _verdict(
        self,
        auc_holdout: float,
        auc_train: float,
    ) -> tuple[str, list[str]]:
        """
        Classify the result against expected ranges.
        """
        notes: list[str] = []

        overfit_gap = auc_train - auc_holdout
        if overfit_gap > 0.05:
            notes.append(
                f"Train-holdout AUC gap of {overfit_gap:.4f} suggests overfitting."
            )

        if auc_holdout > LEAKAGE_AUC_THRESHOLD:
            notes.append(
                f"Holdout AUC {auc_holdout:.4f} exceeds leakage threshold "
                f"{LEAKAGE_AUC_THRESHOLD}. Investigate for target leakage."
            )
            return "leakage_suspected", notes

        if auc_holdout < EXPECTED_AUC_MIN:
            notes.append(
                f"Holdout AUC {auc_holdout:.4f} below expected minimum "
                f"{EXPECTED_AUC_MIN}. Check aggregation logic."
            )
            return "below_expected", notes

        if auc_holdout > EXPECTED_AUC_MAX:
            notes.append(
                f"Holdout AUC {auc_holdout:.4f} above expected maximum "
                f"{EXPECTED_AUC_MAX}. Verify no leakage."
            )
            return "above_expected", notes

        notes.append(
            f"Holdout AUC {auc_holdout:.4f} within expected range "
            f"[{EXPECTED_AUC_MIN}, {EXPECTED_AUC_MAX}]."
        )
        return "ok", notes

    # ---- reporting -------------------------------------------------------

    def _write_report(self, result: SmokeTestResult) -> Path:
        out_path = self.config.output_path()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("Report written: %s", out_path.name)
        return out_path

    def _log_summary(self, result: SmokeTestResult) -> None:
        logger.info("=" * 70)
        logger.info("Baseline smoke test complete")
        logger.info("  rows:               %d", result.n_rows)
        logger.info("  features:           %d", result.n_features)
        logger.info("  positive rate:      %.4f", result.positive_rate)
        logger.info("  ---")
        logger.info("  AUC (train):        %.4f", result.auc_train)
        logger.info("  AUC (holdout):      %.4f", result.auc_holdout)
        logger.info("  Gini (holdout):     %.4f", result.gini_holdout)
        logger.info("  KS (holdout):       %.4f", result.ks_holdout)
        logger.info("  Brier (holdout):    %.4f", result.brier_holdout)
        logger.info("  ---")
        logger.info("  Verdict:            %s", result.verdict)
        for note in result.notes:
            logger.info("  Note: %s", note)
        logger.info("  ---")
        logger.info("  Top 20 features by importance:")
        for feat in result.top_features:
            logger.info(
                "    #%2d  %-50s  %d",
                feat["rank"], feat["feature"], feat["importance"],
            )
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def main() -> int:
    configure_logging()
    project_root = Path(__file__).resolve().parents[1]

    config = SmokeTestConfig(
        project_root=project_root,
        processed_dir=project_root / "data" / "processed",
        reports_dir=project_root / "artifacts" / "reports",
    )

    try:
        result = BaselineSmokeTest(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Setup error: %s", exc)
        return 3
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        return 3
    except Exception:
        logger.exception("Smoke test failed")
        return 1

    # Exit code reflects the verdict
    if result.verdict == "ok":
        return 0
    if result.verdict == "leakage_suspected":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())