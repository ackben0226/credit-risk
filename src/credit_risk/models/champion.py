"""
Champion scorecard: Elastic-Net logistic regression on WoE-encoded features.

The champion model is a transparent, monotonic, regulator-friendly
scorecard. It consumes WoE-encoded features (nulls resolved to their own
bin) and produces calibrated probability-of-default estimates.

Training protocol
-----------------
    1. Fit on TRAIN with each hyperparameter combination.
    2. Evaluate on VAL for every combination.
    3. Select the combination with highest VAL AUC.
    4. Refit nothing — the selected model stays as-is.
    5. Evaluate on HOLDOUT once, as the unbiased estimate.

The holdout is touched exactly once. Everything else is measured on val.

Metrics
-------
For every split (train/val/holdout), we report:
    - AUC               discrimination (rank-ordering)
    - Gini              2 * AUC - 1
    - KS                Kolmogorov-Smirnov separability
    - Brier             mean squared error of probabilities
    - Calibration slope regression of observed on predicted (log-odds)
    - Calibration intercept offset in log-odds space

Outputs
-------
- artifacts/models/champion/<version>/model.pkl
- artifacts/models/champion/<version>/metadata.json
- artifacts/reports/champion_metrics.json
- artifacts/reports/champion_tuning.json
- artifacts/reports/champion_predictions.parquet
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


logger = logging.getLogger("credit_risk.models.champion")


# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------

# C is inverse regularization strength. Smaller → stronger regularization.
# Values chosen to span a wide range; the grid search will narrow down.
C_GRID: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1.0)

# Elastic-Net mixing parameter. 0.0 = L2 (ridge), 1.0 = L1 (lasso).
L1_RATIO_GRID: tuple[float, ...] = (0.0, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChampionConfig:
    """Resolved configuration for champion training."""

    project_root: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path

    train_file: str = "champion_train.parquet"
    val_file: str = "champion_val.parquet"
    holdout_file: str = "champion_holdout.parquet"

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    model_version: str = "0.1.0"

    c_grid: tuple[float, ...] = C_GRID
    l1_ratio_grid: tuple[float, ...] = L1_RATIO_GRID

    max_iter: int = 2000
    random_state: int = 42

    def train_path(self) -> Path:
        return self.processed_dir / self.train_file

    def val_path(self) -> Path:
        return self.processed_dir / self.val_file

    def holdout_path(self) -> Path:
        return self.processed_dir / self.holdout_file

    def model_dir(self) -> Path:
        return self.models_dir / "champion" / self.model_version

    def model_path(self) -> Path:
        return self.model_dir() / "model.pkl"

    def metadata_path(self) -> Path:
        return self.model_dir() / "metadata.json"

    def metrics_path(self) -> Path:
        return self.reports_dir / "champion_metrics.json"

    def tuning_path(self) -> Path:
        return self.reports_dir / "champion_tuning.json"

    def predictions_path(self) -> Path:
        return self.reports_dir / "champion_predictions.parquet"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SplitMetrics:
    """Metrics for one split (train/val/holdout)."""

    split: str
    n_rows: int
    n_positive: int
    positive_rate: float
    auc: float
    gini: float
    ks: float
    brier: float
    calibration_slope: float
    calibration_intercept: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "positive_rate": round(self.positive_rate, 6),
            "auc": round(self.auc, 6),
            "gini": round(self.gini, 6),
            "ks": round(self.ks, 6),
            "brier": round(self.brier, 6),
            "calibration_slope": round(self.calibration_slope, 6),
            "calibration_intercept": round(self.calibration_intercept, 6),
        }


@dataclass
class TuningResult:
    """One hyperparameter combination and its validation AUC."""

    C: float
    l1_ratio: float
    val_auc: float
    val_gini: float
    n_nonzero_coefs: int
    fit_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "C": self.C,
            "l1_ratio": self.l1_ratio,
            "val_auc": round(self.val_auc, 6),
            "val_gini": round(self.val_gini, 6),
            "n_nonzero_coefs": self.n_nonzero_coefs,
            "fit_seconds": round(self.fit_seconds, 3),
        }


@dataclass
class ChampionReport:
    """Aggregated champion training report."""

    generated_at: str
    model_version: str
    n_features: int
    best_C: float
    best_l1_ratio: float
    n_nonzero_coefs: int
    training_seconds: float
    split_metrics: list[SplitMetrics] = field(default_factory=list)
    tuning_results: list[TuningResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "n_features": self.n_features,
            "best_C": self.best_C,
            "best_l1_ratio": self.best_l1_ratio,
            "n_nonzero_coefs": self.n_nonzero_coefs,
            "training_seconds": round(self.training_seconds, 3),
            "split_metrics": [m.to_dict() for m in self.split_metrics],
            "tuning_results": [t.to_dict() for t in self.tuning_results],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC."""
    return float(roc_auc_score(y_true, y_score))


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation of positive/negative CDFs."""
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
    """Brier score: mean squared error of probabilities."""
    return float(brier_score_loss(y_true, y_prob))


def compute_calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """
    Fit a logistic regression of y_true on logit(y_prob).

    The fitted slope should be close to 1.0 for a well-calibrated model.
    The intercept should be close to 0.0.

    Uses C=1e10 to effectively disable regularization — the calibration
    regression should reflect the observed data, not a shrinkage estimate.
    """
    eps = 1e-9
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))

    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1e10,          # effectively unregularized
    )
    lr.fit(logit_p.reshape(-1, 1), y_true)

    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    return slope, intercept


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class ChampionTrainer:
    """
    Trains and evaluates the champion scorecard.

    Grid-searches over (C, l1_ratio), selects by validation AUC, fits
    the final model on train, and evaluates on holdout.
    """

    config: ChampionConfig
    report: ChampionReport | None = None
    model: Any = None

    # ---- public API ------------------------------------------------------

    def run(self) -> ChampionReport:
        start = time.monotonic()

        self.config.model_dir().mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        X_train, y_train = self._load_split(self.config.train_path(), "train")
        X_val, y_val = self._load_split(self.config.val_path(), "val")
        X_holdout, y_holdout = self._load_split(self.config.holdout_path(), "holdout")

        logger.info(
            "Loaded splits: train=%d, val=%d, holdout=%d",
            len(X_train), len(X_val), len(X_holdout),
        )
        logger.info("Feature count: %d", X_train.shape[1])

        # Grid search on val
        tuning_results, best_model, best_C, best_l1_ratio = self._grid_search(
            X_train, y_train, X_val, y_val
        )

        self.model = best_model

        # Evaluate on all three splits
        metrics = self._evaluate_all(
            model=best_model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_holdout=X_holdout, y_holdout=y_holdout,
        )

        # Persist model
        self._persist_model(best_model, X_train.shape[1], best_C, best_l1_ratio)

        # Predictions on all splits
        self._persist_predictions(
            model=best_model,
            X_train=X_train, X_val=X_val, X_holdout=X_holdout,
        )

        # Report
        duration = time.monotonic() - start
        self.report = self._build_report(
            model=best_model,
            best_C=best_C,
            best_l1_ratio=best_l1_ratio,
            n_features=X_train.shape[1],
            metrics=metrics,
            tuning_results=tuning_results,
            duration=duration,
        )
        self._write_reports(self.report)
        self._log_summary(self.report)
        return self.report

    # ---- input loading ---------------------------------------------------

    def _load_split(
        self,
        path: Path,
        name: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if not path.exists():
            raise FileNotFoundError(
                f"Split not found: {path}. Run binning first."
            )

        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("  loaded %s: %d rows, %d cols", name, len(df), len(df.columns))

        key = self.config.key_column
        target = self.config.target_column

        if key not in df.columns:
            raise KeyError(f"Key '{key}' not found in {path.name}")
        if target not in df.columns:
            raise KeyError(f"Target '{target}' not found in {path.name}")

        X = df.drop(columns=[key, target])
        y = df[target].astype(int)

        return X, y

    # ---- grid search -----------------------------------------------------

    def _grid_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[list[TuningResult], Any, float, float]:
        """
        Iterate over the hyperparameter grid, fit each model on train,
        evaluate on val, and return the results + best model.
        """
        n_combos = len(self.config.c_grid) * len(self.config.l1_ratio_grid)
        logger.info("Grid search: %d combinations", n_combos)

        results: list[TuningResult] = []
        best_model: Any = None
        best_auc: float = -1.0
        best_C: float = 0.0
        best_l1_ratio: float = 0.0

        for C in self.config.c_grid:
            for l1_ratio in self.config.l1_ratio_grid:
                fit_start = time.monotonic()

                model = self._fit_model(X_train, y_train, C, l1_ratio)
                p_val = model.predict_proba(X_val)[:, 1]

                val_auc = compute_auc(np.asarray(y_val), p_val)
                val_gini = 2 * val_auc - 1

                n_nonzero = int((model.coef_[0] != 0).sum())

                fit_seconds = time.monotonic() - fit_start

                result = TuningResult(
                    C=C,
                    l1_ratio=l1_ratio,
                    val_auc=val_auc,
                    val_gini=val_gini,
                    n_nonzero_coefs=n_nonzero,
                    fit_seconds=fit_seconds,
                )
                results.append(result)

                logger.info(
                    "  C=%.4f l1_ratio=%.2f  val_auc=%.4f  n_nonzero=%d  %.1fs",
                    C, l1_ratio, val_auc, n_nonzero, fit_seconds,
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model = model
                    best_C = C
                    best_l1_ratio = l1_ratio

        logger.info(
            "Best: C=%.4f l1_ratio=%.2f  val_auc=%.4f",
            best_C, best_l1_ratio, best_auc,
        )
        return results, best_model, best_C, best_l1_ratio

    def _fit_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        C: float,
        l1_ratio: float,
    ) -> LogisticRegression:
        """
        Fit a single Elastic-Net logistic regression.

        No `class_weight="balanced"`: the model needs to produce calibrated
        probabilities reflecting the true ~8% event rate, not reweighted
        ones. Discrimination (AUC) is unaffected by class weighting, but
        calibration is destroyed by it.

        No `penalty` argument: sklearn 1.8+ infers L2/L1/Elastic-Net from
        the `l1_ratio` value. Passing `penalty="elasticnet"` triggers a
        deprecation warning.
        """
        model = LogisticRegression(
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        model.fit(X, y)
        return model

    # ---- evaluation ------------------------------------------------------

    def _evaluate_all(
        self,
        model: LogisticRegression,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_holdout: pd.DataFrame, y_holdout: pd.Series,
    ) -> list[SplitMetrics]:
        """Compute metrics on all three splits."""
        metrics: list[SplitMetrics] = []

        for name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("holdout", X_holdout, y_holdout),
        ]:
            metrics.append(self._evaluate_split(model, X, y, name))

        return metrics

    def _evaluate_split(
        self,
        model: LogisticRegression,
        X: pd.DataFrame,
        y: pd.Series,
        name: str,
    ) -> SplitMetrics:
        p = model.predict_proba(X)[:, 1]
        y_arr = np.asarray(y)

        auc = compute_auc(y_arr, p)
        ks = compute_ks(y_arr, p)
        brier = compute_brier(y_arr, p)
        slope, intercept = compute_calibration_slope_intercept(y_arr, p)

        return SplitMetrics(
            split=name,
            n_rows=len(X),
            n_positive=int(y_arr.sum()),
            positive_rate=float(y_arr.mean()),
            auc=auc,
            gini=2 * auc - 1,
            ks=ks,
            brier=brier,
            calibration_slope=slope,
            calibration_intercept=intercept,
        )

    # ---- persistence -----------------------------------------------------

    def _persist_model(
        self,
        model: LogisticRegression,
        n_features: int,
        C: float,
        l1_ratio: float,
    ) -> None:
        """Save model pickle + metadata."""
        with self.config.model_path().open("wb") as f:
            pickle.dump(model, f)

        metadata = {
            "version": self.config.model_version,
            "algorithm": "LogisticRegression",
            "solver": "saga",
            "C": C,
            "l1_ratio": l1_ratio,
            "n_features": n_features,
            "n_nonzero_coefs": int((model.coef_[0] != 0).sum()),
            "intercept": float(model.intercept_[0]),
            "trained_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }
        with self.config.metadata_path().open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model persisted: %s", self.config.model_path())

    def _persist_predictions(
        self,
        model: LogisticRegression,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_holdout: pd.DataFrame,
    ) -> None:
        """Save predicted probabilities for all splits to a single Parquet."""
        train_p = model.predict_proba(X_train)[:, 1]
        val_p = model.predict_proba(X_val)[:, 1]
        holdout_p = model.predict_proba(X_holdout)[:, 1]

        df = pd.concat([
            pd.DataFrame({
                "split": "train",
                "row_index": np.arange(len(train_p)),
                "champion_pd": train_p,
            }),
            pd.DataFrame({
                "split": "val",
                "row_index": np.arange(len(val_p)),
                "champion_pd": val_p,
            }),
            pd.DataFrame({
                "split": "holdout",
                "row_index": np.arange(len(holdout_p)),
                "champion_pd": holdout_p,
            }),
        ], ignore_index=True)

        df.to_parquet(
            self.config.predictions_path(),
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        logger.info("Predictions written: %s", self.config.predictions_path().name)

    # ---- report ----------------------------------------------------------

    def _build_report(
        self,
        model: LogisticRegression,
        best_C: float,
        best_l1_ratio: float,
        n_features: int,
        metrics: list[SplitMetrics],
        tuning_results: list[TuningResult],
        duration: float,
    ) -> ChampionReport:
        notes: list[str] = []

        # Sanity notes
        holdout = next((m for m in metrics if m.split == "holdout"), None)
        train = next((m for m in metrics if m.split == "train"), None)

        if train and holdout:
            gap = train.auc - holdout.auc
            if gap > 0.05:
                notes.append(
                    f"Train-holdout AUC gap of {gap:.4f} suggests overfitting."
                )
            if holdout.auc < 0.70:
                notes.append(
                    f"Holdout AUC {holdout.auc:.4f} below expected range "
                    "for a well-tuned scorecard."
                )
            if holdout.auc > 0.85:
                notes.append(
                    f"Holdout AUC {holdout.auc:.4f} suspiciously high; "
                    "investigate for leakage."
                )
            if abs(holdout.calibration_slope - 1.0) > 0.15:
                notes.append(
                    f"Holdout calibration slope {holdout.calibration_slope:.3f} "
                    "outside [0.85, 1.15]; recalibration may be needed."
                )
            if abs(holdout.calibration_intercept) > 0.15:
                notes.append(
                    f"Holdout calibration intercept "
                    f"{holdout.calibration_intercept:.3f} outside [-0.15, 0.15]; "
                    "recalibration may be needed."
                )

        return ChampionReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            model_version=self.config.model_version,
            n_features=n_features,
            best_C=best_C,
            best_l1_ratio=best_l1_ratio,
            n_nonzero_coefs=int((model.coef_[0] != 0).sum()),
            training_seconds=duration,
            split_metrics=metrics,
            tuning_results=tuning_results,
            notes=notes,
        )

    def _write_reports(self, report: ChampionReport) -> None:
        with self.config.metrics_path().open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        with self.config.tuning_path().open("w", encoding="utf-8") as f:
            json.dump(
                [t.to_dict() for t in report.tuning_results],
                f, indent=2, default=str,
            )

        logger.info("Metrics written: %s", self.config.metrics_path().name)
        logger.info("Tuning written: %s", self.config.tuning_path().name)

    def _log_summary(self, report: ChampionReport) -> None:
        logger.info("=" * 70)
        logger.info("Champion training complete")
        logger.info("  model version:       %s", report.model_version)
        logger.info("  features:            %d", report.n_features)
        logger.info("  best C:              %.4f", report.best_C)
        logger.info("  best l1_ratio:       %.2f", report.best_l1_ratio)
        logger.info("  nonzero coefs:       %d", report.n_nonzero_coefs)
        logger.info("  training duration:   %.1fs", report.training_seconds)
        logger.info("  ---")
        logger.info("  Metrics by split:")
        for m in report.split_metrics:
            logger.info(
                "    %-8s  AUC=%.4f  Gini=%.4f  KS=%.4f  Brier=%.4f  "
                "cal_slope=%.3f  cal_int=%.3f",
                m.split, m.auc, m.gini, m.ks, m.brier,
                m.calibration_slope, m.calibration_intercept,
            )
        if report.notes:
            logger.info("  ---")
            for note in report.notes:
                logger.info("  Note: %s", note)
        logger.info("=" * 70)