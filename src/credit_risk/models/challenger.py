"""
Challenger model: LightGBM gradient boosting on raw-encoded features.

The challenger tests whether nonlinear interactions and native null
handling extract material lift over the linear champion. It consumes
the raw-encoded challenger feature store (categoricals factorized to
integer codes, nulls preserved, missing indicators included).

Training protocol
-----------------
    1. Optuna studies over a broad hyperparameter space.
    2. Objective: maximize validation AUC.
    3. Early stopping on validation AUC per trial.
    4. Best trial's configuration is retrained on train for a fixed
       number of rounds equal to the best trial's best_iteration.
    5. Evaluate on HOLDOUT once.

The holdout is touched exactly once. Hyperparameter selection uses val.

Design notes
------------
- Class imbalance is not corrected with `is_unbalance` or
  `scale_pos_weight`. The default binary objective with an appropriate
  decision threshold performs well, and reweighting would break
  calibration in the same way `class_weight="balanced"` did for the
  champion.
- LightGBM handles nulls natively during tree growth. No imputation.
- Categoricals are already integer codes from the challenger store.
- Early stopping uses validation AUC with a patience of 50 rounds.
- Optuna sampler: TPE (Tree-structured Parzen Estimator).
- The final model is trained WITHOUT a validation set, so LightGBM
  does not set `model.best_iteration`. The number of boosting rounds
  used is tracked explicitly as `final_n_rounds` and passed to every
  subsequent `predict()` call.
- Predictions are clamped to [0, 1] before metric computation and
  before persistence. LightGBM's sigmoid can produce tiny
  out-of-range values on extreme inputs due to floating-point
  imprecision; clamping ensures downstream consumers (metrics,
  decision engine, serving) never see invalid probabilities.

Outputs
-------
- artifacts/models/challenger/<version>/model.txt
- artifacts/models/challenger/<version>/metadata.json
- artifacts/reports/challenger_metrics.json
- artifacts/reports/challenger_tuning.json
- artifacts/reports/challenger_predictions.parquet
- artifacts/reports/challenger_feature_importance.csv
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
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import brier_score_loss, roc_auc_score


logger = logging.getLogger("credit_risk.models.challenger")


# ---------------------------------------------------------------------------
# Optuna configuration
# ---------------------------------------------------------------------------

N_TRIALS = 30
EARLY_STOPPING_ROUNDS = 50
MAX_BOOST_ROUNDS = 2000


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChallengerConfig:
    """Resolved configuration for challenger training."""

    project_root: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path

    train_file: str = "challenger_train.parquet"
    val_file: str = "challenger_val.parquet"
    holdout_file: str = "challenger_holdout.parquet"

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    model_version: str = "0.1.0"

    n_trials: int = N_TRIALS
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS
    max_boost_rounds: int = MAX_BOOST_ROUNDS

    random_state: int = 42

    def train_path(self) -> Path:
        return self.processed_dir / self.train_file

    def val_path(self) -> Path:
        return self.processed_dir / self.val_file

    def holdout_path(self) -> Path:
        return self.processed_dir / self.holdout_file

    def model_dir(self) -> Path:
        return self.models_dir / "challenger" / self.model_version

    def model_path(self) -> Path:
        return self.model_dir() / "model.txt"

    def metadata_path(self) -> Path:
        return self.model_dir() / "metadata.json"

    def metrics_path(self) -> Path:
        return self.reports_dir / "challenger_metrics.json"

    def tuning_path(self) -> Path:
        return self.reports_dir / "challenger_tuning.json"

    def predictions_path(self) -> Path:
        return self.reports_dir / "challenger_predictions.parquet"

    def importance_path(self) -> Path:
        return self.reports_dir / "challenger_feature_importance.csv"


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
class TrialResult:
    """One Optuna trial's outcome."""

    trial_number: int
    val_auc: float
    best_iteration: int
    params: dict[str, Any]
    fit_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "val_auc": round(self.val_auc, 6),
            "best_iteration": self.best_iteration,
            "params": self.params,
            "fit_seconds": round(self.fit_seconds, 3),
        }


@dataclass
class ChallengerReport:
    """Aggregated challenger training report."""

    generated_at: str
    model_version: str
    n_features: int
    n_trials_completed: int
    best_trial_number: int
    best_iteration: int
    best_params: dict[str, Any]
    training_seconds: float
    split_metrics: list[SplitMetrics] = field(default_factory=list)
    trial_results: list[TrialResult] = field(default_factory=list)
    top_features: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "model_version": self.model_version,
            "n_features": self.n_features,
            "n_trials_completed": self.n_trials_completed,
            "best_trial_number": self.best_trial_number,
            "best_iteration": self.best_iteration,
            "best_params": self.best_params,
            "training_seconds": round(self.training_seconds, 3),
            "split_metrics": [m.to_dict() for m in self.split_metrics],
            "trial_results": [t.to_dict() for t in self.trial_results],
            "top_features": self.top_features,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Metric helpers (same as champion; will be extracted to evaluate.py later)
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_ks(y_true: np.ndarray, y_score: np.ndarray) -> float:
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
    return float(brier_score_loss(y_true, y_prob))


def compute_calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression

    eps = 1e-9
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)

    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class ChallengerTrainer:
    """
    Trains and evaluates the challenger LightGBM model.

    Optuna hyperparameter search on validation AUC, then evaluation on
    all three splits with the best model.
    """

    config: ChallengerConfig
    report: ChallengerReport | None = None
    model: lgb.Booster | None = None
    feature_names: list[str] = field(default_factory=list)
    final_n_rounds: int = 0

    # ---- public API ------------------------------------------------------

    def run(self) -> ChallengerReport:
        start = time.monotonic()

        self.config.model_dir().mkdir(parents=True, exist_ok=True)
        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        X_train, y_train = self._load_split(self.config.train_path(), "train")
        X_val, y_val = self._load_split(self.config.val_path(), "val")
        X_holdout, y_holdout = self._load_split(self.config.holdout_path(), "holdout")

        self.feature_names = list(X_train.columns)

        logger.info(
            "Loaded splits: train=%d, val=%d, holdout=%d",
            len(X_train), len(X_val), len(X_holdout),
        )
        logger.info("Feature count: %d", X_train.shape[1])

        # Optuna hyperparameter search
        trial_results, best_params, best_iteration = self._optuna_search(
            X_train, y_train, X_val, y_val
        )

        best_val_auc = max(t.val_auc for t in trial_results)
        logger.info(
            "Best trial: iteration=%d  val_auc=%.4f",
            best_iteration, best_val_auc,
        )

        # Retrain on train with best params and best_iteration rounds
        logger.info("Retraining best model with %d rounds", best_iteration)
        self.model, self.final_n_rounds = self._fit_final_model(
            X_train, y_train, best_params, best_iteration
        )
        logger.info("Final model trained with %d trees", self.final_n_rounds)

        # Evaluate on all three splits
        metrics = self._evaluate_all(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_holdout=X_holdout, y_holdout=y_holdout,
        )

        # Persist model
        self._persist_model(best_params, X_train.shape[1])

        # Predictions
        self._persist_predictions(X_train, X_val, X_holdout)

        # Feature importance
        top_features = self._persist_feature_importance()

        # Report
        duration = time.monotonic() - start
        self.report = self._build_report(
            best_params=best_params,
            best_iteration=self.final_n_rounds,
            n_features=X_train.shape[1],
            metrics=metrics,
            trial_results=trial_results,
            top_features=top_features,
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
                f"Split not found: {path}. Run store first."
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

    # ---- Optuna search ---------------------------------------------------

    def _optuna_search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[list[TrialResult], dict[str, Any], int]:
        """
        Run Optuna with TPE sampler. Returns:
            (trial_results, best_params, best_iteration)
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info("Optuna search: %d trials", self.config.n_trials)

        trial_results: list[TrialResult] = []

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)

            fit_start = time.monotonic()
            model = self._fit_trial_model(X_train, y_train, X_val, y_val, params)
            fit_seconds = time.monotonic() - fit_start

            # Per-trial model was fit WITH a validation set, so
            # `model.best_iteration` is set correctly by early stopping.
            trial_best_iteration = int(
                model.best_iteration or self.config.max_boost_rounds
            )

            p_val = model.predict(X_val, num_iteration=trial_best_iteration)
            p_val = np.clip(p_val, 0.0, 1.0)  # defensive
            val_auc = compute_auc(np.asarray(y_val), p_val)

            trial_results.append(
                TrialResult(
                    trial_number=trial.number,
                    val_auc=val_auc,
                    best_iteration=trial_best_iteration,
                    params=params,
                    fit_seconds=fit_seconds,
                )
            )

            logger.info(
                "  trial %d: val_auc=%.4f  best_iter=%d  %.1fs",
                trial.number, val_auc, trial_best_iteration, fit_seconds,
            )

            return val_auc

        sampler = TPESampler(seed=self.config.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        best_trial_number = study.best_trial.number

        best_trial = next(
            t for t in trial_results if t.trial_number == best_trial_number
        )

        logger.info(
            "Optuna complete: best trial=%d  val_auc=%.4f  iter=%d",
            best_trial_number, best_trial.val_auc, best_trial.best_iteration,
        )

        return trial_results, best_params, best_trial.best_iteration

    @staticmethod
    def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
        """Define the Optuna search space."""
        return {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.10, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, step=8),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 200, step=10
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "random_state": 42,
            "n_jobs": -1,
        }

    def _fit_trial_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: dict[str, Any],
    ) -> lgb.Booster:
        """Fit one model with early stopping on validation AUC."""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.config.early_stopping_rounds,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.max_boost_rounds,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=callbacks,
        )
        return model

    def _fit_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: dict[str, Any],
        n_rounds: int,
    ) -> tuple[lgb.Booster, int]:
        """
        Refit the best configuration on train for a fixed number of rounds.

        Returns (model, n_rounds). Because the model is trained without a
        validation set, LightGBM does not set `model.best_iteration`. The
        caller must use the returned `n_rounds` when calling predict().
        """
        if n_rounds <= 0:
            raise ValueError(
                f"_fit_final_model requires a positive round count, got {n_rounds}"
            )

        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_rounds,
        )

        actual = model.current_iteration()
        if actual != n_rounds:
            logger.warning(
                "Model has %d trees, expected %d", actual, n_rounds
            )

        return model, n_rounds

    # ---- evaluation ------------------------------------------------------

    def _evaluate_all(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series,
        X_holdout: pd.DataFrame, y_holdout: pd.Series,
    ) -> list[SplitMetrics]:
        metrics: list[SplitMetrics] = []
        for name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("holdout", X_holdout, y_holdout),
        ]:
            metrics.append(self._evaluate_split(X, y, name))
        return metrics

    def _evaluate_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        name: str,
    ) -> SplitMetrics:
        p = self.model.predict(X, num_iteration=self.final_n_rounds)
        # Defensive: LightGBM's sigmoid can produce tiny out-of-range values
        # on extreme inputs due to floating-point imprecision. Clamp to [0, 1]
        # so metrics and downstream consumers always see valid probabilities.
        p = np.clip(p, 0.0, 1.0)

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
        best_params: dict[str, Any],
        n_features: int,
    ) -> None:
        """Save LightGBM model as text, plus metadata JSON."""
        self.model.save_model(
            str(self.config.model_path()),
            num_iteration=self.final_n_rounds,
        )

        metadata = {
            "version": self.config.model_version,
            "algorithm": "LightGBM",
            "n_rounds": self.final_n_rounds,
            "params": best_params,
            "n_features": n_features,
            "trained_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }
        with self.config.metadata_path().open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Model persisted: %s", self.config.model_path())

    def _persist_predictions(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_holdout: pd.DataFrame,
    ) -> None:
        iters = self.final_n_rounds
        train_p = np.clip(self.model.predict(X_train, num_iteration=iters), 0.0, 1.0)
        val_p = np.clip(self.model.predict(X_val, num_iteration=iters), 0.0, 1.0)
        holdout_p = np.clip(
            self.model.predict(X_holdout, num_iteration=iters), 0.0, 1.0
        )

        df = pd.concat([
            pd.DataFrame({
                "split": "train",
                "row_index": np.arange(len(train_p)),
                "challenger_pd": train_p,
            }),
            pd.DataFrame({
                "split": "val",
                "row_index": np.arange(len(val_p)),
                "challenger_pd": val_p,
            }),
            pd.DataFrame({
                "split": "holdout",
                "row_index": np.arange(len(holdout_p)),
                "challenger_pd": holdout_p,
            }),
        ], ignore_index=True)

        df.to_parquet(
            self.config.predictions_path(),
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        logger.info("Predictions written: %s", self.config.predictions_path().name)

    def _persist_feature_importance(self) -> list[dict[str, Any]]:
        """Emit feature importance as CSV + return top 20 for the report."""
        importance_gain = self.model.feature_importance(importance_type="gain")
        importance_split = self.model.feature_importance(importance_type="split")

        df = pd.DataFrame({
            "feature": self.feature_names,
            "gain": importance_gain,
            "split": importance_split,
        }).sort_values("gain", ascending=False).reset_index(drop=True)

        df.to_csv(self.config.importance_path(), index=False)
        logger.info("Feature importance written: %s", self.config.importance_path().name)

        return df.head(20).to_dict(orient="records")

    # ---- report ----------------------------------------------------------

    def _build_report(
        self,
        best_params: dict[str, Any],
        best_iteration: int,
        n_features: int,
        metrics: list[SplitMetrics],
        trial_results: list[TrialResult],
        top_features: list[dict[str, Any]],
        duration: float,
    ) -> ChallengerReport:
        notes: list[str] = []

        holdout = next((m for m in metrics if m.split == "holdout"), None)
        train = next((m for m in metrics if m.split == "train"), None)

        if train and holdout:
            gap = train.auc - holdout.auc
            if gap > 0.10:
                notes.append(
                    f"Train-holdout AUC gap of {gap:.4f} suggests overfitting."
                )
            if holdout.auc < 0.75:
                notes.append(
                    f"Holdout AUC {holdout.auc:.4f} below expected range "
                    "for gradient boosting on this dataset."
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
            if abs(holdout.calibration_intercept) > 0.20:
                notes.append(
                    f"Holdout calibration intercept "
                    f"{holdout.calibration_intercept:.3f} outside [-0.20, 0.20]; "
                    "recalibration may be needed."
                )

        best_trial = max(trial_results, key=lambda t: t.val_auc) if trial_results else None
        best_trial_number = best_trial.trial_number if best_trial else -1

        return ChallengerReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            model_version=self.config.model_version,
            n_features=n_features,
            n_trials_completed=len(trial_results),
            best_trial_number=best_trial_number,
            best_iteration=best_iteration,
            best_params=best_params,
            training_seconds=duration,
            split_metrics=metrics,
            trial_results=trial_results,
            top_features=top_features,
            notes=notes,
        )

    def _write_reports(self, report: ChallengerReport) -> None:
        with self.config.metrics_path().open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        with self.config.tuning_path().open("w", encoding="utf-8") as f:
            json.dump(
                [t.to_dict() for t in report.trial_results],
                f, indent=2, default=str,
            )

        logger.info("Metrics written: %s", self.config.metrics_path().name)
        logger.info("Tuning written: %s", self.config.tuning_path().name)

    def _log_summary(self, report: ChallengerReport) -> None:
        logger.info("=" * 70)
        logger.info("Challenger training complete")
        logger.info("  model version:       %s", report.model_version)
        logger.info("  features:            %d", report.n_features)
        logger.info("  trials completed:    %d", report.n_trials_completed)
        logger.info("  best trial:          %d", report.best_trial_number)
        logger.info("  best iteration:      %d", report.best_iteration)
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
        logger.info("  ---")
        logger.info("  Top 10 features by gain:")
        for i, feat in enumerate(report.top_features[:10], 1):
            logger.info(
                "    #%2d  %-40s  gain=%.1f",
                i, feat["feature"], feat["gain"],
            )
        if report.notes:
            logger.info("  ---")
            for note in report.notes:
                logger.info("  Note: %s", note)
        logger.info("=" * 70)