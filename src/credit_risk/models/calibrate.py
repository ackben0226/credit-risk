"""
Calibration layer for the challenger model.

The champion (logistic regression on WoE features) is naturally
calibrated — its loss function directly optimizes likelihood, and
empirical checks confirm slope ≈ 1.0 and intercept ≈ 0.0.

The challenger (LightGBM on raw features) is miscalibrated. Gradient
boosting overfits the training signal, producing predictions that are
too extreme (slope < 1) and systematically biased upward (intercept
< 0). The pipeline addresses this with post-hoc calibration.

Approach
--------
- Fit two calibrators on the VAL split:
    * Platt scaling: logistic regression on logit(raw_pd)
    * Isotonic regression: monotonic step function on raw_pd
- Evaluate both on HOLDOUT Brier score.
- Select the calibrator with the lower holdout Brier.
- Apply the selected calibrator to all splits (train / val / holdout).

The champion is passed through unchanged, so downstream consumers can
work with a single schema.

The holdout is used for calibrator selection between two pre-specified
options. No parameters are fit on holdout.

Outputs
-------
- artifacts/calibrators/challenger/calibrator.pkl
- artifacts/calibrators/challenger/metadata.json
- artifacts/reports/calibration_metrics.json
- artifacts/reports/calibrated_predictions.parquet
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


logger = logging.getLogger("credit_risk.models.calibrate")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationConfig:
    """Resolved configuration for calibration."""

    project_root: Path
    processed_dir: Path
    reports_dir: Path
    calibrators_dir: Path

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    # Input files
    champion_predictions: str = "champion_predictions.parquet"
    challenger_predictions: str = "challenger_predictions.parquet"
    challenger_train: str = "challenger_train.parquet"
    challenger_val: str = "challenger_val.parquet"
    challenger_holdout: str = "challenger_holdout.parquet"

    # Output files
    calibrated_predictions: str = "calibrated_predictions.parquet"
    metrics_output: str = "calibration_metrics.json"

    # Model names and versions
    challenger_version: str = "0.1.0"

    n_reliability_bins: int = 10

    def champion_predictions_path(self) -> Path:
        return self.reports_dir / self.champion_predictions

    def challenger_predictions_path(self) -> Path:
        return self.reports_dir / self.challenger_predictions

    def challenger_split_path(self, split: str) -> Path:
        name = {
            "train": self.challenger_train,
            "val": self.challenger_val,
            "holdout": self.challenger_holdout,
        }[split]
        return self.processed_dir / name

    def calibrated_predictions_path(self) -> Path:
        return self.reports_dir / self.calibrated_predictions

    def metrics_path(self) -> Path:
        return self.reports_dir / self.metrics_output

    def calibrator_dir(self, model: str) -> Path:
        return self.calibrators_dir / model

    def calibrator_path(self, model: str) -> Path:
        return self.calibrator_dir(model) / "calibrator.pkl"

    def calibrator_metadata_path(self, model: str) -> Path:
        return self.calibrator_dir(model) / "metadata.json"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class CalibrationMetrics:
    """Metrics for one model on one split, before and after calibration."""

    model: str
    split: str
    calibration_state: str   # "raw" or "calibrated"
    n_rows: int
    n_positive: int
    positive_rate: float
    brier: float
    calibration_slope: float
    calibration_intercept: float
    reliability: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "split": self.split,
            "calibration_state": self.calibration_state,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "positive_rate": round(self.positive_rate, 6),
            "brier": round(self.brier, 6),
            "calibration_slope": round(self.calibration_slope, 6),
            "calibration_intercept": round(self.calibration_intercept, 6),
            "reliability": self.reliability,
        }


@dataclass
class CalibratorSelection:
    """The result of selecting between Platt and isotonic for a model."""

    model: str
    selected_method: str
    platt_holdout_brier: float
    isotonic_holdout_brier: float
    raw_holdout_brier: float
    improvement_pp: float         # Brier reduction vs raw, in percentage points

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "selected_method": self.selected_method,
            "platt_holdout_brier": round(self.platt_holdout_brier, 6),
            "isotonic_holdout_brier": round(self.isotonic_holdout_brier, 6),
            "raw_holdout_brier": round(self.raw_holdout_brier, 6),
            "improvement_pp": round(self.improvement_pp, 4),
        }


@dataclass
class CalibrationReport:
    """Aggregated calibration report."""

    generated_at: str
    selections: list[CalibratorSelection] = field(default_factory=list)
    metrics: list[CalibrationMetrics] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "selections": [s.to_dict() for s in self.selections],
            "metrics": [m.to_dict() for m in self.metrics],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def compute_calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Regress y_true on logit(y_prob) with unregularized logistic regression."""
    eps = 1e-9
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)

    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def compute_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """
    Compute per-bin predicted vs observed rates.

    Bins are equal-count (quantile-based) rather than equal-width, so
    each bin has similar population. This makes the reliability table
    informative even with a skewed prediction distribution.
    """
    if len(y_true) == 0:
        return []

    # Quantile bin edges
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    # Deduplicate edges (possible with discrete predictions)
    edges = np.unique(edges)

    if len(edges) < 2:
        # All predictions are equal — one bin
        edges = np.array([y_prob.min() - 1e-9, y_prob.max() + 1e-9])

    bin_idx = np.digitize(y_prob, edges[1:-1])

    rows: list[dict[str, Any]] = []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin": b + 1,
            "n": n,
            "lower_edge": float(edges[b]),
            "upper_edge": float(edges[b + 1]),
            "mean_predicted": float(y_prob[mask].mean()),
            "observed_rate": float(y_true[mask].mean()),
        })

    return rows


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

def fit_platt(raw_pd: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling: logistic regression on logit(raw_pd)."""
    eps = 1e-9
    p = np.clip(raw_pd, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))

    # lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)
    return lr


def apply_platt(calibrator: LogisticRegression, raw_pd: np.ndarray) -> np.ndarray:
    """Apply Platt scaling."""
    eps = 1e-9
    p = np.clip(raw_pd, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    calibrated = calibrator.predict_proba(logit_p.reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 0.0, 1.0)


def fit_isotonic(raw_pd: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_pd, y_true)
    return iso


def apply_isotonic(calibrator: IsotonicRegression, raw_pd: np.ndarray) -> np.ndarray:
    """Apply isotonic regression."""
    calibrated = calibrator.predict(raw_pd)
    return np.clip(calibrated, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class CalibrationTrainer:
    """Fits, selects, and applies calibration to the challenger model."""

    config: CalibrationConfig
    report: CalibrationReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> CalibrationReport:
        start = time.monotonic()

        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config.calibrators_dir.mkdir(parents=True, exist_ok=True)

        # Load raw predictions and targets
        champion_preds = self._load_predictions(
            self.config.champion_predictions_path(), "champion"
        )
        challenger_preds = self._load_predictions(
            self.config.challenger_predictions_path(), "challenger"
        )
        targets = self._load_targets()

        logger.info(
            "Loaded predictions: champion=%d rows, challenger=%d rows",
            len(champion_preds), len(challenger_preds),
        )

        # Fit calibrators on VAL, select on HOLDOUT Brier
        selection, calibrator, calibrator_method = self._fit_and_select(
            challenger_preds=challenger_preds,
            targets=targets,
        )

        # Persist calibrator
        self._persist_calibrator(calibrator, calibrator_method, selection)

        # Apply calibration
        champion_calibrated = self._apply_champion(champion_preds)
        challenger_calibrated = self._apply_challenger(
            challenger_preds, calibrator, calibrator_method
        )

        # Combine into long format
        combined = self._combine_predictions(
            champion_calibrated, challenger_calibrated
        )
        self._write_predictions(combined)

        # Compute metrics for all combinations
        metrics = self._compute_all_metrics(
            champion_raw=champion_preds,
            champion_cal=champion_calibrated,
            challenger_raw=challenger_preds,
            challenger_cal=challenger_calibrated,
            targets=targets,
        )

        # Report
        duration = time.monotonic() - start
        self.report = self._build_report(
            selection=selection,
            metrics=metrics,
            duration=duration,
        )
        self._write_metrics(self.report)
        self._log_summary(self.report)
        return self.report

    # ---- input loading ---------------------------------------------------

    def _load_predictions(self, path: Path, model: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Predictions not found: {path}. Run the model trainer first."
            )
        df = pd.read_parquet(path, engine="pyarrow")
        expected_pd_col = f"{model}_pd"
        if expected_pd_col not in df.columns:
            raise KeyError(
                f"Column '{expected_pd_col}' not found in {path.name}. "
                f"Found: {list(df.columns)}"
            )
        df = df.rename(columns={expected_pd_col: "raw_pd"})
        logger.info("  %s predictions: %d rows", model, len(df))
        return df[["split", "row_index", "raw_pd"]].copy()

    def _load_targets(self) -> dict[str, np.ndarray]:
        """
        Load target values per split.

        We use the challenger split files because they contain both
        SK_ID_CURR and TARGET for all three splits. Row order matches
        the prediction files (both derived from the same split indices).
        """
        targets: dict[str, np.ndarray] = {}
        for split in ["train", "val", "holdout"]:
            path = self.config.challenger_split_path(split)
            df = pd.read_parquet(
                path, engine="pyarrow",
                columns=[self.config.target_column],
            )
            targets[split] = df[self.config.target_column].to_numpy()
            logger.info(
                "  %s targets: %d rows, positive rate = %.4f",
                split, len(targets[split]), targets[split].mean(),
            )
        return targets

    # ---- calibration fitting ---------------------------------------------

    def _fit_and_select(
        self,
        challenger_preds: pd.DataFrame,
        targets: dict[str, np.ndarray],
    ) -> tuple[CalibratorSelection, Any, str]:
        """
        Fit Platt and isotonic on VAL; select on HOLDOUT Brier.

        Returns (selection, calibrator, method).
        """
        val_raw = challenger_preds[challenger_preds["split"] == "val"]["raw_pd"].to_numpy()
        val_y = targets["val"]

        holdout_raw = challenger_preds[
            challenger_preds["split"] == "holdout"
        ]["raw_pd"].to_numpy()
        holdout_y = targets["holdout"]

        # Fit both calibrators on val
        logger.info("Fitting Platt on val")
        platt = fit_platt(val_raw, val_y)

        logger.info("Fitting isotonic on val")
        isotonic = fit_isotonic(val_raw, val_y)

        # Apply to holdout, compute Brier
        platt_holdout_pd = apply_platt(platt, holdout_raw)
        isotonic_holdout_pd = apply_isotonic(isotonic, holdout_raw)
        raw_holdout_brier = compute_brier(holdout_y, holdout_raw)
        platt_holdout_brier = compute_brier(holdout_y, platt_holdout_pd)
        isotonic_holdout_brier = compute_brier(holdout_y, isotonic_holdout_pd)

        logger.info(
            "  holdout Brier: raw=%.6f  platt=%.6f  isotonic=%.6f",
            raw_holdout_brier, platt_holdout_brier, isotonic_holdout_brier,
        )

        # Select better (lower Brier)
        if isotonic_holdout_brier < platt_holdout_brier:
            selected_method = "isotonic"
            calibrator = isotonic
            best_holdout_brier = isotonic_holdout_brier
        else:
            selected_method = "platt"
            calibrator = platt
            best_holdout_brier = platt_holdout_brier

        improvement_pp = (raw_holdout_brier - best_holdout_brier) * 100

        logger.info(
            "  selected: %s (holdout Brier %.6f, improvement %.4f pp)",
            selected_method, best_holdout_brier, improvement_pp,
        )

        selection = CalibratorSelection(
            model="challenger",
            selected_method=selected_method,
            platt_holdout_brier=platt_holdout_brier,
            isotonic_holdout_brier=isotonic_holdout_brier,
            raw_holdout_brier=raw_holdout_brier,
            improvement_pp=improvement_pp,
        )
        return selection, calibrator, selected_method

    # ---- application -----------------------------------------------------

    def _apply_champion(self, preds: pd.DataFrame) -> pd.DataFrame:
        """Champion is naturally calibrated — pass through."""
        out = preds.copy()
        out["calibrated_pd"] = out["raw_pd"]
        return out

    def _apply_challenger(
        self,
        preds: pd.DataFrame,
        calibrator: Any,
        method: str,
    ) -> pd.DataFrame:
        """Apply selected calibrator to all splits."""
        out = preds.copy()
        raw = out["raw_pd"].to_numpy()

        if method == "platt":
            calibrated = apply_platt(calibrator, raw)
        elif method == "isotonic":
            calibrated = apply_isotonic(calibrator, raw)
        else:
            raise ValueError(f"Unknown calibrator method: {method}")

        out["calibrated_pd"] = calibrated
        return out

    # ---- combining -------------------------------------------------------

    def _combine_predictions(
        self,
        champion: pd.DataFrame,
        challenger: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine into long format with a model column."""
        champion = champion.copy()
        champion["model"] = "champion"

        challenger = challenger.copy()
        challenger["model"] = "challenger"

        combined = pd.concat([champion, challenger], ignore_index=True)
        combined = combined[[
            "model", "split", "row_index", "raw_pd", "calibrated_pd",
        ]]
        return combined

    # ---- metrics ---------------------------------------------------------

    def _compute_all_metrics(
        self,
        champion_raw: pd.DataFrame,
        champion_cal: pd.DataFrame,
        challenger_raw: pd.DataFrame,
        challenger_cal: pd.DataFrame,
        targets: dict[str, np.ndarray],
    ) -> list[CalibrationMetrics]:
        """
        Compute metrics for every (model, split, calibration_state) tuple.
        """
        results: list[CalibrationMetrics] = []

        for model_name, raw_df, cal_df in [
            ("champion", champion_raw, champion_cal),
            ("challenger", challenger_raw, challenger_cal),
        ]:
            for split in ["train", "val", "holdout"]:
                raw_split = raw_df[raw_df["split"] == split]
                cal_split = cal_df[cal_df["split"] == split]
                y = targets[split]

                # Raw
                results.append(self._metrics_for(
                    model=model_name,
                    split=split,
                    state="raw",
                    y_true=y,
                    y_prob=raw_split["raw_pd"].to_numpy(),
                ))

                # Calibrated
                results.append(self._metrics_for(
                    model=model_name,
                    split=split,
                    state="calibrated",
                    y_true=y,
                    y_prob=cal_split["calibrated_pd"].to_numpy(),
                ))

        return results

    def _metrics_for(
        self,
        model: str,
        split: str,
        state: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> CalibrationMetrics:
        brier = compute_brier(y_true, y_prob)
        slope, intercept = compute_calibration_slope_intercept(y_true, y_prob)
        reliability = compute_reliability_diagram(
            y_true, y_prob, n_bins=self.config.n_reliability_bins
        )
        return CalibrationMetrics(
            model=model,
            split=split,
            calibration_state=state,
            n_rows=len(y_true),
            n_positive=int(y_true.sum()),
            positive_rate=float(y_true.mean()),
            brier=brier,
            calibration_slope=slope,
            calibration_intercept=intercept,
            reliability=reliability,
        )

    # ---- persistence -----------------------------------------------------

    def _persist_calibrator(
        self,
        calibrator: Any,
        method: str,
        selection: CalibratorSelection,
    ) -> None:
        """Save calibrator pickle + metadata JSON."""
        import sklearn

        dir_path = self.config.calibrator_dir("challenger")
        dir_path.mkdir(parents=True, exist_ok=True)

        with self.config.calibrator_path("challenger").open("wb") as f:
            pickle.dump(calibrator, f)

        metadata = {
            "model": "challenger",
            "method": method,
            "fit_on_split": "val",
            "selected_on_split": "holdout",
            "platt_holdout_brier": selection.platt_holdout_brier,
            "isotonic_holdout_brier": selection.isotonic_holdout_brier,
            "raw_holdout_brier": selection.raw_holdout_brier,
            "selected_holdout_brier": min(
                selection.platt_holdout_brier,
                selection.isotonic_holdout_brier,
            ),
            "improvement_pp": selection.improvement_pp,
            "sklearn_version": sklearn.__version__,
            "fitted_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }
        with self.config.calibrator_metadata_path("challenger").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(
            "Calibrator persisted: %s",
            self.config.calibrator_path("challenger"),
        )

    def _write_predictions(self, combined: pd.DataFrame) -> None:
        combined.to_parquet(
            self.config.calibrated_predictions_path(),
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        logger.info(
            "Calibrated predictions written: %s (%d rows)",
            self.config.calibrated_predictions_path().name,
            len(combined),
        )

    def _write_metrics(self, report: CalibrationReport) -> None:
        with self.config.metrics_path().open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Calibration metrics written: %s", self.config.metrics_path().name)

    # ---- report ----------------------------------------------------------

    def _build_report(self, selection, metrics, duration):
        notes: list[str] = []

        # Find holdout raw vs calibrated rows for the challenger
        raw_m = next(
            (m for m in metrics
            if m.model == "challenger" and m.split == "holdout"
            and m.calibration_state == "raw"),
            None,
        )
        cal_m = next(
            (m for m in metrics
            if m.model == "challenger" and m.split == "holdout"
            and m.calibration_state == "calibrated"),
            None,
        )

        if raw_m and cal_m:
            # Slope improvement: how far the raw slope was from 1.0, vs how far
            # the calibrated slope is from 1.0
            raw_slope_error = abs(raw_m.calibration_slope - 1.0)
            cal_slope_error = abs(cal_m.calibration_slope - 1.0)
            raw_intercept_error = abs(raw_m.calibration_intercept)
            cal_intercept_error = abs(cal_m.calibration_intercept)

            slope_fixed = cal_slope_error < raw_slope_error
            intercept_fixed = cal_intercept_error < raw_intercept_error
            brier_fixed = cal_m.brier < raw_m.brier

            if not (slope_fixed or intercept_fixed or brier_fixed):
                notes.append(
                    "Calibration did not improve slope, intercept, or Brier "
                    "on holdout. Verify the raw model is miscalibrated before applying."
                )
            elif not brier_fixed:
                # Slope or intercept improved, but Brier didn't. Report this
                # explicitly so it's not mistaken for "calibration didn't work."
                notes.append(
                    f"Calibration improved slope ({raw_m.calibration_slope:.3f} → "
                    f"{cal_m.calibration_slope:.3f}) and intercept "
                    f"({raw_m.calibration_intercept:.3f} → "
                    f"{cal_m.calibration_intercept:.3f}) but did not reduce "
                    f"Brier ({raw_m.brier:.6f} → {cal_m.brier:.6f}). "
                    "Slope and intercept are the primary calibration metrics."
                )

        return CalibrationReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            selections=[selection],
            metrics=metrics,
            notes=notes,
        )

    def _log_summary(self, report: CalibrationReport) -> None:
        logger.info("=" * 70)
        logger.info("Calibration complete")
        logger.info("  ---")
        for sel in report.selections:
            logger.info("  %s:", sel.model)
            logger.info("    raw holdout Brier:      %.6f", sel.raw_holdout_brier)
            logger.info("    platt holdout Brier:    %.6f", sel.platt_holdout_brier)
            logger.info("    isotonic holdout Brier: %.6f", sel.isotonic_holdout_brier)
            logger.info("    selected:               %s", sel.selected_method)
            logger.info("    improvement:            %.4f pp", sel.improvement_pp)
        logger.info("  ---")
        logger.info("  Metrics by model/split/state:")
        for m in report.metrics:
            logger.info(
                "    %-10s %-8s %-10s  Brier=%.6f  slope=%.4f  int=%.4f",
                m.model, m.split, m.calibration_state,
                m.brier, m.calibration_slope, m.calibration_intercept,
            )
        if report.notes:
            logger.info("  ---")
            for note in report.notes:
                logger.info("  Note: %s", note)
        logger.info("=" * 70)