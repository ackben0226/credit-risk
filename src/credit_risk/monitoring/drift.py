"""
Drift detection and monitoring for the credit risk model.

Compares a monitoring population (typically recent data or holdout)
against a reference population (typically the training split) on:

    - Per-feature distribution drift (PSI, KS)
    - Prediction distribution drift (PSI on calibrated PD)
    - Calibration drift (slope, intercept, Brier)
    - Performance drift (AUC, Gini, KS)

Each metric is classified as stable / moderate / material based on
configurable thresholds. The report identifies the top-N drifted
features so the team can investigate.

Design notes
------------
- PSI (Population Stability Index) compares the binned distribution
  of a feature between reference and monitoring populations. It is
  the standard metric in credit risk monitoring.
- KS (Kolmogorov-Smirnov) provides a complementary view: it measures
  the maximum separation between the two CDFs.
- Nulls are treated as their own bin for PSI. This preserves the
  missingness signal, which is often a drift indicator in itself.
- All comparisons are deterministic; the same inputs produce the
  same output.

Outputs
-------
- artifacts/reports/monitoring_report.json
- artifacts/reports/feature_drift.parquet
- artifacts/reports/prediction_drift.parquet
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
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score


logger = logging.getLogger("credit_risk.monitoring.drift")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonitoringConfig:
    """Resolved monitoring configuration."""

    project_root: Path
    processed_dir: Path
    reports_dir: Path

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    reference_split: str = "train"
    monitoring_split: str = "holdout"

    # PSI thresholds
    psi_stable_max: float = 0.10
    psi_moderate_max: float = 0.25

    # KS threshold
    ks_drift_threshold: float = 0.05

    # Calibration thresholds
    calibration_slope_min: float = 0.90
    calibration_slope_max: float = 1.10
    calibration_intercept_abs_max: float = 0.15

    # Performance thresholds
    min_auc: float = 0.70
    max_auc_drop: float = 0.03

    # Reporting
    top_n_drifted: int = 20
    top_n_features_ranked_by: str = "psi"

    def challenger_split_path(self, split: str) -> Path:
        return self.processed_dir / f"challenger_{split}.parquet"

    def calibrated_predictions_path(self) -> Path:
        return self.reports_dir / "calibrated_predictions.parquet"

    def report_path(self) -> Path:
        return self.reports_dir / "monitoring_report.json"

    def feature_drift_path(self) -> Path:
        return self.reports_dir / "feature_drift.parquet"

    def prediction_drift_path(self) -> Path:
        return self.reports_dir / "prediction_drift.parquet"

    @classmethod
    def from_yaml(cls, path: Path, project_root: Path) -> "MonitoringConfig":
        """Load config from YAML, overriding defaults."""
        if not path.exists():
            raise FileNotFoundError(f"Monitoring config not found: {path}")

        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        populations = raw.get("populations", {})
        psi = raw.get("psi", {})
        ks = raw.get("ks", {})
        cal = raw.get("calibration", {})
        perf = raw.get("performance", {})
        rep = raw.get("reporting", {})

        return cls(
            project_root=project_root,
            processed_dir=project_root / "data" / "processed",
            reports_dir=project_root / "artifacts" / "reports",
            reference_split=populations.get("reference_split", "train"),
            monitoring_split=populations.get("monitoring_split", "holdout"),
            psi_stable_max=float(psi.get("stable_max", 0.10)),
            psi_moderate_max=float(psi.get("moderate_max", 0.25)),
            ks_drift_threshold=float(ks.get("drift_threshold", 0.05)),
            calibration_slope_min=float(cal.get("slope_min", 0.90)),
            calibration_slope_max=float(cal.get("slope_max", 1.10)),
            calibration_intercept_abs_max=float(cal.get("intercept_abs_max", 0.15)),
            min_auc=float(perf.get("min_auc", 0.70)),
            max_auc_drop=float(perf.get("max_auc_drop", 0.03)),
            top_n_drifted=int(rep.get("top_n_drifted", 20)),
            top_n_features_ranked_by=rep.get("top_n_features_ranked_by", "psi"),
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    """Drift metrics for one feature."""

    feature: str
    reference_mean: float | None
    monitoring_mean: float | None
    reference_null_rate: float
    monitoring_null_rate: float
    psi: float
    ks: float
    severity: str  # stable | moderate | material

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "reference_mean": (
                round(self.reference_mean, 6)
                if self.reference_mean is not None else None
            ),
            "monitoring_mean": (
                round(self.monitoring_mean, 6)
                if self.monitoring_mean is not None else None
            ),
            "reference_null_rate": round(self.reference_null_rate, 6),
            "monitoring_null_rate": round(self.monitoring_null_rate, 6),
            "psi": round(self.psi, 6),
            "ks": round(self.ks, 6),
            "severity": self.severity,
        }


@dataclass
class PredictionDriftResult:
    """Drift metrics for the model's predictions."""

    reference_mean: float
    monitoring_mean: float
    reference_positive_rate: float  # fraction with PD > 0.5 (as a sanity check)
    monitoring_positive_rate: float
    psi: float
    ks: float
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_mean": round(self.reference_mean, 6),
            "monitoring_mean": round(self.monitoring_mean, 6),
            "reference_positive_rate": round(self.reference_positive_rate, 6),
            "monitoring_positive_rate": round(self.monitoring_positive_rate, 6),
            "psi": round(self.psi, 6),
            "ks": round(self.ks, 6),
            "severity": self.severity,
        }


@dataclass
class CalibrationDriftResult:
    """Calibration metrics on the monitoring population."""

    reference_auc: float
    monitoring_auc: float
    reference_brier: float
    monitoring_brier: float
    monitoring_slope: float
    monitoring_intercept: float
    calibration_ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_auc": round(self.reference_auc, 6),
            "monitoring_auc": round(self.monitoring_auc, 6),
            "reference_brier": round(self.reference_brier, 6),
            "monitoring_brier": round(self.monitoring_brier, 6),
            "monitoring_slope": round(self.monitoring_slope, 6),
            "monitoring_intercept": round(self.monitoring_intercept, 6),
            "calibration_ok": self.calibration_ok,
        }


@dataclass
class MonitoringReport:
    """Aggregated monitoring report."""

    generated_at: str
    reference_split: str
    monitoring_split: str
    n_reference_rows: int
    n_monitoring_rows: int
    n_features: int
    n_features_stable: int
    n_features_moderate: int
    n_features_material: int
    prediction_drift: PredictionDriftResult
    calibration_drift: CalibrationDriftResult
    top_drifted_features: list[FeatureDriftResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "reference_split": self.reference_split,
            "monitoring_split": self.monitoring_split,
            "n_reference_rows": self.n_reference_rows,
            "n_monitoring_rows": self.n_monitoring_rows,
            "n_features": self.n_features,
            "n_features_stable": self.n_features_stable,
            "n_features_moderate": self.n_features_moderate,
            "n_features_material": self.n_features_material,
            "prediction_drift": self.prediction_drift.to_dict(),
            "calibration_drift": self.calibration_drift.to_dict(),
            "top_drifted_features": [f.to_dict() for f in self.top_drifted_features],
            "duration_seconds": round(self.duration_seconds, 3),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_psi(
    reference: np.ndarray,
    monitoring: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index.

    Bins are defined on the reference distribution (quantile-based).
    Both populations are binned with the same edges, and PSI is
    computed as sum((ref_pct - mon_pct) * ln(ref_pct / mon_pct)).

    Nulls are excluded from the PSI computation but tracked separately
    by the caller (null rate comparison). This makes PSI purely about
    the non-null distribution shift.
    """
    ref = reference[~np.isnan(reference)]
    mon = monitoring[~np.isnan(monitoring)]

    if len(ref) == 0 or len(mon) == 0:
        return 0.0

    # Compute bin edges on reference using quantiles
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    # Bin both populations using the same edges
    ref_counts, _ = np.histogram(ref, bins=edges)
    mon_counts, _ = np.histogram(mon, bins=edges)

    ref_pct = ref_counts / len(ref)
    mon_pct = mon_counts / len(mon)

    # Replace zeros with a small epsilon to avoid log(0) and division by zero
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    mon_pct = np.clip(mon_pct, eps, None)

    psi = np.sum((ref_pct - mon_pct) * np.log(ref_pct / mon_pct))
    return float(psi)


def compute_ks_two_sample(
    reference: np.ndarray,
    monitoring: np.ndarray,
) -> float:
    """
    Two-sample Kolmogorov-Smirnov statistic.

    Returns the maximum separation between the empirical CDFs of the
    two samples.
    """
    ref = np.sort(reference[~np.isnan(reference)])
    mon = np.sort(monitoring[~np.isnan(monitoring)])

    if len(ref) == 0 or len(mon) == 0:
        return 0.0

    # Merge the two sorted arrays and compute CDFs at each point
    all_values = np.concatenate([ref, mon])
    all_values.sort()

    cdf_ref = np.searchsorted(ref, all_values, side="right") / len(ref)
    cdf_mon = np.searchsorted(mon, all_values, side="right") / len(mon)

    return float(np.max(np.abs(cdf_ref - cdf_mon)))


def classify_severity(psi: float, config: MonitoringConfig) -> str:
    """Classify a PSI value into stable / moderate / material."""
    if psi < config.psi_stable_max:
        return "stable"
    if psi < config.psi_moderate_max:
        return "moderate"
    return "material"


def compute_calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Regress y_true on logit(y_prob) with unregularized logistic regression."""
    from sklearn.linear_model import LogisticRegression

    eps = 1e-9
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))

    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)

    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

@dataclass
class ModelMonitor:
    """Runs drift detection for the model."""

    config: MonitoringConfig
    report: MonitoringReport | None = None

    def run(self) -> MonitoringReport:
        start = time.monotonic()

        self.config.reports_dir.mkdir(parents=True, exist_ok=True)

        # Load reference and monitoring populations
        ref_df = self._load_split(self.config.reference_split)
        mon_df = self._load_split(self.config.monitoring_split)

        logger.info(
            "Reference: %s, %d rows; Monitoring: %s, %d rows",
            self.config.reference_split, len(ref_df),
            self.config.monitoring_split, len(mon_df),
        )

        # Feature drift
        logger.info("Computing feature drift across %d features",
                    ref_df.shape[1] - 2)  # exclude key + target
        feature_drift = self._compute_feature_drift(ref_df, mon_df)
        logger.info(
            "Feature drift: %d stable, %d moderate, %d material",
            sum(1 for f in feature_drift if f.severity == "stable"),
            sum(1 for f in feature_drift if f.severity == "moderate"),
            sum(1 for f in feature_drift if f.severity == "material"),
        )

        # Save feature drift table
        feature_drift_df = pd.DataFrame([f.to_dict() for f in feature_drift])
        feature_drift_df.to_parquet(
            self.config.feature_drift_path(),
            engine="pyarrow", compression="snappy", index=False,
        )

        # Prediction drift
        logger.info("Computing prediction drift")
        prediction_drift = self._compute_prediction_drift()

        # Calibration drift
        logger.info("Computing calibration drift")
        calibration_drift = self._compute_calibration_drift()

        # Top drifted features (ranked by PSI or KS)
        top_n = self._top_drifted(feature_drift)

        # Notes
        notes = self._build_notes(
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            calibration_drift=calibration_drift,
        )

        duration = time.monotonic() - start
        self.report = MonitoringReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            reference_split=self.config.reference_split,
            monitoring_split=self.config.monitoring_split,
            n_reference_rows=len(ref_df),
            n_monitoring_rows=len(mon_df),
            n_features=ref_df.shape[1] - 2,
            n_features_stable=sum(
                1 for f in feature_drift if f.severity == "stable"
            ),
            n_features_moderate=sum(
                1 for f in feature_drift if f.severity == "moderate"
            ),
            n_features_material=sum(
                1 for f in feature_drift if f.severity == "material"
            ),
            prediction_drift=prediction_drift,
            calibration_drift=calibration_drift,
            top_drifted_features=top_n,
            duration_seconds=duration,
            notes=notes,
        )
        self._write_report(self.report)
        self._log_summary(self.report)
        return self.report

    # ---- data loading ----------------------------------------------------

    def _load_split(self, split: str) -> pd.DataFrame:
        path = self.config.challenger_split_path(split)
        if not path.exists():
            raise FileNotFoundError(
                f"Split not found: {path}. Run store first."
            )
        df = pd.read_parquet(path, engine="pyarrow")
        return df

    # ---- feature drift ---------------------------------------------------

    def _compute_feature_drift(
        self,
        ref_df: pd.DataFrame,
        mon_df: pd.DataFrame,
    ) -> list[FeatureDriftResult]:
        """Compute PSI and KS for every feature."""
        exclude = {self.config.key_column, self.config.target_column}
        features = [c for c in ref_df.columns if c not in exclude]

        results: list[FeatureDriftResult] = []
        for i, feature in enumerate(features, 1):
            if feature not in mon_df.columns:
                continue
            ref_vals = ref_df[feature].to_numpy()
            mon_vals = mon_df[feature].to_numpy()

            # PSI and KS on non-null values
            psi = compute_psi(ref_vals, mon_vals)
            ks = compute_ks_two_sample(ref_vals, mon_vals)

            # Null rates
            ref_null = float(np.isnan(ref_vals).mean()) if np.issubdtype(ref_vals.dtype, np.number) else float(ref_df[feature].isna().mean())
            mon_null = float(np.isnan(mon_vals).mean()) if np.issubdtype(mon_vals.dtype, np.number) else float(mon_df[feature].isna().mean())

            # Means (numeric only)
            try:
                ref_mean = float(np.nanmean(ref_vals.astype(float)))
                mon_mean = float(np.nanmean(mon_vals.astype(float)))
            except (TypeError, ValueError):
                ref_mean = None
                mon_mean = None

            severity = classify_severity(psi, self.config)

            results.append(FeatureDriftResult(
                feature=feature,
                reference_mean=ref_mean,
                monitoring_mean=mon_mean,
                reference_null_rate=ref_null,
                monitoring_null_rate=mon_null,
                psi=psi,
                ks=ks,
                severity=severity,
            ))

            if i % 50 == 0:
                logger.debug("  feature drift progress: %d/%d", i, len(features))

        return results

    # ---- prediction drift -----------------------------------------------

    def _compute_prediction_drift(self) -> PredictionDriftResult:
        """PSI and KS on calibrated PD between reference and monitoring splits."""
        preds_path = self.config.calibrated_predictions_path()
        if not preds_path.exists():
            raise FileNotFoundError(f"Predictions not found: {preds_path}")

        preds = pd.read_parquet(preds_path, engine="pyarrow")
        preds = preds[preds["model"] == "challenger"]

        ref_pd = preds[
            preds["split"] == self.config.reference_split
        ]["calibrated_pd"].to_numpy()
        mon_pd = preds[
            preds["split"] == self.config.monitoring_split
        ]["calibrated_pd"].to_numpy()

        psi = compute_psi(ref_pd, mon_pd)
        ks = compute_ks_two_sample(ref_pd, mon_pd)
        severity = classify_severity(psi, self.config)

        return PredictionDriftResult(
            reference_mean=float(ref_pd.mean()),
            monitoring_mean=float(mon_pd.mean()),
            reference_positive_rate=float((ref_pd > 0.5).mean()),
            monitoring_positive_rate=float((mon_pd > 0.5).mean()),
            psi=psi,
            ks=ks,
            severity=severity,
        )

    # ---- calibration drift ----------------------------------------------

    def _compute_calibration_drift(self) -> CalibrationDriftResult:
        """Compute calibration metrics on both reference and monitoring."""
        preds = pd.read_parquet(
            self.config.calibrated_predictions_path(), engine="pyarrow"
        )
        preds = preds[preds["model"] == "challenger"]

        ref_pd = preds[
            preds["split"] == self.config.reference_split
        ]["calibrated_pd"].to_numpy()
        mon_pd = preds[
            preds["split"] == self.config.monitoring_split
        ]["calibrated_pd"].to_numpy()

        ref_y = self._load_split(self.config.reference_split)[
            self.config.target_column
        ].to_numpy()
        mon_y = self._load_split(self.config.monitoring_split)[
            self.config.target_column
        ].to_numpy()

        ref_auc = float(roc_auc_score(ref_y, ref_pd))
        mon_auc = float(roc_auc_score(mon_y, mon_pd))
        ref_brier = float(brier_score_loss(ref_y, ref_pd))
        mon_brier = float(brier_score_loss(mon_y, mon_pd))

        mon_slope, mon_intercept = compute_calibration_slope_intercept(
            mon_y, mon_pd
        )

        calibration_ok = (
            self.config.calibration_slope_min <= mon_slope <= self.config.calibration_slope_max
            and abs(mon_intercept) <= self.config.calibration_intercept_abs_max
        )

        return CalibrationDriftResult(
            reference_auc=ref_auc,
            monitoring_auc=mon_auc,
            reference_brier=ref_brier,
            monitoring_brier=mon_brier,
            monitoring_slope=mon_slope,
            monitoring_intercept=mon_intercept,
            calibration_ok=calibration_ok,
        )

    # ---- helpers ---------------------------------------------------------

    def _top_drifted(
        self, feature_drift: list[FeatureDriftResult]
    ) -> list[FeatureDriftResult]:
        """Return top-N features ranked by configured metric."""
        key = (
            lambda f: -f.psi
            if self.config.top_n_features_ranked_by == "psi"
            else -f.ks
        )
        sorted_features = sorted(feature_drift, key=key)
        return sorted_features[: self.config.top_n_drifted]

    def _build_notes(
        self,
        feature_drift: list[FeatureDriftResult],
        prediction_drift: PredictionDriftResult,
        calibration_drift: CalibrationDriftResult,
    ) -> list[str]:
        notes: list[str] = []

        n_material = sum(1 for f in feature_drift if f.severity == "material")
        n_moderate = sum(1 for f in feature_drift if f.severity == "moderate")

        if n_material > 0:
            notes.append(
                f"{n_material} features show material drift (PSI >= "
                f"{self.config.psi_moderate_max}). Investigate before "
                "continuing to use the model."
            )
        if n_moderate > 0:
            notes.append(
                f"{n_moderate} features show moderate drift "
                f"({self.config.psi_stable_max} <= PSI < "
                f"{self.config.psi_moderate_max}). Monitor closely."
            )

        if prediction_drift.severity == "material":
            notes.append(
                f"Prediction distribution has drifted materially "
                f"(PSI = {prediction_drift.psi:.4f}). The population "
                "the model is scoring has changed."
            )

        if not calibration_drift.calibration_ok:
            notes.append(
                f"Calibration on the monitoring split is out of tolerance: "
                f"slope = {calibration_drift.monitoring_slope:.4f}, "
                f"intercept = {calibration_drift.monitoring_intercept:.4f}. "
                "Recalibration may be required."
            )

        auc_drop = calibration_drift.reference_auc - calibration_drift.monitoring_auc
        if auc_drop > self.config.max_auc_drop:
            notes.append(
                f"AUC dropped by {auc_drop:.4f} between reference and "
                f"monitoring (from {calibration_drift.reference_auc:.4f} to "
                f"{calibration_drift.monitoring_auc:.4f})."
            )

        if not notes:
            notes.append(
                "All monitored features are stable and calibration is "
                "within tolerance."
            )

        return notes

    # ---- persistence -----------------------------------------------------

    def _write_report(self, report: MonitoringReport) -> None:
        with self.config.report_path().open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Monitoring report written: %s",
                    self.config.report_path().name)

    def _log_summary(self, report: MonitoringReport) -> None:
        logger.info("=" * 70)
        logger.info("Monitoring Report")
        logger.info("  reference: %s (%d rows)",
                    report.reference_split, report.n_reference_rows)
        logger.info("  monitoring: %s (%d rows)",
                    report.monitoring_split, report.n_monitoring_rows)
        logger.info("  ---")
        logger.info("  Feature drift (%d features):", report.n_features)
        logger.info("    stable:   %d", report.n_features_stable)
        logger.info("    moderate: %d", report.n_features_moderate)
        logger.info("    material: %d", report.n_features_material)
        logger.info("  ---")
        logger.info(
            "  Prediction drift: PSI=%.4f  KS=%.4f  severity=%s",
            report.prediction_drift.psi,
            report.prediction_drift.ks,
            report.prediction_drift.severity,
        )
        logger.info(
            "  Calibration drift: AUC %.4f -> %.4f, Brier %.6f -> %.6f, "
            "slope=%.4f, intercept=%.4f, ok=%s",
            report.calibration_drift.reference_auc,
            report.calibration_drift.monitoring_auc,
            report.calibration_drift.reference_brier,
            report.calibration_drift.monitoring_brier,
            report.calibration_drift.monitoring_slope,
            report.calibration_drift.monitoring_intercept,
            report.calibration_drift.calibration_ok,
        )
        logger.info("  ---")
        logger.info("  Top 10 drifted features:")
        for f in report.top_drifted_features[:10]:
            logger.info(
                "    %-40s  PSI=%.4f  KS=%.4f  %s",
                f.feature, f.psi, f.ks, f.severity,
            )
        if report.notes:
            logger.info("  ---")
            for note in report.notes:
                logger.info("  Note: %s", note)
        logger.info("  duration: %.1fs", report.duration_seconds)
        logger.info("=" * 70)