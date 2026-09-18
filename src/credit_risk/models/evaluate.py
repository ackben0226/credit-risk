"""
Champion/challenger comparison report.

Produces the artifact that drives the model promotion decision. Given
two trained, calibrated models, this module:

    1. Cross-checks the training-time metrics against recomputed values
       from the saved predictions. Catches drift between components.
    2. Compares discrimination (AUC, Gini, KS) on holdout.
    3. Compares calibration (Brier, slope, intercept) on holdout.
    4. Compares cost-sensitive performance across a range of cost ratios.
    5. Compares feature importance top-20.
    6. Produces a production recommendation with override conditions.

The recommendation is not a simple "challenger wins". It reports the
tradeoff and documents when each model is preferred.

Design notes
------------
- Cost-sensitive comparison: for each model and each hypothetical cost
  ratio (C_FN / C_FP), sweep thresholds to find the one minimizing
  expected cost on holdout. Compare the resulting costs.
- C_FN = cost of approving a defaulter (false negative in risk terms).
- C_FP = cost of rejecting a good applicant.
- Expected cost = C_FN * P(approve & default) + C_FP * P(reject & good).
- Prediction on holdout is used for the cost analysis; the champion
  and challenger are compared on the same rows.

Outputs
-------
- artifacts/reports/model_comparison.json
- docs/champion_challenger.md
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
from sklearn.metrics import brier_score_loss, roc_auc_score


logger = logging.getLogger("credit_risk.models.evaluate")


# ---------------------------------------------------------------------------
# Cost ratios to sweep
# ---------------------------------------------------------------------------
# C_FN / C_FP: how many times more expensive is approving a defaulter
# compared to rejecting a good applicant. Typical retail lending values
# range from 5 (consumer credit) to 50 (unsecured micro-lending).
#
COST_RATIOS: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0, 50.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationConfig:
    """Resolved configuration for the comparison report."""

    project_root: Path
    processed_dir: Path
    reports_dir: Path
    docs_dir: Path

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    calibrated_predictions: str = "calibrated_predictions.parquet"
    champion_metrics: str = "champion_metrics.json"
    challenger_metrics: str = "challenger_metrics.json"
    calibration_metrics: str = "calibration_metrics.json"
    challenger_importance: str = "challenger_feature_importance.csv"

    comparison_output: str = "model_comparison.json"
    report_output: str = "champion_challenger.md"

    challenger_split_prefix: str = "challenger_"

    cost_ratios: tuple[float, ...] = COST_RATIOS

    cross_check_tolerance: float = 0.001  # 0.1% tolerance on metric agreement

    def calibrated_predictions_path(self) -> Path:
        return self.reports_dir / self.calibrated_predictions

    def champion_metrics_path(self) -> Path:
        return self.reports_dir / self.champion_metrics

    def challenger_metrics_path(self) -> Path:
        return self.reports_dir / self.challenger_metrics

    def calibration_metrics_path(self) -> Path:
        return self.reports_dir / self.calibration_metrics

    def challenger_importance_path(self) -> Path:
        return self.reports_dir / self.challenger_importance

    def challenger_split_path(self, split: str) -> Path:
        return self.processed_dir / f"{self.challenger_split_prefix}{split}.parquet"

    def comparison_output_path(self) -> Path:
        return self.reports_dir / self.comparison_output

    def report_output_path(self) -> Path:
        return self.docs_dir / self.report_output


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Discrimination and calibration metrics for one model on holdout."""

    model: str
    auc: float
    gini: float
    ks: float
    brier: float
    calibration_slope: float
    calibration_intercept: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "auc": round(self.auc, 6),
            "gini": round(self.gini, 6),
            "ks": round(self.ks, 6),
            "brier": round(self.brier, 6),
            "calibration_slope": round(self.calibration_slope, 6),
            "calibration_intercept": round(self.calibration_intercept, 6),
        }


@dataclass
class CostSweepResult:
    """Expected-cost comparison at one cost ratio."""

    cost_ratio: float
    champion_threshold: float
    challenger_threshold: float
    champion_expected_cost: float
    challenger_expected_cost: float
    challenger_cost_savings: float
    challenger_cost_savings_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_ratio": self.cost_ratio,
            "champion_threshold": round(self.champion_threshold, 6),
            "challenger_threshold": round(self.challenger_threshold, 6),
            "champion_expected_cost": round(self.champion_expected_cost, 6),
            "challenger_expected_cost": round(self.challenger_expected_cost, 6),
            "challenger_cost_savings": round(self.challenger_cost_savings, 6),
            "challenger_cost_savings_pct": round(self.challenger_cost_savings_pct, 4),
        }


@dataclass
class CrossCheckResult:
    """Whether training-time metrics agree with recomputed values."""

    check_name: str
    stored_value: float
    recomputed_value: float
    delta: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check_name,
            "stored": round(self.stored_value, 6),
            "recomputed": round(self.recomputed_value, 6),
            "delta": round(self.delta, 6),
            "passed": self.passed,
        }


@dataclass
class ComparisonReport:
    """Aggregated champion/challenger comparison."""

    generated_at: str
    champion_metrics: ModelMetrics
    challenger_metrics: ModelMetrics
    cost_sweep: list[CostSweepResult] = field(default_factory=list)
    cross_checks: list[CrossCheckResult] = field(default_factory=list)
    champion_top_features: list[dict[str, Any]] = field(default_factory=list)
    challenger_top_features: list[dict[str, Any]] = field(default_factory=list)
    statistical_recommendation: str = ""
    operational_recommendation: str = ""
    override_conditions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "champion_metrics": self.champion_metrics.to_dict(),
            "challenger_metrics": self.challenger_metrics.to_dict(),
            "cost_sweep": [c.to_dict() for c in self.cost_sweep],
            "cross_checks": [c.to_dict() for c in self.cross_checks],
            "champion_top_features": self.champion_top_features,
            "challenger_top_features": self.challenger_top_features,
            "recommendation": {
                "statistical": self.statistical_recommendation,
                "operational": self.operational_recommendation,
                "override_conditions": self.override_conditions,
            },
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Metric helpers
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

    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)

    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def optimal_threshold_for_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> tuple[float, float]:
    """
    Find the threshold that minimizes expected cost.

    Sweeps a grid of candidate thresholds; for each, computes the
    expected cost of the resulting decisions and selects the minimum.

    Returns (threshold, expected_cost) at the optimal point.
    """
    candidates = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.5
    best_cost = float("inf")

    n = len(y_true)

    for t in candidates:
        approve = y_prob < t       # approve if PD below threshold
        decline = ~approve

        # Approved defaulters → cost_fn per default
        n_fn = int((approve & (y_true == 1)).sum())
        # Declined good applicants → cost_fp per good applicant
        n_fp = int((decline & (y_true == 0)).sum())

        cost = (cost_fn * n_fn + cost_fp * n_fp) / n

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)

    return best_threshold, best_cost


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

@dataclass
class ModelEvaluator:
    """Computes the champion/challenger comparison report."""

    config: EvaluationConfig
    report: ComparisonReport | None = None

    # ---- public API ------------------------------------------------------

    def run(self) -> ComparisonReport:
        start = time.monotonic()

        self.config.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config.docs_dir.mkdir(parents=True, exist_ok=True)

        # Load predictions and targets
        predictions, targets = self._load_predictions_and_targets()

        # Compute metrics on holdout
        champion_metrics = self._compute_model_metrics(
            predictions, targets, "champion"
        )
        challenger_metrics = self._compute_model_metrics(
            predictions, targets, "challenger"
        )

        logger.info(
            "Holdout AUC: champion=%.4f, challenger=%.4f",
            champion_metrics.auc, challenger_metrics.auc,
        )

        # Cross-check
        cross_checks = self._run_cross_checks(
            champion_metrics, challenger_metrics
        )

        # Cost sweep
        cost_sweep = self._cost_sweep(predictions, targets)
        for c in cost_sweep:
            logger.info(
                "  C_FN/C_FP=%.0f: champion cost=%.6f, challenger cost=%.6f, "
                "challenger savings=%.4f pp",
                c.cost_ratio, c.champion_expected_cost,
                c.challenger_expected_cost, c.challenger_cost_savings_pct,
            )

        # Feature importance
        champion_top, challenger_top = self._load_feature_importance()

        # Recommendation
        stat_rec, op_rec, overrides, notes = self._build_recommendation(
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            cost_sweep=cost_sweep,
            cross_checks=cross_checks,
        )

        duration = time.monotonic() - start
        self.report = ComparisonReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            cost_sweep=cost_sweep,
            cross_checks=cross_checks,
            champion_top_features=champion_top,
            challenger_top_features=challenger_top,
            statistical_recommendation=stat_rec,
            operational_recommendation=op_rec,
            override_conditions=overrides,
            notes=notes + [f"Computation time: {duration:.1f}s"],
        )

        self._write_json(self.report)
        self._write_markdown(self.report)
        self._log_summary(self.report)
        return self.report

    # ---- input loading ---------------------------------------------------

    def _load_predictions_and_targets(
        self,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Load calibrated predictions and holdout targets."""
        path = self.config.calibrated_predictions_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Predictions not found: {path}. Run calibration first."
            )

        preds = pd.read_parquet(path, engine="pyarrow")
        logger.info("  loaded predictions: %d rows", len(preds))

        # Load targets for all splits; we'll use holdout for comparison
        targets: dict[str, np.ndarray] = {}
        for split in ["train", "val", "holdout"]:
            split_path = self.config.challenger_split_path(split)
            df = pd.read_parquet(
                split_path, engine="pyarrow",
                columns=[self.config.target_column],
            )
            targets[split] = df[self.config.target_column].to_numpy()

        return preds, targets

    def _load_feature_importance(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Load feature importance for both models.

        The champion's importance comes from its coefficients; the
        challenger's from the saved feature_importance CSV.
        """
        champion_top: list[dict[str, Any]] = []
        challenger_top: list[dict[str, Any]] = []

        # Challenger
        path = self.config.challenger_importance_path()
        if path.exists():
            df = pd.read_csv(path)
            df = df.sort_values("gain", ascending=False).head(20)
            challenger_top = df.to_dict(orient="records")

        # Champion: read from metadata and the model pickle
        # We can produce coefficient-based importance for the champion
        champion_top = self._champion_feature_importance()

        return champion_top, challenger_top

    def _champion_feature_importance(self) -> list[dict[str, Any]]:
        """
        Load champion coefficients and rank by absolute value.

        Champions with WoE-encoded features have interpretable
        coefficients: larger absolute value → larger influence.
        """
        import pickle

        model_path = (
            self.config.project_root
            / "artifacts" / "models" / "champion" / "0.1.0" / "model.pkl"
        )
        if not model_path.exists():
            logger.warning("Champion model not found at %s", model_path)
            return []

        with model_path.open("rb") as f:
            model = pickle.load(f)

        # Feature names come from the champion training data
        train_path = self.config.project_root / "data" / "processed" / "champion_train.parquet"
        if not train_path.exists():
            return []

        df = pd.read_parquet(train_path, engine="pyarrow")
        feature_cols = [
            c for c in df.columns
            if c not in (self.config.key_column, self.config.target_column)
        ]

        coefs = model.coef_[0]
        if len(coefs) != len(feature_cols):
            logger.warning(
                "Coefficient count (%d) does not match feature count (%d)",
                len(coefs), len(feature_cols),
            )
            return []

        rows = sorted(
            [
                {"feature": f, "coefficient": float(c),
                 "abs_coefficient": float(abs(c))}
                for f, c in zip(feature_cols, coefs)
            ],
            key=lambda x: -x["abs_coefficient"],
        )[:20]

        return rows

    # ---- metrics ---------------------------------------------------------

    def _compute_model_metrics(
        self,
        predictions: pd.DataFrame,
        targets: dict[str, np.ndarray],
        model: str,
    ) -> ModelMetrics:
        """Compute holdout metrics for one model."""
        mask = (predictions["model"] == model) & (predictions["split"] == "holdout")
        subset = predictions[mask]
        y_prob = subset["calibrated_pd"].to_numpy()
        y_true = targets["holdout"]

        auc = compute_auc(y_true, y_prob)
        ks = compute_ks(y_true, y_prob)
        brier = compute_brier(y_true, y_prob)
        slope, intercept = compute_calibration_slope_intercept(y_true, y_prob)

        return ModelMetrics(
            model=model,
            auc=auc,
            gini=2 * auc - 1,
            ks=ks,
            brier=brier,
            calibration_slope=slope,
            calibration_intercept=intercept,
        )

    # ---- cross-check -----------------------------------------------------

    def _run_cross_checks(
        self,
        champion_metrics: ModelMetrics,
        challenger_metrics: ModelMetrics,
    ) -> list[CrossCheckResult]:
        """
        Compare recomputed metrics against the values stored by the
        trainers. If they diverge beyond tolerance, flag it.
        """
        results: list[CrossCheckResult] = []
        tol = self.config.cross_check_tolerance

        # Champion
        champ_path = self.config.champion_metrics_path()
        if champ_path.exists():
            with champ_path.open() as f:
                stored = json.load(f)
            holdout = next(
                (s for s in stored["split_metrics"] if s["split"] == "holdout"),
                None,
            )
            if holdout:
                results.append(self._make_check(
                    "champion_auc",
                    holdout["auc"], champion_metrics.auc, tol,
                ))
                results.append(self._make_check(
                    "champion_brier",
                    holdout["brier"], champion_metrics.brier, tol,
                ))

        # Challenger (raw metrics, not calibrated)
        chall_path = self.config.challenger_metrics_path()
        if chall_path.exists():
            with chall_path.open() as f:
                stored = json.load(f)
            holdout = next(
                (s for s in stored["split_metrics"] if s["split"] == "holdout"),
                None,
            )
            if holdout:
                # Note: challenge metrics are raw, not calibrated. AUC should
                # be unchanged by calibration. Brier will differ.
                results.append(self._make_check(
                    "challenger_auc",
                    holdout["auc"], challenger_metrics.auc, tol,
                ))

        return results

    @staticmethod
    def _make_check(
        name: str,
        stored: float,
        recomputed: float,
        tol: float,
    ) -> CrossCheckResult:
        delta = abs(stored - recomputed)
        return CrossCheckResult(
            check_name=name,
            stored_value=stored,
            recomputed_value=recomputed,
            delta=delta,
            passed=delta <= tol,
        )

    # ---- cost sweep ------------------------------------------------------

    def _cost_sweep(
        self,
        predictions: pd.DataFrame,
        targets: dict[str, np.ndarray],
    ) -> list[CostSweepResult]:
        """Compute expected cost for both models across the cost ratio grid."""
        holdout = predictions[predictions["split"] == "holdout"]
        y_true = targets["holdout"]

        champion_pd = holdout[
            holdout["model"] == "champion"
        ]["calibrated_pd"].to_numpy()
        challenger_pd = holdout[
            holdout["model"] == "challenger"
        ]["calibrated_pd"].to_numpy()

        results: list[CostSweepResult] = []

        for ratio in self.config.cost_ratios:
            cost_fn = ratio
            cost_fp = 1.0

            champ_thresh, champ_cost = optimal_threshold_for_cost(
                y_true, champion_pd, cost_fn, cost_fp
            )
            chall_thresh, chall_cost = optimal_threshold_for_cost(
                y_true, challenger_pd, cost_fn, cost_fp
            )

            savings = champ_cost - chall_cost
            savings_pct = (savings / champ_cost * 100) if champ_cost > 0 else 0.0

            results.append(CostSweepResult(
                cost_ratio=ratio,
                champion_threshold=champ_thresh,
                challenger_threshold=chall_thresh,
                champion_expected_cost=champ_cost,
                challenger_expected_cost=chall_cost,
                challenger_cost_savings=savings,
                challenger_cost_savings_pct=savings_pct,
            ))

        return results

    # ---- recommendation --------------------------------------------------

    def _build_recommendation(
        self,
        champion_metrics: ModelMetrics,
        challenger_metrics: ModelMetrics,
        cost_sweep: list[CostSweepResult],
        cross_checks: list[CrossCheckResult],
    ) -> tuple[str, str, list[str], list[str]]:
        """Produce statistical + operational recommendation and overrides."""
        notes: list[str] = []

        # Check cross-checks
        failed = [c for c in cross_checks if not c.passed]
        if failed:
            notes.append(
                f"{len(failed)} cross-checks failed. Metrics may have drifted."
            )

        # Discrimination
        auc_delta = challenger_metrics.auc - champion_metrics.auc
        gini_delta = challenger_metrics.gini - champion_metrics.gini
        brier_delta = champion_metrics.brier - challenger_metrics.brier

        # Calibration tolerance
        chall_cal_ok = (
            abs(challenger_metrics.calibration_slope - 1.0) <= 0.10
            and abs(challenger_metrics.calibration_intercept) <= 0.10
        )
        champ_cal_ok = (
            abs(champion_metrics.calibration_slope - 1.0) <= 0.10
            and abs(champion_metrics.calibration_intercept) <= 0.10
        )

        # Cost advantage (take mid-range ratio for the summary)
        mid_cost = next(
            (c for c in cost_sweep if c.cost_ratio == 20.0),
            cost_sweep[len(cost_sweep) // 2] if cost_sweep else None,
        )

        # Statistical recommendation
        if auc_delta > 0.005 and chall_cal_ok and champ_cal_ok:
            statistical_rec = (
                f"Challenger is statistically superior: "
                f"+{auc_delta*100:.2f} pp AUC, "
                f"+{gini_delta*100:.2f} pp Gini, "
                f"-{brier_delta:.4f} Brier (lower is better). "
                "Both models are within calibration tolerance."
            )
        elif auc_delta > 0.005 and not chall_cal_ok:
            statistical_rec = (
                f"Challenger wins on discrimination (+{auc_delta*100:.2f} pp AUC) "
                "but is not within calibration tolerance. "
                "Recalibration required before promotion."
            )
        elif abs(auc_delta) <= 0.005:
            statistical_rec = (
                "Discrimination is statistically equivalent. "
                "Prefer the simpler model (champion)."
            )
        else:
            statistical_rec = (
                f"Champion outperforms challenger on discrimination. "
                "Recommend keeping champion in production."
            )

        # Operational recommendation
        if "superior" in statistical_rec and chall_cal_ok:
            if mid_cost and mid_cost.challenger_cost_savings_pct > 0.5:
                operational_rec = (
                    f"Challenger is the primary production model. "
                    f"At C_FN/C_FP={mid_cost.cost_ratio:.0f}, "
                    f"challenger expected cost is "
                    f"{mid_cost.challenger_cost_savings_pct:.2f}% lower. "
                    "The champion remains available for regulatory "
                    "demonstration and linear-scorecard-required contexts."
                )
            else:
                operational_rec = (
                    "Challenger is the primary production model. "
                    "Cost advantage over champion is marginal at the "
                    "assumed cost ratios; verify against actual business "
                    "cost parameters before final promotion."
                )
        else:
            operational_rec = (
                "Champion remains the primary production model. "
                "Challenger stays in shadow mode for continued monitoring."
            )

        # Override conditions
        overrides = [
            "Regulatory context requires a linear, points-based scorecard: "
            "use the champion.",
            "Explainability to a non-technical audience is the primary "
            "requirement: use the champion.",
            "Inference latency must be under 5 ms p99: use the champion "
            "(linear scoring is faster than tree traversal).",
            "Maximum discrimination is the priority and cost tolerance "
            "permits: use the challenger with calibrated PD.",
            "Subgroup fairness analysis reveals disparate impact in the "
            "challenger but not the champion: prefer the champion until "
            "the fairness issue is resolved.",
        ]

        return statistical_rec, operational_rec, overrides, notes

    # ---- output writing --------------------------------------------------

    def _write_json(self, report: ComparisonReport) -> None:
        path = self.config.comparison_output_path()
        with path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info("Comparison written: %s", path.name)

    def _write_markdown(self, report: ComparisonReport) -> None:
        path = self.config.report_output_path()
        md = self._render_markdown(report)
        with path.open("w", encoding="utf-8") as f:
            f.write(md)
        logger.info("Markdown report written: %s", path.name)

    def _render_markdown(self, report: ComparisonReport) -> str:
        lines: list[str] = []

        lines.append("# Champion / Challenger Comparison")
        lines.append("")
        lines.append(f"*Generated: {report.generated_at}*")
        lines.append("")
        lines.append("---")
        lines.append("")

        # --- Holdout metrics side-by-side ---
        lines.append("## Holdout Metrics")
        lines.append("")
        lines.append("| Metric | Champion | Challenger | Δ (Ch - Cg) |")
        lines.append("|---|---:|---:|---:|")
        lines.append(self._metric_row(
            "AUC",
            report.champion_metrics.auc,
            report.challenger_metrics.auc,
        ))
        lines.append(self._metric_row(
            "Gini",
            report.champion_metrics.gini,
            report.challenger_metrics.gini,
        ))
        lines.append(self._metric_row(
            "KS",
            report.champion_metrics.ks,
            report.challenger_metrics.ks,
        ))
        lines.append(self._metric_row(
            "Brier (lower better)",
            report.champion_metrics.brier,
            report.challenger_metrics.brier,
        ))
        lines.append(self._metric_row(
            "Calibration slope (target 1.0)",
            report.champion_metrics.calibration_slope,
            report.challenger_metrics.calibration_slope,
        ))
        lines.append(self._metric_row(
            "Calibration intercept (target 0.0)",
            report.champion_metrics.calibration_intercept,
            report.challenger_metrics.calibration_intercept,
        ))
        lines.append("")

        # --- Cross-checks ---
        lines.append("## Cross-checks")
        lines.append("")
        lines.append("Recomputed metrics vs stored metrics. Tolerance 0.1%.")
        lines.append("")
        lines.append("| Check | Stored | Recomputed | Delta | Status |")
        lines.append("|---|---:|---:|---:|:---:|")
        for c in report.cross_checks:
            status = "✅" if c.passed else "❌"
            lines.append(
                f"| {c.check_name} | {c.stored_value:.6f} | "
                f"{c.recomputed_value:.6f} | {c.delta:.6f} | {status} |"
            )
        lines.append("")

        # --- Cost sweep ---
        lines.append("## Cost-Sensitive Comparison")
        lines.append("")
        lines.append(
            "Expected cost per applicant on holdout, at the optimal "
            "threshold for each model, across a range of C_FN/C_FP ratios."
        )
        lines.append("")
        lines.append(
            "| C_FN/C_FP | Champion thresh | Challenger thresh | "
            "Champion cost | Challenger cost | Savings |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for c in report.cost_sweep:
            lines.append(
                f"| {c.cost_ratio:.0f} | "
                f"{c.champion_threshold:.4f} | "
                f"{c.challenger_threshold:.4f} | "
                f"{c.champion_expected_cost:.6f} | "
                f"{c.challenger_expected_cost:.6f} | "
                f"{c.challenger_cost_savings_pct:.3f}% |"
            )
        lines.append("")

        # --- Top features ---
        lines.append("## Top 10 Features")
        lines.append("")
        lines.append("### Champion (WoE coefficient magnitude)")
        lines.append("")
        lines.append("| Rank | Feature | Coefficient |")
        lines.append("|---:|---|---:|")
        for i, f in enumerate(report.champion_top_features[:10], 1):
            lines.append(
                f"| {i} | `{f['feature']}` | {f['coefficient']:.4f} |"
            )
        lines.append("")
        lines.append("### Challenger (LightGBM gain)")
        lines.append("")
        lines.append("| Rank | Feature | Gain |")
        lines.append("|---:|---|---:|")
        for i, f in enumerate(report.challenger_top_features[:10], 1):
            lines.append(
                f"| {i} | `{f['feature']}` | {f['gain']:.1f} |"
            )
        lines.append("")

        # --- Recommendation ---
        lines.append("## Recommendation")
        lines.append("")
        lines.append("### Statistical")
        lines.append("")
        lines.append(report.statistical_recommendation)
        lines.append("")
        lines.append("### Operational")
        lines.append("")
        lines.append(report.operational_recommendation)
        lines.append("")
        lines.append("### Override Conditions")
        lines.append("")
        lines.append(
            "The operational recommendation above assumes no overriding "
            "constraint. The following conditions supersede it:"
        )
        lines.append("")
        for o in report.override_conditions:
            lines.append(f"- {o}")
        lines.append("")

        # --- Notes ---
        if report.notes:
            lines.append("## Notes")
            lines.append("")
            for note in report.notes:
                lines.append(f"- {note}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(
            "*End of comparison. This document is a companion to the "
            "Model Card and the Problem Statement.*"
        )
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _metric_row(label: str, champion_val: float, challenger_val: float) -> str:
        delta = challenger_val - champion_val
        return f"| {label} | {champion_val:.6f} | {challenger_val:.6f} | {delta:+.6f} |"

    # ---- summary ---------------------------------------------------------

    def _log_summary(self, report: ComparisonReport) -> None:
        logger.info("=" * 70)
        logger.info("Champion/Challenger Comparison")
        logger.info("  ---")
        logger.info("  Holdout metrics:")
        logger.info(
            "    %-10s  AUC=%.4f  Gini=%.4f  KS=%.4f  Brier=%.6f",
            "champion",
            report.champion_metrics.auc,
            report.champion_metrics.gini,
            report.champion_metrics.ks,
            report.champion_metrics.brier,
        )
        logger.info(
            "    %-10s  AUC=%.4f  Gini=%.4f  KS=%.4f  Brier=%.6f",
            "challenger",
            report.challenger_metrics.auc,
            report.challenger_metrics.gini,
            report.challenger_metrics.ks,
            report.challenger_metrics.brier,
        )
        logger.info("  ---")
        logger.info("  Cross-checks: %d passed, %d failed",
                    sum(1 for c in report.cross_checks if c.passed),
                    sum(1 for c in report.cross_checks if not c.passed))
        logger.info("  ---")
        logger.info("  Statistical recommendation:")
        logger.info("    %s", report.statistical_recommendation)
        logger.info("  Operational recommendation:")
        logger.info("    %s", report.operational_recommendation)
        logger.info("=" * 70)