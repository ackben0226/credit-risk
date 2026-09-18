"""
SHAP-based explainability for the challenger (LightGBM).

Computes global and local SHAP values for the challenger model. Produces:

- Global feature importance (mean |SHAP| top-20)
- Per-applicant feature contributions for a sample
- Contribution matrix for reason code generation

The champion (logistic regression) is explained via coefficients
directly — no SHAP needed. See `linear_contributions` for that path.
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


logger = logging.getLogger("credit_risk.explainability.shap_analysis")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapConfig:
    """Configuration for SHAP analysis."""

    project_root: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path

    key_column: str = "SK_ID_CURR"
    target_column: str = "TARGET"

    challenger_split: str = "challenger_holdout.parquet"
    challenger_model_dir: str = "challenger/0.1.0"

    sample_size: int = 5000       # number of applicants for local SHAP
    top_k: int = 20               # number of top features to emit

    def challenger_split_path(self) -> Path:
        return self.processed_dir / self.challenger_split

    def challenger_model_path(self) -> Path:
        return self.models_dir / self.challenger_model_dir / "model.txt"

    def challenger_metadata_path(self) -> Path:
        return self.models_dir / self.challenger_model_dir / "metadata.json"

    def global_shap_path(self) -> Path:
        return self.reports_dir / "challenger_shap_global.json"

    def local_shap_path(self) -> Path:
        return self.reports_dir / "challenger_shap_local.parquet"

    def contribution_matrix_path(self) -> Path:
        return self.reports_dir / "challenger_contributions.parquet"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class GlobalFeatureImportance:
    """One feature's global importance."""

    rank: int
    feature: str
    mean_abs_shap: float
    mean_shap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "feature": self.feature,
            "mean_abs_shap": round(self.mean_abs_shap, 6),
            "mean_shap": round(self.mean_shap, 6),
        }


@dataclass
class ShapReport:
    """Aggregated SHAP analysis output."""

    generated_at: str
    model: str
    model_version: str
    n_samples_global: int
    n_samples_local: int
    n_features: int
    top_features: list[GlobalFeatureImportance] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "model": self.model,
            "model_version": self.model_version,
            "n_samples_global": self.n_samples_global,
            "n_samples_local": self.n_samples_local,
            "n_features": self.n_features,
            "top_features": [f.to_dict() for f in self.top_features],
            "duration_seconds": round(self.duration_seconds, 3),
        }


# ---------------------------------------------------------------------------
# SHAP analyzer
# ---------------------------------------------------------------------------

@dataclass
class ShapAnalyzer:
    """Computes SHAP explanations for the challenger model."""

    config: ShapConfig
    report: ShapReport | None = None

    def run(self) -> ShapReport:
        start = time.monotonic()

        try:
            import shap  # noqa: F401
        except ImportError:
            raise ImportError(
                "shap is not installed. Install with: pip install shap"
            )
        import lightgbm as lgb
        import shap

        # Load model and data
        model = lgb.Booster(model_file=str(self.config.challenger_model_path()))
        X = self._load_features()
        logger.info("Loaded model (%d trees) and data (%d rows, %d cols)",
                    model.num_trees(), len(X), X.shape[1])

        # Sample for SHAP computation (SHAP is expensive)
        sample_n = min(self.config.sample_size, len(X))
        X_sample = X.sample(n=sample_n, random_state=42).reset_index(drop=True)
        logger.info("Sampled %d rows for SHAP computation", sample_n)

        # Compute SHAP values
        logger.info("Computing SHAP values (this may take a few minutes)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values may be a list [class_0, class_1].
        # We want class_1 (default) contributions.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_df = pd.DataFrame(
            shap_values, columns=X_sample.columns, index=X_sample.index
        )

        # Global importance
        global_importance = self._global_importance(shap_df)
        logger.info(
            "Top feature by |SHAP|: %s (mean |SHAP| = %.4f)",
            global_importance[0].feature, global_importance[0].mean_abs_shap,
        )

        # Persist global report
        self._write_global(global_importance)

        # Persist local SHAP values
        local_path = self.config.local_shap_path()
        shap_df.to_parquet(
            local_path, engine="pyarrow", compression="snappy", index=False,
        )
        logger.info("Local SHAP values written: %s", local_path.name)

        # Persist contribution matrix (same as SHAP for challenger)
        contrib_path = self.config.contribution_matrix_path()
        shap_df.to_parquet(
            contrib_path, engine="pyarrow", compression="snappy", index=False,
        )
        logger.info("Contribution matrix written: %s", contrib_path.name)

        # Build report
        duration = time.monotonic() - start
        self.report = ShapReport(
            generated_at=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            model="challenger",
            model_version="0.1.0",
            n_samples_global=sample_n,
            n_samples_local=sample_n,
            n_features=X.shape[1],
            top_features=global_importance[: self.config.top_k],
            duration_seconds=duration,
        )
        self._log_summary(self.report)
        return self.report

    def _load_features(self) -> pd.DataFrame:
        path = self.config.challenger_split_path()
        if not path.exists():
            raise FileNotFoundError(f"Split not found: {path}")

        df = pd.read_parquet(path, engine="pyarrow")
        drop = [self.config.key_column, self.config.target_column]
        X = df.drop(columns=[c for c in drop if c in df.columns])
        logger.info("Loaded features: %d rows, %d cols", len(X), X.shape[1])
        return X

    def _global_importance(
        self, shap_df: pd.DataFrame
    ) -> list[GlobalFeatureImportance]:
        """Compute mean |SHAP| per feature, ranked."""
        mean_abs = shap_df.abs().mean()
        mean_signed = shap_df.mean()

        rows = sorted(
            [
                GlobalFeatureImportance(
                    rank=0,  # filled below
                    feature=feat,
                    mean_abs_shap=float(mean_abs[feat]),
                    mean_shap=float(mean_signed[feat]),
                )
                for feat in shap_df.columns
            ],
            key=lambda r: -r.mean_abs_shap,
        )

        for i, r in enumerate(rows, 1):
            rows[i - 1] = GlobalFeatureImportance(
                rank=i,
                feature=r.feature,
                mean_abs_shap=r.mean_abs_shap,
                mean_shap=r.mean_shap,
            )

        return rows

    def _write_global(self, importance: list[GlobalFeatureImportance]) -> None:
        path = self.config.global_shap_path()
        with path.open("w", encoding="utf-8") as f:
            import json
            json.dump(
                {
                    "generated_at": datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%SZ"
                    ),
                    "model": "challenger",
                    "features": [r.to_dict() for r in importance],
                },
                f, indent=2,
            )
        logger.info("Global SHAP written: %s", path.name)

    def _log_summary(self, report: ShapReport) -> None:
        logger.info("=" * 70)
        logger.info("SHAP Analysis Complete")
        logger.info("  model:              %s", report.model)
        logger.info("  samples:            %d", report.n_samples_global)
        logger.info("  features:           %d", report.n_features)
        logger.info("  duration:           %.1fs", report.duration_seconds)
        logger.info("  ---")
        logger.info("  Top 10 features by mean |SHAP|:")
        for f in report.top_features[:10]:
            logger.info(
                "    #%2d  %-40s  mean|SHAP|=%.6f  mean_SHAP=%+.6f",
                f.rank, f.feature, f.mean_abs_shap, f.mean_shap,
            )
        logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Champion contributions (coefficient-based)
# ---------------------------------------------------------------------------

def linear_contributions(
    model: Any,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute coefficient contributions for a linear model.

    For an applicant with features x and coefficients β, the log-odds
    contribution of feature j is β_j × x_j. This is exactly the champion's
    additive explanation.
    """
    coefs = model.coef_[0]
    if len(coefs) != X.shape[1]:
        raise ValueError(
            f"Coefficient count ({len(coefs)}) does not match feature count "
            f"({X.shape[1]})"
        )
    contributions = X.to_numpy() * coefs  # broadcasting
    return pd.DataFrame(contributions, columns=X.columns, index=X.index)