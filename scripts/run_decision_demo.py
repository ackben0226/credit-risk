"""
Decision engine demo.

Applies the decision engine to holdout predictions from both models,
shows decision distribution, and verifies the config's threshold is
close to the cost-optimal threshold for the reference population.

Outputs:
    artifacts/reports/decision_demo.json
    artifacts/reports/decision_outcomes.parquet

Exit codes:
    0 — success
    1 — error
    2 — missing input
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.decision.engine import (  # noqa: E402
    DecisionConfig,
    DecisionEngine,
    optimal_threshold_for_cost,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_decision_demo")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "decision.yaml"
    reports_dir = project_root / "artifacts" / "reports"
    processed_dir = project_root / "data" / "processed"

    # Load decision config
    config = DecisionConfig.from_yaml(config_path)
    logger.info(
        "Loaded decision config: version=%s, C_FN=%.1f, C_FP=%.1f, "
        "default approve_max=%.4f",
        config.version, config.c_fn, config.c_fp,
        config.default_segment.approve_max,
    )

    # Load calibrated predictions
    preds_path = reports_dir / "calibrated_predictions.parquet"
    if not preds_path.exists():
        logger.error("Calibrated predictions not found: %s", preds_path)
        return 2

    preds = pd.read_parquet(preds_path)
    logger.info("Loaded predictions: %d rows", len(preds))

    # Load targets
    holdout = pd.read_parquet(
        processed_dir / "challenger_holdout.parquet",
        columns=["TARGET"],
    )
    y_true = holdout["TARGET"].to_numpy()

    # Compute cost-optimal threshold on challenger holdout
    chall_holdout = preds[
        (preds["model"] == "challenger") & (preds["split"] == "holdout")
    ]
    chall_pd = chall_holdout["calibrated_pd"].to_numpy()

    opt_threshold, opt_cost = optimal_threshold_for_cost(
        y_true=y_true, y_prob=chall_pd,
        cost_fn=config.c_fn, cost_fp=config.c_fp,
    )
    logger.info(
        "Cost-optimal threshold (challenger holdout): %.4f  "
        "expected cost: %.6f",
        opt_threshold, opt_cost,
    )
    logger.info(
        "Config default threshold: %.4f  (difference: %+.4f)",
        config.default_segment.approve_max,
        config.default_segment.approve_max - opt_threshold,
    )

    # Apply engine to holdout for both models
    outcomes: list[dict] = []
    for model in ["champion", "challenger"]:
        model_preds = preds[
            (preds["model"] == model) & (preds["split"] == "holdout")
        ].copy()
        model_preds = model_preds.reset_index(drop=True)
        model_preds["applicant_id"] = [f"{model}_{i}" for i in range(len(model_preds))]
        model_preds["segment"] = "default"

        engine = DecisionEngine(config=config, model_name=model)
        batch = engine.decide_batch(
            applicants=model_preds[["applicant_id", "calibrated_pd", "segment"]].rename(
                columns={"calibrated_pd": "pd"}
            )
        )
        outcomes.append(batch)

    all_outcomes = pd.concat(outcomes, ignore_index=True)

    # Save outcomes
    out_path = reports_dir / "decision_outcomes.parquet"
    all_outcomes.to_parquet(
        out_path, engine="pyarrow", compression="snappy", index=False,
    )
    logger.info("Decision outcomes written: %s", out_path.name)

    # Distribution summary
    summary: dict = {
        "config_version": config.version,
        "cost_ratio": config.c_fn / config.c_fp,
        "default_approve_max": config.default_segment.approve_max,
        "default_review_max": config.default_segment.review_max,
        "cost_optimal_threshold": round(opt_threshold, 6),
        "cost_optimal_expected_cost": round(opt_cost, 6),
        "threshold_delta": round(
            config.default_segment.approve_max - opt_threshold, 6
        ),
        "decisions": {},
    }

    for model in ["champion", "challenger"]:
        subset = all_outcomes[all_outcomes["model"] == model]
        dist = subset["decision"].value_counts().to_dict()
        total = len(subset)
        summary["decisions"][model] = {
            "n_total": total,
            "approve_count": dist.get("APPROVE", 0),
            "approve_pct": round(dist.get("APPROVE", 0) / total * 100, 2),
            "refer_count": dist.get("REFER", 0),
            "refer_pct": round(dist.get("REFER", 0) / total * 100, 2),
            "decline_count": dist.get("DECLINE", 0),
            "decline_pct": round(dist.get("DECLINE", 0) / total * 100, 2),
        }

        # Actual positive rate among each decision class
        y_split = y_true[: len(subset)]
        subset_reset = subset.reset_index(drop=True)
        for decision in ["APPROVE", "REFER", "DECLINE"]:
            mask = subset_reset["decision"] == decision
            if mask.any():
                actual_rate = float(y_split[mask.values].mean())
                summary["decisions"][model][f"actual_rate_{decision}"] = round(
                    actual_rate, 6
                )

    # Save summary
    summary_path = reports_dir / "decision_demo.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Decision demo summary written: %s", summary_path.name)

    # Log summary
    logger.info("=" * 70)
    logger.info("Decision Engine Demo")
    logger.info("  ---")
    logger.info("  Config: version=%s", summary["config_version"])
    logger.info("  Cost ratio C_FN/C_FP: %.1f", summary["cost_ratio"])
    logger.info("  Config approve_max: %.4f", summary["default_approve_max"])
    logger.info("  Config review_max: %.4f", summary["default_review_max"])
    logger.info("  Cost-optimal threshold: %.4f", summary["cost_optimal_threshold"])
    logger.info("  Delta: %+.4f", summary["threshold_delta"])
    logger.info("  ---")
    for model in ["champion", "challenger"]:
        d = summary["decisions"][model]
        logger.info("  %s:", model)
        logger.info(
            "    APPROVE: %d (%.2f%%)   actual default rate: %.4f",
            d["approve_count"], d["approve_pct"],
            d.get("actual_rate_APPROVE", 0.0),
        )
        logger.info(
            "    REFER:   %d (%.2f%%)   actual default rate: %.4f",
            d["refer_count"], d["refer_pct"],
            d.get("actual_rate_REFER", 0.0),
        )
        logger.info(
            "    DECLINE: %d (%.2f%%)   actual default rate: %.4f",
            d["decline_count"], d["decline_pct"],
            d.get("actual_rate_DECLINE", 0.0),
        )
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())