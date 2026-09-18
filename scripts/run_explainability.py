"""
Explainability demo.

1. Computes global and local SHAP values for the challenger model.
2. Loads champion model and computes coefficient contributions.
3. Runs the reason code generator on a sample of declined applicants.
4. Writes example adverse-action notices.

Outputs:
    artifacts/reports/challenger_shap_global.json
    artifacts/reports/challenger_shap_local.parquet
    artifacts/reports/challenger_contributions.parquet
    artifacts/reports/champion_contributions.parquet
    artifacts/reports/example_adverse_actions.json

Exit codes:
    0 — success
    1 — error
    2 — missing input
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.explainability.reason_codes import (  # noqa: E402
    ReasonCodeConfig,
    ReasonCodeGenerator,
    render_notice_text,
)
from credit_risk.explainability.shap_analysis import (  # noqa: E402
    ShapAnalyzer,
    ShapConfig,
    linear_contributions,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_explainability")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    # ---- 1. SHAP for challenger ----
    shap_config = ShapConfig(
        project_root=project_root,
        processed_dir=project_root / "data" / "processed",
        models_dir=project_root / "artifacts" / "models",
        reports_dir=project_root / "artifacts" / "reports",
    )

    shap_analyzer = ShapAnalyzer(config=shap_config)
    shap_report = shap_analyzer.run()

    # ---- 2. Champion coefficient contributions ----
    logger.info("Computing champion coefficient contributions")

    champion_model_path = (
        project_root / "artifacts" / "models" / "champion" / "0.1.0" / "model.pkl"
    )
    with champion_model_path.open("rb") as f:
        champion_model = pickle.load(f)

    champion_train = pd.read_parquet(
        project_root / "data" / "processed" / "champion_holdout.parquet",
    )
    champion_X = champion_train.drop(
        columns=["SK_ID_CURR", "TARGET"], errors="ignore"
    )
    # Sample to match SHAP sample for consistency
    n_sample = min(5000, len(champion_X))
    champion_X_sample = champion_X.sample(n=n_sample, random_state=42).reset_index(drop=True)

    champion_contribs = linear_contributions(champion_model, champion_X_sample)
    champion_path = (
        project_root / "artifacts" / "reports" / "champion_contributions.parquet"
    )
    champion_contribs.to_parquet(
        champion_path, engine="pyarrow", compression="snappy", index=False,
    )
    logger.info("Champion contributions written: %s (%d rows)",
                champion_path.name, len(champion_contribs))

    # ---- 3. Reason codes on a sample of declined applicants ----
    logger.info("Generating reason codes for sample of declined applicants")

    reason_config_path = project_root / "configs" / "reason_codes.yaml"
    reason_config = ReasonCodeConfig.from_yaml(reason_config_path)
    generator = ReasonCodeGenerator(config=reason_config)

    # Load decisions from the decision demo
    decisions_path = project_root / "artifacts" / "reports" / "decision_outcomes.parquet"
    if not decisions_path.exists():
        logger.error("Decision outcomes not found: %s", decisions_path)
        return 2

    decisions = pd.read_parquet(decisions_path)
    challenger_decisions = decisions[decisions["model"] == "challenger"].reset_index(drop=True)
    declined = challenger_decisions[challenger_decisions["decision"] == "DECLINE"].head(10)

    logger.info("Selected %d declined applicants for reason code demo", len(declined))

    # Use challenger contributions for the declined applicants
    # NOTE: contributions are on the sampled feature matrix, not on the full
    # holdout. For a real system, we'd compute SHAP on demand for each
    # declined applicant. Here, we use the first N rows of the contributions
    # matrix as a stand-in.
    challenger_contribs_path = project_root / "artifacts" / "reports" / "challenger_contributions.parquet"
    challenger_contribs = pd.read_parquet(challenger_contribs_path)

    # Take first N rows
    n_declined = min(len(declined), len(challenger_contribs))
    sample_contribs = challenger_contribs.iloc[:n_declined]

    notices: list[dict] = []
    example_texts: list[str] = []

    for i in range(n_declined):
        applicant_id = f"demo_{i}"
        # Use the PD from the decision demo, and the actual contribution row
        pd_val = float(declined.iloc[i]["pd"])
        contribs = sample_contribs.iloc[i].to_dict()

        notice = generator.generate(
            applicant_id=applicant_id,
            pd=pd_val,
            decision="DECLINE",
            contributions=contribs,
        )
        notices.append(notice.to_dict())

        if i < 3:
            example_texts.append(render_notice_text(notice))

    # ---- 4. Write example notices ----
    output_path = (
        project_root / "artifacts" / "reports" / "example_adverse_actions.json"
    )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_notices": len(notices),
                "notices": notices,
                "example_texts": example_texts,
            },
            f, indent=2,
        )
    logger.info("Example adverse-action notices written: %s", output_path.name)

    # ---- 5. Summary ----
    logger.info("=" * 70)
    logger.info("Explainability Complete")
    logger.info("  SHAP (challenger):   top feature = %s (mean|SHAP|=%.6f)",
                shap_report.top_features[0].feature,
                shap_report.top_features[0].mean_abs_shap)
    logger.info("  Champion:            %d coefficient rows", len(champion_contribs))
    logger.info("  Reason code demo:    %d notices generated", len(notices))
    logger.info("  Example notices:     first 3 rendered as text")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Example adverse-action notice:")
    logger.info("-" * 70)
    for line in example_texts[0].split("\n"):
        logger.info("  %s", line)
    logger.info("-" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())