"""
Run champion/challenger comparison.

Outputs:
    artifacts/reports/model_comparison.json
    docs/champion_challenger.md

Exit codes:
    0 — success
    1 — evaluation error
    2 — missing input
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.models.evaluate import (  # noqa: E402
    EvaluationConfig,
    ModelEvaluator,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_evaluation")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config = EvaluationConfig(
        project_root=project_root,
        processed_dir=project_root / "data" / "processed",
        reports_dir=project_root / "artifacts" / "reports",
        docs_dir=project_root / "docs",
    )

    try:
        report = ModelEvaluator(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except Exception:
        logger.exception("Evaluation failed")
        return 1

    logger.info("Evaluation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())