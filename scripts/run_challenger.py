"""
Train the challenger LightGBM model.

Outputs:
    artifacts/models/challenger/<version>/model.txt
    artifacts/models/challenger/<version>/metadata.json
    artifacts/reports/challenger_metrics.json
    artifacts/reports/challenger_tuning.json
    artifacts/reports/challenger_predictions.parquet
    artifacts/reports/challenger_feature_importance.csv

Exit codes:
    0 — success
    1 — training error
    2 — missing input
    3 — missing dependency (pip install lightgbm optuna)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.models.challenger import (  # noqa: E402
    ChallengerConfig,
    ChallengerTrainer,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_challenger")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config = ChallengerConfig(
        project_root=project_root,
        processed_dir=project_root / "data" / "processed",
        models_dir=project_root / "artifacts" / "models",
        reports_dir=project_root / "artifacts" / "reports",
    )

    try:
        report = ChallengerTrainer(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        return 3
    except Exception:
        logger.exception("Challenger training failed")
        return 1

    logger.info("Challenger trained: best trial val AUC = %.4f",
                max(t.val_auc for t in report.trial_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())