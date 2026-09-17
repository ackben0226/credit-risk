"""
Train the champion scorecard on WoE-encoded features.

Outputs:
    artifacts/models/champion/<version>/model.pkl
    artifacts/models/champion/<version>/metadata.json
    artifacts/reports/champion_metrics.json
    artifacts/reports/champion_tuning.json
    artifacts/reports/champion_predictions.parquet

Exit codes:
    0 — success
    1 — training error
    2 — missing input
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.models.champion import (  # noqa: E402
    ChampionConfig,
    ChampionTrainer,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_champion")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config = ChampionConfig(
        project_root=project_root,
        processed_dir=project_root / "data" / "processed",
        models_dir=project_root / "artifacts" / "models",
        reports_dir=project_root / "artifacts" / "reports",
    )

    try:
        report = ChampionTrainer(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except Exception:
        logger.exception("Champion training failed")
        return 1

    logger.info("Champion trained: val AUC target check")
    return 0


if __name__ == "__main__":
    sys.exit(main())