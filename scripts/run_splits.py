"""
Generate deterministic train/validation/holdout splits.

Outputs:
    artifacts/splits/splits.json

Exit codes:
    0 — success
    1 — split generation error
    2 — missing input file
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.features.splits import SplitsConfig, Splitter  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_splits")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config = SplitsConfig(
        project_root=project_root,
        processed_dir=project_root / "data" / "processed",
        splits_dir=project_root / "artifacts" / "splits",
    )

    try:
        report = Splitter(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except Exception:
        logger.exception("Split generation failed")
        return 1

    logger.info(
        "Splits generated: dev=%d, val=%d, test=%d",
        report.n_dev, report.n_val, report.n_test,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())