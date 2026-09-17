from __future__ import annotations
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.features.aggregations.bureau import (
    BureauAggregationConfig,
    BureauAggregator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger = logging.getLogger("credit_risk.run_bureau_aggregation")

def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config = BureauAggregationConfig(
        project_root=project_root,
        interim_dir=project_root / "data" / "interim",
        reports_dir=project_root / "artifacts" / "reports"
    )

    try:
        report = BureauAggregator(config=config).run()
    except FileNotFoundError as exc:
        logger.info(f"Missing input: %s", exc)
        return 2

    except Exception:
        logger.exception("Bureau aggregation failed")
        return 1

    logger.info("Bureau aggregation succeeded: %d features", report.n_features)
    return 0

if __name__ == "__main__":
    sys.exit(main())

