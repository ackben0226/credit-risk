"""
Run previous_application aggregation: previous_application → per-applicant features.

Outputs:
    data/interim/previous_application_aggregated.parquet
    artifacts/reports/previous_application_aggregated_features.json

Exit codes:
    0 — success
    1 — aggregation error
    2 — missing input files
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.features.aggregations.previous_application import (  # noqa: E402
    PreviousApplicationAggregationConfig,
    PreviousApplicationAggregator,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_previous_application_aggregation")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config = PreviousApplicationAggregationConfig(
        project_root=project_root,
        interim_dir=project_root / "data" / "interim",
        reports_dir=project_root / "artifacts" / "reports",
    )

    try:
        report = PreviousApplicationAggregator(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except Exception:
        logger.exception("Previous application aggregation failed")
        return 1

    logger.info(
        "Previous application aggregation succeeded: %d features",
        report.n_features,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())