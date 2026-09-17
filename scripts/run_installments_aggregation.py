"""
Run installments aggregation: installments_payments → per-applicant features.

Outputs:
    data/interim/installments_aggregated.parquet
    artifacts/reports/installments_aggregated_features.json

Exit codes:
    0 — success
    1 — aggregation error
    2 — missing input files (run ingestion first)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.features.aggregations.installments import (  # noqa: E402
    InstallmentsAggregationConfig,
    InstallmentsAggregator,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_installments_aggregation")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config = InstallmentsAggregationConfig(
        project_root=project_root,
        interim_dir=project_root / "data" / "interim",
        reports_dir=project_root / "artifacts" / "reports",
    )

    try:
        report = InstallmentsAggregator(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except Exception:
        logger.exception("Installments aggregation failed")
        return 1

    logger.info("Installments aggregation succeeded: %d features", report.n_features)
    return 0


if __name__ == "__main__":
    sys.exit(main())