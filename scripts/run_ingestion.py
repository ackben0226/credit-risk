"""
Run the ingestion pipeline against the Home Credit raw data.

Converts raw CSVs to interim Parquet under data/interim/. Runs validation
as a precondition; aborts on validation failure.

Exit codes:
    0 — ingestion succeeded
    1 — validation failed or ingestion error
    2 — setup error (missing config, contracts, or inspection)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.data.ingest import ingest_all  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_ingestion")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    try:
        report = ingest_all(project_root)
    except FileNotFoundError as exc:
        logger.error("Setup error: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("Ingestion aborted: %s", exc)
        return 1

    if not report.validation_passed:
        logger.error(
            "Ingestion did not complete: validation failed with %d errors",
            report.validation_error_count,
        )
        return 1

    if not report.results:
        logger.error("Ingestion did not produce any results")
        return 1

    logger.info("Ingestion succeeded: %d tables written", len(report.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())