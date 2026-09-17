"""
Run validation against the current Home Credit raw data.

Standalone runner for the validation layer. In production, validation is
invoked automatically by the ingestion pipeline and CI. This script exists
for manual invocation during development.

Exit codes:
    0 — validation passed (no errors)
    1 — validation failed (one or more errors)
    2 — setup error (missing contracts, missing raw data)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make src importable without an editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.data.validate import (  # noqa: E402
    Validator,
    load_contracts,
    save_validation_report,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_validation")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    contracts_dir = project_root / "configs" / "contracts"
    raw_dir = project_root / "data" / "raw" / "home_credit"
    reports_dir = project_root / "artifacts" / "reports"

    # ---- Preconditions --------------------------------------------------

    if not contracts_dir.exists():
        logger.error("Contracts directory not found: %s", contracts_dir)
        logger.error("Run: python scripts/build_contracts.py")
        return 2

    if not raw_dir.exists():
        logger.error("Raw data directory not found: %s", raw_dir)
        return 2

    tables, joins = load_contracts(contracts_dir)
    if not tables:
        logger.error("No table contracts found in %s", contracts_dir)
        logger.error("Run: python scripts/build_contracts.py")
        return 2

    logger.info("Loaded %d table contracts, %d join contracts", len(tables), len(joins))

    # ---- Run validation -------------------------------------------------

    validator = Validator(
        raw_dir=raw_dir,
        tables=tables,
        joins=joins,
    )
    report = validator.run()

    # ---- Persist report -------------------------------------------------

    out_path = reports_dir / f"validation_{report.generated_at}.json"
    save_validation_report(report, out_path)

    # ---- Exit code ------------------------------------------------------

    if not report.passed:
        logger.error(
            "VALIDATION FAILED: %d errors, %d warnings",
            report.error_count,
            report.warning_count,
        )
        return 1

    logger.info(
        "VALIDATION PASSED: %d checks, %d warnings",
        report.checks_run,
        report.warning_count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())