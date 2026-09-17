"""
Generate schema contracts from an inspection JSON.

Run this once after `scripts/inspect_data.py`, then review the emitted
YAMLs under configs/contracts/. Commit them. They become the source of
truth for src/credit_risk/data/validate.py.

The contract builder reads the *full-data* statistics emitted by
inspection (null counts, min/max, unique values). It does not fall back
to sampled statistics — a contract derived from a sample is a governance
liability.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Make src importable without an editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.data.validate import build_contracts_from_inspection  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.build_contracts")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    contracts_dir = project_root / "configs" / "contracts"
    reports_dir = project_root / "artifacts" / "reports"

    reports = sorted(reports_dir.glob("inspection_*.json"))
    if not reports:
        sys.exit(
            f"No inspection reports found in {reports_dir}. "
            "Run scripts/inspect_data.py first."
        )

    latest = reports[-1]
    logger.info("Using inspection report: %s", latest.name)

    build_contracts_from_inspection(latest, contracts_dir)
    logger.info("Contracts written to: %s", contracts_dir)


if __name__ == "__main__":
    main()