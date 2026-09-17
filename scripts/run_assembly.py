"""
Run feature assembly: join all aggregated tables into modelling matrices.

Outputs:
    data/processed/application_train_features.parquet
    data/processed/application_test_features.parquet
    artifacts/reports/assembled_feature_catalogue.json

Exit codes:
    0 — success
    1 — assembly error
    2 — missing input files
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.features.assembly import (  # noqa: E402
    AssemblyConfig,
    Assembler,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_assembly")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config = AssemblyConfig(
        project_root=project_root,
        interim_dir=project_root / "data" / "interim",
        processed_dir=project_root / "data" / "processed",
        reports_dir=project_root / "artifacts" / "reports",
    )

    try:
        report = Assembler(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except Exception:
        logger.exception("Assembly failed")
        return 1

    logger.info(
        "Assembly succeeded: train %d × %d, test %d × %d",
        report.train_output_rows, report.train_output_columns,
        report.test_output_rows, report.test_output_columns,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())