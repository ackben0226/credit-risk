"""
Run WoE binning, IV computation, and feature selection.

Outputs:
    artifacts/binning/<feature>.json            (per-feature binning tables)
    artifacts/reports/binning_summary.json
    artifacts/reports/feature_selection.json
    data/processed/champion_train.parquet
    data/processed/champion_val.parquet
    data/processed/champion_holdout.parquet

Exit codes:
    0 — success
    1 — binning error
    2 — missing input (run earlier stages first)
    3 — missing dependency (pip install optbinning)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.features.binning import (  # noqa: E402
    BinningConfig,
    BinningPipeline,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_binning")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config = BinningConfig(
        project_root=project_root,
        processed_dir=project_root / "data" / "processed",
        splits_dir=project_root / "artifacts" / "splits",
        binning_dir=project_root / "artifacts" / "binning",
        reports_dir=project_root / "artifacts" / "reports",
    )

    try:
        report = BinningPipeline(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        return 3
    except Exception:
        logger.exception("Binning failed")
        return 1

    logger.info(
        "Binning succeeded: %d features kept, train shape %s",
        report.n_features_kept, report.train_shape,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())