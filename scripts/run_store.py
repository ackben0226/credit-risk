"""
Build challenger feature stores from the assembled feature matrix.

Outputs:
    data/processed/challenger_train.parquet
    data/processed/challenger_val.parquet
    data/processed/challenger_holdout.parquet
    artifacts/reports/challenger_feature_catalogue.json

Exit codes:
    0 — success
    1 — build error
    2 — missing input
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.features.store import (  # noqa: E402
    ChallengerStoreBuilder,
    StoreConfig,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_store")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config = StoreConfig(
        project_root=project_root,
        processed_dir=project_root / "data" / "processed",
        splits_dir=project_root / "artifacts" / "splits",
        reports_dir=project_root / "artifacts" / "reports",
    )

    try:
        report = ChallengerStoreBuilder(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except Exception:
        logger.exception("Store build failed")
        return 1

    logger.info(
        "Challenger store built: %d features, train shape %s",
        report.n_features_output, report.train_shape,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())