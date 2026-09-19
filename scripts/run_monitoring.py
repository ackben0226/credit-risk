"""
Run monitoring: compute drift between reference and monitoring populations.

Outputs:
    artifacts/reports/monitoring_report.json
    artifacts/reports/feature_drift.parquet
    artifacts/reports/prediction_drift.parquet

Exit codes:
    0 — success
    1 — error
    2 — missing input
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.monitoring.drift import (  # noqa: E402
    ModelMonitor,
    MonitoringConfig,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("credit_risk.run_monitoring")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    config_path = project_root / "configs" / "monitoring_config.yaml"
    try:
        config = MonitoringConfig.from_yaml(config_path, project_root)
    except FileNotFoundError as exc:
        logger.error("Config not found: %s", exc)
        return 2

    try:
        report = ModelMonitor(config=config).run()
    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 2
    except Exception:
        logger.exception("Monitoring failed")
        return 1

    logger.info(
        "Monitoring complete: %d material drift features",
        report.n_features_material,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())