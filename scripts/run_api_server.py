"""
Run the FastAPI service locally for development.

Usage:
    python scripts/run_api_server.py

Server listens on http://localhost:8000 by default.

Endpoints:
    GET  /health          liveness
    GET  /readiness       readiness
    GET  /model-info      metadata
    POST /score           calibrated PD + decision
    POST /explain         SHAP contributions + reason codes
    POST /report          adverse-action notice text

Interactive API docs at:
    http://localhost:8000/docs
    http://localhost:8000/redoc
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def main() -> int:
    uvicorn.run(
        "credit_risk.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())