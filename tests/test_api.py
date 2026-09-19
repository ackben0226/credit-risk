"""
Smoke tests for the FastAPI service.

These tests verify:
- Basic endpoint availability
- Request/response schema validation
- End-to-end scoring with a synthetic feature dict
- Readiness state reporting

Tests run against the FastAPI app via TestClient, without spawning
a real server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_features():
    """Load one row of holdout features as a synthetic applicant."""
    root = Path(__file__).resolve().parents[1]
    df = pd.read_parquet(
        root / "data" / "processed" / "challenger_holdout.parquet",
        engine="pyarrow",
    )
    row = df.iloc[0]
    features = {
        col: (None if pd.isna(row[col]) else float(row[col]))
        for col in df.columns
        if col not in ("SK_ID_CURR", "TARGET")
    }
    return features


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness(client):
    r = client.get("/readiness")
    assert r.status_code == 200
    body = r.json()
    assert "ready" in body
    assert "model_loaded" in body
    assert "calibrator_loaded" in body


def test_model_info(client):
    r = client.get("/model-info")
    if r.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] > 0
    assert body["model"] in ("champion", "challenger")


def test_score(client, sample_features):
    r = client.post(
        "/score",
        json={
            "applicant_id": "test_001",
            "features": sample_features,
        },
    )
    if r.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applicant_id"] == "test_001"
    assert 0.0 <= body["pd"] <= 1.0
    assert body["decision"] in ("APPROVE", "REFER", "DECLINE")


def test_score_rejects_missing_fields(client):
    r = client.post("/score", json={"features": {}})
    assert r.status_code == 422  # Pydantic validation error


def test_explain(client, sample_features):
    r = client.post(
        "/explain",
        json={
            "applicant_id": "test_002",
            "features": sample_features,
            "top_k": 5,
        },
    )
    if r.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applicant_id"] == "test_002"
    assert len(body["top_contributions"]) <= 5


def test_report(client, sample_features):
    r = client.post(
        "/report",
        json={
            "applicant_id": "test_003",
            "features": sample_features,
        },
    )
    if r.status_code == 503:
        pytest.skip("Model not loaded in test environment")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["applicant_id"] == "test_003"
    assert "notice_text" in body