"""
FastAPI application for the credit risk PD model.

Exposes:
    GET  /health           liveness check
    GET  /readiness        readiness check
    GET  /model-info       model and config metadata
    POST /score            calibrated PD + decision
    POST /explain          SHAP contributions + reason codes
    POST /report           adverse-action notice text

The service loads:
    - The challenger LightGBM model (artifacts/models/challenger/)
    - The challenger calibrator (artifacts/calibrators/challenger/)
    - The decision config (configs/decision.yaml)
    - The reason code config (configs/reason_codes.yaml)

Model loading happens lazily on the first request and is cached. This
keeps the container start fast while allowing readiness checks to
report accurate state.

For production, the model would be loaded at startup via a lifespan
handler. For the demo, lazy loading with a cache is simpler and lets
the service start even if the artifacts aren't yet present.
"""

from __future__ import annotations

import logging
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from credit_risk.api.schemas import (
    ExplainRequest,
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    ModelInfoResponse,
    ReadinessResponse,
    ReasonCodeResponse,
    ReportResponse,
    ScoreRequest,
    ScoreResponse,
)
from credit_risk.decision.engine import (
    DecisionConfig,
    DecisionEngine,
)
from credit_risk.explainability.reason_codes import (
    ReasonCodeConfig,
    ReasonCodeGenerator,
    render_notice_text,
)


logger = logging.getLogger("credit_risk.api.main")


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class ModelState:
    """
    Cached model, calibrator, and configuration.

    Loaded lazily on first request; kept in memory for subsequent requests.
    """

    def __init__(self) -> None:
        self.challenger_model: lgb.Booster | None = None
        self.calibrator: Any = None
        self.decision_config: DecisionConfig | None = None
        self.reason_config: ReasonCodeConfig | None = None
        self.feature_names: list[str] = []
        self.feature_columns: list[str] = []

    @property
    def model_loaded(self) -> bool:
        return self.challenger_model is not None

    @property
    def calibrator_loaded(self) -> bool:
        return self.calibrator is not None

    @property
    def reason_codes_loaded(self) -> bool:
        return self.reason_config is not None

    @property
    def ready(self) -> bool:
        return (
            self.model_loaded
            and self.calibrator_loaded
            and self.reason_codes_loaded
            and self.decision_config is not None
        )


state = ModelState()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Resolve the project root from this file's location."""
    return Path(__file__).resolve().parents[3]


def _load_state() -> None:
    """Load all resources into the global state object."""
    if state.ready:
        return

    root = _project_root()

    # Load decision config
    if state.decision_config is None:
        config_path = root / "configs" / "decision.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Decision config not found: {config_path}")
        state.decision_config = DecisionConfig.from_yaml(config_path)
        logger.info("Decision config loaded: version=%s",
                    state.decision_config.version)

    # Load reason code config
    if state.reason_config is None:
        reason_path = root / "configs" / "reason_codes.yaml"
        if not reason_path.exists():
            raise FileNotFoundError(f"Reason code config not found: {reason_path}")
        state.reason_config = ReasonCodeConfig.from_yaml(reason_path)
        logger.info("Reason code config loaded: %d reasons",
                    len(state.reason_config.reasons))

    # Load challenger model
    if state.challenger_model is None:
        model_path = (
            root / "artifacts" / "models" / "challenger" / "0.1.0" / "model.txt"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Challenger model not found: {model_path}")
        state.challenger_model = lgb.Booster(model_file=str(model_path))
        logger.info("Challenger model loaded: %d trees",
                    state.challenger_model.num_trees())

    # Load calibrator
    if state.calibrator is None:
        cal_path = (
            root / "artifacts" / "calibrators" / "challenger" / "calibrator.pkl"
        )
        if not cal_path.exists():
            raise FileNotFoundError(f"Calibrator not found: {cal_path}")
        with cal_path.open("rb") as f:
            state.calibrator = pickle.load(f)
        logger.info("Calibrator loaded: %s", type(state.calibrator).__name__)

    # Load feature names from the challenger holdout (deterministic order)
    if not state.feature_names:
        sample_path = root / "data" / "processed" / "challenger_holdout.parquet"
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Feature sample not found: {sample_path}"
            )
        df = pd.read_parquet(sample_path, engine="pyarrow")
        state.feature_columns = [
            c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")
        ]
        state.feature_names = list(state.feature_columns)
        logger.info("Feature contract loaded: %d features",
                    len(state.feature_names))


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Attempt eager load on startup; not fatal if it fails."""
    try:
        _load_state()
        logger.info("Startup: all resources loaded")
    except Exception as exc:
        logger.warning(
            "Startup: could not load all resources (%s). "
            "Readiness will report false until resources are available.",
            exc,
        )
    yield
    logger.info("Shutdown: releasing resources")
    state.challenger_model = None
    state.calibrator = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Credit Risk PD Service",
    description=(
        "Calibrated probability of default and decision endpoints for "
        "the Home Credit credit risk model."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: returns 200 if the process is running."""
    return HealthResponse(status="ok")


@app.get("/readiness", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    """Readiness: returns 200 if all resources are loaded and usable."""
    return ReadinessResponse(
        ready=state.ready,
        model_loaded=state.model_loaded,
        calibrator_loaded=state.calibrator_loaded,
        reason_codes_loaded=state.reason_codes_loaded,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model and configuration metadata."""
    _ensure_ready()
    assert state.decision_config is not None  # checked by _ensure_ready

    return ModelInfoResponse(
        model=state.decision_config.default_model,
        model_version=state.decision_config.model_version,
        n_features=len(state.feature_names),
        config_version=state.decision_config.version,
        default_approve_max=state.decision_config.default_segment.approve_max,
        default_review_max=state.decision_config.default_segment.review_max,
        cost_fn=state.decision_config.c_fn,
        cost_fp=state.decision_config.c_fp,
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """
    Score a single applicant: calibrated PD and decision.

    The request must include the applicant's feature values keyed by
    feature name. Missing features trigger a 422 error.
    """
    _ensure_ready()

    t0 = time.monotonic()
    pd_value = _score_features(request.features)
    elapsed_ms = (time.monotonic() - t0) * 1000

    engine = DecisionEngine(config=state.decision_config)
    outcome = engine.decide_one(
        applicant_id=request.applicant_id,
        pd=pd_value,
        segment=request.segment,
        policy_flags=request.policy_flags,
    )

    logger.info(
        "score: applicant=%s pd=%.4f decision=%s latency_ms=%.1f",
        request.applicant_id, pd_value, outcome.decision, elapsed_ms,
    )

    return ScoreResponse(
        applicant_id=outcome.applicant_id,
        pd=outcome.pd,
        decision=outcome.decision,
        threshold_used=outcome.threshold_used,
        segment=outcome.segment,
        override_applied=outcome.override_applied,
        override_reason=outcome.override_reason,
        expected_cost=outcome.expected_cost,
        model=outcome.model,
        model_version=outcome.model_version,
        config_version=outcome.config_version,
        decided_at=outcome.decided_at,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> ExplainResponse:
    """
    Explain a decision for one applicant.

    Returns top feature contributions and mapped reason codes.
    Intended for internal use (underwriters, auditors).
    """
    _ensure_ready()

    pd_value = _score_features(request.features)
    engine = DecisionEngine(config=state.decision_config)
    outcome = engine.decide_one(
        applicant_id=request.applicant_id,
        pd=pd_value,
        segment=request.segment,
        policy_flags=request.policy_flags,
    )

    # Compute per-feature contributions for this applicant
    contributions = _compute_contributions(request.features)

    # Top-K by absolute contribution
    sorted_contribs = sorted(
        contributions.items(), key=lambda kv: -abs(kv[1])
    )[: request.top_k]

    top_contributions = [
        FeatureContribution(
            feature=feat,
            contribution=float(val),
            value=request.features.get(feat),
        )
        for feat, val in sorted_contribs
    ]

    # Reason codes
    generator = ReasonCodeGenerator(config=state.reason_config)
    notice = generator.generate(
        applicant_id=request.applicant_id,
        pd=pd_value,
        decision=outcome.decision,
        contributions=contributions,
    )

    return ExplainResponse(
        applicant_id=outcome.applicant_id,
        pd=outcome.pd,
        decision=outcome.decision,
        top_contributions=top_contributions,
        primary_reasons=[
            ReasonCodeResponse(
                feature=r.feature,
                code=r.code,
                description=r.description,
                contribution=r.contribution,
            )
            for r in notice.primary_reasons
        ],
        additional_reasons=[
            ReasonCodeResponse(
                feature=r.feature,
                code=r.code,
                description=r.description,
                contribution=r.contribution,
            )
            for r in notice.additional_reasons
        ],
    )


@app.post("/report", response_model=ReportResponse)
def report(request: ScoreRequest) -> ReportResponse:
    """
    Return the full adverse-action notice text for one applicant.

    Returns the rendered plain-language notice suitable for sending
    to the applicant in the case of a decline.
    """
    _ensure_ready()

    pd_value = _score_features(request.features)
    engine = DecisionEngine(config=state.decision_config)
    outcome = engine.decide_one(
        applicant_id=request.applicant_id,
        pd=pd_value,
        segment=request.segment,
        policy_flags=request.policy_flags,
    )

    contributions = _compute_contributions(request.features)
    generator = ReasonCodeGenerator(config=state.reason_config)
    notice = generator.generate(
        applicant_id=request.applicant_id,
        pd=pd_value,
        decision=outcome.decision,
        contributions=contributions,
    )

    return ReportResponse(
        applicant_id=outcome.applicant_id,
        pd=outcome.pd,
        decision=outcome.decision,
        notice_text=render_notice_text(notice),
        primary_reasons=[
            ReasonCodeResponse(
                feature=r.feature,
                code=r.code,
                description=r.description,
                contribution=r.contribution,
            )
            for r in notice.primary_reasons
        ],
        additional_reasons=[
            ReasonCodeResponse(
                feature=r.feature,
                code=r.code,
                description=r.description,
                contribution=r.contribution,
            )
            for r in notice.additional_reasons
        ],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_ready() -> None:
    """Ensure resources are loaded; raise 503 if not."""
    if not state.ready:
        try:
            _load_state()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Service not ready: {exc}",
            )
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="Service not ready: model or calibrator missing",
        )


def _score_features(features: dict[str, Any]) -> float:
    """
    Score a feature dict with the challenger model, apply calibration,
    and return the calibrated PD.

    Features not present in the request are treated as NaN. The model
    handles nulls natively. Features in the request but not in the
    model's feature contract are silently ignored.
    """
    assert state.challenger_model is not None
    assert state.calibrator is not None

    # Build a 1-row DataFrame with the model's expected columns
    row = {name: features.get(name, np.nan) for name in state.feature_names}
    X = pd.DataFrame([row], columns=state.feature_names).astype("float64")

    n_iter = state.challenger_model.current_iteration()
    raw_pd = state.challenger_model.predict(X, num_iteration=n_iter or None)
    raw_pd = float(np.clip(raw_pd[0], 0.0, 1.0))

    # Predict raw PD
    raw_pd = state.challenger_model.predict(X, num_iteration=state.challenger_model.current_iteration())
    raw_pd = float(np.clip(raw_pd[0], 0.0, 1.0))

    # Apply calibration
    calibrated = _apply_calibrator(raw_pd)
    return float(np.clip(calibrated, 0.0, 1.0))


def _apply_calibrator(raw_pd: float) -> float:
    """
    Apply the loaded calibrator to a single PD value.

    The calibrator is either LogisticRegression (Platt) or
    IsotonicRegression. Both accept a 1-D array and return calibrated
    probabilities.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    cal = state.calibrator
    arr = np.array([raw_pd])

    if isinstance(cal, IsotonicRegression):
        return float(cal.predict(arr)[0])

    if isinstance(cal, LogisticRegression):
        eps = 1e-9
        p = np.clip(arr, eps, 1 - eps)
        logit_p = np.log(p / (1 - p))
        prob = cal.predict_proba(logit_p.reshape(-1, 1))[:, 1]
        return float(prob[0])

    raise TypeError(f"Unsupported calibrator type: {type(cal).__name__}")


def _compute_contributions(features: dict[str, Any]) -> dict[str, float]:
    """
    Compute SHAP contributions for a single applicant.

    Uses the challenger model's TreeExplainer. This is the slow path
    (SHAP requires a full tree traversal), but returns exactly
    interpretable contributions.
    """
    import shap

    assert state.challenger_model is not None

    row = {name: features.get(name, np.nan) for name in state.feature_names}
    X = pd.DataFrame([row], columns=state.feature_names).astype("float64")

    explainer = shap.TreeExplainer(state.challenger_model)
    shap_values = explainer.shap_values(X)

    # For LightGBM binary classification, shap_values is a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = shap_values[0]
    return {
        name: float(val) for name, val in zip(state.feature_names, values)
    }