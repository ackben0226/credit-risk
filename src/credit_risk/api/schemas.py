"""
Pydantic schemas for the FastAPI service.

Defines request and response models for all endpoints. Uses Pydantic v2
strict mode where possible to catch malformed input early.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    """Score a single applicant."""

    model_config = ConfigDict(extra="forbid")

    applicant_id: str = Field(..., description="Unique identifier for the applicant")
    features: dict[str, float | int | None] = Field(
        ...,
        description=(
            "Feature values keyed by feature name. All features the model "
            "expects must be present; nulls are allowed for sparse features."
        ),
    )
    segment: str = Field(
        "default",
        description="Segment name for threshold selection (e.g. new_customer).",
    )
    policy_flags: dict[str, bool] = Field(
        default_factory=dict,
        description="Policy override flags (e.g. sanctions_list, fraud_flag).",
    )


class ExplainRequest(ScoreRequest):
    """Explain a decision for a single applicant."""

    top_k: int = Field(
        10, ge=1, le=50,
        description="Number of top feature contributions to return.",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    ready: bool
    model_loaded: bool
    calibrator_loaded: bool
    reason_codes_loaded: bool


class ModelInfoResponse(BaseModel):
    model: str
    model_version: str
    n_features: int
    config_version: str
    default_approve_max: float
    default_review_max: float
    cost_fn: float
    cost_fp: float


class ScoreResponse(BaseModel):
    applicant_id: str
    pd: float
    decision: str
    threshold_used: float
    segment: str
    override_applied: str | None
    override_reason: str | None
    expected_cost: float
    model: str
    model_version: str
    config_version: str
    decided_at: str


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    value: float | None


class ReasonCodeResponse(BaseModel):
    feature: str
    code: str
    description: str
    contribution: float


class ExplainResponse(BaseModel):
    applicant_id: str
    pd: float
    decision: str
    top_contributions: list[FeatureContribution]
    primary_reasons: list[ReasonCodeResponse]
    additional_reasons: list[ReasonCodeResponse]


class ReportResponse(BaseModel):
    applicant_id: str
    pd: float
    decision: str
    notice_text: str
    primary_reasons: list[ReasonCodeResponse]
    additional_reasons: list[ReasonCodeResponse]