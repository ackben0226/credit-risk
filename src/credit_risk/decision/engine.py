"""
Decision engine: converts calibrated PD into business decisions.

The engine is the last mile between the model and the business action.
It reads a decision policy from configuration (cost parameters, segment
thresholds, policy overrides) and applies it to calibrated probabilities
of default, producing APPROVE / REFER / DECLINE decisions.

Design notes
------------
- The threshold is a business decision, not a model output. It is
  configurable without retraining and audit-traceable.
- Cost parameters C_FN and C_FP are externally supplied. The
  cost-optimal threshold is derived from them by minimizing expected
  cost on a reference population (see `scripts/run_decision_demo.py`).
- Segment-specific thresholds supersede defaults. Segments not listed
  use the `default` segment.
- Policy overrides fire before threshold logic. A sanctions hit forces
  DECLINE regardless of PD.
- Every decision is auditable: the engine records the PD, the
  threshold used, the segment, any override that fired, and the
  expected cost of the decision.
- Batch decisions preserve input order. The engine is deterministic.

Outputs
-------
A `DecisionOutcome` per applicant, containing the decision and its
metadata. The caller is responsible for logging and downstream routing.

Config schema
-------------
The engine reads from `configs/decision.yaml`. Required fields:

    costs.C_FN, costs.C_FP
    thresholds.approve_max, thresholds.review_max
    segments.<name>.approve_max, segments.<name>.review_max
    policy_overrides.<name>.force_decision, policy_overrides.<name>.reason
    model.default_model, model.version
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


logger = logging.getLogger("credit_risk.decision.engine")


# ---------------------------------------------------------------------------
# Decision labels
# ---------------------------------------------------------------------------

DECISION_APPROVE = "APPROVE"
DECISION_REFER = "REFER"
DECISION_DECLINE = "DECLINE"

VALID_DECISIONS = frozenset({DECISION_APPROVE, DECISION_REFER, DECISION_DECLINE})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentThresholds:
    """Thresholds for one segment."""

    approve_max: float
    review_max: float

    def validate(self) -> None:
        if not (0.0 <= self.approve_max <= 1.0):
            raise ValueError(
                f"approve_max must be in [0, 1], got {self.approve_max}"
            )
        if not (0.0 <= self.review_max <= 1.0):
            raise ValueError(
                f"review_max must be in [0, 1], got {self.review_max}"
            )
        if self.approve_max >= self.review_max:
            raise ValueError(
                f"approve_max ({self.approve_max}) must be < review_max "
                f"({self.review_max})"
            )


@dataclass(frozen=True)
class PolicyOverride:
    """A policy-level override that forces a decision."""

    name: str
    force_decision: str
    reason: str


@dataclass(frozen=True)
class DecisionConfig:
    """Resolved decision configuration."""

    version: str
    effective_date: str
    c_fn: float
    c_fp: float
    default_segment: SegmentThresholds
    segments: dict[str, SegmentThresholds]
    policy_overrides: dict[str, PolicyOverride]
    default_model: str
    model_version: str

    def segment_thresholds(self, segment: str) -> SegmentThresholds:
        """Look up segment thresholds; fall back to default if unknown."""
        return self.segments.get(segment, self.default_segment)

    @classmethod
    def from_yaml(cls, path: Path) -> "DecisionConfig":
        """Load configuration from a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Decision config not found: {path}")

        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Costs
        costs = raw["costs"]
        c_fn = float(costs["C_FN"])
        c_fp = float(costs["C_FP"])
        if c_fn <= 0 or c_fp <= 0:
            raise ValueError("C_FN and C_FP must be positive")

        # Default thresholds
        thresholds = raw["thresholds"]
        default_segment = SegmentThresholds(
            approve_max=float(thresholds["approve_max"]),
            review_max=float(thresholds["review_max"]),
        )
        default_segment.validate()

        # Segment overrides
        segments: dict[str, SegmentThresholds] = {"default": default_segment}
        for name, seg in raw.get("segments", {}).items():
            if name == "default":
                continue
            st = SegmentThresholds(
                approve_max=float(seg["approve_max"]),
                review_max=float(seg["review_max"]),
            )
            st.validate()
            segments[name] = st

        # Policy overrides
        overrides: dict[str, PolicyOverride] = {}
        for name, ov in raw.get("policy_overrides", {}).items():
            fd = ov["force_decision"]
            if fd not in VALID_DECISIONS:
                raise ValueError(
                    f"Invalid force_decision '{fd}' for override '{name}'. "
                    f"Must be one of {sorted(VALID_DECISIONS)}"
                )
            overrides[name] = PolicyOverride(
                name=name,
                force_decision=fd,
                reason=ov["reason"],
            )

        # Model
        model = raw.get("model", {})
        default_model = model.get("default_model", "challenger")
        model_version = model.get("version", "0.0.0")

        return cls(
            version=raw.get("version", "0.0.0"),
            effective_date=raw.get("effective_date", ""),
            c_fn=c_fn,
            c_fp=c_fp,
            default_segment=default_segment,
            segments=segments,
            policy_overrides=overrides,
            default_model=default_model,
            model_version=model_version,
        )


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

@dataclass
class DecisionOutcome:
    """The engine's output for one applicant."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicant_id": self.applicant_id,
            "pd": round(self.pd, 6),
            "decision": self.decision,
            "threshold_used": round(self.threshold_used, 6),
            "segment": self.segment,
            "override_applied": self.override_applied,
            "override_reason": self.override_reason,
            "expected_cost": round(self.expected_cost, 6),
            "model": self.model,
            "model_version": self.model_version,
            "config_version": self.config_version,
            "decided_at": self.decided_at,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

@dataclass
class DecisionEngine:
    """
    Converts calibrated PD into APPROVE / REFER / DECLINE decisions.

    The engine is stateless aside from its config. It is safe to call
    from multiple threads.
    """

    config: DecisionConfig
    model_name: str = ""       # overrides config.default_model if set

    def decide_one(
        self,
        applicant_id: str,
        pd: float,
        segment: str = "default",
        policy_flags: dict[str, bool] | None = None,
    ) -> DecisionOutcome:
        """
        Produce a decision for one applicant.

        Parameters
        ----------
        applicant_id : str
            Identifier for the applicant (for audit log).
        pd : float
            Calibrated probability of default, in [0, 1].
        segment : str
            Segment name; must exist in config or fall back to default.
        policy_flags : dict[str, bool], optional
            Keys match policy_overrides in config. Any flag set to True
            triggers the corresponding override.
        """
        if not (0.0 <= pd <= 1.0):
            raise ValueError(f"pd must be in [0, 1], got {pd}")

        # Resolve model name
        model = self.model_name or self.config.default_model

        # Apply policy overrides first
        override = self._check_policy_overrides(policy_flags or {})

        if override is not None:
            return DecisionOutcome(
                applicant_id=applicant_id,
                pd=float(pd),
                decision=override.force_decision,
                threshold_used=0.0,
                segment=segment,
                override_applied=override.name,
                override_reason=override.reason,
                expected_cost=0.0,
                model=model,
                model_version=self.config.model_version,
                config_version=self.config.version,
                decided_at=self._now_iso(),
            )

        # Segment thresholds
        seg_thresholds = self.config.segment_thresholds(segment)
        approve_max = seg_thresholds.approve_max
        review_max = seg_thresholds.review_max

        # Threshold decision
        if pd < approve_max:
            decision = DECISION_APPROVE
            threshold_used = approve_max
        elif pd < review_max:
            decision = DECISION_REFER
            threshold_used = review_max
        else:
            decision = DECISION_DECLINE
            threshold_used = review_max

        # Expected cost given the decision
        expected_cost = self._expected_cost(
            pd=pd, decision=decision, threshold=threshold_used
        )

        return DecisionOutcome(
            applicant_id=applicant_id,
            pd=float(pd),
            decision=decision,
            threshold_used=float(threshold_used),
            segment=segment,
            override_applied=None,
            override_reason=None,
            expected_cost=float(expected_cost),
            model=model,
            model_version=self.config.model_version,
            config_version=self.config.version,
            decided_at=self._now_iso(),
        )

    def decide_batch(
        self,
        applicants: pd.DataFrame,
        policy_flags: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Produce decisions for a batch of applicants.

        Parameters
        ----------
        applicants : pd.DataFrame
            Must have columns: applicant_id, pd. Optional: segment.
        policy_flags : pd.DataFrame, optional
            Same index/length as applicants; columns are policy override names,
            values are booleans.

        Returns
        -------
        pd.DataFrame with one row per input applicant, columns matching
        DecisionOutcome.
        """
        required_cols = {"applicant_id", "pd"}
        missing = required_cols - set(applicants.columns)
        if missing:
            raise KeyError(f"applicants missing required columns: {missing}")

        if policy_flags is not None and len(policy_flags) != len(applicants):
            raise ValueError(
                f"policy_flags has {len(policy_flags)} rows, "
                f"applicants has {len(applicants)}"
            )

        outcomes: list[dict[str, Any]] = []
        for i, row in applicants.iterrows():
            applicant_id = str(row["applicant_id"])
            pd_val = float(row["pd"])
            segment = str(row.get("segment", "default"))

            flags: dict[str, bool] = {}
            if policy_flags is not None:
                flag_row = policy_flags.iloc[i] if isinstance(i, int) else policy_flags.loc[i]
                flags = {
                    col: bool(flag_row[col])
                    for col in policy_flags.columns
                    if flag_row[col]
                }

            outcome = self.decide_one(
                applicant_id=applicant_id,
                pd=pd_val,
                segment=segment,
                policy_flags=flags,
            )
            outcomes.append(outcome.to_dict())

        return pd.DataFrame(outcomes)

    # ---- internals -------------------------------------------------------

    def _check_policy_overrides(
        self, flags: dict[str, bool]
    ) -> PolicyOverride | None:
        """
        Return the first firing policy override, or None.

        Ordering: overrides fire in the order they appear in the config.
        If multiple fire, the first in config order wins.
        """
        for name, override in self.config.policy_overrides.items():
            if flags.get(name, False):
                return override
        return None

    def _expected_cost(
        self,
        pd: float,
        decision: str,
        threshold: float,
    ) -> float:
        """
        Expected cost of the decision, in the same units as C_FN / C_FP.

        - APPROVE: expected loss is C_FN * pd (cost if the applicant defaults).
        - DECLINE: expected loss is C_FP * (1 - pd) (cost of losing a good applicant).
        - REFER: cost of manual review, assumed 0.1 * C_FP. Adjust per policy.
        """
        c_fn = self.config.c_fn
        c_fp = self.config.c_fp

        if decision == DECISION_APPROVE:
            return c_fn * pd
        if decision == DECISION_DECLINE:
            return c_fp * (1.0 - pd)
        # REFER: manual review cost
        return 0.1 * c_fp

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Cost-optimal threshold helper
# ---------------------------------------------------------------------------

def optimal_threshold_for_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float,
    cost_fp: float,
    n_grid: int = 99,
) -> tuple[float, float]:
    """
    Compute the cost-optimal decision threshold on a reference population.

    Sweeps a grid of candidate thresholds; for each, computes expected
    cost of the resulting decisions and returns the minimum.

    Returns (threshold, expected_cost).
    """
    candidates = np.linspace(0.01, 0.99, n_grid)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    best_threshold = 0.5
    best_cost = float("inf")

    for t in candidates:
        approve = y_prob < t
        n_fn = int((approve & (y_true == 1)).sum())
        n_fp = int((~approve & (y_true == 0)).sum())
        cost = (cost_fn * n_fn + cost_fp * n_fp) / n
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)

    return best_threshold, best_cost