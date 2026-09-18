"""
Reason code generation for adverse-action notices.

Converts feature contributions (from SHAP for the challenger, from
coefficients for the champion) into structured, human-readable reason
codes suitable for rejected applicants.

Design notes
------------
- Reason codes are a deterministic mapping from feature names to
  structured codes and human-readable descriptions. The mapping is
  defined in configs/reason_codes.yaml.
- The contribution magnitude determines the ordering: the top K
  features by absolute contribution are selected, filtered by
  direction (positive = increasing risk).
- The module is model-agnostic: it operates on the signed feature
  contributions list, regardless of how they were computed.
- Reason codes must be traceable to model features and stable across
  runs. They must not be misleading or reference features the
  applicant cannot understand.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


logger = logging.getLogger("credit_risk.explainability.reason_codes")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReasonCodeMapping:
    """Configuration for one reason code."""

    feature: str
    code: str
    description: str
    direction: str  # "positive" or "negative"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "code": self.code,
            "description": self.description,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class ReasonCodeConfig:
    """All reason code mappings."""

    version: str
    reasons: dict[str, ReasonCodeMapping]

    @classmethod
    def from_yaml(cls, path: Path) -> "ReasonCodeConfig":
        if not path.exists():
            raise FileNotFoundError(f"Reason code config not found: {path}")

        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        reasons: dict[str, ReasonCodeMapping] = {}
        for feature, spec in raw["reasons"].items():
            direction = spec.get("direction", "positive")
            if direction not in ("positive", "negative"):
                raise ValueError(
                    f"Invalid direction '{direction}' for reason '{feature}'"
                )
            reasons[feature] = ReasonCodeMapping(
                feature=feature,
                code=spec["code"],
                description=spec["description"],
                direction=direction,
            )

        return cls(
            version=raw.get("version", "0.0.0"),
            reasons=reasons,
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ReasonCode:
    """A single reason code for one applicant."""

    feature: str
    code: str
    description: str
    contribution: float  # signed SHAP/coefficient value

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "code": self.code,
            "description": self.description,
            "contribution": round(self.contribution, 6),
        }


@dataclass
class AdverseActionNotice:
    """
    A complete adverse-action notice for one applicant.

    Compliant with the structure required by ECOA / Reg B (US) and
    analogous requirements in other jurisdictions: identifies the
    primary factors that contributed to the decision, in language
    the applicant can understand.
    """

    applicant_id: str
    pd: float
    decision: str
    primary_reasons: list[ReasonCode] = field(default_factory=list)
    additional_reasons: list[ReasonCode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicant_id": self.applicant_id,
            "pd": round(self.pd, 6),
            "decision": self.decision,
            "primary_reasons": [r.to_dict() for r in self.primary_reasons],
            "additional_reasons": [r.to_dict() for r in self.additional_reasons],
        }


# ---------------------------------------------------------------------------
# Reason Code Generator
# ---------------------------------------------------------------------------

@dataclass
class ReasonCodeGenerator:
    """
    Generates reason codes from feature contributions.

    The generator takes per-applicant, per-feature contributions
    (SHAP values for tree models, coefficient contributions for
    linear models), and produces structured reason codes for adverse-
    action notices.
    """

    config: ReasonCodeConfig
    n_primary: int = 4      # number of primary reasons
    n_additional: int = 6   # additional reasons if primary is insufficient

    def generate(
        self,
        applicant_id: str,
        pd: float,
        decision: str,
        contributions: dict[str, float],
    ) -> AdverseActionNotice:
        """
        Produce a reason code notice for one applicant.

        Parameters
        ----------
        applicant_id : str
        pd : float
            Calibrated probability of default.
        decision : str
            APPROVE / REFER / DECLINE. Only DECLINE and REFER typically
            need reason codes; APPROVE is passed through with empty reasons.
        contributions : dict[str, float]
            Signed contribution per feature. Positive = increases risk.
            Typically the SHAP value or coefficient contribution.
        """
        # Only DECLINE and REFER need reason codes
        if decision == "APPROVE":
            return AdverseActionNotice(
                applicant_id=applicant_id,
                pd=pd,
                decision=decision,
                primary_reasons=[],
                additional_reasons=[],
            )

        # Build ranked list of (feature, contribution) for positive
        # contributions only — reasons should explain why risk is high
        positive_contribs: list[tuple[str, float]] = []
        for feature, value in contributions.items():
            if value <= 0:
                continue
            mapping = self.config.reasons.get(feature)
            if mapping is None:
                # Feature not in reason-code config — skip
                continue
            # Direction check: if config says "negative", higher feature
            # values are protective, not risk-increasing, so skip
            if mapping.direction == "negative":
                continue
            positive_contribs.append((feature, float(value)))

        # Sort by contribution descending
        positive_contribs.sort(key=lambda x: -x[1])

        primary: list[ReasonCode] = []
        additional: list[ReasonCode] = []

        for feature, value in positive_contribs:
            mapping = self.config.reasons[feature]
            reason = ReasonCode(
                feature=feature,
                code=mapping.code,
                description=mapping.description,
                contribution=value,
            )
            if len(primary) < self.n_primary:
                primary.append(reason)
            elif len(additional) < self.n_additional:
                additional.append(reason)
            else:
                break

        return AdverseActionNotice(
            applicant_id=applicant_id,
            pd=pd,
            decision=decision,
            primary_reasons=primary,
            additional_reasons=additional,
        )

    def generate_batch(
        self,
        applicant_ids: list[str],
        pds: list[float],
        decisions: list[str],
        contribution_matrix: pd.DataFrame,
    ) -> list[AdverseActionNotice]:
        """
        Generate notices for a batch of applicants.

        Parameters
        ----------
        applicant_ids : list of str
        pds : list of float
        decisions : list of str
        contribution_matrix : pd.DataFrame
            Rows are applicants (same order), columns are features, values
            are signed contributions.
        """
        if not (len(applicant_ids) == len(pds) == len(decisions) == len(contribution_matrix)):
            raise ValueError("Lengths must all match")

        notices: list[AdverseActionNotice] = []
        for i, (aid, pd_val, dec) in enumerate(zip(applicant_ids, pds, decisions)):
            contributions = contribution_matrix.iloc[i].to_dict()
            notices.append(self.generate(
                applicant_id=aid,
                pd=pd_val,
                decision=dec,
                contributions=contributions,
            ))
        return notices


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_notice_text(notice: AdverseActionNotice) -> str:
    """
    Render an adverse-action notice as a plain text document.

    Suitable for inclusion in a rejection letter or a customer-facing
    explanation. Uses simple, non-technical language.
    """
    lines: list[str] = []

    if notice.decision == "APPROVE":
        return f"Applicant {notice.applicant_id}: APPROVED"

    lines.append(f"Adverse Action Notice")
    lines.append(f"Applicant ID: {notice.applicant_id}")
    lines.append(f"Decision: {notice.decision}")
    lines.append(f"")
    lines.append(
        f"Your application was reviewed and could not be approved at this "
        f"time. The primary factors considered in this decision were:"
    )
    lines.append("")

    for i, reason in enumerate(notice.primary_reasons, 1):
        lines.append(f"  {i}. {reason.description}")

    if notice.additional_reasons:
        lines.append("")
        lines.append("Additional factors considered:")
        lines.append("")
        for i, reason in enumerate(notice.additional_reasons, 1):
            lines.append(f"  {i}. {reason.description}")

    lines.append("")
    lines.append(
        "This decision was made using an automated system. You have the "
        "right to request a human review of this decision."
    )

    return "\n".join(lines)