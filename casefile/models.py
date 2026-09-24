from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PHOTO_ESTIMATE = "photo_estimate"
    POLICE_REPORT = "police_report"
    POLICY_DOC = "policy_doc"


class ClaimDocument(BaseModel):
    id: str
    type: DocumentType
    raw_text: str


class DamageLineItem(BaseModel):
    description: str
    cost: float


class ExtractedFields(BaseModel):
    policy_number: str
    claimant_name: str
    incident_date: str
    damage_line_items: list[DamageLineItem]
    reported_total: float


class PriorClaimsCheck(BaseModel):
    prior_claim_count: int
    prior_claim_flags: list[str]
    policy_limit: float
    within_limit: bool


class DamageSanityCheck(BaseModel):
    estimate_total: float
    repair_shop_total: float
    variance_pct: float
    flagged: bool
    notes: str


class Decision(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    ESCALATE = "escalate"


class Recommendation(BaseModel):
    decision: Decision
    payout_amount: float
    rationale: str
    requires_human_approval: bool


class RevisionRequest(BaseModel):
    reason: str
    target_node: str = "investigator"
    notes: str


class ClaimStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    DENIED_BY_HUMAN = "denied_by_human"
    TERMINATED_OVER_BUDGET = "terminated_over_budget"
    ERRORED = "errored"
    DONE = "done"


class ClaimState(BaseModel):
    claim_id: str
    documents: list[ClaimDocument] = Field(default_factory=list)
    repair_shop_total: float = 0.0
    prior_claims_history: dict = Field(default_factory=dict)
    extracted: Optional[ExtractedFields] = None
    prior_check: Optional[PriorClaimsCheck] = None
    damage_check: Optional[DamageSanityCheck] = None
    recommendation: Optional[Recommendation] = None
    revision_request: Optional[RevisionRequest] = None
    status: ClaimStatus = ClaimStatus.IN_PROGRESS
    step_count: int = 0
    tokens_used: int = 0
    dollars_used: float = 0.0
    revision_count: int = 0
    history: list[str] = Field(default_factory=list)
