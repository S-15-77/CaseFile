import pytest
from pydantic import ValidationError

from casefile.models import (
    ClaimDocument, DocumentType, DamageLineItem, ExtractedFields,
    PriorClaimsCheck, DamageSanityCheck, Decision, Recommendation,
    RevisionRequest, ClaimStatus, ClaimState,
)


def test_claim_state_defaults():
    state = ClaimState(claim_id="C-001")
    assert state.status == ClaimStatus.IN_PROGRESS
    assert state.step_count == 0
    assert state.dollars_used == 0.0
    assert state.revision_count == 0
    assert state.documents == []
    assert state.history == []


def test_claim_document_requires_valid_type():
    with pytest.raises(ValidationError):
        ClaimDocument(id="d1", type="not_a_type", raw_text="x")


def test_extracted_fields_round_trip():
    fields = ExtractedFields(
        policy_number="P-1",
        claimant_name="Jane Doe",
        incident_date="2026-01-01",
        damage_line_items=[DamageLineItem(description="bumper", cost=500.0)],
        reported_total=500.0,
    )
    assert fields.damage_line_items[0].cost == 500.0


def test_recommendation_decision_enum():
    rec = Recommendation(
        decision=Decision.APPROVE,
        payout_amount=1200.0,
        rationale="within limits",
        requires_human_approval=True,
    )
    assert rec.decision == Decision.APPROVE


def test_claim_state_history_append():
    state = ClaimState(claim_id="C-002")
    state.history.append("extractor")
    state.history.append("investigator")
    assert state.history == ["extractor", "investigator"]
