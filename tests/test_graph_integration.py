from unittest.mock import patch

from casefile.graph import run_claim
from casefile.models import (
    ClaimDocument, DocumentType, ClaimState, ClaimStatus,
    ExtractedFields, DamageLineItem, Recommendation, Decision,
)
from casefile.nodes.reviewer import ReviewerOutput


def test_full_claim_reaches_pending_approval():
    state = ClaimState(
        claim_id="C-100",
        documents=[ClaimDocument(id="d1", type=DocumentType.PHOTO_ESTIMATE, raw_text="bumper $500")],
        repair_shop_total=490.0,
        prior_claims_history={"prior_claim_count": 0, "prior_claim_flags": [], "policy_limit": 5000.0},
    )
    extracted = ExtractedFields(
        policy_number="P-1", claimant_name="Jane Doe", incident_date="2026-01-01",
        damage_line_items=[DamageLineItem(description="bumper", cost=500.0)], reported_total=500.0,
    )
    reviewer_output = ReviewerOutput(
        action="recommend",
        recommendation=Recommendation(decision=Decision.APPROVE, payout_amount=500.0, rationale="clean", requires_human_approval=True),
    )

    # casefile.nodes.extractor.llm and casefile.nodes.reviewer.llm are the
    # same shared `casefile.llm` module object (both bound via `from
    # casefile import llm`), so patching call_structured on either path
    # patches it for both callers. Dispatch by schema instead of using two
    # separate patches, which would silently overwrite each other.
    def fake_call_structured(system_prompt, user_prompt, schema):
        if schema is ExtractedFields:
            return extracted, 50, 20
        if schema is ReviewerOutput:
            return reviewer_output, 50, 20
        raise AssertionError(f"unexpected schema {schema}")

    with patch("casefile.llm.call_structured", side_effect=fake_call_structured):
        result = run_claim(state)

    assert result.status == ClaimStatus.PENDING_APPROVAL
    assert result.history == ["extractor", "investigator", "reviewer"]
