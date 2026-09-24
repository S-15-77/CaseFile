from unittest.mock import patch

from casefile.models import (
    ClaimState, ExtractedFields, DamageLineItem, PriorClaimsCheck,
    DamageSanityCheck, Recommendation, Decision, RevisionRequest,
)
from casefile.nodes import reviewer
from casefile.nodes.reviewer import ReviewerOutput


def _base_state():
    return ClaimState(
        claim_id="C-001",
        extracted=ExtractedFields(
            policy_number="P-1", claimant_name="Jane Doe", incident_date="2026-01-01",
            damage_line_items=[DamageLineItem(description="bumper", cost=500.0)],
            reported_total=500.0,
        ),
        prior_check=PriorClaimsCheck(prior_claim_count=0, prior_claim_flags=[], policy_limit=5000.0, within_limit=True),
        damage_check=DamageSanityCheck(estimate_total=500.0, repair_shop_total=490.0, variance_pct=2.0, flagged=False, notes="ok"),
    )


def test_reviewer_recommends_when_action_recommend():
    state = _base_state()
    output = ReviewerOutput(
        action="recommend",
        recommendation=Recommendation(decision=Decision.APPROVE, payout_amount=500.0, rationale="clean", requires_human_approval=True),
    )
    with patch("casefile.nodes.reviewer.llm.call_structured", return_value=(output, 80, 40)):
        result = reviewer.run(state)
    assert result.recommendation.decision == Decision.APPROVE
    assert result.revision_request is None
    assert result.history == ["reviewer"]


def test_reviewer_requests_revision_when_action_revise():
    state = _base_state()
    output = ReviewerOutput(
        action="revise",
        revision_request=RevisionRequest(reason="damage check looks incomplete", notes="re-check variance"),
    )
    with patch("casefile.nodes.reviewer.llm.call_structured", return_value=(output, 80, 40)):
        result = reviewer.run(state)
    assert result.revision_request.reason == "damage check looks incomplete"
    assert result.recommendation is None
