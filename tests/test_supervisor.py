from casefile.budget import MAX_STEPS, MAX_REVISIONS
from casefile.models import ClaimState, ClaimStatus, Recommendation, Decision, RevisionRequest
from casefile.nodes import supervisor


def test_route_after_extractor_normal():
    state = ClaimState(claim_id="C-1", step_count=1)
    assert supervisor.route_after_extractor(state) == "investigator"


def test_route_after_extractor_over_budget():
    state = ClaimState(claim_id="C-1", step_count=MAX_STEPS)
    assert supervisor.route_after_extractor(state) == "terminated_over_budget"
    assert state.status == ClaimStatus.TERMINATED_OVER_BUDGET


def test_route_after_investigator_normal():
    state = ClaimState(claim_id="C-1", step_count=2)
    assert supervisor.route_after_investigator(state) == "reviewer"


def test_route_after_reviewer_sends_back_when_revision_requested():
    state = ClaimState(
        claim_id="C-1", step_count=3, revision_count=0,
        revision_request=RevisionRequest(reason="incomplete", notes="check again"),
    )
    assert supervisor.route_after_reviewer(state) == "investigator"
    assert state.revision_count == 1


def test_route_after_reviewer_forces_final_after_max_revisions():
    state = ClaimState(
        claim_id="C-1", step_count=3, revision_count=MAX_REVISIONS,
        revision_request=RevisionRequest(reason="incomplete", notes="check again"),
        recommendation=Recommendation(decision=Decision.ESCALATE, payout_amount=0.0, rationale="forced", requires_human_approval=False),
    )
    result = supervisor.route_after_reviewer(state)
    assert result == "done"
    assert state.status == ClaimStatus.DONE


def test_route_after_reviewer_pending_approval_when_payout_requires_it():
    state = ClaimState(
        claim_id="C-1", step_count=3,
        recommendation=Recommendation(decision=Decision.APPROVE, payout_amount=500.0, rationale="clean", requires_human_approval=True),
    )
    assert supervisor.route_after_reviewer(state) == "pending_approval"
    assert state.status == ClaimStatus.PENDING_APPROVAL


def test_route_after_reviewer_done_when_no_approval_needed():
    state = ClaimState(
        claim_id="C-1", step_count=3,
        recommendation=Recommendation(decision=Decision.DENY, payout_amount=0.0, rationale="over limit", requires_human_approval=False),
    )
    assert supervisor.route_after_reviewer(state) == "done"
    assert state.status == ClaimStatus.DONE


def test_route_after_reviewer_over_budget_wins():
    state = ClaimState(
        claim_id="C-1", step_count=MAX_STEPS,
        recommendation=Recommendation(decision=Decision.APPROVE, payout_amount=500.0, rationale="clean", requires_human_approval=True),
    )
    assert supervisor.route_after_reviewer(state) == "terminated_over_budget"
