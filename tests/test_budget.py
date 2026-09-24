from casefile.budget import (
    MAX_STEPS, MAX_DOLLARS, MAX_REVISIONS, tokens_to_dollars, over_budget,
)
from casefile.models import ClaimState


def test_tokens_to_dollars_is_positive_and_scales():
    assert tokens_to_dollars(0) == 0.0
    assert tokens_to_dollars(1000) > tokens_to_dollars(100)


def test_over_budget_false_when_under_limits():
    state = ClaimState(claim_id="C-001", step_count=1, dollars_used=0.01)
    assert over_budget(state) is False


def test_over_budget_true_when_step_count_at_ceiling():
    state = ClaimState(claim_id="C-001", step_count=MAX_STEPS)
    assert over_budget(state) is True


def test_over_budget_true_when_dollars_at_ceiling():
    state = ClaimState(claim_id="C-001", dollars_used=MAX_DOLLARS)
    assert over_budget(state) is True


def test_max_revisions_is_two():
    assert MAX_REVISIONS == 2
