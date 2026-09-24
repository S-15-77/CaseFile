from casefile.models import ClaimState, ExtractedFields, DamageLineItem
from casefile.nodes import investigator


def _state_with_extracted(reported_total, repair_shop_total, prior_history):
    return ClaimState(
        claim_id="C-001",
        repair_shop_total=repair_shop_total,
        prior_claims_history=prior_history,
        extracted=ExtractedFields(
            policy_number="P-1",
            claimant_name="Jane Doe",
            incident_date="2026-01-01",
            damage_line_items=[DamageLineItem(description="bumper", cost=reported_total)],
            reported_total=reported_total,
        ),
    )


def test_investigator_flags_large_variance():
    state = _state_with_extracted(
        reported_total=2000.0,
        repair_shop_total=1000.0,
        prior_history={"prior_claim_count": 0, "prior_claim_flags": [], "policy_limit": 5000.0},
    )
    result = investigator.run(state)
    assert result.damage_check.flagged is True
    assert result.prior_check.within_limit is True
    assert result.step_count == 1
    assert result.history == ["investigator"]


def test_investigator_no_flag_within_variance():
    state = _state_with_extracted(
        reported_total=1000.0,
        repair_shop_total=980.0,
        prior_history={"prior_claim_count": 1, "prior_claim_flags": [], "policy_limit": 5000.0},
    )
    result = investigator.run(state)
    assert result.damage_check.flagged is False


def test_investigator_over_policy_limit():
    state = _state_with_extracted(
        reported_total=6000.0,
        repair_shop_total=6000.0,
        prior_history={"prior_claim_count": 0, "prior_claim_flags": [], "policy_limit": 5000.0},
    )
    result = investigator.run(state)
    assert result.prior_check.within_limit is False
