from casefile.models import ClaimState, PriorClaimsCheck, DamageSanityCheck

VARIANCE_FLAG_THRESHOLD_PCT = 15.0


def run(state: ClaimState) -> ClaimState:
    history = state.prior_claims_history
    policy_limit = history.get("policy_limit", 0.0)
    reported_total = state.extracted.reported_total if state.extracted else 0.0

    state.prior_check = PriorClaimsCheck(
        prior_claim_count=history.get("prior_claim_count", 0),
        prior_claim_flags=history.get("prior_claim_flags", []),
        policy_limit=policy_limit,
        within_limit=reported_total <= policy_limit,
    )

    estimate_total = reported_total
    repair_shop_total = state.repair_shop_total
    if repair_shop_total > 0:
        variance_pct = abs(estimate_total - repair_shop_total) / repair_shop_total * 100
    else:
        variance_pct = 0.0
    flagged = variance_pct >= VARIANCE_FLAG_THRESHOLD_PCT
    state.damage_check = DamageSanityCheck(
        estimate_total=estimate_total,
        repair_shop_total=repair_shop_total,
        variance_pct=round(variance_pct, 2),
        flagged=flagged,
        notes=(
            f"estimate vs repair-shop variance {variance_pct:.1f}%"
            + (" — flagged" if flagged else " — within tolerance")
        ),
    )

    state.step_count += 1
    state.history.append("investigator")
    return state
