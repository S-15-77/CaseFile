from casefile.synthetic import generate_claims


def test_generates_thirty_claims():
    claims = generate_claims(seed=42)
    assert len(claims) == 30
    ids = [c["claim_id"] for c in claims]
    assert len(set(ids)) == 30


def test_mix_includes_variance_flag_and_over_limit_and_budget_stress_cases():
    claims = generate_claims(seed=42)
    over_limit = [c for c in claims if c["prior_claims_history"]["policy_limit"] < 3000]
    assert len(over_limit) >= 1
    many_docs = [c for c in claims if len(c["documents"]) >= 4]
    assert len(many_docs) >= 1


def test_deterministic_with_same_seed():
    a = generate_claims(seed=42)
    b = generate_claims(seed=42)
    assert [c["claim_id"] for c in a] == [c["claim_id"] for c in b]
