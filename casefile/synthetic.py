import json
import random

DAMAGE_DESCRIPTIONS = ["bumper", "windshield", "door panel", "headlight", "fender"]


def _make_claim(i: int, rng: random.Random) -> dict:
    claim_id = f"SYN-{i:03d}"
    reported_total = round(rng.uniform(300, 4000), 2)

    # ~5 claims get a large estimate/repair-shop mismatch (triggers reviewer send-back)
    if 20 <= i < 25:
        repair_shop_total = round(reported_total * rng.uniform(1.3, 1.8), 2)
    else:
        repair_shop_total = round(reported_total * rng.uniform(0.95, 1.05), 2)

    # ~3 claims exceed the policy limit
    if 25 <= i < 28:
        policy_limit = round(reported_total * 0.5, 2)
    else:
        policy_limit = round(rng.uniform(4000, 8000), 2)

    num_docs = 2
    # ~2 claims get many documents to stress the budget ceiling
    if 28 <= i < 30:
        num_docs = 6

    documents = [
        {
            "id": f"{claim_id}-D{j}",
            "type": "photo_estimate" if j % 2 == 0 else "police_report",
            "raw_text": f"{rng.choice(DAMAGE_DESCRIPTIONS)} damage, estimated ${reported_total / num_docs:.2f}",
        }
        for j in range(num_docs)
    ]

    return {
        "claim_id": claim_id,
        "documents": documents,
        "repair_shop_total": repair_shop_total,
        "prior_claims_history": {
            "prior_claim_count": rng.randint(0, 3),
            "prior_claim_flags": [] if rng.random() > 0.1 else ["late_filing"],
            "policy_limit": policy_limit,
        },
    }


def generate_claims(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [_make_claim(i, rng) for i in range(30)]


def write_claims_file(path: str, seed: int = 42) -> None:
    claims = generate_claims(seed=seed)
    with open(path, "w") as f:
        json.dump(claims, f, indent=2)
