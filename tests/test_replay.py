import os
import uuid
from unittest.mock import patch

import psycopg
import pytest

from casefile.db import db
from casefile.models import (
    ClaimState, ClaimDocument, DocumentType, ExtractedFields, DamageLineItem,
    Recommendation, Decision, ClaimStatus,
)
from casefile.nodes.reviewer import ReviewerOutput
from casefile.graph import run_claim
from casefile.replay import replay_claim

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://casefile:casefile@localhost:5432/casefile")


def _skip_if_unreachable():
    try:
        conn = psycopg.connect(DATABASE_URL, connect_timeout=2)
        conn.close()
    except Exception:
        pytest.skip("Postgres not reachable at DATABASE_URL; run `docker compose up -d db`")


def _fake_call_structured(extracted, reviewer_output):
    def _call(system_prompt, user_prompt, schema):
        if schema is ExtractedFields:
            return extracted, 50, 20
        if schema is ReviewerOutput:
            return reviewer_output, 50, 20
        raise AssertionError(f"unexpected schema {schema}")
    return _call


def test_replay_reaches_same_terminal_state():
    _skip_if_unreachable()
    conn = db.get_connection()
    db.init_schema(conn)
    claim_id = f"TEST-{uuid.uuid4().hex[:8]}"
    db.insert_claim(conn, claim_id, {"raw": "data"})

    state = ClaimState(
        claim_id=claim_id,
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
        recommendation=Recommendation(decision=Decision.DENY, payout_amount=0.0, rationale="test", requires_human_approval=False),
    )
    with patch("casefile.llm.call_structured", side_effect=_fake_call_structured(extracted, reviewer_output)):
        original = run_claim(state)
        db.save_snapshot(conn, claim_id, step_number=original.step_count, node_name=original.history[-1], state=original)

        replayed = replay_claim(claim_id)

    assert replayed.status == original.status == ClaimStatus.DONE
    conn.close()


def test_replay_raises_for_unknown_claim():
    _skip_if_unreachable()
    with pytest.raises(ValueError):
        replay_claim("NO-SUCH-CLAIM")
