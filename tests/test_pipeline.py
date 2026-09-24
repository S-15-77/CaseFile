import os
import uuid
from unittest.mock import patch

import psycopg
import pytest

from casefile.db import db
from casefile.models import ExtractedFields, DamageLineItem, Recommendation, Decision, ClaimStatus
from casefile.nodes.reviewer import ReviewerOutput
from casefile.pipeline import process_claim

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://casefile:casefile@localhost:5432/casefile")


def _skip_if_unreachable():
    try:
        conn = psycopg.connect(DATABASE_URL, connect_timeout=2)
        conn.close()
    except Exception:
        pytest.skip("Postgres not reachable at DATABASE_URL; run `docker compose up -d db`")


def test_process_claim_auto_approves_and_writes_back():
    _skip_if_unreachable()
    conn = db.get_connection()
    db.init_schema(conn)
    claim_id = f"TEST-{uuid.uuid4().hex[:8]}"
    claim_dict = {
        "claim_id": claim_id,
        "documents": [{"id": "d1", "type": "photo_estimate", "raw_text": "bumper $500"}],
        "repair_shop_total": 490.0,
        "prior_claims_history": {"prior_claim_count": 0, "prior_claim_flags": [], "policy_limit": 5000.0},
    }
    extracted = ExtractedFields(
        policy_number="P-1", claimant_name="Jane Doe", incident_date="2026-01-01",
        damage_line_items=[DamageLineItem(description="bumper", cost=500.0)], reported_total=500.0,
    )
    reviewer_output = ReviewerOutput(
        action="recommend",
        recommendation=Recommendation(decision=Decision.APPROVE, payout_amount=500.0, rationale="clean", requires_human_approval=True),
    )

    def fake_call_structured(system_prompt, user_prompt, schema):
        if schema is ExtractedFields:
            return extracted, 50, 20
        if schema is ReviewerOutput:
            return reviewer_output, 50, 20
        raise AssertionError(f"unexpected schema {schema}")

    with patch("casefile.llm.call_structured", side_effect=fake_call_structured):
        result = process_claim(conn, claim_dict, auto_approve=True)

    assert result.status == ClaimStatus.DONE
    writeback = conn.execute(
        "SELECT * FROM claims_system_writeback WHERE claim_id = %s", (claim_id,)
    ).fetchone()
    assert writeback is not None
    assert float(writeback["payout_amount"]) == 500.0
    conn.close()
