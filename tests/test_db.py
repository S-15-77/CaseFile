import os
import uuid

import psycopg
import pytest

from casefile.db import db
from casefile.models import ClaimState, ClaimDocument, DocumentType

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://casefile:casefile@localhost:5432/casefile")


def _skip_if_unreachable():
    try:
        conn = psycopg.connect(DATABASE_URL, connect_timeout=2)
        conn.close()
    except Exception:
        pytest.skip("Postgres not reachable at DATABASE_URL; run `docker compose up -d db`")


def test_snapshot_round_trip():
    _skip_if_unreachable()
    conn = db.get_connection()
    db.init_schema(conn)
    claim_id = f"TEST-{uuid.uuid4().hex[:8]}"
    db.insert_claim(conn, claim_id, {"raw": "data"})

    state = ClaimState(
        claim_id=claim_id,
        documents=[ClaimDocument(id="d1", type=DocumentType.PHOTO_ESTIMATE, raw_text="x")],
        step_count=1,
        history=["extractor"],
    )
    db.save_snapshot(conn, claim_id, step_number=1, node_name="extractor", state=state)

    loaded = db.load_latest_snapshot(conn, claim_id)
    assert loaded is not None
    assert loaded.claim_id == claim_id
    assert loaded.history == ["extractor"]
    conn.close()


def test_trace_events_and_pending_approvals():
    _skip_if_unreachable()
    conn = db.get_connection()
    db.init_schema(conn)
    claim_id = f"TEST-{uuid.uuid4().hex[:8]}"
    db.insert_claim(conn, claim_id, {"raw": "data"})

    db.record_trace_event(conn, claim_id, 1, "extractor", 100, 50, 0.001, "ok")
    events = db.list_trace_events(conn, claim_id)
    assert len(events) == 1
    assert events[0]["node_name"] == "extractor"

    db.create_pending_approval(conn, claim_id, {"decision": "approve", "payout_amount": 500.0})
    pending = db.list_pending_approvals(conn)
    assert any(p["claim_id"] == claim_id for p in pending)

    db.decide_approval(conn, claim_id, approved=True, decided_by="test-user")
    db.write_back(conn, claim_id, payout_amount=500.0)
    conn.close()
