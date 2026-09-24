import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from casefile.models import ClaimState

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://casefile:casefile@localhost:5432/casefile"
)
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_PATH.read_text())


def insert_claim(conn: psycopg.Connection, claim_id: str, raw_input: dict) -> None:
    conn.execute(
        "INSERT INTO claims (claim_id, raw_input) VALUES (%s, %s) "
        "ON CONFLICT (claim_id) DO NOTHING",
        (claim_id, json.dumps(raw_input)),
    )


def save_snapshot(
    conn: psycopg.Connection, claim_id: str, step_number: int, node_name: str, state: ClaimState
) -> None:
    conn.execute(
        "INSERT INTO claim_state_snapshots (claim_id, step_number, node_name, state_json) "
        "VALUES (%s, %s, %s, %s)",
        (claim_id, step_number, node_name, state.model_dump_json()),
    )


def load_latest_snapshot(conn: psycopg.Connection, claim_id: str) -> ClaimState | None:
    row = conn.execute(
        "SELECT state_json FROM claim_state_snapshots WHERE claim_id = %s "
        "ORDER BY step_number DESC LIMIT 1",
        (claim_id,),
    ).fetchone()
    if row is None:
        return None
    return ClaimState.model_validate(row["state_json"])


def record_trace_event(
    conn: psycopg.Connection, claim_id: str, step_number: int, node_name: str,
    tokens_in: int, tokens_out: int, dollars: float, status: str,
) -> None:
    conn.execute(
        "INSERT INTO trace_events (claim_id, step_number, node_name, tokens_in, tokens_out, dollars, status) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (claim_id, step_number, node_name, tokens_in, tokens_out, dollars, status),
    )


def list_trace_events(conn: psycopg.Connection, claim_id: str) -> list[dict]:
    return conn.execute(
        "SELECT * FROM trace_events WHERE claim_id = %s ORDER BY step_number ASC",
        (claim_id,),
    ).fetchall()


def create_pending_approval(conn: psycopg.Connection, claim_id: str, recommendation: dict) -> None:
    conn.execute(
        "INSERT INTO pending_approvals (claim_id, recommendation_json, status) "
        "VALUES (%s, %s, 'pending') "
        "ON CONFLICT (claim_id) DO UPDATE SET recommendation_json = EXCLUDED.recommendation_json, status = 'pending'",
        (claim_id, json.dumps(recommendation)),
    )


def list_pending_approvals(conn: psycopg.Connection) -> list[dict]:
    return conn.execute(
        "SELECT * FROM pending_approvals WHERE status = 'pending'"
    ).fetchall()


def decide_approval(conn: psycopg.Connection, claim_id: str, approved: bool, decided_by: str) -> None:
    conn.execute(
        "UPDATE pending_approvals SET status = %s, decided_by = %s, decided_at = now() "
        "WHERE claim_id = %s",
        ("approved" if approved else "rejected", decided_by, claim_id),
    )


def write_back(conn: psycopg.Connection, claim_id: str, payout_amount: float) -> None:
    conn.execute(
        "INSERT INTO claims_system_writeback (claim_id, payout_amount) VALUES (%s, %s) "
        "ON CONFLICT (claim_id) DO NOTHING",
        (claim_id, payout_amount),
    )
