from casefile.db import db
from casefile.graph import run_claim
from casefile.models import ClaimState


def replay_claim(claim_id: str) -> ClaimState:
    conn = db.get_connection()
    try:
        snapshot = db.load_latest_snapshot(conn, claim_id)
        if snapshot is None:
            raise ValueError(f"no snapshot found for claim {claim_id}")
        # Already-terminal claims just return the snapshot as-is; otherwise
        # re-run the graph from the persisted state.
        if snapshot.status.value not in ("in_progress",):
            return snapshot
        return run_claim(snapshot)
    finally:
        conn.close()
