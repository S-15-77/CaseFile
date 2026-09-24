from casefile.db import db
from casefile.graph import run_claim
from casefile.models import ClaimState, ClaimStatus
from casefile.replay import replay_claim


def process_claim(conn, claim_dict: dict, auto_approve: bool = False) -> ClaimState:
    claim_id = claim_dict["claim_id"]
    db.insert_claim(conn, claim_id, claim_dict)

    state = ClaimState.model_validate(claim_dict)
    result = run_claim(state)

    db.save_snapshot(conn, claim_id, step_number=result.step_count, node_name=(result.history[-1] if result.history else "start"), state=result)
    db.record_trace_event(
        conn, claim_id, step_number=result.step_count, node_name=",".join(result.history),
        tokens_in=result.tokens_used, tokens_out=0, dollars=result.dollars_used,
        status=result.status.value,
    )

    if result.status == ClaimStatus.PENDING_APPROVAL:
        db.create_pending_approval(conn, claim_id, result.recommendation.model_dump())
        if auto_approve:
            db.decide_approval(conn, claim_id, approved=True, decided_by="batch-auto-approve")
            result = replay_claim(claim_id)
            db.write_back(conn, claim_id, payout_amount=result.recommendation.payout_amount)
            result.status = ClaimStatus.DONE
            db.save_snapshot(conn, claim_id, step_number=result.step_count + 1, node_name="write_back", state=result)

    return result
