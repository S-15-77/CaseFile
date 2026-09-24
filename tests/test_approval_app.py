from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from casefile.approval_app import app

client = TestClient(app)


def test_index_lists_pending_approvals():
    fake_rows = [{"claim_id": "C-1", "recommendation_json": {"payout_amount": 500.0, "rationale": "clean"}, "status": "pending"}]
    with patch("casefile.approval_app.db.get_connection") as get_conn, \
         patch("casefile.approval_app.db.list_pending_approvals", return_value=fake_rows):
        get_conn.return_value = MagicMock()
        resp = client.get("/")
    assert resp.status_code == 200
    assert "C-1" in resp.text


def test_approve_calls_decide_approval_and_replay_then_redirects():
    with patch("casefile.approval_app.db.get_connection") as get_conn, \
         patch("casefile.approval_app.db.decide_approval") as decide, \
         patch("casefile.approval_app.replay_claim") as replay:
        get_conn.return_value = MagicMock()
        resp = client.post("/approve/C-1", follow_redirects=False)
    decide.assert_called_once()
    assert decide.call_args.kwargs.get("approved") is True or decide.call_args[0][2] is True
    replay.assert_called_once_with("C-1")
    assert resp.status_code in (302, 303, 307)


def test_reject_never_calls_replay():
    with patch("casefile.approval_app.db.get_connection") as get_conn, \
         patch("casefile.approval_app.db.decide_approval") as decide, \
         patch("casefile.approval_app.replay_claim") as replay:
        get_conn.return_value = MagicMock()
        client.post("/reject/C-1", follow_redirects=False)
    decide.assert_called_once()
    replay.assert_not_called()
