from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse

from casefile.db import db
from casefile.replay import replay_claim

app = FastAPI(title="CaseFile Approval Queue")


@app.get("/", response_class=HTMLResponse)
def index():
    conn = db.get_connection()
    rows = db.list_pending_approvals(conn)
    conn.close()
    rows_html = "".join(
        f"<tr><td>{r['claim_id']}</td><td>{r['recommendation_json']}</td>"
        f"<td><form method='post' action='/approve/{r['claim_id']}' style='display:inline'>"
        f"<button type='submit'>Approve</button></form> "
        f"<form method='post' action='/reject/{r['claim_id']}' style='display:inline'>"
        f"<button type='submit'>Reject</button></form></td></tr>"
        for r in rows
    )
    return f"""
    <html><body>
    <h1>Pending Approvals</h1>
    <table border="1"><tr><th>Claim</th><th>Recommendation</th><th>Action</th></tr>
    {rows_html}
    </table>
    </body></html>
    """


@app.post("/approve/{claim_id}")
def approve(claim_id: str):
    conn = db.get_connection()
    db.decide_approval(conn, claim_id, approved=True, decided_by="ui")
    conn.close()
    replay_claim(claim_id)
    return RedirectResponse(url="/", status_code=303)


@app.post("/reject/{claim_id}")
def reject(claim_id: str):
    conn = db.get_connection()
    db.decide_approval(conn, claim_id, approved=False, decided_by="ui")
    conn.close()
    return RedirectResponse(url="/", status_code=303)
