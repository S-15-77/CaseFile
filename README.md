# CaseFile

A bounded, observable, replayable multi-agent claims-triage pipeline.
Supervisor + extractor/investigator/reviewer agents process a claim through
a LangGraph state machine with typed (Pydantic) handoffs, a code-enforced
budget ceiling, a loop guard on reviewer send-backs, and a human approval
gate before any payout write-back.

## Setup

1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill in a free Groq API key from
   https://console.groq.com
4. Start Docker Desktop, then `docker compose up -d db`
5. `python -c "from casefile.db import db; conn = db.get_connection(); db.init_schema(conn)"`

## Run the tests

`pytest -v`

DB-dependent tests (`test_db.py`, `test_replay.py`, `test_pipeline.py`)
auto-skip if Postgres isn't reachable.

## Run the ship gate

```bash
# 30 recorded claim runs, auto-approving payouts (same code path as manual UI approval)
python scripts/run_batch.py --auto-approve

# Pick any claim_id printed above and replay it from its stored snapshot
python scripts/replay_one.py SYN-000

# See the exact node path taken for one claim
python scripts/trace_view.py SYN-000

# Cost-per-claim chart with the budget ceiling drawn as a line
python scripts/cost_report.py
```

## Manual approval UI

`uvicorn casefile.approval_app:app --reload` then open http://localhost:8000
to approve/reject pending payouts by hand instead of `--auto-approve`.

## Architecture

See `docs/superpowers/specs/2026-09-23-casefile-design.md` for the full
design. In short: `extractor` → `investigator` → `reviewer`, routed by pure
Python functions in `casefile/nodes/supervisor.py` that check a budget
ceiling (`casefile/budget.py`) and a revision-count loop guard before every
transition — never an LLM decision. Every node transition is snapshotted to
Postgres so any claim can be resumed via `casefile/replay.py`.
