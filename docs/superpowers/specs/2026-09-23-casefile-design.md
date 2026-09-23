# CaseFile: Multi-Agent Claims Triage — Design Spec

Date: 2026-09-23

## Purpose

Prove that a task too large for one prompt can be decomposed across specialized
agents and kept bounded, observable, and replayable. Concretely: a supervisor
routes a claim through extractor → investigator → reviewer, enforces a budget
ceiling (tokens + dollars + step count) with code (not prompting), gates any
payout/write-back behind a human approval step, and produces a trace for every
run that can be replayed from a stored snapshot to the same terminal state.

This is a demo/prototype built to hit the project's "ship gate," not a real
insurer integration. All 30 claims are synthetic.

## Non-goals

- No real document OCR / PDF parsing — "documents" are synthetic plain-text
  fields standing in for extracted OCR output.
- No real claims-system integration — write-back is a local JSON/Postgres
  table simulating one.
- No multi-tenant auth, no production deployment.

## Stack

- Python 3.11+
- LangGraph for the state machine
- Postgres via Docker Compose (local, free) for claim state, snapshots, and
  trace events
- Groq API free tier (`llama-3.3-70b-versatile`) as the LLM, via an env var
  API key
- Pydantic for all typed handoff schemas
- FastAPI + a plain HTML page for the human approval gate and a trace viewer
- matplotlib for the cost-per-claim chart
- pytest for tests

## Data model (Pydantic)

- `ClaimDocument`: id, type (photo_estimate / police_report / policy_doc),
  raw_text (simulated OCR)
- `ExtractedFields`: policy_number, claimant_name, incident_date,
  damage_line_items (list of {description, cost}), reported_total
- `PriorClaimsCheck`: prior_claim_count, prior_claim_flags (list[str]),
  policy_limit, within_limit (bool)
- `DamageSanityCheck`: estimate_total, repair_shop_total, variance_pct,
  flagged (bool), notes
- `Recommendation`: decision (approve / deny / escalate), payout_amount,
  rationale, requires_human_approval (bool)
- `RevisionRequest`: reason, target_node ("investigator"), notes
- `ClaimState`: claim_id, documents, extracted, prior_check, damage_check,
  recommendation, status, step_count, tokens_used, dollars_used,
  revision_count, history (list of node names in order)

All of the above live in `casefile/models.py`. No node passes free text to
another node — every edge passes a `ClaimState` (or a slice of it).

## Graph / nodes

`casefile/graph.py` builds a LangGraph `StateGraph[ClaimState]`:

1. **supervisor** (entry + routing): checks `step_count` and `dollars_used`
   against ceilings before every transition. Routes:
   `start → extractor → investigator → reviewer → (approve|deny|escalate) → END`.
   If reviewer emits a `RevisionRequest` and `revision_count < 2`, routes back
   to `investigator`; on the 2nd send-back, forces reviewer to produce a final
   decision instead (`revision_count` used as the loop guard, not a prompt).
   If budget/step ceiling is exceeded at any check, routes to
   `terminated_over_budget` (a terminal state), independent of what the LLM
   would have chosen.
2. **extractor**: LLM call over `documents` → `ExtractedFields`.
3. **investigator**: looks up prior claims + policy limit (from a synthetic
   claims-history fixture) → `PriorClaimsCheck`; compares damage estimate to
   repair-shop total → `DamageSanityCheck`.
4. **reviewer**: LLM call synthesizing extracted fields + checks →
   `Recommendation`. May instead return a `RevisionRequest` if the
   investigator's output looks incomplete/contradictory (LLM's judgment, but
   the *routing* on that request is fixed code, see supervisor above).

Any `Recommendation` with `decision in {approve, escalate}` and a nonzero
`payout_amount` sets `requires_human_approval = True` and the graph pauses at
a `pending_approval` state — it does not reach `END` or write to the
simulated claims system until a human approves via the web UI.

## Budget ceiling (code-enforced)

Per claim, constants in `casefile/budget.py`:
`MAX_STEPS = 12`, `MAX_DOLLARS = 0.50`, `MAX_REVISIONS = 2`.
Token → dollar conversion uses Groq's published per-token rate for the model
in use (even though the free tier charges $0 today, the accounting is real
so the ceiling is provably enforced by arithmetic, not by the free tier
happening to be generous). `supervisor` checks these before *every* node
dispatch and short-circuits to `terminated_over_budget` on breach — this is
a plain `if` in Python, not something the LLM can reason its way around.

## Persistence & resume

Postgres tables (`casefile/db/schema.sql`):

- `claims`: claim_id, raw synthetic input, created_at
- `claim_state_snapshots`: claim_id, step_number, node_name, state_json
  (full `ClaimState` serialized), created_at — one row per node transition
- `trace_events`: claim_id, step_number, node_name, started_at, finished_at,
  tokens_in, tokens_out, dollars, status
- `pending_approvals`: claim_id, recommendation_json, status
  (pending/approved/rejected), decided_by, decided_at
- `claims_system_writeback`: claim_id, payout_amount, written_at — the
  simulated external system; a row is only inserted after
  `pending_approvals.status = 'approved'`

**Resume**: `casefile/replay.py` loads the latest `claim_state_snapshots` row
for a claim, deserializes `ClaimState`, and re-enters the LangGraph graph at
`state.history[-1]`'s successor node. A replayed run must reach the same
`status` as the original run for the same claim (asserted in the ship-gate
script and in a pytest test).

## Human approval gate

`casefile/approval_app.py`: a small FastAPI app with one page listing
`pending_approvals` rows (claim id, recommendation, payout, rationale) and
Approve/Reject buttons. Approving updates `pending_approvals.status` and
resumes the graph via `replay.py` logic, which then performs the
`claims_system_writeback` insert and moves the claim to `END`. Rejecting also
resumes but routes to a `denied_by_human` terminal state, no write-back.

For the batch ship-gate run (30 claims), approvals can also be granted via a
CLI flag (`--auto-approve`) so the 30-run batch doesn't require 30 manual
clicks, but the code path exercised is identical to the UI path (same
`pending_approvals` row/state machine) — only the *actor* differs.

## Synthetic data

`casefile/synthetic.py` generates 30 claims across these categories (mixed
so the ship gate's requirements are naturally hit):
- ~20 clean claims (approve path, no reviewer send-back)
- ~5 claims with a damage-estimate/repair-cost mismatch large enough to
  trigger a `RevisionRequest` from reviewer → investigator (satisfies "the
  reviewer sends work back at least once and the graph still terminates")
- ~3 claims that exceed the policy limit (escalate/deny path)
- ~2 claims deliberately shaped to blow the budget ceiling (e.g. many
  documents forcing extra extractor calls) to prove
  `terminated_over_budget` triggers from code

Output: `data/synthetic_claims.json`, checked into the repo so runs are
reproducible without regenerating.

## Ship gate deliverables

- `scripts/run_batch.py --auto-approve`: runs all 30 synthetic claims through
  the graph, writes snapshots + trace_events for each, prints a summary
  (terminal state per claim).
- `scripts/replay_one.py <claim_id>`: restores a claim from its last
  snapshot, re-runs to completion, asserts the terminal state matches the
  original run's recorded terminal state.
- `scripts/cost_report.py`: reads `trace_events`, prints a per-claim
  dollars/tokens table and saves `reports/cost_per_claim.png` (matplotlib bar
  chart with the `MAX_DOLLARS` ceiling drawn as a horizontal line, visually
  confirming no claim crossed it).
- `scripts/trace_view.py <claim_id>`: prints the exact node path taken for
  one claim, in order, from `trace_events`.

## Testing

- `tests/test_models.py`: Pydantic schema validation edge cases.
- `tests/test_nodes.py`: each node (extractor/investigator/reviewer) tested
  with a mocked Groq client — deterministic fixture responses in, correct
  typed output out.
- `tests/test_budget.py`: constructs a `ClaimState` at the ceiling boundary,
  asserts `supervisor` routes to `terminated_over_budget` and does **not**
  call the LLM again (proves code enforcement, not prompt-based).
  Also a synthetic-looping scenario (revision_count forced past
  MAX_REVISIONS) asserting the graph still reaches `END`.
- `tests/test_graph_integration.py`: one full claim through the real graph
  (mocked LLM) from `start` to a terminal state.
- `tests/test_replay.py`: run a claim, snapshot it, replay from snapshot,
  assert same terminal state.

## Error handling

- LLM call failures (network/timeout/malformed JSON from Groq): retried once
  with a stricter "return valid JSON matching this schema" reminder, then on
  second failure the node returns a `Recommendation(decision="escalate",
  requires_human_approval=True, rationale="extraction/review failed, needs
  human")` rather than crashing the graph — every claim reaches a terminal
  state or a human queue, never an unhandled exception mid-graph.
- DB write failures: run aborts that claim (marked `errored` in `claims`),
  batch script continues to the next claim rather than stopping the whole
  run.

## Repo structure

```
CaseFile/
  casefile/
    models.py
    graph.py
    nodes/ (extractor.py, investigator.py, reviewer.py, supervisor.py)
    budget.py
    db/ (schema.sql, db.py)
    replay.py
    approval_app.py
    synthetic.py
    llm.py (Groq client wrapper)
  scripts/
    run_batch.py
    replay_one.py
    cost_report.py
    trace_view.py
  data/synthetic_claims.json
  reports/ (generated: cost_per_claim.png)
  tests/
  docker-compose.yml (Postgres only)
  requirements.txt
  README.md
  LOG.md  (running build log: what was done, issues hit, how it works/runs)
  .env.example (GROQ_API_KEY=)
```
