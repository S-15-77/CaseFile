# Build Log

Running log of what was built, issues hit, and how the system works — kept
up to date as the project progresses.

## 2026-09-23 — Design

- Brainstormed and wrote the design spec:
  `docs/superpowers/specs/2026-09-23-casefile-design.md`.
- Decisions: Groq free-tier API, Python + LangGraph, synthetic claim data,
  local Postgres via Docker (free, local), a hand-rolled JSON trace log
  instead of a full OpenTelemetry collector, and a small FastAPI page for
  the human approval gate.
- Sandbox issue: this session's Bash sandbox blocks all writes under
  `.git` (even `git init` and `git commit`, even after the repo directory
  existed and the user ran `git init` manually outside the sandbox). Worked
  around it by having the user run all `git init`/`add`/`commit` steps
  themselves outside the sandboxed session; this session only writes
  regular files.

## 2026-09-23 — Implementation plan

- Wrote the task-by-task implementation plan:
  `docs/superpowers/plans/2026-09-23-casefile-implementation.md`.
- 14 tasks, TDD throughout: models → budget ceiling → LLM wrapper →
  extractor/investigator/reviewer nodes → supervisor routing → LangGraph
  assembly → Postgres persistence → replay → synthetic data → approval UI →
  batch/replay/cost-report scripts → ship gate run.

## 2026-09-23 — Implementation

Built directly in-session (skipped the subagent-driven-development workflow
since it depends on committing after every task, which the git sandbox
restriction rules out here).

- **Environment:** created `.venv`, installed `requirements.txt`. Two
  sandbox network issues hit along the way: (1) the sandbox's outbound
  proxy needed `pypi.org` / `files.pythonhosted.org` explicitly allow-listed
  per-command; (2) even allow-listed, pip's own bundled CA bundle
  (`certifi`) didn't trust the sandbox's inspecting proxy certificate,
  while `curl` (using the macOS system trust store) worked fine against the
  same host — fixed by passing `--trusted-host pypi.org --trusted-host
  files.pythonhosted.org` to `pip install`, a targeted fix scoped to those
  two hosts rather than disabling TLS verification generally.
- **Docker:** the daemon isn't running, and this sandboxed session can't
  launch GUI apps (`open -a Docker` fails with a Launch Services error), so
  Postgres-backed tests (`test_db.py`, `test_replay.py`, `test_pipeline.py`)
  currently skip via their own `_skip_if_unreachable()` guard. Start Docker
  Desktop yourself and run `docker compose up -d db` to exercise those.
- **LangGraph state bug:** the first working version of `casefile/graph.py`
  used `StateGraph(ClaimState)` directly (a Pydantic model as the graph's
  state schema) and had nodes return the full updated `ClaimState`. This
  produced validation errors and, worse, silently cross-contaminated field
  values between unrelated nodes on each superstep — LangGraph's per-field
  channel reconciliation for a Pydantic-model state schema does not behave
  like "last full object wins" the way a plain dict does. Fixed by wrapping
  state in a `TypedDict` (`{"claim": ClaimState, "next_route": str}`) so
  there's exactly one channel holding the whole claim object (always fully
  overwritten, never field-merged), and by having each node compute its own
  routing decision (via `casefile/nodes/supervisor.py`) and store the result
  in `next_route` for the conditional edge to read — rather than the
  conditional edge re-invoking (and re-mutating) the supervisor function on
  a freshly-recoerced, disconnected copy of the state, which silently
  discarded status/revision-count updates.
- **Mock-patching bug (same root shape, different code):** the first
  integration test patched `casefile.nodes.extractor.llm.call_structured`
  and `casefile.nodes.reviewer.llm.call_structured` as two separate mocks.
  Both names resolve to the exact same `casefile.llm` module object (both
  modules do `from casefile import llm`), so the second `patch()` context
  silently overwrote the first's effect for the whole duration — extractor
  ended up calling the reviewer's mock. Fixed by patching
  `casefile.llm.call_structured` once with a single `side_effect` that
  dispatches on the requested Pydantic `schema` argument. Applied the same
  fix to `test_replay.py` and `test_pipeline.py`.
- All non-DB tests pass (`pytest -v`, DB tests skipped pending Docker).
- Generated the checked-in `data/synthetic_claims.json` (30 claims, seed 42).

<!-- Append further entries below as implementation proceeds, including the
     actual ship-gate run results once Postgres is available. -->
