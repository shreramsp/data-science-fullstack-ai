# 09 — FlowForge DAG Engine

## Purpose

A type-safe workflow DAG (directed acyclic graph) orchestration engine: define
a graph of typed nodes and edges, validate it with Kahn's topological sort
(including cycle detection), and execute it level-by-level with real
thread-pool parallelism and conditional branch short-circuiting. Ships with a
Streamlit frontend and an optional FastAPI backend service.

## Approach

**Type-safety patterns** (`src/models.py`) — Python analogues of robust
TypeScript modeling patterns:
- `NewType`-based `WorkflowId` / `NodeId` give nominal (branded) typing, so a
  static checker like mypy flags mixing the two even though both are `str`
  at runtime.
- Each node kind (`source`, `map`, `filter`, `branch`, `aggregate`) is its own
  Pydantic model with a `Literal["..."]` discriminator, joined into a
  `NodeConfig` discriminated union via `Field(discriminator="type")`.
- `Ok` / `Err` dataclasses model node execution outcomes as a `Result` type
  instead of relying on exceptions for expected failure paths.
- `assert_never()` is used at the end of every exhaustive `if/elif` chain over
  a `Literal` op field in `src/engine.py`, so adding a new op without handling
  it in the executor is a static-analysis error, not a silent runtime bug.

**Algorithm** (`src/engine.py`) — Kahn's topological sort (`O(V + E)`):
in-degree counts are reduced level by level; each level's nodes have no
dependency on one another and are dispatched concurrently to a
`ThreadPoolExecutor`. If nodes remain with a non-zero in-degree after the
queue drains, a cycle exists and `CycleError` reports exactly which nodes are
involved. `branch` nodes evaluate a predicate on their input and mark only the
matching outgoing edge ("true"/"false") as live; any node with no live
inbound edge is recorded as `skipped` and does not execute — a minimal
Airflow-style conditional short-circuit.

**Frontend** (`app.py`, Streamlit) — pick one of five sample workflows (or
paste a custom workflow as JSON), see the DAG rendered by topological level
(matplotlib/networkx — no external Graphviz binary required), run it, and
inspect the per-node event timeline (started/succeeded/skipped/failed,
timing, output) plus final outputs. A sidebar toggle switches between running
the engine in-process or via the FastAPI backend over HTTP.

**Backend** (`src/api.py`, FastAPI) — `POST /workflows`, `POST
/workflows/{id}/run`, `GET /workflows/{id}` wrap the same engine as a REST
service, run separately with `uvicorn`. This is the genuine "full stack"
half: a real HTTP backend the Streamlit frontend can call over the network,
independent of the in-process fallback.

## Setup

```bash
cd 09_flowforge_dag_engine
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Run

Frontend only (in-process engine, no backend server needed):

```bash
./.venv/bin/streamlit run app.py
```

Full stack (backend + frontend calling it over HTTP) — in two terminals:

```bash
# Terminal 1
./.venv/bin/uvicorn src.api:app --port 8009

# Terminal 2
./.venv/bin/streamlit run app.py
# then in the sidebar, set "Execution backend" to "FastAPI service (localhost:8009)"
```

Tests:

```bash
./.venv/bin/python -m pytest tests/ -v
```

## Honest results

- `pytest tests/` → **7 passed, 0 failed** (topological ordering, diamond
  parallel-level grouping and correct aggregation, cycle detection reporting
  the exact cyclic node set, branch skip-propagation, filter→aggregate value
  correctness, 20 randomly generated DAGs all producing a valid topological
  order, and duplicate-node-id rejection).
- Manually verified end-to-end: started `uvicorn src.api:app`, created and
  ran the "Diamond Parallel" workflow via `curl` against `POST /workflows`
  and `POST /workflows/{id}/run` — returned the correct level grouping
  (`[[src], [left, right], [join]]`) and correct output (`join = 34`).
  Started `streamlit run app.py` headless and confirmed it serves (HTTP 200,
  `/_stcore/health` → `ok`) with no exceptions in its log, and separately
  executed all five sample workflows through the exact code paths the UI
  calls (diagram rendering + engine execution), confirming each produces the
  expected output or, for the intentionally cyclic demo workflow, the
  expected `CycleError`.
- Two real bugs were caught and fixed by the test suite during development:
  (1) a `branch` node was passing its own boolean decision downstream instead
  of the original input value; (2) `aggregate(sum/average)` broke when its
  single upstream edge carried a list (e.g. from a `filter` node) rather than
  multiple scalar edges. Both are covered by regression tests
  (`test_branch_skips_non_taken_path`, `test_filter_then_aggregate_keeps_only_matching_elements`).

## Limitations

- The FastAPI backend keeps workflows in an in-memory dict (no persistence);
  restarting the server loses stored workflows.
- The `simulate_latency` toggle only affects the in-process engine mode in
  the current UI wiring; the FastAPI mode is always called with an explicit
  query param but is not re-toggled live mid-session — not a correctness
  issue, just a minor UX asymmetry.
- No drag-and-drop graph editor; custom workflows are authored as JSON pasted
  into a text area. A visual editor is outside the current scope.
- The Kahn's-algorithm parallelism is real (a `ThreadPoolExecutor` per level)
  but Python's GIL means the demonstrated speed-up is from I/O-bound
  simulated latency (`time.sleep`), not CPU-bound work.
