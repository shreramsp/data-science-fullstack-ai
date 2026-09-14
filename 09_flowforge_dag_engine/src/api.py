"""Minimal FastAPI backend exposing the DAG engine as a REST service.

Run with:  uvicorn src.api:app --port 8009
This is the "backend" half of the full-stack demo; the Streamlit frontend
(app.py) can call it directly, or fall back to the in-process engine if the
service isn't running -- see the "Execution backend" toggle in the sidebar.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .engine import CycleError, ValidationError, execute_workflow
from .models import WorkflowDefinition

app = FastAPI(title="FlowForge DAG Engine API")

_workflows: dict[str, WorkflowDefinition] = {}


@app.post("/workflows")
def create_workflow(wf: WorkflowDefinition) -> dict:
    _workflows[wf.id] = wf
    return {"id": wf.id, "node_count": len(wf.nodes), "edge_count": len(wf.edges)}


@app.get("/workflows/{workflow_id}")
def get_workflow(workflow_id: str) -> WorkflowDefinition:
    wf = _workflows.get(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    return wf


@app.post("/workflows/{workflow_id}/run")
def run_workflow(workflow_id: str, simulate_latency: bool = True) -> dict:
    wf = _workflows.get(workflow_id)
    if wf is None:
        raise HTTPException(status_code=404, detail="workflow not found")
    try:
        run = execute_workflow(wf, simulate_latency=simulate_latency)
    except CycleError as exc:
        raise HTTPException(status_code=400, detail=f"cycle detected: {exc.remaining}") from exc
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "levels": run.levels,
        "outputs": run.outputs,
        "total_duration_ms": run.total_duration_ms,
        "events": [vars(e) for e in run.events],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
