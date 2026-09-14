"""FlowForge DAG Engine -- Streamlit frontend."""
from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st

from src.engine import CycleError, NodeEvent, ValidationError, WorkflowRun, execute_workflow
from src.models import WorkflowDefinition
from src.sample_workflows import SAMPLE_WORKFLOWS
from src.viz import draw_workflow

API_BASE = "http://127.0.0.1:8009"

st.set_page_config(page_title="FlowForge DAG Engine", page_icon="🏗️", layout="wide")

st.title("🏗️ FlowForge DAG Engine")
st.caption(
    "A type-safe workflow DAG engine built on Kahn's topological sort, with "
    "parallel level execution and conditional branch skipping."
)

with st.sidebar:
    st.header("Workflow")
    choice = st.selectbox("Sample workflow", list(SAMPLE_WORKFLOWS.keys()) + ["Custom (paste JSON)"])
    simulate_latency = st.checkbox("Simulate node latency (50ms/node)", value=True)
    backend_mode = st.radio(
        "Execution backend",
        ["In-process engine", "FastAPI service (localhost:8009)"],
        help="The FastAPI option requires `uvicorn src.api:app --port 8009` running separately.",
    )
    st.markdown("---")
    st.markdown(
        "**Node types:** `source`, `map`, `filter`, `branch`, `aggregate`\n\n"
        "**Algorithm:** Kahn's topological sort groups independent nodes into "
        "levels that execute concurrently; `branch` nodes short-circuit the "
        "non-taken path."
    )

parse_error: str | None = None
if choice == "Custom (paste JSON)":
    default_json = json.dumps(SAMPLE_WORKFLOWS["Linear Pipeline"].model_dump(), indent=2)
    raw = st.text_area("Workflow JSON", value=default_json, height=280)
    try:
        wf: WorkflowDefinition | None = WorkflowDefinition.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as validation feedback
        wf = None
        parse_error = str(exc)
else:
    wf = SAMPLE_WORKFLOWS[choice]

col1, col2 = st.columns([1, 1])

if parse_error:
    st.error(f"Invalid workflow JSON:\n\n{parse_error}")
elif wf is not None:
    with col1:
        st.subheader("DAG structure")
        try:
            fig = draw_workflow(wf)
            st.pyplot(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001 - visualization is best-effort
            st.warning(f"Could not render diagram: {exc}")
        with st.expander("Raw workflow definition"):
            st.json(wf.model_dump())

    with col2:
        st.subheader("Execution")

        def run_via_api(workflow: WorkflowDefinition, latency: bool) -> WorkflowRun:
            requests.post(f"{API_BASE}/workflows", json=workflow.model_dump(mode="json"), timeout=5)
            resp = requests.post(
                f"{API_BASE}/workflows/{workflow.id}/run",
                params={"simulate_latency": latency},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return WorkflowRun(
                levels=data["levels"],
                events=[NodeEvent(**e) for e in data["events"]],
                outputs=data["outputs"],
                total_duration_ms=data["total_duration_ms"],
            )

        def render_run(run: WorkflowRun) -> None:
            st.success(
                f"Completed in {run.total_duration_ms:.1f} ms across {len(run.levels)} topological level(s)"
            )
            st.write("**Topological levels (each runs its nodes in parallel):**")
            for i, level in enumerate(run.levels):
                st.write(f"Level {i}: {', '.join(level)}")

            status_icon = {"started": "\U0001F535", "succeeded": "✅", "skipped": "⏭️", "failed": "❌"}
            rows = [
                {
                    "node": e.node_id,
                    "status": f"{status_icon.get(e.status, '')} {e.status}",
                    "t (ms)": round(e.timestamp * 1000, 1),
                    "duration (ms)": round(e.duration_ms, 1),
                    "output": e.output,
                    "error": e.error,
                }
                for e in run.events
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.write("**Final node outputs:**")
            st.json(run.outputs)

        if st.button("▶ Run workflow", type="primary"):
            try:
                if backend_mode.startswith("FastAPI"):
                    run_result = run_via_api(wf, simulate_latency)
                else:
                    run_result = execute_workflow(wf, simulate_latency=simulate_latency)
                render_run(run_result)
            except CycleError as exc:
                st.error(f"Cycle detected -- cannot topologically sort nodes: {exc.remaining}")
            except ValidationError as exc:
                st.error(f"Invalid workflow: {exc}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the FastAPI backend at "
                    f"{API_BASE}. Start it with `uvicorn src.api:app --port 8009`, "
                    "or switch to 'In-process engine'."
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
                st.error(f"Execution failed: {exc}")
