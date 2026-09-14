"""FlowForge execution engine: Kahn's topological sort + level-parallel executor.

Algorithm: Kahn's algorithm (in-degree reduction, O(V + E)) is used both to
validate that a workflow is a true DAG (a non-empty in-degree remainder after
the queue drains means a cycle exists) and to group nodes into "levels" --
each level's nodes have no dependency on each other and are executed
concurrently via a thread pool, giving real (measurable) parallel speed-up.

Conditional `branch` nodes support short-circuit skipping: only the edge
matching the branch's boolean decision is "active"; a node with no active
inbound edge is marked `skipped` and does not execute, mirroring Airflow-style
branch operators.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .models import (
    AggregateNode,
    BranchNode,
    Edge,
    Err,
    FilterNode,
    MapNode,
    NodeConfig,
    NodeId,
    Ok,
    SourceNode,
    WorkflowDefinition,
    assert_never,
)


class CycleError(Exception):
    def __init__(self, remaining: list[NodeId]):
        self.remaining = remaining
        super().__init__(f"cycle detected among nodes: {remaining}")


class ValidationError(Exception):
    pass


@dataclass
class NodeEvent:
    node_id: str
    status: str  # started | succeeded | failed | skipped
    timestamp: float
    duration_ms: float = 0.0
    output: Any = None
    error: str | None = None


@dataclass
class WorkflowRun:
    levels: list[list[str]]
    events: list[NodeEvent] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    total_duration_ms: float = 0.0


def validate_workflow(wf: WorkflowDefinition) -> None:
    ids = [n.id for n in wf.nodes]
    if len(ids) != len(set(ids)):
        raise ValidationError("duplicate node ids found")
    id_set = set(ids)
    for e in wf.edges:
        if e.source not in id_set:
            raise ValidationError(f"edge references unknown source node '{e.source}'")
        if e.target not in id_set:
            raise ValidationError(f"edge references unknown target node '{e.target}'")


def topological_levels(nodes: list[NodeConfig], edges: list[Edge]) -> list[list[NodeId]]:
    """Group nodes into parallel-executable levels via Kahn's algorithm."""
    node_ids = [n.id for n in nodes]
    in_degree: dict[NodeId, int] = {nid: 0 for nid in node_ids}
    adjacency: dict[NodeId, list[NodeId]] = defaultdict(list)
    for e in edges:
        adjacency[e.source].append(e.target)
        in_degree[e.target] += 1

    frontier = deque(sorted(nid for nid, deg in in_degree.items() if deg == 0))
    remaining_in_degree = dict(in_degree)
    levels: list[list[NodeId]] = []
    processed = 0

    while frontier:
        level = sorted(frontier)
        levels.append(level)
        frontier.clear()
        for nid in level:
            processed += 1
            for neighbor in adjacency[nid]:
                remaining_in_degree[neighbor] -= 1
                if remaining_in_degree[neighbor] == 0:
                    frontier.append(neighbor)

    if processed != len(node_ids):
        remaining = sorted(nid for nid, deg in remaining_in_degree.items() if deg > 0)
        raise CycleError(remaining)

    return levels


def _run_node(node: NodeConfig, inputs: list[Any]) -> Ok[Any] | Err:
    try:
        if isinstance(node, SourceNode):
            return Ok(node.value)
        elif isinstance(node, MapNode):
            v = inputs[0]
            if node.op == "add":
                return Ok(v + node.operand)
            elif node.op == "multiply":
                return Ok(v * node.operand)
            elif node.op == "upper":
                return Ok(str(v).upper())
            elif node.op == "negate":
                return Ok(-v)
            else:
                assert_never(node.op)
        elif isinstance(node, FilterNode):
            items = inputs[0]
            if node.op == "gt":
                return Ok([x for x in items if x > node.threshold])
            elif node.op == "lt":
                return Ok([x for x in items if x < node.threshold])
            elif node.op == "even":
                return Ok([x for x in items if x % 2 == 0])
            elif node.op == "odd":
                return Ok([x for x in items if x % 2 != 0])
            else:
                assert_never(node.op)
        elif isinstance(node, BranchNode):
            # Passes its input through unchanged; the routing decision (which
            # outgoing edge is "live") is computed separately by the engine
            # via `_branch_decision` so downstream nodes still receive the
            # original value rather than the boolean decision itself.
            return Ok(inputs[0])
        elif isinstance(node, AggregateNode):
            # A single upstream edge carrying a list (e.g. from a `filter`
            # node) is treated as the collection to aggregate; multiple
            # upstream edges are treated as the collection themselves.
            values = inputs[0] if len(inputs) == 1 and isinstance(inputs[0], list) else inputs
            if node.op == "sum":
                return Ok(sum(values))
            elif node.op == "average":
                return Ok(sum(values) / len(values) if values else 0)
            elif node.op == "concat":
                flat = [x for sub in inputs for x in (sub if isinstance(sub, list) else [sub])]
                return Ok(flat)
            else:
                assert_never(node.op)
        else:
            assert_never(node)
    except Exception as exc:  # noqa: BLE001 - node ops are user-configured
        return Err(str(exc))


def _branch_decision(node: BranchNode, v: Any) -> bool:
    if node.op == "gt":
        return v > node.threshold
    elif node.op == "lt":
        return v < node.threshold
    elif node.op == "eq":
        return v == node.threshold
    else:
        assert_never(node.op)


def _execute_single(node: NodeConfig, inputs: list[Any], simulate_latency: bool) -> tuple[Ok[Any] | Err, float]:
    t0 = time.perf_counter()
    if simulate_latency:
        time.sleep(0.05)
    result = _run_node(node, inputs)
    duration_ms = (time.perf_counter() - t0) * 1000
    return result, duration_ms


def execute_workflow(wf: WorkflowDefinition, simulate_latency: bool = True) -> WorkflowRun:
    validate_workflow(wf)
    nodes_by_id = {n.id: n for n in wf.nodes}
    levels = topological_levels(wf.nodes, wf.edges)

    incoming: dict[NodeId, list[Edge]] = defaultdict(list)
    for e in wf.edges:
        incoming[e.target].append(e)

    outputs: dict[NodeId, Any] = {}
    active: dict[NodeId, bool] = {}
    branch_decisions: dict[NodeId, bool] = {}
    events: list[NodeEvent] = []
    start_all = time.perf_counter()

    def edge_is_live(e: Edge) -> bool:
        if not active.get(e.source, False):
            return False
        src_node = nodes_by_id[e.source]
        if isinstance(src_node, BranchNode) and e.label is not None:
            return branch_decisions.get(e.source) == (e.label == "true")
        return True

    for level in levels:
        pending: dict[NodeId, NodeConfig] = {}
        pending_inputs: dict[NodeId, list[Any]] = {}
        for nid in level:
            node = nodes_by_id[nid]
            in_edges = incoming[nid]
            is_active = (not in_edges) or any(edge_is_live(e) for e in in_edges)
            active[nid] = is_active
            if not is_active:
                events.append(
                    NodeEvent(node_id=nid, status="skipped", timestamp=time.perf_counter() - start_all)
                )
                continue
            pending[nid] = node
            pending_inputs[nid] = [outputs[e.source] for e in in_edges if edge_is_live(e) and e.source in outputs]

        if not pending:
            continue

        with ThreadPoolExecutor(max_workers=len(pending)) as pool:
            futures = {}
            for nid, node in pending.items():
                events.append(NodeEvent(node_id=nid, status="started", timestamp=time.perf_counter() - start_all))
                futures[nid] = pool.submit(_execute_single, node, pending_inputs[nid], simulate_latency)

            for nid, fut in futures.items():
                result, duration_ms = fut.result()
                if isinstance(result, Ok):
                    outputs[nid] = result.value
                    node = nodes_by_id[nid]
                    if isinstance(node, BranchNode):
                        branch_decisions[nid] = _branch_decision(node, pending_inputs[nid][0])
                    events.append(
                        NodeEvent(
                            node_id=nid,
                            status="succeeded",
                            timestamp=time.perf_counter() - start_all,
                            duration_ms=duration_ms,
                            output=result.value,
                        )
                    )
                else:
                    events.append(
                        NodeEvent(
                            node_id=nid,
                            status="failed",
                            timestamp=time.perf_counter() - start_all,
                            duration_ms=duration_ms,
                            error=result.error,
                        )
                    )

    total_duration_ms = (time.perf_counter() - start_all) * 1000
    return WorkflowRun(
        levels=[list(map(str, lvl)) for lvl in levels],
        events=events,
        outputs={str(k): v for k, v in outputs.items()},
        total_duration_ms=total_duration_ms,
    )
