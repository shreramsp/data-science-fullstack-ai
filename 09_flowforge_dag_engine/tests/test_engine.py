import random

import pytest

from src.engine import CycleError, execute_workflow, topological_levels
from src.models import Edge, MapNode, SourceNode, WorkflowDefinition
from src.sample_workflows import (
    conditional_branch,
    diamond_parallel,
    filter_then_aggregate,
    invalid_cycle,
    linear_pipeline,
)


def test_linear_pipeline_produces_one_node_per_level_in_order():
    wf = linear_pipeline()
    levels = topological_levels(wf.nodes, wf.edges)
    assert levels == [["n1"], ["n2"], ["n3"]]

    run = execute_workflow(wf, simulate_latency=False)
    assert run.outputs["n3"] == (10 + 5) * 2


def test_diamond_parallel_groups_independent_nodes_in_same_level():
    wf = diamond_parallel()
    levels = topological_levels(wf.nodes, wf.edges)
    assert levels == [["src"], ["left", "right"], ["join"]]

    run = execute_workflow(wf, simulate_latency=False)
    # src=8 -> left = 8+2=10, right = 8*3=24 -> join = sum = 34
    assert run.outputs["join"] == 34


def test_cycle_detection_raises_with_remaining_nodes():
    wf = invalid_cycle()
    with pytest.raises(CycleError) as exc_info:
        topological_levels(wf.nodes, wf.edges)
    assert set(exc_info.value.remaining) == {"a", "b", "c"}


def test_branch_skips_non_taken_path():
    wf = conditional_branch()
    run = execute_workflow(wf, simulate_latency=False)

    # src=42 > 10 -> true branch taken: big_path runs, small_path is skipped.
    statuses = {e.node_id: e.status for e in run.events if e.status in ("succeeded", "skipped")}
    assert statuses["big_path"] == "succeeded"
    assert statuses["small_path"] == "skipped"
    # aggregate should only see the live branch's output (42 * 10 = 420)
    assert run.outputs["join"] == 420


def test_filter_then_aggregate_keeps_only_matching_elements():
    wf = filter_then_aggregate()
    run = execute_workflow(wf, simulate_latency=False)
    assert run.outputs["evens"] == [2, 4, 6, 8, 10]
    assert run.outputs["total"] == 30


def test_random_dag_topological_order_is_always_valid():
    """Generate random acyclic graphs and assert every edge respects level order."""
    rng = random.Random(42)
    for _ in range(20):
        n = rng.randint(3, 12)
        nodes = [MapNode(id=f"m{i}", op="add", operand=1) for i in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.3:
                    edges.append(Edge(source=f"m{i}", target=f"m{j}"))
        levels = topological_levels(nodes, edges)
        level_of = {nid: lvl_idx for lvl_idx, level in enumerate(levels) for nid in level}
        for e in edges:
            assert level_of[e.source] < level_of[e.target]


def test_duplicate_node_ids_are_rejected():
    wf = WorkflowDefinition(
        id="dup",
        nodes=[SourceNode(id="x", value=1), SourceNode(id="x", value=2)],
        edges=[],
    )
    with pytest.raises(Exception):
        execute_workflow(wf, simulate_latency=False)
