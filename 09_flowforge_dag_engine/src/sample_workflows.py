"""A handful of example workflows demonstrating each engine feature."""
from __future__ import annotations

from .models import (
    AggregateNode,
    BranchNode,
    Edge,
    FilterNode,
    MapNode,
    SourceNode,
    WorkflowDefinition,
)


def linear_pipeline() -> WorkflowDefinition:
    return WorkflowDefinition(
        id="linear-pipeline",
        name="Linear Pipeline",
        nodes=[
            SourceNode(id="n1", value=10),
            MapNode(id="n2", op="add", operand=5),
            MapNode(id="n3", op="multiply", operand=2),
        ],
        edges=[
            Edge(source="n1", target="n2"),
            Edge(source="n2", target="n3"),
        ],
    )


def diamond_parallel() -> WorkflowDefinition:
    return WorkflowDefinition(
        id="diamond-parallel",
        name="Diamond Parallel",
        nodes=[
            SourceNode(id="src", value=8),
            MapNode(id="left", op="add", operand=2),
            MapNode(id="right", op="multiply", operand=3),
            AggregateNode(id="join", op="sum"),
        ],
        edges=[
            Edge(source="src", target="left"),
            Edge(source="src", target="right"),
            Edge(source="left", target="join"),
            Edge(source="right", target="join"),
        ],
    )


def conditional_branch() -> WorkflowDefinition:
    return WorkflowDefinition(
        id="conditional-branch",
        name="Conditional Branch",
        nodes=[
            SourceNode(id="src", value=42),
            BranchNode(id="check", op="gt", threshold=10),
            MapNode(id="big_path", op="multiply", operand=10),
            MapNode(id="small_path", op="negate"),
            AggregateNode(id="join", op="sum"),
        ],
        edges=[
            Edge(source="src", target="check"),
            Edge(source="check", target="big_path", label="true"),
            Edge(source="check", target="small_path", label="false"),
            Edge(source="big_path", target="join"),
            Edge(source="small_path", target="join"),
        ],
    )


def filter_then_aggregate() -> WorkflowDefinition:
    return WorkflowDefinition(
        id="filter-aggregate",
        name="Filter and Aggregate",
        nodes=[
            SourceNode(id="src", value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            FilterNode(id="evens", op="even"),
            AggregateNode(id="total", op="sum"),
        ],
        edges=[
            Edge(source="src", target="evens"),
            Edge(source="evens", target="total"),
        ],
    )


def invalid_cycle() -> WorkflowDefinition:
    """A 3-node cycle: used to demonstrate cycle detection, not for execution."""
    return WorkflowDefinition(
        id="invalid-cycle",
        name="Invalid Cycle (demo)",
        nodes=[
            MapNode(id="a", op="add", operand=1),
            MapNode(id="b", op="add", operand=1),
            MapNode(id="c", op="add", operand=1),
        ],
        edges=[
            Edge(source="a", target="b"),
            Edge(source="b", target="c"),
            Edge(source="c", target="a"),
        ],
    )


SAMPLE_WORKFLOWS = {
    "Linear Pipeline": linear_pipeline(),
    "Diamond Parallel": diamond_parallel(),
    "Conditional Branch": conditional_branch(),
    "Filter and Aggregate": filter_then_aggregate(),
    "Invalid Cycle (demo)": invalid_cycle(),
}
