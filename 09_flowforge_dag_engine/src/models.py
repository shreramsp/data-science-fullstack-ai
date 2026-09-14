"""Type-safe workflow data model.

Approximates the "Total TypeScript" patterns (branded nominal IDs, discriminated
unions, exhaustive `never` narrowing, `Result` types) in idiomatic Python:

- `NewType` gives `WorkflowId`/`NodeId` distinct static types (mypy catches a
  `WorkflowId` used where a `NodeId` is expected) even though both are `str`
  at runtime.
- Each node kind is its own Pydantic model with a `Literal["..."]` discriminator
  field, joined into `NodeConfig` via `Field(discriminator="type")` -- the
  Python analogue of a TypeScript discriminated union.
- `assert_never` is used in `engine.py`'s exhaustive `match` statements so an
  unhandled node/op variant is caught by static analysis (mypy) instead of at
  runtime.
- `Ok`/`Err` model node execution outcomes without relying on exceptions for
  expected failure paths, mirroring a `Result<T, E>` type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Generic, Literal, NewType, NoReturn, Optional, TypeVar, Union

from pydantic import BaseModel, Field

WorkflowId = NewType("WorkflowId", str)
NodeId = NewType("NodeId", str)

T = TypeVar("T")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err:
    error: str


Result = Union[Ok[T], Err]


def assert_never(x: NoReturn) -> NoReturn:
    """Exhaustiveness guard: reaching this call means a variant was unhandled."""
    raise AssertionError(f"Unhandled variant: {x!r}")


class SourceNode(BaseModel):
    id: NodeId
    type: Literal["source"] = "source"
    value: Any


class MapNode(BaseModel):
    id: NodeId
    type: Literal["map"] = "map"
    op: Literal["add", "multiply", "upper", "negate"]
    operand: Optional[float] = None


class FilterNode(BaseModel):
    id: NodeId
    type: Literal["filter"] = "filter"
    op: Literal["gt", "lt", "even", "odd"]
    threshold: Optional[float] = None


class BranchNode(BaseModel):
    id: NodeId
    type: Literal["branch"] = "branch"
    op: Literal["gt", "lt", "eq"]
    threshold: float


class AggregateNode(BaseModel):
    id: NodeId
    type: Literal["aggregate"] = "aggregate"
    op: Literal["sum", "concat", "average"]


NodeConfig = Annotated[
    Union[SourceNode, MapNode, FilterNode, BranchNode, AggregateNode],
    Field(discriminator="type"),
]


class Edge(BaseModel):
    source: NodeId
    target: NodeId
    label: Optional[Literal["true", "false"]] = None


class WorkflowDefinition(BaseModel):
    id: WorkflowId
    name: str = ""
    nodes: list[NodeConfig]
    edges: list[Edge]
