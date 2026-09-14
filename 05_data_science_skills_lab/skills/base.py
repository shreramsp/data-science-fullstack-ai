"""The result contract every skill executor returns.

The follow-up requirement for this project was that running a skill live must
not dump raw JSON on screen. So executors never return free-form dicts: they
return a ``SkillResult`` made of typed presentation blocks (metric tiles,
tables, charts, narrative, code), which `skills/render.py` turns into a
dashboard and `src/run_all.py` turns into a Markdown report. The same result
object therefore drives both the UI and the batch report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

BlockKind = Literal["metrics", "table", "chart", "narrative", "code", "artifact"]


@dataclass
class Block:
    kind: BlockKind
    title: str = ""
    payload: Any = None
    meta: dict = field(default_factory=dict)


@dataclass
class Metric:
    label: str
    value: str
    help: str = ""


@dataclass
class SkillResult:
    headline: str
    blocks: list[Block] = field(default_factory=list)

    # ---- builder helpers (keep executors short and uniform) ----
    def metrics(self, *metrics: Metric, title: str = "") -> "SkillResult":
        self.blocks.append(Block("metrics", title, list(metrics)))
        return self

    def table(self, df: pd.DataFrame, title: str = "", note: str = "") -> "SkillResult":
        self.blocks.append(Block("table", title, df, {"note": note}))
        return self

    def chart(
        self,
        df: pd.DataFrame,
        *,
        kind: Literal["bar", "line", "hbar", "scatter", "area"] = "bar",
        x: str,
        y: str,
        color: str | None = None,
        title: str = "",
        note: str = "",
        sort: str | None = None,
    ) -> "SkillResult":
        self.blocks.append(
            Block("chart", title, df,
                  {"kind": kind, "x": x, "y": y, "color": color,
                   "note": note, "sort": sort})
        )
        return self

    def narrative(self, markdown: str, title: str = "") -> "SkillResult":
        self.blocks.append(Block("narrative", title, markdown.strip()))
        return self

    def code(self, snippet: str, title: str = "", language: str = "python") -> "SkillResult":
        self.blocks.append(Block("code", title, snippet.strip(), {"language": language}))
        return self

    def artifact(self, markdown: str, filename: str, title: str = "") -> "SkillResult":
        """A document deliverable (spec, memo, catalog entry) generated from real data."""
        self.blocks.append(
            Block("artifact", title or filename, markdown.strip(), {"filename": filename})
        )
        return self


@dataclass
class Skill:
    id: str
    name: str
    pack: str            # "agent-ml-skills" | "data-analytics-skills"
    category: str
    crisp_dm: str        # CRISP-DM phase this skill serves
    summary: str
    run: Any             # Callable[[dict], SkillResult]


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
