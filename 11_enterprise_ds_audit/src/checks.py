"""Static, heuristic-based data-science governance checks.

Every check here is a heuristic over source text/AST, not a formal proof.
Findings are phrased as risk flags for a human auditor to verify, not as
certain defects.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

SEVERITY_WEIGHT = {"critical": 15, "high": 8, "medium": 4, "low": 1, "info": 0}
SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]

SPLIT_NAMES = {
    "train_test_split", "TimeSeriesSplit", "KFold", "StratifiedKFold",
    "GroupKFold", "TimeSeriesSplitOOS",
}
SEED_CALL_NAMES = {"seed", "manual_seed", "set_seed", "random_seed"}
SEED_KEYWORDS = {"random_state", "seed"}
FIT_ATTRS = {"fit", "fit_transform", "fit_predict"}

_PLACEHOLDER_TOKENS = (
    "changeme", "your_api_key", "xxxx", "placeholder", "example",
    "dummy", "fake", "replace_me", "todo", "insert_key",
)

SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access-key-shaped literal"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "OpenAI/Anthropic-style secret-key-shaped literal"),
    (
        re.compile(r"(?i)\b(password|passwd|api[_-]?key|secret|token)\s*=\s*[\"']([^\"']{6,})[\"']"),
        "hardcoded credential-like assignment",
    ),
]


@dataclass
class Finding:
    severity: str
    category: str
    message: str
    file: str = ""
    line: int = 0

    def key(self):
        return (SEVERITY_ORDER.index(self.severity), self.category, self.file, self.line)


@dataclass
class FileStats:
    fit_calls: int = 0
    has_split: bool = False
    has_seed: bool = False
    bare_except: int = 0
    eval_exec: int = 0
    shell_true: int = 0
    loc: int = 0


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.stats = FileStats()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name.split(".")[-1] in SPLIT_NAMES:
                self.stats.has_split = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            if alias.name in SPLIT_NAMES:
                self.stats.has_split = True
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.type is None:
            self.stats.bare_except += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr

        if func_name in SPLIT_NAMES:
            self.stats.has_split = True
        if func_name in SEED_CALL_NAMES:
            self.stats.has_seed = True
        if func_name in FIT_ATTRS:
            self.stats.fit_calls += 1
        if func_name in {"eval", "exec"} and isinstance(func, ast.Name):
            self.stats.eval_exec += 1

        for kw in node.keywords:
            if kw.arg in SEED_KEYWORDS:
                self.stats.has_seed = True
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self.stats.shell_true += 1

        self.generic_visit(node)


def _scan_secrets(text: str, relpath: str) -> list[Finding]:
    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, label in SECRET_PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            literal = m.group(0).lower()
            if any(tok in literal for tok in _PLACEHOLDER_TOKENS):
                continue
            findings.append(Finding(
                severity="critical", category="hardcoded-secret",
                message=f"Possible {label}.", file=relpath, line=lineno,
            ))
    return findings


def audit_python_file(path: Path, project_root: Path) -> tuple[FileStats, list[Finding]]:
    relpath = str(path.relative_to(project_root))
    text = path.read_text(encoding="utf-8", errors="replace")
    findings = _scan_secrets(text, relpath)
    stats = FileStats(loc=len(text.splitlines()))

    try:
        tree = ast.parse(text, filename=relpath)
    except SyntaxError as exc:
        findings.append(Finding(
            severity="medium", category="parse-error",
            message=f"File could not be parsed as Python: {exc.msg}",
            file=relpath, line=exc.lineno or 0,
        ))
        return stats, findings

    visitor = _Visitor()
    visitor.visit(tree)
    v = visitor.stats
    v.loc = stats.loc
    stats = v

    if stats.eval_exec:
        findings.append(Finding(
            severity="high", category="dangerous-builtin",
            message=f"{stats.eval_exec} call(s) to eval()/exec() found.",
            file=relpath,
        ))
    if stats.shell_true:
        findings.append(Finding(
            severity="high", category="shell-injection-risk",
            message=f"{stats.shell_true} subprocess call(s) with shell=True found.",
            file=relpath,
        ))
    if stats.bare_except:
        findings.append(Finding(
            severity="low", category="bare-except",
            message=f"{stats.bare_except} bare `except:` clause(s) found (may hide real errors).",
            file=relpath,
        ))
    return stats, findings


def audit_other_source_file(path: Path, project_root: Path) -> list[Finding]:
    relpath = str(path.relative_to(project_root))
    text = path.read_text(encoding="utf-8", errors="replace")
    return _scan_secrets(text, relpath)
