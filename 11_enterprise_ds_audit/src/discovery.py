"""Locate audited project directories and their source files."""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_DIR_PATTERN = re.compile(r"^\d{2}_")

EXCLUDE_DIR_NAMES = {
    ".venv", "venv", "__pycache__", "node_modules", ".git", "dist", "build",
    ".next", ".cache", ".pytest_cache", ".wrangler", "coverage", "egg-info",
}

CODE_EXTENSIONS = {".py"}
OTHER_TEXT_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}


def discover_projects(root: Path) -> list[Path]:
    """Return every top-level `NN_name` project directory under root, sorted by number."""
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and PROJECT_DIR_PATTERN.match(p.name)),
        key=lambda p: p.name,
    )


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_NAMES for part in path.parts)


def iter_python_files(project_dir: Path):
    for path in project_dir.rglob("*.py"):
        if path.is_file() and not _is_excluded(path):
            yield path


def iter_other_source_files(project_dir: Path):
    for ext in OTHER_TEXT_EXTENSIONS:
        for path in project_dir.rglob(f"*{ext}"):
            if path.is_file() and not _is_excluded(path):
                yield path


def detect_stack(project_dir: Path) -> str:
    if (project_dir / "package.json").exists() and not any(iter_python_files(project_dir)):
        return "typescript"
    return "python"
