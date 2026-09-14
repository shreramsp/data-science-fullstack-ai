"""Aggregate per-project static-analysis findings into a portfolio audit report."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import discovery
from .checks import SEVERITY_WEIGHT, Finding, audit_other_source_file, audit_python_file

TITLES = {
    "00_dynamic_todo_workspace": "Dynamic Todo Workspace",
    "01_nyc_taxi_trip_prediction": "NYC Taxi Trip Duration & Fare Prediction",
    "02_nano_llm_transformer": "NanoLlama Autoregressive SFT LLM",
    "03_customer_segmentation_clustering": "Customer Intelligence & Segmentation Clustering",
    "04_associative_pattern_mining": "Market Basket Pattern Mining",
    "05_data_science_skills_lab": "Data Science Skills Mastery Lab",
    "06_anomaly_detection": "Autonomous Anomaly Detection Platform",
    "07_automl_autogluon": "AutoGluon Multi-Layer Stacking Platform",
    "08_datascience_visual_mastery": "Data Science Visual Foundations Curriculum",
    "09_flowforge_dag_engine": "FlowForge DAG Engine",
    "10_crispdm_masters_curriculum": "CRISP-DM Master's Data Science Platform",
    "11_enterprise_ds_audit": "Enterprise Data Science Audit Platform",
    "12_timeseries_forecasting": "Time Series Forecasting Engine",
    "13_crispdm_nyc_taxi_audit_platform": "CRISP-DM NYC TLC Audit Platform",
    "14_autogluon_multimodal_automl_suite": "AutoGluon Multimodal AutoML Suite",
    "15_spy_timeseries_sota_forecasting": "SPY SOTA Time Series Forecasting Platform",
}


def grade_for(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


@dataclass
class ProjectAudit:
    name: str
    title: str
    stack: str
    python_files: int
    other_files: int
    total_loc: int
    has_readme: bool
    has_requirements: bool
    fit_calls: int
    has_split_import: bool
    has_seed_usage: bool
    findings: list[Finding] = field(default_factory=list)
    score: float = 100.0
    grade: str = "A"

    def to_dict(self):
        d = asdict(self)
        d["findings"] = [asdict(f) for f in sorted(self.findings, key=lambda f: f.key())]
        return d


def _structural_findings(project_dir: Path, has_readme: bool, has_requirements: bool, stack: str) -> list[Finding]:
    findings = []
    if not has_readme:
        findings.append(Finding(severity="medium", category="missing-doc", message="No README.md found."))
    if stack == "python" and not has_requirements:
        findings.append(Finding(
            severity="medium", category="missing-doc",
            message="No requirements.txt or pyproject.toml found.",
        ))
    return findings


def audit_project(project_dir: Path) -> ProjectAudit:
    stack = discovery.detect_stack(project_dir)
    has_readme = (project_dir / "README.md").exists()
    has_requirements = (project_dir / "requirements.txt").exists() or (project_dir / "pyproject.toml").exists()

    findings = _structural_findings(project_dir, has_readme, has_requirements, stack)
    total_loc = 0
    fit_calls = 0
    has_split_import = False
    has_seed_usage = False
    py_count = 0
    other_count = 0

    for path in discovery.iter_python_files(project_dir):
        py_count += 1
        stats, file_findings = audit_python_file(path, project_dir)
        findings.extend(file_findings)
        total_loc += stats.loc
        fit_calls += stats.fit_calls
        has_split_import = has_split_import or stats.has_split
        has_seed_usage = has_seed_usage or stats.has_seed

    for path in discovery.iter_other_source_files(project_dir):
        other_count += 1
        findings.extend(audit_other_source_file(path, project_dir))

    if fit_calls > 0 and not has_split_import:
        findings.append(Finding(
            severity="medium", category="leakage-risk",
            message=(
                f"{fit_calls} .fit()/.fit_transform() call(s) detected but no "
                "train_test_split/TimeSeriesSplit/KFold import found in this project's "
                "source — verify preprocessing and models are fit only on training data."
            ),
        ))
    if fit_calls > 0 and not has_seed_usage:
        findings.append(Finding(
            severity="low", category="reproducibility-risk",
            message=(
                f"{fit_calls} .fit()-style call(s) detected but no random_state/seed "
                "usage found — results may not be exactly reproducible across runs."
            ),
        ))

    score = 100.0
    for f in findings:
        score -= SEVERITY_WEIGHT[f.severity]
    score = max(0.0, min(100.0, score))

    return ProjectAudit(
        name=project_dir.name,
        title=TITLES.get(project_dir.name, project_dir.name),
        stack=stack,
        python_files=py_count,
        other_files=other_count,
        total_loc=total_loc,
        has_readme=has_readme,
        has_requirements=has_requirements,
        fit_calls=fit_calls,
        has_split_import=has_split_import,
        has_seed_usage=has_seed_usage,
        findings=findings,
        score=round(score, 1),
        grade=grade_for(score),
    )


def run_audit(root: Path) -> dict:
    projects = discovery.discover_projects(root)
    audits = [audit_project(p) for p in projects]
    avg_score = round(sum(a.score for a in audits) / len(audits), 1) if audits else 0.0

    severity_totals = {sev: 0 for sev in SEVERITY_WEIGHT}
    for a in audits:
        for f in a.findings:
            severity_totals[f.severity] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "portfolio_score": avg_score,
        "portfolio_grade": grade_for(avg_score),
        "severity_totals": severity_totals,
        "projects": [a.to_dict() for a in audits],
    }


def save_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    report = run_audit(project_root)
    save_report(report, Path(__file__).resolve().parents[1] / "artifacts" / "audit_report.json")
    print(f"Portfolio score: {report['portfolio_score']} ({report['portfolio_grade']})")
    print(f"Projects audited: {len(report['projects'])}")
    print(f"Severity totals: {report['severity_totals']}")
