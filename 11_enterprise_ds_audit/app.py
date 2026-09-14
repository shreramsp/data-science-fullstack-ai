"""Enterprise Data Science Audit Platform.

A Streamlit dashboard that statically audits every numbered project in this
repository for documentation completeness, reproducibility hygiene
(seed pinning), leakage risk (fit calls without a visible train/test split),
and basic security smells (secrets, eval/exec, shell=True, bare excepts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.report import run_audit, save_report  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
PORTFOLIO_ROOT = APP_DIR.parent
REPORT_PATH = APP_DIR / "artifacts" / "audit_report.json"

SEVERITY_COLOR = {
    "critical": "#8b0000", "high": "#c1440e", "medium": "#b8860b",
    "low": "#5b7a9d", "info": "#6b7280",
}

st.set_page_config(page_title="Enterprise DS Audit Platform", layout="wide")


@st.cache_data(show_spinner=False)
def _load_or_build(force: bool = False) -> dict:
    if not force and REPORT_PATH.exists():
        import json
        return json.loads(REPORT_PATH.read_text())
    report = run_audit(PORTFOLIO_ROOT)
    save_report(report, REPORT_PATH)
    return report


st.title("🛡️ Enterprise Data Science Audit Platform")
st.caption(
    "Static, heuristic governance audit of every numbered project in this portfolio — "
    "documentation completeness, reproducibility hygiene, leakage-risk flags, and "
    "security smells. Findings are risk flags for human review, not proof of defects."
)

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Re-run audit now", use_container_width=True):
        st.cache_data.clear()
        report = _load_or_build(force=True)
        st.success("Audit re-run complete.")
    else:
        report = _load_or_build(force=False)
    st.caption(f"Report generated: {report['generated_at']}")
    st.divider()
    st.subheader("Methodology")
    st.markdown(
        "- **Docs**: README.md / requirements.txt presence\n"
        "- **Leakage risk**: `.fit()`/`.fit_transform()` calls with no "
        "`train_test_split`/`TimeSeriesSplit`/`KFold` import in the project\n"
        "- **Reproducibility**: fit calls with no `random_state`/seed usage\n"
        "- **Security**: `eval`/`exec`, `subprocess(shell=True)`, hardcoded "
        "secret-shaped literals, bare `except:`\n"
        "- Scoring starts at 100 per project; deductions: critical −15, "
        "high −8, medium −4, low −1."
    )

projects = report["projects"]
df = pd.DataFrame([
    {
        "Project": p["name"],
        "Title": p["title"],
        "Stack": p["stack"],
        "Score": p["score"],
        "Grade": p["grade"],
        "Python files": p["python_files"],
        "LOC": p["total_loc"],
        "Critical": sum(1 for f in p["findings"] if f["severity"] == "critical"),
        "High": sum(1 for f in p["findings"] if f["severity"] == "high"),
        "Medium": sum(1 for f in p["findings"] if f["severity"] == "medium"),
        "Low": sum(1 for f in p["findings"] if f["severity"] == "low"),
    }
    for p in projects
])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Portfolio score", f"{report['portfolio_score']}/100")
col2.metric("Portfolio grade", report["portfolio_grade"])
col3.metric("Projects audited", len(projects))
col4.metric("Critical findings", report["severity_totals"]["critical"])
col5.metric("High findings", report["severity_totals"]["high"])

st.subheader("Portfolio scorecard")
st.bar_chart(df.set_index("Project")["Score"])
st.dataframe(
    df.sort_values("Project"),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Project detail")
selected = st.selectbox(
    "Choose a project to inspect",
    options=[p["name"] for p in projects],
    format_func=lambda name: f"{name} — {next(p['title'] for p in projects if p['name'] == name)}",
)
proj = next(p for p in projects if p["name"] == selected)

d1, d2, d3, d4 = st.columns(4)
d1.metric("Score", f"{proj['score']}/100 ({proj['grade']})")
d2.metric("Python files", proj["python_files"])
d3.metric("Lines of code", proj["total_loc"])
d4.metric("Fit-style calls", proj["fit_calls"])

st.markdown(
    f"- README present: {'✅' if proj['has_readme'] else '❌'}\n"
    f"- requirements/pyproject present: {'✅' if proj['has_requirements'] else '❌'}\n"
    f"- Train/test split import detected: {'✅' if proj['has_split_import'] else '❌'}\n"
    f"- Seed/random_state usage detected: {'✅' if proj['has_seed_usage'] else '❌'}"
)

if proj["findings"]:
    fdf = pd.DataFrame(proj["findings"])
    fdf["severity"] = fdf["severity"].str.upper()
    st.dataframe(
        fdf[["severity", "category", "message", "file", "line"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.success("No findings for this project.")

st.download_button(
    "⬇️ Download full audit report (JSON)",
    data=REPORT_PATH.read_text(),
    file_name="audit_report.json",
    mime="application/json",
)
