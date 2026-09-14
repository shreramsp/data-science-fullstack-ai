# 11 — Enterprise Data Science Audit Platform

A static, heuristic governance auditor for this repository's own project
portfolio, served through a Streamlit dashboard. It scans every numbered
project directory (`00_...` through `15_...`, including this one) and scores
it on documentation completeness, reproducibility hygiene, leakage-risk
flags, and basic security smells.

This implementation uses an original auditing pipeline, interface,
documentation, metrics, and rule set.

## What was built

| File | Role |
|---|---|
| `src/discovery.py` | Finds every `NN_project_name` directory under the repo root and walks its source files, excluding `.venv`, `node_modules`, `__pycache__`, build output, etc. |
| `src/checks.py` | Per-file static checks: an `ast`-based visitor detects `.fit()`/`.fit_transform()` calls, `train_test_split`/`TimeSeriesSplit`/`KFold` imports, `random_state`/seed usage, bare `except:`, `eval`/`exec`, and `subprocess(shell=True)`; a regex pass flags secret-shaped literals (AWS-key-shaped, `sk-...`-shaped, hardcoded `password=`/`api_key=` assignments) in both Python and TypeScript/JS files. |
| `src/report.py` | Aggregates per-file findings into a per-project scorecard (structural checks + leakage-risk / reproducibility-risk flags derived from the aggregated stats), computes a 0–100 score and letter grade per project and for the portfolio as a whole, and serializes everything to `artifacts/audit_report.json`. |
| `app.py` | Streamlit dashboard: portfolio score/grade, a bar chart of per-project scores, a sortable scorecard table, a per-project drill-down with the full findings list, a methodology explainer, and a JSON report download. |

## Approach

This is a **static heuristic audit**, not a formal proof or a dynamic test
run of every project (most projects have their own heavy ML dependencies —
re-running all of them would be redundant with their own validation). Every
finding is phrased as a risk flag for a human reviewer, never as a
certainty:

- **Leakage-risk**: a project with `.fit()`-style calls but no
  `train_test_split`/`TimeSeriesSplit`/`KFold` import anywhere in its source
  is flagged for manual verification — this correctly does *not* penalize
  legitimate full-data fits (e.g. unsupervised clustering, anomaly
  detection) beyond a "please verify" flag.
- **Reproducibility-risk**: fit calls with no `random_state`/seed usage
  anywhere in the project are flagged low-severity.
- **Security**: `eval`/`exec`, `shell=True`, hardcoded secret-shaped
  literals (obvious placeholders like `changeme`/`your_api_key`/`xxxx` are
  excluded to cut false positives), and bare `except:` clauses.
- **Docs**: missing `README.md` or `requirements.txt`/`pyproject.toml`.

Scoring starts at 100 per project; deductions are critical −15, high −8,
medium −4, low −1 (floored at 0). The portfolio score is the mean of all
project scores. This project audits itself as project `11` in the same
pass.

## Honest results (as of this build)

Running the auditor against the full portfolio (16 projects, `00`–`15`)
currently produces:

- **Portfolio score: ~98–99 / 100 (A)**
- **0 critical, 0 high findings**
- A handful of medium "leakage-risk — please verify" flags on projects that
  fit unsupervised models (clustering/anomaly detection) or preprocessors
  without an explicit split import in that file — these are review flags,
  not confirmed leakage.
- No hardcoded secrets, no `eval`/`exec`, no `shell=True` detected anywhere
  in the portfolio.

Exact numbers will shift slightly as other projects are edited; re-run the
audit to get current numbers (see below).

## Setup

```bash
cd 11_enterprise_ds_audit
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Regenerate artifacts/audit_report.json from the current repo state
python3.13 -m src.report

# Launch the dashboard
streamlit run app.py
```

Open the printed local URL (default `http://localhost:8501`). Use the
sidebar "Re-run audit now" button to re-scan the repo live from within the
app, or select a project from the dropdown to see its full findings list.

## Limitations

- Heuristic, static-only: it does not execute any other project's code, so
  it cannot catch runtime-only issues (e.g. a scaler actually being fit on
  test data at runtime despite a `train_test_split` import existing in the
  file).
- Secret-detection is regex-based and pattern-limited; it is a smell test,
  not a substitute for a dedicated secret scanner.
- "Leakage-risk" is a single-file heuristic (import presence), not
  data-flow analysis — it can produce false positives/negatives.
