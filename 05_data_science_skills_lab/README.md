# 05 — Data Science Skills Mastery Lab

An interactive lab that installs two public skill packs as a **catalog of 46
skills**, executes every one of them live against a real Kaggle dataset (Telco
Customer Churn, 7,043 subscribers), maps each skill to the CRISP-DM phase it
serves, and renders every result as a visual dashboard rather than raw JSON.

This implementation is an independent build. The **skill names, categories and
pack structure** mirror the public layout of
[`param087/agent-ml-skills`](https://github.com/param087/agent-ml-skills)
(15 skills) and
[`nimrodfisher/data-analytics-skills`](https://github.com/nimrodfisher/data-analytics-skills)
(31 unique skills across 6 categories), because "demonstrate every skill"
requires knowing which skills exist. Every executor, the result contract,
renderer, registry, interface, metrics, and reported results were implemented
specifically for this project.

## Dataset

The **Telco Customer Churn** table — the same 7,043-row, 21-column dataset
published on Kaggle, fetched from the identical IBM sample-data mirror so that
no Kaggle account is required and `git clone && pip install && run` stays
reproducible. `data/load_data.py` downloads and caches it; if the network is
unavailable it generates a seeded synthetic table with the same schema instead
and records that fact in `data/source.txt`, which the app displays. The lab
never presents synthetic rows as the real dataset.

## Setup

```bash
cd 05_data_science_skills_lab
python3.13 -m venv .venv        # 3.13 recommended; sklearn/pandas wheels
source .venv/bin/activate       # were not yet available for 3.14
pip install -r requirements.txt
```

## Run

```bash
# 1. Fetch (or regenerate) the dataset — runs automatically if missing
python data/load_data.py

# 2. Execute all 46 skills and write reports/skill_run_report.md
python src/run_all.py
python src/run_all.py --pack agent-ml-skills      # one pack
python src/run_all.py --skill model-evaluation    # one skill

# 3. Launch the lab UI
streamlit run app.py
```

Then open the printed local URL (default `http://localhost:8501`).

## What was built

- **`skills/registry.py`** — the catalog: all 46 skills with pack, category,
  CRISP-DM phase, one-line summary, and the executor that runs it.
- **`skills/ml_skills.py`** — 15 executors for the `agent-ml-skills` pack.
- **`skills/analytics_skills.py`** — 31 executors for the
  `data-analytics-skills` pack, across its 6 categories.
- **`skills/base.py`** — the `SkillResult` contract. Executors never return
  free-form dicts; they return typed presentation blocks (metric tiles, tables,
  charts, narrative, code, document artifacts). This is what makes the
  follow-up requirement structural rather than cosmetic — raw JSON cannot reach
  the screen because no executor produces any.
- **`skills/render.py`** — renders those blocks as a Streamlit dashboard with
  Altair charts, metric tiles and downloadable artifacts.
- **`src/data_prep.py` / `src/model.py`** — one shared cleaning + feature layer
  and one leakage-safe pipeline, so all 46 skills report mutually consistent
  numbers instead of 46 private re-cleanings.
- **`src/run_all.py`** — CLI batch runner producing
  `reports/skill_run_report.md` and `reports/skill_run_summary.json`.
- **`app.py`** — four tabs: **Skills Lab** (filter by pack/category/phase,
  execute any skill live), **CRISP-DM** (phase map built from the registry),
  **Live inference** (score a customer through the same fitted pipeline),
  **Batch run** (execute all 46 with status and timings).
- **`CRISP_DM.md`** — the six phases mapped to this project's actual decisions.

## Honest results

**Skill execution.** 46/46 skills execute successfully in **5.8 s** total
(`python src/run_all.py`, and the same batch run from inside the app). The
slowest is `hyperparameter-tuning` at 3.5 s; every other skill is under 0.6 s.

**Churn model** — logistic regression, 5,282 train / 1,761 held-out test rows,
`random_state=42`:

| Metric | Value |
|---|---|
| ROC-AUC | 0.849 |
| PR-AUC | 0.654 |
| Accuracy | 0.798 |
| Precision | 0.654 |
| Recall | 0.505 |
| F1 (@0.50) | 0.570 |
| Brier | 0.135 |

These are real numbers on the real dataset, and they are unremarkable by
design — 0.85 ROC-AUC is roughly what a well-specified linear model gets on
Telco churn, and the lab's point is the *practice*, not a leaderboard score.
Two findings worth stating because they cut against expectation: grid search
over 8 configurations moves CV ROC-AUC by less than 0.002, while moving the
decision threshold from 0.50 to 0.30 lifts F1 from 0.570 to 0.634. The seven
engineered features add +0.0014 CV ROC-AUC — reported as-is rather than
inflated.

### Limitations, stated plainly

- **Three skills are demonstrated in scope-reduced form, and say so on screen.**
  `pytorch-training-loop` implements and runs the canonical
  forward/loss/backward/step loop in NumPy with analytic gradients — real
  convergence, real per-epoch train/val logging, but no `torch`, autograd or
  GPU. `llm-finetuning` builds and profiles a genuine 800-example SFT corpus
  from real customer rows and shows the LoRA config it is formatted for, but
  **no fine-tuning run was executed** and no loss curve is claimed.
  `rag-pipeline` performs real chunking, TF-IDF vectorization, cosine
  retrieval and citation over the lab's computed findings, but its generation
  step is extractive rather than an LLM call.
- **The A/B test skill analyses observational data.** The two-proportion z-test
  is computed correctly, but customers chose their own billing method, so the
  result is confounded with contract type. The skill says this in-line rather
  than implying causality.
- **Campaign economics are placeholders.** `impact-quantification` uses an
  assumed $25 offer cost and 30% save rate; those are not measured from this
  dataset, every dollar figure scales linearly with them, and the skill states
  this. The defensible output is the shape of the net-value curve, not the
  amounts.
- **Cohort and time-series dates are reconstructed.** The dataset has no signup
  timestamps, so calendar months are derived from `tenure` against a fixed
  anchor. The *shape* is real; the calendar labels are relative. The downward
  churn trend across older cohorts is survivorship, not improvement.
- **The validation split is random, not temporal**, because no event timestamps
  exist to split on. This is recorded as a standing concern in the project's own
  peer review.
- `segmentation-analysis` fixes k=4 for interpretability rather than selecting
  it by silhouette or elbow analysis.

## Validation performed

- `python src/run_all.py` — **46/46 skills succeed**, report written.
- App launched with `streamlit run app.py`; health endpoint returns `ok`,
  page returns HTTP 200.
- The app was driven headlessly with Streamlit's own `AppTest` runner: initial
  render raises no exception; **Execute skill live** renders dashboards for
  skills from both packs (`exploratory-data-analysis`, `model-evaluation`,
  `root-cause-investigation`, `executive-summary-generator` — including its
  artifact download button); the skill selector switches packs (15 ↔ 31
  skills); the live-inference sliders re-score correctly (tenure 5 → 2 months
  moved churn probability 70.4% → 71.9% at ~3 ms); and the in-app **Run all
  skills** button completed 46/46 in 5.8 s with 0 failures.
- Reproducibility was verified by the lab itself: two independent fits of the
  same pipeline produce byte-identical held-out ROC-AUC.

## Files

```
05_data_science_skills_lab/
├── README.md
├── CRISP_DM.md
├── requirements.txt
├── app.py                       Streamlit UI (4 tabs)
├── data/
│   ├── load_data.py             download + cache, seeded synthetic fallback
│   ├── telco_churn.csv          cached dataset (7,043 rows)
│   └── source.txt               provenance of the cached copy
├── skills/
│   ├── base.py                  SkillResult / Block / Skill contracts
│   ├── registry.py              the 46-skill catalog
│   ├── ml_skills.py             15 agent-ml-skills executors
│   ├── analytics_skills.py      31 data-analytics-skills executors
│   └── render.py                blocks -> Streamlit dashboard
├── src/
│   ├── data_prep.py             shared cleaning + feature engineering
│   ├── model.py                 leakage-safe pipeline, shared metrics
│   └── run_all.py               batch runner + report writer
└── reports/                     generated by src/run_all.py
    ├── skill_run_report.md
    └── skill_run_summary.json
```
