# 07 — AutoGluon Multi-Layer Stacking Platform

AutoML illustrated with [AutoGluon](https://auto.gluon.ai/)'s `TabularPredictor`
across three task types (binary classification, multiclass classification,
regression), built end to end under CRISP-DM: clean, small benchmark data →
baseline fit → an **AutoResearch hill climb** over AutoGluon's own bagging +
stacking architecture → one honest held-out evaluation per task → a
Streamlit **data-science admin dashboard** that serves the saved predictors
live.

This implementation uses an original AutoML pipeline, interface,
documentation, metrics, and reported results.

## What was built

| File | Role |
|---|---|
| `data/load_data.py` | Task registry: three scikit-learn built-in datasets, one per problem type, no download needed. |
| `src/automl.py` | Fixed AutoGluon fit recipe (CatBoost + Random Forest + Extra Trees) parameterized by bagging/stacking depth. |
| `src/autoresearch.py` | Steepest-ascent hill climbing over `(num_bag_folds, num_stack_levels)`; `LITERATURE` table backing every design decision. |
| `src/train.py` | The CRISP-DM run per task: split → baseline → AutoResearch → best-config refit → one test evaluation → artifacts. |
| `app.py` | Six-tab Streamlit admin dashboard: Overview, Data, Methods, AutoResearch, Evaluation, Live Scoring. |
| `CRISP_DM.md` | The six phases mapped to this project's concrete decisions and honest results. |

## Setup

```bash
cd 07_automl_autogluon
python3.13 -m venv .venv        # AutoGluon 1.6.1 supports Python 3.13
source .venv/bin/activate
pip install -r requirements.txt
```

LightGBM and XGBoost are intentionally left out of the model pool so this
install step is sufficient on a clean checkout — see "Known limitations"
below.

## Run

```bash
python src/train.py     # ~2-3 minutes: fits + searches + evaluates all 3 tasks
streamlit run app.py    # opens the dashboard at http://localhost:8501
```

`src/train.py` must run at least once before `app.py` — the dashboard reads
its output from `artifacts/` and, for the Live Scoring tab, loads the fitted
predictors from `models/` (both regenerated bit-identically from the fixed
seed; `models/` is gitignored because a fitted, bagged, multi-layer predictor
for one task alone runs 3-100MB).

## Approach

Same recipe on every task, only the architecture depth changes:

1. **Split once**: 80% train/search, 20% held-out test (stratified for the
   two classification tasks). The test split is touched exactly once, after
   the search is frozen.
2. **AutoResearch hill climb**: starting from a `(0, 0)` baseline (no
   bagging, no stacking), steepest-ascent search over `num_bag_folds ∈ {0, 2,
   3, 5, 8}` and `num_stack_levels ∈ {0, 1, 2}`, scored by AutoGluon's own
   out-of-fold validation score. Capped at 7 fits per task, 20s each.
3. **Final refit** at the winning configuration (45s budget), then one
   `predictor.evaluate()` call against the held-out test split.

The search space and the mechanics it's exercising (repeated k-fold bagging,
Wolpert-style stacked generalization, Caruana-style greedy weighted
ensembling) come from the AutoGluon-Tabular paper (Erickson et al., 2020,
arXiv:2003.06505) — see `CRISP_DM.md` and the dashboard's Methods tab for the
full literature table.

## Honest results

On these three small, clean benchmark tables, heavier bagging/stacking mostly
did **not** beat the plain baseline on AutoGluon's own validation score —
breast cancer picked up a small ~0.004 AUC improvement from 2-fold bagging,
while wine and housing both kept the unbagged, unstacked configuration. This
is reported as found, not re-run until a bigger number appeared; it's
consistent with bagging/stacking being a variance-reduction and capacity
argument that pays off more on larger or noisier data than a few hundred to a
few thousand rows. Held-out test metrics (from the checked-in run — rerun
`python src/train.py` to reproduce):

| Task | Metric | Value |
|---|---|---|
| Breast Cancer Diagnosis (binary) | ROC-AUC | 0.999 |
| Breast Cancer Diagnosis (binary) | Accuracy | 0.965 |
| Wine Cultivar Classification (multiclass) | Accuracy | 0.972 |
| California Housing Price (regression) | R² | 0.794 |
| California Housing Price (regression) | RMSE | 0.519 |

## Known limitations

- **LightGBM excluded**: needs the system `libomp` runtime, absent on a clean
  macOS/Python install (`brew install libomp` would fix it, but that's an
  extra system-package step this project avoids requiring).
- **XGBoost excluded**: its sdist needs `cmake` to build on Python 3.13/arm64,
  where no prebuilt wheel is published yet.
- Model pool is therefore CatBoost + Random Forest + Extra Trees only — still
  enough for genuine bagging and multi-layer stacking, but not AutoGluon's
  full default model zoo.
- California Housing is downsampled to a fixed 3,000-row seed (from 20,640)
  to keep bagged, multi-level fits laptop-fast; results are on that sample,
  not the full dataset.
- Search budget (7 evals × 20s, single restart) is small next to a
  production AutoGluon run; it is enough to show the hill climb moving and
  converging, not to claim an exhaustive architecture search.
