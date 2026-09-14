# CRISP-DM — AutoGluon Multi-Layer Stacking Platform

## 1. Business Understanding

Illustrate AutoML with AutoGluon across more than one kind of data science
task, using the same recipe each time: bag base models, stack a layer of
weighted ensembles on top, and let a hill-climbing search choose how deep the
architecture should go rather than fixing it by hand. The deliverable is a
dashboard a data scientist or ML engineer could use to inspect exactly what
AutoGluon built and why.

## 2. Data Understanding

Three tasks, one from each problem type AutoGluon's `TabularPredictor`
targets, all built into scikit-learn so there is no download step:

| Task | Type | Rows | Features | Source |
|---|---|---|---|---|
| Breast Cancer Diagnosis | binary | 569 | 30 | `sklearn.datasets.load_breast_cancer` |
| Wine Cultivar Classification | multiclass (3) | 178 | 13 | `sklearn.datasets.load_wine` |
| California Housing Price | regression | 3,000 (seeded sample of 20,640) | 8 | `sklearn.datasets.fetch_california_housing` |

All three are well-known, clean benchmark sets with no missing values, chosen
so that data cleaning is not the point of this project — the AutoML
architecture is.

## 3. Data Preparation

Each task is split once, up front: 80% train/search, 20% held-out test,
stratified for the two classification tasks. The 20% test split is never
touched by fitting or by the architecture search — it is scored exactly once,
at the end, by the final refit predictor. No manual feature engineering is
applied; AutoGluon's own preprocessing (imputation, categorical encoding)
runs inside `TabularPredictor.fit`.

## 4. Modeling

Model family is fixed across every fit: CatBoost, Random Forest, Extra Trees
(`src/automl.py`). LightGBM and XGBoost are deliberately left out — LightGBM
needs the system `libomp` runtime that a clean macOS/Python install lacks,
and XGBoost's sdist needs `cmake` to build on Python 3.13/arm64 where no
prebuilt wheel exists yet — so `pip install -r requirements.txt` stays
sufficient with no system package manager step.

What AutoResearch searches over (`src/autoresearch.py`) is the *architecture*,
not the model family:

- `num_bag_folds` ∈ {0, 2, 3, 5, 8} — bagging depth (Breiman, 1996)
- `num_stack_levels` ∈ {0, 1, 2} — stacking depth (Wolpert, 1992)

Every layer's models are additionally combined by AutoGluon's built-in
`WeightedEnsemble`, a greedy forward-selection ensembler (Caruana et al.,
2004). This whole recipe — repeated bagging plus multi-layer stacking — is
the one described in Erickson et al. (2020), *AutoGluon-Tabular*
(arXiv:2003.06505).

Search procedure: steepest-ascent hill climbing from the (0, 0) baseline,
capped at 7 evaluations per task, each fit capped at 20s. A move is accepted
only if it improves AutoGluon's own out-of-fold validation score
(`score_val`); the climb stops at a local optimum or the eval budget.

## 5. Evaluation

The winning `(num_bag_folds, num_stack_levels)` per task is refit once more
(45s time budget) and evaluated exactly once against the held-out test split.
Honest result: on all three of these small, clean benchmark tables, the
search's own out-of-fold score shows bagging/stacking either barely helps
(breast cancer: `folds=2` beats `folds=0` by ~0.004 AUC) or does not help at
all (wine and housing both keep the `(0, 0)` baseline) — consistent with the
literature's finding that bagging's variance reduction and stacking's
extra-layer capacity mostly pay off on larger, noisier data than a few
hundred to a few thousand rows. That is reported here rather than hidden or
re-run until a bigger number appeared; see `artifacts/*/metrics.json` for the
exact `score_val` at every step the climb visited.

Held-out test metrics per task (from the actual run — regenerate with
`python src/train.py` to reproduce bit-for-bit):

- **Breast Cancer** (binary): ROC-AUC 0.999, accuracy 0.965
- **Wine** (multiclass): accuracy 0.972, log-loss -0.049
- **California Housing** (regression): R² 0.794, RMSE 0.519

## 6. Deployment

`app.py`, a six-tab Streamlit dashboard (Overview, Data, Methods,
AutoResearch, Evaluation, Live Scoring), reads everything from
`artifacts/<task>/` — it computes no metric itself. The Live Scoring tab
additionally loads the actual fitted `TabularPredictor` from `models/<task>/`
to run real single-row inference; those model directories are regenerable
(`python src/train.py`) but gitignored, since a fitted, bagged, multi-layer
predictor for one task alone runs 3-100MB.
