# CRISP-DM — Autonomous Anomaly Detection Platform

The six phases mapped to the concrete decisions and code in this project.

## 1. Business Understanding

**Problem.** A payments operation receives a high-volume transaction stream in
which a very small fraction is fraudulent. Labels arrive late (chargebacks take
weeks) or not at all, so the detector must work **unsupervised** — it ranks
transactions by how unusual they are, and does not learn "what fraud looks like"
from labelled fraud.

**The real constraint is analyst capacity.** A fraud team can review a fixed
number of alerts per day, so the deliverable is not a classifier but a *ranking*
plus a defensible cut point. Success is therefore measured as *precision and
recall at a fixed alert budget* (0.5% of traffic here), alongside average
precision over the whole ranking.

**Why average precision is the objective.** At ~0.5% prevalence, ROC-AUC is
dominated by the enormous negative class and stays high even for weak detectors
(this run: every one of the five methods scores between 0.93 and 0.96 test
ROC-AUC while their average precision spans 0.13 to 0.27 — a 2x spread the ROC
number hides completely). Davis & Goadrich (ICML 2006) and Saito & Rehmsmeier
(PLOS ONE 2015) are the standard references for preferring PR over ROC in this
regime. ROC-AUC is still reported everywhere; it just does not steer decisions.

## 2. Data Understanding

Target dataset: the ULB **Credit Card Fraud Detection** benchmark (Dal Pozzolo
et al., IEEE CIDM 2015) — 284,807 transactions, 492 frauds, features `V1..V28`
being PCA components of the original (confidential) fields, plus `Time`,
`Amount`, `Class`.

It is ~150 MB behind a Kaggle login, so `data/load_data.py` resolves in order:
the real `data/creditcard.csv` if present, otherwise a **seeded synthetic
stand-in** with the same schema and the same *shape* of problem — uncorrelated
PCA-style components with decaying variance, heavy-tailed legitimate outliers,
extreme imbalance, and a minority class of deliberately heterogeneous difficulty
(some blatant, most marginal). Which source was used is stamped into
`artifacts/metrics.json` and shown in the dashboard header.

`single_feature_auc()` in `src/train.py` produces the per-feature diagnostic:
how well each feature separates the classes *alone*. It carries a caveat the
dashboard states explicitly — single-feature AUC only sees one-sided shifts, so
components whose anomalies sit in both tails score near 0.5 despite being
informative jointly.

## 3. Data Preparation

`src/features.py`, all rules reported in `artifacts/cleaning_report.csv`:

- **Exact duplicates dropped** — re-submitted/settlement echoes would
  double-count in both training and scoring.
- **Negative amounts sign-corrected** (refund artefacts) rather than dropped.
- **Missing amounts median-imputed** rather than dropped — dropping rows risks
  biasing away the minority class, which is the entire signal.
- **Derived features**: `log_amount` (amounts are heavy-tailed), and
  `hour_sin`/`hour_cos` — a cyclical encoding so 23:00 and 00:00 are neighbours
  rather than opposite extremes.
- **Feature sets** are a *search dimension*, not a fixed choice: `components`,
  `components_amount`, `components_amount_time`.
- **Split**: stratified 60/20/20 train/validation/test. Stratification only
  guarantees each split holds enough anomalies for its metrics to mean anything.
  **Labels never fit a detector.**

## 4. Modeling

`src/detectors.py` wraps five classic detectors behind one interface — fit on
unlabelled data, emit a score where higher = more anomalous:

| Method | Source | Idea |
|---|---|---|
| Isolation Forest | Liu, Ting & Zhou (ICDM 2008) | Random-split isolation depth |
| Local Outlier Factor | Breunig et al. (SIGMOD 2000) | Local density ratio vs neighbours |
| One-Class SVM | Schölkopf et al. (Neural Comp. 2001) | Kernel support boundary |
| Elliptic Envelope (MCD) | Rousseeuw & Van Driessen (Technometrics 1999) | Robust Mahalanobis distance |
| PCA reconstruction error | Shyu et al. (2003) | Residual off the principal subspace |

One-Class SVM is fitted on a bounded 6,000-row subsample because the kernel
solver is roughly quadratic; the cap is a named constant and is documented
rather than hidden.

**Baseline sweep first.** All five run at default settings before any tuning, so
the autoresearch result has something to be better *than*.

**AutoResearch** (`src/autoresearch.py`) then hill-climbs the joint space of
*feature set × scaler × detection method × that method's hyperparameters* — the
method itself is a search dimension. Multi-restart steepest ascent (Russell &
Norvig, AIMA ch. 4): one move changes exactly one decision, take the best
improving move, restart randomly when stuck. Every evaluation is logged to
`artifacts/search_history.csv`, **including rejected moves**, because a
trajectory showing only improvements says nothing about the landscape.

## 5. Evaluation

- The search selects on validation average precision, so **validation numbers
  are optimistically biased** — they are reported as such.
- The **test split is scored exactly once**, after the search finishes.
- The **operating threshold is fixed on validation** at the alert budget and
  applied unchanged to test, which is how a threshold behaves in production.
- The dashboard's Evaluation tab derives the PR curve, ROC curve, score
  distribution, and threshold sweep directly from the saved test scores.
- The cost model in the threshold explorer uses *user-supplied* costs and is
  labelled a decision aid, not a measured saving.

## 6. Deployment

`src/train.py` saves the fitted scaler + detector + threshold to
`artifacts/best_pipeline.joblib`. The dashboard's **Live scoring** tab loads that
artefact and scores a real held-out transaction (editable amount and hour),
showing the score, its rank against test traffic, the alert decision, and the
ground-truth label — including when the model is wrong. Scope is local
deployment through Streamlit.
