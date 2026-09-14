# 10 — CRISP-DM Master's Data Science Platform

A textbook-quality tour of the full CRISP-DM lifecycle — Business
Understanding through Deployment — on one synthetic retail-transactions
dataset, exercising **all five** required techniques (clustering, anomaly
detection, supervised learning, association rule mining, sub-linear/LSH
search) end to end, plus a short conceptual quiz per phase, served through a
Streamlit dashboard.

This implementation uses an original synthetic dataset. A Kaggle-style
dataset was replaced with a small reproducible
generator (see "Approach" below) so every modeling phase has a guaranteed,
inspectable signal to find.

## What was built

| File | Role |
|---|---|
| `data/generate_data.py` | Seeded synthetic online-retail transaction generator (customers, invoices, products, baskets) — no download needed. |
| `src/features.py` | Cleaning + RFM/invoice feature engineering (CRISP-DM Data Preparation). |
| `src/clustering.py` | KMeans customer segmentation, model-selected by silhouette score. |
| `src/anomaly.py` | Isolation Forest invoice-level outlier detection. |
| `src/supervised.py` | RandomForest churn-risk classifier with a leakage-safe label/feature split. |
| `src/association.py` | Apriori + association rules (`mlxtend`) over purchase baskets. |
| `src/lsh.py` | Custom random-hyperplane (SimHash) LSH index, benchmarked against brute-force cosine search. |
| `src/quizzes.py` | Static conceptual quiz bank, one set per CRISP-DM sub-phase. |
| `src/pipeline.py` | Runs every phase once, writes all `artifacts/`. |
| `app.py` | Nine-page Streamlit dashboard (one CRISP-DM sub-phase per page + quiz), reads only from `artifacts/`. |
| `CRISP_DM.md` | The six CRISP-DM phases mapped to this project's concrete decisions and honest results. |

## Setup

```bash
cd 10_crispdm_masters_curriculum
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3.13 data/generate_data.py   # regenerate the synthetic dataset (already checked in)
python3.13 src/pipeline.py         # <1s: runs all 5 techniques, writes artifacts/summary.json + tables
streamlit run app.py               # opens the dashboard at http://localhost:8501
```

`src/pipeline.py` must run at least once before `app.py` — the dashboard
reads everything from `artifacts/` and computes nothing itself.

## Approach

A single **synthetic** transaction log (`data/generate_data.py`, numpy,
seed=42) replaces a downloaded Kaggle CSV so every technique in the prompt
has a real, guaranteed signal at a small, inspectable scale:

- Five customer archetypes (champion/loyal/regular/at-risk/dormant) give RFM
  clustering genuine structure.
- ~1% of invoices are generated as anomalous bulk orders for Isolation
  Forest to recover.
- Two products per category are wired as reinforced "bundle" pairs so
  Apriori finds real high-lift rules.
- Real 300-product co-purchase embeddings are expanded into a 20K-item
  synthetic corpus specifically so the LSH-vs-brute-force speedup is visible
  at a realistic catalog scale (see `CRISP_DM.md` §4e for why).

Every number shown in the dashboard is read from `artifacts/summary.json`
and companion CSVs, produced by one deterministic run of `src/pipeline.py` —
the app performs no computation of its own, so app and pipeline cannot
silently disagree.

## Honest results (from the checked-in `artifacts/`, reproducible bit-for-bit)

| Technique | Result |
|---|---|
| Clustering | best silhouette 0.500 at k=2 (silhouette curve for k=2..6 shown in-app) |
| Anomaly detection | 30/1,500 invoices flagged (2.0%), matching the ~1% injection rate |
| Supervised (churn-risk) | ROC-AUC 0.783, accuracy 0.689, F1 0.680 on a 106-customer held-out set |
| Association rules | 82 rules at lift ≥ 1.5, top lift ≈ 35 |
| LSH search | 8.2x faster than brute force at recall@5 = 0.64, scanning ~4.5% of a 20,100-item corpus |

See `CRISP_DM.md` for the full phase-by-phase writeup, including honest
limitations (k=2's coarseness, the LSH corpus being a synthetic expansion,
no real seasonality in the data).

## Known limitations

- Dataset is synthetic, not a real Kaggle download — a deliberate trade-off
  for full reproducibility and guaranteed per-technique signal at minimum
  scope; nothing here claims to be real transaction data.
- Authentication, a database, and deployment infrastructure are outside the
  current scope.
- Recall@5 of 0.64 for LSH is a tunable trade-off (more hash tables raise
  recall, lower speedup); the shipped parameters were chosen from a small
  sweep documented in `CRISP_DM.md`, not the only valid choice.
