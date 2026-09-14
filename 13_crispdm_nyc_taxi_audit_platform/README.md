# 13 — CRISP-DM NYC TLC Audit Platform

An independent, end-to-end CRISP-DM implementation on a synthetic NYC
TLC-style trip dataset, built around a distinguishing feature: an
automated **audit report** (leakage check, split-integrity check,
reproducibility check, raw data-quality check) that a data-science or
code auditor could read without re-deriving the pipeline. A Streamlit
front end walks through all six CRISP-DM phases plus the audit report and
a live fare-estimator tab.

This focused build intentionally does **not** include a long-form paper,
separate TypeScript architecture, TreeSHAP force plots,
FastAPI + REST inference, load testing, or an MLOps demonstration. It keeps the
CRISP-DM-plus-audit-trail core: Python/pandas/scikit-learn/Streamlit,
honest held-out metrics, and a genuinely automated audit report.

### Why not the real NYC TLC trip record archive

The official TLC trip record files are tens of GB across parquet files
per month and require a separate download step, which breaks
reproducibility for a plain `git clone && pip install`. This project
instead generates a small, seeded, synthetic dataset
(`data/generate_data.py`, 8,000 base rows) shaped like a TLC trip record
(pickup/dropoff zone + coordinates, distance, fare components, payment
type), with a deliberate ~3% slice of injected data-quality defects so the
audit stage has real issues to catch. Metrics below describe fit quality
on **this synthetic data**, not the real TLC archive.

## What was built

- `data/generate_data.py` — seeded synthetic TLC-style trip generator
  (8,000 rows across 7 zone hotspots, with injected missing values,
  duplicates, out-of-bounds GPS, and negative fares).
- `src/features.py` — shared cleaning rules and feature engineering (no
  target leakage into features).
- `src/clustering.py` — K-Means (k=5) spatial clustering of pickup
  locations, profiled by fare/distance.
- `src/modeling.py` — trains and compares 4 regressors (Linear, Ridge,
  Random Forest, Gradient Boosting) on an identical 80/20 split; honest
  MAE/RMSE/R² on the held-out test split; permutation-importance
  explainability.
- `src/audit.py` — the platform's core: 5 automated audit checks (raw data
  quality, cleaning transparency, target-leakage check, train/test split
  integrity, reproducibility) that write `artifacts/audit_report.json`.
- `src/pipeline.py` — orchestrates the full CRISP-DM run and writes all
  artifacts.
- `app.py` — Streamlit app with 8 tabs: Business Understanding, Data
  Understanding, Data Preparation, Modeling & Evaluation, Explainability,
  Spatial Clustering, Audit Report, and a live Fare Estimator.
- `CRISP_DM.md` — the six CRISP-DM phases mapped to this project's
  concrete decisions and code.

## Setup

```bash
cd 13_crispdm_nyc_taxi_audit_platform
python3.13 -m venv .venv        # 3.13 recommended; sklearn/pandas wheels
source .venv/bin/activate       # not yet available for 3.14 at build time
pip install -r requirements.txt
```

## Run

```bash
# 1. Run the full CRISP-DM pipeline once from the command line
#    (also runs automatically, cached, the first time the app loads)
python -m src.pipeline

# 2. Launch the front end
streamlit run app.py
```

Then open the printed local URL (default `http://localhost:8501`).

## Honest results

From the last `python -m src.pipeline` run on the generated dataset
(8,040 raw rows → 7,900 after cleaning → 6,320 train / 1,580 test):

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Linear Regression | 4.91 | 6.91 | 0.941 |
| Ridge | 4.91 | 6.91 | 0.941 |
| Random Forest | 4.60 | 6.77 | 0.944 |
| **Gradient Boosting (best)** | **4.40** | **6.45** | **0.949** |

Permutation importance on the held-out set for the selected Gradient
Boosting model: `trip_distance_miles` dominates (importance ≈ 1.85 mean
R² drop when shuffled), followed distantly by `is_rush_hour` (≈ 0.07) and
`payment_type` (≈ 0.02) — consistent with the fare formula used to
generate the data, which is distance-driven with smaller time-of-day
effects.

Audit report highlights from the same run: raw data-quality check flagged
81 missing `passenger_count` rows, 81 missing `payment_type` rows, 40
duplicate rows, 80 out-of-NYC-bounds pickups, and 20 negative-fare rows —
all by design (injected defects) and all resolved or explained by the
cleaning step. Leakage check, split-integrity check (zero `trip_id`
overlap, max category-share drift 1.2%), and reproducibility check all
pass.

**Limitations, stated plainly:**
- The data is synthetic; these metrics show model fit to the generating
  process, not real-world NYC taxi fare-prediction accuracy.
- Explainability uses permutation importance (model-agnostic, computed on
  the held-out set), not a full TreeSHAP force-plot workbench.
- No REST API, load testing, or MLOps monitoring is included; this is a
  local Streamlit deployment.
- Model comparison uses a small fixed set of regressors with mostly
  default hyperparameters, not an exhaustive hyperparameter search.

## Files

```
13_crispdm_nyc_taxi_audit_platform/
├── README.md
├── CRISP_DM.md
├── requirements.txt
├── app.py
├── data/
│   ├── generate_data.py
│   └── nyc_tlc_trips.csv         (generated, reproducible via fixed seed)
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── clustering.py
│   ├── modeling.py
│   ├── audit.py
│   └── pipeline.py
└── artifacts/                    (generated by src/pipeline.py)
    ├── best_model.joblib
    ├── metrics.json
    ├── audit_report.json
    ├── cluster_profile.csv
    └── feature_importance.csv
```
