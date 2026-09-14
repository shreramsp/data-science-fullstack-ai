# CRISP-DM Phases — CRISP-DM NYC TLC Audit Platform

## 1. Business Understanding

**Objective:** predict `total_amount` for a NYC taxi-style trip from trip
distance, pickup/dropoff zone, and time context, while giving a
data-science/code auditor full visibility into every step that produced
that number.

**Success criteria:** honest held-out regression metrics, zero target
leakage, a fixed-seed reproducible pipeline, and an automated audit trail
readable without re-deriving the code (`src/audit.py`, "Audit Report" tab).

## 2. Data Understanding

The real NYC TLC trip record archive is many GB per month and requires a
separate download step that breaks `git clone && pip install`
reproducibility. `data/generate_data.py` instead builds a seeded synthetic
dataset (8,000 base trips, deterministic RNG) shaped like a TLC trip
record: pickup/dropoff zone + coordinates around seven NYC borough/airport
hotspots, trip distance (haversine + noise), rush-hour-aware speed and
duration, and a TLC-style fare formula (base + per-mile + per-minute +
tolls + congestion surcharge + tip). A ~3% slice of rows is deliberately
corrupted (missing fields, out-of-bounds GPS, negative fares, duplicate
rows) so the audit stage has real defects to catch. `app.py`'s "Data
Understanding" tab shows the raw schema, summary statistics, and per-zone
trip counts.

## 3. Data Preparation

`src/features.py::clean_raw` — deterministic, documented cleaning rules:
drop duplicates, drop out-of-NYC-bounds pickups, drop non-positive
fare/distance rows, impute missing `passenger_count` (median) and
`payment_type` (`"unknown"`). `build_features` then derives pickup
hour/day-of-week, weekend flag, rush-hour flag, and airport-trip flag —
none of which touch `fare_amount`/`total_amount`, to avoid leaking the
target into the features.

## 4. Modeling

`src/modeling.py` trains four regressors (Linear Regression, Ridge,
Random Forest, Gradient Boosting) on an identical 80/20 split (fixed seed
42) inside a `ColumnTransformer` + `Pipeline` (one-hot encoding for
categoricals, numeric features passed through). This is a small, honest
stand-in for the brief's "auto research of various techniques" — a fixed
set of models compared by held-out RMSE, not an open-ended hyperparameter
search.

## 5. Evaluation

Every metric (MAE, RMSE, R²) in `artifacts/metrics.json` and the
"Modeling & Evaluation" tab is computed on the 20% test split, which is
never used for fitting. The best model is selected by lowest test RMSE.
Explainability uses **permutation importance** on the same held-out set
(`src/modeling.py::explain_with_permutation_importance`) — a
dependency-light, honest substitute for a full TreeSHAP workbench.

Also included, as unsupervised data-understanding steps: K-Means spatial
clustering of pickup coordinates (`src/clustering.py`, "Spatial
Clustering" tab), profiled by average fare/distance per cluster.

## 6. Deployment

The Streamlit app (`app.py`) loads the pipeline's saved best model and
serves live fare estimates through a "Fare Estimator" tab (pickup/dropoff
zone, distance, time, passengers, payment type → predicted total fare with
its held-out MAE shown alongside as an honest error margin). This is a
local deployment; no REST API, load
testing, or cloud infrastructure is included (see README "Honest results
and limitations").

## Audit trail (the platform's distinguishing feature)

`src/audit.py` runs five deterministic checks and writes
`artifacts/audit_report.json`, surfaced in the "Audit Report" tab:

1. **Raw data quality** — missing values, duplicate rows, out-of-bounds
   coordinates, negative fares, non-positive distances.
2. **Cleaning transparency** — raw vs. clean row counts and % dropped.
3. **Leakage check** — confirms the target column is not present in the
   feature set used to fit the model.
4. **Split integrity** — confirms zero `trip_id` overlap between train and
   test, and reports category-share drift (payment type) between splits.
5. **Reproducibility** — confirms a single fixed seed drives data
   generation, the train/test split, and model fitting.
