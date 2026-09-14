# CRISP-DM Notes — NYC Taxi Trip Duration & Fare Prediction

Concrete mapping of this project's code and decisions to the six CRISP-DM
phases. See [`README.md`](./README.md) for setup/run commands and results.

## 1. Business Understanding

**Goal:** given a pickup location, dropoff location, pickup time, and
passenger count, estimate (a) how long the trip will take and (b) what it
will cost — surfaced through an interactive map so a user can pick
locations visually rather than typing raw coordinates.

**Success criteria:** a working local pipeline (data → trained models →
served predictions) with honestly reported held-out error, not a specific
accuracy target — this is a learning/demonstration build, not a production
pricing system.

## 2. Data Understanding

Real-world Kaggle NYC taxi data was reviewed only for its **schema and
scale** (pickup/dropoff lat-lon, pickup timestamp, passenger count, vendor
id, trip duration; fare data available separately in NYC TLC trip records),
generated locally. Because the real dataset is 1.5+ GB and gated behind Kaggle
auth, `data/generate_data.py` produces a schema-equivalent synthetic
dataset instead (see README "Why not the real Kaggle dataset").

Inspecting the generated data (`data/nyc_taxi_trips.csv`, 9,000 rows)
surfaces the deliberately injected issues that Data Preparation must
handle:
- ~1% of rows have identical pickup/dropoff coordinates (GPS glitch
  simulation).
- ~1% of rows have 3–6 hour trip durations (stuck-meter simulation).
- ~1% of rows have a missing `passenger_count`.

## 3. Data Preparation

Implemented in `src/features.py`:

- `clean_raw()` — drops rows outside a generous NYC bounding box, rows with
  near-zero trip distance, and duration/fare values outside a plausible
  range (30s–3h; $2.50–$250). On the generated dataset this removes ~2% of
  rows.
- `build_features()` — derives the actual model inputs from raw fields:
  - `trip_distance_miles` via the haversine formula between pickup and
    dropoff coordinates.
  - `pickup_hour`, `pickup_dayofweek`, `pickup_month` from the timestamp.
  - `is_weekend`, `is_rush_hour` (7-9am, 4-7pm), `is_overnight` (12-5am)
    indicator flags.
  - `is_airport_trip` — within 1.5 miles of JFK or LaGuardia on either end.
  - Missing `passenger_count` imputed with the column median.

This module is imported by both `src/train.py` and `app.py` so training
and serving use identical feature logic (no train/serve skew).

## 4. Modeling

`src/train.py` trains two independent `RandomForestRegressor` models
(scikit-learn), one per target:

- Duration model → predicts `trip_duration` (seconds).
- Fare model → predicts `fare_amount` (USD).

Random forests were chosen over a linear model because trip duration and
fare depend on nonlinear interactions (e.g., rush-hour slowdowns only
matter for longer trips) that a plain linear regression would underfit,
while staying fast to train and easy to inspect via feature importances —
appropriate for a compact demonstration rather than a heavily tuned
production model.

An 80/20 train/test split with a fixed `random_state=42` keeps results
reproducible run to run.

## 5. Evaluation

Metrics (MAE, RMSE, R²) are computed **only on the 20% test split**, never
seen during training, via `sklearn.metrics`. Results are written to
`models/metrics.json` and summarized in the README's "Honest results"
section. Feature importances are also recorded per model to sanity-check
that the models are learning sensible relationships (distance dominates
both; rush-hour matters most for duration; airport proximity matters most
for fare after distance).

No test-set leakage into training: cleaning and feature engineering are
deterministic transforms of a single row (no target-derived or
future-derived features), and the split happens after feature engineering
but the transforms themselves use no cross-row statistics beyond the
median imputation, which is acceptable for a demo of this scope.

## 6. Deployment

`app.py` is the deployment surface: a Streamlit app that loads the two
saved model artifacts (`models/*.joblib`) and `models/metrics.json`,
builds features from user input through the same `src/features.py`
functions used in training, and serves live predictions plus an
interactive pydeck map. This is a local deployment (`streamlit run app.py`)
appropriate to this project's scope; cloud hosting, containers, and
authentication are outside the current implementation.
