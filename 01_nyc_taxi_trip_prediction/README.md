# 01 — NYC Taxi Trip Duration & Fare Prediction

An independent, end-to-end CRISP-DM implementation: synthetic-but-realistic
NYC taxi trip data → cleaning/feature engineering → two regression models
(trip duration, fare) → an interactive Streamlit map front end for live
trip estimation.

This implementation uses an original pipeline, interface, documentation, and
reported results.

### Why not the real Kaggle dataset

The Kaggle "New York City Taxi Trip Duration" dataset is 1.5+ GB and
requires Kaggle account authentication, which breaks reproducibility for a
plain `git clone && pip install`. To keep the project reproducible,
it instead generates a **small, seeded, synthetic
dataset** (`data/generate_data.py`) with the same schema shape (pickup/
dropoff coordinates, timestamps, passenger count, vendor id) and
realistic NYC structure (hotspot clusters around Midtown, FiDi, both
airports, etc.; rush-hour/overnight speed effects; an NYC-taxi-style fare
formula; a small share of injected bad rows so cleaning has real work to
do). Metrics below describe how well the models fit **this synthetic
data**, not the real Kaggle competition — see "Honest results" for that
distinction.

## What was built

- `data/generate_data.py` — seeded synthetic trip generator (9,000 rows).
- `src/features.py` — shared feature engineering (haversine distance,
  time-of-day/rush-hour/airport flags) used identically by training and
  the app, plus the data-cleaning rules.
- `src/train.py` — CRISP-DM Data Preparation → Modeling → Evaluation
  pipeline. Trains two `RandomForestRegressor` models (duration, fare) and
  writes them plus `models/metrics.json` to `models/`.
- `app.py` — Streamlit front end: pick pickup/dropoff (NYC landmark presets
  or manual lat/lon), see both plotted with a route line on an interactive
  pydeck map, set date/time/passengers/vendor, click **Estimate trip** to
  get predicted duration + fare, and expand a panel with honest held-out
  model metrics.
- `CRISP_DM.md` — the six CRISP-DM phases mapped to this project's
  concrete decisions and code.

## Setup

```bash
cd 01_nyc_taxi_trip_prediction
python3.13 -m venv .venv        # 3.13 recommended; sklearn/pandas wheels
source .venv/bin/activate       # not yet available for 3.14 at build time
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate the dataset (also runs automatically from train.py if missing)
python data/generate_data.py

# 2. Train both models and write models/metrics.json
python src/train.py

# 3. Launch the front end
streamlit run app.py
```

Then open the printed local URL (default `http://localhost:8501`).

## Approach (CRISP-DM summary — full detail in `CRISP_DM.md`)

1. **Business understanding** — estimate trip duration and fare from
   pickup/dropoff location, time, and trip context, exposed through a
   usable map UI for interactive trip estimation.
2. **Data understanding** — inspect the generated data's schema, ranges,
   and known injected defects (zero-distance rows, duration outliers,
   missing passenger counts).
3. **Data preparation** — `clean_raw()` drops out-of-NYC-bounds
   coordinates, zero/near-zero-distance rows, and duration/fare outliers;
   `build_features()` derives distance (haversine), calendar parts, and
   rush-hour/overnight/airport indicator flags.
4. **Modeling** — one `RandomForestRegressor` per target (duration, fare),
   80/20 train/test split, fixed random seed for reproducibility.
5. **Evaluation** — MAE/RMSE/R² computed on the held-out test split only
   (never seen during training); see results below.
6. **Deployment** — the Streamlit app loads the saved models and serves
   live predictions through the interactive map UI as a local deployment.

## Honest results

From the last `python src/train.py` run on the generated dataset
(9,000 raw rows → 8,807 after cleaning → 7,045 train / 1,762 test):

| Target | MAE | RMSE | R² |
|---|---|---|---|
| Trip duration | 336.8 s (~5.6 min) | 582.3 s | 0.846 |
| Fare amount | $2.60 | $3.96 | 0.971 |

`trip_distance_miles` dominates both models' feature importance (80% for
duration, 93% for fare), which matches the data-generating process — the
synthetic fare formula is distance-driven with smaller time-of-day/airport
adjustments, and duration additionally depends on rush-hour traffic
slowdowns (its second-most-important feature at ~14%).

**Limitations, stated plainly:**
- The data is synthetic. These metrics show the models fit the
  generating process well; they are not a claim about accuracy on real
  NYC taxi trips or the real Kaggle leaderboard.
- Fare and duration are generated from a hand-built formula plus noise,
  not observed real-world data, so the "ground truth" itself is a
  simplification (no real traffic incidents, weather, or fare-rule edge
  cases).
- The map's location picker uses preset landmarks plus manual
  latitude/longitude entry rather than click-to-drop-pin selection, to
  avoid adding extra front-end dependencies beyond Streamlit + pydeck.

## Files

```
01_nyc_taxi_trip_prediction/
├── README.md
├── CRISP_DM.md
├── requirements.txt
├── app.py
├── data/
│   ├── generate_data.py
│   └── nyc_taxi_trips.csv        (generated, reproducible via fixed seed)
├── src/
│   ├── features.py
│   └── train.py
└── models/                       (generated by src/train.py)
    ├── duration_model.joblib
    ├── fare_model.joblib
    └── metrics.json
```
