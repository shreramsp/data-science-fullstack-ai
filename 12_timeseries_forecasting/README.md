# 12 — Time Series Forecasting Engine

An independent, end-to-end CRISP-DM time-series forecasting project:
synthetic multi-series daily retail sales → calendar/lag/rolling feature
engineering with a chronological holdout → four forecasting candidates
(naive, seasonal naive, Holt-Winters, gradient-boosted trees) → a Streamlit
admin dashboard for exploration, model comparison, and forward forecasting.

This project uses an original CRISP-DM time-series forecasting pipeline,
administrative dashboard, documentation, and reported results.

## What was built

| File | Purpose |
|---|---|
| `data/generate_data.py` | Seeded generator for 9 daily sales series (3 stores x 3 categories) with trend, weekly/yearly seasonality, and promo effects. |
| `src/features.py` | Calendar + lag/rolling feature engineering; chronological (time-forward) train/test split. |
| `src/train.py` | CRISP-DM pipeline — prepare, fit 4 models, evaluate on the holdout, persist artifacts. |
| `src/forecast.py` | Walks the winning model forward past the end of history for a requested horizon. |
| `app.py` | Six-tab Streamlit admin dashboard. |
| `CRISP_DM.md` | The six phases mapped to concrete decisions and code. |

### On the dataset

Projects in this portfolio use small, seeded, reproducible datasets rather
than depend on a large authenticated Kaggle download (see project 01/03/06
for the same pattern). This project generates 9 daily sales series over 3
years (1,095 days) with the qualitative properties a real retail
time-series dataset (e.g. a Favorita/Walmart-style forecasting competition)
has: trend, weekly seasonality, an annual holiday bump, and promo weeks.

**Every number in this README and in `CRISP_DM.md` describes that generated
dataset.** None of it is a claim about real store sales or about any
published forecasting benchmark result.

## Setup

```bash
cd 12_timeseries_forecasting
python3.13 -m venv .venv          # 3.13 recommended; sklearn/pandas/statsmodels
source .venv/bin/activate         # wheels are not yet published for 3.14
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate the sales data (train.py does this automatically if missing)
python data/generate_data.py

# 2. Run the CRISP-DM pipeline: features, 4 models, evaluation, artifacts/
python src/train.py                    # ~2s

# 3. Launch the dashboard
streamlit run app.py
```

Then open the printed local URL (default `http://localhost:8501`).

## Approach

### Feature engineering & the holdout

Each series gets calendar features (day of week, month, holiday-season
flag, promo flag) plus lag features at 1/7/14/28 days and rolling mean/std
at 7/28 days, computed with `shift()` before `rolling()` so no feature for
day *t* can see day *t*'s own value. The holdout is the **last 56 calendar
days of every series**, not a random sample of rows — a random split on
time-series data would let a model train on rows that are chronologically
*after* rows in its own test set, which inflates reported accuracy.

### Four candidates, one comparison

- **Naive** — tomorrow = today.
- **Seasonal naive** — tomorrow = same weekday last week.
- **Holt-Winters** — one additive exponential-smoothing model per series
  (`statsmodels`), fit on that series' own history.
- **Gradient-boosted trees** — one global `HistGradientBoostingRegressor`
  (scikit-learn) over the pooled lag/calendar feature table across all 9
  series.

No hyperparameter search — each model uses one fixed configuration. This
compares model *classes* honestly rather than tuning one candidate harder
than the others.

## Honest results

From the last `python src/train.py` run (seed 20260913, ~2s), pooled MAE /
RMSE / SMAPE over all 9 series' 56-day holdouts:

| Model | MAE | RMSE | SMAPE % |
|---|---|---|---|
| Naive | 27.35 | 35.42 | 17.53 |
| Seasonal naive | 29.90 | 39.20 | 19.03 |
| Holt-Winters | 26.14 | 34.50 | 16.96 |
| **Gradient-boosted trees (winner)** | **19.47** | **25.14** | **12.66** |

**The seasonal-naive baseline underperforms plain naive** in this run
(29.90 vs 27.35 MAE) — the promo/holiday variance in this generated data
makes a week-old value noisier than yesterday's value. Holt-Winters is a
modest step up over both naive baselines. The gradient-boosted-trees model
is the clear winner, a ~29% MAE reduction versus Holt-Winters, because it is
the only candidate using multiple lag horizons, rolling statistics, and the
promo/holiday flags together, pooled across all 9 series in one fit.

SMAPE (symmetric mean absolute percentage error) is reported instead of
plain MAPE because the data has occasional zero-sale days, where a plain
percentage error is undefined.

## Limitations

- **Synthetic data.** Trend, seasonality and promo effects are generator
  parameters, not observed retail behaviour — no real stock-outs,
  competitor actions, weather, or price changes.
- **The evaluation is not a recursive multi-step-forecast test.** The GBM
  holdout uses each row's true historical lag values, which is a legitimate
  evaluation for a system that refits/re-lags daily, but it does not measure
  how error compounds if the model is instead run forward untouched for the
  whole horizon. The Forecast Explorer tab's GBM path *is* recursive (each
  future day's lags come from the model's own prior predictions), and the
  dashboard's model card calls this distinction out explicitly.
- **No hyperparameter search.** Each model uses one fixed, reasonable
  configuration; none were tuned.
- **9 series, 3 years of history.** Enough to compare these four candidates,
  not enough to detect slow regime changes or fit anything more data-hungry.
- **Holt-Winters uses one fixed seasonal period (7 days)** and no
  multiplicative-seasonality search.

## Validation performed

- `python data/generate_data.py` and `python src/train.py` run clean
  end to end and reproduce the table above under the fixed seed.
- The forward-forecast helper (`src/forecast.py`) was run standalone for a
  sample series and produces a 14-day forecast with sensible weekday
  variation (higher on weekends), confirming the recursive lag-feature
  construction works.
- The dashboard was executed headlessly with Streamlit's `AppTest` runner
  (`streamlit.testing.v1.AppTest`), which runs the full script including the
  content of all six tabs: zero exceptions raised. It was also started as a
  live server (`streamlit run app.py --server.headless true`) and returned
  HTTP 200 with no errors in the server log.

## Files

```
12_timeseries_forecasting/
├── README.md
├── CRISP_DM.md
├── requirements.txt
├── app.py
├── .streamlit/config.toml          (light theme)
├── data/
│   ├── generate_data.py
│   └── sales.csv                   (generated, reproducible via fixed seed)
├── src/
│   ├── features.py
│   ├── train.py
│   └── forecast.py
└── artifacts/                      (generated by src/train.py)
    ├── best_model.joblib
    ├── metrics.json
    ├── forecasts.csv
    ├── model_comparison.csv
    └── per_series_metrics.csv
```
