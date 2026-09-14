# CRISP-DM — Time Series Forecasting Engine

The six phases mapped to what this project actually does. Numbers come from
`artifacts/metrics.json` (seed 20260913, `python src/train.py`).

---

## 1. Business understanding

**Question.** A retail operation has daily unit sales for several store x
category combinations and needs an 8-week-ahead forecast to drive
replenishment and promo planning. Which forecasting approach is worth the
added complexity over "assume tomorrow looks like today"?

**Success criteria fixed before modelling.** A candidate model only counts as
an improvement if it beats both a naive last-value baseline and a seasonal
(week-ago) baseline on a holdout the model never trained on. Comparing only
against a "good-sounding" model risks reporting a win that a free baseline
already achieves.

## 2. Data understanding

**Source.** `data/generate_data.py` generates 3 stores x 3 categories (9
series), daily, 2023-01-01 → 2025-12-30 (1,095 days, 9,855 rows), with a
fixed seed. A real multi-year, multi-series Kaggle forecasting dataset (e.g.
Favorita/Walmart-style competitions) runs into the hundreds of MB; this
project generates a small, reproducible
series with the same *shape* of the problem instead: trend, weekly
seasonality, an annual holiday bump, and store-run promo weeks.

**Series-level structure.**

| Element | Detail |
|---|---|
| Trend | Category-specific annual growth (6–18%/yr) |
| Weekly seasonality | Weekend uplift (Fri/Sat/Sun highest) |
| Yearly seasonality | Smooth annual cycle + Nov–Dec holiday bump |
| Promo | ~1 five-day promo week every ~5 weeks per store, +35% uplift |
| Noise | Multiplicative Gaussian noise + Poisson count draw (integer, occasional zeros) |

**Distribution shape.** Counts are non-negative integers with occasional
zero-sale days — the reason SMAPE, not plain MAPE, is used for percentage
error in phase 5.

## 3. Data preparation

`src/features.py`. Calendar features (`day_of_week`, `month`, `day_of_year`,
`is_weekend`, `is_holiday_season`, `promo`) plus per-series lag features
(`lag_1/7/14/28`) and rolling statistics (`roll_mean_7/28`,
`roll_std_7/28`), each computed with `shift()` before `rolling()` so a
feature for day *t* never sees day *t*'s own sales.

**Split.** The last 56 days of **each** series are held out as test; every
earlier day is train (`chronological_split`). This is a forward-in-time
split, not a random row split — a random split would let the model see
future days' lag features during training and report an inflated score.

Rows whose lag/rolling window would reach before the start of history are
dropped (9,855 → 9,603 feature rows) rather than imputed with a placeholder
value.

## 4. Modeling

`src/train.py` fits four candidates:

| Model | What it is |
|---|---|
| `naive` | Prediction = yesterday's actual value (`lag_1`) |
| `seasonal_naive` | Prediction = same weekday last week (`lag_7`) |
| `holt_winters` | One additive Holt-Winters exponential-smoothing model per series (`statsmodels`, weekly seasonal period), fit on that series' own history only |
| `gbm` | One global `HistGradientBoostingRegressor` (scikit-learn) trained on the lag/calendar feature table pooled across all 9 series |

No hyperparameter search was run — each model uses one fixed, reasonable
configuration. This is a comparison of model *classes*, not a tuned
leaderboard.

## 5. Evaluation

Pooled over all 9 series, last 56 days each:

| Model | MAE | RMSE | SMAPE % |
|---|---|---|---|
| Naive | 27.35 | 35.42 | 17.53 |
| Seasonal naive | 29.90 | 39.20 | 19.03 |
| Holt-Winters | 26.14 | 34.50 | 16.96 |
| **Gradient-boosted trees (winner)** | **19.47** | **25.14** | **12.66** |

**Reading this honestly.** The seasonal-naive baseline is *worse* than the
plain naive baseline here (29.90 vs 27.35 MAE) — the week-ago value is
noisier than yesterday's value for this generated data, because the
week-over-week promo/holiday shifts add variance a 7-day lag doesn't average
out. Holt-Winters, which fits trend + weekly seasonality explicitly per
series, is a modest improvement over both naive baselines (26.14 MAE). The
global gradient-boosted-trees model is the clear winner (19.47 MAE, a ~29%
reduction versus Holt-Winters) — it is the only candidate that can use
*multiple* lag horizons, rolling statistics, and the promo/holiday flags
jointly, and the only one that pools information across all 9 series in a
single fit.

**What this does not show.** The GBM evaluation uses true historical lag
values for every test-set row (a legitimate one-step-style evaluation, since
those lags are real past data, not model-generated). It is not a test of
recursive multi-step forecasting, where the model's own errors compound into
its own future lag features — see the Forecast Explorer tab and the model
card for that distinction.

## 6. Deployment

`app.py` — a Streamlit dashboard, six tabs:

| Tab | Contents |
|---|---|
| Overview | KPI tiles, full-history chart for all 9 series, pooled model comparison table. |
| Explore | Per-series raw sales with promo markers, weekday-effect bar chart. |
| Model Comparison | MAE/RMSE/SMAPE bar charts, per-series metric breakdown table. |
| Forecast Explorer | Recent history + a forward forecast (7–90 days) from the winning model; holdout actual-vs-all-models chart per series. |
| CRISP-DM | This phase map. |
| Model Card | Selected model, metrics, reproducibility, feature list, limitations. |

**Scoring path.** `artifacts/best_model.joblib` stores the fitted GBM plus
its feature-column order, and every fitted per-series Holt-Winters model, so
the same artifact works regardless of which model type wins a given run.
`src/forecast.py` walks the winning model forward day by day for the
requested horizon; for the GBM path this is recursive (each day's lag
features come from the model's own prior predictions once real history runs
out), which is explicitly called out as a source of compounding error in the
model card.

**Not in scope:** containerisation,
authentication, a served API, a scheduled retrain, or drift monitoring.

**Reproducibility.** One seed (20260913) drives data generation and the GBM
fit. `python data/generate_data.py && python src/train.py` reproduces every
number in this document.
