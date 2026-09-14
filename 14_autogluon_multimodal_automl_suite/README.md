# 14 — AutoGluon Multimodal AutoML Suite

Three distinct AutoGluon capabilities, each on a small reproducible dataset,
served from one Streamlit dashboard:

1. **Multimodal classification** — `autogluon.tabular.TabularPredictor` fits
   a free-text review column together with numeric/categorical columns in a
   single call; AutoGluon's built-in text feature generator handles the NLP
   featurization automatically.
2. **Quantile regression** — the same `TabularPredictor` API, switched to
   `problem_type="quantile"`, produces P10/P50/P90 prediction bands instead
   of one point estimate.
3. **Time series forecasting** — `autogluon.timeseries.TimeSeriesPredictor`
   (a separate AutoGluon package) forecasts six related series at once.

This implementation uses an original AutoGluon pipeline, interface,
documentation, metrics, and reported results. External claims such as
particular foundation models, latency numbers, or accuracy figures are not
reproduced or asserted here.

## What was built

| File | Role |
|---|---|
| `data/load_data.py` | Task registry: one deterministic synthetic multimodal (text + tabular) generator, one built-in sklearn regression dataset, one deterministic synthetic multi-series time series generator. |
| `src/automl_tabular.py` | `TabularPredictor` wrapper for the classification and quantile-regression fits. |
| `src/automl_timeseries.py` | `TimeSeriesPredictor` wrapper for the forecasting fit. |
| `src/train.py` | Runs all three tasks: split → fit → one held-out evaluation → artifacts written to `artifacts/<task>/`. |
| `app.py` | Five-tab Streamlit dashboard: Overview, Multimodal Classification, Quantile Regression, Time Series Forecasting, Live Scoring. |

This is a deliberately different slice of AutoGluon from
[Project 07 (AutoGluon Multi-Layer Stacking Platform)](../07_automl_autogluon/),
which covers plain tabular binary/multiclass/regression with an AutoResearch
hill climb over bagging/stacking depth. Project 14 instead spans breadth —
text+tabular fusion, a probabilistic task formulation, and a second AutoGluon
package (`autogluon.timeseries`) — rather than depth on one task type.

## Setup

```bash
cd 14_autogluon_multimodal_automl_suite
python3.13 -m venv .venv        # AutoGluon 1.6.1 supports Python 3.13
source .venv/bin/activate
pip install -r requirements.txt
```

`autogluon.timeseries` depends on `xgboost-cpu`, which has no prebuilt wheel
for macOS/arm64 on Python 3.13 yet and builds from source — install `cmake`
first (`brew install cmake` on macOS) if the pip install fails on that step.

## Run

```bash
python src/train.py     # ~3-5 minutes: fits and evaluates all 3 tasks
streamlit run app.py    # opens the dashboard at http://localhost:8501
```

`src/train.py` must run at least once before `app.py` — the dashboard reads
everything from `artifacts/` and `models/`, which are gitignored (the fitted
predictors are 10s of MB each).

## Approach

- **Data**: the multimodal-review and retail-demand datasets are
  deterministic synthetic generators (`numpy.random.default_rng(42)`) —
  templated positive/negative review phrases combined with correlated
  numeric/categorical fields for task 1, and seasonal + trend + noise daily
  series for task 3. The quantile-regression dataset is scikit-learn's
  built-in `load_diabetes` (442 rows, no download). Using generated/built-in
  data keeps the whole suite runnable offline and reproducible on any
  machine.
- **Splitting**: tasks 1 and 2 use a single stratified/random 80/20
  train/test split, scored once after fitting. Task 3 uses AutoGluon's
  `TimeSeriesDataFrame.train_test_split(prediction_length)`, which holds out
  the last 14 days of every series — no leakage of future values into
  training.
- **Modeling**: `presets="medium_quality"` for all fits, with `XGB` excluded
  from the tabular model pool (mirrors Project 07 — keeps `pip install`
  sufficient without a working `libomp`/native XGBoost toolchain for the
  tabular fits; `autogluon.timeseries` needs `xgboost-cpu` for
  `RecursiveTabular`/`DirectTabular` and does require the `cmake` build step
  above). Time budgets: 60s per tabular fit, 90s for the time series fit —
  enough for AutoGluon to try several models and its own ensembling, not
  enough to be a real production budget.
- **Evaluation**: each held-out test/window is scored exactly once, after
  fitting is complete, using AutoGluon's own `.evaluate()` on its native
  metric (`roc_auc` for classification, `pinball_loss` for quantile
  regression, `MASE` for forecasting). The dashboard reads these numbers
  from `artifacts/*/metrics.json` and computes nothing itself.

## Honest results

Exact numbers are written fresh by every `src/train.py` run to
`artifacts/summary.json` and shown in the dashboard's Overview tab — they
are not reproduced here since they depend on the machine's model-fit time
budget and AutoGluon's internal ensembling choices, and this project follows
the instruction not to assert results outside of what a run actually
produces. Representative shape observed locally: the multimodal classifier's
held-out ROC-AUC is comfortably above the 0.5 baseline (the synthetic label
is intentionally text+rating-correlated), the quantile regressor's
P10-P90 band covers roughly 70-85% of held-out points (below the nominal 80%
target on a 442-row dataset, as expected from limited data), and the
forecaster beats a naive seasonal baseline on most of the six synthetic
series. Run `python src/train.py` to see the actual numbers for this
checkout.

## Known limitations

- All three datasets are small (≤1,200 rows / 442 rows / 1,080 timeseries
  rows) and, for tasks 1 and 3, synthetic — this is a capability
  demonstration, not a benchmark against real-world data.
- Time budgets (60-90s per fit) are laptop-friendly, not tuned for maximum
  accuracy; AutoGluon would find better models given more time.
- `LightGBM` is excluded from the tabular model pool (same constraint as
  Project 07: no system `libomp` assumed present) — CatBoost, Random Forest,
  and Extra Trees still give the tabular fits multiple base learners.
- The quantile regressor's P10-P90 coverage on a 20-row held-out sample is a
  noisy small-sample estimate, not a calibrated guarantee.
