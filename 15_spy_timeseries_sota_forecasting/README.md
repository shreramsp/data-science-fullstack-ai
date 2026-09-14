# 15 — SOTA SPY Time Series Forecasting & Trading Platform

An independent, end-to-end CRISP-DM probabilistic time-series forecasting
project: a synthetic, SPY-like daily price series → leakage-safe
technical-feature engineering with a chronological, embargoed
train/test split → a point-forecast model tournament plus P10/P50/P90
quantile forecasts at two horizons (t+1, t+5) → a simple long/flat
trading backtest against buy-and-hold → a Streamlit dashboard.

This is a focused implementation of a zero-leakage, multi-horizon
probabilistic forecasting and trading platform for SPY with explainability.
The full-scale architecture (a 7-backbone tournament including
Chronos-T5/PatchTST/TFT, a separate FastAPI+React stack, Playwright
screenshots, and a UML diagram suite) and its associated performance analysis
are outside the current scope — see "Deliberate scope boundaries" below.

## What was built

| File | Purpose |
|---|---|
| `data/generate_data.py` | Seeded generator for a synthetic, SPY-like daily OHLCV series (regime-switching volatility, fat-tailed shocks, mild autocorrelation). |
| `src/features.py` | Leakage-safe technical-indicator feature engineering; chronological train/test split with a 5-day embargo. |
| `src/train.py` | CRISP-DM pipeline — point-forecast tournament, P10/P50/P90 quantile models, evaluation, permutation importance, backtest, artifact persistence. |
| `src/backtest.py` | Long/flat trading rule driven by the median forecast, vs. buy-and-hold. |
| `app.py` | Six-tab Streamlit dashboard. |
| `CRISP_DM.md` | The six phases mapped to concrete decisions and code. |

### On the dataset

A downloaded SPY price history is small enough to fit in this repo, but
this portfolio's convention (see projects 01/03/06/12) is a small,
seeded, reproducible **generated** dataset so the project has no runtime
dependency on an external data provider and every number in this README
is exactly reproducible from the fixed seed. `data/generate_data.py`
simulates 1,260 trading days (~5 years) with the qualitative statistical
properties a real equity index has — volatility clustering via a
two-state Markov regime (calm ≈12% / stressed ≈28% annualized vol),
fat-tailed daily shocks (Student-t, df=5), and mild lag-1/lag-5 return
autocorrelation so technical features have real, if weak, signal.

**Every number below describes that generated series.** None of it is a
claim about real SPY prices, returns, or any published trading-strategy
result.

### Deliberate scope boundaries

To keep the project reproducible and its results honest, this build
intentionally does **not** include: the 7-backbone
tournament (Chronos-T5, PatchTST, TFT+VSN, a 2-level LightGBM/XGBoost/
CatBoost/Ridge stacking DAG, Bi-LSTM, AutoARIMA+GARCH, Caruana
ensembling); a separate FastAPI backend + React frontend; SHAP
waterfall plots; a macro stress simulator; a UML diagram suite; a LaTeX
research paper; or a Playwright screenshot suite. Instead: a 3-candidate
point tournament (naive / Ridge / gradient-boosted trees) plus
gradient-boosted quantile regression for the P10/P50/P90 envelope, with
permutation-importance explainability and a single-page Streamlit
dashboard — the same modeling *rigor* (chronological + embargoed split,
scaler fit only on train, honest backtest vs. a real baseline) at a
fraction of the surface area.

## Setup

```bash
cd 15_spy_timeseries_sota_forecasting
python3.13 -m venv .venv          # 3.13 recommended; sklearn/pandas wheels
source .venv/bin/activate         # not yet published for 3.14
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate the synthetic price series (train.py does this automatically if missing)
python data/generate_data.py

# 2. Run the CRISP-DM pipeline: features, tournament, quantiles, backtest, artifacts/
python src/train.py                    # ~3s

# 3. Launch the dashboard
streamlit run app.py
```

Then open the printed local URL (default `http://localhost:8501`).

## Approach

### Leakage controls

- Every lag/rolling/EWM feature is computed on a `shift(1)`-ed series,
  so day *t*'s features only see information available at the close of
  day *t-1*.
- The train/test split is **chronological** (first 80% / last 20%,
  982/242 rows), not random — with a **5-day embargo dropped on both
  sides of the boundary** so no rolling window or forward-looking
  target row spans train and test.
- `RobustScaler` is fit on the training features only and applied
  unchanged to test.

### Point tournament, then quantiles

Three candidates per horizon (t+1 next-day, t+5 next-week log return):
naive (zero-return random walk), Ridge regression, and
`HistGradientBoostingRegressor`, compared honestly with one fixed
configuration each (no hyperparameter search). The winning algorithm
family (gradient-boosted trees) is then refit three times per horizon
with `loss="quantile"` at q=0.10/0.50/0.90 to produce the P10/P50/P90
envelope.

### Backtest

A simple long/flat rule: go long when the t+1 P50 forecast exceeds a
threshold (default 0.0), otherwise stay flat — no shorting, no
leverage. A 2 bps slippage cost is charged only on days the position
changes. Compared against passive buy-and-hold on the same out-of-
sample window.

## Honest results

From the last `python src/train.py` run (seed 20260913, ~3s, 982
train / 242 test days, split cutoff 2025-09-25):

**Point-forecast tournament (out-of-sample):**

| Horizon | Model | MAE | RMSE | Directional hit rate |
|---|---|---|---|---|
| t+1 | naive | 0.00779 | 0.01343 | 0.0% |
| t+1 | **Ridge (winner, MAE)** | **0.00778** | **0.01338** | 51.8% |
| t+1 | GBM | 0.00815 | 0.01375 | 51.7% |
| t+5 | naive | 0.02026 | 0.02997 | 0.4% |
| t+5 | **Ridge (winner, MAE)** | **0.02026** | **0.02971** | 48.8% |
| t+5 | GBM | 0.02456 | 0.03369 | 42.6% |

Ridge and naive are within noise of each other on MAE at both
horizons — **next-day and next-week log returns on this series are
close to a random walk**, and directional hit rates cluster around
42-52%, i.e. close to a coin flip. This is the expected, honest outcome
for return-level forecasting (as opposed to volatility forecasting,
which is typically far more predictable) and is consistent with how
hard real equity-index return prediction is; it is *not* a bug.

**Quantile calibration (P10-P90 band, nominal target 80% coverage):**

| Horizon | Empirical coverage | Pinball loss (P50) |
|---|---|---|
| t+1 | 76.9% | 0.00394 |
| t+5 | 63.6% | 0.01101 |

The t+1 band is reasonably calibrated (76.9% vs. 80% target); the t+5
band is meaningfully under-covered (63.6%), i.e. the P10/P90 envelope
is too narrow for the 5-day-ahead return at this sample size.

**Trading backtest (t+1 P50-driven, long/flat, 2 bps slippage, same
242-day out-of-sample window):**

| | Strategy | Buy & Hold |
|---|---|---|
| Total return | 13.1% | 27.2% |
| Annualized return | 13.7% | 28.5% |
| Annualized volatility | 11.0% | 21.3% |
| Sharpe ratio | 1.25 | **1.34** |
| Sortino ratio | 1.17 | 2.11 |
| Max drawdown | **-5.7%** | -10.3% |

Directional hit rate: 52.1%.

**The strategy does not beat buy-and-hold on a risk-adjusted (Sharpe)
basis in this test window** — it earns a lower Sharpe and Sortino
ratio, because the underlying test period has strong, fairly steady
upward drift that a partially-flat long/flat rule (52% hit rate)
partly sits out of. It does deliver materially lower volatility and a
smaller max drawdown (-5.7% vs. -10.3%), which is a legitimate
risk-management outcome even where it is not a return-maximizing one.
Reporting this honestly is the point: a 52%-hit-rate signal is weak,
and a weak signal does not reliably beat a strong-drift buy-and-hold
baseline risk-adjusted, even though it can still reduce drawdown.

## Limitations

- **Synthetic data.** Volatility regimes, fat tails, and
  autocorrelation are generator parameters, not observed market
  behaviour — no real macro shocks, earnings, or liquidity events.
- **Single test window.** All out-of-sample numbers come from one
  242-day holdout of one simulated path; no walk-forward /
  purged-K-fold cross-validation across multiple windows was run, so
  these results should not be read as a stable expectation.
- **t+5 quantile band is under-covered** (63.6% vs. 80% target) — the
  5-day-ahead envelope is currently too narrow at this sample size.
- **No hyperparameter search.** Every model uses one fixed, reasonable
  configuration; none were tuned.
- **No transaction-cost realism beyond a flat 2 bps slippage** on
  position changes — no bid/ask spread modeling, market impact, or
  financing cost for the flat/long position.
- **The strategy underperforms buy-and-hold on Sharpe/Sortino** in this
  run, as reported above; it is not presented as a superior strategy.

## Validation performed

- `python data/generate_data.py` and `python src/train.py` run clean
  end to end and reproduce the tables above under the fixed seed.
- The dashboard was executed headlessly with Streamlit's `AppTest`
  runner (`streamlit.testing.v1.AppTest`), which runs the full script
  including all six tabs: zero exceptions raised.
- The dashboard was also started as a live server
  (`streamlit run app.py --server.headless true`) and returned HTTP 200
  with no errors in the server log.

## Files

```
15_spy_timeseries_sota_forecasting/
├── README.md
├── CRISP_DM.md
├── requirements.txt
├── app.py
├── .streamlit/config.toml          (light theme)
├── data/
│   ├── generate_data.py
│   └── spy_prices.csv              (generated, reproducible via fixed seed)
├── src/
│   ├── features.py
│   ├── train.py
│   └── backtest.py
└── artifacts/                      (generated by src/train.py)
    ├── best_model.joblib
    ├── metrics.json
    ├── model_comparison.csv
    ├── forecasts.csv
    ├── feature_importance.csv
    └── backtest_results.csv
```
