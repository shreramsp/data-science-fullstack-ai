# CRISP-DM — SOTA SPY Time Series Forecasting & Trading Platform

## 1. Business Understanding

Forecast a broad equity index's short-horizon return distribution (not
just a point estimate) and evaluate whether that forecast supports a
simple, honestly-backtested trading rule — with strict controls against
temporal data leakage, since leakage is the single most common way
time-series ML projects report inflated, non-reproducible performance.

## 2. Data Understanding

`data/generate_data.py` produces 1,260 trading days (~5 years) of a
single synthetic, SPY-like daily OHLCV series: a two-state Markov
volatility regime (calm ≈12% / stressed ≈28% annualized vol, 78%/22%
observed mix), Student-t (df=5) fat-tailed daily shocks, and mild
lag-1/lag-5 return autocorrelation so lag and rolling-window technical
features have genuine (if weak) signal to find. See `README.md` for why
a generated series is used instead of a downloaded one.

## 3. Data Preparation

`src/features.py`:
- Log returns, then lagged returns (1/2/3/5/10 days), rolling mean/std
  of returns (5/20 days), 10-day momentum, RSI(14), MACD(12,26,9),
  20-day volume z-score, and prior-day high-low range.
- **Leakage discipline**: every rolling/EWM window is computed on a
  `shift(1)`-ed series first, so the feature row for day *t* only uses
  information available at the close of day *t-1*.
- Two forward targets: `fwd_return_1` (t+1) and `fwd_return_5` (t+5)
  log returns, computed with negative shifts, never used as features.
- **Chronological split** at 80%/20% (982 train / 242 test rows) with a
  **5-day embargo** dropped on both sides of the boundary, so no
  rolling-window or forward-target row spans train and test.
- `RobustScaler` fit on the training features only, applied unchanged
  to test.

## 4. Modeling

`src/train.py`, per horizon (t+1, t+5):
- **Point-forecast tournament**: naive (zero-return random walk),
  Ridge regression, `HistGradientBoostingRegressor` — compared on the
  held-out test set by MAE, RMSE, and directional hit rate.
- **Probabilistic envelope**: three `HistGradientBoostingRegressor`
  models per horizon with `loss="quantile"` at q=0.10/0.50/0.90,
  evaluated by pinball loss and empirical P10-P90 coverage against the
  nominal 80%.
- No hyperparameter search — each model uses one fixed, reasonable
  configuration, to compare model classes honestly.

## 5. Evaluation

- Point metrics and quantile calibration are reported in
  `artifacts/model_comparison.csv` / `artifacts/metrics.json`.
- **Explainability**: permutation importance (MAE increase under
  feature shuffling on the test set) for the t+1 GBM model, in
  `artifacts/feature_importance.csv`.
- **Trading backtest** (`src/backtest.py`): a long/flat rule driven by
  the t+1 P50 forecast (long when predicted log return > threshold,
  else flat; 2 bps slippage only on position changes), compared against
  a passive buy-and-hold baseline on the same out-of-sample window —
  Sharpe, Sortino, max drawdown, and directional hit rate for both.
- See `README.md` → "Honest results" for the actual numbers from the
  fixed-seed run, including where the strategy underperforms buy-and-
  hold and where quantile coverage misses its target.

## 6. Deployment

`app.py` — a six-tab Streamlit dashboard (Overview, Data & Features,
Model Tournament, Probabilistic Forecast, Explainability, Trading
Backtest) reading the artifacts produced by `src/train.py`. No
separate serving layer; this is a local analyst tool, not a production
trading system.
