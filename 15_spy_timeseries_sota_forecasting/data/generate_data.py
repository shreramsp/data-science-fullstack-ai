"""Seeded generator for a synthetic SPY-like daily OHLCV price series.

Simulates a single broad-market index series with:
  - a small positive long-run drift,
  - a two-state Markov volatility regime (calm / stressed) so realized
    volatility clusters the way real equity index volatility does,
  - mild short-horizon return autocorrelation (momentum then mean
    reversion), which gives lag/rolling-window technical features
    genuine (if modest) predictive signal to find,
  - OHLV built around the simulated close with a small realistic
    intraday range and volume that rises with volatility.

This is a *generated* series with the qualitative statistical properties
of a real equity index (fat-tailed, vol-clustered daily returns) — not a
download of, or a claim about, actual SPY prices or returns.

Run:  python data/generate_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260913
N_DAYS = 1260  # ~5 trading years
START_PRICE = 420.0
OUT_PATH = Path(__file__).resolve().parent / "spy_prices.csv"

# Two-state annualized volatility regime: calm vs. stressed.
VOL_CALM = 0.12
VOL_STRESSED = 0.28
P_CALM_TO_STRESSED = 0.02
P_STRESSED_TO_CALM = 0.10
DRIFT_ANNUAL = 0.08
TRADING_DAYS = 252


def simulate_returns(rng: np.random.Generator, n_days: int) -> tuple[np.ndarray, np.ndarray]:
    daily_drift = DRIFT_ANNUAL / TRADING_DAYS
    regime = np.zeros(n_days, dtype=int)  # 0 = calm, 1 = stressed
    state = 0
    for t in range(n_days):
        regime[t] = state
        flip = P_STRESSED_TO_CALM if state == 1 else P_CALM_TO_STRESSED
        if rng.random() < flip:
            state = 1 - state

    daily_vol = np.where(regime == 0, VOL_CALM, VOL_STRESSED) / np.sqrt(TRADING_DAYS)
    shocks = rng.standard_t(df=5, size=n_days) * daily_vol / np.sqrt(5 / 3)

    # Mild short-horizon autocorrelation: a touch of momentum at lag 1,
    # a touch of mean reversion at lag 5 — both weak, neither deterministic.
    returns = np.zeros(n_days)
    momentum = 0.05
    reversion = -0.04
    for t in range(n_days):
        ar_term = 0.0
        if t >= 1:
            ar_term += momentum * returns[t - 1]
        if t >= 5:
            ar_term += reversion * returns[t - 5]
        returns[t] = ar_term + shocks[t]

    # The AR feedback loop interacting with fat-tailed shocks can shift the
    # realized sample mean well away from the target drift for a given seed
    # (a few large shocks propagate through the momentum term). Recenter the
    # whole series onto the intended daily drift so the annualized return is
    # close to the target regardless of that finite-sample noise, while
    # leaving the autocorrelation/vol-clustering shape untouched.
    returns = returns - returns.mean() + daily_drift
    return returns, regime


def build_ohlcv(rng: np.random.Generator, close: np.ndarray, regime: np.ndarray) -> pd.DataFrame:
    n = len(close)
    prev_close = np.roll(close, 1)
    prev_close[0] = START_PRICE
    intraday_vol = np.where(regime == 0, 0.006, 0.016)
    open_ = prev_close * (1 + rng.normal(0, intraday_vol * 0.3, n))
    high_low_span = np.abs(rng.normal(0, intraday_vol, n)) + np.abs(close - open_) / open_
    high = np.maximum(open_, close) * (1 + high_low_span)
    low = np.minimum(open_, close) * (1 - high_low_span)
    base_volume = 60_000_000
    volume = (base_volume * (1 + 2.5 * (intraday_vol / intraday_vol.min() - 1))
              * (1 + rng.normal(0, 0.15, n)))
    volume = np.clip(volume, 20_000_000, None).astype(np.int64)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


def main() -> None:
    rng = np.random.default_rng(SEED)
    returns, regime = simulate_returns(rng, N_DAYS)
    close = START_PRICE * np.exp(np.cumsum(returns))

    ohlcv = build_ohlcv(rng, close, regime)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=N_DAYS)

    df = pd.DataFrame({"date": dates})
    df = pd.concat([df, ohlcv], axis=1)
    df["vol_regime"] = np.where(regime == 0, "calm", "stressed")
    df = df.round({"open": 2, "high": 2, "low": 2, "close": 2})

    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_PATH}")
    print(f"Close range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"Regime mix: {df['vol_regime'].value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    main()
