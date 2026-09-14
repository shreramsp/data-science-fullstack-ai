"""Leakage-safe feature engineering and chronological train/test split for
the synthetic SPY-like daily series.

Every feature is computed with `shift()` applied before any `rolling()` /
`ewm()` window, so the feature for day *t* only ever sees information from
day *t-1* and earlier. Forward targets (`fwd_return_1`, `fwd_return_5`) are
computed with negative shifts and are dropped from the feature matrix.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "spy_prices.csv"
HORIZONS = (1, 5)
EMBARGO_DAYS = 5  # dropped on both sides of the split boundary
TEST_FRACTION = 0.2

FEATURE_COLUMNS = [
    "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_5", "lag_return_10",
    "roll_mean_return_5", "roll_std_return_5", "roll_mean_return_20", "roll_std_return_20",
    "momentum_10", "rsi_14", "macd", "macd_signal", "macd_hist",
    "volume_z_20", "high_low_range",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.shift(1)  # value known as of the close *before* today


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).shift(1)
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def load_raw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "data" / "generate_data.py")], check=True)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    for lag in (1, 2, 3, 5, 10):
        df[f"lag_return_{lag}"] = df["log_return"].shift(lag)

    # shift(1) first so the rolling window at day t only covers days < t
    shifted_return = df["log_return"].shift(1)
    df["roll_mean_return_5"] = shifted_return.rolling(5).mean()
    df["roll_std_return_5"] = shifted_return.rolling(5).std()
    df["roll_mean_return_20"] = shifted_return.rolling(20).mean()
    df["roll_std_return_20"] = shifted_return.rolling(20).std()

    df["momentum_10"] = (df["close"].shift(1) / df["close"].shift(11) - 1)
    df["rsi_14"] = _rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["close"])

    volume_shifted = df["volume"].shift(1)
    vol_roll_mean = volume_shifted.rolling(20).mean()
    vol_roll_std = volume_shifted.rolling(20).std()
    df["volume_z_20"] = (volume_shifted - vol_roll_mean) / vol_roll_std.replace(0, np.nan)

    df["high_low_range"] = ((df["high"].shift(1) - df["low"].shift(1)) / df["close"].shift(1))

    for h in HORIZONS:
        df[f"fwd_return_{h}"] = np.log(df["close"].shift(-h) / df["close"])

    return df


def chronological_split(feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Time-ordered split with an embargo gap around the boundary so no
    rolling/forward-looking window straddles train and test."""
    feat = feat.sort_values("date").reset_index(drop=True)
    split_idx = int(len(feat) * (1 - TEST_FRACTION))
    cutoff_date = feat.loc[split_idx, "date"]

    train_end = split_idx - EMBARGO_DAYS
    test_start = split_idx + EMBARGO_DAYS
    train = feat.iloc[:train_end].copy()
    test = feat.iloc[test_start:].copy()
    return train, test, cutoff_date


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Returns (raw, full_feature_table, train_clean, test_clean, cutoff_date).

    train_clean / test_clean drop rows with any NaN in the feature or
    target columns needed for the requested horizons (warm-up window at
    the start of history, and the last H rows of history for horizon H).
    """
    raw = load_raw()
    feat = engineer_features(raw)
    train, test, cutoff = chronological_split(feat)

    needed = FEATURE_COLUMNS + [f"fwd_return_{h}" for h in HORIZONS]
    train_clean = train.dropna(subset=needed).reset_index(drop=True)
    test_clean = test.dropna(subset=needed).reset_index(drop=True)
    return raw, feat, train_clean, test_clean, cutoff


if __name__ == "__main__":
    raw, feat, train, test, cutoff = prepare()
    print(f"raw={len(raw)} feat={len(feat)} train={len(train)} test={len(test)} cutoff={cutoff.date()}")
