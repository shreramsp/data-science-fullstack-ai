"""Feature engineering and the chronological train/test split.

All lag and rolling-window features for a given date are computed only from
that series' own past (``shift`` before ``rolling``), and the holdout is the
last block of calendar days per series rather than a random row sample — the
two guards against leaking future information into the model that a naive
random split on time-series data would introduce.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SALES_PATH = DATA_DIR / "sales.csv"

TEST_DAYS = 56  # 8 weeks held out per series, forward in time
LAGS = (1, 7, 14, 28)
ROLL_WINDOWS = (7, 28)


def load_raw() -> pd.DataFrame:
    if not SALES_PATH.exists():
        from generate_data import generate  # noqa: PLC0415 (script fallback)
        generate().to_csv(SALES_PATH, index=False)
    df = pd.read_csv(SALES_PATH, parse_dates=["date"])
    return df.sort_values(["store_id", "category", "date"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar + lag/rolling features, grouped per (store, category) series."""
    df = df.copy()
    df["series_id"] = df["store_id"] + "_" + df["category"]

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

    grp = df.groupby("series_id")["sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = grp.shift(lag)
    for win in ROLL_WINDOWS:
        # shift(1) first so the window never includes the target day itself.
        df[f"roll_mean_{win}"] = grp.shift(1).rolling(win).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{win}"] = grp.shift(1).rolling(win).std().reset_index(level=0, drop=True)

    return df


FEATURE_COLUMNS = [
    "day_of_week", "month", "day_of_year", "is_weekend", "is_holiday_season", "promo",
    *[f"lag_{l}" for l in LAGS],
    *[f"roll_mean_{w}" for w in ROLL_WINDOWS],
    *[f"roll_std_{w}" for w in ROLL_WINDOWS],
]


def chronological_split(df: pd.DataFrame, test_days: int = TEST_DAYS):
    """Split each series' tail `test_days` off as test; everything earlier is train."""
    cutoff = df["date"].max() - pd.Timedelta(days=test_days - 1)
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test


def prepare():
    """Load raw sales, engineer features, drop warm-up rows with NaN lags, split."""
    raw = load_raw()
    feat = add_features(raw)
    feat = feat.dropna(subset=[f"lag_{max(LAGS)}", f"roll_mean_{max(ROLL_WINDOWS)}"]).reset_index(drop=True)
    train, test = chronological_split(feat)
    return raw, feat, train, test
