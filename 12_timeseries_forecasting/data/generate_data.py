"""Seeded synthetic daily retail-sales generator.

Mimics the shape of a popular Kaggle time-series forecasting dataset (e.g.
Store Sales / Walmart Sales Forecasting): a date axis crossed with a handful
of store x category series, each with its own trend, weekly seasonality,
yearly seasonality, a promo effect and noise. Used instead of downloading a
real multi-hundred-MB competition dataset. Every number this project reports
describes this generated
dataset, not any real store's sales.

Run:  python data/generate_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260913
DATA_DIR = Path(__file__).resolve().parent
OUT_PATH = DATA_DIR / "sales.csv"

START_DATE = "2023-01-01"
N_DAYS = 3 * 365  # ~3 years of daily history

STORES = ["S1", "S2", "S3"]
CATEGORIES = ["Grocery", "Electronics", "Apparel"]

# Per-category base level, yearly-growth rate, and weekly/yearly amplitude —
# fixed so the generator is deterministic given SEED.
CATEGORY_PARAMS = {
    "Grocery":     dict(base=220.0, annual_growth=0.06, weekly_amp=0.25, yearly_amp=0.15, noise_sigma=0.09),
    "Electronics": dict(base=90.0,  annual_growth=0.18, weekly_amp=0.40, yearly_amp=0.35, noise_sigma=0.16),
    "Apparel":     dict(base=140.0, annual_growth=0.10, weekly_amp=0.30, yearly_amp=0.45, noise_sigma=0.13),
}
STORE_MULT = {"S1": 1.15, "S2": 1.00, "S3": 0.80}

# Monday..Sunday multiplicative weekday effect (retail: weekends busier).
WEEKDAY_EFFECT = np.array([0.90, 0.88, 0.92, 0.97, 1.10, 1.35, 1.25])


def _yearly_seasonal(day_of_year: np.ndarray, amp: float) -> np.ndarray:
    # Smooth annual cycle plus a November-December holiday bump.
    base = 1.0 + amp * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    holiday_bump = amp * 0.8 * np.exp(-0.5 * ((day_of_year - 340) / 12) ** 2)
    return base + holiday_bump


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()  # Monday=0
    t = np.arange(N_DAYS)

    rows = []
    for store in STORES:
        # Deterministic per-store promo calendar: ~1 promo week every ~5 weeks.
        promo = np.zeros(N_DAYS, dtype=int)
        promo_starts = rng.choice(np.arange(14, N_DAYS - 7), size=N_DAYS // 35, replace=False)
        for s in promo_starts:
            promo[s:s + 5] = 1

        for cat in CATEGORIES:
            p = CATEGORY_PARAMS[cat]
            trend = 1.0 + p["annual_growth"] * (t / 365.25)
            weekly = 1.0 + p["weekly_amp"] * (WEEKDAY_EFFECT[dow] - 1.0)
            yearly = _yearly_seasonal(doy, p["yearly_amp"])
            promo_effect = 1.0 + 0.35 * promo
            level = p["base"] * STORE_MULT[store] * trend * weekly * yearly * promo_effect
            noise = rng.normal(1.0, p["noise_sigma"], size=N_DAYS)
            sales = np.clip(level * noise, a_min=0, a_max=None)
            sales = rng.poisson(np.maximum(sales, 0.5)).astype(float)

            rows.append(pd.DataFrame({
                "date": dates,
                "store_id": store,
                "category": cat,
                "promo": promo,
                "sales": sales,
            }))

    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values(["store_id", "category", "date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows ({df['date'].min().date()} -> {df['date'].max().date()}, "
          f"{df['store_id'].nunique()} stores x {df['category'].nunique()} categories) -> {OUT_PATH}")
