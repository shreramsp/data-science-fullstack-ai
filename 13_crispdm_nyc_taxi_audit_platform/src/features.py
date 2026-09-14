"""Data preparation: cleaning rules and feature engineering.

Kept in one shared module so both the training pipeline and the Streamlit
app apply identical transformations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NYC_LAT_RANGE = (40.48, 40.93)
NYC_LON_RANGE = (-74.27, -73.68)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cleaning: drop duplicates, out-of-bounds GPS, negative
    fares, non-positive distances, and impute the small share of missing
    categorical/count fields. All rules are documented here so an auditor
    can see exactly what happened between raw and clean data."""
    out = df.drop_duplicates(subset=[c for c in df.columns if c != "trip_id"]).copy()

    in_bounds = out["pickup_latitude"].between(*NYC_LAT_RANGE) & out["pickup_longitude"].between(
        *NYC_LON_RANGE
    )
    out = out[in_bounds]
    out = out[out["fare_amount"] > 0]
    out = out[out["trip_distance_miles"] > 0]

    out["passenger_count"] = out["passenger_count"].fillna(out["passenger_count"].median())
    out["payment_type"] = out["payment_type"].fillna("unknown")

    return out.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering applied after cleaning. Never uses
    `fare_amount`/`total_amount` to derive predictor columns (would be
    target leakage for the fare-prediction model)."""
    out = df.copy()
    out["pickup_datetime"] = pd.to_datetime(out["pickup_datetime"])
    out["pickup_hour"] = out["pickup_datetime"].dt.hour
    out["pickup_dayofweek"] = out["pickup_datetime"].dt.dayofweek
    out["is_weekend"] = out["pickup_dayofweek"].isin([5, 6]).astype(int)
    out["is_rush_hour"] = out["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    out["is_airport_trip"] = (
        out["pu_zone"].isin(["JFK Airport", "LaGuardia Airport"])
        | out["do_zone"].isin(["JFK Airport", "LaGuardia Airport"])
    ).astype(int)
    return out


NUMERIC_FEATURES = [
    "trip_distance_miles",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "is_rush_hour",
    "is_airport_trip",
]
CATEGORICAL_FEATURES = ["pu_zone", "do_zone", "payment_type", "rate_code", "vendor_id"]
TARGET_COLUMN = "total_amount"
