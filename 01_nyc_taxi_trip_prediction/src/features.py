"""Shared feature engineering for training and inference.

Keeping this in one module guarantees the Streamlit app builds features
identically to the training pipeline (no train/serve skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NYC_LAT_BOUNDS = (40.4, 41.1)
NYC_LON_BOUNDS = (-74.35, -73.60)

FEATURE_COLUMNS = [
    "passenger_count",
    "trip_distance_miles",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "is_overnight",
    "is_airport_trip",
    "vendor_id",
]


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def is_within_nyc(lat, lon) -> pd.Series:
    lat_ok = (lat >= NYC_LAT_BOUNDS[0]) & (lat <= NYC_LAT_BOUNDS[1])
    lon_ok = (lon >= NYC_LON_BOUNDS[0]) & (lon <= NYC_LON_BOUNDS[1])
    return lat_ok & lon_ok


JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
AIRPORT_RADIUS_MILES = 1.5


def _near_airport(lat, lon) -> np.ndarray:
    near_jfk = haversine_miles(lat, lon, JFK[0], JFK[1]) <= AIRPORT_RADIUS_MILES
    near_lga = haversine_miles(lat, lon, LGA[0], LGA[1]) <= AIRPORT_RADIUS_MILES
    return near_jfk | near_lga


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw trip rows (coords + datetime) into model-ready features."""
    out = df.copy()
    out["pickup_datetime"] = pd.to_datetime(out["pickup_datetime"])

    out["trip_distance_miles"] = haversine_miles(
        out["pickup_latitude"], out["pickup_longitude"],
        out["dropoff_latitude"], out["dropoff_longitude"],
    )
    out["pickup_hour"] = out["pickup_datetime"].dt.hour
    out["pickup_dayofweek"] = out["pickup_datetime"].dt.dayofweek
    out["pickup_month"] = out["pickup_datetime"].dt.month
    out["is_weekend"] = (out["pickup_dayofweek"] >= 5).astype(int)
    out["is_rush_hour"] = out["pickup_hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    out["is_overnight"] = out["pickup_hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    out["is_airport_trip"] = (
        _near_airport(out["pickup_latitude"], out["pickup_longitude"])
        | _near_airport(out["dropoff_latitude"], out["dropoff_longitude"])
    ).astype(int)

    out["passenger_count"] = out["passenger_count"].fillna(out["passenger_count"].median())
    return out


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """CRISP-DM 'Data Preparation' cleaning rules applied to raw trip rows."""
    clean = df.copy()
    before = len(clean)

    clean = clean[is_within_nyc(clean["pickup_latitude"], clean["pickup_longitude"])]
    clean = clean[is_within_nyc(clean["dropoff_latitude"], clean["dropoff_longitude"])]

    dist = haversine_miles(
        clean["pickup_latitude"], clean["pickup_longitude"],
        clean["dropoff_latitude"], clean["dropoff_longitude"],
    )
    clean = clean[dist > 0.05]

    clean = clean[(clean["trip_duration"] >= 30) & (clean["trip_duration"] <= 3 * 3600)]
    clean = clean[(clean["fare_amount"] >= 2.5) & (clean["fare_amount"] <= 250)]

    removed = before - len(clean)
    clean.attrs["rows_removed"] = removed
    clean.attrs["rows_before"] = before
    return clean
