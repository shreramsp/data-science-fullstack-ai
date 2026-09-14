"""Generate a small, reproducible, synthetic NYC-taxi-like trip dataset.

The real Kaggle "New York City Taxi Trip Duration" dataset is > 1.5 GB and
requires Kaggle authentication, which makes it unsuitable for a reproducible
local project. This script generates a structurally equivalent dataset
(same columns, same kind of geospatial/temporal patterns, same targets) from
scratch using a fixed random seed, so anyone can regenerate it with
`python data/generate_data.py`.

Columns mimic the raw Kaggle schema plus a fare_amount target:
    id, vendor_id, pickup_datetime, passenger_count,
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
    trip_duration (seconds, target), fare_amount (dollars, target)

A small fraction of rows are deliberately injected with unrealistic values
(zero-distance trips, near-instant/absurdly long durations, missing
passenger counts) so the modeling pipeline has real data-cleaning work to do
during the CRISP-DM "Data Preparation" phase, instead of starting from an
artificially perfect dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 9000
START_DATE = pd.Timestamp("2016-01-01")
END_DATE = pd.Timestamp("2016-06-30 23:59:59")

# (name, lat, lon, weight, std_dev_degrees) — rough NYC hotspot clusters.
HOTSPOTS = [
    ("Midtown Manhattan", 40.7549, -73.9840, 0.22, 0.010),
    ("Downtown / FiDi", 40.7075, -74.0113, 0.14, 0.008),
    ("Upper West Side", 40.7870, -73.9754, 0.10, 0.008),
    ("Upper East Side", 40.7736, -73.9566, 0.10, 0.008),
    ("Chelsea / Village", 40.7420, -74.0000, 0.12, 0.008),
    ("Williamsburg, Brooklyn", 40.7081, -73.9571, 0.08, 0.010),
    ("Long Island City, Queens", 40.7447, -73.9485, 0.06, 0.010),
    ("JFK Airport", 40.6413, -73.7781, 0.09, 0.004),
    ("LaGuardia Airport", 40.7769, -73.8740, 0.07, 0.004),
    ("Central Park", 40.7812, -73.9665, 0.02, 0.006),
]

NYC_LAT_BOUNDS = (40.55, 40.95)
NYC_LON_BOUNDS = (-74.10, -73.70)


def _sample_points(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = [h[0] for h in HOTSPOTS]
    weights = np.array([h[3] for h in HOTSPOTS])
    weights = weights / weights.sum()
    idx = rng.choice(len(HOTSPOTS), size=n, p=weights)
    lats = np.empty(n)
    lons = np.empty(n)
    for i, cluster_idx in enumerate(idx):
        _, lat, lon, _, std = HOTSPOTS[cluster_idx]
        lats[i] = rng.normal(lat, std)
        lons[i] = rng.normal(lon, std)
    lats = np.clip(lats, *NYC_LAT_BOUNDS)
    lons = np.clip(lons, *NYC_LON_BOUNDS)
    cluster_names = np.array(names)[idx]
    return lats, lons, cluster_names


def _haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 3958.8  # earth radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _hour_weights() -> np.ndarray:
    # Two demand peaks (morning commute, evening commute/nightlife), quiet overnight.
    hours = np.arange(24)
    morning = np.exp(-((hours - 8.5) ** 2) / (2 * 2.0**2))
    evening = np.exp(-((hours - 18.5) ** 2) / (2 * 3.0**2))
    base = 0.15
    w = base + morning + 0.9 * evening
    return w / w.sum()


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pickup_lat, pickup_lon, pickup_cluster = _sample_points(rng, n_rows)
    dropoff_lat, dropoff_lon, dropoff_cluster = _sample_points(rng, n_rows)
    # Nudge a fraction of dropoffs to differ meaningfully from an identical
    # pickup cluster so we don't get too many trivially short trips.
    same_cluster = pickup_cluster == dropoff_cluster
    jitter = rng.normal(0, 0.015, size=(same_cluster.sum(), 2))
    dropoff_lat[same_cluster] += jitter[:, 0]
    dropoff_lon[same_cluster] += jitter[:, 1]

    # Temporal pattern: random days in range, hour drawn from a demand curve.
    total_days = (END_DATE - START_DATE).days
    day_offsets = rng.integers(0, total_days + 1, size=n_rows)
    hour_probs = _hour_weights()
    hours = rng.choice(24, size=n_rows, p=hour_probs)
    minutes = rng.integers(0, 60, size=n_rows)
    seconds = rng.integers(0, 60, size=n_rows)
    pickup_datetime = (
        START_DATE
        + pd.to_timedelta(day_offsets, unit="D")
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    )

    vendor_id = rng.choice([1, 2], size=n_rows, p=[0.45, 0.55])
    passenger_count = rng.choice(
        [1, 2, 3, 4, 5, 6], size=n_rows, p=[0.60, 0.18, 0.08, 0.06, 0.05, 0.03]
    )

    distance_miles = _haversine_miles(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    distance_miles = np.clip(distance_miles, 0.05, None)

    is_rush_hour = np.isin(hours, [7, 8, 9, 16, 17, 18, 19])
    is_overnight = np.isin(hours, [0, 1, 2, 3, 4, 5])
    avg_speed_mph = np.where(is_rush_hour, rng.normal(9, 2, n_rows),
                     np.where(is_overnight, rng.normal(24, 4, n_rows),
                              rng.normal(15, 3, n_rows)))
    avg_speed_mph = np.clip(avg_speed_mph, 3, 35)

    dwell_seconds = rng.normal(90, 30, n_rows).clip(20, None)
    duration_noise = rng.normal(1.0, 0.08, n_rows)
    trip_duration = (distance_miles / avg_speed_mph) * 3600 * duration_noise + dwell_seconds
    trip_duration = trip_duration.clip(30, None)

    is_airport = np.isin(pickup_cluster, ["JFK Airport", "LaGuardia Airport"]) | np.isin(
        dropoff_cluster, ["JFK Airport", "LaGuardia Airport"]
    )
    peak_surcharge = np.where((is_rush_hour) & (pickup_datetime.dayofweek < 5), 1.0, 0.0)
    overnight_surcharge = np.where(is_overnight, 0.5, 0.0)
    airport_fee = np.where(is_airport, rng.normal(15, 3, n_rows).clip(5, None), 0.0)
    fare_noise = rng.normal(0, 1.2, n_rows)
    fare_amount = (
        2.50
        + 2.5 * distance_miles
        + 0.35 * (trip_duration / 60.0)
        + peak_surcharge
        + overnight_surcharge
        + airport_fee
        + fare_noise
    )
    fare_amount = fare_amount.clip(3.0, None)

    df = pd.DataFrame(
        {
            "id": [f"trip_{i:06d}" for i in range(n_rows)],
            "vendor_id": vendor_id,
            "pickup_datetime": pickup_datetime,
            "passenger_count": passenger_count,
            "pickup_longitude": pickup_lon,
            "pickup_latitude": pickup_lat,
            "dropoff_longitude": dropoff_lon,
            "dropoff_latitude": dropoff_lat,
            "trip_duration": trip_duration.round(0).astype(int),
            "fare_amount": fare_amount.round(2),
        }
    )

    # --- Inject realistic messiness for the pipeline to clean up. ---
    n_dirty = int(n_rows * 0.03)
    dirty_idx = rng.choice(n_rows, size=n_dirty, replace=False)
    thirds = np.array_split(dirty_idx, 3)

    # Zero-distance / same-point GPS glitches.
    df.loc[thirds[0], "dropoff_longitude"] = df.loc[thirds[0], "pickup_longitude"]
    df.loc[thirds[0], "dropoff_latitude"] = df.loc[thirds[0], "pickup_latitude"]

    # Absurd duration outliers (stuck meter / bad GPS ping).
    df.loc[thirds[1], "trip_duration"] = rng.integers(3 * 3600, 6 * 3600, size=len(thirds[1]))

    # Missing passenger counts (sensor/logging gap).
    df.loc[thirds[2], "passenger_count"] = np.nan

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    frame = generate()
    out_path = __file__.replace("generate_data.py", "nyc_taxi_trips.csv")
    frame.to_csv(out_path, index=False)
    print(f"Wrote {len(frame):,} rows to {out_path}")
    print(frame.describe(include="all").T)
