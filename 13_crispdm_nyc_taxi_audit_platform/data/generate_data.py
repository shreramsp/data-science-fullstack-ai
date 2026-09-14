"""Seeded synthetic NYC TLC-style trip generator.

The real NYC TLC trip record parquet files are multiple GB per month and
require no-auth-but-large downloads that break `git clone && pip install`
reproducibility. This generator produces a small, fully reproducible
dataset with the same *shape* as a TLC trip record (pickup/dropoff
zone + coordinates, distance, fare components, payment type) and
deliberately injects a slice of realistic data-quality defects
(nulls, duplicates, negative fares, out-of-bounds coordinates) so the
audit stage of this platform has real issues to detect rather than a
synthetic report of a clean dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_TRIPS = 8000

# Approximate borough centroids (lat, lon) used as trip-generation hotspots.
# Coordinates are coarse public-knowledge borough centers.
ZONES = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
}
ZONE_NAMES = list(ZONES.keys())
# Manhattan and the airports are far busier than the outer-borough zones.
ZONE_WEIGHTS = np.array([0.42, 0.16, 0.12, 0.07, 0.03, 0.10, 0.10])
ZONE_WEIGHTS = ZONE_WEIGHTS / ZONE_WEIGHTS.sum()

PAYMENT_TYPES = ["card", "cash", "mobile_wallet", "no_charge"]
PAYMENT_WEIGHTS = [0.62, 0.28, 0.08, 0.02]

NYC_LAT_RANGE = (40.48, 40.93)
NYC_LON_RANGE = (-74.27, -73.68)


def _haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _sample_point(rng, zone_name):
    lat0, lon0 = ZONES[zone_name]
    # ~0.5-2.5 mile jitter around the zone centroid.
    lat = lat0 + rng.normal(0, 0.02)
    lon = lon0 + rng.normal(0, 0.02)
    return lat, lon


def generate(n_trips: int = N_TRIPS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pu_zone = rng.choice(ZONE_NAMES, size=n_trips, p=ZONE_WEIGHTS)
    do_zone = rng.choice(ZONE_NAMES, size=n_trips, p=ZONE_WEIGHTS)

    pu_lat, pu_lon, do_lat, do_lon = [], [], [], []
    for pz, dz in zip(pu_zone, do_zone):
        la, lo = _sample_point(rng, pz)
        pu_lat.append(la)
        pu_lon.append(lo)
        la, lo = _sample_point(rng, dz)
        do_lat.append(la)
        do_lon.append(lo)
    pu_lat, pu_lon = np.array(pu_lat), np.array(pu_lon)
    do_lat, do_lon = np.array(do_lat), np.array(do_lon)

    distance = _haversine_miles(pu_lat, pu_lon, do_lat, do_lon)
    distance = np.clip(distance + rng.normal(0, 0.3, n_trips), 0.1, None)

    start = pd.Timestamp("2024-01-01")
    minutes_offset = rng.integers(0, 60 * 24 * 90, n_trips)  # 90-day window
    pickup_dt = start + pd.to_timedelta(minutes_offset, unit="m")
    hour = pickup_dt.hour.to_numpy()
    is_rush = np.isin(hour, [7, 8, 9, 16, 17, 18, 19]).astype(float)
    is_airport = np.isin(pu_zone, ["JFK Airport", "LaGuardia Airport"]) | np.isin(
        do_zone, ["JFK Airport", "LaGuardia Airport"]
    )

    avg_speed_mph = 14 - 5 * is_rush + rng.normal(0, 2, n_trips)
    avg_speed_mph = np.clip(avg_speed_mph, 3, None)
    duration_min = (distance / avg_speed_mph) * 60
    duration_min = np.clip(duration_min + rng.normal(0, 2, n_trips), 1, None)
    dropoff_dt = pickup_dt + pd.to_timedelta(duration_min, unit="m")

    passenger_count = rng.integers(1, 5, n_trips)
    vendor_id = rng.choice(["V1", "V2"], size=n_trips)
    payment_type = rng.choice(PAYMENT_TYPES, size=n_trips, p=PAYMENT_WEIGHTS)
    rate_code = np.where(is_airport, "airport_flat", "standard")

    base_fare = 3.00
    per_mile = 2.75
    per_minute = 0.45
    fare_amount = (
        base_fare
        + per_mile * distance
        + per_minute * duration_min
        + rng.normal(0, 1.2, n_trips)
    )
    fare_amount = np.clip(fare_amount, 3.5, None)
    congestion_surcharge = np.where(pu_zone == "Manhattan", 2.75, 0.0)
    tolls_amount = np.where(is_airport, rng.choice([0.0, 6.94], size=n_trips, p=[0.4, 0.6]), 0.0)
    tip_rate = np.where(payment_type == "cash", 0.0, rng.uniform(0.05, 0.25, n_trips))
    tip_amount = np.round(fare_amount * tip_rate, 2)
    total_amount = np.round(fare_amount + tolls_amount + congestion_surcharge + tip_amount, 2)

    df = pd.DataFrame(
        {
            "trip_id": np.arange(1, n_trips + 1),
            "vendor_id": vendor_id,
            "pickup_datetime": pickup_dt,
            "dropoff_datetime": dropoff_dt,
            "passenger_count": passenger_count,
            "pu_zone": pu_zone,
            "do_zone": do_zone,
            "pickup_latitude": pu_lat,
            "pickup_longitude": pu_lon,
            "dropoff_latitude": do_lat,
            "dropoff_longitude": do_lon,
            "trip_distance_miles": np.round(distance, 3),
            "rate_code": rate_code,
            "payment_type": payment_type,
            "fare_amount": np.round(fare_amount, 2),
            "tip_amount": tip_amount,
            "tolls_amount": tolls_amount,
            "congestion_surcharge": congestion_surcharge,
            "total_amount": total_amount,
        }
    )

    # --- Inject realistic data-quality defects for the audit stage ---
    n_bad = int(n_trips * 0.03)
    bad_idx = rng.choice(n_trips, size=n_bad, replace=False)

    # a) missing passenger_count / payment_type
    df.loc[bad_idx[: n_bad // 3], "passenger_count"] = np.nan
    df.loc[bad_idx[n_bad // 3 : 2 * n_bad // 3], "payment_type"] = None

    # b) out-of-NYC-bounds coordinates (GPS glitches)
    glitch_idx = bad_idx[2 * n_bad // 3 :]
    df.loc[glitch_idx, "pickup_latitude"] = 0.0
    df.loc[glitch_idx, "pickup_longitude"] = 0.0

    # c) negative / zero fare rows (billing system errors)
    neg_idx = rng.choice(n_trips, size=max(1, n_trips // 400), replace=False)
    df.loc[neg_idx, "fare_amount"] = -abs(df.loc[neg_idx, "fare_amount"])
    df.loc[neg_idx, "total_amount"] = df.loc[neg_idx, "fare_amount"]

    # d) exact duplicate rows (upstream ingestion double-write)
    dup_rows = df.sample(n=max(1, n_trips // 200), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = generate()
    out.to_csv(__file__.replace("generate_data.py", "nyc_tlc_trips.csv"), index=False)
    print(f"Wrote {len(out)} rows to data/nyc_tlc_trips.csv")
