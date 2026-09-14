"""Unsupervised spatial clustering of pickup locations (data understanding)."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans

N_CLUSTERS = 5


def cluster_pickups(df: pd.DataFrame, n_clusters: int = N_CLUSTERS, seed: int = 42):
    coords = df[["pickup_latitude", "pickup_longitude"]].to_numpy()
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = model.fit_predict(coords)
    out = df.copy()
    out["cluster"] = labels

    profile = (
        out.groupby("cluster")
        .agg(
            trip_count=("trip_id", "count"),
            avg_fare=("total_amount", "mean"),
            avg_distance=("trip_distance_miles", "mean"),
            centroid_lat=("pickup_latitude", "mean"),
            centroid_lon=("pickup_longitude", "mean"),
            dominant_zone=("pu_zone", lambda s: s.mode().iat[0]),
        )
        .round(3)
        .reset_index()
    )
    return out, profile, model
