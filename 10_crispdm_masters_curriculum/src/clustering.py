"""CRISP-DM Modeling phase — unsupervised learning: customer segmentation."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_COLS = ["recency_days", "frequency", "monetary", "avg_basket_value", "product_diversity"]


def segment_customers(customer_features: pd.DataFrame, k_range=range(2, 7)) -> dict:
    X = customer_features[RFM_COLS].to_numpy()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    scores = {}
    models = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        scores[k] = float(silhouette_score(Xs, labels))
        models[k] = (km, labels)

    best_k = max(scores, key=scores.get)
    best_model, best_labels = models[best_k]

    result = customer_features.copy()
    result["segment"] = best_labels

    profile = (
        result.groupby("segment")[RFM_COLS]
        .mean()
        .round(2)
        .reset_index()
    )
    profile["n_customers"] = result.groupby("segment").size().values

    return {
        "assignments": result[["customer_id", "segment"]],
        "profile": profile,
        "silhouette_by_k": {int(k): round(v, 4) for k, v in scores.items()},
        "best_k": int(best_k),
        "best_silhouette": round(scores[best_k], 4),
    }
