"""CRISP-DM Modeling phase — anomaly / outlier detection on invoices.

Isolation Forest flags invoices whose (basket size, quantity, value) profile
is far from the bulk of normal single-customer purchases — this is where the
injected bulk-order invoices from data/generate_data.py should surface.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["n_lines", "total_quantity", "invoice_value", "avg_unit_qty"]


def detect_anomalies(invoice_features: pd.DataFrame, contamination: float = 0.02) -> dict:
    X = invoice_features[FEATURES].to_numpy()
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=42
    )
    labels = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    scores = model.score_samples(X)  # higher = more normal

    result = invoice_features.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = labels == -1

    flagged = result[result["is_anomaly"]].sort_values("anomaly_score").reset_index(drop=True)

    return {
        "scored_invoices": result,
        "flagged": flagged,
        "n_flagged": int(result["is_anomaly"].sum()),
        "contamination": contamination,
        "flagged_share": round(float(result["is_anomaly"].mean()), 4),
    }
