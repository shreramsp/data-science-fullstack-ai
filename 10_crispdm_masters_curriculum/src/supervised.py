"""CRISP-DM Modeling phase — supervised learning: churn-risk classification.

Target: `churn_risk` = 1 if a customer's recency is above the population
median (i.e. they haven't purchased in a while), else 0. Recency itself is
excluded from the feature set (it defines the label) so the model has to
learn the risk signal from frequency/monetary/diversity behavior instead of
trivially re-deriving the label — the standard leakage guard for this setup.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

FEATURE_COLS = ["frequency", "monetary", "avg_basket_value", "product_diversity", "category_diversity"]


def train_churn_model(customer_features: pd.DataFrame, test_size: float = 0.25) -> dict:
    df = customer_features.copy()
    median_recency = df["recency_days"].median()
    df["churn_risk"] = (df["recency_days"] > median_recency).astype(int)

    X = df[FEATURE_COLS]
    y = df["churn_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
    }

    importance = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "metrics": metrics,
        "feature_importance": importance,
        "median_recency_days": round(float(median_recency), 1),
    }
