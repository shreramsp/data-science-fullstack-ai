"""CRISP-DM modeling + evaluation stage for the NYC taxi project.

Run with:  python src/train.py

Loads the generated trip data (regenerating it if missing), cleans it,
engineers features, trains one RandomForestRegressor for trip_duration and
one for fare_amount, evaluates both honestly on a held-out test split, and
saves the models + metrics to models/ for the Streamlit app to load.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_COLUMNS, build_features, clean_raw  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nyc_taxi_trips.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_raw() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        gen_path = os.path.join(BASE_DIR, "data", "generate_data.py")
        os.system(f"{sys.executable} {gen_path}")
    return pd.read_csv(DATA_PATH, parse_dates=["pickup_datetime"])


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    raw = load_raw()
    cleaned = clean_raw(raw)
    rows_removed = cleaned.attrs.get("rows_removed", 0)
    rows_before = cleaned.attrs.get("rows_before", len(raw))
    print(f"Data Preparation: removed {rows_removed}/{rows_before} rows during cleaning "
          f"({rows_removed / rows_before:.1%})")

    features = build_features(cleaned)
    X = features[FEATURE_COLUMNS]
    y_duration = features["trip_duration"]
    y_fare = features["fare_amount"]

    X_train, X_test, ydur_train, ydur_test, yfare_train, yfare_test = train_test_split(
        X, y_duration, y_fare, test_size=0.2, random_state=42
    )

    duration_model = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42
    )
    duration_model.fit(X_train, ydur_train)
    duration_metrics = evaluate(ydur_test, duration_model.predict(X_test))

    fare_model = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42
    )
    fare_model.fit(X_train, yfare_train)
    fare_metrics = evaluate(yfare_test, fare_model.predict(X_test))

    duration_importance = dict(
        sorted(zip(FEATURE_COLUMNS, duration_model.feature_importances_.tolist()),
               key=lambda kv: kv[1], reverse=True)
    )
    fare_importance = dict(
        sorted(zip(FEATURE_COLUMNS, fare_model.feature_importances_.tolist()),
               key=lambda kv: kv[1], reverse=True)
    )

    joblib.dump(duration_model, os.path.join(MODELS_DIR, "duration_model.joblib"))
    joblib.dump(fare_model, os.path.join(MODELS_DIR, "fare_model.joblib"))

    metrics = {
        "n_rows_raw": int(rows_before),
        "n_rows_cleaned": int(len(cleaned)),
        "rows_removed_in_cleaning": int(rows_removed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": FEATURE_COLUMNS,
        "duration_model": {
            "type": "RandomForestRegressor",
            "target": "trip_duration (seconds)",
            "test_metrics": duration_metrics,
            "feature_importance": duration_importance,
        },
        "fare_model": {
            "type": "RandomForestRegressor",
            "target": "fare_amount (USD)",
            "test_metrics": fare_metrics,
            "feature_importance": fare_importance,
        },
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation (held-out test set):")
    print(f"  Duration model -> MAE: {duration_metrics['mae']:.1f}s  "
          f"RMSE: {duration_metrics['rmse']:.1f}s  R2: {duration_metrics['r2']:.3f}")
    print(f"  Fare model     -> MAE: ${fare_metrics['mae']:.2f}  "
          f"RMSE: ${fare_metrics['rmse']:.2f}  R2: {fare_metrics['r2']:.3f}")
    print(f"\nSaved models + metrics.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
