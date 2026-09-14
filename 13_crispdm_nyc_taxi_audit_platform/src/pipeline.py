"""End-to-end CRISP-DM pipeline: load -> audit raw -> clean -> feature
engineer -> cluster -> split -> train/compare -> explain -> audit split &
leakage -> write artifacts. Run directly with `python -m src.pipeline` or
imported by `app.py` (which caches the same steps for the UI)."""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd

from . import audit
from .clustering import cluster_pickups
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, build_features, clean_raw
from .modeling import explain_with_permutation_importance, select_best, split_data, train_and_compare

SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nyc_tlc_trips.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def load_raw() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        from data.generate_data import generate

        df = generate()
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH, parse_dates=["pickup_datetime", "dropoff_datetime"])


def run_pipeline() -> dict:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    raw_df = load_raw()
    raw_quality = audit.audit_raw_data(raw_df)

    clean_df = clean_raw(raw_df)
    feat_df = build_features(clean_df)

    clustered_df, cluster_profile, _ = cluster_pickups(feat_df, seed=SEED)

    train_df, test_df = split_data(feat_df, seed=SEED)
    results = train_and_compare(train_df, test_df)
    best_name = select_best(results)
    best_result = results[best_name]

    importance_df = explain_with_permutation_importance(best_result, test_df, seed=SEED)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    full_audit = audit.build_full_audit_report(
        raw_df=raw_df,
        clean_df=clean_df,
        feature_columns=feature_cols,
        target_column=TARGET_COLUMN,
        train_df=train_df,
        test_df=test_df,
        seed=SEED,
    )

    metrics = {name: r.metrics for name, r in results.items()}

    joblib.dump(best_result.pipeline, os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "model_comparison": metrics}, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "audit_report.json"), "w") as f:
        json.dump(full_audit, f, indent=2, default=str)
    cluster_profile.to_csv(os.path.join(ARTIFACTS_DIR, "cluster_profile.csv"), index=False)
    importance_df.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False)

    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "feat_df": feat_df,
        "clustered_df": clustered_df,
        "cluster_profile": cluster_profile,
        "train_df": train_df,
        "test_df": test_df,
        "results": results,
        "best_name": best_name,
        "importance_df": importance_df,
        "audit_report": full_audit,
        "raw_quality": raw_quality,
    }


if __name__ == "__main__":
    out = run_pipeline()
    print(f"Best model: {out['best_name']}")
    print(json.dumps(out["results"][out["best_name"]].metrics, indent=2))
    print("Artifacts written to", ARTIFACTS_DIR)
