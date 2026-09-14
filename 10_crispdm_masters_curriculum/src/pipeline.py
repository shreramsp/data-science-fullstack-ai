"""Runs the full CRISP-DM pipeline end to end and writes artifacts/ for app.py.

Usage:
    python3.13 src/pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import association, anomaly, clustering, features, lsh, supervised

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "online_retail.csv")


def main() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    t_start = time.time()

    # --- Data Understanding + Data Preparation ---
    raw = features.load_raw(DATA_PATH)
    clean_df, cleaning_report = features.clean(raw)

    eda_summary = {
        "n_line_items": int(len(clean_df)),
        "n_invoices": int(clean_df["invoice_id"].nunique()),
        "n_customers": int(clean_df["customer_id"].nunique()),
        "n_products": int(clean_df["product_id"].nunique()),
        "n_categories": int(clean_df["category"].nunique()),
        "date_min": str(clean_df["invoice_date"].min().date()),
        "date_max": str(clean_df["invoice_date"].max().date()),
        "total_revenue": round(float(clean_df["line_total"].sum()), 2),
        "avg_line_value": round(float(clean_df["line_total"].mean()), 2),
        "cleaning_report": cleaning_report,
    }

    customer_features = features.build_customer_features(clean_df)
    invoice_features = features.build_invoice_features(clean_df)
    baskets = features.build_baskets(clean_df)

    customer_features.to_csv(os.path.join(ARTIFACTS_DIR, "customer_features.csv"), index=False)
    invoice_features.to_csv(os.path.join(ARTIFACTS_DIR, "invoice_features.csv"), index=False)

    category_mix = (
        clean_df.groupby("category")["line_total"].sum().sort_values(ascending=False).round(2)
    )
    category_mix.to_csv(os.path.join(ARTIFACTS_DIR, "category_mix.csv"))

    monthly_revenue = (
        clean_df.set_index("invoice_date")["line_total"].resample("ME").sum().round(2)
    )
    monthly_revenue.to_csv(os.path.join(ARTIFACTS_DIR, "monthly_revenue.csv"))

    # --- Modeling: Clustering ---
    cluster_result = clustering.segment_customers(customer_features)
    cluster_result["assignments"].to_csv(os.path.join(ARTIFACTS_DIR, "customer_segments.csv"), index=False)
    cluster_result["profile"].to_csv(os.path.join(ARTIFACTS_DIR, "segment_profile.csv"), index=False)

    # --- Modeling: Anomaly Detection ---
    anomaly_result = anomaly.detect_anomalies(invoice_features)
    anomaly_result["scored_invoices"].to_csv(os.path.join(ARTIFACTS_DIR, "invoice_anomaly_scores.csv"), index=False)
    anomaly_result["flagged"].to_csv(os.path.join(ARTIFACTS_DIR, "flagged_invoices.csv"), index=False)

    # --- Modeling: Supervised churn-risk ---
    supervised_result = supervised.train_churn_model(customer_features)
    supervised_result["feature_importance"].to_csv(
        os.path.join(ARTIFACTS_DIR, "churn_feature_importance.csv"), index=False
    )

    # --- Modeling: Association rules ---
    assoc_result = association.mine_rules(baskets)
    assoc_result["rules"].to_csv(os.path.join(ARTIFACTS_DIR, "association_rules.csv"), index=False)

    # --- Modeling: LSH sub-linear search ---
    lsh_result = lsh.run_benchmark(clean_df)

    summary = {
        "eda": eda_summary,
        "clustering": {
            "silhouette_by_k": cluster_result["silhouette_by_k"],
            "best_k": cluster_result["best_k"],
            "best_silhouette": cluster_result["best_silhouette"],
        },
        "anomaly_detection": {
            "n_flagged": anomaly_result["n_flagged"],
            "contamination": anomaly_result["contamination"],
            "flagged_share": anomaly_result["flagged_share"],
        },
        "supervised_learning": {
            "metrics": supervised_result["metrics"],
            "median_recency_days": supervised_result["median_recency_days"],
        },
        "association_rules": {
            "n_baskets": assoc_result["n_baskets"],
            "n_frequent_itemsets": assoc_result["n_frequent_itemsets"],
            "n_rules": assoc_result["n_rules"],
        },
        "lsh": lsh_result,
        "pipeline_runtime_sec": round(time.time() - t_start, 2),
    }

    with open(os.path.join(ARTIFACTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
