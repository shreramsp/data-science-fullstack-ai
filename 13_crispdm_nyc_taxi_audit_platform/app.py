"""CRISP-DM NYC TLC Audit Platform — Streamlit front end.

Walks a data-science / code auditor through all six CRISP-DM phases for a
NYC-taxi-style fare-prediction project, ending in a dedicated Audit Report
tab that surfaces the automated leakage, reproducibility, and data-quality
checks from `src/audit.py`.
"""

from __future__ import annotations

import json

import pandas as pd
import pydeck as pdk
import streamlit as st

from src.pipeline import run_pipeline

st.set_page_config(page_title="CRISP-DM NYC TLC Audit Platform", layout="wide")


@st.cache_resource(show_spinner="Running CRISP-DM pipeline (generate -> audit -> clean -> cluster -> model -> explain)...")
def get_pipeline_output():
    return run_pipeline()


out = get_pipeline_output()

st.title("CRISP-DM NYC TLC Audit Platform")
st.caption(
    "An end-to-end, auditor-transparent CRISP-DM walkthrough on a synthetic "
    "NYC TLC-style trip dataset: every phase below shows the data behind it "
    "and the Audit Report tab shows the automated checks a code/data-science "
    "auditor would run."
)

tabs = st.tabs(
    [
        "1. Business Understanding",
        "2. Data Understanding",
        "3. Data Preparation",
        "4. Modeling & Evaluation",
        "5. Explainability",
        "6. Spatial Clustering",
        "7. Audit Report",
        "8. Fare Estimator",
    ]
)

with tabs[0]:
    st.header("Business Understanding")
    st.markdown(
        """
**Objective.** Predict `total_amount` (fare + tolls + congestion surcharge +
tip) for a NYC taxi-style trip from pickup/dropoff zone, trip distance, and
time context, while giving a data-science/code auditor full visibility into
how the numbers behind that prediction were produced.

**Success criteria.**
- A regression model with honest, held-out RMSE/MAE/R² (no leakage).
- A reproducible pipeline (fixed seeds throughout).
- An automated audit trail an auditor can read without re-deriving the code.

**Scope note.** This is a focused local build using a
seeded synthetic dataset (see Data Understanding for why), not the real
NYC TLC trip record archive.
"""
    )

with tabs[1]:
    st.header("Data Understanding")
    raw_df = out["raw_df"]
    st.write(f"Raw dataset: **{len(raw_df):,} rows**, {raw_df.shape[1]} columns.")
    st.dataframe(raw_df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Numeric summary")
        st.dataframe(
            raw_df[["trip_distance_miles", "fare_amount", "total_amount", "passenger_count"]].describe().round(2),
            use_container_width=True,
        )
    with col2:
        st.subheader("Trips per zone")
        st.bar_chart(raw_df["pu_zone"].value_counts())

    st.subheader("Known data-quality issues (see Audit Report tab for the full check)")
    quality = out["raw_quality"]
    for check in quality["checks"]:
        icon = "✅" if check["status"] == "pass" else "⚠️"
        st.write(f"{icon} **{check['check']}** — {check['detail']}")

with tabs[2]:
    st.header("Data Preparation")
    st.markdown(
        """
Cleaning rules applied (see `src/features.py::clean_raw`):
1. Drop exact duplicate rows.
2. Drop rows with pickup coordinates outside the NYC bounding box.
3. Drop non-positive `fare_amount` / `trip_distance_miles` rows.
4. Impute missing `passenger_count` with the median, missing `payment_type` with `"unknown"`.

Feature engineering applied after cleaning (see `build_features`): pickup
hour/day-of-week, weekend flag, rush-hour flag, airport-trip flag. None of
these derive from the target column.
"""
    )
    audit_clean = out["audit_report"]["cleaning_transparency"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw rows", f"{audit_clean['raw_rows']:,}")
    c2.metric("Clean rows", f"{audit_clean['clean_rows']:,}")
    c3.metric("Dropped", f"{audit_clean['rows_dropped']} ({audit_clean['pct_dropped']}%)")
    st.subheader("Sample of engineered features")
    st.dataframe(out["feat_df"].head(15), use_container_width=True)

with tabs[3]:
    st.header("Modeling & Evaluation")
    st.markdown(
        "Four regressors trained on an identical 80/20 split (fixed seed) "
        "and compared by held-out RMSE — a small, honest stand-in for a "
        "full hyperparameter-search 'AutoResearch' tournament."
    )
    metrics_df = pd.DataFrame(
        {name: r.metrics for name, r in out["results"].items()}
    ).T.round(3)
    metrics_df = metrics_df.sort_values("rmse")
    st.dataframe(metrics_df, use_container_width=True)
    st.success(f"Best model by held-out RMSE: **{out['best_name']}**")
    st.caption(
        "MAE/RMSE are in dollars (target = total_amount). Metrics are computed "
        "on the test split only, which was never used for fitting."
    )

with tabs[4]:
    st.header("Explainability")
    st.markdown(
        "Model-agnostic **permutation importance** on the held-out test set "
        f"for the selected model (`{out['best_name']}`) — how much test R² "
        "drops when a feature's values are shuffled. A lightweight, "
        "dependency-free substitute for a full TreeSHAP workbench."
    )
    st.dataframe(out["importance_df"], use_container_width=True)
    st.bar_chart(out["importance_df"].set_index("feature")["importance_mean"])

with tabs[5]:
    st.header("Spatial Clustering (Data Understanding — Unsupervised)")
    st.markdown("K-Means (k=5) on pickup latitude/longitude, profiled by fare and distance.")
    profile = out["cluster_profile"]
    st.dataframe(profile, use_container_width=True)

    clustered = out["clustered_df"]
    palette = [
        [230, 57, 70], [69, 123, 157], [42, 157, 143], [244, 162, 97], [155, 93, 229],
    ]
    plot_df = clustered.sample(min(1500, len(clustered)), random_state=42).copy()
    plot_df["color"] = plot_df["cluster"].apply(lambda c: palette[int(c) % len(palette)])
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[pickup_longitude, pickup_latitude]",
        get_fill_color="color",
        get_radius=60,
        pickable=True,
    )
    view_state = pdk.ViewState(latitude=40.73, longitude=-73.93, zoom=9.3)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

with tabs[6]:
    st.header("Audit Report")
    st.markdown(
        "Automated checks a data-science / code auditor should run before "
        "trusting this pipeline. See `src/audit.py` for the check logic."
    )
    report = out["audit_report"]

    st.subheader("Leakage check")
    lk = report["leakage_check"]
    st.write(f"Target `{lk['target_column']}` present in feature set: **{lk['target_present_in_features']}** → **{lk['status']}**")

    st.subheader("Reproducibility check")
    rc = report["reproducibility"]
    st.write(f"Fixed seed used throughout: **{rc['fixed_seed']}** → **{rc['status']}**")
    st.caption(rc["note"])

    st.subheader("Train/test split integrity")
    si = report["split_integrity"]
    st.write(
        f"Train rows: {si['train_rows']} · Test rows: {si['test_rows']} · "
        f"ID overlap: **{si['id_overlap_count']}** → **{si['status']}**"
    )
    st.write(f"Max payment-type share drift between train/test: {si['max_category_share_drift']}")

    st.subheader("Raw data quality")
    for check in report["raw_data_quality"]["checks"]:
        icon = "✅" if check["status"] == "pass" else "⚠️"
        st.write(f"{icon} **{check['check']}** — {check['detail']}")

    st.subheader("Cleaning transparency")
    ct = report["cleaning_transparency"]
    st.write(
        f"{ct['raw_rows']} raw rows → {ct['clean_rows']} clean rows "
        f"({ct['rows_dropped']} dropped, {ct['pct_dropped']}%)."
    )

    st.download_button(
        "Download full audit_report.json",
        data=json.dumps(report, indent=2, default=str),
        file_name="audit_report.json",
        mime="application/json",
    )

with tabs[7]:
    st.header("Fare Estimator (Deployment)")
    st.markdown("Live inference using the deployed best model from the Modeling tab.")
    best_pipeline = out["results"][out["best_name"]].pipeline

    zones = sorted(out["feat_df"]["pu_zone"].unique())
    col1, col2 = st.columns(2)
    with col1:
        pu_zone = st.selectbox("Pickup zone", zones, index=zones.index("Manhattan") if "Manhattan" in zones else 0)
        distance = st.slider("Trip distance (miles)", 0.2, 20.0, 3.0, 0.1)
        passengers = st.slider("Passenger count", 1, 4, 1)
    with col2:
        do_zone = st.selectbox("Dropoff zone", zones, index=zones.index("Brooklyn") if "Brooklyn" in zones else 1)
        hour = st.slider("Pickup hour", 0, 23, 18)
        payment = st.selectbox("Payment type", sorted(out["feat_df"]["payment_type"].dropna().unique()))

    is_weekend = 0
    is_rush = int(hour in [7, 8, 9, 16, 17, 18, 19])
    is_airport = int(pu_zone in ["JFK Airport", "LaGuardia Airport"] or do_zone in ["JFK Airport", "LaGuardia Airport"])
    rate_code = "airport_flat" if is_airport else "standard"
    vendor_id = "V1"

    input_row = pd.DataFrame(
        [
            {
                "trip_distance_miles": distance,
                "passenger_count": passengers,
                "pickup_hour": hour,
                "pickup_dayofweek": 2,
                "is_weekend": is_weekend,
                "is_rush_hour": is_rush,
                "is_airport_trip": is_airport,
                "pu_zone": pu_zone,
                "do_zone": do_zone,
                "payment_type": payment,
                "rate_code": rate_code,
                "vendor_id": vendor_id,
            }
        ]
    )
    if st.button("Estimate fare"):
        pred = best_pipeline.predict(input_row)[0]
        st.metric("Estimated total fare", f"${pred:,.2f}")
        st.caption(
            f"Prediction from `{out['best_name']}`, held-out test MAE ≈ "
            f"${out['results'][out['best_name']].metrics['mae']:.2f} — treat this "
            "estimate with that margin of error in mind."
        )
