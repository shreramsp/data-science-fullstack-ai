"""Streamlit front end for NYC Taxi Trip Duration & Fare Prediction.

Run with:  streamlit run app.py

Lets a user pick a pickup and dropoff location (preset NYC landmarks or
manual coordinates), see both plotted on an interactive map, and get a
trip duration + fare estimate from the trained RandomForest models.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
import pydeck as pdk
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import FEATURE_COLUMNS, build_features  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

LANDMARKS = {
    "Times Square": (40.7580, -73.9855),
    "Central Park (South)": (40.7676, -73.9787),
    "Empire State Building": (40.7484, -73.9857),
    "Wall Street / FiDi": (40.7075, -74.0113),
    "Brooklyn Bridge": (40.7061, -73.9969),
    "Williamsburg, Brooklyn": (40.7081, -73.9571),
    "Long Island City, Queens": (40.7447, -73.9485),
    "Yankee Stadium": (40.8296, -73.9262),
    "JFK Airport": (40.6413, -73.7781),
    "LaGuardia Airport": (40.7769, -73.8740),
    "Custom (enter coordinates)": None,
}

st.set_page_config(page_title="NYC Taxi Trip Estimator", layout="wide")


@st.cache_resource
def load_models():
    duration_model = joblib.load(os.path.join(MODELS_DIR, "duration_model.joblib"))
    fare_model = joblib.load(os.path.join(MODELS_DIR, "fare_model.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return duration_model, fare_model, metrics


def location_picker(label: str, default_key: str, key_prefix: str):
    choice = st.selectbox(label, list(LANDMARKS.keys()),
                           index=list(LANDMARKS.keys()).index(default_key),
                           key=f"{key_prefix}_select")
    if LANDMARKS[choice] is None:
        col1, col2 = st.columns(2)
        lat = col1.number_input("Latitude", value=40.7549, format="%.5f",
                                 key=f"{key_prefix}_lat")
        lon = col2.number_input("Longitude", value=-73.9840, format="%.5f",
                                 key=f"{key_prefix}_lon")
    else:
        lat, lon = LANDMARKS[choice]
        st.caption(f"{lat:.5f}, {lon:.5f}")
    return lat, lon


def main():
    st.title("NYC Taxi Trip Duration & Fare Prediction")
    st.caption(
        "CRISP-DM end-to-end demo trained on a reproducible synthetic NYC-taxi-style "
        "dataset (see README for data + methodology notes)."
    )

    if not os.path.exists(os.path.join(MODELS_DIR, "metrics.json")):
        st.error("No trained models found. Run `python src/train.py` first.")
        st.stop()

    duration_model, fare_model, metrics = load_models()

    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Trip details")
        pickup_lat, pickup_lon = location_picker("Pickup location", "Times Square", "pickup")
        dropoff_lat, dropoff_lon = location_picker(
            "Dropoff location", "JFK Airport", "dropoff"
        )

        c1, c2 = st.columns(2)
        pickup_date = c1.date_input("Pickup date", value=datetime(2016, 3, 15))
        pickup_time = c2.time_input("Pickup time", value=datetime(2016, 3, 15, 18, 30).time())
        passenger_count = st.slider("Passenger count", 1, 6, 1)
        vendor_id = st.radio("Vendor", [1, 2], horizontal=True)

        estimate = st.button("Estimate trip", type="primary", width="stretch")

    with right:
        st.subheader("Route map")
        route_df = pd.DataFrame(
            [
                {"name": "Pickup", "lat": pickup_lat, "lon": pickup_lon, "color": [16, 185, 129]},
                {"name": "Dropoff", "lat": dropoff_lat, "lon": dropoff_lon, "color": [239, 68, 68]},
            ]
        )
        line_df = pd.DataFrame(
            [{"from": [pickup_lon, pickup_lat], "to": [dropoff_lon, dropoff_lat]}]
        )
        mid_lat = (pickup_lat + dropoff_lat) / 2
        mid_lon = (pickup_lon + dropoff_lon) / 2

        deck = pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=mid_lat, longitude=mid_lon, zoom=10.5, pitch=0
            ),
            layers=[
                pdk.Layer(
                    "LineLayer",
                    data=line_df,
                    get_source_position="from",
                    get_target_position="to",
                    get_color=[100, 116, 139],
                    get_width=3,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=route_df,
                    get_position=["lon", "lat"],
                    get_fill_color="color",
                    get_radius=180,
                    pickable=True,
                ),
            ],
            tooltip={"text": "{name}"},
        )
        st.pydeck_chart(deck, width="stretch")

    if estimate:
        pickup_dt = datetime.combine(pickup_date, pickup_time)
        row = pd.DataFrame(
            [
                {
                    "pickup_datetime": pickup_dt,
                    "passenger_count": passenger_count,
                    "pickup_longitude": pickup_lon,
                    "pickup_latitude": pickup_lat,
                    "dropoff_longitude": dropoff_lon,
                    "dropoff_latitude": dropoff_lat,
                    "vendor_id": vendor_id,
                }
            ]
        )
        feats = build_features(row)[FEATURE_COLUMNS]
        pred_duration_s = float(duration_model.predict(feats)[0])
        pred_fare = float(fare_model.predict(feats)[0])
        distance = float(feats["trip_distance_miles"].iloc[0])

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated duration", f"{pred_duration_s / 60:.1f} min")
        m2.metric("Estimated fare", f"${pred_fare:.2f}")
        m3.metric("Straight-line distance", f"{distance:.2f} mi")

    with st.expander("Model performance (honest, held-out test metrics)"):
        d = metrics["duration_model"]["test_metrics"]
        f = metrics["fare_model"]["test_metrics"]
        st.markdown(
            f"**Duration model** — MAE: {d['mae']:.1f}s · RMSE: {d['rmse']:.1f}s · "
            f"R²: {d['r2']:.3f}  \n"
            f"**Fare model** — MAE: ${f['mae']:.2f} · RMSE: ${f['rmse']:.2f} · "
            f"R²: {f['r2']:.3f}  \n\n"
            f"Trained on {metrics['n_train']:,} rows, evaluated on "
            f"{metrics['n_test']:,} held-out rows "
            f"({metrics['rows_removed_in_cleaning']} of {metrics['n_rows_raw']:,} "
            "raw rows dropped during cleaning)."
        )


if __name__ == "__main__":
    main()
