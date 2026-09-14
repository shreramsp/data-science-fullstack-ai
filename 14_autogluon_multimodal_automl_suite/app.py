"""Streamlit dashboard for the AutoGluon Multimodal AutoML Suite.

Every number here is read from ``artifacts/`` (written once by
``src/train.py``) or, for live scoring, from a fitted predictor under
``models/``. The dashboard computes no metric of its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"

st.set_page_config(page_title="AutoGluon Multimodal AutoML Suite", page_icon="🧬", layout="wide")


@st.cache_data
def load_summary():
    path = ARTIFACTS / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_tabular_artifacts(task_key: str):
    d = ARTIFACTS / task_key
    return {
        "metrics": json.loads((d / "metrics.json").read_text()),
        "leaderboard": pd.read_csv(d / "leaderboard.csv"),
        "importance": pd.read_csv(d / "feature_importance.csv", index_col=0) if (d / "feature_importance.csv").exists() else None,
        "sample": pd.read_csv(d / "holdout_sample.csv"),
    }


@st.cache_data
def load_timeseries_artifacts():
    d = ARTIFACTS / "retail_demand"
    return {
        "metrics": json.loads((d / "metrics.json").read_text()),
        "leaderboard": pd.read_csv(d / "leaderboard.csv"),
        "history": pd.read_csv(d / "history.csv", parse_dates=["timestamp"]),
        "actuals": pd.read_csv(d / "actuals.csv", parse_dates=["timestamp"]),
        "forecast": pd.read_csv(d / "forecast.csv", parse_dates=["timestamp"]),
    }


@st.cache_resource
def load_tabular_predictor(task_key: str):
    from autogluon.tabular import TabularPredictor
    path = MODELS / task_key
    if not path.exists():
        return None
    return TabularPredictor.load(str(path))


@st.cache_resource
def load_ts_predictor():
    from autogluon.timeseries import TimeSeriesPredictor
    path = MODELS / "retail_demand"
    if not path.exists():
        return None
    return TimeSeriesPredictor.load(str(path))


summary = load_summary()

st.title("🧬 AutoGluon Multimodal AutoML Suite")
st.caption(
    "Three distinct AutoGluon capabilities on one small, reproducible dataset each: "
    "tabular + text multimodal classification, multi-quantile probabilistic regression, "
    "and multi-series time series forecasting."
)

if summary is None:
    st.error(
        "No artifacts found. Run `python src/train.py` first — it fits every task and "
        "writes the results this dashboard reads under `artifacts/`."
    )
    st.stop()

overview, multimodal_tab, quantile_tab, forecast_tab, scoring = st.tabs(
    ["Overview", "Multimodal Classification", "Quantile Regression", "Time Series Forecasting", "Live Scoring"]
)

# ---------------------------------------------------------------- Overview
with overview:
    st.subheader("Capabilities at a glance")
    cols = st.columns(3)
    for col, key in zip(cols, summary.keys()):
        m = summary[key]
        primary_metric = m["eval_metric"]
        test_val = m["test_metrics"].get(primary_metric)
        with col:
            st.markdown(f"**{m['display_name']}**")
            st.caption(m.get("problem_type", "forecasting"))
            st.metric(f"test {primary_metric}", f"{test_val:.4f}" if test_val is not None else "n/a")
            st.metric("models fit", m["n_models_fit"])
            st.metric("fit time", f"{m['fit_seconds']}s")
            st.caption(m["source"])

    st.divider()
    st.markdown(
        """
**Why these three tasks?** `autogluon.tabular` and `autogluon.timeseries` are separate
AutoGluon packages with different strengths — this suite exercises both plus a capability
inside `TabularPredictor` (automatic text feature handling) that a plain numeric/categorical
demo would not show:

1. **Multimodal Classification** — `TabularPredictor` ingests a free-text review column
   alongside numeric and categorical columns in the same fit call; AutoGluon's built-in
   text feature generator handles the NLP featurization automatically.
2. **Quantile Regression** — the same `TabularPredictor` API, switched to
   `problem_type="quantile"`, produces P10/P50/P90 prediction bands instead of one point
   estimate — a different *task formulation* of the identical AutoML engine.
3. **Time Series Forecasting** — `TimeSeriesPredictor` (the `autogluon.timeseries`
   package) forecasts six related series at once with a shared model search.
        """
    )

# ---------------------------------------------------------------- Multimodal Classification
with multimodal_tab:
    key = "multimodal_reviews"
    art = load_tabular_artifacts(key)
    m = art["metrics"]
    st.subheader(m["display_name"])
    st.write(f"**Source:** {m['source']}")
    st.write(f"**Rows:** {m['n_rows']} total → {m['n_train']} train, {m['n_test']} held-out test")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Model leaderboard:**")
        st.dataframe(art["leaderboard"], use_container_width=True, hide_index=True)
    with c2:
        st.write("**Held-out test metrics:**")
        st.json(m["test_metrics"])
        st.caption(f"Best model: `{m['best_model']}`")

    if art["importance"] is not None:
        st.write("**Permutation feature importance** (includes the auto-generated text features):")
        imp = art["importance"].reset_index().rename(columns={"index": "feature"})
        fig = px.bar(imp.sort_values("importance", ascending=True).tail(15), x="importance", y="feature", orientation="h")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.write("**Held-out sample with predictions:**")
    st.dataframe(art["sample"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Quantile Regression
with quantile_tab:
    key = "diabetes_quantile"
    art = load_tabular_artifacts(key)
    m = art["metrics"]
    st.subheader(m["display_name"])
    st.write(f"**Source:** {m['source']}")
    st.write(f"**Rows:** {m['n_rows']} total → {m['n_train']} train, {m['n_test']} held-out test")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Model leaderboard:**")
        st.dataframe(art["leaderboard"], use_container_width=True, hide_index=True)
    with c2:
        st.write("**Held-out test metrics:**")
        st.json(m["test_metrics"])
        st.metric("P10-P90 holdout coverage", f"{m['p10_p90_holdout_coverage']:.0%}")
        st.caption(f"Best model: `{m['best_model']}`")

    st.write("**Held-out predictions with P10/P50/P90 bands (sorted by actual value):**")
    sample = art["sample"].sort_values("disease_progression").reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample.index, y=sample["pred_p90"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=sample.index, y=sample["pred_p10"], mode="lines", fill="tonexty", line=dict(width=0), name="P10-P90 band"))
    fig.add_trace(go.Scatter(x=sample.index, y=sample["pred_p50"], mode="lines+markers", name="P50 (median)"))
    fig.add_trace(go.Scatter(x=sample.index, y=sample["disease_progression"], mode="markers", name="actual", marker=dict(size=8, symbol="x")))
    fig.update_layout(height=420, xaxis_title="held-out row (sorted)", yaxis_title="disease progression")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(art["sample"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Time Series Forecasting
with forecast_tab:
    ts = load_timeseries_artifacts()
    m = ts["metrics"]
    st.subheader(m["display_name"])
    st.write(f"**Source:** {m['source']}")
    st.write(f"**Series:** {m['n_series']} items, {m['n_rows']} total rows, forecasting the last {m['prediction_length']} days")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Model leaderboard** (scored on held-out window):")
        st.dataframe(ts["leaderboard"], use_container_width=True, hide_index=True)
    with c2:
        st.write("**Held-out test metrics:**")
        st.json(m["test_metrics"])
        st.caption(f"Best model: `{m['best_model']}`")

    item_ids = sorted(ts["history"]["item_id"].unique())
    item = st.selectbox("Item", item_ids)
    hist = ts["history"][ts["history"]["item_id"] == item]
    actual = ts["actuals"][ts["actuals"]["item_id"] == item]
    fc = ts["forecast"][ts["forecast"]["item_id"] == item]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["timestamp"], y=hist["demand"], mode="lines", name="history"))
    fig.add_trace(go.Scatter(x=actual["timestamp"], y=actual["demand"], mode="lines+markers", name="actual (held out)"))
    if "0.1" in fc.columns and "0.9" in fc.columns:
        fig.add_trace(go.Scatter(x=fc["timestamp"], y=fc["0.9"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=fc["timestamp"], y=fc["0.1"], mode="lines", fill="tonexty", line=dict(width=0), name="P10-P90 band"))
    if "mean" in fc.columns:
        fig.add_trace(go.Scatter(x=fc["timestamp"], y=fc["mean"], mode="lines+markers", name="forecast (mean)"))
    fig.update_layout(height=440, xaxis_title="date", yaxis_title="demand", title=f"{item}: history, forecast, and actual")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Live Scoring
with scoring:
    st.subheader("Score a held-out row with a deployed predictor")
    task_key = st.selectbox(
        "Task",
        ["multimodal_reviews", "diabetes_quantile"],
        format_func=lambda k: summary[k]["display_name"],
    )
    predictor = load_tabular_predictor(task_key)
    art = load_tabular_artifacts(task_key)
    sample = art["sample"]

    if predictor is None:
        st.warning(
            f"No fitted predictor found at `models/{task_key}/`. Model directories are "
            "gitignored — run `python src/train.py` once locally to regenerate them, then "
            "reload this page."
        )
    else:
        label_col = predictor.label
        drop_cols = [c for c in sample.columns if c.startswith("predicted_") or c.startswith("pred_p")]
        row_idx = st.selectbox("Held-out row", sample.index, format_func=lambda i: f"row {i}")
        row = sample.loc[[row_idx]].drop(columns=drop_cols, errors="ignore")
        true_value = row[label_col].iloc[0]
        features = row.drop(columns=[label_col])

        edited = st.data_editor(features, use_container_width=True, num_rows="fixed")

        if st.button("Predict", type="primary"):
            pred = predictor.predict(edited)
            st.metric("True value", str(true_value))
            if predictor.problem_type == "quantile":
                st.write("Predicted quantiles:")
                st.dataframe(pred, use_container_width=True)
            else:
                st.metric("Prediction", str(pred.iloc[0]))
                proba = predictor.predict_proba(edited).iloc[0]
                st.write("Class probabilities:")
                st.bar_chart(proba)

st.divider()
st.caption(
    "Reference: Erickson, N. et al. (2020), \"AutoGluon-Tabular: Robust and Accurate AutoML "
    "for Structured Data\", arXiv:2003.06505; Shchur, O. et al. (2023), \"AutoGluon-TimeSeries\", "
    "AutoML Conference. The multimodal-review and retail-demand datasets are deterministic "
    "synthetic generators (seed=42); the quantile-regression dataset is scikit-learn's "
    "built-in `load_diabetes` — no external downloads required."
)
