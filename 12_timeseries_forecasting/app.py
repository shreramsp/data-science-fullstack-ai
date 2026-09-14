"""Time Series Forecasting Engine — admin dashboard (CRISP-DM).

Run:  streamlit run app.py      (after `python src/train.py`)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
ARTIFACTS = ROOT / "artifacts"

from features import FEATURE_COLUMNS, TEST_DAYS  # noqa: E402
from forecast import forecast_series  # noqa: E402

st.set_page_config(page_title="Time Series Forecasting Engine", page_icon="📈", layout="wide")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
MODEL_COLORS = {
    "naive": "#8a8a86",
    "seasonal_naive": "#eda100",
    "holt_winters": "#2a78d6",
    "gbm": "#1baf7a",
}
MODEL_LABELS = {
    "naive": "Naive (last value)",
    "seasonal_naive": "Seasonal naive (t-7)",
    "holt_winters": "Holt-Winters",
    "gbm": "Gradient-boosted trees",
}

CRISP_DM = [
    ("1. Business understanding",
     "Forecast daily unit sales for each store x category series far enough ahead "
     "(8 weeks) to drive a replenishment / promo-planning cycle, and be honest about "
     "which class of model actually earns its complexity over a naive baseline.",
     "README.md, CRISP_DM.md"),
    ("2. Data understanding",
     "Daily sales for 3 stores x 3 categories (9 series), ~3 years, with trend, "
     "weekly seasonality, an annual holiday bump, and a promo flag.",
     "data/generate_data.py, the Explore tab"),
    ("3. Data preparation",
     "Calendar features (day of week, month, holiday season, promo) plus lag "
     "(1/7/14/28-day) and rolling mean/std (7/28-day) features per series, computed "
     "only from each series' own past; a chronological 56-day-per-series holdout.",
     "src/features.py"),
    ("4. Modeling",
     "Four candidates: naive, seasonal naive, per-series Holt-Winters exponential "
     "smoothing, and one global gradient-boosted-trees model over lag/calendar "
     "features across all series.",
     "src/train.py"),
    ("5. Evaluation",
     "MAE, RMSE and SMAPE on the holdout, pooled across series and broken out per "
     "series; the winner is whichever model has the lowest pooled MAE.",
     "src/train.py, the Model Comparison tab"),
    ("6. Deployment",
     "This dashboard: EDA, model comparison, a forecast explorer that runs the "
     "winning model forward past the end of the historical data, and a model card.",
     "app.py, src/forecast.py"),
]


@st.cache_data(show_spinner=False)
def load_artifacts():
    needed = ["metrics.json", "forecasts.csv", "model_comparison.csv", "per_series_metrics.csv"]
    if not all((ARTIFACTS / f).exists() for f in needed):
        return None
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    forecasts = pd.read_csv(ARTIFACTS / "forecasts.csv", parse_dates=["date"])
    comparison = pd.read_csv(ARTIFACTS / "model_comparison.csv")
    per_series = pd.read_csv(ARTIFACTS / "per_series_metrics.csv")
    return {"metrics": metrics, "forecasts": forecasts, "comparison": comparison, "per_series": per_series}


@st.cache_data(show_spinner=False)
def load_raw():
    from features import load_raw as _load
    df = _load()
    df["series_id"] = df["store_id"] + "_" + df["category"]
    return df


@st.cache_resource(show_spinner=False)
def load_bundle():
    path = ARTIFACTS / "best_model.joblib"
    return joblib.load(path) if path.exists() else None


data = load_artifacts()
raw = load_raw()
bundle = load_bundle()

st.title("📈 Time Series Forecasting Engine")
st.caption("Synthetic multi-series retail sales • CRISP-DM • naive / seasonal-naive / "
           "Holt-Winters / gradient-boosted-trees comparison")

if data is None:
    st.warning("No artifacts found. Run `python src/train.py` first, then reload this page.")
    st.stop()

metrics = data["metrics"]
series_ids = sorted(raw["series_id"].unique())

tab_overview, tab_explore, tab_compare, tab_forecast, tab_crisp, tab_card = st.tabs(
    ["Overview", "Explore", "Model Comparison", "Forecast Explorer", "CRISP-DM", "Model Card"]
)

# ----------------------------------------------------------------------------- Overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Series tracked", metrics["n_series"])
    c2.metric("History (days)", int(metrics["n_rows_raw"] / metrics["n_series"]))
    c3.metric("Best model", MODEL_LABELS[metrics["best_model"]])
    c4.metric("Best MAE (units/day)", f"{metrics['overall'][metrics['best_model']]['mae']:.1f}")

    st.subheader("All series, full history")
    chart = alt.Chart(raw).mark_line(strokeWidth=1.1).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("sales:Q", title="units/day"),
        color=alt.Color("series_id:N", title="series"),
        tooltip=["date:T", "series_id:N", "sales:Q"],
    ).properties(height=380).configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=INK)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("How each model did (pooled over all 9 series)")
    comp = data["comparison"].copy()
    comp["model_label"] = comp["model"].map(MODEL_LABELS)
    st.dataframe(comp[["model_label", "mae", "rmse", "smape"]]
                 .rename(columns={"model_label": "model", "mae": "MAE", "rmse": "RMSE", "smape": "SMAPE %"})
                 .round(2), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------- Explore
with tab_explore:
    sid = st.selectbox("Series", series_ids, key="explore_sid")
    g = raw[raw["series_id"] == sid].sort_values("date")
    st.write(f"**{len(g):,} days** — {g['date'].min().date()} → {g['date'].max().date()}, "
             f"promo active on {int(g['promo'].sum())} days")

    base = alt.Chart(g).encode(x=alt.X("date:T", title=None))
    line = base.mark_line(color=MODEL_COLORS["gbm"]).encode(y=alt.Y("sales:Q", title="units/day"))
    promo_marks = base.transform_filter(alt.datum.promo == 1).mark_tick(
        color=MODEL_COLORS["seasonal_naive"], thickness=2, size=8
    ).encode(y=alt.value(0))
    st.altair_chart((line + promo_marks).properties(height=320)
                     .configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=INK),
                     use_container_width=True)

    st.caption("Weekday effect (mean sales by day of week, this series)")
    dow = g.assign(dow=g["date"].dt.day_name()).groupby("dow", as_index=False)["sales"].mean()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_chart = alt.Chart(dow).mark_bar(color=MODEL_COLORS["holt_winters"]).encode(
        x=alt.X("dow:N", sort=order, title=None),
        y=alt.Y("sales:Q", title="mean units/day"),
    ).properties(height=220).configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=INK)
    st.altair_chart(dow_chart, use_container_width=True)

# ----------------------------------------------------------------------------- Model Comparison
with tab_compare:
    st.subheader(f"Holdout evaluation — last {TEST_DAYS} days per series")
    comp_long = data["comparison"].melt(id_vars="model", value_vars=["mae", "rmse", "smape"],
                                         var_name="metric", value_name="value")
    comp_long["model_label"] = comp_long["model"].map(MODEL_LABELS)
    bars = alt.Chart(comp_long).mark_bar().encode(
        x=alt.X("model_label:N", title=None, sort=None),
        y=alt.Y("value:Q"),
        color=alt.Color("model:N", scale=alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values())),
                         legend=None),
        column=alt.Column("metric:N", title=None),
        tooltip=["model_label:N", "metric:N", alt.Tooltip("value:Q", format=".2f")],
    ).properties(height=280, width=180).configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=INK)
    st.altair_chart(bars, use_container_width=False)

    st.subheader("Per-series breakdown")
    metric_choice = st.radio("Metric", ["mae", "rmse", "smape"], horizontal=True)
    per_series = data["per_series"].copy()
    per_series["model_label"] = per_series["model"].map(MODEL_LABELS)
    pivot = per_series.pivot(index="series_id", columns="model_label", values=metric_choice).round(2)
    st.dataframe(pivot, use_container_width=True)
    st.caption(
        "SMAPE (symmetric mean absolute percentage error) is reported instead of plain MAPE "
        "because several series have zero-sale days, where a plain percentage error is undefined."
    )

# ----------------------------------------------------------------------------- Forecast Explorer
with tab_forecast:
    col1, col2 = st.columns([2, 1])
    with col1:
        sid = st.selectbox("Series", series_ids, key="forecast_sid")
    with col2:
        horizon = st.slider("Days ahead", 7, 90, 28, step=7)

    hist = raw[raw["series_id"] == sid].sort_values("date").tail(120)
    fwd = forecast_series(raw, bundle, sid, horizon)

    hist_chart = alt.Chart(hist).mark_line(color=MUTED).encode(
        x=alt.X("date:T", title=None), y=alt.Y("sales:Q", title="units/day"),
    )
    fwd_chart = alt.Chart(fwd).mark_line(color=MODEL_COLORS["gbm"], strokeDash=[4, 3]).encode(
        x="date:T", y="pred:Q",
    )
    st.altair_chart((hist_chart + fwd_chart).properties(height=340)
                     .configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=INK),
                     use_container_width=True)
    st.caption(f"Solid line: last 120 observed days. Dashed line: {horizon}-day forward forecast "
               f"from the winning model ({MODEL_LABELS[metrics['best_model']]}).")

    st.subheader("Holdout: actual vs. every model, this series")
    fc = data["forecasts"]
    fc_sid = fc[fc["series_id"] == sid]
    plot_df = fc_sid.melt(
        id_vars=["date"], value_vars=["sales", "pred_naive", "pred_seasonal_naive", "pred_holt_winters", "pred_gbm"],
        var_name="series", value_name="value",
    )
    name_map = {"sales": "actual", "pred_naive": "naive", "pred_seasonal_naive": "seasonal_naive",
                "pred_holt_winters": "holt_winters", "pred_gbm": "gbm"}
    plot_df["series"] = plot_df["series"].map(name_map)
    colors = {"actual": INK, **MODEL_COLORS}
    holdout_chart = alt.Chart(plot_df).mark_line().encode(
        x=alt.X("date:T", title=None), y=alt.Y("value:Q", title="units/day"),
        color=alt.Color("series:N", scale=alt.Scale(domain=list(colors), range=list(colors.values()))),
        strokeWidth=alt.condition(alt.datum.series == "actual", alt.value(2.4), alt.value(1.4)),
    ).properties(height=320).configure_axis(gridColor=GRID, labelColor=MUTED, titleColor=INK)
    st.altair_chart(holdout_chart, use_container_width=True)

# ----------------------------------------------------------------------------- CRISP-DM
with tab_crisp:
    for phase, summary, where in CRISP_DM:
        with st.expander(phase, expanded=(phase.startswith("1"))):
            st.write(summary)
            st.caption(f"See: {where}")

# ----------------------------------------------------------------------------- Model Card
with tab_card:
    st.subheader("Selected model")
    st.write(f"**{MODEL_LABELS[metrics['best_model']]}**, chosen by lowest pooled MAE on the holdout.")
    st.json(metrics["overall"][metrics["best_model"]])

    st.subheader("Reproducibility")
    st.code(
        "python data/generate_data.py   # writes data/sales.csv (seed 20260913)\n"
        "python src/train.py            # writes artifacts/\n"
        "streamlit run app.py",
        language="bash",
    )
    st.write(f"Seed: `{metrics['seed']}` · Holdout: last `{metrics['test_days']}` days per series · "
             f"Cutoff date: `{metrics['cutoff_date']}` · Training runtime: `{metrics['runtime_seconds']}s`")

    if metrics["best_model"] == "gbm":
        st.subheader("Feature columns (global gradient-boosted-trees model)")
        st.code(", ".join(FEATURE_COLUMNS))

    st.subheader("Limitations")
    st.markdown(
        "- **Synthetic data.** Trend/seasonality/promo effects are generator parameters, not "
        "observed retail behaviour; no real stock-outs, competitor actions, or price changes.\n"
        "- **Evaluation uses true historical lags**, not recursively-generated ones — realistic "
        "for a system that refits/re-lags daily, optimistic for a model left to run forward "
        "untouched for the whole holdout (which is what the Forecast Explorer's GBM path does "
        "recursively, and where errors can compound).\n"
        "- **No hyperparameter search.** The gradient-boosted-trees model uses one fixed "
        "configuration; Holt-Winters uses fixed additive trend/seasonal terms.\n"
        "- **9 series, 3 years.** Too little history to fit anything more data-hungry than these "
        "four candidates, and too little to detect slow regime changes."
    )
