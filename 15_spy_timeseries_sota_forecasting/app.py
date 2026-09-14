"""Streamlit dashboard for the SPY-like probabilistic forecasting &
trading backtest platform. Six tabs: Overview, Data & Features, Model
Tournament, Probabilistic Forecast, Explainability, Trading Backtest.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_PATH = ROOT / "data" / "spy_prices.csv"

st.set_page_config(page_title="SPY SOTA Forecasting Platform", layout="wide")


@st.cache_data
def load_artifacts():
    if not (ARTIFACTS / "metrics.json").exists():
        subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], check=True)
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    comparison = pd.read_csv(ARTIFACTS / "model_comparison.csv")
    forecasts = pd.read_csv(ARTIFACTS / "forecasts.csv", parse_dates=["date"])
    importance = pd.read_csv(ARTIFACTS / "feature_importance.csv")
    backtest = pd.read_csv(ARTIFACTS / "backtest_results.csv", parse_dates=["date"])
    prices = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return metrics, comparison, forecasts, importance, backtest, prices


metrics, comparison, forecasts, importance, backtest, prices = load_artifacts()

st.title("SOTA SPY Time Series Forecasting & Trading Platform")
st.caption(
    "Independent build on a **synthetic, SPY-like** daily price series (regime-switching "
    "volatility, fat-tailed shocks). Not real market data and not a claim about real SPY "
    "performance — see the Limitations note at the bottom of this page."
)

tabs = st.tabs([
    "Overview", "Data & Features", "Model Tournament",
    "Probabilistic Forecast", "Explainability", "Trading Backtest",
])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    st.subheader("Price history")
    chart = alt.Chart(prices).mark_line().encode(
        x="date:T", y=alt.Y("close:Q", title="Close (synthetic)"),
        color=alt.value("#1f77b4"),
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (days)", f"{metrics['n_rows_raw']:,}")
    c2.metric("Train days", f"{metrics['n_train']:,}")
    c3.metric("Test days (out-of-sample)", f"{metrics['n_test']:,}")
    c4.metric("Split cutoff", metrics["split_cutoff_date"])
    st.markdown(
        f"- Chronological split, **embargo of {metrics['embargo_days']} days** dropped "
        "around the boundary so no rolling/forward-looking window straddles train and test.\n"
        "- RobustScaler fit on the training features only, then applied to test.\n"
        "- Two forecast horizons: next trading day (t+1) and next trading week (t+5), each as "
        "a P10 / P50 / P90 log-return envelope."
    )

# ---------------------------------------------------------- Data & Features
with tabs[1]:
    st.subheader("Raw price sample")
    st.dataframe(prices.tail(10), use_container_width=True)

    st.subheader("Engineered features (test set sample)")
    feat_cols = [c for c in forecasts.columns if c not in
                 ("date", "horizon", "actual_return", "pred_naive", "pred_ridge", "pred_gbm",
                  "pred_p10", "pred_p50", "pred_p90")]
    st.dataframe(forecasts[forecasts["horizon"] == 1].head(10), use_container_width=True)
    st.markdown(
        "**Feature families** (all computed with `shift()` before any `rolling()`/`ewm()` "
        "window, so day *t*'s features never see day *t*'s own bar):\n"
        "- Lagged log returns (1, 2, 3, 5, 10 days)\n"
        "- Rolling mean/std of returns (5, 20 days)\n"
        "- Momentum (10-day), RSI(14), MACD(12,26,9)\n"
        "- Volume z-score (20-day), prior-day high-low range"
    )

# --------------------------------------------------------- Model Tournament
with tabs[2]:
    st.subheader("Point-forecast tournament (MAE / RMSE / directional hit rate)")
    horizon_pick = st.radio("Horizon", [1, 5], horizontal=True, format_func=lambda h: f"t+{h}")
    sub = comparison[comparison["horizon"] == horizon_pick]
    st.dataframe(sub.style.format({"mae": "{:.5f}", "rmse": "{:.5f}", "directional_hit_rate": "{:.1%}"}),
                 use_container_width=True)

    bar = alt.Chart(sub).mark_bar().encode(
        x=alt.X("model:N", sort="-y"), y="mae:Q", color="model:N",
    ).properties(height=280)
    st.altair_chart(bar, use_container_width=True)
    winner = sub.sort_values("mae").iloc[0]["model"]
    st.info(f"Lowest MAE at horizon t+{horizon_pick}: **{winner}**. Candidates: naive "
            "(zero-return random walk), Ridge (linear), HistGradientBoostingRegressor.")

# ---------------------------------------------------- Probabilistic Forecast
with tabs[3]:
    st.subheader("P10 / P50 / P90 return envelope vs. actual (out-of-sample)")
    horizon_pick2 = st.radio("Horizon", [1, 5], horizontal=True, format_func=lambda h: f"t+{h}",
                              key="horizon2")
    sub2 = forecasts[forecasts["horizon"] == horizon_pick2].sort_values("date")

    band = alt.Chart(sub2).mark_area(opacity=0.25, color="#1f77b4").encode(
        x="date:T", y="pred_p10:Q", y2="pred_p90:Q",
    )
    median_line = alt.Chart(sub2).mark_line(color="#1f77b4").encode(x="date:T", y="pred_p50:Q")
    actual_line = alt.Chart(sub2).mark_line(color="black", strokeDash=[4, 2]).encode(
        x="date:T", y="actual_return:Q",
    )
    st.altair_chart((band + median_line + actual_line).properties(height=340), use_container_width=True)
    st.caption("Shaded band = P10-P90 forecast; solid line = P50 median forecast; "
               "dashed black = actual realized log return.")

    qm = metrics["quantile_metrics"][f"h{horizon_pick2}"]
    c1, c2 = st.columns(2)
    c1.metric("P10-P90 empirical coverage", f"{qm['coverage_p10_p90']:.1%}",
              help=f"Target (nominal) coverage: {qm['target_coverage']:.0%}")
    c2.metric("Pinball loss (P50)", f"{qm['pinball_loss']['p50']:.5f}")

# -------------------------------------------------------------- Explainability
with tabs[4]:
    st.subheader("Permutation feature importance — horizon t+1 GBM (MAE increase)")
    chart_imp = alt.Chart(importance).mark_bar().encode(
        x="importance:Q", y=alt.Y("feature:N", sort="-x"),
    ).properties(height=400)
    st.altair_chart(chart_imp, use_container_width=True)
    st.caption(
        "Permutation importance: each feature's column is shuffled on the held-out test set "
        "and the resulting MAE increase is reported (mean over 10 repeats). Larger = the model "
        "relies on that feature more."
    )

# --------------------------------------------------------------- Trading Backtest
with tabs[5]:
    st.subheader("Long/flat backtest driven by the t+1 P50 forecast")
    threshold = st.slider("Long-entry threshold on predicted next-day log return", -0.005, 0.005, 0.0, 0.0005)

    if threshold == 0.0:
        bt_summary = metrics["backtest"]
        bt = backtest
    else:
        sys.path.insert(0, str(ROOT / "src"))
        from backtest import run_backtest, summarize
        h1 = forecasts[forecasts["horizon"] == 1].sort_values("date")
        bt = run_backtest(h1["actual_return"].to_numpy(), h1["pred_p50"].to_numpy(), threshold=threshold)
        bt["date"] = h1["date"].to_numpy()
        bt_summary = summarize(bt)

    strat_curve = (1 + bt["strategy_return"]).cumprod()
    bh_curve = (1 + bt["buy_hold_return"]).cumprod()
    curve_df = pd.DataFrame({"date": bt["date"], "Strategy": strat_curve, "Buy & Hold": bh_curve})
    curve_long = curve_df.melt("date", var_name="series", value_name="growth_of_1")
    line = alt.Chart(curve_long).mark_line().encode(x="date:T", y="growth_of_1:Q", color="series:N")
    st.altair_chart(line.properties(height=320), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe (strategy)", f"{bt_summary['strategy']['sharpe_ratio']:.2f}",
              delta=f"{bt_summary['strategy']['sharpe_ratio'] - bt_summary['buy_and_hold']['sharpe_ratio']:.2f} vs B&H")
    c2.metric("Max drawdown (strategy)", f"{bt_summary['strategy']['max_drawdown']:.1%}")
    c3.metric("Directional hit rate", f"{bt_summary['directional_hit_rate']:.1%}")
    c4.metric("Annualized return (strategy)", f"{bt_summary['strategy']['annualized_return']:.1%}")

    st.markdown(
        "Rule: **long when the predicted next-day log return exceeds the threshold, otherwise "
        "flat** (no shorting, no leverage). A 2 bps slippage cost is charged only on days the "
        "position changes."
    )

st.divider()
st.warning(
    "**Limitations.** All prices are a seeded synthetic series with the qualitative "
    "statistical properties of an equity index (fat tails, volatility clustering) — not real "
    "SPY data. Directional hit rates near 50-52% reflect how hard next-day return direction is "
    "to predict even on data generated with genuine (if weak) autocorrelation; this is expected, "
    "not a bug. The t+5 quantile band is under-covered relative to its 80% target, and the trading "
    "strategy underperforms buy-and-hold on a risk-adjusted (Sharpe) basis in this particular test "
    "window, though with materially lower volatility and drawdown. No hyperparameter search was run."
)
