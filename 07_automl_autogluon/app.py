"""Streamlit admin dashboard for the AutoGluon multi-layer stacking platform.

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

st.set_page_config(page_title="AutoGluon Stacking Platform", page_icon="🧱", layout="wide")


@st.cache_data
def load_summary():
    summary_path = ARTIFACTS / "summary.json"
    lit_path = ARTIFACTS / "literature.json"
    if not summary_path.exists():
        return None, None
    return json.loads(summary_path.read_text()), json.loads(lit_path.read_text())


@st.cache_data
def load_task_artifacts(task_key: str):
    d = ARTIFACTS / task_key
    return {
        "metrics": json.loads((d / "metrics.json").read_text()),
        "leaderboard": pd.read_csv(d / "leaderboard.csv"),
        "history": pd.read_csv(d / "search_history.csv"),
        "importance": pd.read_csv(d / "feature_importance.csv", index_col=0),
        "sample": pd.read_csv(d / "holdout_sample.csv"),
    }


@st.cache_resource
def load_predictor(task_key: str):
    from autogluon.tabular import TabularPredictor
    path = MODELS / task_key
    if not path.exists():
        return None
    return TabularPredictor.load(str(path))


summary, literature = load_summary()

st.title("🧱 AutoGluon Multi-Layer Stacking Platform")
st.caption(
    "AutoML illustrated across three task types (binary, multiclass, regression) with an "
    "AutoResearch hill climb over AutoGluon's bagging + stacking architecture."
)

if summary is None:
    st.error(
        "No artifacts found. Run `python src/train.py` first — it fits every task and "
        "writes the results this dashboard reads under `artifacts/`."
    )
    st.stop()

task_keys = list(summary.keys())
overview, data_tab, methods, research, evaluation, scoring = st.tabs(
    ["Overview", "Data", "Methods", "AutoResearch", "Evaluation", "Live Scoring"]
)

# ---------------------------------------------------------------- Overview
with overview:
    st.subheader("Portfolio at a glance")
    cols = st.columns(len(task_keys))
    for col, key in zip(cols, task_keys):
        m = summary[key]
        primary_metric = m["eval_metric"]
        test_val = m["test_metrics"].get(primary_metric)
        with col:
            st.markdown(f"**{m['display_name']}**")
            st.caption(m["problem_type"])
            st.metric(f"test {primary_metric}", f"{test_val:.4f}" if test_val is not None else "n/a")
            st.metric("stacking layers fit", m["stack_levels_fit"])
            st.metric("best config", f"folds={m['best_config']['num_bag_folds']}, stack={m['best_config']['num_stack_levels']}")

    st.divider()
    st.subheader("What AutoResearch found")
    rows = []
    for key in task_keys:
        m = summary[key]
        rows.append({
            "task": m["display_name"],
            "baseline (no bag/stack) score_val": m["baseline_score_val"],
            "best score_val": m["best_score_val"],
            "improvement": m["best_score_val"] - m["baseline_score_val"],
            "winning config": f"folds={m['best_config']['num_bag_folds']}, stack_levels={m['best_config']['num_stack_levels']}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "score_val is AutoGluon's own out-of-fold validation score, always higher-is-better "
        "in its convention regardless of the underlying metric (so a negative RMSE going up "
        "toward 0 is an improvement)."
    )

# ---------------------------------------------------------------- Data
with data_tab:
    st.subheader("Task registry")
    for key in task_keys:
        m = summary[key]
        with st.expander(f"{m['display_name']} — {m['problem_type']}", expanded=False):
            st.write(f"**Source:** {m['source']}")
            st.write(f"**Rows:** {m['n_rows']} total → {m['n_train_search']} train/search, {m['n_test']} held-out test")
            st.write(f"**Eval metric:** `{m['eval_metric']}`")
            art = load_task_artifacts(key)
            st.write("Sample of held-out rows with model predictions:")
            st.dataframe(art["sample"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- Methods
with methods:
    st.subheader("How AutoGluon's stacking architecture works here")
    st.markdown(
        """
Each fit below is the same recipe, only the depth changes:

1. **Layer 1 (base models):** CatBoost, Random Forest, and Extra Trees are each
   trained on the training split. When `num_bag_folds > 0`, each one is
   trained as *k* bagged copies (k-fold cross-validation repeated across the
   whole model), and predictions are the bag's average — Breiman's variance-
   reduction argument for bagging.
2. **Weighted ensemble:** at the top of every layer, AutoGluon runs Caruana's
   greedy forward-selection ensembling over that layer's models, picking a
   non-negative weighted combination that maximizes validation score.
3. **Layer 2+ (stacking):** if `num_stack_levels > 0`, the next layer's base
   models train on the original features **plus** the out-of-fold predictions
   of every layer-1 model — Wolpert's stacked generalization — and get their
   own weighted ensemble on top.

LightGBM and XGBoost are intentionally excluded from the model pool: LightGBM
needs the system `libomp` runtime that a stock macOS/Python install lacks, and
XGBoost's source build needs `cmake` on Python 3.13/arm64 where no prebuilt
wheel is published yet. Excluding both keeps `pip install -r requirements.txt`
sufficient on a clean checkout; CatBoost + Random Forest + Extra Trees still
gives AutoResearch three genuinely different learners to bag and stack.
        """
    )
    st.subheader("Literature behind the AutoResearch search space")
    st.dataframe(pd.DataFrame(literature), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- AutoResearch
with research:
    st.subheader("Hill climb over (num_bag_folds, num_stack_levels)")
    task_key = st.selectbox("Task", task_keys, format_func=lambda k: summary[k]["display_name"], key="research_task")
    art = load_task_artifacts(task_key)
    hist = art["history"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["step"], y=hist["score_val"], mode="lines+markers",
        text=[f"folds={r.num_bag_folds}, stack={r.num_stack_levels}" for r in hist.itertuples()],
        hovertemplate="%{text}<br>score_val=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Evaluation order during the climb (best-so-far config only advances)",
        xaxis_title="evaluation #", yaxis_title="score_val (higher is better)",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(hist, use_container_width=True, hide_index=True)
    m = art["metrics"]
    st.info(
        f"Winner: **{m['best_config']}** at score_val={m['best_score_val']:.4f}, vs. baseline "
        f"(no bagging, no stacking) score_val={m['baseline_score_val']:.4f}, "
        f"searched in {m['search_seconds']}s over {m['search_evals']} fits."
    )

# ---------------------------------------------------------------- Evaluation
with evaluation:
    st.subheader("Final leaderboard and held-out test metrics")
    task_key = st.selectbox("Task", task_keys, format_func=lambda k: summary[k]["display_name"], key="eval_task")
    art = load_task_artifacts(task_key)
    m = art["metrics"]

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Model leaderboard** (validation score, by stack level):")
        st.dataframe(art["leaderboard"], use_container_width=True, hide_index=True)
    with c2:
        st.write("**Held-out test metrics** (scored once, after the search):")
        st.json(m["test_metrics"])
        st.caption(f"Best model selected for deployment: `{m['best_model']}`")

    st.write("**Permutation feature importance** (computed on the held-out test split):")
    imp = art["importance"].reset_index().rename(columns={"index": "feature"})
    fig2 = px.bar(imp.sort_values("importance", ascending=True).tail(15), x="importance", y="feature", orientation="h")
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- Live Scoring
with scoring:
    st.subheader("Score a held-out row with the deployed predictor")
    task_key = st.selectbox("Task", task_keys, format_func=lambda k: summary[k]["display_name"], key="score_task")
    predictor = load_predictor(task_key)
    art = load_task_artifacts(task_key)
    sample = art["sample"]

    if predictor is None:
        st.warning(
            f"No fitted predictor found at `models/{task_key}/`. Model directories are "
            "gitignored (50-250MB of bagged/stacked models each) — run `python src/train.py` "
            "once locally to regenerate them, then reload this page."
        )
    else:
        label_col = predictor.label
        row_idx = st.selectbox("Held-out row", sample.index, format_func=lambda i: f"row {i}")
        row = sample.loc[[row_idx]].drop(columns=["prediction"], errors="ignore")
        true_value = row[label_col].iloc[0]
        features = row.drop(columns=[label_col])

        edited = st.data_editor(features, use_container_width=True, num_rows="fixed")

        if st.button("Predict", type="primary"):
            pred = predictor.predict(edited).iloc[0]
            st.metric("Prediction", str(pred))
            st.metric("True value", str(true_value))
            if predictor.problem_type in ("binary", "multiclass"):
                proba = predictor.predict_proba(edited).iloc[0]
                st.write("Class probabilities:")
                st.bar_chart(proba)

st.divider()
st.caption(
    "Reference: Erickson, N. et al. (2020), \"AutoGluon-Tabular: Robust and Accurate AutoML "
    "for Structured Data\", arXiv:2003.06505. Datasets are scikit-learn's built-in "
    "breast_cancer, wine, and california_housing — no external download required."
)
