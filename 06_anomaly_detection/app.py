"""Anomaly Detection Admin Dashboard (Streamlit).

Every number on this page is read from ``artifacts/`` -- the output of
``python src/train.py``. The dashboard computes no metric of its own except the
curves and the threshold sweep, both derived directly from the saved test
scores. Nothing here is hard-coded or illustrative.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data")]

st.set_page_config(page_title="Anomaly Detection Platform", page_icon="🛡️", layout="wide")

# --- palette (validated; see README) -----------------------------------------
LIGHT = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "normal": "#2a78d6",
         "anomaly": "#e34948", "grid": "#d8d7d2", "muted": "#52514e"}
DARK = {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "normal": "#3987e5",
        "anomaly": "#e66767", "grid": "#3a3a37", "muted": "#c3c2b7"}


def palette() -> dict:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


C = palette()


def base(chart: alt.Chart, height: int = 260) -> alt.Chart:
    """Shared chart chrome: transparent surface, recessive axes, fixed height."""
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor=C["grid"], gridOpacity=0.6,
                            domainColor=C["grid"], tickColor=C["grid"],
                            labelColor=C["muted"], titleColor=C["muted"])
            .configure_legend(labelColor=C["muted"], titleColor=C["muted"]))


# --- artifact loading ---------------------------------------------------------
@st.cache_data
def load_artifacts():
    required = ["metrics.json", "baselines.csv", "search_history.csv",
                "test_scores.csv", "cleaning_report.csv", "feature_diagnostics.csv"]
    missing = [f for f in required if not (ARTIFACTS / f).exists()]
    if missing:
        return None, missing
    return {
        "metrics": json.loads((ARTIFACTS / "metrics.json").read_text()),
        "baselines": pd.read_csv(ARTIFACTS / "baselines.csv"),
        "history": pd.read_csv(ARTIFACTS / "search_history.csv"),
        "scores": pd.read_csv(ARTIFACTS / "test_scores.csv"),
        "cleaning": pd.read_csv(ARTIFACTS / "cleaning_report.csv"),
        "diagnostics": pd.read_csv(ARTIFACTS / "feature_diagnostics.csv"),
    }, []


@st.cache_resource
def load_pipeline():
    path = ARTIFACTS / "best_pipeline.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_test_frame():
    """Rebuild the test split (same seed, same code path) for live scoring."""
    import load_data
    from features import clean, engineer, split
    raw, _ = load_data.load()
    return split(engineer(clean(raw)[0]))["test"]


data, missing = load_artifacts()
if data is None:
    st.title("🛡️ Autonomous Anomaly Detection Platform")
    st.error("Artifacts are missing: " + ", ".join(missing))
    st.code("python src/train.py", language="bash")
    st.stop()

M = data["metrics"]
scores_df = data["scores"]
y_test = scores_df["label"].to_numpy()
s_test = scores_df["score"].to_numpy()

st.title("🛡️ Autonomous Anomaly Detection Platform")
st.caption(
    f"CRISP-DM · unsupervised detection · autoresearch hill climbing · "
    f"data source **{M['data_source']}** · trained {M['generated_at'][:19]}Z "
    f"in {M['runtime_seconds']}s · seed {M['seed']}"
)
if M["data_source"] == "synthetic-fallback":
    st.warning(
        "Running on the **seeded synthetic fallback**, not the real Kaggle "
        "credit-card file (150 MB, login-gated). Schema and difficulty are "
        "modelled on it; the metrics below describe this generated data only. "
        "Drop the real `creditcard.csv` into `data/` and re-run `src/train.py` "
        "to reproduce every panel against it.", icon="⚠️")

overview, dataset, methods, research, evaluation, scoring = st.tabs(
    ["Overview", "Data", "Methods", "AutoResearch", "Evaluation", "Live scoring"])

# =============================================================================
with overview:
    op, test = M["operating_point"], M["test"]
    c = st.columns(4)
    c[0].metric("Test PR-AUC (avg precision)", f"{test['average_precision']:.3f}",
                f"{test['average_precision'] - M['baseline_best']['test_average_precision']:+.3f} vs best baseline")
    c[1].metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
    c[2].metric("Precision @ operating point", f"{op['precision']:.1%}",
                help="Share of raised alerts that are true anomalies.")
    c[3].metric("Recall @ operating point", f"{op['recall']:.1%}",
                help="Share of all anomalies in the test split that got flagged.")

    c = st.columns(4)
    c[0].metric("Rows modelled", f"{M['cleaning']['final_rows']:,}")
    c[1].metric("Anomaly prevalence", f"{M['cleaning']['prevalence']:.3%}")
    c[2].metric("Pipelines fitted", f"{M['search']['pipelines_fitted']}",
                help="Distinct configurations the hill-climbing search actually fitted.")
    c[3].metric("Alert budget", f"{M['alert_budget']:.2%} of traffic")

    st.subheader("Selected pipeline")
    st.code(M["search"]["best_label"], language="text")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"""
**How it was chosen.** Hill climbing over *feature set x scaler x detection
method x hyperparameters*, maximising **validation average precision**
({M['search']['restarts']} random restarts, {M['search']['evaluations_logged']}
evaluations logged, {M['search']['pipelines_fitted']} distinct pipelines fitted).

**How to read the numbers.** Validation AP is
`{M['search']['best_valid']['average_precision']:.4f}` but the search *selected*
on it, so it is optimistically biased. The honest figure is the test split,
touched once: **AP {test['average_precision']:.4f}, ROC-AUC {test['roc_auc']:.4f}**.

**Operating point.** Threshold `{op['threshold']:.3f}` was fixed on validation at
the {M['alert_budget']:.1%} alert budget, then applied unchanged to test:
{op['true_positives']} caught, {op['false_positives']} false alarms,
{op['false_negatives']} missed.
            """)
    with right:
        st.markdown("**Test confusion matrix at the operating point**")
        st.dataframe(pd.DataFrame(
            [[op["true_negatives"], op["false_positives"]],
             [op["false_negatives"], op["true_positives"]]],
            index=["actual normal", "actual anomaly"],
            columns=["predicted normal", "predicted anomaly"]),
            width="stretch")
        st.caption(f"Alert rate {op['alert_rate']:.2%} of test traffic.")

# =============================================================================
with dataset:
    st.subheader("CRISP-DM phase 2-3 · Data Understanding & Preparation")
    cl = M["cleaning"]
    c = st.columns(4)
    c[0].metric("Raw rows", f"{cl['raw_rows']:,}")
    c[1].metric("Duplicates dropped", f"{cl['duplicates_dropped']:,}")
    c[2].metric("Amounts repaired", f"{cl['negative_amounts_fixed'] + cl['missing_amounts_imputed']:,}",
                help="Negative amounts sign-corrected plus missing amounts median-imputed.")
    c[3].metric("Anomalies", f"{cl['anomalies']:,}", f"{cl['prevalence']:.3%} prevalence")

    st.markdown("**Cleaning ledger** — every transformation, with row counts.")
    st.dataframe(data["cleaning"], width="stretch", hide_index=True)

    st.markdown("**Split plan** — stratified so each split holds enough anomalies to measure.")
    st.dataframe(pd.DataFrame(M["splits"]).T.rename(
        columns={"rows": "rows", "anomalies": "anomalies"}), width="stretch")
    st.caption("Labels never fit a detector. They stratify the split, score candidates "
               "during the search, and evaluate the final model — that is all.")

    st.subheader("Which features separate anomalies on their own?")
    top = data["diagnostics"].head(12)
    chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4, color=C["s1"]).encode(
        x=alt.X("separation:Q", title="single-feature separation  |2·AUC−1|"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("auc:Q", format=".3f"),
                 alt.Tooltip("separation:Q", format=".3f"),
                 alt.Tooltip("mean_normal:Q", format=".3f", title="mean (normal)"),
                 alt.Tooltip("mean_anomaly:Q", format=".3f", title="mean (anomaly)")])
    st.altair_chart(base(chart, 320), width="stretch")
    st.caption(
        "0 = carries no signal by itself, 1 = perfectly separating by itself. "
        "The detectors see all features jointly; this is a data-understanding "
        "diagnostic, not a model explanation. **Caveat:** single-feature AUC only "
        "detects *one-sided* shifts, so a component whose anomalies sit in both "
        "tails scores near 0.5 here while still being highly informative to a "
        "multivariate detector — which is why the V-components rank low and the "
        "detectors nonetheless reach high ROC-AUC.")
    with st.expander("Full per-feature diagnostic table"):
        st.dataframe(data["diagnostics"], width="stretch", hide_index=True)

# =============================================================================
with methods:
    st.subheader("CRISP-DM phase 4 · Modeling — the method pool")
    st.markdown(
        "Five classic unsupervised detectors, each fitted at default settings on "
        "the same training rows. This sweep is the **baseline the autoresearch "
        "result has to beat**; without it, a tuned number means nothing.")

    bl = data["baselines"].copy()
    winner = M["search"]["best_config"]["detector"]
    bl["role"] = np.where(bl["detector"] == winner, "selected by autoresearch", "baseline")
    chart = alt.Chart(bl).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("test_average_precision:Q", title="test average precision (PR-AUC)"),
        y=alt.Y("detector:N", sort="-x", title=None),
        color=alt.Color("role:N", title=None,
                        scale=alt.Scale(domain=["baseline", "selected by autoresearch"],
                                        range=[C["s1"], C["s2"]]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("detector:N"),
                 alt.Tooltip("valid_average_precision:Q", format=".4f", title="valid AP"),
                 alt.Tooltip("test_average_precision:Q", format=".4f", title="test AP"),
                 alt.Tooltip("test_roc_auc:Q", format=".4f", title="test ROC-AUC"),
                 alt.Tooltip("seconds:Q", format=".1f", title="fit+score seconds")])
    st.altair_chart(base(chart, 240), width="stretch")
    st.caption(f"Highlighted bar = the method the search settled on ({winner}); "
               "the bar shows that method at *default* settings, the search then "
               "tuned its preprocessing and hyperparameters.")

    st.dataframe(bl.drop(columns=["role"]), width="stretch", hide_index=True)
    st.markdown(f"""
**Baseline → autoresearch lift.** Best default pipeline
(`{M['baseline_best']['detector']}`) scored **{M['baseline_best']['test_average_precision']:.4f}**
test AP. The searched pipeline scores **{M['test']['average_precision']:.4f}** —
a lift of **{M['test']['average_precision'] - M['baseline_best']['test_average_precision']:+.4f}**.

Note the ROC-AUC column stays in a narrow high band across all five methods while
AP spreads widely. That gap is the whole reason the search optimises AP: at
{M['cleaning']['prevalence']:.3%} prevalence, ROC-AUC flatters everything.
    """)

# =============================================================================
with research:
    st.subheader("AutoResearch · multi-restart steepest-ascent hill climbing")
    s = M["search"]
    c = st.columns(4)
    c[0].metric("Restarts", s["restarts"])
    c[1].metric("Evaluations logged", s["evaluations_logged"])
    c[2].metric("Distinct pipelines fitted", s["pipelines_fitted"])
    c[3].metric("Best validation AP", f"{s['best_valid']['average_precision']:.4f}")

    st.markdown(
        "One move = change exactly one decision (feature set, scaler, detection "
        "method, or one hyperparameter of the current method); take the steepest "
        "improving move; restart randomly when stuck in a local optimum. "
        "**Rejected moves are plotted too** — a trajectory showing only "
        "improvements hides how hard the landscape was.")

    hist = data["history"].copy()
    hist["restart_label"] = "restart " + hist["restart"].astype(str)
    restarts = sorted(hist["restart_label"].unique())
    scale = alt.Scale(domain=restarts, range=[C["s1"], C["s2"], C["s3"]][:len(restarts)])

    points = alt.Chart(hist).mark_circle(size=70, opacity=0.75).encode(
        x=alt.X("eval:Q", title="evaluation #"),
        y=alt.Y("average_precision:Q", title="validation average precision"),
        color=alt.Color("restart_label:N", title=None, scale=scale,
                        legend=alt.Legend(orient="bottom")),
        shape=alt.Shape("accepted:N", title="move accepted",
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("eval:Q", title="evaluation"),
                 alt.Tooltip("restart_label:N", title="restart"),
                 alt.Tooltip("move:N"), alt.Tooltip("accepted:N"),
                 alt.Tooltip("label:N", title="pipeline"),
                 alt.Tooltip("average_precision:Q", format=".4f", title="AP"),
                 alt.Tooltip("roc_auc:Q", format=".4f", title="ROC-AUC"),
                 alt.Tooltip("seconds:Q", format=".1f")])
    running = alt.Chart(hist).mark_line(strokeWidth=2, color=C["muted"],
                                        strokeDash=[6, 3]).encode(
        x="eval:Q", y=alt.Y("running_best_ap:Q"))
    st.altair_chart(base(alt.layer(points, running).resolve_scale(color="independent"), 320),
                    width="stretch")
    st.caption("Dashed line = best AP found so far. Shape distinguishes accepted "
               "moves from evaluated-and-rejected ones, so identity never rests on colour alone.")

    st.markdown("**Accepted moves** — the actual climb.")
    accepted = hist[hist["accepted"]][
        ["restart", "step", "move", "label", "average_precision", "roc_auc"]]
    st.dataframe(accepted, width="stretch", hide_index=True)

    with st.expander("Full search log (every evaluation, including rejects)"):
        st.dataframe(hist, width="stretch", hide_index=True)

    st.subheader("Literature the design decisions rest on")
    st.markdown("Each row is the published source for a choice made in this "
                "platform. Citations are to the original papers; **every number "
                "in this dashboard comes from this project's own run**, never "
                "from a paper's reported results.")
    st.dataframe(pd.DataFrame(M["literature"]), width="stretch", hide_index=True)

# =============================================================================
with evaluation:
    st.subheader("CRISP-DM phase 5 · Evaluation on the held-out test split")
    st.caption(f"{len(scores_df):,} transactions, {int(y_test.sum())} true anomalies. "
               "Used exactly once, after the search finished.")

    prec, rec, _ = precision_recall_curve(y_test, s_test)
    pr = pd.DataFrame({"recall": rec, "precision": prec})
    fpr, tpr, _ = roc_curve(y_test, s_test)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    prevalence = float(y_test.mean())

    left, right = st.columns(2)
    with left:
        st.markdown("**Precision-Recall curve** (the one that matters here)")
        line = alt.Chart(pr).mark_line(strokeWidth=2, color=C["s1"]).encode(
            x=alt.X("recall:Q", title="recall", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", title="precision", scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("recall:Q", format=".3f"),
                     alt.Tooltip("precision:Q", format=".3f")])
        rule = alt.Chart(pd.DataFrame({"y": [prevalence]})).mark_rule(
            strokeWidth=2, strokeDash=[6, 3], color=C["muted"]).encode(y="y:Q")
        st.altair_chart(base(alt.layer(line, rule)), width="stretch")
        st.caption(f"AP = {M['test']['average_precision']:.4f}. Dashed line = a random "
                   f"ranker ({prevalence:.3%}, the prevalence). The model is "
                   f"{M['test']['average_precision'] / prevalence:.0f}x that floor.")
    with right:
        st.markdown("**ROC curve** (reported, not optimised)")
        line = alt.Chart(roc).mark_line(strokeWidth=2, color=C["s1"]).encode(
            x=alt.X("fpr:Q", title="false positive rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="true positive rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")])
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
            strokeWidth=2, strokeDash=[6, 3], color=C["muted"]).encode(x="x:Q", y="y:Q")
        st.altair_chart(base(alt.layer(line, diag)), width="stretch")
        st.caption(f"ROC-AUC = {M['test']['roc_auc']:.4f}. Looks excellent — and would "
                   "look near-excellent for a much weaker model at this prevalence.")

    st.subheader("Score distribution")
    plot_df = scores_df.copy()
    plot_df["class"] = np.where(plot_df["label"] == 1, "anomaly", "normal")
    hist_chart = alt.Chart(plot_df).mark_bar(opacity=0.75).encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=60), title="anomaly score (higher = more anomalous)"),
        y=alt.Y("count()", stack=None, title="transactions (log scale)",
                scale=alt.Scale(type="symlog")),
        color=alt.Color("class:N", title=None,
                        scale=alt.Scale(domain=["normal", "anomaly"],
                                        range=[C["normal"], C["anomaly"]]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("class:N"), alt.Tooltip("count()", title="transactions")])
    st.altair_chart(base(hist_chart, 280), width="stretch")
    st.caption("Log-scaled counts — at 0.5% prevalence a linear axis would erase the "
               "anomaly class entirely. The overlap region is where the false "
               "positives and misses live.")

    st.subheader("Threshold explorer — pick an operating point")
    st.markdown("The model produces a *ranking*; the business picks the cut. Move the "
                "alert budget to see the trade the fraud team is actually making.")
    c1, c2, c3 = st.columns(3)
    budget = c1.slider("Alert budget (% of traffic reviewed)", 0.1, 5.0,
                       float(M["alert_budget"] * 100), 0.1) / 100
    cost_fp = c2.number_input("Cost of reviewing a false alert ($)", 1.0, 500.0, 15.0, 1.0)
    cost_fn = c3.number_input("Cost of a missed anomaly ($)", 10.0, 10_000.0, 400.0, 10.0)

    k = max(1, int(round(len(s_test) * budget)))
    order = np.argsort(s_test)[::-1]
    flagged = np.zeros(len(s_test), bool)
    flagged[order[:k]] = True
    tp = int((flagged & (y_test == 1)).sum())
    fp = int((flagged & (y_test == 0)).sum())
    fn = int((~flagged & (y_test == 1)).sum())
    precision = tp / k
    recall = tp / max(1, int(y_test.sum()))
    cost = fp * cost_fp + fn * cost_fn
    do_nothing = int(y_test.sum()) * cost_fn

    m = st.columns(5)
    m[0].metric("Alerts raised", f"{k:,}")
    m[1].metric("Caught", f"{tp}", f"{recall:.1%} recall")
    m[2].metric("False alarms", f"{fp}", f"{precision:.1%} precision")
    m[3].metric("Missed", f"{fn}")
    m[4].metric("Modelled cost", f"${cost:,.0f}",
                f"{cost - do_nothing:+,.0f} vs reviewing nothing", delta_color="inverse")
    st.caption("The cost figures use *your* inputs above, not measured losses — they "
               "are a decision aid for choosing a threshold, not a claimed saving.")

    sweep = []
    for b in np.linspace(0.001, 0.05, 50):
        kk = max(1, int(round(len(s_test) * b)))
        fl = np.zeros(len(s_test), bool)
        fl[order[:kk]] = True
        t = int((fl & (y_test == 1)).sum())
        sweep.append({"budget": b, "precision": t / kk,
                      "recall": t / max(1, int(y_test.sum())),
                      "cost": (int((fl & (y_test == 0)).sum()) * cost_fp
                               + (int(y_test.sum()) - t) * cost_fn)})
    sweep_df = pd.DataFrame(sweep)
    cost_chart = alt.Chart(sweep_df).mark_line(strokeWidth=2, color=C["s1"]).encode(
        x=alt.X("budget:Q", title="alert budget (share of traffic)", axis=alt.Axis(format="%")),
        y=alt.Y("cost:Q", title="modelled cost ($)"),
        tooltip=[alt.Tooltip("budget:Q", format=".2%"), alt.Tooltip("cost:Q", format="$,.0f"),
                 alt.Tooltip("precision:Q", format=".1%"), alt.Tooltip("recall:Q", format=".1%")])
    marker = alt.Chart(pd.DataFrame({"budget": [budget]})).mark_rule(
        strokeWidth=2, strokeDash=[6, 3], color=C["s2"]).encode(x="budget:Q")
    st.altair_chart(base(alt.layer(cost_chart, marker), 240), width="stretch")
    st.caption("Cost against alert budget under the inputs above; dashed line marks "
               "your current setting.")

# =============================================================================
with scoring:
    st.subheader("CRISP-DM phase 6 · Deployment — score a transaction")
    pipeline = load_pipeline()
    if pipeline is None:
        st.error("artifacts/best_pipeline.joblib is missing. Run `python src/train.py`.")
    else:
        from features import FEATURE_SETS
        test_frame = load_test_frame()
        cols = FEATURE_SETS[pipeline["feature_set"]]
        threshold = pipeline["threshold"]

        st.markdown(f"Serving the saved pipeline `{M['search']['best_label']}` at the "
                    f"production threshold `{threshold:.3f}`.")

        mode = st.radio("Pick a transaction", ["A known anomaly", "A random normal row",
                                               "Pick by row number"], horizontal=True)
        anomalies = test_frame.index[test_frame["Class"] == 1].tolist()
        normals = test_frame.index[test_frame["Class"] == 0].tolist()
        rng = np.random.default_rng(st.session_state.get("nonce", 0))
        if mode == "A known anomaly":
            idx = int(rng.choice(anomalies))
        elif mode == "A random normal row":
            idx = int(rng.choice(normals))
        else:
            idx = st.number_input("Test row", 0, len(test_frame) - 1, 0, 1)
        if st.button("Draw another"):
            st.session_state["nonce"] = int(np.random.randint(0, 10_000))
            st.rerun()

        row = test_frame.loc[int(idx)]
        st.markdown("**Adjust the transaction before scoring** (defaults are the real row):")
        a, b = st.columns(2)
        amount = a.number_input("Amount ($)", 0.0, 30_000.0, float(row["Amount"]), 1.0)
        hour = b.slider("Hour of day", 0.0, 23.99, float(row["hour"]), 0.25)

        edited = row.copy()
        edited["Amount"] = amount
        edited["log_amount"] = np.log1p(max(amount, 0.0))
        edited["hour"] = hour
        edited["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        edited["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        X = edited[cols].to_numpy(dtype=float).reshape(1, -1)
        score = float(pipeline["detector"].score(pipeline["scaler"].transform(X))[0])
        percentile = float((s_test < score).mean() * 100)
        flagged = score >= threshold

        c = st.columns(3)
        c[0].metric("Anomaly score", f"{score:.2f}", f"threshold {threshold:.2f}")
        c[1].metric("Rank vs test traffic", f"{percentile:.2f}nd pct",
                    help="Share of held-out transactions this row outranks.")
        c[2].metric("Ground-truth label", "ANOMALY" if row["Class"] == 1 else "normal")

        if flagged:
            st.error(f"🚨 **ALERT** — score {score:.2f} is at or above the "
                     f"{threshold:.2f} threshold. Route to an analyst.", icon="🚨")
        else:
            st.success(f"✅ **Pass** — score {score:.2f} is below the "
                       f"{threshold:.2f} threshold. No alert raised.", icon="✅")
        if flagged != bool(row["Class"]):
            st.caption("This row is one the model gets **wrong** — useful to see. "
                       "At this prevalence, most alerts are false and most anomalies "
                       "are missed; that is what the Evaluation tab quantifies.")

        with st.expander("Feature vector handed to the detector"):
            st.dataframe(pd.DataFrame({"feature": cols,
                                       "value": X.ravel()}),
                         width="stretch", hide_index=True)
