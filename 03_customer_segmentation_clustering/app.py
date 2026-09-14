"""Customer Intelligence admin dashboard (CRISP-DM + AutoResearch).

Run:  streamlit run app.py      (after `python src/train.py`)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Customer Intelligence Console",
                   page_icon="◧", layout="wide")

# Categorical slots 1-6 of the validated light-surface palette, bound to segment
# NAMES (the entity) rather than cluster ids, so a colour never migrates between
# segments across retrains. Every chart also ships labels or a table, which is
# the required relief for the three slots under 3:1 contrast.
SEGMENT_ORDER = ["Champions", "Loyal Regulars", "Big-Ticket Buyers",
                 "Promising / New", "At Risk", "Hibernating"]
SEGMENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID, FOCUS = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"

FEATURE_DOCS = {
    "recency_days": "Days from the customer's last purchase to the snapshot date.",
    "frequency": "Distinct purchase invoices (cancellations excluded).",
    "monetary": "Total spend across all purchase lines.",
    "avg_order_value": "monetary / frequency.",
    "avg_basket_size": "Mean number of line items per invoice.",
    "tenure_days": "Days from first purchase to the snapshot date.",
    "return_rate": "Cancelled invoices / purchase invoices.",
    "distinct_products": "Number of unique StockCodes bought.",
    "RFM_score": "Sum of the 1-5 recency, frequency and monetary quintile scores (Hughes 1994).",
}

CRISP_DM = [
    ("1. Business understanding",
     "Split a retail customer base into segments a marketing team can actually run "
     "campaigns against, and justify the split with published validity indices rather "
     "than an eyeballed elbow.",
     "README.md, the actionability constraint in src/autoresearch.py"),
    ("2. Data understanding",
     "Transaction-level retail log: invoices, line items, prices, customers, countries. "
     "Known defects: guest checkouts with no customer id, cancellations, zero-price "
     "lines, duplicated rows.",
     "data/generate_data.py, the Data & CRISP-DM tab"),
    ("3. Data preparation",
     "Drop duplicates, guest checkouts and zero-price lines; keep cancellations as "
     "negative revenue; aggregate to one row per customer with RFM plus derived "
     "behaviour features.",
     "src/features.py"),
    ("4. Modeling",
     "AutoResearch hill climbing over 180 preprocessing + clustering pipelines "
     "(feature set x log transform x scaler x algorithm x k).",
     "src/autoresearch.py"),
    ("5. Evaluation",
     "Internal indices, subsample stability (ARI), agreement with the RFM-quintile "
     "baseline, an external check against the generator's latent archetypes, and an "
     "exhaustive-grid optimality gap.",
     "src/train.py, the AutoResearch and Model card tabs"),
    ("6. Deployment",
     "This dashboard: segment profiles, per-customer lookup, and a nearest-centroid "
     "scoring path for a customer that was not in the training snapshot.",
     "app.py, artifacts/best_pipeline.joblib"),
]


# ----------------------------------------------------------------------------- loading
@st.cache_data(show_spinner=False)
def load_artifacts() -> dict | None:
    needed = ["metrics.json", "customer_segments.csv", "segment_profile.csv",
              "search_history.csv", "cleaning_report.csv"]
    if not all((ARTIFACTS / f).exists() for f in needed):
        return None
    out = {
        "metrics": json.loads((ARTIFACTS / "metrics.json").read_text()),
        "customers": pd.read_csv(ARTIFACTS / "customer_segments.csv"),
        "profile": pd.read_csv(ARTIFACTS / "segment_profile.csv"),
        "history": pd.read_csv(ARTIFACTS / "search_history.csv"),
        "cleaning": pd.read_csv(ARTIFACTS / "cleaning_report.csv"),
    }
    for optional in ["grid_results.csv", "archetype_crosstab.csv"]:
        path = ARTIFACTS / optional
        out[optional[:-4]] = pd.read_csv(path) if path.exists() else None
    return out


@st.cache_resource(show_spinner=False)
def load_pipeline():
    import joblib
    path = ARTIFACTS / "best_pipeline.joblib"
    return joblib.load(path) if path.exists() else None


def segment_scale(domain: list[str]) -> alt.Scale:
    """Fixed name -> hue binding; never cycled, never rank-ordered."""
    order = [s for s in SEGMENT_ORDER if s in domain] + \
            [s for s in domain if s not in SEGMENT_ORDER]
    return alt.Scale(domain=order, range=SEGMENT_COLORS[:len(order)])


def styled(chart: alt.Chart) -> alt.Chart:
    return (chart
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor=GRID, gridWidth=1, domain=False,
                            tickColor=GRID, labelColor=MUTED, titleColor=MUTED,
                            labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=11,
                              titleFontSize=11, symbolType="square", symbolSize=90)
            .configure_axisX(grid=False))


def ranked_bar(df: pd.DataFrame, value: str, title: str, fmt: str,
               height: int = 240) -> alt.Chart:
    """Horizontal bar, sorted by magnitude, with a direct label on every bar."""
    order = alt.EncodingSortField(field=value, op="max", order="descending")
    base = alt.Chart(df).encode(
        y=alt.Y("segment:N", sort=order, title=None,
                axis=alt.Axis(labelColor=INK, labelFontSize=12)),
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(format=fmt),
                scale=alt.Scale(domain=[0, float(df[value].max()) * 1.15], nice=False)),
        color=alt.Color("segment:N", scale=segment_scale(df["segment"].tolist()),
                        legend=None),
        tooltip=[alt.Tooltip("segment:N", title="Segment"),
                 alt.Tooltip(f"{value}:Q", title=title, format=fmt)],
    )
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18)
    labels = base.mark_text(align="left", dx=6, color=INK, fontSize=11).encode(
        text=alt.Text(f"{value}:Q", format=fmt), color=alt.value(INK))
    return (bars + labels).properties(height=height)


# ----------------------------------------------------------------------------- shell
data = load_artifacts()
st.title("Customer Intelligence Console")
st.caption("RFM segmentation discovered by an automated hill-climbing search over "
           "clustering pipelines, scored with published cluster-validity indices.")

if data is None:
    st.error("No artifacts found. Build them first:")
    st.code("python data/generate_data.py\npython src/train.py", language="bash")
    st.stop()

M = data["metrics"]
customers, profile, history = data["customers"], data["profile"], data["history"]
best, search = M["best"], M["search"]
cfg = best["config"]

with st.sidebar:
    st.subheader("Run metadata")
    st.write(f"**Trained** {M['generated_at']}")
    st.write(f"**Python** {M['python']}  ·  **seed** {M['seed']}")
    st.write(f"**Pipeline runtime** {M['runtime_seconds']} s")
    st.divider()
    st.subheader("Winning pipeline")
    st.json(cfg, expanded=True)
    st.divider()
    st.caption("Rebuild artifacts with `python src/train.py`, then press R to rerun.")

tabs = st.tabs(["Overview", "Segments", "AutoResearch", "Research grounding",
                "Data & CRISP-DM", "Model card"])

# ----------------------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Customers", f"{M['data']['customers']:,}")
    c[1].metric("Revenue in window", f"${M['data']['total_revenue']:,.0f}")
    c[2].metric("Segments", cfg["k"])
    c[3].metric("Silhouette", f"{best['silhouette']:.3f}")
    c[4].metric("Stability (ARI)", f"{M['stability']['ari_mean']:.3f}",
                help=f"Mean adjusted Rand index over {M['stability']['n_trials']} refits on "
                     f"{M['stability']['subsample_frac']:.0%} subsamples "
                     f"(min {M['stability']['ari_min']:.3f}).")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Customers per segment")
        st.altair_chart(styled(ranked_bar(profile, "customers", "Customers", ",.0f")),
                        width="stretch")
    with right:
        st.subheader("Share of revenue")
        st.altair_chart(styled(ranked_bar(profile, "revenue_share", "Share of revenue", ".1%")),
                        width="stretch")

    st.info(
        f"**{profile.loc[profile['revenue_share'].idxmax(), 'segment']}** holds "
        f"{profile['revenue_share'].max():.0%} of revenue from "
        f"{profile.loc[profile['revenue_share'].idxmax(), 'customers'] / profile['customers'].sum():.0%} "
        "of customers — the concentration that makes retention spend worthwhile.")

    st.subheader("Segment profile")
    st.caption("Medians per segment, except return rate (mean) and revenue (sum).")
    show = profile[["segment", "customers", "recency_days", "frequency", "monetary",
                    "avg_order_value", "avg_basket_size", "tenure_days", "return_rate",
                    "rfm_score", "revenue", "revenue_share", "mean_silhouette",
                    "recommended_action"]].copy()
    # NumberColumn's printf format prints the number as-is, so scale the fraction here
    show["revenue_share"] = show["revenue_share"] * 100.0
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={
            "segment": "Segment", "customers": "Customers",
            "recency_days": st.column_config.NumberColumn("Recency (d)", format="%.0f"),
            "frequency": st.column_config.NumberColumn("Frequency", format="%.0f"),
            "monetary": st.column_config.NumberColumn("Monetary", format="$%.0f"),
            "avg_order_value": st.column_config.NumberColumn("AOV", format="$%.0f"),
            "avg_basket_size": st.column_config.NumberColumn("Basket", format="%.1f"),
            "tenure_days": st.column_config.NumberColumn("Tenure (d)", format="%.0f"),
            "return_rate": st.column_config.NumberColumn("Return rate", format="%.2f"),
            "rfm_score": st.column_config.NumberColumn("RFM score", format="%.0f"),
            "revenue": st.column_config.NumberColumn("Revenue", format="$%.0f"),
            "revenue_share": st.column_config.NumberColumn("Rev. share", format="%.1f%%"),
            "mean_silhouette": st.column_config.NumberColumn("Silhouette", format="%.3f"),
            "recommended_action": "Recommended action"})

# ----------------------------------------------------------------------------- segments
with tabs[1]:
    names = [s for s in SEGMENT_ORDER if s in set(profile["segment"])] + \
            [s for s in profile["segment"] if s not in SEGMENT_ORDER]
    picked = st.selectbox("Focus segment", names)
    row = profile[profile["segment"] == picked].iloc[0]
    members = customers[customers["segment"] == picked]

    c = st.columns(5)
    c[0].metric("Customers", f"{int(row['customers']):,}")
    c[1].metric("Median recency", f"{row['recency_days']:.0f} d")
    c[2].metric("Median frequency", f"{row['frequency']:.0f}")
    c[3].metric("Median spend", f"${row['monetary']:,.0f}")
    c[4].metric("Mean silhouette", f"{row['mean_silhouette']:.3f}",
                help="Below ~0.25 means the segment's members sit close to a neighbouring "
                     "segment's boundary and the split is weakly supported there.")
    st.success(f"**Recommended action —** {row['recommended_action']}")

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Where this segment sits")
        st.caption("Recency against spend, log spend axis. The focus segment is "
                   "highlighted; everyone else is grey — two colours, so the scatter "
                   "stays readable for colour-vision-deficient viewers.")
        plot = customers.assign(focus=np.where(customers["segment"] == picked, picked, "Other"))
        scatter = alt.Chart(plot).mark_circle(size=42, opacity=0.75,
                                              stroke="#ffffff", strokeWidth=1).encode(
            x=alt.X("recency_days:Q", title="Recency (days since last purchase)"),
            y=alt.Y("monetary:Q", title="Total spend ($)",
                    scale=alt.Scale(type="log")),
            color=alt.Color("focus:N",
                            scale=alt.Scale(domain=[picked, "Other"], range=[FOCUS, "#c9c8c3"]),
                            legend=alt.Legend(title=None, orient="top")),
            size=alt.Size("focus:N", scale=alt.Scale(domain=[picked, "Other"], range=[60, 22]),
                          legend=None),
            tooltip=[alt.Tooltip("CustomerID:Q", title="Customer", format=".0f"),
                     alt.Tooltip("segment:N", title="Segment"),
                     alt.Tooltip("recency_days:Q", title="Recency (d)", format=".0f"),
                     alt.Tooltip("frequency:Q", title="Frequency", format=".0f"),
                     alt.Tooltip("monetary:Q", title="Spend", format="$,.0f")],
        ).properties(height=380)
        st.altair_chart(styled(scatter), width="stretch")
    with right:
        st.subheader("Profile vs. the whole base")
        overall = customers[["recency_days", "frequency", "monetary",
                             "avg_order_value", "avg_basket_size"]].median()
        seg_med = members[overall.index].median()
        comp = pd.DataFrame({
            "feature": overall.index,
            "ratio": (seg_med / overall.replace(0, np.nan)).to_numpy(),
            "segment_median": seg_med.to_numpy(),
            "base_median": overall.to_numpy(),
        })
        bar = alt.Chart(comp).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                                       height=18, color=FOCUS).encode(
            y=alt.Y("feature:N", sort=None, title=None,
                    axis=alt.Axis(labelColor=INK, labelFontSize=12)),
            x=alt.X("ratio:Q", title="Segment median / base median (1.0 = typical)",
                    scale=alt.Scale(domain=[0, float(comp["ratio"].max()) * 1.16],
                                    nice=False)),
            tooltip=[alt.Tooltip("feature:N", title="Feature"),
                     alt.Tooltip("ratio:Q", title="x base median", format=".2f"),
                     alt.Tooltip("segment_median:Q", title="Segment", format=",.1f"),
                     alt.Tooltip("base_median:Q", title="Base", format=",.1f")])
        rule = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(
            color=MUTED, strokeDash=[4, 3]).encode(x="x:Q")
        text = alt.Chart(comp).mark_text(align="left", dx=6, color=INK, fontSize=11).encode(
            y=alt.Y("feature:N", sort=None), x="ratio:Q",
            text=alt.Text("ratio:Q", format=".2f"))
        st.altair_chart(styled((bar + rule + text).properties(height=240)),
                        width="stretch")
        st.dataframe(comp.round(2), hide_index=True, width="stretch")

    st.subheader(f"Members of {picked}")
    cols = ["CustomerID", "country", "recency_days", "frequency", "monetary",
            "avg_order_value", "avg_basket_size", "tenure_days", "return_rate",
            "RFM_score", "silhouette"]
    st.dataframe(members[cols].sort_values("monetary", ascending=False),
                 hide_index=True, width="stretch", height=280)
    st.download_button(f"Download {picked} as CSV",
                       members[cols].to_csv(index=False).encode(),
                       file_name=f"segment_{picked.lower().replace(' / ', '_').replace(' ', '_')}.csv",
                       mime="text/csv")

# ----------------------------------------------------------------------------- autoresearch
with tabs[2]:
    st.subheader("Hill-climbing search")
    st.markdown(
        f"""**Strategy** {search['strategy']} · **space** {search['space_size']} candidate
pipelines · **restarts** {search['restarts']} · **distinct pipelines actually fitted**
{search['distinct_pipelines_fitted']} (every fit is cached, so this is the true cost).

Each step evaluates all one-coordinate neighbours of the current configuration and
moves to the best strictly-improving one; when none improves, the restart has hit a
local optimum.""")

    c = st.columns(4)
    c[0].metric("Composite objective", f"{best['objective']:.4f}")
    c[1].metric("Silhouette", f"{best['silhouette']:.3f}")
    c[2].metric("Davies–Bouldin", f"{best['davies_bouldin']:.3f}", help="Lower is better.")
    c[3].metric("Calinski–Harabasz", f"{best['calinski_harabasz']:,.0f}")

    grid = search.get("grid_reference")
    if grid:
        st.divider()
        g = st.columns(4)
        g[0].metric("Grid best objective", f"{grid['grid_best_objective']:.4f}")
        g[1].metric("Optimality gap", f"{grid['optimality_gap']:+.4f}")
        g[2].metric("Reached global optimum",
                    "yes" if grid["hill_climb_found_global_optimum"] else "no")
        g[3].metric("Fits saved vs. full grid", f"{grid['fits_saved_vs_grid']}")
        st.caption(f"{grid['n_feasible']} of {grid['n_configs']} configurations satisfied the "
                   f"actionability constraint. The exhaustive grid is a verification run only "
                   f"— hill climbing never sees it.")

    st.divider()
    st.subheader("Search trajectory")
    traj = history.copy()
    traj["evaluation"] = np.arange(1, len(traj) + 1)
    traj["restart_label"] = "restart " + traj["restart"].astype(str)
    traj["running_best"] = traj["cur_objective"].cummax()
    traj["outcome"] = np.where(traj["moved"], "improved", "local optimum")

    best_line = alt.Chart(traj).mark_line(interpolate="step-after", strokeWidth=2,
                                          color=INK).encode(
        x=alt.X("evaluation:Q", title="Hill-climbing step (all restarts, in order)"),
        y=alt.Y("running_best:Q", title="Best objective so far",
                scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("evaluation:Q", title="Step"),
                 alt.Tooltip("running_best:Q", title="Best so far", format=".4f"),
                 alt.Tooltip("n_evaluated:Q", title="Pipelines fitted")])
    st.altair_chart(styled(best_line.properties(height=190)), width="stretch")
    st.caption("Best objective found so far, against cumulative search effort. The flat "
               "tail is the search confirming it cannot do better, not the search stalling.")

    st.markdown("**Each restart on its own axes**")
    inner = alt.Chart(traj).encode(
        x=alt.X("step:Q", title="Step within restart", axis=alt.Axis(tickMinStep=1),
                scale=alt.Scale(padding=14)),
        y=alt.Y("cur_objective:Q", title="Objective", scale=alt.Scale(zero=False)))
    climb = inner.mark_line(strokeWidth=2, color=FOCUS)
    # mark_point, not mark_circle: only point marks honour the shape channel, so
    # the improved/stopped distinction survives greyscale and CVD.
    marks = inner.mark_point(size=85, filled=True, stroke="#ffffff", strokeWidth=1.5).encode(
        color=alt.Color("outcome:N",
                        scale=alt.Scale(domain=["improved", "local optimum"],
                                        range=[FOCUS, "#8a8983"]),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("outcome:N",
                        scale=alt.Scale(domain=["improved", "local optimum"],
                                        range=["circle", "triangle-down"]), legend=None),
        tooltip=[alt.Tooltip("restart:Q", title="Restart"),
                 alt.Tooltip("step:Q", title="Step"),
                 alt.Tooltip("cur_algorithm:N", title="Algorithm"),
                 alt.Tooltip("cur_k:Q", title="k"),
                 alt.Tooltip("cur_scaler:N", title="Scaler"),
                 alt.Tooltip("cur_feature_set:N", title="Features"),
                 alt.Tooltip("cur_log_transform:N", title="log1p"),
                 alt.Tooltip("cur_objective:Q", title="Objective", format=".4f"),
                 alt.Tooltip("cur_silhouette:Q", title="Silhouette", format=".4f")])
    target = alt.Chart(pd.DataFrame({"y": [best["objective"]]})).mark_rule(
        color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
    panels = ((climb + marks + target).properties(width=230, height=150)
              .facet(facet=alt.Facet("restart_label:N", title=None,
                                     header=alt.Header(labelColor=INK, labelFontSize=12,
                                                       labelFontWeight="bold")),
                     columns=3))
    st.altair_chart(styled(panels), width="content")
    st.caption("One panel per random restart — small multiples rather than six colours on "
               "one axis. The dashed line is the best objective found overall; a "
               "down-triangle is where that restart hit a local optimum and stopped. "
               "Restarts that plateau below the dashed line are exactly the failure mode "
               "random restarts exist to cover.")

    with st.expander("Full search history"):
        st.dataframe(traj, hide_index=True, width="stretch", height=300)

    st.divider()
    st.subheader("Objective definition")
    obj = search["objective"]
    st.latex(r"J = 0.5\cdot\frac{s+1}{2} \;+\; 0.25\cdot\frac{1}{1+DB} \;+\; "
             r"0.25\cdot\frac{CH}{CH+500}")
    st.markdown(
        f"""`s` = silhouette (Rousseeuw 1987), `DB` = Davies–Bouldin (1979), `CH` =
Calinski–Harabasz (1974). Each term is rescaled to roughly [0, 1] and is monotone in
the direction the index prefers, so the blend never rewards a worse index. Weights
{obj['weights']}; the constant {obj['ch_squash_constant']:.0f} squashes the unbounded
CH term.""")

    con = search["constraint"]
    st.warning(
        f"**Actionability constraint** — feasible solutions need k between "
        f"{con['min_k']} and {con['max_k']} and no segment below "
        f"{con['min_cluster_share']:.0%} of the base. {con['why']}")

    uref = search.get("unconstrained_reference")
    if uref:
        st.caption("Best achievable composite at each k below the feasible band, "
                   "searched over the same preprocessing/algorithm space — the number "
                   "behind the claim above:")
        udf = pd.DataFrame(uref)
        udf["pipeline"] = udf["config"].apply(
            lambda c: f"{c['feature_set']}, {c['scaler']}, {c['algorithm']}")
        udf["min_cluster_share"] = udf["min_cluster_share"] * 100.0
        st.dataframe(
            udf[["k", "best_composite", "silhouette", "pipeline", "min_cluster_share",
                "beats_constrained_winner"]],
            hide_index=True, width="stretch",
            column_config={
                "k": "k", "best_composite": st.column_config.NumberColumn(
                    "Best composite", format="%.4f"),
                "silhouette": st.column_config.NumberColumn("Silhouette", format="%.4f"),
                "pipeline": "Best pipeline",
                "min_cluster_share": st.column_config.NumberColumn(
                    "Smallest segment", format="%.1f%%"),
                "beats_constrained_winner": st.column_config.CheckboxColumn(
                    "Beats k=6 winner")})

    st.subheader("Number-of-clusters diagnostics")
    st.caption("Both curves are diagnostics; neither selects the model. Shown on separate "
               "axes on purpose — inertia and silhouette are not comparable scales.")
    elbow = pd.DataFrame(M["elbow"])
    e1, e2 = st.columns(2)
    for col, field, title, fmt in [(e1, "inertia", "Within-cluster sum of squares", ".2f"),
                                   (e2, "silhouette", "Silhouette", ".3f")]:
        line = alt.Chart(elbow).mark_line(strokeWidth=2, color=INK, point=False).encode(
            x=alt.X("k:O", title="k"), y=alt.Y(f"{field}:Q", title=title,
                                               scale=alt.Scale(zero=False)))
        dots = alt.Chart(elbow).mark_point(size=90, filled=True, stroke="#ffffff",
                                           strokeWidth=1.5).encode(
            x="k:O", y=f"{field}:Q",
            color=alt.Color("feasible:N",
                            scale=alt.Scale(domain=[True, False], range=[FOCUS, "#c9c8c3"]),
                            legend=alt.Legend(title="Within feasible k band", orient="top")),
            tooltip=[alt.Tooltip("k:O"), alt.Tooltip(f"{field}:Q", format=fmt),
                     alt.Tooltip("min_cluster_share:Q", title="Smallest segment", format=".1%"),
                     alt.Tooltip("feasible:N", title="Feasible")])
        col.altair_chart(styled((line + dots).properties(height=250)),
                         width="stretch")
    st.caption("The silhouette curve is why the constraint exists: it climbs again below "
               "k=4, where the extra 'segments' are outlier pockets rather than audiences.")

    if data["grid_results"] is not None:
        st.divider()
        st.subheader("Full-grid leaderboard")
        gr = data["grid_results"]
        only_feasible = st.checkbox("Feasible configurations only", value=True)
        view = gr[gr["feasible"]] if only_feasible else gr
        st.dataframe(view.sort_values("objective", ascending=False).head(25),
                     hide_index=True, width="stretch", height=320)

# ----------------------------------------------------------------------------- literature
with tabs[3]:
    st.subheader("What each number on this dashboard comes from")
    st.markdown(
        "Every metric the search optimises and every algorithm in the search space is a "
        "published method, cited below at its original source. These are the sources of "
        "the *methods*; the numbers on this dashboard are from this project's own run and "
        "are not comparable to results reported in any of these papers.")
    lit = pd.DataFrame(M["literature"])
    st.dataframe(lit, hide_index=True, width="stretch", height=420,
                 column_config={"element": st.column_config.TextColumn("Dashboard element",
                                                                       width="medium"),
                                "role": st.column_config.TextColumn("Role here", width="large"),
                                "source": st.column_config.TextColumn("Source", width="large")})
    st.divider()
    st.markdown(
        """**How the implementation follows the sources**

- *Rousseeuw (1987)* defines the silhouette per observation, so the dashboard reports
  both the mean (the search objective) and the per-segment mean, and flags segments
  whose members sit near a boundary. Per-customer values are in the segment tables.
- *Calinski & Harabasz (1974)* and *Davies & Bouldin (1979)* are unbounded and
  reverse-signed respectively, so both are rescaled before entering the blend rather
  than being compared raw.
- *Hughes (1994)* supplies both the RFM features and the quintile-tier baseline the
  clustering is measured against, so the clustering has to beat the cheap method to
  justify itself.
- *Russell & Norvig* describe steepest-ascent hill climbing's failure mode — local
  optima — so the implementation uses random restarts and this dashboard reports the
  optimality gap against an exhaustive grid instead of assuming the search succeeded.""")

# ----------------------------------------------------------------------------- data
with tabs[4]:
    st.subheader("CRISP-DM phase map")
    st.dataframe(pd.DataFrame(CRISP_DM, columns=["Phase", "What was done", "Where"]),
                 hide_index=True, width="stretch",
                 column_config={"Phase": st.column_config.TextColumn(width="small"),
                                "What was done": st.column_config.TextColumn(width="large"),
                                "Where": st.column_config.TextColumn(width="medium")})

    st.divider()
    st.subheader("Data preparation audit")
    d = M["data"]
    c = st.columns(4)
    c[0].metric("Raw transaction lines", f"{d['raw_lines']:,}")
    c[1].metric("Lines after cleaning", f"{d['clean_lines']:,}")
    c[2].metric("Customers", f"{d['customers']:,}")
    c[3].metric("Snapshot date", d["snapshot_date"])

    clean = data["cleaning"]
    clean = clean[clean["rows_removed"] > 0]
    clean_order = alt.EncodingSortField(field="rows_removed", op="max", order="descending")
    bar = alt.Chart(clean).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                                    height=20, color=FOCUS).encode(
        y=alt.Y("step:N", sort=clean_order, title=None,
                axis=alt.Axis(labelColor=INK, labelFontSize=12, labelLimit=320)),
        x=alt.X("rows_removed:Q", title="Transaction lines removed",
                scale=alt.Scale(domain=[0, float(clean["rows_removed"].max()) * 1.15],
                                nice=False)),
        tooltip=[alt.Tooltip("step:N", title="Step"),
                 alt.Tooltip("rows_removed:Q", title="Rows removed", format=",.0f")])
    txt = alt.Chart(clean).mark_text(align="left", dx=6, color=INK, fontSize=11).encode(
        y=alt.Y("step:N", sort=clean_order), x="rows_removed:Q",
        text=alt.Text("rows_removed:Q", format=",.0f"))
    st.altair_chart(styled((bar + txt).properties(height=170)), width="stretch")
    st.caption("Cancellation lines are kept, not dropped: they carry negative revenue and "
               "feed the per-customer return rate.")

    st.divider()
    st.subheader("Feature dictionary")
    st.dataframe(pd.DataFrame({"feature": list(FEATURE_DOCS),
                               "definition": list(FEATURE_DOCS.values())}),
                 hide_index=True, width="stretch",
                 column_config={"definition": st.column_config.TextColumn(width="large")})

    st.subheader("Feature distributions")
    feat_pick = st.selectbox("Feature", ["monetary", "frequency", "recency_days",
                                         "avg_order_value", "avg_basket_size",
                                         "tenure_days", "return_rate"])
    hist = alt.Chart(customers).mark_bar(color=FOCUS, cornerRadiusTopLeft=3,
                                         cornerRadiusTopRight=3).encode(
        x=alt.X(f"{feat_pick}:Q", bin=alt.Bin(maxbins=45), title=feat_pick),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")])
    st.altair_chart(styled(hist.properties(height=230)), width="stretch")
    st.dataframe(customers[[feat_pick]].describe().T.round(2), width="stretch")

    with st.expander("Customer feature table (all customers)"):
        st.dataframe(customers, hide_index=True, width="stretch", height=320)
        st.download_button("Download all customer features",
                           customers.to_csv(index=False).encode(),
                           file_name="customer_segments.csv", mime="text/csv")

# ----------------------------------------------------------------------------- model card
with tabs[5]:
    st.subheader("Model card")
    left, right = st.columns(2)
    with left:
        st.markdown("**Selected pipeline**")
        st.dataframe(pd.DataFrame({"setting": list(cfg), "value": [str(v) for v in cfg.values()]}),
                     hide_index=True, width="stretch")
    with right:
        st.markdown("**Internal indices for the selected pipeline**")
        st.dataframe(pd.DataFrame([
            {"index": "Silhouette", "value": f"{best['silhouette']:.4f}", "better": "higher"},
            {"index": "Davies–Bouldin", "value": f"{best['davies_bouldin']:.4f}",
             "better": "lower"},
            {"index": "Calinski–Harabasz", "value": f"{best['calinski_harabasz']:,.1f}",
             "better": "higher"},
            {"index": "Composite objective", "value": f"{best['objective']:.4f}",
             "better": "higher"},
            {"index": "Smallest segment share",
             "value": f"{best['min_cluster_share']:.1%}", "better": "≥ 3%"},
        ]), hide_index=True, width="stretch")
        st.markdown("**Reproducibility**")
        st.code(f"seed        {M['seed']}\npython      {M['python']}\n"
                f"runtime     {M['runtime_seconds']} s\n"
                f"pipelines   {search['distinct_pipelines_fitted']} fitted / "
                f"{search['space_size']} in space", language="text")

    st.markdown("**Held-out-free validation**")
    st.markdown(
        "Clustering has no labels, so there is no test split. Validity is established "
        "three ways instead:")
    stab = M["stability"]
    st.dataframe(pd.DataFrame([
        {"check": "Subsample stability (ARI)",
         "value": f"{stab['ari_mean']:.3f} ± {stab['ari_std']:.3f} "
                  f"(min {stab['ari_min']:.3f}, {stab['n_trials']} refits at "
                  f"{stab['subsample_frac']:.0%})",
         "reads as": "≈1.0 means the same customers land together when the data changes."},
        {"check": "Agreement with RFM quintile tiers (ARI)",
         "value": f"{M['baseline_agreement']['adjusted_rand_index_vs_rfm_quintiles']:.3f}",
         "reads as": "Related to the cheap baseline but not a restatement of it."},
        {"check": "External archetype recovery (ARI / NMI)",
         "value": (f"{M['external_check']['adjusted_rand_index']:.3f} / "
                   f"{M['external_check']['normalized_mutual_info']:.3f}")
                  if M.get("external_check") else "n/a",
         "reads as": "Sanity check only — see the caveat below."},
    ]), hide_index=True, width="stretch",
        column_config={"reads as": st.column_config.TextColumn(width="large")})

    if M.get("external_check"):
        st.info(f"**External check caveat** — {M['external_check']['note']}")
    if data["archetype_crosstab"] is not None:
        with st.expander("Latent archetype × discovered segment crosstab"):
            st.dataframe(data["archetype_crosstab"], hide_index=True, width="stretch")

    st.divider()
    st.subheader("Score a customer")
    st.caption("Assigns a customer to the nearest segment centroid in the fitted feature "
               "space — the same transform the training run used. Works for any algorithm "
               "in the search space, including ones without a native `predict`.")
    bundle = load_pipeline()
    if bundle is None:
        st.warning("`artifacts/best_pipeline.joblib` is missing — rerun `python src/train.py`.")
    else:
        mode = st.radio("Input", ["Look up an existing customer", "Enter values manually"],
                        horizontal=True, label_visibility="collapsed")
        cols = bundle["feature_columns"]
        if mode == "Look up an existing customer":
            cid = st.selectbox("Customer ID", customers["CustomerID"].astype(int).tolist())
            src = customers[customers["CustomerID"].astype(int) == cid].iloc[0]
            vals = [float(src[c]) for c in cols]
            st.dataframe(pd.DataFrame({"feature": cols, "value": vals}),
                         hide_index=True, width="stretch")
        else:
            fields = st.columns(len(cols))
            vals = [float(fields[i].number_input(
                c, min_value=0.0, value=float(customers[c].median()), step=1.0))
                for i, c in enumerate(cols)]

        if st.button("Assign segment", type="primary"):
            x = np.array(vals, dtype=float).reshape(1, -1)
            if bundle["log_transform"]:
                x = np.log1p(np.clip(x, 0.0, None))
            xs = bundle["scaler"].transform(x)
            d = np.linalg.norm(bundle["centroids"] - xs, axis=1)
            order = np.argsort(d)
            winner = bundle["centroid_labels"][int(order[0])]
            name, action = bundle["segment_names"][winner]
            runner = bundle["segment_names"][bundle["centroid_labels"][int(order[1])]][0]
            st.success(f"### {name}\n{action}")
            st.caption(f"Nearest centroid distance {d[order[0]]:.3f}; "
                       f"next-closest segment is **{runner}** at {d[order[1]]:.3f}. "
                       "Similar distances mean a borderline customer.")
            st.dataframe(pd.DataFrame({
                "segment": [bundle["segment_names"][c][0] for c in bundle["centroid_labels"]],
                "distance": np.round(d, 4)}).sort_values("distance"),
                hide_index=True, width="stretch")

    st.divider()
    st.subheader("Known limitations")
    st.markdown(
        """- **The data is synthetic.** It reproduces the schema and defect profile of a
  public retail transaction log, not its real customers. Every number here describes
  this generated dataset; none of it is a claim about real retail behaviour.
- **Clustering has no ground truth.** The indices measure geometry, not business value.
  A high silhouette means the segments are well separated in RFM space — it does not
  prove a campaign targeted at them will convert.
- **The archetype recovery score is not a benchmark.** The generator's archetypes
  overlap heavily in RFM space, so partial agreement is the expected outcome; it was
  never optimised for, and it exists only to confirm the clustering is not noise.
- **The feasible-k band is a business assumption**, declared before the search. Change
  `MIN_K` / `MAX_K` in `src/autoresearch.py` and the winning pipeline may change.
- **Segment names come from a template match**, not from the algorithm. They are a
  readability aid over the RFM profile, and a small dataset shift can rename a segment
  without the underlying cluster changing much.""")
