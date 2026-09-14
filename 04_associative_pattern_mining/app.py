"""Market Basket Pattern Mining — data science admin dashboard.

Streamlit front end over the CRISP-DM experiment in ``src/``. Reads the
artifacts written by ``python -m src.run_experiment`` and also lets an analyst
re-mine interactively with their own thresholds.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.mining import (
    MiningConfig,
    clean_raw,
    evaluate,
    fmt_items,
    load_raw,
    mine_rules,
    objective,
    to_basket_matrix,
)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="Market Basket Pattern Mining", page_icon="🧺", layout="wide")


# --------------------------------------------------------------------------
# Cached data access
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading and cleaning transactions ...")
def load_prepared():
    raw = load_raw()
    clean, quality = clean_raw(raw)
    matrix = to_basket_matrix(clean)
    return raw, clean, quality, matrix


@st.cache_data(show_spinner="Mining rules ...")
def mine_cached(_matrix: pd.DataFrame, config_dict: dict):
    config = MiningConfig(**config_dict)
    itemsets, rules = mine_rules(_matrix, config)
    metrics = evaluate(_matrix, itemsets, rules)
    metrics["score"] = objective(metrics)
    return itemsets, rules, metrics


def load_artifact(name: str):
    path = ARTIFACTS / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return pd.read_csv(path)


try:
    raw, clean, quality, matrix = load_prepared()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

experiment = load_artifact("metrics.json")
history = load_artifact("autoresearch_history.csv")


# --------------------------------------------------------------------------
# Sidebar: live mining controls
# --------------------------------------------------------------------------

st.sidebar.title("🧺 Mining controls")

default_cfg = (experiment or {}).get("best_config") or MiningConfig().as_dict()
if st.sidebar.button("Reset to autoresearch best", width="stretch"):
    st.session_state.clear()
    st.rerun()

algorithm = st.sidebar.selectbox(
    "Algorithm", ["fpgrowth", "apriori"],
    index=["fpgrowth", "apriori"].index(default_cfg["algorithm"]),
    help="Both are exact: they find the same frequent itemsets. FP-Growth avoids "
         "candidate generation and is usually faster on dense data.",
)
min_support = st.sidebar.slider("Minimum support", 0.005, 0.10, float(default_cfg["min_support"]), 0.005,
                                help="Fraction of baskets an itemset must appear in.")
min_confidence = st.sidebar.slider("Minimum confidence", 0.05, 0.95, float(default_cfg["min_confidence"]), 0.05,
                                   help="P(consequent | antecedent).")
min_lift = st.sidebar.slider("Minimum lift", 1.0, 5.0, float(default_cfg["min_lift"]), 0.05,
                             help="Lift = 1 means independence; > 1 means positive association.")
max_len = st.sidebar.slider("Max itemset size", 2, 4, int(default_cfg["max_len"]))

config = MiningConfig(algorithm, min_support, min_confidence, min_lift, max_len)
itemsets, rules, metrics = mine_cached(matrix, config.as_dict())

st.sidebar.divider()
st.sidebar.caption(
    f"**{quality['baskets_out']:,}** baskets · **{quality['distinct_items_out']}** items · "
    f"avg basket **{quality['avg_basket_size']}**"
)


# --------------------------------------------------------------------------
# Header KPIs
# --------------------------------------------------------------------------

st.title("Market Basket Pattern Mining")
st.caption(
    "Association-rule mining on grocery transactions, following CRISP-DM, with "
    "hill-climbing autoresearch over the mining hyperparameters."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Baskets", f"{quality['baskets_out']:,}")
k2.metric("Frequent itemsets", f"{metrics['n_itemsets']:,}")
k3.metric("Rules", f"{metrics['n_rules']:,}")
k4.metric("Mean lift", f"{metrics['mean_lift']:.2f}")
k5.metric("Basket coverage", f"{metrics['coverage']:.1%}")

if metrics["n_rules"] == 0:
    st.warning("No rules at these thresholds. Lower minimum support or confidence.")

tabs = st.tabs(
    ["📋 CRISP-DM", "🔍 Data", "🧠 Rules", "🧪 Autoresearch", "✅ Evaluation", "🛒 Recommender"]
)


# --------------------------------------------------------------------------
# Tab 1 — CRISP-DM
# --------------------------------------------------------------------------

with tabs[0]:
    st.subheader("CRISP-DM phase map")
    phases = pd.DataFrame(
        [
            ("1. Business Understanding",
             "Which products are bought together, strongly enough to drive placement, bundling and "
             "next-item recommendations?",
             "README.md"),
            ("2. Data Understanding",
             "Item-level transaction log; profile item frequency, basket sizes and injected data defects.",
             "Data tab · artifacts/data_quality.json"),
            ("3. Data Preparation",
             "Normalise item text, drop blank/duplicate rows, drop single-item baskets, pivot to a "
             "one-hot basket × item matrix.",
             "src/mining.py — clean_raw / to_basket_matrix"),
            ("4. Modeling",
             "Frequent-itemset mining (Apriori / FP-Growth) → rules; hyperparameters chosen by "
             "hill-climbing autoresearch rather than by hand.",
             "src/mining.py · src/autoresearch.py"),
            ("5. Evaluation",
             "Support/confidence/lift/leverage/conviction/Zhang's metric, basket coverage, and a "
             "70/30 held-out rule-stability check.",
             "Evaluation tab · artifacts/metrics.json"),
            ("6. Deployment",
             "This dashboard: live re-mining, a rule explorer, and a basket recommender that serves "
             "the mined rules.",
             "app.py"),
        ],
        columns=["Phase", "What was done here", "Where"],
    )
    st.dataframe(phases, hide_index=True, width="stretch")

    st.subheader("Interestingness measures reported")
    st.markdown(
        """
| Measure | Definition | Reads as | Source |
|---|---|---|---|
| Support | P(A ∪ B) | How often the rule applies at all | Agrawal, Imielinski & Swami (1993) |
| Confidence | P(B \\| A) | Reliability given the antecedent | Agrawal, Imielinski & Swami (1993) |
| Lift | P(B \\| A) / P(B) | 1 = independent, > 1 = positive association | Brin, Motwani, Ullman & Tsur (1997) |
| Leverage | P(A ∪ B) − P(A)P(B) | Extra co-occurrence in absolute basket share | Piatetsky-Shapiro (1991) |
| Conviction | (1 − P(B)) / (1 − conf) | Sensitivity to the rule being wrong; ∞ = never violated | Brin et al. (1997) |
| Zhang's metric | ∈ [−1, 1] | Direction + strength; negative = substitution | Zhang (2000) |

Algorithms: **Apriori** — Agrawal & Srikant, VLDB 1994 (downward-closure pruning).
**FP-Growth** — Han, Pei & Yin, SIGMOD 2000 (prefix-tree, no candidate generation).
Both are exact, so they return identical itemsets; they differ only in runtime.
        """
    )


# --------------------------------------------------------------------------
# Tab 2 — Data
# --------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Data quality — what preparation removed")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Raw rows", f"{quality['rows_in']:,}")
    q2.metric("Blank/missing item rows", f"{quality['rows_dropped_missing_item']:,}")
    q3.metric("Duplicate rows", f"{quality['rows_dropped_duplicate']:,}")
    q4.metric("Single-item baskets", f"{quality['baskets_dropped_singleton']:,}")
    st.caption(
        f"Distinct item labels fell from **{quality['distinct_items_in']}** to "
        f"**{quality['distinct_items_out']}** once casing and whitespace were normalised — the "
        "difference was duplicate spellings of the same product."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Top items by support")
        freq = (
            clean["itemDescription"].value_counts().head(20).rename_axis("item").reset_index(name="baskets")
        )
        freq["support"] = freq["baskets"] / quality["baskets_out"]
        st.altair_chart(
            alt.Chart(freq).mark_bar().encode(
                x=alt.X("support:Q", axis=alt.Axis(format="%"), title="Support"),
                y=alt.Y("item:N", sort="-x", title=None),
                tooltip=["item", "baskets", alt.Tooltip("support:Q", format=".2%")],
            ).properties(height=420),
            width="stretch",
        )
    with right:
        st.subheader("Basket size distribution")
        sizes = clean.groupby("basket_id").size().rename("items").reset_index()
        st.altair_chart(
            alt.Chart(sizes).mark_bar().encode(
                x=alt.X("items:Q", bin=alt.Bin(maxbins=20), title="Items per basket"),
                y=alt.Y("count():Q", title="Baskets"),
                tooltip=["count():Q"],
            ).properties(height=200),
            width="stretch",
        )
        st.metric("Median basket size", int(sizes["items"].median()))
        st.metric("Largest basket", int(sizes["items"].max()))
        st.subheader("Raw sample")
        st.dataframe(raw.head(8), hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# Tab 3 — Rules
# --------------------------------------------------------------------------

with tabs[2]:
    if metrics["n_rules"] == 0:
        st.info("No rules to explore at the current thresholds.")
    else:
        st.subheader("Rule explorer")
        display = rules.copy()
        display["antecedents"] = display["antecedents"].apply(fmt_items)
        display["consequents"] = display["consequents"].apply(fmt_items)

        item_filter = st.multiselect(
            "Show only rules involving these items", sorted(matrix.columns), default=[]
        )
        if item_filter:
            wanted = set(item_filter)
            mask = display.apply(
                lambda r: bool(wanted & set(r["rule"].replace(" → ", ", ").split(", "))), axis=1
            )
            display = display[mask]

        st.caption(f"{len(display):,} rules shown, sorted by lift.")
        st.dataframe(
            display[
                ["antecedents", "consequents", "support", "confidence", "lift",
                 "leverage", "conviction", "zhangs_metric"]
            ].style.format(
                {
                    "support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.2f}",
                    "leverage": "{:.4f}", "conviction": "{:.2f}", "zhangs_metric": "{:.3f}",
                }
            ),
            hide_index=True,
            width="stretch",
            height=380,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Support vs confidence (colour = lift)")
            st.altair_chart(
                alt.Chart(display).mark_circle(size=70, opacity=0.7).encode(
                    x=alt.X("support:Q", title="Support"),
                    y=alt.Y("confidence:Q", title="Confidence"),
                    color=alt.Color("lift:Q", scale=alt.Scale(scheme="viridis"), title="Lift"),
                    tooltip=["rule", alt.Tooltip("support:Q", format=".3f"),
                             alt.Tooltip("confidence:Q", format=".3f"),
                             alt.Tooltip("lift:Q", format=".2f")],
                ).properties(height=340).interactive(),
                width="stretch",
            )
        with c2:
            st.subheader("Top 15 rules by lift")
            top = display.head(15)
            st.altair_chart(
                alt.Chart(top).mark_bar().encode(
                    x=alt.X("lift:Q", title="Lift"),
                    y=alt.Y("rule:N", sort="-x", title=None),
                    tooltip=["rule", alt.Tooltip("lift:Q", format=".2f"),
                             alt.Tooltip("confidence:Q", format=".3f")],
                ).properties(height=340),
                width="stretch",
            )

        st.download_button(
            "Download these rules (CSV)",
            display.to_csv(index=False).encode(),
            file_name="rules.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# Tab 4 — Autoresearch
# --------------------------------------------------------------------------

with tabs[3]:
    st.subheader("Hill-climbing hyperparameter search")
    st.markdown(
        "Steepest-ascent hill climbing with random restarts over "
        "`{algorithm, min_support, min_confidence, min_lift, max_len}`. Each step evaluates every "
        "configuration one grid move away and takes the best strict improvement; when none improves, "
        "the restart has hit a local optimum. The objective balances **mean lift** against **basket "
        "coverage**, penalising rule sets that are too small or too large to act on "
        "(`src/mining.py — objective`). The weighting is our stated modelling choice, not a result "
        "taken from any paper."
    )

    if history is None or experiment is None:
        st.warning("Run `python -m src.run_experiment` to populate the search trace.")
    else:
        a, b, c = st.columns(3)
        a.metric("Configurations evaluated", experiment["configs_evaluated"])
        b.metric("Baseline score", f"{experiment['baseline_metrics']['score']:.4f}")
        c.metric(
            "Best score", f"{experiment['best_metrics']['score']:.4f}",
            delta=f"{experiment['best_metrics']['score'] - experiment['baseline_metrics']['score']:+.4f}",
        )

        st.altair_chart(
            alt.layer(
                alt.Chart(history).mark_circle(size=45, opacity=0.45).encode(
                    x=alt.X("trial:Q", title="Trial"),
                    y=alt.Y("score:Q", title="Objective score"),
                    color=alt.Color("restart:N", title="Restart"),
                    tooltip=["trial", "restart", "kind", "algorithm", "min_support",
                             "min_confidence", "min_lift", "max_len",
                             alt.Tooltip("score:Q", format=".4f"), "n_rules"],
                ),
                alt.Chart(history).mark_line(color="#d62728", strokeWidth=2).encode(
                    x="trial:Q", y=alt.Y("best_so_far:Q", title="Objective score"),
                ),
            ).properties(height=320).interactive(),
            width="stretch",
        )
        st.caption("Red line = best score found so far. Each colour is one restart.")

        st.subheader("Winning configuration vs baseline")
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        cmp = pd.DataFrame(
            {
                "Baseline (hand-picked)": {
                    **{k: str(v) for k, v in experiment["baseline_config"].items()},
                    **{k: _fmt(v) for k, v in experiment["baseline_metrics"].items()},
                },
                "Autoresearch best": {
                    **{k: str(v) for k, v in experiment["best_config"].items()},
                    **{k: _fmt(v) for k, v in experiment["best_metrics"].items()},
                },
            }
        )
        st.dataframe(cmp, width="stretch")

        with st.expander("Full search trace"):
            st.dataframe(history, hide_index=True, width="stretch", height=300)


# --------------------------------------------------------------------------
# Tab 5 — Evaluation
# --------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Held-out rule stability")
    if experiment is None:
        st.warning("Run `python -m src.run_experiment` to populate evaluation results.")
    else:
        stab = experiment["holdout_stability"]
        st.markdown(
            "Association rules are descriptive, so they are often reported without any held-out "
            "check. Here rules are mined on a random **70%** of baskets and then re-scored on the "
            "unseen **30%**; a rule *holds* if its lift on the held-out baskets still clears the "
            "mining threshold."
        )
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Rules mined on train", f"{stab['rules_mined_on_train']:,}")
        e2.metric("Still holding on test", f"{stab['rules_holding_on_test']:,}",
                  delta=f"{stab['pct_rules_holding']}%")
        e3.metric("Mean lift (train)", f"{stab['mean_lift_train']:.2f}")
        e4.metric("Mean lift (held out)", f"{stab['mean_lift_holdout']:.2f}")
        if stab["lift_correlation"] is not None:
            st.metric("Train vs held-out lift correlation", f"{stab['lift_correlation']:.3f}")

        holdout_rules = load_artifact("holdout_rules.csv")
        if holdout_rules is not None:
            st.altair_chart(
                alt.Chart(holdout_rules).mark_circle(size=60, opacity=0.6).encode(
                    x=alt.X("lift:Q", title="Lift on training baskets"),
                    y=alt.Y("holdout_lift:Q", title="Lift on held-out baskets"),
                    tooltip=["antecedents", "consequents",
                             alt.Tooltip("lift:Q", format=".2f"),
                             alt.Tooltip("holdout_lift:Q", format=".2f")],
                ).properties(height=340).interactive(),
                width="stretch",
            )
            st.caption(
                "Points on the diagonal are stable rules; points far below it were overfitted to "
                "the training baskets."
            )

        recovery = experiment.get("ground_truth_recovery")
        if recovery:
            st.subheader("Ground-truth recovery")
            st.markdown(
                "Because this dataset is synthetic, the co-purchase bundles it contains are *known*. "
                "That makes a check possible that real transaction data never allows: did mining "
                "actually find the patterns that were planted?"
            )
            r1, r2 = st.columns(2)
            r1.metric(
                "Bundles fully recovered",
                f"{recovery['bundles_fully_recovered']}/{recovery['planted_bundles']}",
            )
            r2.metric(
                "Item pairs recovered",
                f"{recovery['pairs_recovered']}/{recovery['pairs_planted']}",
                delta=f"{recovery['pct_pairs_recovered']}%",
            )
            st.dataframe(
                pd.DataFrame(recovery["per_bundle"]), hide_index=True, width="stretch"
            )
            st.caption(
                "This measures the miner against the generator, not against real shopper behaviour."
            )

        st.subheader("Current rule-set summary")
        st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})


# --------------------------------------------------------------------------
# Tab 6 — Recommender
# --------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Next-item recommender")
    st.write("Pick what is in the basket; the mined rules suggest what to recommend next.")

    basket = st.multiselect("Items in basket", sorted(matrix.columns), default=[])

    if not basket:
        st.info("Select at least one item.")
    elif metrics["n_rules"] == 0:
        st.warning("No rules available at the current thresholds.")
    else:
        selected = set(basket)
        matched = rules[
            rules["antecedents"].apply(lambda a: set(a).issubset(selected))
            & rules["consequents"].apply(lambda c: not (set(c) & selected))
        ]
        if matched.empty:
            st.warning("No rule fires for this basket. Try lowering minimum support or confidence.")
        else:
            recs = (
                matched.assign(recommendation=matched["consequents"].apply(fmt_items))
                .sort_values(["lift", "confidence"], ascending=False)
                .drop_duplicates("recommendation")
                .head(10)
            )
            for _, row in recs.iterrows():
                c1, c2 = st.columns([3, 2])
                c1.markdown(f"**{row['recommendation']}**")
                c1.caption(f"because of: {fmt_items(row['antecedents'])}")
                c2.markdown(
                    f"lift **{row['lift']:.2f}** · confidence **{row['confidence']:.0%}** · "
                    f"support **{row['support']:.3f}**"
                )
                st.divider()
