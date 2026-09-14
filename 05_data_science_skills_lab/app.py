"""Data Science Skills Mastery Lab — Streamlit front end.

Four views:
  * Skills Lab      — browse all 46 installed skills and execute any one live
  * CRISP-DM        — the six phases, each with the skills that serve it
  * Live inference  — score a customer against the same fitted pipeline
  * Batch run       — execute the whole catalog and inspect the run log
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from skills.base import SkillResult  # noqa: E402
from skills.ml_skills import rag_pipeline  # noqa: E402
from skills.registry import (CRISP_PHASES, DA_PACK, ML_PACK, SKILLS,  # noqa: E402
                             by_pack, by_phase, categories)
from skills.render import render_result  # noqa: E402
from src.data_prep import get_data  # noqa: E402
from src.model import get_model  # noqa: E402

st.set_page_config(page_title="Data Science Skills Mastery Lab",
                   page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner="Loading data and fitting the baseline model ...")
def context() -> dict:
    bundle = get_data()
    ctx = dict(bundle)
    ctx["model"] = get_model(bundle["df"])
    return ctx


ctx = context()
df, mb = ctx["df"], ctx["model"]

st.title("🧪 Data Science Skills Mastery Lab")
st.caption(
    f"{len(SKILLS)} skills from two installed packs, each demonstrated on the "
    f"Telco Customer Churn dataset ({ctx['rows_clean']:,} customers) and mapped "
    f"to a CRISP-DM phase."
)

top = st.columns(5)
top[0].metric("Skills installed", len(SKILLS))
top[1].metric("Customers", f"{ctx['rows_clean']:,}")
top[2].metric("Churn rate", f"{df['churn_flag'].mean():.1%}")
top[3].metric("Model ROC-AUC", f"{mb['metrics']['roc_auc']:.3f}", "held-out split")
top[4].metric("Data source", "real" if ctx["source"].startswith("real") else "synthetic")
st.caption(f"Dataset provenance: {ctx['source']}")

tab_lab, tab_crisp, tab_infer, tab_batch = st.tabs(
    ["🎛️ Skills Lab", "🔄 CRISP-DM", "⚡ Live inference", "📋 Batch run"])

# --------------------------------------------------------------------------
# Skills Lab
# --------------------------------------------------------------------------
with tab_lab:
    left, right = st.columns([1, 3], gap="large")

    with left:
        st.subheader("Catalog")
        pack = st.radio("Installed pack", [ML_PACK, DA_PACK, "both"], index=0,
                        format_func=lambda p: {"both": "both packs"}.get(p, p))
        pool = SKILLS if pack == "both" else by_pack(pack)
        cats = ["all categories"] + (
            categories(pack) if pack != "both"
            else categories(ML_PACK) + categories(DA_PACK))
        cat = st.selectbox("Category", cats)
        if cat != "all categories":
            pool = [s for s in pool if s.category == cat]
        phase = st.selectbox("CRISP-DM phase", ["all phases"] + CRISP_PHASES)
        if phase != "all phases":
            pool = [s for s in pool if s.crisp_dm == phase]

        st.caption(f"{len(pool)} skill(s) match")
        if not pool:
            st.stop()
        chosen_id = st.radio("Skill", [s.id for s in pool],
                             format_func=lambda i: next(s.name for s in pool if s.id == i))
        skill = next(s for s in pool if s.id == chosen_id)

    with right:
        st.subheader(skill.name)
        meta = st.columns(3)
        meta[0].caption(f"**Pack** · {skill.pack}")
        meta[1].caption(f"**Category** · {skill.category}")
        meta[2].caption(f"**CRISP-DM** · {skill.crisp_dm}")
        st.write(skill.summary)

        query = None
        if skill.id == "rag-pipeline":
            query = st.text_input("Retrieval query",
                                  "which customers churn the most?")

        if st.button("▶ Execute skill live", type="primary", key=f"run-{skill.id}"):
            t0 = time.perf_counter()
            with st.spinner(f"Executing {skill.id} ..."):
                try:
                    result = (rag_pipeline(ctx, query) if skill.id == "rag-pipeline"
                              else skill.run(ctx))
                    st.session_state[f"res-{skill.id}"] = (result, time.perf_counter() - t0)
                except Exception as exc:
                    st.error(f"Skill failed: {type(exc).__name__}: {exc}")
                    st.session_state.pop(f"res-{skill.id}", None)

        stored = st.session_state.get(f"res-{skill.id}")
        if stored:
            result, elapsed = stored
            st.caption(f"Executed in {elapsed:.2f}s · "
                       f"{len(result.blocks)} output panels")
            render_result(result, key_prefix=skill.id)
        else:
            st.info("Press **Execute skill live** to run this skill against the "
                    "dataset and render its output as a dashboard.")

# --------------------------------------------------------------------------
# CRISP-DM
# --------------------------------------------------------------------------
with tab_crisp:
    st.subheader("CRISP-DM phases and the skills that serve them")
    st.write(
        "Every skill in the catalog is mapped to the phase where it does its "
        "work. The counts below are computed from the registry, so the map "
        "cannot drift out of sync with the code."
    )
    counts = pd.DataFrame([{"phase": p, "skills": len(by_phase(p))}
                           for p in CRISP_PHASES])
    st.bar_chart(counts.set_index("phase"), horizontal=True, height=260)

    narratives = {
        CRISP_PHASES[0]: (
            "Frame the decision before touching a model: which customers get a "
            "retention offer, what action follows, and how success is measured. "
            "The metric layer and assumptions register are written here."),
        CRISP_PHASES[1]: (
            f"Profile the {ctx['rows_raw']:,} raw rows — quality dimensions, "
            f"per-column statistics, cohorts, funnels and a contribution "
            f"decomposition of the {df['churn_flag'].mean():.1%} churn rate."),
        CRISP_PHASES[2]: (
            "Repair types, impute the 11 blank `TotalCharges` values that belong "
            "to never-billed accounts, and derive seven features. Every decision "
            "is logged with the row count it affected."),
        CRISP_PHASES[3]: (
            "A leakage-safe ColumnTransformer pipeline into logistic regression, "
            "plus grid search, class-imbalance handling, k-means segmentation and "
            "a hand-written gradient-descent loop."),
        CRISP_PHASES[4]: (
            f"Held-out ROC-AUC {mb['metrics']['roc_auc']:.3f}, PR-AUC "
            f"{mb['metrics']['pr_auc']:.3f}, calibration by decile, a leakage scan "
            f"with a positive control, a 10-point QA checklist and a peer review "
            f"of this project's own weaknesses."),
        CRISP_PHASES[5]: (
            "Live scoring through the same fitted pipeline, a measured serving "
            "contract, the executive summary, the dashboard spec and the handoff "
            "package."),
    }
    for phase in CRISP_PHASES:
        with st.expander(f"{phase} — {len(by_phase(phase))} skills", expanded=False):
            st.write(narratives[phase])
            st.dataframe(
                pd.DataFrame([{"skill": s.name, "id": s.id, "pack": s.pack,
                               "what it does here": s.summary}
                              for s in by_phase(phase)]),
                width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Live inference
# --------------------------------------------------------------------------
with tab_infer:
    st.subheader("Score a customer")
    st.write("This calls the same fitted pipeline object the metrics above come "
             "from — training-time and serving-time preprocessing cannot drift apart.")

    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 5)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 85.0, step=0.5)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with c2:
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    with c3:
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        addons = st.slider("Add-on services (of 6)", 0, 6, 0)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.30, step=0.05,
                          help="0.30 maximizes F1 on the held-out split; "
                               "0.50 is the untuned default.")

    template = df.iloc[0].copy()
    record = {c: template[c] for c in mb["numeric"] + mb["categorical"]}
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]
    for i, col in enumerate(addon_cols):
        record[col] = "Yes" if i < addons else ("No internet service"
                                                if internet == "No" else "No")
    record.update({
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
        "avg_monthly_spend": monthly, "tenure_years": round(tenure / 12, 2),
        "SeniorCitizen": senior, "Partner": partner, "Contract": contract,
        "InternetService": internet, "PaymentMethod": payment,
        "PaperlessBilling": paperless, "addon_count": addons,
        "has_protection": int(record["OnlineSecurity"] == "Yes"
                              or record["TechSupport"] == "Yes"),
        "is_autopay": int("automatic" in payment),
        "tenure_bucket": ("0-6m" if tenure <= 6 else "7-12m" if tenure <= 12
                          else "13-24m" if tenure <= 24 else "25-48m" if tenure <= 48
                          else "49-72m"),
        "spend_tier": ("Q1 low" if monthly <= df["MonthlyCharges"].quantile(.25)
                       else "Q2" if monthly <= df["MonthlyCharges"].quantile(.5)
                       else "Q3" if monthly <= df["MonthlyCharges"].quantile(.75)
                       else "Q4 high"),
    })
    X = pd.DataFrame([record])[mb["numeric"] + mb["categorical"]]

    t0 = time.perf_counter()
    prob = float(mb["pipeline"].predict_proba(X)[0, 1])
    latency = (time.perf_counter() - t0) * 1000

    r1, r2, r3 = st.columns(3)
    r1.metric("Churn probability", f"{prob:.1%}")
    r2.metric("Decision", "FLAG for retention" if prob >= threshold else "no action",
              f"threshold {threshold:.2f}")
    r3.metric("Scoring latency", f"{latency:.1f} ms")
    st.progress(min(prob, 1.0))

    peers = df[(df["Contract"] == contract) & (df["tenure_bucket"] == record["tenure_bucket"])]
    if len(peers) >= 30:
        st.caption(
            f"Observed churn rate among the {len(peers):,} customers in the same "
            f"contract x tenure group: **{peers['churn_flag'].mean():.1%}** "
            f"(base rate {df['churn_flag'].mean():.1%})."
        )
    st.caption(
        "A single score should rank a retention list, not decide a case on its "
        f"own: at threshold 0.50 the model's precision is "
        f"{mb['metrics']['precision']:.0%}."
    )

# --------------------------------------------------------------------------
# Batch run
# --------------------------------------------------------------------------
with tab_batch:
    st.subheader("Execute the whole catalog")
    st.write("Runs all 46 skills in one pass and reports status and timing for "
             "each — the same thing `python src/run_all.py` does from the CLI.")

    if st.button("▶ Run all skills", type="primary"):
        from src.run_all import run_skills, write_report
        progress = st.progress(0.0, text="starting ...")
        records = []
        for i, skill in enumerate(sorted(SKILLS, key=lambda s: s.id == "analysis-retrospective")):
            progress.progress((i + 1) / len(SKILLS), text=f"running {skill.id} ...")
            records += run_skills([skill], ctx, verbose=False)
        md_path, json_path = write_report(records, ctx)
        progress.empty()
        st.session_state["batch"] = (records, str(md_path), str(json_path))

    if "batch" in st.session_state:
        records, md_path, json_path = st.session_state["batch"]
        ok = sum(r["status"] == "ok" for r in records)
        m = st.columns(4)
        m[0].metric("Skills run", len(records))
        m[1].metric("Succeeded", ok)
        m[2].metric("Failed", len(records) - ok)
        m[3].metric("Total time", f"{sum(r['seconds'] for r in records):.1f}s")

        log = pd.DataFrame([{k: r[k] for k in
                             ("id", "pack", "crisp_dm", "status", "seconds", "headline")}
                            for r in records])
        st.dataframe(log, width="stretch", hide_index=True, height=420)
        st.caption(f"Report written to `{Path(md_path).name}` and "
                   f"`{Path(json_path).name}` under `reports/`.")
        st.download_button("Download run report (Markdown)",
                           Path(md_path).read_text(),
                           file_name="skill_run_report.md", mime="text/markdown")
    else:
        st.info("No batch run in this session yet.")
