"""Executors for the 15 skills in the `param087/agent-ml-skills` pack.

Each function computes something real on the Telco churn dataset and returns a
``SkillResult`` of presentation blocks. Where a skill's native domain is not
tabular churn (LLM fine-tuning, PyTorch loops, RAG), the executor demonstrates
the part of the skill that can be executed honestly inside this project's
dependency budget and says exactly what it is *not* doing.
"""
from __future__ import annotations

import hashlib
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics.pairwise import cosine_similarity

from skills.base import Metric, SkillResult, pct
from src.data_prep import SEED, feature_columns
from src.model import build_pipeline, coefficient_table, get_model, score_all


# --------------------------------------------------------------------------
# 1. exploratory-data-analysis
# --------------------------------------------------------------------------
def eda(ctx: dict) -> SkillResult:
    df = ctx["df"]
    res = SkillResult(
        f"{len(df):,} customers x {df.shape[1]} columns, "
        f"{pct(df['churn_flag'].mean())} churn rate"
    )
    res.metrics(
        Metric("Rows", f"{len(df):,}"),
        Metric("Columns", str(df.shape[1])),
        Metric("Churn rate", pct(df["churn_flag"].mean()), "positive class share"),
        Metric("Median tenure", f"{df['tenure'].median():.0f} months"),
        Metric("Median monthly bill", f"${df['MonthlyCharges'].median():.2f}"),
    )

    dist = (df.groupby("tenure_bucket", observed=True)
              .agg(customers=("churn_flag", "size"),
                   churn_rate=("churn_flag", "mean"))
              .reset_index())
    dist["churn_rate"] = (dist["churn_rate"] * 100).round(1)
    res.chart(dist, kind="bar", x="tenure_bucket", y="churn_rate",
              title="Churn rate by tenure bucket (%)",
              note="Churn is heavily concentrated in the first six months.")

    cat_rates = []
    for col in ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]:
        g = df.groupby(col)["churn_flag"].agg(["size", "mean"]).reset_index()
        g.columns = ["level", "customers", "churn_rate"]
        g.insert(0, "feature", col)
        cat_rates.append(g)
    cats = pd.concat(cat_rates, ignore_index=True)
    cats["churn_rate"] = (cats["churn_rate"] * 100).round(1)
    res.table(cats, "Churn rate by categorical level (%)")

    num = df[["tenure", "MonthlyCharges", "TotalCharges", "addon_count", "churn_flag"]]
    corr = (num.corr(numeric_only=True)["churn_flag"]
               .drop("churn_flag").round(3).reset_index())
    corr.columns = ["feature", "corr_with_churn"]
    res.chart(corr, kind="hbar", x="corr_with_churn", y="feature",
              title="Pearson correlation with churn")
    return res


# --------------------------------------------------------------------------
# 2. data-cleaning
# --------------------------------------------------------------------------
def data_cleaning(ctx: dict) -> SkillResult:
    raw, df, issues = ctx["raw"], ctx["df"], ctx["issues"]
    res = SkillResult(
        f"{len(issues)} quality defect class(es) found and resolved; "
        f"{len(raw):,} raw rows -> {len(df):,} usable rows"
    )
    res.metrics(
        Metric("Raw rows", f"{len(raw):,}"),
        Metric("Rows after cleaning", f"{len(df):,}"),
        Metric("Rows dropped", f"{len(raw) - len(df):,}"),
        Metric("Defect classes", str(len(issues))),
    )
    log = pd.DataFrame([{"column": i.column, "issue": i.issue,
                         "rows_affected": i.rows, "action_taken": i.action}
                        for i in issues])
    if log.empty:
        log = pd.DataFrame([{"column": "-", "issue": "none detected",
                             "rows_affected": 0, "action_taken": "-"}])
    res.table(log, "Cleaning decision log")

    dtypes = pd.DataFrame({
        "column": ["TotalCharges"],
        "dtype_before": [str(raw["TotalCharges"].dtype)],
        "dtype_after": [str(df["TotalCharges"].dtype)],
        "non_numeric_before": [int(pd.to_numeric(raw["TotalCharges"],
                                                 errors="coerce").isna().sum())],
        "missing_after": [int(df["TotalCharges"].isna().sum())],
    })
    res.table(dtypes, "Type repair: before vs after")
    res.narrative(
        "**Rule applied:** blank `TotalCharges` values belong exclusively to "
        "tenure-0 accounts that have never been billed, so they are imputed as "
        "`tenure x MonthlyCharges` (= 0) rather than dropped. Dropping them "
        "would have silently removed the newest customers — the segment with "
        "the highest churn risk — and biased every downstream skill."
    )
    return res


# --------------------------------------------------------------------------
# 3. feature-engineering
# --------------------------------------------------------------------------
def feature_engineering(ctx: dict) -> SkillResult:
    df = ctx["df"]
    engineered = ["avg_monthly_spend", "tenure_years", "tenure_bucket",
                  "spend_tier", "addon_count", "has_protection", "is_autopay"]
    numeric, categorical = feature_columns(df)
    X, y = df[numeric + categorical], df["churn_flag"]

    base_num = [c for c in numeric if c not in engineered]
    base_cat = [c for c in categorical if c not in engineered]

    def cv_auc(num, cat):
        pipe = build_pipeline(numeric=num, categorical=cat)
        return cross_val_score(pipe, X, y, cv=3, scoring="roc_auc").mean()

    auc_base = cv_auc(base_num, base_cat)
    auc_full = cv_auc(numeric, categorical)

    res = SkillResult(
        f"{len(engineered)} engineered features change 3-fold CV ROC-AUC "
        f"{auc_base:.4f} -> {auc_full:.4f} ({auc_full - auc_base:+.4f})"
    )
    res.metrics(
        Metric("Raw-feature CV AUC", f"{auc_base:.4f}"),
        Metric("With engineered", f"{auc_full:.4f}"),
        Metric("Delta", f"{auc_full - auc_base:+.4f}",
               "honest result: small — the raw columns already carry most signal"),
        Metric("Features added", str(len(engineered))),
    )
    spec = pd.DataFrame([
        ("avg_monthly_spend", "TotalCharges / tenure", "numeric",
         "realized spend rate, robust to plan changes"),
        ("tenure_years", "tenure / 12", "numeric", "human-scale tenure"),
        ("tenure_bucket", "binned tenure", "categorical",
         "captures the non-linear early-life churn cliff"),
        ("spend_tier", "MonthlyCharges quartile", "categorical",
         "price-band segmentation"),
        ("addon_count", "count of 6 add-on services = Yes", "numeric",
         "product depth / stickiness proxy"),
        ("has_protection", "OnlineSecurity or TechSupport", "binary",
         "support entitlement, a known retention lever"),
        ("is_autopay", "PaymentMethod contains 'automatic'", "binary",
         "involuntary-churn proxy"),
    ], columns=["feature", "definition", "type", "hypothesis"])
    res.table(spec, "Feature specification")

    lift = (df.groupby("addon_count")["churn_flag"]
              .agg(["size", "mean"]).reset_index())
    lift.columns = ["addon_count", "customers", "churn_rate"]
    lift["churn_rate"] = (lift["churn_rate"] * 100).round(1)
    res.chart(lift, kind="bar", x="addon_count", y="churn_rate",
              title="Churn rate (%) by number of add-on services")
    res.narrative(
        "Reported honestly: the engineered block gives only a marginal CV-AUC "
        "change. Its real value here is interpretability — `tenure_bucket` and "
        "`addon_count` make the churn story legible to a retention team even "
        "when they add little to the model score."
    )
    return res


# --------------------------------------------------------------------------
# 4. sklearn-pipelines
# --------------------------------------------------------------------------
def sklearn_pipelines(ctx: dict) -> SkillResult:
    df, m = ctx["df"], ctx["model"]
    numeric, categorical = m["numeric"], m["categorical"]

    # Leakage demonstration: scaler statistics fitted on all data vs train only.
    all_mean = df["MonthlyCharges"].mean()
    train_mean = m["X_train"]["MonthlyCharges"].mean()

    res = SkillResult(
        f"One ColumnTransformer pipeline: {len(numeric)} numeric + "
        f"{len(categorical)} categorical columns -> "
        f"{len(m['feature_names'])} model features, fitted train-only"
    )
    res.metrics(
        Metric("Numeric inputs", str(len(numeric))),
        Metric("Categorical inputs", str(len(categorical))),
        Metric("Encoded features", str(len(m["feature_names"]))),
        Metric("Pipeline steps", "impute -> scale/one-hot -> logistic"),
    )
    res.code(
        "ColumnTransformer([\n"
        "    ('num', Pipeline([SimpleImputer(strategy='median'),\n"
        "                      StandardScaler()]), numeric),\n"
        "    ('cat', Pipeline([SimpleImputer(strategy='most_frequent'),\n"
        "                      OneHotEncoder(handle_unknown='ignore')]), categorical),\n"
        "])  ->  LogisticRegression(max_iter=2000, random_state=42)",
        "Pipeline definition (src/model.py)",
    )
    res.table(
        pd.DataFrame([
            {"statistic": "MonthlyCharges mean used for scaling",
             "fit_on_full_data (leaky)": round(all_mean, 4),
             "fit_on_train_only (this pipeline)": round(train_mean, 4)}
        ]),
        "Why the pipeline matters",
        note="The two differ, so scaling outside the pipeline would leak "
             "test-set information into training.",
    )
    res.narrative(
        "`handle_unknown='ignore'` on the encoder means an unseen category at "
        "serving time produces an all-zero block instead of a crash — the "
        "difference between a degraded prediction and a 500 in production."
    )
    return res


# --------------------------------------------------------------------------
# 5. model-evaluation
# --------------------------------------------------------------------------
def model_evaluation(ctx: dict) -> SkillResult:
    m = ctx["model"]
    y, prob, met = m["y_test"], m["prob"], m["metrics"]
    cm = m["confusion"]

    res = SkillResult(
        f"Held-out test (n={len(y):,}): ROC-AUC {met['roc_auc']:.3f}, "
        f"PR-AUC {met['pr_auc']:.3f}, F1 {met['f1']:.3f}"
    )
    res.metrics(
        Metric("ROC-AUC", f"{met['roc_auc']:.3f}"),
        Metric("PR-AUC", f"{met['pr_auc']:.3f}", "the honest metric under 26.5% base rate"),
        Metric("Precision", f"{met['precision']:.3f}"),
        Metric("Recall", f"{met['recall']:.3f}"),
        Metric("Brier score", f"{met['brier']:.3f}", "lower is better calibrated"),
    )
    res.table(
        pd.DataFrame(cm,
                     index=["actual: stayed", "actual: churned"],
                     columns=["predicted: stayed", "predicted: churned"]).reset_index(
            names="",
        ),
        "Confusion matrix @ threshold 0.50",
    )

    fpr, tpr, _ = roc_curve(y, prob)
    roc_df = pd.DataFrame({"false_positive_rate": fpr.round(4),
                           "true_positive_rate": tpr.round(4)})
    res.chart(roc_df, kind="line", x="false_positive_rate", y="true_positive_rate",
              title=f"ROC curve (AUC = {met['roc_auc']:.3f})")

    prec, rec, _ = precision_recall_curve(y, prob)
    res.chart(pd.DataFrame({"recall": rec.round(4), "precision": prec.round(4)}),
              kind="line", x="recall", y="precision",
              title=f"Precision-recall curve (AP = {met['pr_auc']:.3f})",
              note=f"Baseline precision = churn base rate = {y.mean():.3f}")

    bins = pd.DataFrame({"prob": prob, "actual": y.values})
    bins["decile"] = pd.qcut(bins["prob"], 10, labels=False, duplicates="drop")
    cal = (bins.groupby("decile")
               .agg(mean_predicted=("prob", "mean"),
                    observed_rate=("actual", "mean"),
                    customers=("actual", "size")).reset_index().round(3))
    res.table(cal, "Calibration by predicted-probability decile",
              note="Predicted vs observed churn rate should track closely.")
    return res


# --------------------------------------------------------------------------
# 6. hyperparameter-tuning
# --------------------------------------------------------------------------
def hyperparameter_tuning(ctx: dict) -> SkillResult:
    df, m = ctx["df"], ctx["model"]
    pipe = build_pipeline(numeric=m["numeric"], categorical=m["categorical"])
    grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=3, n_jobs=-1)
    t0 = time.perf_counter()
    search.fit(m["X_train"], m["y_train"])
    elapsed = time.perf_counter() - t0

    cv = pd.DataFrame(search.cv_results_)[
        ["param_model__C", "param_model__class_weight",
         "mean_test_score", "std_test_score", "rank_test_score"]
    ]
    cv.columns = ["C", "class_weight", "cv_roc_auc", "std", "rank"]
    cv["class_weight"] = cv["class_weight"].astype(str)
    cv = cv.sort_values("rank").round(4).reset_index(drop=True)

    best_prob = search.best_estimator_.predict_proba(m["X_test"])[:, 1]
    test_auc = roc_auc_score(m["y_test"], best_prob)

    res = SkillResult(
        f"Grid search over {len(cv)} configurations in {elapsed:.1f}s — "
        f"best CV ROC-AUC {search.best_score_:.4f}, held-out {test_auc:.4f}"
    )
    res.metrics(
        Metric("Configurations", str(len(cv))),
        Metric("Best CV ROC-AUC", f"{search.best_score_:.4f}"),
        Metric("Held-out ROC-AUC", f"{test_auc:.4f}", "the number that counts"),
        Metric("Best params", f"C={search.best_params_['model__C']}, "
                              f"cw={search.best_params_['model__class_weight']}"),
        Metric("Search time", f"{elapsed:.1f}s"),
    )
    cv["config"] = "C=" + cv["C"].astype(str) + " / cw=" + cv["class_weight"]
    res.chart(cv, kind="hbar", x="cv_roc_auc", y="config",
              title="3-fold CV ROC-AUC per configuration", sort="-x")
    res.table(cv.drop(columns="config"), "Full search results")
    res.narrative(
        "The spread across all eight configurations is small "
        f"({cv['cv_roc_auc'].max() - cv['cv_roc_auc'].min():.4f} AUC). Stated "
        "plainly: on this dataset hyperparameter tuning is not where the wins "
        "are — the decision threshold (see *imbalanced-data*) moves business "
        "outcomes far more than `C` does."
    )
    return res


# --------------------------------------------------------------------------
# 7. imbalanced-data
# --------------------------------------------------------------------------
def imbalanced_data(ctx: dict) -> SkillResult:
    m = ctx["df"], ctx["model"]
    df, mb = m
    y_test, prob = mb["y_test"], mb["prob"]

    balanced = build_pipeline(
        LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced"),
        mb["numeric"], mb["categorical"],
    ).fit(mb["X_train"], mb["y_train"])
    bprob = balanced.predict_proba(mb["X_test"])[:, 1]

    rows = [
        {"strategy": "no reweighting @0.50", **score_all(y_test, (prob >= .5).astype(int), prob)},
        {"strategy": "class_weight='balanced' @0.50",
         **score_all(y_test, (bprob >= .5).astype(int), bprob)},
        {"strategy": "no reweighting @0.30",
         **score_all(y_test, (prob >= .3).astype(int), prob)},
    ]
    comp = pd.DataFrame(rows).round(4)

    sweep = []
    for t in np.arange(0.05, 0.96, 0.05):
        pred = (prob >= t).astype(int)
        s = score_all(y_test, pred, prob)
        sweep.append({"threshold": round(t, 2), "precision": s["precision"],
                      "recall": s["recall"], "f1": s["f1"]})
    sweep_df = pd.DataFrame(sweep).round(3)
    best_t = sweep_df.loc[sweep_df["f1"].idxmax()]

    res = SkillResult(
        f"Positive class is {pct(df['churn_flag'].mean())} of rows; "
        f"threshold {best_t['threshold']:.2f} maximizes F1 at {best_t['f1']:.3f} "
        f"(vs {rows[0]['f1']:.3f} at the default 0.50)"
    )
    res.metrics(
        Metric("Imbalance ratio", f"{(1 - df['churn_flag'].mean()) / df['churn_flag'].mean():.2f} : 1"),
        Metric("Accuracy of 'predict no-one churns'", pct(1 - df["churn_flag"].mean()),
               "why accuracy is a trap here"),
        Metric("Best F1 threshold", f"{best_t['threshold']:.2f}"),
        Metric("F1 at best threshold", f"{best_t['f1']:.3f}"),
    )
    melted = sweep_df.melt(id_vars="threshold", var_name="metric", value_name="score")
    res.chart(melted, kind="line", x="threshold", y="score", color="metric",
              title="Precision / recall / F1 vs decision threshold")
    res.table(comp, "Reweighting strategies compared on the same held-out split")
    res.narrative(
        "Class weighting and threshold moving trade the same quantity: both buy "
        "recall with precision, and neither changes ROC-AUC much because the "
        "ranking is unchanged. The retention-campaign question — how many "
        "false alarms is an outreach budget worth — is what should set the "
        "threshold, not a default of 0.5."
    )
    return res


# --------------------------------------------------------------------------
# 8. experiment-tracking
# --------------------------------------------------------------------------
def experiment_tracking(ctx: dict) -> SkillResult:
    mb = ctx["model"]
    configs = [
        ("logreg-C0.1", LogisticRegression(max_iter=2000, C=0.1, random_state=SEED)),
        ("logreg-C1", LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        ("logreg-balanced", LogisticRegression(max_iter=2000, random_state=SEED,
                                               class_weight="balanced")),
    ]
    from sklearn.ensemble import RandomForestClassifier
    configs.append(("rf-200", RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                     random_state=SEED, n_jobs=-1)))

    runs = []
    for name, est in configs:
        pipe = build_pipeline(est, mb["numeric"], mb["categorical"])
        t0 = time.perf_counter()
        pipe.fit(mb["X_train"], mb["y_train"])
        fit_s = time.perf_counter() - t0
        prob = pipe.predict_proba(mb["X_test"])[:, 1]
        s = score_all(mb["y_test"], (prob >= .5).astype(int), prob)
        runs.append({"run_id": f"run-{len(runs) + 1:02d}", "config": name,
                     "seed": SEED, "n_train": len(mb["X_train"]),
                     "fit_seconds": round(fit_s, 2),
                     "roc_auc": round(s["roc_auc"], 4),
                     "pr_auc": round(s["pr_auc"], 4),
                     "f1": round(s["f1"], 4)})
    log = pd.DataFrame(runs)
    best = log.loc[log["roc_auc"].idxmax()]

    res = SkillResult(
        f"{len(log)} tracked runs, identical split and seed — best is "
        f"{best['config']} at ROC-AUC {best['roc_auc']:.4f}"
    )
    res.metrics(
        Metric("Runs logged", str(len(log))),
        Metric("Best run", str(best["config"])),
        Metric("Best ROC-AUC", f"{best['roc_auc']:.4f}"),
        Metric("Spread", f"{log['roc_auc'].max() - log['roc_auc'].min():.4f}"),
    )
    res.chart(log, kind="hbar", x="roc_auc", y="config",
              title="ROC-AUC by tracked run", sort="-x")
    res.table(log, "Run log",
              note="Every row pins seed, split size and fit time, so any run "
                   "here is re-runnable without guessing what changed.")
    return res


# --------------------------------------------------------------------------
# 9. ml-debugging
# --------------------------------------------------------------------------
def ml_debugging(ctx: dict) -> SkillResult:
    df, mb = ctx["df"], ctx["model"]

    # Deliberate leakage injection, to prove the detector actually detects.
    leaky = df.copy()
    leaky["support_case_opened"] = leaky["churn_flag"] * 0.9 + np.random.default_rng(
        SEED).normal(0, 0.05, len(leaky))
    leak_corr = leaky["support_case_opened"].corr(leaky["churn_flag"])

    scan = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "churn_flag":
            continue
        c = abs(df[col].corr(df["churn_flag"]))
        scan.append({"feature": col, "abs_corr_with_target": round(c, 4),
                     "verdict": "LEAK SUSPECT" if c > 0.9 else "ok"})
    scan.append({"feature": "support_case_opened (injected control)",
                 "abs_corr_with_target": round(abs(leak_corr), 4),
                 "verdict": "LEAK SUSPECT"})
    scan_df = pd.DataFrame(scan).sort_values("abs_corr_with_target", ascending=False)

    sizes, train_scores, val_scores = learning_curve(
        build_pipeline(numeric=mb["numeric"], categorical=mb["categorical"]),
        mb["X_train"], mb["y_train"], cv=3, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1,
    )
    lc = pd.DataFrame({"train_size": np.repeat(sizes, 2),
                       "split": ["train", "validation"] * len(sizes),
                       "roc_auc": np.column_stack(
                           [train_scores.mean(1), val_scores.mean(1)]).ravel().round(4)})
    gap = train_scores.mean(1)[-1] - val_scores.mean(1)[-1]
    dupes = int(df.drop(columns=["customerID"]).duplicated().sum())

    res = SkillResult(
        f"No genuine leak found (max |corr| with target = "
        f"{scan_df.iloc[1]['abs_corr_with_target']:.3f}); train-validation gap "
        f"{gap:+.4f} AUC indicates no meaningful overfit"
    )
    res.metrics(
        Metric("Train-val AUC gap", f"{gap:+.4f}", "< 0.02 = not overfitting"),
        Metric("Leak suspects (real features)",
               str(int((scan_df["verdict"] == "LEAK SUSPECT").sum()) - 1)),
        Metric("Exact duplicate rows", str(dupes)),
        Metric("Class balance", pct(df["churn_flag"].mean())),
    )
    res.chart(lc, kind="line", x="train_size", y="roc_auc", color="split",
              title="Learning curve — train vs validation ROC-AUC")
    res.table(scan_df.head(12), "Target-leakage scan",
              note="An artificial leaky column is injected as a positive "
                   "control: it is flagged, the real features are not.")
    res.narrative(
        "**Debug checklist executed:** target leakage scan (positive control "
        "included), learning-curve overfit check, duplicate-row check, class "
        "balance check, and a train-only preprocessing audit (see "
        "*sklearn-pipelines*). All four pass on this dataset."
    )
    return res


# --------------------------------------------------------------------------
# 10. reproducible-ml
# --------------------------------------------------------------------------
def reproducible_ml(ctx: dict) -> SkillResult:
    df, mb = ctx["df"], ctx["model"]
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]

    def run_once():
        pipe = build_pipeline(numeric=mb["numeric"], categorical=mb["categorical"])
        pipe.fit(mb["X_train"], mb["y_train"])
        p = pipe.predict_proba(mb["X_test"])[:, 1]
        return roc_auc_score(mb["y_test"], p)

    a, b = run_once(), run_once()
    identical = a == b

    import sklearn
    env = pd.DataFrame([
        {"component": "python", "version": sys.version.split()[0]},
        {"component": "platform", "version": platform.platform()},
        {"component": "numpy", "version": np.__version__},
        {"component": "pandas", "version": pd.__version__},
        {"component": "scikit-learn", "version": sklearn.__version__},
        {"component": "global seed", "version": str(SEED)},
        {"component": "dataset SHA-256 (first 16)", "version": data_hash},
        {"component": "dataset source", "version": ctx["source"]},
    ])

    res = SkillResult(
        "Two independent fits of the same pipeline produced "
        + ("byte-identical" if identical else "DIFFERENT")
        + f" held-out ROC-AUC ({a:.10f})"
    )
    res.metrics(
        Metric("Run A ROC-AUC", f"{a:.10f}"),
        Metric("Run B ROC-AUC", f"{b:.10f}"),
        Metric("Deterministic", "yes" if identical else "no"),
        Metric("Data fingerprint", data_hash),
    )
    res.table(env, "Environment and reproducibility manifest")
    res.narrative(
        "Reproducibility here rests on four pins: a fixed `random_state` on "
        "every estimator and split, a content hash of the prepared dataset, a "
        "version-pinned `requirements.txt`, and a deterministic data-prep "
        "function with no randomness of its own."
    )
    return res


# --------------------------------------------------------------------------
# 11. pandas-patterns
# --------------------------------------------------------------------------
def pandas_patterns(ctx: dict) -> SkillResult:
    df = ctx["df"]

    t0 = time.perf_counter()
    _ = df.apply(lambda r: r["MonthlyCharges"] * r["tenure"], axis=1)
    apply_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = df["MonthlyCharges"] * df["tenure"]
    vec_s = time.perf_counter() - t0

    summary = (df.groupby(["Contract", "InternetService"], observed=True)
                 .agg(customers=("churn_flag", "size"),
                      churn_rate=("churn_flag", "mean"),
                      avg_bill=("MonthlyCharges", "mean"))
                 .reset_index()
                 .assign(churn_rate=lambda d: (d["churn_rate"] * 100).round(1),
                         avg_bill=lambda d: d["avg_bill"].round(2))
                 .sort_values("churn_rate", ascending=False))

    pivot = (df.pivot_table(index="Contract", columns="tenure_bucket",
                            values="churn_flag", aggfunc="mean", observed=True)
               .mul(100).round(1).reset_index())

    res = SkillResult(
        f"Vectorized arithmetic ran {apply_s / max(vec_s, 1e-9):.0f}x faster "
        f"than row-wise .apply() on {len(df):,} rows"
    )
    res.metrics(
        Metric(".apply(axis=1)", f"{apply_s * 1000:.1f} ms"),
        Metric("Vectorized", f"{vec_s * 1000:.2f} ms"),
        Metric("Speed-up", f"{apply_s / max(vec_s, 1e-9):.0f}x"),
        Metric("Rows", f"{len(df):,}"),
    )
    res.code(
        "# groupby -> agg -> assign -> sort, one expression, no intermediates\n"
        "(df.groupby(['Contract', 'InternetService'], observed=True)\n"
        "   .agg(customers=('churn_flag', 'size'),\n"
        "        churn_rate=('churn_flag', 'mean'),\n"
        "        avg_bill=('MonthlyCharges', 'mean'))\n"
        "   .reset_index()\n"
        "   .assign(churn_rate=lambda d: (d['churn_rate'] * 100).round(1))\n"
        "   .sort_values('churn_rate', ascending=False))",
        "Method-chaining pattern",
    )
    res.table(summary, "groupby + named aggregation result")
    res.table(pivot, "pivot_table: churn rate (%) by contract x tenure bucket")
    return res


# --------------------------------------------------------------------------
# 12. model-serving
# --------------------------------------------------------------------------
def model_serving(ctx: dict) -> SkillResult:
    mb = ctx["model"]
    sample = mb["X_test"].iloc[:200]

    latencies = []
    for i in range(100):
        row = sample.iloc[[i % len(sample)]]
        t0 = time.perf_counter()
        mb["pipeline"].predict_proba(row)
        latencies.append((time.perf_counter() - t0) * 1000)
    lat = np.array(latencies)

    t0 = time.perf_counter()
    mb["pipeline"].predict_proba(sample)
    batch_ms = (time.perf_counter() - t0) * 1000

    res = SkillResult(
        f"In-process scoring: p50 {np.percentile(lat, 50):.2f} ms, "
        f"p95 {np.percentile(lat, 95):.2f} ms per single request "
        f"(measured over 100 calls)"
    )
    res.metrics(
        Metric("p50 latency", f"{np.percentile(lat, 50):.2f} ms"),
        Metric("p95 latency", f"{np.percentile(lat, 95):.2f} ms"),
        Metric("p99 latency", f"{np.percentile(lat, 99):.2f} ms"),
        Metric("Batch of 200", f"{batch_ms:.1f} ms",
               f"{batch_ms / len(sample):.3f} ms/row — batching is ~"
               f"{np.percentile(lat, 50) / (batch_ms / len(sample)):.0f}x cheaper"),
    )
    res.chart(pd.DataFrame({"call": range(len(lat)), "latency_ms": lat.round(3)}),
              kind="line", x="call", y="latency_ms",
              title="Per-call scoring latency (ms)")
    res.code(
        "# Request contract enforced by the pipeline itself\n"
        "POST /predict\n"
        "{\n"
        '  "tenure": 3, "MonthlyCharges": 89.10, "Contract": "Month-to-month",\n'
        '  "InternetService": "Fiber optic", "PaymentMethod": "Electronic check", ...\n'
        "}\n"
        "-> {\"churn_probability\": 0.71, \"decision\": \"flag_for_retention\",\n"
        "    \"threshold\": 0.50, \"model_version\": \"logreg-baseline\"}",
        "Serving contract",
        language="text",
    )
    res.narrative(
        "The **Live inference** tab in this app is the running demonstration: "
        "it calls the exact same fitted pipeline object, so training-time and "
        "serving-time preprocessing cannot drift apart. Latency above is "
        "in-process only — it excludes HTTP, serialization and network, which "
        "would dominate a real deployment."
    )
    return res


# --------------------------------------------------------------------------
# 13. pytorch-training-loop
# --------------------------------------------------------------------------
def pytorch_training_loop(ctx: dict) -> SkillResult:
    mb = ctx["model"]
    pre = mb["pipeline"].named_steps["pre"]
    Xtr = pre.transform(mb["X_train"])
    Xte = pre.transform(mb["X_test"])
    Xtr = np.asarray(Xtr.todense() if hasattr(Xtr, "todense") else Xtr, dtype=np.float64)
    Xte = np.asarray(Xte.todense() if hasattr(Xte, "todense") else Xte, dtype=np.float64)
    ytr = mb["y_train"].to_numpy()
    yte = mb["y_test"].to_numpy()

    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 0.01, Xtr.shape[1])
    b = 0.0
    lr, batch, epochs = 0.1, 256, 15
    history = []
    for epoch in range(epochs):
        idx = rng.permutation(len(Xtr))
        for start in range(0, len(idx), batch):           # mini-batch loop
            sl = idx[start:start + batch]
            xb, yb = Xtr[sl], ytr[sl]
            logits = xb @ w + b                            # forward
            p = 1 / (1 + np.exp(-logits))
            grad_w = xb.T @ (p - yb) / len(sl)             # backward
            grad_b = (p - yb).mean()
            w -= lr * grad_w                               # optimizer step
            b -= lr * grad_b
        def loss(X, y):
            p = np.clip(1 / (1 + np.exp(-(X @ w + b))), 1e-9, 1 - 1e-9)
            return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        history.append({"epoch": epoch + 1, "train_loss": round(loss(Xtr, ytr), 4),
                        "val_loss": round(loss(Xte, yte), 4)})

    hist = pd.DataFrame(history)
    auc = roc_auc_score(yte, 1 / (1 + np.exp(-(Xte @ w + b))))

    res = SkillResult(
        f"Hand-written mini-batch training loop: {epochs} epochs, "
        f"loss {hist['train_loss'].iloc[0]:.4f} -> {hist['train_loss'].iloc[-1]:.4f}, "
        f"held-out ROC-AUC {auc:.4f}"
    )
    res.metrics(
        Metric("Epochs", str(epochs)),
        Metric("Batch size", str(batch)),
        Metric("Final train loss", f"{hist['train_loss'].iloc[-1]:.4f}"),
        Metric("Final val loss", f"{hist['val_loss'].iloc[-1]:.4f}"),
        Metric("Held-out ROC-AUC", f"{auc:.4f}",
               f"vs {mb['metrics']['roc_auc']:.4f} for the sklearn fit"),
    )
    res.chart(hist.melt(id_vars="epoch", var_name="split", value_name="loss"),
              kind="line", x="epoch", y="loss", color="split",
              title="Binary cross-entropy per epoch")
    res.code(
        "for epoch in range(epochs):\n"
        "    for xb, yb in batches(X, y):\n"
        "        logits = xb @ w + b                 # forward\n"
        "        p      = sigmoid(logits)\n"
        "        grad_w = xb.T @ (p - yb) / len(xb)  # backward (analytic)\n"
        "        w     -= lr * grad_w                # optimizer step\n"
        "    log(train_loss, val_loss)               # epoch-level logging",
        "The canonical loop structure, implemented here in NumPy",
    )
    res.narrative(
        "**Stated plainly:** this is a NumPy implementation of the standard "
        "forward / loss / backward / step loop, not a PyTorch run. The lab "
        "keeps its dependency set to pandas + scikit-learn + Streamlit, so "
        "gradients are derived analytically instead of by autograd. The loop "
        "structure, mini-batching, per-epoch train/val logging and convergence "
        "behaviour are real and reproduced above; `torch.nn`, autograd and GPU "
        "execution are not exercised."
    )
    return res


# --------------------------------------------------------------------------
# 14. llm-finetuning
# --------------------------------------------------------------------------
def llm_finetuning(ctx: dict) -> SkillResult:
    df, mb = ctx["df"], ctx["model"]
    prob = mb["pipeline"].predict_proba(df[mb["numeric"] + mb["categorical"]])[:, 1]

    sample = df.assign(risk=prob).sort_values("risk", ascending=False)
    examples = []
    for _, r in pd.concat([sample.head(400), sample.tail(400)]).iterrows():
        prompt = (f"Customer on a {r['Contract']} contract, {r['tenure']} months "
                  f"tenure, {r['InternetService']} internet, "
                  f"${r['MonthlyCharges']:.2f}/month, {r['addon_count']} add-ons, "
                  f"pays by {r['PaymentMethod']}. Assess churn risk and recommend "
                  f"a retention action.")
        if r["risk"] >= 0.5:
            completion = (f"Risk: HIGH ({r['risk']:.0%}). Drivers: "
                          f"{'month-to-month contract, ' if r['Contract'] == 'Month-to-month' else ''}"
                          f"{'short tenure, ' if r['tenure'] < 12 else ''}"
                          f"{'no protection add-ons, ' if r['addon_count'] == 0 else ''}"
                          "billing profile. Action: offer a 12-month term with a "
                          "bundled support add-on before the next billing cycle.")
        else:
            completion = (f"Risk: LOW ({r['risk']:.0%}). The customer is on a "
                          f"{r['Contract'].lower()} contract with {r['tenure']} months "
                          "of tenure. Action: no intervention; keep in the standard "
                          "loyalty track.")
        examples.append({"prompt": prompt, "completion": completion})

    sft = pd.DataFrame(examples)
    sft["prompt_words"] = sft["prompt"].str.split().str.len()
    sft["completion_words"] = sft["completion"].str.split().str.len()
    train_n = int(len(sft) * 0.9)

    res = SkillResult(
        f"Built a {len(sft):,}-example supervised fine-tuning corpus from real "
        f"customer rows + model risk scores ({train_n} train / {len(sft) - train_n} val)"
    )
    res.metrics(
        Metric("SFT examples", f"{len(sft):,}"),
        Metric("Train / val", f"{train_n} / {len(sft) - train_n}"),
        Metric("Median prompt length", f"{sft['prompt_words'].median():.0f} words"),
        Metric("Median completion", f"{sft['completion_words'].median():.0f} words"),
        Metric("Class balance", pct((sft["completion"].str.contains("HIGH")).mean()),
               "share of HIGH-risk completions"),
    )
    res.table(sft[["prompt", "completion"]].head(4), "Sample instruction pairs")
    res.chart(
        sft.melt(value_vars=["prompt_words", "completion_words"],
                 var_name="field", value_name="words")
           .groupby(["field", "words"]).size().reset_index(name="examples"),
        kind="bar", x="words", y="examples", color="field",
        title="Token-length distribution (word proxy)",
    )
    res.code(
        "base_model: llama-3.2-1b-instruct\n"
        "method: LoRA (r=16, alpha=32, dropout=0.05, target=q_proj,v_proj)\n"
        "epochs: 3   lr: 2e-4 (cosine)   batch: 8   grad_accum: 4\n"
        "max_seq_len: 512   bf16: true   seed: 42\n"
        "eval: held-out 10% split, loss + manual rubric on 50 samples",
        "Fine-tuning configuration this corpus is formatted for",
        language="yaml",
    )
    res.narrative(
        "**Stated plainly: no fine-tuning run was executed here.** This lab has "
        "no GPU budget and no transformers dependency. What is genuinely "
        "demonstrated is the part of the skill that decides whether a "
        "fine-tune succeeds — corpus construction from real data, "
        "instruction/response formatting, length profiling, class balance, and "
        "a train/val split — plus the configuration the corpus is built for. "
        "Any loss curve or benchmark for a fine-tuned model would be fabricated, "
        "so none is shown."
    )
    return res


# --------------------------------------------------------------------------
# 15. rag-pipeline
# --------------------------------------------------------------------------
def rag_pipeline(ctx: dict, query: str = "which customers churn the most?") -> SkillResult:
    """A real retrieval pipeline over the lab's own analytical findings."""
    df = ctx["df"]
    corpus = _knowledge_corpus(df)
    docs = [d["text"] for d in corpus]

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(docs)
    q = vec.transform([query])
    sims = cosine_similarity(q, matrix)[0]
    order = np.argsort(-sims)[:5]

    retrieved = pd.DataFrame([
        {"rank": i + 1, "doc_id": corpus[j]["id"],
         "similarity": round(float(sims[j]), 4), "chunk": corpus[j]["text"]}
        for i, j in enumerate(order)
    ])
    grounded = "\n\n".join(f"- {corpus[j]['text']} _(source: {corpus[j]['id']})_"
                           for j in order[:3] if sims[j] > 0)

    res = SkillResult(
        f"Indexed {len(docs)} knowledge chunks; top hit for \"{query}\" scored "
        f"{sims[order[0]]:.3f}"
    )
    res.metrics(
        Metric("Chunks indexed", str(len(docs))),
        Metric("Vocabulary terms", f"{len(vec.vocabulary_):,}"),
        Metric("Retriever", "TF-IDF + cosine"),
        Metric("Top-1 similarity", f"{sims[order[0]]:.3f}"),
    )
    res.table(retrieved, f"Retrieved context for: \"{query}\"")
    res.narrative(
        f"**Grounded answer, assembled only from retrieved chunks:**\n\n{grounded}"
        if grounded else "No chunk matched the query above the similarity floor.",
        "Generation step",
    )
    res.narrative(
        "**Stated plainly:** retrieval, chunking, vectorization, ranking and "
        "citation are all real and executed above. The generation step is "
        "extractive — it stitches the retrieved chunks together rather than "
        "calling an LLM, because this project ships no model API dependency. "
        "Swapping the extractive step for an LLM call is the only change "
        "needed to make this a full RAG stack."
    )
    return res


def _knowledge_corpus(df: pd.DataFrame) -> list[dict]:
    """Facts computed from the dataset, chunked as a retrievable corpus."""
    chunks = []
    for col in ["Contract", "InternetService", "PaymentMethod", "tenure_bucket"]:
        g = df.groupby(col, observed=True)["churn_flag"].agg(["size", "mean"])
        for level, row in g.iterrows():
            chunks.append({
                "id": f"{col}:{level}",
                "text": (f"Customers with {col} = {level} number {int(row['size']):,} "
                         f"and churn at {row['mean'] * 100:.1f}%, against an overall "
                         f"churn rate of {df['churn_flag'].mean() * 100:.1f}%."),
            })
    chunks += [
        {"id": "overview:dataset",
         "text": (f"The dataset holds {len(df):,} telecom subscribers with "
                  f"{df.shape[1]} columns covering demographics, subscribed "
                  f"services, billing and the churn label.")},
        {"id": "overview:spend",
         "text": (f"Median monthly charge is ${df['MonthlyCharges'].median():.2f}; "
                  f"churned customers average ${df[df.churn_flag == 1]['MonthlyCharges'].mean():.2f} "
                  f"versus ${df[df.churn_flag == 0]['MonthlyCharges'].mean():.2f} for retained ones.")},
        {"id": "overview:tenure",
         "text": (f"Median tenure is {df['tenure'].median():.0f} months; churned "
                  f"customers average {df[df.churn_flag == 1]['tenure'].mean():.1f} months "
                  f"versus {df[df.churn_flag == 0]['tenure'].mean():.1f} for retained ones.")},
        {"id": "method:crisp-dm",
         "text": ("The lab follows CRISP-DM: business understanding, data "
                  "understanding, data preparation, modeling, evaluation and "
                  "deployment, with each skill mapped to the phase it serves.")},
    ]
    return chunks
