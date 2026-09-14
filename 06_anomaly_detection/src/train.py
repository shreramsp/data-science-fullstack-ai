"""CRISP-DM pipeline: data prep -> baseline sweep -> autoresearch -> honest test.

Run ``python src/train.py``. Everything the dashboard shows is written here to
``artifacts/`` -- the dashboard computes no metric of its own, so what you see
in the UI is exactly what this run produced.

Order of operations, and why it matters:

1. Clean and split (60/20/20, stratified).
2. **Baseline sweep** -- all five detectors at their default settings, so the
   autoresearch result has something to be better *than*.
3. **AutoResearch** -- hill climbing on validation average precision.
4. **Operating point** -- the alert threshold is chosen on validation, at the
   analyst alert budget. The test split is not consulted.
5. **Test** -- the winning pipeline is scored once on the held-out split.
   Validation numbers are optimistically biased (the search selected on them);
   the test numbers are the ones to quote.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data")]

import load_data  # noqa: E402
from autoresearch import (ALERT_BUDGET, Candidate, Evaluator, LITERATURE,  # noqa: E402
                          alert_metrics, hill_climb)
from detectors import DETECTORS, build, default_params  # noqa: E402
from features import SEED, clean, engineer, make_scaler, matrix, split  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
RESTARTS = 3
BUDGET = 140


def single_feature_auc(frame: pd.DataFrame) -> pd.DataFrame:
    """Data Understanding: how much does each raw feature separate on its own?"""
    y = frame["Class"].to_numpy()
    rows = []
    for col in [c for c in frame.columns if c.startswith("V")] + ["Amount", "hour"]:
        auc = roc_auc_score(y, frame[col].to_numpy())
        rows.append({
            "feature": col,
            "auc": auc,
            "separation": abs(auc - 0.5) * 2,  # 0 = useless alone, 1 = perfect alone
            "mean_normal": float(frame.loc[y == 0, col].mean()),
            "mean_anomaly": float(frame.loc[y == 1, col].mean()),
        })
    return pd.DataFrame(rows).sort_values("separation", ascending=False).reset_index(drop=True)


def evaluate_on(cand: Candidate, train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, dict]:
    Xtr, Xte = matrix(train, cand.feature_set), matrix(target, cand.feature_set)
    scaler = make_scaler(cand.scaler).fit(Xtr)
    det = build(cand.detector, cand.params).fit(scaler.transform(Xtr))
    scores = det.score(scaler.transform(Xte))
    y = target["Class"].to_numpy()
    metrics = {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        **alert_metrics(y, scores),
    }
    return scores, metrics


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    started = time.perf_counter()

    # --- Phase 2/3: Data Understanding & Preparation -------------------------
    raw, source = load_data.load()
    print(f"[data] source={source} rows={len(raw):,}")
    cleaned, report = clean(raw)
    engineered = engineer(cleaned)
    parts = split(engineered)
    train, valid, test = parts["train"], parts["valid"], parts["test"]
    print(f"[split] train={len(train):,}({int(train.Class.sum())}) "
          f"valid={len(valid):,}({int(valid.Class.sum())}) test={len(test):,}({int(test.Class.sum())})")

    pd.DataFrame(report.to_rows()).to_csv(ARTIFACTS / "cleaning_report.csv", index=False)
    diagnostics = single_feature_auc(engineered)
    diagnostics.to_csv(ARTIFACTS / "feature_diagnostics.csv", index=False)

    # --- Phase 4a: baseline sweep -------------------------------------------
    print("[baseline] five detectors at default settings")
    baseline_rows = []
    for kind in DETECTORS:
        cand = Candidate("components_amount", "standard", kind, default_params(kind))
        t0 = time.perf_counter()
        _, vm = evaluate_on(cand, train, valid)
        _, tm = evaluate_on(cand, train, test)
        baseline_rows.append({
            "detector": kind, "label": cand.label(),
            "valid_average_precision": vm["average_precision"], "valid_roc_auc": vm["roc_auc"],
            "test_average_precision": tm["average_precision"], "test_roc_auc": tm["roc_auc"],
            "test_precision_at_budget": tm["precision_at_budget"],
            "test_recall_at_budget": tm["recall_at_budget"],
            "seconds": time.perf_counter() - t0,
        })
        print(f"  {kind:10s} valid AP={vm['average_precision']:.4f} test AP={tm['average_precision']:.4f}")
    baselines = pd.DataFrame(baseline_rows).sort_values("valid_average_precision", ascending=False)
    baselines.to_csv(ARTIFACTS / "baselines.csv", index=False)

    # --- Phase 4b: autoresearch ---------------------------------------------
    print(f"[autoresearch] hill climbing, {RESTARTS} restarts, budget {BUDGET} evaluations")
    best, best_valid, history = hill_climb(train, valid, restarts=RESTARTS, budget=BUDGET)
    hist = pd.DataFrame(history)
    hist.to_csv(ARTIFACTS / "search_history.csv", index=False)
    print(f"[autoresearch] winner valid AP={best_valid['average_precision']:.4f} :: {best.label()}")

    # --- Phase 5: operating point on validation, then one look at test ------
    valid_scores = Evaluator(train, valid).score_valid(best)
    k = max(1, int(round(len(valid_scores) * ALERT_BUDGET)))
    threshold = float(np.sort(valid_scores)[::-1][k - 1])

    test_scores, test_metrics = evaluate_on(best, train, test)
    y_test = test["Class"].to_numpy()
    flagged = test_scores >= threshold
    tp = int((flagged & (y_test == 1)).sum())
    fp = int((flagged & (y_test == 0)).sum())
    fn = int((~flagged & (y_test == 1)).sum())
    tn = int((~flagged & (y_test == 0)).sum())
    operating = {
        "threshold": threshold,
        "threshold_chosen_on": "validation split, top 0.5% alert budget",
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "alert_rate": float(flagged.mean()),
    }
    print(f"[test] AP={test_metrics['average_precision']:.4f} ROC={test_metrics['roc_auc']:.4f} "
          f"precision@op={operating['precision']:.3f} recall@op={operating['recall']:.3f}")

    pd.DataFrame({"score": test_scores, "label": y_test,
                  "amount": test["Amount"].to_numpy(),
                  "hour": test["hour"].to_numpy().round(2)}).to_csv(
        ARTIFACTS / "test_scores.csv", index=False)

    # --- Phase 6: deployable artefact ---------------------------------------
    Xtr = matrix(train, best.feature_set)
    scaler = make_scaler(best.scaler).fit(Xtr)
    detector = build(best.detector, best.params).fit(scaler.transform(Xtr))
    joblib.dump({"candidate": best.as_dict(), "feature_set": best.feature_set,
                 "scaler": scaler, "detector": detector, "threshold": threshold},
                ARTIFACTS / "best_pipeline.joblib")

    best_baseline = baselines.iloc[0]
    metrics = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "runtime_seconds": round(time.perf_counter() - started, 1),
        "data_source": source,
        "seed": SEED,
        "cleaning": {"raw_rows": report.raw_rows, "duplicates_dropped": report.duplicates_dropped,
                     "negative_amounts_fixed": report.negative_amounts_fixed,
                     "missing_amounts_imputed": report.missing_amounts_imputed,
                     "final_rows": report.final_rows, "anomalies": report.frauds,
                     "prevalence": report.prevalence},
        "splits": {name: {"rows": len(part), "anomalies": int(part.Class.sum())}
                   for name, part in parts.items()},
        "alert_budget": ALERT_BUDGET,
        "search": {"restarts": RESTARTS, "evaluation_budget": BUDGET,
                   # The budget caps *distinct* pipelines fitted; repeat visits
                   # to a configuration are served from cache and logged anyway.
                   "evaluations_logged": int(len(hist)),
                   "pipelines_fitted": int(hist["label"].nunique()),
                   "objective": "validation average precision",
                   "best_config": best.as_dict(), "best_label": best.label(),
                   "best_valid": best_valid},
        "baseline_best": {"detector": best_baseline["detector"],
                          "valid_average_precision": float(best_baseline["valid_average_precision"]),
                          "test_average_precision": float(best_baseline["test_average_precision"])},
        "test": test_metrics,
        "operating_point": operating,
        "literature": LITERATURE,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] {metrics['runtime_seconds']}s -> artifacts/")


if __name__ == "__main__":
    main()
