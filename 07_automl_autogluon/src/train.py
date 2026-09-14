"""CRISP-DM pipeline: for each of the three tasks --
clean/split -> baseline AutoGluon fit -> AutoResearch hill climb over the
stacking architecture -> one refit at the winning config -> one honest
held-out evaluation -> artifacts written to ``artifacts/<task>/``.

Run with ``python src/train.py``. Everything the Streamlit dashboard shows is
read from what this script writes -- the dashboard computes no metric of its
own, and the held-out test split is scored exactly once per task, after the
architecture search is finished and frozen.

AutoGluon's own out-of-fold validation score (``score_val``, always
"higher is better" in its convention regardless of the underlying metric) is
the signal AutoResearch climbs on. It never sees the test split.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data")]

import load_data  # noqa: E402
from automl import best_val_score, fit_predictor  # noqa: E402
from autoresearch import Candidate, LITERATURE, cleanup, hill_climb  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
SEARCH_TMP = MODELS / "_search_tmp"
SEED = 42

BASELINE = Candidate(num_bag_folds=0, num_stack_levels=0)
SEARCH_TIME_LIMIT = 20
FINAL_TIME_LIMIT = 45
MAX_EVALS = 7


def split_task(df: pd.DataFrame, task: load_data.Task) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = df[task.label] if task.problem_type in ("binary", "multiclass") else None
    train_search, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=stratify)
    return train_search.reset_index(drop=True), test.reset_index(drop=True)


def run_task(task_key: str) -> dict:
    print(f"\n=== {task_key} ===")
    df, task = load_data.load(task_key)
    train_search, test = split_task(df, task)
    task_artifacts = ARTIFACTS / task_key
    task_artifacts.mkdir(parents=True, exist_ok=True)

    def evaluate(cand: Candidate) -> float:
        predictor = fit_predictor(
            train_search, task.label, task.problem_type, task.eval_metric,
            SEARCH_TMP, cand.num_bag_folds, cand.num_stack_levels, SEARCH_TIME_LIMIT,
        )
        return best_val_score(predictor)

    t0 = time.time()
    best_cand, best_score, trace = hill_climb(evaluate, BASELINE, max_evals=MAX_EVALS)
    cleanup(SEARCH_TMP)
    search_seconds = time.time() - t0
    print(f"AutoResearch winner: {best_cand.label} (val={best_score:.4f}, {search_seconds:.1f}s, {len(trace)} evals)")

    baseline_score = next(r["score_val"] for r in trace if r["num_bag_folds"] == 0 and r["num_stack_levels"] == 0)

    final_path = MODELS / task_key
    final_predictor = fit_predictor(
        train_search, task.label, task.problem_type, task.eval_metric,
        final_path, best_cand.num_bag_folds, best_cand.num_stack_levels, FINAL_TIME_LIMIT,
    )
    leaderboard = final_predictor.leaderboard(silent=True)
    leaderboard.to_csv(task_artifacts / "leaderboard.csv", index=False)

    test_metrics = final_predictor.evaluate(test, silent=True)
    importance = final_predictor.feature_importance(test, silent=True)
    importance.to_csv(task_artifacts / "feature_importance.csv")

    sample = test.sample(n=min(30, len(test)), random_state=SEED).reset_index(drop=True)
    preds = final_predictor.predict(sample.drop(columns=[task.label]))
    sample_out = sample.copy()
    sample_out["prediction"] = preds.values
    sample_out.to_csv(task_artifacts / "holdout_sample.csv", index=False)

    pd.DataFrame(trace).to_csv(task_artifacts / "search_history.csv", index=False)

    metrics = {
        "task_key": task_key,
        "display_name": task.display_name,
        "problem_type": task.problem_type,
        "eval_metric": task.eval_metric,
        "source": task.source,
        "n_rows": len(df),
        "n_train_search": len(train_search),
        "n_test": len(test),
        "baseline_config": {"num_bag_folds": 0, "num_stack_levels": 0},
        "baseline_score_val": baseline_score,
        "best_config": {"num_bag_folds": best_cand.num_bag_folds, "num_stack_levels": best_cand.num_stack_levels},
        "best_score_val": best_score,
        "search_seconds": round(search_seconds, 1),
        "search_evals": len(trace),
        "best_model": final_predictor.model_best,
        "stack_levels_fit": int(leaderboard["stack_level"].max()),
        "n_models_fit": int(len(leaderboard)),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    (task_artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Held-out test metrics: {metrics['test_metrics']}")
    return metrics


def main() -> None:
    all_metrics = {}
    for task_key in load_data.TASKS:
        all_metrics[task_key] = run_task(task_key)
    (ARTIFACTS / "literature.json").write_text(json.dumps(LITERATURE, indent=2))
    (ARTIFACTS / "summary.json").write_text(json.dumps(all_metrics, indent=2))
    print("\nAll tasks complete. Artifacts written under artifacts/, models under models/.")


if __name__ == "__main__":
    main()
