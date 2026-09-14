"""Runs all three AutoGluon capability demos end to end:

1. multimodal_reviews  -- TabularPredictor with an automatic text feature
                          generator alongside numeric/categorical columns.
2. diabetes_quantile   -- TabularPredictor in ``quantile`` problem-type mode,
                          producing P10/P50/P90 prediction bands.
3. retail_demand       -- TimeSeriesPredictor forecasting six related series.

Each task is split once, fit once with a fixed time budget, evaluated once on
a held-out slice, and every artifact the Streamlit dashboard reads is written
under ``artifacts/<task_key>/``. No metric is computed twice or overwritten.
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
from automl_tabular import fit_classifier, fit_quantile_regressor  # noqa: E402
from automl_timeseries import fit_forecaster, to_ts_dataframe  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
SEED = 42
TABULAR_TIME_LIMIT = 60
TIMESERIES_TIME_LIMIT = 90


def run_multimodal_reviews() -> dict:
    key = "multimodal_reviews"
    print(f"\n=== {key} ===")
    df, task = load_data.load(key)
    task_artifacts = ARTIFACTS / key
    task_artifacts.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[task["label"]])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    t0 = time.time()
    predictor = fit_classifier(train_df, task["label"], MODELS / key, TABULAR_TIME_LIMIT, task["eval_metric"])
    fit_seconds = time.time() - t0

    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(task_artifacts / "leaderboard.csv", index=False)

    test_metrics = predictor.evaluate(test_df, silent=True)
    importance = predictor.feature_importance(test_df, silent=True)
    importance.to_csv(task_artifacts / "feature_importance.csv")

    sample = test_df.sample(n=min(20, len(test_df)), random_state=SEED).reset_index(drop=True)
    proba = predictor.predict_proba(sample.drop(columns=[task["label"]]))
    sample_out = sample.copy()
    sample_out["predicted_recommended"] = predictor.predict(sample.drop(columns=[task["label"]])).values
    sample_out["predicted_proba_yes"] = proba["yes"].values
    sample_out.to_csv(task_artifacts / "holdout_sample.csv", index=False)

    metrics = {
        "task_key": key,
        "display_name": task["display_name"],
        "problem_type": task["problem_type"],
        "eval_metric": task["eval_metric"],
        "source": task["source"],
        "n_rows": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fit_seconds": round(fit_seconds, 1),
        "best_model": predictor.model_best,
        "n_models_fit": int(len(leaderboard)),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    (task_artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Held-out test metrics: {metrics['test_metrics']}")
    return metrics


def run_diabetes_quantile() -> dict:
    key = "diabetes_quantile"
    print(f"\n=== {key} ===")
    df, task = load_data.load(key)
    task_artifacts = ARTIFACTS / key
    task_artifacts.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    t0 = time.time()
    predictor = fit_quantile_regressor(
        train_df, task["label"], task["quantile_levels"], MODELS / key, TABULAR_TIME_LIMIT
    )
    fit_seconds = time.time() - t0

    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(task_artifacts / "leaderboard.csv", index=False)

    test_metrics = predictor.evaluate(test_df, silent=True)

    sample = test_df.sample(n=min(20, len(test_df)), random_state=SEED).reset_index(drop=True)
    preds = predictor.predict(sample.drop(columns=[task["label"]]))
    sample_out = sample.copy()
    for q in task["quantile_levels"]:
        sample_out[f"pred_p{int(q * 100)}"] = preds[q].values
    sample_out.to_csv(task_artifacts / "holdout_sample.csv", index=False)

    coverage = (
        (sample_out[task["label"]] >= sample_out["pred_p10"]) & (sample_out[task["label"]] <= sample_out["pred_p90"])
    ).mean()

    metrics = {
        "task_key": key,
        "display_name": task["display_name"],
        "problem_type": task["problem_type"],
        "eval_metric": task["eval_metric"],
        "source": task["source"],
        "n_rows": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fit_seconds": round(fit_seconds, 1),
        "best_model": predictor.model_best,
        "n_models_fit": int(len(leaderboard)),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "p10_p90_holdout_coverage": float(coverage),
    }
    (task_artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Held-out test metrics: {metrics['test_metrics']}, P10-P90 coverage: {coverage:.2f}")
    return metrics


def run_retail_demand() -> dict:
    key = "retail_demand"
    print(f"\n=== {key} ===")
    df, task = load_data.load(key)
    task_artifacts = ARTIFACTS / key
    task_artifacts.mkdir(parents=True, exist_ok=True)

    full_ts = to_ts_dataframe(df, task["id_column"], task["timestamp_column"])
    prediction_length = task["prediction_length"]
    train_ts, test_ts = full_ts.train_test_split(prediction_length)

    t0 = time.time()
    predictor = fit_forecaster(
        train_ts, task["target"], prediction_length, MODELS / key, TIMESERIES_TIME_LIMIT, task["eval_metric"]
    )
    fit_seconds = time.time() - t0

    leaderboard = predictor.leaderboard(test_ts, silent=True)
    leaderboard.to_csv(task_artifacts / "leaderboard.csv", index=False)

    test_metrics = predictor.evaluate(test_ts)

    forecast = predictor.predict(train_ts)
    forecast.reset_index().to_csv(task_artifacts / "forecast.csv", index=False)
    test_ts.reset_index().to_csv(task_artifacts / "actuals.csv", index=False)
    train_ts.reset_index().to_csv(task_artifacts / "history.csv", index=False)

    metrics = {
        "task_key": key,
        "display_name": task["display_name"],
        "eval_metric": task["eval_metric"],
        "source": task["source"],
        "n_series": df[task["id_column"]].nunique(),
        "n_rows": len(df),
        "prediction_length": prediction_length,
        "fit_seconds": round(fit_seconds, 1),
        "best_model": predictor.model_best,
        "n_models_fit": int(len(leaderboard)),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    (task_artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Held-out test metrics: {metrics['test_metrics']}")
    return metrics


def main() -> None:
    all_metrics = {
        "multimodal_reviews": run_multimodal_reviews(),
        "diabetes_quantile": run_diabetes_quantile(),
        "retail_demand": run_retail_demand(),
    }
    (ARTIFACTS / "summary.json").write_text(json.dumps(all_metrics, indent=2))
    print("\nAll tasks complete. Artifacts written under artifacts/, models under models/.")


if __name__ == "__main__":
    main()
