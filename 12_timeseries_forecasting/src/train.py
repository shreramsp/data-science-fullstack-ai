"""CRISP-DM pipeline: generate/load data -> engineer features -> fit four
forecasting models -> evaluate on a chronological holdout -> persist
artifacts for the dashboard.

Run:  python src/train.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_COLUMNS, TEST_DAYS, chronological_split, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
SEED = 20260913
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


def compute_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    smape = float(np.mean(2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-6)) * 100)
    return {"mae": float(mae), "rmse": float(rmse), "smape": smape}


def fit_holt_winters(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit one additive Holt-Winters model per series; forecast the test horizon."""
    series_models: dict[str, ExponentialSmoothing] = {}
    preds = []
    for sid, g_train in train_raw.groupby("series_id"):
        g_train = g_train.sort_values("date")
        g_test = test_raw[test_raw["series_id"] == sid].sort_values("date")
        y = g_train["sales"].to_numpy(dtype=float)
        model = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated",
        ).fit(optimized=True)
        series_models[sid] = model
        fc = model.forecast(len(g_test))
        preds.append(pd.DataFrame({"date": g_test["date"].to_numpy(), "series_id": sid, "pred_holt_winters": fc}))
    return series_models, pd.concat(preds, ignore_index=True)


def main() -> None:
    t0 = time.time()
    ARTIFACTS.mkdir(exist_ok=True)

    raw, feat, train, test = prepare()
    raw["series_id"] = raw["store_id"] + "_" + raw["category"]
    cutoff = feat["date"].max() - pd.Timedelta(days=TEST_DAYS - 1)
    raw_train, raw_test = chronological_split(raw, TEST_DAYS)
    print(f"Rows: raw={len(raw):,}  feature={len(feat):,}  train={len(train):,}  test={len(test):,}  "
          f"series={raw['store_id'].nunique() * raw['category'].nunique()}  cutoff={cutoff.date()}")

    # --- naive baselines (row-level, already available as feature columns) ---
    test = test.copy()
    test["pred_naive"] = test["lag_1"]
    test["pred_seasonal_naive"] = test["lag_7"]

    # --- Holt-Winters, one model per series ---
    hw_models, hw_preds = fit_holt_winters(raw_train, raw_test)
    test = test.merge(hw_preds, on=["date", "series_id"], how="left")

    # --- global gradient-boosted trees over lag/calendar features ---
    gbm = HistGradientBoostingRegressor(random_state=SEED, max_depth=6, max_iter=300, learning_rate=0.06)
    gbm.fit(train[FEATURE_COLUMNS], train["sales"])
    test["pred_gbm"] = gbm.predict(test[FEATURE_COLUMNS])

    model_cols = {
        "naive": "pred_naive",
        "seasonal_naive": "pred_seasonal_naive",
        "holt_winters": "pred_holt_winters",
        "gbm": "pred_gbm",
    }

    # --- evaluate: per-series metrics, then macro-averaged + pooled overall ---
    per_series_rows, overall = [], {}
    for name, col in model_cols.items():
        overall[name] = compute_metrics(test["sales"], test[col])
        for sid, g in test.groupby("series_id"):
            m = compute_metrics(g["sales"], g[col])
            per_series_rows.append({"model": name, "series_id": sid, **m})

    comparison_df = pd.DataFrame(
        [{"model": name, **m} for name, m in overall.items()]
    ).sort_values("mae").reset_index(drop=True)
    per_series_df = pd.DataFrame(per_series_rows)

    best_model = comparison_df.iloc[0]["model"]
    print("\nModel comparison (pooled over all series, test = last "
          f"{TEST_DAYS} days/series):")
    print(comparison_df.to_string(index=False))
    print(f"\nBest model: {best_model}")

    # --- persist ---
    forecast_cols = ["date", "series_id", "store_id", "category", "sales", *model_cols.values()]
    test[forecast_cols].to_csv(ARTIFACTS / "forecasts.csv", index=False)
    comparison_df.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    per_series_df.to_csv(ARTIFACTS / "per_series_metrics.csv", index=False)

    metrics = {
        "seed": SEED,
        "test_days": TEST_DAYS,
        "cutoff_date": str(cutoff.date()),
        "n_series": int(raw["store_id"].nunique() * raw["category"].nunique()),
        "n_rows_raw": int(len(raw)),
        "overall": overall,
        "best_model": best_model,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    bundle = {
        "type": "gbm" if best_model == "gbm" else ("holt_winters" if best_model == "holt_winters" else best_model),
        "gbm": gbm,
        "feature_columns": FEATURE_COLUMNS,
        "holt_winters_models": hw_models,
        "best_model": best_model,
    }
    joblib.dump(bundle, ARTIFACTS / "best_model.joblib")

    print(f"\nWrote artifacts to {ARTIFACTS} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
