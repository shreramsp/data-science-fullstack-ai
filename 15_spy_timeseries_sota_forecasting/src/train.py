"""CRISP-DM pipeline for the SPY-like probabilistic forecasting tournament.

prepare data -> engineer leakage-safe features -> chronological split with
an embargo -> fit a point-forecast model tournament per horizon -> fit
P10/P50/P90 quantile models with the winning algorithm family -> evaluate
(MAE/RMSE/directional hit rate, pinball loss, band coverage) -> run the
long/flat trading backtest on the horizon-1 median forecast -> persist
artifacts for the dashboard.

Run:  python src/train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import run_backtest, summarize  # noqa: E402
from features import FEATURE_COLUMNS, HORIZONS, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
SEED = 20260913
QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


def directional_hit_rate(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.sign(actual) == np.sign(pred)))


def point_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(mean_squared_error(actual, pred) ** 0.5),
        "directional_hit_rate": directional_hit_rate(actual, pred),
    }


def pinball_loss(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = actual - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def main() -> None:
    t0 = time.time()
    ARTIFACTS.mkdir(exist_ok=True)

    raw, feat, train, test, cutoff = prepare()
    print(f"Rows: raw={len(raw):,} feature_table={len(feat):,} train={len(train):,} "
          f"test={len(test):,} split_cutoff={cutoff.date()}")

    scaler = RobustScaler().fit(train[FEATURE_COLUMNS])
    x_train = pd.DataFrame(scaler.transform(train[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS)
    x_test = pd.DataFrame(scaler.transform(test[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS)

    comparison_rows = []
    forecast_frames = []
    quantile_models: dict[int, dict[str, HistGradientBoostingRegressor]] = {}
    point_models: dict[int, HistGradientBoostingRegressor] = {}
    quantile_metrics: dict[str, dict] = {}
    feature_importance_rows = []

    for h in HORIZONS:
        y_train = train[f"fwd_return_{h}"].to_numpy()
        y_test = test[f"fwd_return_{h}"].to_numpy()

        naive_pred = np.zeros_like(y_test)  # random-walk: expect zero drift over the horizon
        ridge = Ridge(alpha=5.0, random_state=SEED).fit(x_train, y_train)
        ridge_pred = ridge.predict(x_test)
        gbm = HistGradientBoostingRegressor(random_state=SEED, max_depth=4, max_iter=250, learning_rate=0.05)
        gbm.fit(x_train, y_train)
        gbm_pred = gbm.predict(x_test)
        point_models[h] = gbm

        candidates = {"naive": naive_pred, "ridge": ridge_pred, "gbm": gbm_pred}
        for name, pred in candidates.items():
            m = point_metrics(y_test, pred)
            comparison_rows.append({"horizon": h, "model": name, **m})

        winner = min(candidates, key=lambda n: point_metrics(y_test, candidates[n])["mae"])
        print(f"Horizon t+{h}: winner={winner} "
              f"mae={point_metrics(y_test, candidates[winner])['mae']:.5f}")

        # --- quantile regression with the winning algorithm family (gbm) ---
        q_models = {}
        q_preds = {}
        for label, q in QUANTILES.items():
            qm = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, random_state=SEED, max_depth=4, max_iter=250, learning_rate=0.05,
            )
            qm.fit(x_train, y_train)
            q_models[label] = qm
            q_preds[label] = qm.predict(x_test)
        quantile_models[h] = q_models

        coverage = float(np.mean((y_test >= q_preds["p10"]) & (y_test <= q_preds["p90"])))
        pinballs = {label: pinball_loss(y_test, q_preds[label], q) for label, q in QUANTILES.items()}
        quantile_metrics[f"h{h}"] = {
            "coverage_p10_p90": coverage, "target_coverage": 0.80, "pinball_loss": pinballs,
        }

        frame = test[["date"]].copy()
        frame["horizon"] = h
        frame["actual_return"] = y_test
        frame["pred_naive"] = naive_pred
        frame["pred_ridge"] = ridge_pred
        frame["pred_gbm"] = gbm_pred
        for label in QUANTILES:
            frame[f"pred_{label}"] = q_preds[label]
        forecast_frames.append(frame)

        if h == 1:
            perm = permutation_importance(
                gbm, x_test, y_test, n_repeats=10, random_state=SEED, scoring="neg_mean_absolute_error",
            )
            for col, imp in zip(FEATURE_COLUMNS, perm.importances_mean):
                feature_importance_rows.append({"feature": col, "importance": float(imp)})

    comparison_df = pd.DataFrame(comparison_rows)
    forecasts_df = pd.concat(forecast_frames, ignore_index=True)
    importance_df = pd.DataFrame(feature_importance_rows).sort_values("importance", ascending=False)

    # --- trading backtest on the horizon-1 median forecast ---
    h1 = forecasts_df[forecasts_df["horizon"] == 1].reset_index(drop=True)
    bt = run_backtest(h1["actual_return"].to_numpy(), h1["pred_p50"].to_numpy())
    bt["date"] = h1["date"]
    bt_summary = summarize(bt)

    comparison_df.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    forecasts_df.to_csv(ARTIFACTS / "forecasts.csv", index=False)
    importance_df.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    bt.to_csv(ARTIFACTS / "backtest_results.csv", index=False)

    metrics = {
        "seed": SEED,
        "n_rows_raw": int(len(raw)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "split_cutoff_date": str(cutoff.date()),
        "embargo_days": 5,
        "quantile_metrics": quantile_metrics,
        "backtest": bt_summary,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    bundle = {
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "point_models": point_models,
        "quantile_models": quantile_models,
    }
    joblib.dump(bundle, ARTIFACTS / "best_model.joblib")

    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nBacktest (strategy) sharpe={bt_summary['strategy']['sharpe_ratio']:.2f} "
          f"max_dd={bt_summary['strategy']['max_drawdown']:.3f} "
          f"hit_rate={bt_summary['directional_hit_rate']:.3f}")
    print(f"\nWrote artifacts to {ARTIFACTS} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
