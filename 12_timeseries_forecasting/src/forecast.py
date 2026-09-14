"""Forward forecasting helper used by the Streamlit app.

Builds a horizon of future dates for one series with whichever model won
training (`artifacts/best_model.joblib`). The GBM path is recursive: each
future day's lag/rolling features are computed from actuals plus the
model's own prior predictions, since real future lags do not exist yet —
unlike the train.py evaluation, which uses true historical lags throughout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import LAGS, ROLL_WINDOWS  # noqa: E402


def _calendar_features(date: pd.Timestamp, promo: int) -> dict:
    return {
        "day_of_week": date.dayofweek,
        "month": date.month,
        "day_of_year": date.dayofyear,
        "is_weekend": int(date.dayofweek >= 5),
        "is_holiday_season": int(date.month in (11, 12)),
        "promo": promo,
    }


def forecast_series(raw: pd.DataFrame, bundle: dict, series_id: str, horizon: int) -> pd.DataFrame:
    if "series_id" not in raw.columns:
        raw = raw.assign(series_id=raw["store_id"] + "_" + raw["category"])
    history = raw[raw["series_id"] == series_id].sort_values("date")
    last_date = history["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    model_type = bundle["best_model"]
    if model_type == "holt_winters":
        model = bundle["holt_winters_models"][series_id]
        preds = np.clip(model.forecast(horizon), 0, None)
        return pd.DataFrame({"date": future_dates, "series_id": series_id, "pred": preds})

    if model_type == "gbm":
        gbm = bundle["gbm"]
        cols = bundle["feature_columns"]
        values = history["sales"].tolist()
        preds = []
        for d in future_dates:
            row = _calendar_features(d, promo=0)
            for lag in LAGS:
                row[f"lag_{lag}"] = values[-lag]
            for win in ROLL_WINDOWS:
                row[f"roll_mean_{win}"] = float(np.mean(values[-win:]))
                row[f"roll_std_{win}"] = float(np.std(values[-win:])) if len(values) > 1 else 0.0
            x = pd.DataFrame([row])[cols]
            yhat = max(0.0, float(gbm.predict(x)[0]))
            preds.append(yhat)
            values.append(yhat)
        return pd.DataFrame({"date": future_dates, "series_id": series_id, "pred": preds})

    # naive / seasonal_naive fallback
    lag = 1 if model_type == "naive" else 7
    tail = history["sales"].to_numpy()[-lag:]
    reps = int(np.ceil(horizon / lag))
    preds = np.tile(tail, reps)[:horizon]
    return pd.DataFrame({"date": future_dates, "series_id": series_id, "pred": preds})
