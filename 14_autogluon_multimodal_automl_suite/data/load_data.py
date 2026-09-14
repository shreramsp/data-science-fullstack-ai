"""Task registry for the three AutoGluon capabilities this suite illustrates:

1. multimodal_reviews  -- tabular + free-text columns together (AutoGluon's
   automatic text feature generator), binary classification.
2. diabetes_quantile   -- tabular regression solved as multi-quantile
   regression (P10/P50/P90 bands instead of a single point estimate).
3. retail_demand       -- multi-series time series forecasting via
   ``autogluon.timeseries.TimeSeriesPredictor``.

The tabular tasks use a scikit-learn built-in (no download) or a small
deterministic synthetic generator; the time series task is generated
synthetically since AutoGluon's TimeSeriesPredictor needs multiple related
series and no bundled small multi-series dataset ships with sklearn. All
generators are seeded for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

SEED = 42

# ---------------------------------------------------------------------------
# Task 1: multimodal (tabular + text) binary classification
# ---------------------------------------------------------------------------

_POS_PHRASES = [
    "works perfectly and arrived early",
    "excellent build quality, would buy again",
    "exceeded my expectations for the price",
    "very durable and easy to use",
    "fantastic value, highly recommend",
    "customer service was fast and helpful",
    "fits perfectly and looks great",
    "battery life is outstanding",
]
_NEG_PHRASES = [
    "stopped working after two days",
    "cheaply made and broke quickly",
    "not as described, very disappointed",
    "arrived damaged and support was unhelpful",
    "poor build quality for the price",
    "difficult to set up and unreliable",
    "battery drains far too fast",
    "would not recommend this product",
]
_CATEGORIES = ["electronics", "home", "sports", "toys", "kitchen"]
_SHIPPING = ["standard", "expedited", "two_day"]


def _multimodal_reviews(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    quality = rng.normal(0, 1, size=n)  # latent product quality
    price = np.round(rng.uniform(8, 250, size=n), 2)
    category = rng.choice(_CATEGORIES, size=n)
    shipping = rng.choice(_SHIPPING, size=n)
    rating = np.clip(np.round(3 + quality + rng.normal(0, 0.6, size=n)), 1, 5).astype(int)

    texts = []
    for q in quality:
        n_phrases = rng.integers(1, 3)
        pool = _POS_PHRASES if q + rng.normal(0, 0.5) > 0 else _NEG_PHRASES
        texts.append(". ".join(rng.choice(pool, size=n_phrases, replace=False)) + ".")

    recommended = ((quality + 0.4 * (rating - 3) + rng.normal(0, 0.5, size=n)) > 0).astype(int)
    recommended = np.where(recommended == 1, "yes", "no")

    return pd.DataFrame(
        {
            "review_text": texts,
            "rating": rating,
            "price": price,
            "category": category,
            "shipping_method": shipping,
            "recommended": recommended,
        }
    )


# ---------------------------------------------------------------------------
# Task 2: quantile regression
# ---------------------------------------------------------------------------

def _diabetes_quantile() -> pd.DataFrame:
    d = load_diabetes(as_frame=True)
    return d.frame.rename(columns={"target": "disease_progression"})


# ---------------------------------------------------------------------------
# Task 3: multi-series time series forecasting
# ---------------------------------------------------------------------------

def _retail_demand(n_items: int = 6, n_days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for item in range(n_items):
        base = rng.uniform(40, 120)
        weekly_amp = rng.uniform(5, 20)
        trend = rng.uniform(-0.05, 0.15)
        phase = rng.uniform(0, 2 * np.pi)
        t = np.arange(n_days)
        seasonal = weekly_amp * np.sin(2 * np.pi * t / 7 + phase)
        noise = rng.normal(0, 4, size=n_days)
        demand = np.clip(base + trend * t + seasonal + noise, 0, None).round(1)
        rows.append(
            pd.DataFrame(
                {
                    "item_id": f"item_{item}",
                    "timestamp": dates,
                    "demand": demand,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


TASKS = {
    "multimodal_reviews": {
        "display_name": "Multimodal Product Review Classifier",
        "kind": "tabular",
        "problem_type": "binary",
        "label": "recommended",
        "eval_metric": "roc_auc",
        "loader": _multimodal_reviews,
        "source": "Deterministic synthetic generator (text + tabular features, seed=42)",
    },
    "diabetes_quantile": {
        "display_name": "Diabetes Progression Quantile Regression",
        "kind": "tabular_quantile",
        "problem_type": "quantile",
        "label": "disease_progression",
        "quantile_levels": [0.1, 0.5, 0.9],
        "eval_metric": "pinball_loss",
        "loader": _diabetes_quantile,
        "source": "sklearn.datasets.load_diabetes (442 rows, 10 features)",
    },
    "retail_demand": {
        "display_name": "Multi-Series Retail Demand Forecast",
        "kind": "timeseries",
        "id_column": "item_id",
        "timestamp_column": "timestamp",
        "target": "demand",
        "prediction_length": 14,
        "eval_metric": "MASE",
        "loader": _retail_demand,
        "source": "Deterministic synthetic generator (6 items x 180 days, seasonal + trend, seed=42)",
    },
}


def load(task_key: str):
    task = TASKS[task_key]
    return task["loader"](), task
