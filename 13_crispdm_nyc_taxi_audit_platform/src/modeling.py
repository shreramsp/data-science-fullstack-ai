"""Modeling + evaluation: a small honest multi-model comparison.

The automated model research step trains a small, fixed set of
regressors with default-ish hyperparameters and picking the best by
held-out RMSE, rather than a long-running hyperparameter search. Every
metric reported is computed on the held-out test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN

SEED = 42

CANDIDATE_MODELS = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0, random_state=SEED),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED),
    "gradient_boosting": GradientBoostingRegressor(random_state=SEED),
}


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]


def _build_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def split_data(df: pd.DataFrame, seed: int = SEED):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_and_compare(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, ModelResult]:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    x_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]

    results: dict[str, ModelResult] = {}
    for name, estimator in CANDIDATE_MODELS.items():
        pipe = _build_pipeline(estimator)
        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "r2": float(r2_score(y_test, preds)),
        }
        results[name] = ModelResult(name=name, pipeline=pipe, metrics=metrics)
    return results


def select_best(results: dict[str, ModelResult]) -> str:
    return min(results, key=lambda k: results[k].metrics["rmse"])


def explain_with_permutation_importance(
    result: ModelResult, test_df: pd.DataFrame, n_repeats: int = 8, seed: int = SEED
) -> pd.DataFrame:
    """Model-agnostic explainability via permutation importance on the held-out
    set (honest, dependency-light substitute for a full TreeSHAP workbench)."""
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]
    r = permutation_importance(
        result.pipeline, x_test, y_test, n_repeats=n_repeats, random_state=seed, scoring="r2"
    )
    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return imp
