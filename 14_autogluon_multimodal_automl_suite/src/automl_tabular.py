"""Thin wrapper around AutoGluon's TabularPredictor used for both the
multimodal (tabular + text) classification task and the quantile regression
task. AutoGluon detects the text column automatically via its built-in
feature generator; no manual NLP preprocessing is required.
"""
from __future__ import annotations

from pathlib import Path

from autogluon.tabular import TabularPredictor


def fit_classifier(train_df, label: str, save_path: Path, time_limit: int, eval_metric: str):
    predictor = TabularPredictor(
        label=label,
        problem_type="binary",
        eval_metric=eval_metric,
        path=str(save_path),
        verbosity=0,
    ).fit(
        train_df,
        presets="medium_quality",
        time_limit=time_limit,
        excluded_model_types=["XGB"],  # keep dependency footprint aligned with catboost-only install
    )
    return predictor


def fit_quantile_regressor(train_df, label: str, quantile_levels: list[float], save_path: Path, time_limit: int):
    predictor = TabularPredictor(
        label=label,
        problem_type="quantile",
        quantile_levels=quantile_levels,
        path=str(save_path),
        verbosity=0,
    ).fit(
        train_df,
        presets="medium_quality",
        time_limit=time_limit,
        excluded_model_types=["XGB"],
    )
    return predictor
