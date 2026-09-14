"""Thin wrapper around ``TabularPredictor`` fixing everything an AutoGluon run
needs except the two knobs AutoResearch is searching over: bagged-fold count
and stack depth.

Model family is fixed to CatBoost + RandomForest + ExtraTrees. LightGBM is
AutoGluon's usual default top model but needs the system OpenMP runtime
(``libomp``) that is not present on a stock macOS/Python install; rather than
require a Homebrew step outside this project, it is left out so the pipeline
runs unmodified on a clean checkout. XGBoost is left out for the same
"clean checkout" reason: its sdist needs ``cmake`` to build on Python 3.13/
arm64 when no prebuilt wheel is published yet.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

HYPERPARAMETERS = {"CAT": {}, "RF": {}, "XT": {}}


def fit_predictor(
    train_df: pd.DataFrame,
    label: str,
    problem_type: str,
    eval_metric: str,
    path: Path,
    num_bag_folds: int,
    num_stack_levels: int,
    time_limit: int,
) -> TabularPredictor:
    shutil.rmtree(path, ignore_errors=True)
    predictor = TabularPredictor(
        label=label,
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=str(path),
        verbosity=0,
    )
    fit_kwargs = dict(
        time_limit=time_limit,
        presets="medium_quality",
        hyperparameters=HYPERPARAMETERS,
        num_bag_folds=num_bag_folds,
    )
    if num_bag_folds > 0:
        fit_kwargs["num_stack_levels"] = num_stack_levels
    predictor.fit(train_df, **fit_kwargs)
    return predictor


def best_val_score(predictor: TabularPredictor) -> float:
    lb = predictor.leaderboard(silent=True)
    return float(lb.sort_values("score_val", ascending=False).iloc[0]["score_val"])
