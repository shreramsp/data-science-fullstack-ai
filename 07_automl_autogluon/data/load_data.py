"""Task registry: three small, deterministic tabular tasks spanning the three
problem types AutoGluon's tabular predictor targets (binary / multiclass /
regression). All three ship inside scikit-learn, so there is no download step
and no risk of a missing-file surprise for whoever runs this next.

California Housing (20,640 rows) is downsampled to a fixed 3,000-row seed so
that bagged, multi-level AutoGluon fits stay laptop-fast; the other two are
used whole (569 and 178 rows).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_wine

SEED = 42


@dataclass(frozen=True)
class Task:
    key: str
    display_name: str
    problem_type: str  # 'binary' | 'multiclass' | 'regression'
    label: str
    eval_metric: str
    loader: Callable[[], pd.DataFrame]
    source: str


def _breast_cancer() -> pd.DataFrame:
    d = load_breast_cancer(as_frame=True)
    df = d.frame.rename(columns={"target": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].map({0: "malignant", 1: "benign"})
    return df


def _wine() -> pd.DataFrame:
    d = load_wine(as_frame=True)
    df = d.frame.rename(columns={"target": "cultivar"})
    df["cultivar"] = df["cultivar"].map({0: "cultivar_0", 1: "cultivar_1", 2: "cultivar_2"})
    return df


def _california_housing() -> pd.DataFrame:
    d = fetch_california_housing(as_frame=True)
    return d.frame.sample(n=3000, random_state=SEED).reset_index(drop=True)


TASKS: dict[str, Task] = {
    "breast_cancer": Task(
        key="breast_cancer",
        display_name="Breast Cancer Diagnosis",
        problem_type="binary",
        label="diagnosis",
        eval_metric="roc_auc",
        loader=_breast_cancer,
        source="sklearn.datasets.load_breast_cancer (569 rows, 30 features)",
    ),
    "wine": Task(
        key="wine",
        display_name="Wine Cultivar Classification",
        problem_type="multiclass",
        label="cultivar",
        eval_metric="log_loss",
        loader=_wine,
        source="sklearn.datasets.load_wine (178 rows, 13 features, 3 classes)",
    ),
    "california_housing": Task(
        key="california_housing",
        display_name="California Housing Price",
        problem_type="regression",
        label="MedHouseVal",
        eval_metric="root_mean_squared_error",
        loader=_california_housing,
        source="sklearn.datasets.fetch_california_housing, seeded 3,000-row sample of 20,640",
    ),
}


def load(task_key: str) -> tuple[pd.DataFrame, Task]:
    task = TASKS[task_key]
    return task.loader(), task
