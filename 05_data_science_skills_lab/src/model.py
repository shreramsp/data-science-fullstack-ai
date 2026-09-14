"""Shared modeling pipeline for the Modeling / Evaluation CRISP-DM skills.

One leakage-safe pipeline is fitted once per session and reused by every skill
that needs a model, so the numbers shown across the lab are mutually
consistent and all come from the same held-out split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_prep import SEED, feature_columns

_CACHE: dict = {}


def build_pipeline(estimator=None, numeric=(), categorical=()) -> Pipeline:
    """Preprocessing + estimator in one object, so every transform is fitted
    on training folds only (no leakage from the test split)."""
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), list(numeric)),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             list(categorical)),
        ],
        remainder="drop",
    )
    est = (LogisticRegression(max_iter=2000, random_state=SEED)
           if estimator is None else estimator)
    return Pipeline([("pre", pre), ("model", est)])


def score_all(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def get_model(df: pd.DataFrame, force: bool = False) -> dict:
    """Fit (or return the cached) baseline logistic-regression churn model."""
    if "baseline" in _CACHE and not force:
        return _CACHE["baseline"]

    numeric, categorical = feature_columns(df)
    X = df[numeric + categorical]
    y = df["churn_flag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    pipe = build_pipeline(numeric=numeric, categorical=categorical)
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    bundle = {
        "pipeline": pipe,
        "numeric": numeric,
        "categorical": categorical,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "prob": prob, "pred": pred,
        "metrics": score_all(y_test, pred, prob),
        "confusion": confusion_matrix(y_test, pred),
        "feature_names": list(
            pipe.named_steps["pre"].get_feature_names_out()
        ),
    }
    _CACHE["baseline"] = bundle
    return bundle


def random_forest(df: pd.DataFrame, **kwargs) -> Pipeline:
    numeric, categorical = feature_columns(df)
    est = RandomForestClassifier(
        n_estimators=kwargs.pop("n_estimators", 200),
        random_state=SEED, n_jobs=-1, **kwargs
    )
    return build_pipeline(est, numeric, categorical)


def coefficient_table(model_bundle: dict, top: int = 12) -> pd.DataFrame:
    """Signed logistic-regression coefficients as odds ratios."""
    coefs = model_bundle["pipeline"].named_steps["model"].coef_[0]
    names = [n.split("__", 1)[-1] for n in model_bundle["feature_names"]]
    tbl = pd.DataFrame({"feature": names, "coefficient": coefs})
    tbl["odds_ratio"] = np.exp(tbl["coefficient"]).round(3)
    tbl["abs"] = tbl["coefficient"].abs()
    return (tbl.sort_values("abs", ascending=False)
            .head(top).drop(columns="abs")
            .round({"coefficient": 3}).reset_index(drop=True))
