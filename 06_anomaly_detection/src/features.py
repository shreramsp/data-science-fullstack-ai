"""CRISP-DM Data Preparation: cleaning, splitting, and feature assembly.

Shared by the training pipeline, the autoresearch loop, and the dashboard so
that a transaction is transformed identically everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler

SEED = 20260913
COMPONENT_COLS = [f"V{i}" for i in range(1, 29)]

# A "feature set" is one axis of the autoresearch search space.
FEATURE_SETS: dict[str, list[str]] = {
    "components": COMPONENT_COLS,
    "components_amount": COMPONENT_COLS + ["log_amount"],
    "components_amount_time": COMPONENT_COLS + ["log_amount", "hour_sin", "hour_cos"],
}

SCALERS = {
    "standard": lambda: StandardScaler(),
    "robust": lambda: RobustScaler(),
    "minmax": lambda: MinMaxScaler(),
    "quantile": lambda: QuantileTransformer(
        output_distribution="normal", n_quantiles=1000, subsample=100_000, random_state=SEED
    ),
}


@dataclass
class CleaningReport:
    raw_rows: int
    duplicates_dropped: int
    negative_amounts_fixed: int
    missing_amounts_imputed: int
    final_rows: int
    frauds: int

    @property
    def prevalence(self) -> float:
        return self.frauds / self.final_rows if self.final_rows else 0.0

    def to_rows(self) -> list[dict]:
        return [
            {"step": "raw rows loaded", "rows": self.raw_rows, "detail": ""},
            {"step": "exact duplicates dropped", "rows": -self.duplicates_dropped,
             "detail": "re-submitted/settlement echoes would double-count in scoring"},
            {"step": "negative amounts corrected", "rows": self.negative_amounts_fixed,
             "detail": "sign flipped to absolute value (refund artefacts)"},
            {"step": "missing amounts imputed", "rows": self.missing_amounts_imputed,
             "detail": "median amount; dropping rows would bias the minority class"},
            {"step": "rows retained for modelling", "rows": self.final_rows,
             "detail": f"{self.frauds} anomalies ({self.prevalence:.3%} prevalence)"},
        ]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the documented cleaning rules and report exactly what changed."""
    frame = raw.copy()
    raw_rows = len(frame)

    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    duplicates = before - len(frame)

    negatives = int((frame["Amount"] < 0).sum())
    frame.loc[frame["Amount"] < 0, "Amount"] = frame.loc[frame["Amount"] < 0, "Amount"].abs()

    missing = int(frame["Amount"].isna().sum())
    if missing:
        frame["Amount"] = frame["Amount"].fillna(frame["Amount"].median())

    report = CleaningReport(
        raw_rows=raw_rows,
        duplicates_dropped=duplicates,
        negative_amounts_fixed=negatives,
        missing_amounts_imputed=missing,
        final_rows=len(frame),
        frauds=int(frame["Class"].sum()),
    )
    return frame, report


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the feature sets can draw on."""
    out = frame.copy()
    out["log_amount"] = np.log1p(out["Amount"].clip(lower=0))
    hour = (out["Time"] % 86_400) / 3600.0
    # Cyclical encoding: 23:00 and 00:00 must be neighbours, not extremes.
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["hour"] = hour
    return out


def split(frame: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Stratified 60/20/20 train/validation/test split.

    Labels are *never* used to fit a detector -- the detectors here are
    unsupervised. Stratification only guarantees each split holds enough
    anomalies for its metrics to mean anything; validation drives the
    autoresearch search, and test is touched exactly once, at the end.
    """
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "valid": [], "test": []}
    for _, group in frame.groupby("Class"):
        idx = rng.permutation(len(group))
        n_train = int(round(0.6 * len(group)))
        n_valid = int(round(0.2 * len(group)))
        chunks = np.split(idx, [n_train, n_train + n_valid])
        for name, chunk in zip(("train", "valid", "test"), chunks):
            parts[name].append(group.iloc[chunk])
    return {
        name: pd.concat(frames).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for name, frames in parts.items()
    }


def matrix(frame: pd.DataFrame, feature_set: str) -> np.ndarray:
    return frame[FEATURE_SETS[feature_set]].to_numpy(dtype=float)


def make_scaler(name: str):
    return SCALERS[name]()
