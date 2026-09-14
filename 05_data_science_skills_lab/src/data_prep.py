"""Shared CRISP-DM data-preparation layer used by every skill in the lab.

Every skill executor receives the same prepared frame, so the lab demonstrates
skills against one consistent, honestly-documented view of the data rather than
46 ad-hoc re-cleanings.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load_data import data_source, load_raw  # noqa: E402

TARGET = "Churn"
SEED = 42

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
ID_COL = "customerID"
# Reference "today" for turning the tenure column into calendar months, so the
# cohort / time-series skills have a real date axis. Stated openly in the UI.
ANCHOR_MONTH = pd.Timestamp("2024-12-01")


@dataclass
class QualityIssue:
    column: str
    issue: str
    rows: int
    action: str


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[QualityIssue]]:
    """Apply the documented cleaning rules and return the issue log with them."""
    df = raw.copy()
    issues: list[QualityIssue] = []

    # TotalCharges ships as an object column with blank strings for customers
    # whose tenure is 0 (they have not been billed yet).
    coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blanks = int(coerced.isna().sum())
    if blanks:
        issues.append(
            QualityIssue(
                "TotalCharges",
                "non-numeric/blank values (tenure-0 accounts, never billed)",
                blanks,
                "coerced to NaN, then imputed as tenure x MonthlyCharges (= 0)",
            )
        )
    df["TotalCharges"] = coerced.fillna(df["tenure"] * df["MonthlyCharges"])

    dupes = int(df.duplicated(subset=[ID_COL]).sum())
    if dupes:
        issues.append(QualityIssue(ID_COL, "duplicate customer IDs", dupes,
                                   "dropped, keeping first occurrence"))
        df = df.drop_duplicates(subset=[ID_COL], keep="first")

    for col in ("MonthlyCharges", "TotalCharges", "tenure"):
        neg = int((df[col] < 0).sum())
        if neg:
            issues.append(QualityIssue(col, "negative values", neg, "dropped"))
            df = df[df[col] >= 0]

    missing = int(df.isna().sum().sum())
    if missing:
        issues.append(QualityIssue("(any)", "residual missing cells", missing,
                                   "left for the model pipeline's imputer"))

    return df.reset_index(drop=True), issues


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the features the analysis and modeling skills share."""
    out = df.copy()
    out["churn_flag"] = (out[TARGET] == "Yes").astype(int)
    out["avg_monthly_spend"] = np.where(
        out["tenure"] > 0, out["TotalCharges"] / out["tenure"], out["MonthlyCharges"]
    ).round(2)
    out["tenure_years"] = (out["tenure"] / 12).round(2)
    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[-0.1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)
    out["spend_tier"] = pd.qcut(
        out["MonthlyCharges"], q=4, labels=["Q1 low", "Q2", "Q3", "Q4 high"]
    ).astype(str)

    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
    out["addon_count"] = sum((out[c] == "Yes").astype(int) for c in addons)
    out["has_protection"] = (
        (out["OnlineSecurity"] == "Yes") | (out["TechSupport"] == "Yes")
    ).astype(int)
    out["is_autopay"] = out["PaymentMethod"].str.contains("automatic").astype(int)

    # Calendar signup month implied by tenure, for cohort/time-series skills.
    out["signup_month"] = (
        ANCHOR_MONTH - pd.to_timedelta(out["tenure"] * 30, unit="D")
    ).dt.to_period("M").dt.to_timestamp()
    return out


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Model input columns: (numeric, categorical). Leak-prone columns excluded."""
    drop = {ID_COL, TARGET, "churn_flag", "signup_month"}
    numeric = [c for c in df.columns
               if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns
                   if c not in drop and c not in numeric and df[c].nunique() <= 12]
    return numeric, categorical


_CACHE: dict[str, object] = {}


def get_data(force: bool = False) -> dict:
    """Load → clean → engineer once, and memoize for the session."""
    if "bundle" in _CACHE and not force:
        return _CACHE["bundle"]  # type: ignore[return-value]

    raw = load_raw()
    cleaned, issues = clean(raw)
    df = engineer(cleaned)
    bundle = {
        "raw": raw,
        "df": df,
        "issues": issues,
        "source": data_source(),
        "rows_raw": len(raw),
        "rows_clean": len(df),
    }
    _CACHE["bundle"] = bundle
    return bundle
