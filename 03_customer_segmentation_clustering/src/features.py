"""CRISP-DM Data Preparation: transaction log -> per-customer feature matrix.

Shared by the training pipeline and the dashboard so both see identical data.
The RFM construction follows Hughes (1994), "Strategic Database Marketing":
Recency = days since last purchase, Frequency = number of purchase occasions,
Monetary = total spend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_CSV = DATA_DIR / "online_retail.csv"

RFM_FEATURES = ["recency_days", "frequency", "monetary"]
EXTENDED_FEATURES = RFM_FEATURES + ["avg_order_value", "avg_basket_size",
                                    "tenure_days", "return_rate"]
FEATURE_SETS = {"rfm": RFM_FEATURES, "rfm_extended": EXTENDED_FEATURES}


@dataclass
class CleaningReport:
    """Row-level audit trail for the Data Preparation phase."""
    rows_raw: int = 0
    rows_final: int = 0
    steps: list[tuple[str, int]] = field(default_factory=list)

    def log(self, label: str, dropped: int) -> None:
        self.steps.append((label, int(dropped)))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps, columns=["step", "rows_removed"])


def load_raw(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Load the transaction log, generating it on first use."""
    path = Path(path)
    if not path.exists():
        import sys
        sys.path.insert(0, str(DATA_DIR))
        from generate_data import generate  # type: ignore
        generate(DATA_DIR)
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    return df


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the standard Online-Retail cleaning rules, auditing every drop."""
    rep = CleaningReport(rows_raw=len(df))

    before = len(df)
    df = df.drop_duplicates()
    rep.log("exact duplicate lines", before - len(df))

    before = len(df)
    df = df[df["CustomerID"].notna()]
    rep.log("missing CustomerID (guest checkout)", before - len(df))

    # cancellations are real events: net them out rather than silently dropping
    cancels = df["InvoiceNo"].str.startswith("C")
    rep.log("cancellation lines kept as negative revenue", 0)

    before = len(df)
    df = df[~((df["Quantity"] <= 0) & (~cancels))]
    rep.log("non-cancellation rows with Quantity <= 0", before - len(df))

    before = len(df)
    df = df[df["UnitPrice"] > 0]
    rep.log("zero / negative UnitPrice", before - len(df))

    df = df.copy()
    df["line_revenue"] = df["Quantity"] * df["UnitPrice"]
    df["is_return"] = df["InvoiceNo"].str.startswith("C")
    rep.rows_final = len(df)
    return df, rep


def build_customer_features(df: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Aggregate cleaned transactions into one row per customer."""
    if snapshot is None:
        snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    sales = df[~df["is_return"]]
    g = sales.groupby("CustomerID")

    feat = pd.DataFrame({
        "recency_days": (snapshot - g["InvoiceDate"].max()).dt.total_seconds() / 86400.0,
        "frequency": g["InvoiceNo"].nunique(),
        "monetary": g["line_revenue"].sum(),
        "n_lines": g.size(),
        "tenure_days": (snapshot - g["InvoiceDate"].min()).dt.total_seconds() / 86400.0,
        "distinct_products": g["StockCode"].nunique(),
    })
    feat["avg_order_value"] = feat["monetary"] / feat["frequency"]
    feat["avg_basket_size"] = feat["n_lines"] / feat["frequency"]

    returned = df[df["is_return"]].groupby("CustomerID")["InvoiceNo"].nunique()
    feat["n_returns"] = returned.reindex(feat.index).fillna(0.0)
    feat["return_rate"] = feat["n_returns"] / feat["frequency"]

    feat["country"] = df.groupby("CustomerID")["Country"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown")

    # a customer whose only activity is a refund has no positive spend to segment on
    feat = feat[feat["monetary"] > 0]
    return feat.round(4)


def rfm_scores(feat: pd.DataFrame) -> pd.DataFrame:
    """Classic 1-5 RFM quintile scores -- the interpretable baseline the
    unsupervised clustering is compared against."""
    out = pd.DataFrame(index=feat.index)
    out["R_score"] = pd.qcut(feat["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F_score"] = pd.qcut(feat["frequency"].rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)
    out["M_score"] = pd.qcut(feat["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["RFM_score"] = out[["R_score", "F_score", "M_score"]].sum(axis=1)
    return out


def prepare(path: Path | str = RAW_CSV) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Full preparation pass: raw -> cleaned transactions -> customer features."""
    raw = load_raw(path)
    clean, report = clean_transactions(raw)
    feat = build_customer_features(clean)
    feat = feat.join(rfm_scores(feat))
    return clean, feat, report


def matrix(feat: pd.DataFrame, feature_set: str, log_transform: bool) -> np.ndarray:
    """Numeric design matrix for a search configuration."""
    cols = FEATURE_SETS[feature_set]
    X = feat[cols].to_numpy(dtype=float)
    if log_transform:
        # recency/frequency/monetary are strongly right-skewed; log1p is the
        # standard remedy and is safe because all columns are non-negative.
        X = np.log1p(np.clip(X, 0.0, None))
    return X
