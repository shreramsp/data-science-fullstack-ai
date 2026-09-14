"""CRISP-DM Data Preparation phase: cleaning + feature engineering.

Turns raw transaction line items into:
  1. `customer_features` — one row per customer (RFM + diversity), used by
     clustering, anomaly detection, and supervised learning.
  2. `baskets` — one product list per invoice, used by association rule mining.
  3. `invoice_features` — one row per invoice, used by anomaly detection.
"""
from __future__ import annotations

import pandas as pd


def load_raw(path: str = "data/online_retail.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["invoice_date"])
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Minimal, honest cleaning report — this synthetic set has no nulls/dupes
    injected, but we still run the checks a real CRISP-DM prep step requires."""
    report = {"rows_in": int(len(df))}
    df = df.dropna(subset=["customer_id", "invoice_id", "product_id", "quantity", "unit_price"])
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    df = df.drop_duplicates()
    report["rows_out"] = int(len(df))
    report["rows_dropped"] = report["rows_in"] - report["rows_out"]
    return df, report


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["invoice_date"].max() + pd.Timedelta(days=1)
    grp = df.groupby("customer_id")
    feats = grp.agg(
        recency_days=("invoice_date", lambda s: (snapshot - s.max()).days),
        frequency=("invoice_id", "nunique"),
        monetary=("line_total", "sum"),
        avg_basket_value=("line_total", lambda s: s.sum() / df.loc[s.index, "invoice_id"].nunique()),
        product_diversity=("product_id", "nunique"),
        category_diversity=("category", "nunique"),
    ).reset_index()
    archetype = df.groupby("customer_id")["archetype"].first().reset_index()
    feats = feats.merge(archetype, on="customer_id", how="left")
    return feats


def build_invoice_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("invoice_id")
    feats = grp.agg(
        customer_id=("customer_id", "first"),
        invoice_date=("invoice_date", "first"),
        n_lines=("product_id", "nunique"),
        total_quantity=("quantity", "sum"),
        invoice_value=("line_total", "sum"),
    ).reset_index()
    feats["avg_unit_qty"] = feats["total_quantity"] / feats["n_lines"]
    return feats


def build_baskets(df: pd.DataFrame) -> list[list[str]]:
    return df.groupby("invoice_id")["product_id"].apply(lambda s: sorted(set(s))).tolist()
