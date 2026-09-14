"""Seeded synthetic transaction log shaped like the UCI/Kaggle "Online Retail" dataset.

Why synthetic: the real Online Retail II workbook is a ~45 MB authenticated
download, which breaks `git clone && pip install && run`. This generator
reproduces the same *schema* (InvoiceNo, StockCode, Description, Quantity,
InvoiceDate, UnitPrice, CustomerID, Country) and the same *shape of defects*
(cancellations, missing customer ids, zero-price rows, duplicates) so the
CRISP-DM data-preparation phase has real work to do.

Customers are drawn from latent purchasing archetypes. The archetype labels are
written to a separate file and are NEVER used for fitting -- they exist only as
an external sanity check on the unsupervised result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260913
SNAPSHOT = datetime(2011, 12, 10)          # analysis reference date
WINDOW_DAYS = 730                          # two-year transaction history

# name -> (n_customers, orders/yr, days-since-last, basket size, unit price)
ARCHETYPES: dict[str, dict] = {
    "champion":      dict(n=170, freq=(14, 26), recency=(1, 25),   basket=(4, 9), price=(4.0, 14.0)),
    "loyal":         dict(n=300, freq=(7, 14),  recency=(10, 60),  basket=(3, 7), price=(2.5, 8.0)),
    "big_spender":   dict(n=140, freq=(2, 5),   recency=(15, 90),  basket=(6, 14), price=(9.0, 30.0)),
    "at_risk":       dict(n=220, freq=(6, 12),  recency=(150, 300), basket=(3, 6), price=(2.5, 9.0)),
    "hibernating":   dict(n=260, freq=(1, 3),   recency=(240, 600), basket=(1, 4), price=(1.5, 6.0)),
    "new_shopper":   dict(n=190, freq=(1, 2),   recency=(1, 40),   basket=(1, 4), price=(2.0, 7.0)),
}

COUNTRIES = ["United Kingdom"] * 12 + ["Germany", "France", "EIRE", "Spain",
                                       "Netherlands", "Belgium", "Portugal", "Australia"]

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER"), ("71053", "WHITE METAL LANTERN"),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER"),     ("22423", "REGENCY CAKESTAND 3 TIER"),
    ("47566", "PARTY BUNTING"),                        ("20725", "LUNCH BAG RED RETROSPOT"),
    ("22383", "LUNCH BAG SUKI DESIGN"),                ("21212", "PACK OF 72 RETROSPOT CAKE CASES"),
    ("22960", "JAM MAKING SET WITH JARS"),             ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS"),
    ("23203", "JUMBO BAG VINTAGE DOILY"),              ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN"),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER"),    ("22457", "NATURAL SLATE HEART CHALKBOARD"),
    ("23245", "SET OF 3 REGENCY CAKE TINS"),           ("22138", "BAKING SET 9 PIECE RETROSPOT"),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT"),        ("22197", "POPCORN HOLDER"),
    ("23298", "SPOTTY BUNTING"),                       ("21931", "JUMBO STORAGE BAG SUKI"),
]


def generate(out_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    truth: list[dict] = []
    customer_id = 12346
    invoice_no = 536365          # globally unique, like the real dataset

    for name, cfg in ARCHETYPES.items():
        for _ in range(cfg["n"]):
            cid = customer_id
            customer_id += 1
            country = COUNTRIES[rng.integers(len(COUNTRIES))]

            orders_per_year = rng.uniform(*cfg["freq"])
            recency_days = float(rng.uniform(*cfg["recency"]))
            last_order = SNAPSHOT - timedelta(days=recency_days)
            # active tenure: how far back this customer's history reaches
            tenure = min(WINDOW_DAYS - recency_days, rng.uniform(60, WINDOW_DAYS))
            n_orders = max(1, int(round(orders_per_year * max(tenure, 30) / 365.0)))

            # spread the remaining orders backwards from the last one
            gaps = np.sort(rng.uniform(0, max(tenure, 1.0), size=n_orders - 1))[::-1]
            order_dates = [last_order] + [last_order - timedelta(days=float(g)) for g in gaps]

            price_lo, price_hi = cfg["price"]
            for oi, odate in enumerate(order_dates):
                invoice = str(invoice_no)
                invoice_no += 1
                odate = odate + timedelta(hours=float(rng.uniform(7, 20)),
                                          minutes=float(rng.integers(0, 60)))
                n_lines = max(1, int(rng.integers(cfg["basket"][0], cfg["basket"][1] + 1)))
                for _ in range(n_lines):
                    stock, desc = PRODUCTS[rng.integers(len(PRODUCTS))]
                    rows.append(dict(
                        InvoiceNo=invoice, StockCode=stock, Description=desc,
                        Quantity=int(max(1, rng.poisson(6))),
                        InvoiceDate=odate,
                        UnitPrice=round(float(rng.uniform(price_lo, price_hi)), 2),
                        CustomerID=float(cid), Country=country,
                    ))
                # ~4% of orders are later cancelled -> negative-quantity 'C' invoice
                if rng.random() < 0.04 and oi > 0:
                    last = rows[-1]
                    rows.append({**last,
                                 "InvoiceNo": f"C{invoice}",
                                 "Quantity": -last["Quantity"],
                                 "InvoiceDate": odate + timedelta(days=float(rng.uniform(1, 10)))})
            truth.append(dict(CustomerID=float(cid), archetype=name))

    df = pd.DataFrame(rows)

    # --- inject the defects the real dataset is famous for -------------------
    n = len(df)
    guest = rng.choice(n, size=int(0.035 * n), replace=False)          # missing ids
    df.loc[guest, "CustomerID"] = np.nan
    free = rng.choice(n, size=int(0.004 * n), replace=False)           # zero-price rows
    df.loc[free, "UnitPrice"] = 0.0
    dupes = df.sample(n=int(0.01 * n), random_state=SEED)              # duplicated lines
    df = pd.concat([df, dupes], ignore_index=True)

    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.floor("min")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "online_retail.csv", index=False)
    pd.DataFrame(truth).to_csv(out_dir / "customer_archetypes.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()
    frame = generate(Path(args.out))
    print(f"rows            : {len(frame):,}")
    print(f"customers       : {frame['CustomerID'].nunique():,}")
    print(f"invoices        : {frame['InvoiceNo'].nunique():,}")
    print(f"date range      : {frame['InvoiceDate'].min()} -> {frame['InvoiceDate'].max()}")
    print(f"missing CustomerID: {frame['CustomerID'].isna().sum():,}")
    print(f"cancellations   : {frame['InvoiceNo'].astype(str).str.startswith('C').sum():,}")
