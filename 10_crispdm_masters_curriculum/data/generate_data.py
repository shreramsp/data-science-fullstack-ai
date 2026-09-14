"""Reproducible synthetic online-retail transaction generator.

Not a scrape or copy of any Kaggle file. Structured to exercise every CRISP-DM
modeling phase this project demonstrates:
  - customer RFM segments -> clustering + supervised churn-risk labels
  - a handful of injected bulk-order invoices -> anomaly detection
  - correlated product pairs (bundles) -> association rule mining
  - a large product catalog with co-purchase structure -> LSH similarity search

Run directly to (re)write data/online_retail.csv:
    python3.13 data/generate_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_CUSTOMERS = 500
N_PRODUCTS = 300
N_INVOICES = 1500
CATEGORIES = [
    "Home & Kitchen", "Stationery", "Electronics Accessories", "Toys",
    "Garden", "Apparel", "Sports", "Beauty",
]
COUNTRIES = ["United Kingdom", "Germany", "France", "Ireland", "Spain", "Netherlands"]

# Customer archetypes: (share of population, visits/yr, spend multiplier, recency bias)
ARCHETYPES = {
    "champion": dict(share=0.12, freq_lambda=14, spend_mult=2.4, recency_bias=0.15),
    "loyal": dict(share=0.28, freq_lambda=8, spend_mult=1.4, recency_bias=0.35),
    "regular": dict(share=0.35, freq_lambda=4, spend_mult=1.0, recency_bias=0.55),
    "at_risk": dict(share=0.18, freq_lambda=2, spend_mult=0.8, recency_bias=0.85),
    "dormant": dict(share=0.07, freq_lambda=1, spend_mult=0.6, recency_bias=0.97),
}


def _make_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for pid in range(N_PRODUCTS):
        cat = rng.choice(CATEGORIES)
        base_price = rng.gamma(shape=2.0, scale=6.0) + 1.5
        rows.append(
            {
                "product_id": f"P{pid:04d}",
                "product_name": f"{cat.split()[0]} Item {pid:04d}",
                "category": cat,
                "unit_price": round(base_price, 2),
            }
        )
    return pd.DataFrame(rows)


def _make_customers(rng: np.random.Generator) -> pd.DataFrame:
    names = list(ARCHETYPES.keys())
    shares = [ARCHETYPES[n]["share"] for n in names]
    archetype = rng.choice(names, size=N_CUSTOMERS, p=shares)
    country = rng.choice(COUNTRIES, size=N_CUSTOMERS, p=[0.45, 0.15, 0.15, 0.1, 0.08, 0.07])
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(N_CUSTOMERS)],
            "archetype": archetype,
            "country": country,
        }
    )


def _bundle_pairs(rng: np.random.Generator, products: pd.DataFrame) -> list[tuple[str, str]]:
    """Pick product pairs per category that co-occur more than chance -> real lift for Apriori."""
    pairs = []
    for cat in CATEGORIES:
        cat_products = products.loc[products["category"] == cat, "product_id"].tolist()
        if len(cat_products) < 4:
            continue
        chosen = rng.choice(cat_products, size=4, replace=False)
        pairs.append((chosen[0], chosen[1]))
        pairs.append((chosen[2], chosen[3]))
    return pairs


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    products = _make_products(rng)
    customers = _make_customers(rng)
    bundles = _bundle_pairs(rng, products)
    bundle_lookup = {a: b for a, b in bundles}

    start = pd.Timestamp("2024-01-01")
    horizon_days = 365

    lines = []
    invoice_no = 100000
    n_outlier_invoices = max(3, int(N_INVOICES * 0.01))
    outlier_invoice_idx = set(
        rng.choice(N_INVOICES, size=n_outlier_invoices, replace=False).tolist()
    )

    cust_by_archetype = {a: customers.loc[customers["archetype"] == a, "customer_id"].tolist()
                         for a in ARCHETYPES}
    # Weighted customer draw so higher-frequency archetypes generate proportionally more invoices.
    weights = customers["archetype"].map(lambda a: ARCHETYPES[a]["freq_lambda"]).to_numpy(dtype=float)
    weights = weights / weights.sum()

    for i in range(N_INVOICES):
        cust_idx = rng.choice(N_CUSTOMERS, p=weights)
        cust = customers.iloc[cust_idx]
        arche = ARCHETYPES[cust["archetype"]]

        day_offset = int(rng.beta(2 - arche["recency_bias"], 1 + arche["recency_bias"]) * horizon_days)
        invoice_date = start + pd.Timedelta(days=day_offset)

        basket_size = max(1, int(rng.poisson(3) * arche["spend_mult"]))
        is_outlier = i in outlier_invoice_idx
        if is_outlier:
            basket_size = int(rng.integers(25, 45))

        chosen_products = rng.choice(products["product_id"], size=basket_size, replace=False)
        basket = set(chosen_products.tolist())
        # Reinforce bundle structure: if a bundle "trigger" product was drawn, add its pair.
        for trigger, partner in bundle_lookup.items():
            if trigger in basket and rng.random() < 0.65:
                basket.add(partner)

        invoice_id = f"INV{invoice_no:06d}"
        invoice_no += 1
        for pid in basket:
            prod = products.loc[products["product_id"] == pid].iloc[0]
            qty = int(rng.integers(15, 40)) if is_outlier else int(rng.integers(1, 6))
            price = prod["unit_price"] * (1.0 if not is_outlier else 0.9)
            lines.append(
                {
                    "invoice_id": invoice_id,
                    "invoice_date": invoice_date,
                    "customer_id": cust["customer_id"],
                    "country": cust["country"],
                    "product_id": pid,
                    "product_name": prod["product_name"],
                    "category": prod["category"],
                    "quantity": qty,
                    "unit_price": round(float(price), 2),
                }
            )

    df = pd.DataFrame(lines)
    df["line_total"] = (df["quantity"] * df["unit_price"]).round(2)
    df = df.merge(customers[["customer_id", "archetype"]], on="customer_id", how="left")
    return df.sort_values(["invoice_date", "invoice_id"]).reset_index(drop=True)


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/online_retail.csv", index=False)
    print(f"wrote {len(out):,} line items across {out['invoice_id'].nunique():,} invoices "
          f"and {out['customer_id'].nunique():,} customers -> data/online_retail.csv")
