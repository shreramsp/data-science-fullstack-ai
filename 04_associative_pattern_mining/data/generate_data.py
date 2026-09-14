"""Generate a small, seeded, reproducible market-basket dataset.

The output schema deliberately mirrors the Kaggle "Groceries dataset"
(heeraldedhia/groceries-dataset): one row per purchased item, with columns
``Member_number``, ``Date``, ``itemDescription``. That dataset requires a
Kaggle account to download, which breaks a plain ``git clone && pip install``
reproduction, so this generator stands in for it.

Structure that association-rule mining is supposed to find is *planted* here
on purpose (see ``BUNDLES``), mixed with a Zipf-ish popularity background and
a small share of deliberately dirty rows so the Data Preparation phase has
real work to do.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260913
OUT_PATH = Path(__file__).with_name("grocery_transactions.csv")

# Background catalog: items bought largely independently of each other.
CATALOG = [
    "whole milk", "rolls/buns", "soda", "yogurt", "bottled water",
    "root vegetables", "tropical fruit", "shopping bags", "sausage", "citrus fruit",
    "pastry", "canned beer", "newspapers", "frankfurter", "brown bread",
    "domestic eggs", "margarine", "pip fruit", "napkins", "chocolate",
    "frozen vegetables", "curd", "beef", "white bread", "salty snack",
    "dish detergent", "paper towels", "ice cream", "onions", "cat food",
]

# Planted co-purchase bundles: (items, probability a basket contains the bundle).
# These are the patterns a correct Apriori/FP-Growth run should recover.
BUNDLES = [
    (["tortillas", "salsa", "ground beef"], 0.055),
    (["white bread", "butter", "jam"], 0.050),
    (["pasta", "tomato sauce", "parmesan"], 0.060),
    (["coffee", "creamer", "sugar"], 0.045),
    (["diapers", "canned beer"], 0.030),
    (["cereal", "whole milk"], 0.070),
    (["hamburger meat", "rolls/buns", "ketchup"], 0.040),
    (["chips", "soda", "salsa"], 0.045),
    (["bananas", "yogurt", "granola"], 0.050),
    (["red wine", "cheese", "crackers"], 0.035),
]

BUNDLE_ITEMS = sorted({item for items, _ in BUNDLES for item in items})
ALL_ITEMS = sorted(set(CATALOG) | set(BUNDLE_ITEMS))


def _zipf_weights(n: int, exponent: float = 0.8) -> np.ndarray:
    """Popularity weights: a few items dominate, most are rare."""
    ranks = np.arange(1, n + 1)
    weights = 1.0 / ranks**exponent
    return weights / weights.sum()


def generate(n_baskets: int = 6000, n_members: int = 1200, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    catalog = np.array(CATALOG)
    weights = _zipf_weights(len(catalog))
    # Shuffle which item lands on which popularity rank so the ordering above
    # does not itself encode the answer.
    weights = weights[rng.permutation(len(catalog))]

    start = date(2024, 1, 1)
    rows: list[dict] = []

    for basket_id in range(n_baskets):
        member = int(rng.integers(1000, 1000 + n_members))
        day = start + timedelta(days=int(rng.integers(0, 730)))

        items: set[str] = set()

        # 1. Planted bundles. A bundle fires as a unit, but each member item
        #    can still be forgotten (85% retention) so rules are not perfect.
        for bundle_items, prob in BUNDLES:
            if rng.random() < prob:
                for item in bundle_items:
                    if rng.random() < 0.85:
                        items.add(item)

        # 2. Background items drawn by popularity.
        n_background = int(rng.integers(1, 8))
        if n_background:
            items.update(rng.choice(catalog, size=n_background, replace=False, p=weights))

        if not items:
            continue

        # sorted(): set iteration order for strings varies per process with
        # hash randomisation, which would make the whole file non-reproducible.
        for item in sorted(items):
            rows.append(
                {
                    "Member_number": member,
                    "Date": day.strftime("%d-%m-%Y"),
                    "itemDescription": item,
                    "_basket_id": basket_id,
                }
            )

    df = pd.DataFrame(rows)

    # 3. Inject realistic defects for the Data Preparation phase to handle.
    n = len(df)
    dirty = df.sample(frac=0.02, random_state=seed).copy()
    dirty["itemDescription"] = dirty["itemDescription"].str.upper().radd("  ")  # casing + whitespace
    df = pd.concat([df, dirty], ignore_index=True)                              # also creates duplicates

    missing_idx = df.sample(n=int(n * 0.005), random_state=seed + 1).index
    df.loc[missing_idx, "itemDescription"] = np.nan

    blank_idx = df.sample(n=int(n * 0.003), random_state=seed + 2).index
    df.loc[blank_idx, "itemDescription"] = ""

    return df.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baskets", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    df = generate(n_baskets=args.baskets, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df):,} rows -> {args.out}")
    print(f"  baskets: {df['_basket_id'].nunique():,}")
    print(f"  distinct items (raw, pre-cleaning): {df['itemDescription'].nunique():,}")


if __name__ == "__main__":
    main()
