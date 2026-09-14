"""Data preparation + association-rule mining.

Shared by the batch experiment (``run_experiment.py``), the hill-climbing
search (``autoresearch.py``) and the Streamlit dashboard (``app.py``) so all
three see exactly the same baskets and the same rule metrics.

Interestingness measures implemented/reported here follow the standard
association-mining literature:

* support, confidence            — Agrawal, Imielinski & Swami (1993)
* Apriori candidate generation   — Agrawal & Srikant (VLDB 1994)
* FP-Growth (no candidate gen)   — Han, Pei & Yin (SIGMOD 2000)
* lift, conviction               — Brin, Motwani, Ullman & Tsur (SIGMOD 1997)
* leverage                       — Piatetsky-Shapiro (1991)
* Zhang's metric                 — Zhang (2000)

mlxtend supplies the frequent-itemset and rule-metric implementations; this
module supplies the pipeline, the cleaning rules and the scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "grocery_transactions.csv"

RULE_METRICS = [
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
    "zhangs_metric",
]


@dataclass(frozen=True)
class MiningConfig:
    """One point in the association-mining hyperparameter space."""

    algorithm: str = "fpgrowth"       # "apriori" | "fpgrowth"
    min_support: float = 0.01
    min_confidence: float = 0.25
    min_lift: float = 1.10
    max_len: int = 3                  # max itemset size (antecedent + consequent)

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# CRISP-DM phase 3: Data Preparation
# --------------------------------------------------------------------------

def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_data.py` first."
        )
    return pd.read_csv(path)


def clean_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise item names and drop unusable rows.

    Returns the cleaned frame plus a report of what was removed, so the
    dashboard can show the data-quality story instead of hiding it.
    """
    report = {"rows_in": len(raw), "distinct_items_in": raw["itemDescription"].nunique()}

    df = raw.copy()
    df["itemDescription"] = (
        df["itemDescription"].astype("string").str.strip().str.lower()
    )

    blank = df["itemDescription"].isna() | (df["itemDescription"] == "")
    report["rows_dropped_missing_item"] = int(blank.sum())
    df = df[~blank]

    # A basket is a (member, date) shopping trip. The generator also carries an
    # explicit _basket_id; prefer it when present, fall back to member+date so
    # the pipeline also works on the real Kaggle file, which has no basket id.
    if "_basket_id" in df.columns:
        df["basket_id"] = df["_basket_id"].astype(str)
    else:
        df["basket_id"] = df["Member_number"].astype(str) + "_" + df["Date"].astype(str)

    before = len(df)
    df = df.drop_duplicates(subset=["basket_id", "itemDescription"])
    report["rows_dropped_duplicate"] = int(before - len(df))

    # Single-item baskets cannot contribute to any rule.
    sizes = df.groupby("basket_id")["itemDescription"].size()
    singletons = set(sizes[sizes < 2].index)
    report["baskets_dropped_singleton"] = len(singletons)
    df = df[~df["basket_id"].isin(singletons)]

    report["rows_out"] = len(df)
    report["distinct_items_out"] = int(df["itemDescription"].nunique())
    report["baskets_out"] = int(df["basket_id"].nunique())
    report["avg_basket_size"] = round(report["rows_out"] / max(report["baskets_out"], 1), 2)
    return df.reset_index(drop=True), report


def to_basket_matrix(clean: pd.DataFrame) -> pd.DataFrame:
    """One-hot transaction matrix: rows = baskets, columns = items, values = bool."""
    matrix = (
        clean.assign(present=True)
        .pivot_table(
            index="basket_id",
            columns="itemDescription",
            values="present",
            aggfunc="max",
            fill_value=False,
        )
        .astype(bool)
    )
    matrix.columns.name = None
    return matrix


# --------------------------------------------------------------------------
# CRISP-DM phase 4: Modeling
# --------------------------------------------------------------------------

def mine_rules(matrix: pd.DataFrame, config: MiningConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run frequent-itemset mining, then derive rules. Returns (itemsets, rules)."""
    miner = apriori if config.algorithm == "apriori" else fpgrowth
    itemsets = miner(
        matrix,
        min_support=config.min_support,
        use_colnames=True,
        max_len=config.max_len,
    )

    empty_rules = pd.DataFrame(columns=["antecedents", "consequents", *RULE_METRICS])
    if itemsets.empty:
        return itemsets, empty_rules

    rules = association_rules(
        itemsets, metric="confidence", min_threshold=config.min_confidence
    )
    if rules.empty:
        return itemsets, empty_rules

    rules = rules[rules["lift"] >= config.min_lift].copy()
    if rules.empty:
        return itemsets, empty_rules

    rules["antecedent_len"] = rules["antecedents"].apply(len)
    rules["consequent_len"] = rules["consequents"].apply(len)
    rules["rule"] = rules.apply(
        lambda r: f"{fmt_items(r['antecedents'])} → {fmt_items(r['consequents'])}", axis=1
    )
    return itemsets, rules.sort_values("lift", ascending=False).reset_index(drop=True)


def fmt_items(items) -> str:
    return ", ".join(sorted(items))


# --------------------------------------------------------------------------
# CRISP-DM phase 5: Evaluation
# --------------------------------------------------------------------------

def basket_coverage(matrix: pd.DataFrame, rules: pd.DataFrame) -> float:
    """Fraction of baskets matched by at least one rule's full itemset.

    Coverage is the honesty check on a rule set: a handful of very high-lift
    rules that fire on 1% of baskets is not an actionable result.
    """
    if rules.empty:
        return 0.0
    covered = pd.Series(False, index=matrix.index)
    for _, rule in rules.iterrows():
        items = [i for i in (set(rule["antecedents"]) | set(rule["consequents"])) if i in matrix.columns]
        if items:
            covered |= matrix[items].all(axis=1)
    return float(covered.mean())


def evaluate(matrix: pd.DataFrame, itemsets: pd.DataFrame, rules: pd.DataFrame) -> dict:
    """Summary statistics for one mining run."""
    if rules.empty:
        return {
            "n_itemsets": int(len(itemsets)),
            "n_rules": 0,
            "mean_lift": 0.0,
            "max_lift": 0.0,
            "mean_confidence": 0.0,
            "mean_support": 0.0,
            "coverage": 0.0,
        }
    return {
        "n_itemsets": int(len(itemsets)),
        "n_rules": int(len(rules)),
        "mean_lift": float(rules["lift"].mean()),
        "max_lift": float(rules["lift"].max()),
        "mean_confidence": float(rules["confidence"].mean()),
        "mean_support": float(rules["support"].mean()),
        "coverage": basket_coverage(matrix, rules),
    }


def objective(metrics: dict, target_rules: tuple[int, int] = (20, 200)) -> float:
    """Scalar quality score for a rule set — what hill climbing maximises.

    Three terms, all deliberately simple and inspectable:

    1. ``mean_lift - 1`` — average dependence strength above independence
       (lift == 1 means the rule tells you nothing), squashed so a few
       extreme rules cannot dominate.
    2. ``coverage``      — share of baskets the rule set actually explains.
    3. a size penalty    — rule sets outside ``target_rules`` are penalised,
       because both "3 rules" and "40,000 rules" are useless to an analyst.

    This is a stated modelling choice, not a result from any paper: the
    literature supplies the metrics, the weighting is ours.
    """
    n_rules = metrics["n_rules"]
    if n_rules == 0:
        return 0.0

    lift_term = min(max(metrics["mean_lift"] - 1.0, 0.0) / 4.0, 1.0)
    coverage_term = metrics["coverage"]

    low, high = target_rules
    if n_rules < low:
        size_penalty = (low - n_rules) / low
    elif n_rules > high:
        size_penalty = min((n_rules - high) / high, 1.0)
    else:
        size_penalty = 0.0

    return float(0.55 * lift_term + 0.45 * coverage_term - 0.35 * size_penalty)


def run_pipeline(config: MiningConfig, matrix: pd.DataFrame | None = None):
    """Convenience: clean → matrix → mine → evaluate for one config."""
    if matrix is None:
        clean, _ = clean_raw(load_raw())
        matrix = to_basket_matrix(clean)
    itemsets, rules = mine_rules(matrix, config)
    metrics = evaluate(matrix, itemsets, rules)
    metrics["score"] = objective(metrics)
    return matrix, itemsets, rules, metrics


def score_rules_on(matrix: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Recompute support/confidence/lift for existing rules on another basket set.

    Association rules are descriptive, not predictive, so they are rarely
    validated on held-out data — but a rule mined from a sample is still an
    estimate. Re-scoring mined rules against baskets they were not mined from
    is what separates a stable pattern from one fitted to noise.
    """
    n = len(matrix)
    out = []
    for _, rule in rules.iterrows():
        ante = [i for i in rule["antecedents"] if i in matrix.columns]
        cons = [i for i in rule["consequents"] if i in matrix.columns]
        if len(ante) != len(rule["antecedents"]) or len(cons) != len(rule["consequents"]):
            out.append({"support": 0.0, "confidence": 0.0, "lift": 0.0})
            continue

        ante_mask = matrix[ante].all(axis=1)
        cons_mask = matrix[cons].all(axis=1)
        both = int((ante_mask & cons_mask).sum())
        s_ante, s_cons = ante_mask.mean(), cons_mask.mean()
        support = both / n
        confidence = support / s_ante if s_ante > 0 else 0.0
        lift = confidence / s_cons if s_cons > 0 else 0.0
        out.append({"support": support, "confidence": float(confidence), "lift": float(lift)})

    scored = pd.DataFrame(out, index=rules.index)
    return scored.add_prefix("holdout_")


def split_baskets(matrix: pd.DataFrame, test_frac: float = 0.3, seed: int = 42):
    """Random basket-level split for the held-out rule-stability check."""
    shuffled = matrix.sample(frac=1.0, random_state=seed)
    cut = int(len(shuffled) * (1 - test_frac))
    return shuffled.iloc[:cut], shuffled.iloc[cut:]
