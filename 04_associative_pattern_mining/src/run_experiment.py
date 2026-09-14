"""CRISP-DM end-to-end experiment for market-basket pattern mining.

Runs every phase in order and writes the artifacts the dashboard reads:

    artifacts/data_quality.json          phase 2-3  understanding + preparation
    artifacts/autoresearch_history.csv   phase 4    hill-climbing search trace
    artifacts/autoresearch_best.json     phase 4    winning configuration
    artifacts/itemsets.csv               phase 4    frequent itemsets
    artifacts/rules.csv                  phase 4-5  rules + held-out re-scoring
    artifacts/metrics.json               phase 5    baseline vs tuned, stability

Usage:  python -m src.run_experiment  [--restarts 3] [--max-steps 12]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.generate_data import BUNDLES
from .autoresearch import SEED_CONFIG, hill_climb
from .mining import (
    clean_raw,
    evaluate,
    fmt_items,
    load_raw,
    mine_rules,
    objective,
    score_rules_on,
    split_baskets,
    to_basket_matrix,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def ground_truth_recovery(rules: pd.DataFrame) -> dict:
    """Did mining recover the co-purchase bundles the generator planted?

    Only possible because the data is synthetic and its ground truth is known —
    on real transactions there is nothing to check rules against. Every planted
    pair is looked for among the mined rules' full itemsets.
    """
    from itertools import combinations

    mined = {
        frozenset(r["antecedents"]) | frozenset(r["consequents"])
        for _, r in rules.iterrows()
    }
    per_bundle, found_total, pair_total = [], 0, 0
    for items, _prob in BUNDLES:
        pairs = list(combinations(sorted(items), 2))
        found = sum(1 for pair in pairs if frozenset(pair) in mined)
        per_bundle.append(
            {"bundle": " + ".join(sorted(items)), "pairs_found": found, "pairs_total": len(pairs)}
        )
        found_total += found
        pair_total += len(pairs)
    return {
        "planted_bundles": len(BUNDLES),
        "bundles_fully_recovered": sum(1 for b in per_bundle if b["pairs_found"] == b["pairs_total"]),
        "pairs_recovered": found_total,
        "pairs_planted": pair_total,
        "pct_pairs_recovered": round(100 * found_total / max(pair_total, 1), 1),
        "per_bundle": per_bundle,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)

    # -- Phase 2/3: Data Understanding & Preparation -----------------------
    print("[1/5] Loading and cleaning transactions ...")
    raw = load_raw()
    clean, quality = clean_raw(raw)
    matrix = to_basket_matrix(clean)
    print(f"      {quality['rows_in']:,} rows -> {quality['baskets_out']:,} baskets "
          f"x {quality['distinct_items_out']} items")

    item_freq = (
        clean["itemDescription"].value_counts().rename_axis("item").reset_index(name="count")
    )
    item_freq["support"] = item_freq["count"] / quality["baskets_out"]
    item_freq.to_csv(ARTIFACTS / "item_frequency.csv", index=False)

    basket_sizes = clean.groupby("basket_id").size()
    quality["basket_size_median"] = float(basket_sizes.median())
    quality["basket_size_max"] = int(basket_sizes.max())
    (ARTIFACTS / "data_quality.json").write_text(json.dumps(quality, indent=2))

    # -- Phase 4: Modeling — baseline, then autoresearch --------------------
    print("[2/5] Baseline mining run (hand-picked configuration) ...")
    base_itemsets, base_rules = mine_rules(matrix, SEED_CONFIG)
    baseline = evaluate(matrix, base_itemsets, base_rules)
    baseline["score"] = objective(baseline)
    print(f"      {baseline['n_rules']} rules, score {baseline['score']:.4f}")

    print(f"[3/5] Autoresearch hill climbing ({args.restarts} restarts) ...")
    best_config, best_metrics, history = hill_climb(
        matrix, restarts=args.restarts, max_steps=args.max_steps, seed=args.seed
    )
    history.to_csv(ARTIFACTS / "autoresearch_history.csv", index=False)
    (ARTIFACTS / "autoresearch_best.json").write_text(
        json.dumps({"config": best_config.as_dict(), "metrics": best_metrics}, indent=2)
    )
    print(f"      best: {best_config.as_dict()}")
    print(f"      {best_metrics['n_rules']} rules, score {best_metrics['score']:.4f}")

    # -- Phase 4: final model with the winning configuration ---------------
    print("[4/5] Mining final rule set with the winning configuration ...")
    itemsets, rules = mine_rules(matrix, best_config)

    # -- Phase 5: Evaluation — held-out rule stability ---------------------
    print("[5/5] Held-out stability check (70/30 basket split) ...")
    train_matrix, test_matrix = split_baskets(matrix)
    _, train_rules = mine_rules(train_matrix, best_config)
    holdout = score_rules_on(test_matrix, train_rules)
    checked = pd.concat([train_rules.reset_index(drop=True), holdout.reset_index(drop=True)], axis=1)

    held = checked[checked["holdout_lift"] >= best_config.min_lift]
    stability = {
        "train_baskets": int(len(train_matrix)),
        "test_baskets": int(len(test_matrix)),
        "rules_mined_on_train": int(len(train_rules)),
        "rules_holding_on_test": int(len(held)),
        "pct_rules_holding": round(100 * len(held) / max(len(train_rules), 1), 1),
        "mean_lift_train": round(float(train_rules["lift"].mean()), 3) if len(train_rules) else 0.0,
        "mean_lift_holdout": round(float(checked["holdout_lift"].mean()), 3) if len(checked) else 0.0,
        "lift_correlation": (
            round(float(checked["lift"].corr(checked["holdout_lift"])), 3) if len(checked) > 2 else None
        ),
    }
    print(f"      {stability['rules_holding_on_test']}/{stability['rules_mined_on_train']} "
          f"rules hold on held-out baskets ({stability['pct_rules_holding']}%)")

    recovery = ground_truth_recovery(rules)
    print(f"      ground truth: {recovery['bundles_fully_recovered']}/{recovery['planted_bundles']} "
          f"planted bundles fully recovered "
          f"({recovery['pairs_recovered']}/{recovery['pairs_planted']} pairs)")

    # -- Persist artifacts --------------------------------------------------
    export = rules.copy()
    export["antecedents"] = export["antecedents"].apply(fmt_items)
    export["consequents"] = export["consequents"].apply(fmt_items)
    export.to_csv(ARTIFACTS / "rules.csv", index=False)

    itemsets_out = itemsets.copy()
    itemsets_out["itemsets"] = itemsets_out["itemsets"].apply(fmt_items)
    itemsets_out["length"] = itemsets_out["itemsets"].str.count(",") + 1
    itemsets_out.sort_values("support", ascending=False).to_csv(
        ARTIFACTS / "itemsets.csv", index=False
    )

    checked_out = checked.copy()
    checked_out["antecedents"] = checked_out["antecedents"].apply(fmt_items)
    checked_out["consequents"] = checked_out["consequents"].apply(fmt_items)
    checked_out.to_csv(ARTIFACTS / "holdout_rules.csv", index=False)

    (ARTIFACTS / "metrics.json").write_text(
        json.dumps(
            {
                "data_quality": quality,
                "baseline_config": SEED_CONFIG.as_dict(),
                "baseline_metrics": baseline,
                "best_config": best_config.as_dict(),
                "best_metrics": best_metrics,
                "configs_evaluated": int(history["evaluated"].sum()),
                "holdout_stability": stability,
                "ground_truth_recovery": recovery,
            },
            indent=2,
        )
    )
    print(f"\nDone. Artifacts written to {ARTIFACTS}")


if __name__ == "__main__":
    main()
