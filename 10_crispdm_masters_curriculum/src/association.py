"""CRISP-DM Modeling phase — associative rule mining (market-basket analysis)."""
from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def mine_rules(baskets: list[list[str]], min_support: float = 0.003, min_lift: float = 1.5) -> dict:
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = apriori(onehot, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return {
            "frequent_itemsets": frequent,
            "rules": pd.DataFrame(),
            "n_baskets": len(baskets),
            "n_frequent_itemsets": 0,
            "n_rules": 0,
        }

    rules = association_rules(frequent, metric="lift", min_threshold=min_lift)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))

    keep = ["antecedents", "consequents", "support", "confidence", "lift"]
    rules_view = rules[keep].round(4)

    return {
        "frequent_itemsets": frequent,
        "rules": rules_view,
        "n_baskets": len(baskets),
        "n_frequent_itemsets": int(len(frequent)),
        "n_rules": int(len(rules_view)),
    }
