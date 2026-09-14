# CRISP-DM — Market Basket Pattern Mining

The six CRISP-DM phases mapped to the concrete decisions and code in this
project. Numbers come from the last `python -m src.run_experiment` run.

## 1. Business Understanding

**Question.** Which grocery products are bought together strongly enough to
justify a business action — shelf placement, a bundle promotion, or a
next-item recommendation at checkout?

**Why association mining and not classification.** There is no label to
predict. The task is unsupervised pattern discovery over set-valued
transactions, which is exactly what frequent-itemset mining and association
rules were designed for.

**Success criteria, agreed up front.**
- Rules must be *actionable*: enough of them to be interesting, few enough to
  read. Both "4 rules" and "40,000 rules" are failures.
- Rules must cover a meaningful share of real baskets, not just high-lift
  curiosities that fire on 1% of trips.
- Thresholds must be chosen by a stated procedure, not by hand-tuning until
  the output looks good — hence the hill-climbing autoresearch in phase 4.

## 2. Data Understanding

Schema mirrors the Kaggle *Groceries dataset*: one row per purchased item,
with `Member_number`, `Date`, `itemDescription`.

Profiling (`artifacts/data_quality.json`, Data tab in the dashboard) found:

| Observation | Value |
|---|---|
| Raw rows | 30,995 |
| Distinct item labels, raw | 101 |
| Distinct item labels, after normalising casing/whitespace | 51 |
| Baskets after preparation | 5,444 |
| Mean / median / max basket size | 5.44 / 5 / 15 |

The gap between 101 and 51 item labels is the headline data-quality finding:
half the apparent product catalogue was duplicate spellings of the same
product. Mining on the raw labels would split each item's support across its
variants and suppress genuine rules.

Item popularity is heavily skewed — a handful of staples appear in 25–40% of
baskets while most items sit below 5%. This directly constrains `min_support`:
set it too high and only the staples survive, and staples co-occur with
everything, which produces high-support but low-lift noise.

## 3. Data Preparation

Implemented in `src/mining.py` — `clean_raw()` and `to_basket_matrix()`.

| Step | Rule | Effect |
|---|---|---|
| Normalise item text | strip whitespace, lowercase | 101 → 51 distinct items |
| Drop unusable rows | blank / missing `itemDescription` | −242 rows |
| Deduplicate | one row per (basket, item) | −598 rows |
| Define the basket | explicit basket id, else `Member_number` + `Date` | 5,444 baskets |
| Drop singletons | baskets with < 2 items | −551 baskets |
| Encode | pivot to a boolean basket × item matrix | 5,444 × 51 |

Dropping single-item baskets is a deliberate choice, not a convenience: a
one-item basket cannot support or contradict any rule, but it *does* enter the
denominator of every support calculation and so deflates every metric
uniformly. Excluding them keeps support interpretable as "share of baskets
where a rule could have applied".

## 4. Modeling

**Algorithms.** Apriori (Agrawal & Srikant, VLDB 1994) and FP-Growth (Han,
Pei & Yin, SIGMOD 2000), via `mlxtend`. Both are exact and return identical
frequent itemsets; they differ only in how they get there (candidate
generation with downward-closure pruning vs. a prefix-tree with no candidate
generation). Rules are then derived with the standard interestingness
measures — support, confidence, lift, leverage, conviction, Zhang's metric.

**Autoresearch (`src/autoresearch.py`).** Rather than hand-picking thresholds,
the search space

```
algorithm       ∈ {apriori, fpgrowth}
min_support     ∈ {0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05}
min_confidence  ∈ {0.10 … 0.70}
min_lift        ∈ {1.0, 1.1, 1.25, 1.5, 2.0, 3.0}
max_len         ∈ {2, 3, 4}
```

is explored by **steepest-ascent hill climbing with random restarts**: from
the current configuration, evaluate every configuration one grid step away
along one axis, move to the best strict improvement, stop at a local optimum,
then restart from a random point. Random restarts are the standard mitigation
for hill climbing's local-optimum weakness. Every evaluation is cached and
logged to `artifacts/autoresearch_history.csv` so the whole search trace — not
just the winner — is visible in the dashboard.

**Objective** (`mining.objective`), maximised by the search:

```
score = 0.55 · min(max(mean_lift − 1, 0)/4, 1)   # dependence strength
      + 0.45 · basket_coverage                    # share of baskets explained
      − 0.35 · size_penalty(n_rules ∉ [20, 200])  # actionability
```

The three components encode the phase-1 success criteria. The **weights are
our stated modelling choice**, not a result imported from any paper — the
literature supplies the measures, the trade-off between them is a business
decision.

## 5. Evaluation

Three independent checks, all in `artifacts/metrics.json`:

**a. Rule-set quality.** The autoresearch winner
(`fpgrowth, min_support=0.02, min_confidence=0.20, min_lift=1.10, max_len=3`)
produced **188 rules** from 331 frequent itemsets, mean lift **10.29**, basket
coverage **65.5%**, objective **0.8449** — against the hand-picked baseline's
159 rules and **0.8339**. The improvement is real but small (+0.0111): the
search mostly confirmed that a sensible hand-picked configuration was already
near a local optimum, and it got there by trading a little mean lift for more
coverage and more rules.

**b. Held-out stability.** Rules mined on a random 70% of baskets (3,810) were
re-scored on the unseen 30% (1,634). **156 of 211 rules (73.9%)** still cleared
the lift threshold, train/held-out mean lift was 9.23 vs 9.43, and the
per-rule lift correlation was **0.981**. The strong rules transfer; the ~26%
that fail are mostly low-support rules whose estimates were noisy.

**c. Ground-truth recovery.** Because the data is synthetic, the planted
co-purchase bundles are known — a check real transaction data never permits.
Mining recovered **10/10 bundles and 26/26 item pairs**. This validates the
pipeline against the generator, and says nothing about real shopper behaviour.

**Known artifact — redundant and lift-inflated rules.** Two effects, both
expected and both visible in the output:

*Redundancy.* A single 3-item bundle yields every antecedent/consequent split
of its subsets. The `cheese` / `crackers` / `red wine` bundle alone accounts
for **12 of the 188 rules**, all with lift between 25.7 and 26.1. The rule
count overstates the number of distinct findings.

*Lift inflation by a frequent bystander.* `cereal, margarine → whole milk`
reports lift **3.40**, but margarine is independent of both its partners
(lift 0.94 with cereal, 1.01 with whole milk) — the rule inherits the genuine
`cereal → whole milk` lift of **3.41**. Conditioning on a common item leaves
lift almost unchanged, so these rules pass any lift threshold the real one
passes.

Neither is a bug; both are the well-known redundancy problem in rule mining.
No closed/maximal-itemset or minimal-non-redundant-rule filter is applied —
see the README's limitations.

## 6. Deployment

`app.py` — a Streamlit admin dashboard for local deployment.
It loads the saved artifacts and also re-mines live when an analyst moves the
thresholds, so the modelling phase is reproducible from the UI rather than
only from the CLI. Six tabs: CRISP-DM phase map and measure definitions, data
quality and profiling, a filterable rule explorer with support/confidence/lift
plots, the autoresearch search trace, the evaluation results above, and a
basket recommender that serves the mined rules as next-item suggestions.
