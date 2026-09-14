# 04 — Market Basket Pattern Mining

Association-rule mining on grocery transactions, built end to end against the
CRISP-DM framework: seeded transaction data → cleaning and basket encoding →
frequent-itemset mining (Apriori / FP-Growth) → **hill-climbing autoresearch**
over the mining hyperparameters → honest evaluation → a Streamlit data-science
admin dashboard with a live rule explorer and basket recommender.

This implementation uses an original mining pipeline, interface,
documentation, and reported results.

### Why not the real Kaggle dataset

The Kaggle *Groceries dataset* requires account authentication to download,
which breaks a plain `git clone && pip install` reproduction. To keep the
workflow reproducible, `data/generate_data.py` instead
generates a **small, seeded, synthetic transaction log with the same schema**
(`Member_number`, `Date`, `itemDescription`). The generator plants ten known
co-purchase bundles, mixes them with a skewed popularity background, and
injects realistic defects — casing/whitespace variants, blank items, duplicate
rows, single-item baskets — so the preparation phase has real work to do.

The upside of a known generator is a validation that real data cannot offer:
the planted bundles are ground truth, so the miner can be checked against them
directly (see "Honest results"). The downside is stated plainly in
"Limitations".

## What was built

- `data/generate_data.py` — seeded synthetic transaction generator (6,000
  baskets → 30,995 raw item rows).
- `src/mining.py` — cleaning rules, basket-matrix encoding, Apriori/FP-Growth
  mining, the interestingness measures, basket coverage, the search objective,
  and held-out rule re-scoring. Shared by the CLI and the dashboard so both
  see identical baskets and metrics.
- `src/autoresearch.py` — steepest-ascent hill climbing with random restarts
  over `{algorithm, min_support, min_confidence, min_lift, max_len}`, logging
  every evaluated configuration.
- `src/run_experiment.py` — the CRISP-DM pipeline end to end; writes all
  artifacts to `artifacts/`.
- `app.py` — Streamlit admin dashboard, six tabs: CRISP-DM phase map and
  measure definitions, data-quality profiling, filterable rule explorer with
  support/confidence/lift charts, the autoresearch search trajectory,
  evaluation results, and a next-item basket recommender. Sidebar sliders
  re-mine live.
- `CRISP_DM.md` — the six phases mapped to this project's concrete decisions.

## Setup

```bash
cd 04_associative_pattern_mining
python3.13 -m venv .venv        # 3.13 recommended; mlxtend/pandas wheels
source .venv/bin/activate       # not yet available for 3.14 at build time
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate the dataset
python data/generate_data.py

# 2. Run the full CRISP-DM experiment (mining + autoresearch + evaluation)
python -m src.run_experiment          # ~5 s; writes artifacts/

# 3. Launch the dashboard
streamlit run app.py
```

Then open the printed local URL (default `http://localhost:8501`).

The hill-climbing search can also be run on its own:

```bash
python -m src.autoresearch --restarts 3 --max-steps 12
```

## Approach

1. **Business understanding** — find product pairs/triples co-bought strongly
   enough to act on, with success defined up front as *actionable* (20–200
   rules), *well-covering*, and *chosen by a stated procedure* rather than
   hand-tuned.
2. **Data understanding** — profile item frequency, basket sizes, and injected
   defects.
3. **Data preparation** — normalise item text, drop blank/duplicate rows and
   single-item baskets, pivot to a boolean 5,444 × 51 basket matrix.
4. **Modeling** — Apriori (Agrawal & Srikant, 1994) and FP-Growth (Han, Pei &
   Yin, 2000) for frequent itemsets; rules scored with support, confidence,
   lift (Brin et al., 1997), leverage (Piatetsky-Shapiro, 1991), conviction,
   and Zhang's metric. Thresholds selected by hill-climbing autoresearch.
5. **Evaluation** — rule-set quality, a 70/30 held-out stability check, and
   ground-truth bundle recovery.
6. **Deployment** — the Streamlit dashboard serves the rules and re-mines live.

Full detail in [CRISP_DM.md](./CRISP_DM.md).

## Honest results

From the last `python -m src.run_experiment` run (5,444 baskets × 51 items):

**Autoresearch.** 48 distinct configurations evaluated across 3 restarts.

| | Baseline (hand-picked) | Autoresearch best |
|---|---|---|
| Configuration | fpgrowth, sup 0.02, conf 0.30, lift 1.10, len 3 | fpgrowth, sup 0.02, conf **0.20**, lift 1.10, len 3 |
| Rules | 159 | 188 |
| Mean lift | 11.55 | 10.29 |
| Basket coverage | 63.1% | 65.5% |
| Objective score | 0.8339 | **0.8449** |

The search improved the objective by **+0.0111** — real, but small. The honest
reading is that a sensible hand-picked configuration was already close to a
local optimum, and the search's contribution was to confirm that and trade a
little mean lift for more coverage. It did not discover a dramatically better
region of the space.

**Held-out stability** (mine on 70% of baskets, re-score on the unseen 30%):

| Metric | Value |
|---|---|
| Rules mined on train | 211 |
| Still clearing the lift threshold on test | **156 (73.9%)** |
| Mean lift, train → held out | 9.23 → 9.43 |
| Per-rule lift correlation | **0.981** |

Strong rules transfer well; the ~26% that fail are mostly low-support rules
whose estimates were noisy on 3,810 baskets.

**Ground-truth recovery.** **10/10 planted bundles and 26/26 item pairs** were
recovered. This validates the pipeline against its own generator — it is a
correctness check on the code, not evidence about real shoppers.

### Limitations, stated plainly

- **The data is synthetic.** Every number above describes this generated
  dataset. None of it is a claim about real grocery baskets, and the
  ground-truth recovery result in particular measures the miner against the
  generator, nothing more.
- **Many rules are redundant restatements of the same pattern.** One planted
  3-item bundle (`cheese`, `crackers`, `red wine`) generates **12 separate
  rules** — every antecedent/consequent split of its subsets — all with
  near-identical lift (25.7–26.1). The rule count therefore overstates how many
  distinct findings there are.
- **A frequent bystander item inflates lift.** `cereal, margarine → whole milk`
  reports lift **3.40**, but margarine is independent of both (lift 0.94 with
  cereal, 1.01 with whole milk); the rule simply inherits the genuine
  `cereal → whole milk` lift of **3.41**. Adding a common item to an
  antecedent barely changes lift, so such rules survive filtering.
  No redundancy filter (closed/maximal itemsets, minimal non-redundant rules)
  is applied here, so the explorer shows these alongside the genuine ones.
- **The objective's weights are a choice, not a finding.** The measures come
  from the literature; the 0.55/0.45/0.35 trade-off between lift, coverage and
  rule-set size is ours and is stated rather than justified empirically.
- **Hill climbing is local.** Three restarts over a discrete grid gives no
  guarantee of a global optimum, and the grid itself bounds what can be found.
- **`algorithm` is effectively a no-op in the search.** Apriori and FP-Growth
  are exact and return identical itemsets, so the axis only affects runtime.
  It is kept in the space deliberately — the search correctly finds it makes
  no difference to quality, which is the right answer.
- **Association rules are descriptive.** The recommender surfaces
  co-occurrence, not causation, and no online/A-B evaluation was performed.
- **Reproducibility is exact except for one float digit.** The generator and
  the pipeline are fully seeded and rule counts, coverage, stability and
  recovery are identical run to run; `mean_confidence` can differ in its last
  decimal place because rule ordering changes the floating-point summation
  order. This is a ~1e-16 effect and affects no reported figure.

## Files

```
04_associative_pattern_mining/
├── README.md
├── CRISP_DM.md
├── requirements.txt
├── app.py                              Streamlit admin dashboard
├── data/
│   ├── generate_data.py
│   └── grocery_transactions.csv        (generated, reproducible via fixed seed)
├── src/
│   ├── mining.py                       preparation + mining + metrics
│   ├── autoresearch.py                 hill-climbing hyperparameter search
│   └── run_experiment.py               CRISP-DM pipeline entrypoint
└── artifacts/                          (generated by src/run_experiment.py)
    ├── metrics.json                    all reported results
    ├── data_quality.json
    ├── item_frequency.csv
    ├── itemsets.csv
    ├── rules.csv
    ├── holdout_rules.csv
    └── autoresearch_history.csv / autoresearch_best.json
```
