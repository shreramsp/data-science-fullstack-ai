# 03 — Customer Intelligence & Segmentation Clustering

An independent, end-to-end CRISP-DM clustering project: a retail transaction log →
RFM feature engineering → an **AutoResearch hill-climbing search** over 180
clustering pipelines, scored with published cluster-validity indices → a Streamlit
admin dashboard for the resulting six customer segments.

This project uses an original clustering pipeline, interface, documentation,
and reported results.

## What was built

| File | Purpose |
|---|---|
| `data/generate_data.py` | Seeded transaction-log generator (Online Retail schema + its defect profile). |
| `src/features.py` | Cleaning rules with a row-level audit trail; RFM + derived per-customer features; RFM quintile baseline. |
| `src/autoresearch.py` | The hill-climbing search: search space, the composite objective, the actionability constraint, the literature table. |
| `src/train.py` | CRISP-DM pipeline — prepare → search → refit → evaluate → persist artifacts. |
| `app.py` | Six-tab Streamlit admin dashboard. |
| `CRISP_DM.md` | The six phases mapped to concrete decisions and code. |

### On the dataset

The canonical clustering dataset for this use case is **Online Retail / Online
Retail II** (UCI, mirrored on Kaggle) — a ~45 MB
authenticated download, which breaks a plain `git clone && pip install && run`.
To keep the workflow reproducible, this project **generates** a
small seeded transaction log with the same schema (`InvoiceNo`, `StockCode`,
`Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`) and
the same defect profile that makes the real dataset a good teaching case: guest
checkouts with no customer id, `C`-prefixed cancellation invoices, zero-price rows,
and duplicated lines. 60,221 lines / 1,277 customers over two years.

**Every number in this README describes that generated dataset.** None of it is a
claim about real retail customers or about any published result.

## Setup

```bash
cd 03_customer_segmentation_clustering
python3.13 -m venv .venv          # 3.13 recommended; sklearn/pandas wheels
source .venv/bin/activate         # are not yet published for 3.14
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate the transaction log (train.py does this automatically if missing)
python data/generate_data.py

# 2. Run the CRISP-DM pipeline: search, refit, evaluate, write artifacts/
python src/train.py                    # ~12 s; add --skip-exhaustive to halve it

# 3. Launch the dashboard
streamlit run app.py
```

Then open the printed local URL (default `http://localhost:8501`).

Useful flags: `--restarts N` (hill-climbing restarts, default 6), `--skip-exhaustive`
(skip the full-grid verification run).

## Approach

### AutoResearch: what actually searches

One candidate is a **complete pipeline**, not just a value of *k*:

```
feature_set    rfm | rfm_extended          (RFM, or RFM + AOV/basket/tenure/return-rate)
log_transform  False | True                (log1p, for the right-skewed RFM columns)
scaler         standard | robust | minmax
algorithm      kmeans | agglomerative(ward) | gaussian mixture
k              4 .. 8
                                            = 180 candidate pipelines
```

**Steepest-ascent hill climbing with random restarts.** From a starting
configuration, evaluate every neighbour one coordinate away, move to the best
strictly-improving one, and stop when none improves — that restart has found a local
optimum. Six restarts; every fit is cached, so the reported cost is the true number
of distinct model fits.

**The objective** blends three published internal indices, each rescaled to ~[0,1]
and each monotone in the direction its index prefers:

```
J = 0.50 · (s + 1)/2  +  0.25 · 1/(1 + DB)  +  0.25 · CH/(CH + 500)
```

`s` = silhouette (Rousseeuw 1987), `DB` = Davies–Bouldin (1979), `CH` =
Calinski–Harabasz (1974).

**The actionability constraint**, declared *before* the search ran: `k ∈ [4, 8]` and
no segment below 3% of the customer base. This matters, and the project reports why
rather than hiding it — see "What the constraint is doing" below.

### Matching the dashboard to the literature

The dashboard's "Research grounding" tab lists every index and algorithm it uses
against its original source (Rousseeuw 1987; Calinski & Harabasz 1974; Davies &
Bouldin 1979; Hughes 1994; Arthur & Vassilvitskii 2007; Ward 1963; Dempster et al.
1977; Thorndike 1953; Russell & Norvig on local search). Those are citations for the
*methods*. The numbers on the dashboard come from this project's own run and are not
comparable to results reported in any of those papers.

## Honest results

From the last `python src/train.py` run (seed 20260913, ~12 s):

**Search.** 80 distinct pipelines fitted out of 180 in the space. The exhaustive grid
was then run as a *verification* pass the search never sees: hill climbing reached
the constrained global optimum, optimality gap **0.0000**, having skipped 100 fits.
161 of the 180 configurations satisfied the actionability constraint.

**Winning pipeline:** `rfm` features, no log transform, min-max scaling, k-means, k=6.

| Index | Value | Direction |
|---|---|---|
| Silhouette | 0.541 | higher better |
| Davies–Bouldin | 0.637 | lower better |
| Calinski–Harabasz | 2,259.7 | higher better |
| Composite objective | 0.7427 | higher better |

**Segments** (1,276 customers, $2,797,297 of revenue in the window):

| Segment | Customers | Median recency | Median freq. | Median spend | Revenue share | Silhouette |
|---|---|---|---|---|---|---|
| Champions | 86 | 14 d | 32 | $10,655 | 33.9% | 0.574 |
| Loyal Regulars | 253 | 28 d | 16 | $2,679 | 28.9% | 0.394 |
| Big-Ticket Buyers | 94 | 51 d | 5 | $5,134 | 18.4% | 0.457 |
| At Risk | 314 | 258 d | 5 | $743 | 9.6% | 0.491 |
| Promising / New | 373 | 24 d | 2 | $323 | 8.8% | 0.655 |
| Hibernating | 156 | 488 d | 1 | $63 | 0.4% | 0.639 |

**Validation.** Clustering has no labels, so there is no test split. Three checks
instead:

- **Subsample stability:** refitting on ten 80% subsamples gives adjusted Rand index
  **0.990 ± 0.008** (min 0.976) against the full-data labels — the same customers
  group together when the data changes.
- **Against the cheap baseline:** ARI **0.278** with the classic 1–5 RFM quintile
  tiers. Related to the baseline but not a restatement of it, which is the point —
  the clusters are not axis-aligned cuts.
- **External sanity check:** the generator drew customers from six latent archetypes
  that the pipeline never sees. Recovery is ARI **0.481** / NMI **0.660**. This is
  *not* a benchmark: the archetypes overlap heavily in RFM space (the crosstab in the
  dashboard shows `champion` splitting across Champions and Loyal Regulars, and
  `hibernating` splitting across Hibernating and At Risk), so partial agreement is
  the expected outcome. It confirms the clustering is not noise; it proves nothing
  more.

### What the constraint is doing — stated plainly

Without the `k ∈ [4, 8]` bound, this search does **not** return six segments. Scanning
the best achievable composite at each k below the feasible band (written to
`metrics.json.search.unconstrained_reference` on every run) shows both k=2 and k=3
scoring *above* the k=6 winner:

| k | Best composite | Best pipeline | Smallest segment |
|---|---|---|---|
| 2 | **0.8035** | rfm_extended, robust, k-means | 10.9% |
| 3 | 0.7766 | rfm_extended, robust, agglomerative | 5.8% |
| 6 (winner) | 0.7427 | rfm, min-max, k-means | 6.7% |

A two-way split is a valid geometric optimum and not a segmentation. k=3 scores well
too, but the free RFM quintile tiers already rank customers into good/average/poor,
so a clustering needs to resolve more than that to earn its place.

So the honest framing is: **the constraint, not the objective, is what produces six
segments.** It was declared before the search ran rather than chosen after seeing
results, and both the elbow curve and the full-grid leaderboard are shipped in the
dashboard so the trade-off is inspectable rather than asserted. Change `MIN_K` /
`MAX_K` in `src/autoresearch.py` and the winner changes.

### Other limitations

- **The data is synthetic.** Real transaction logs have seasonality, promotions,
  product-mix effects and fraud that this generator does not model.
- **Internal indices measure geometry, not money.** A high silhouette says the
  segments are well separated in RFM space. It does not say a campaign aimed at them
  will convert.
- **Segment names come from a template match** (Hungarian assignment against six
  marketing archetypes on rank-normalised R/F/M), not from the algorithm. They are a
  readability aid; a data shift can rename a segment without the cluster changing
  much.
- **Loyal Regulars has the weakest silhouette (0.394)** — it sits between Champions
  and Promising / New rather than occupying clean space of its own, so its boundary
  is the least trustworthy part of the solution.
- **The search space is coarse.** Three scalers, three algorithms, one linear
  transform, no dimensionality reduction, no per-algorithm hyperparameters beyond
  `k`.

## Validation performed

- `python data/generate_data.py` and `python src/train.py` run clean end to end.
- Hill climbing was checked against the exhaustive 180-configuration grid (it reached
  the global optimum with zero gap).
- The dashboard was executed headlessly with all 10 Altair specs compiled and all 16
  dataframes Arrow-serialised, then rendered in a real headless Chrome over the
  DevTools protocol: 0 skeletons, 0 exceptions, all six tabs visited.
- Nearest-centroid scoring reproduces the training segment for 100% of training
  customers, and manual inputs land where they should (recent/frequent/high-spend →
  Champions; 600 days lapsed → Hibernating).

## Files

```
03_customer_segmentation_clustering/
├── README.md
├── CRISP_DM.md
├── requirements.txt
├── app.py
├── .streamlit/config.toml          (light theme — the chart palette is validated for it)
├── data/
│   ├── generate_data.py
│   ├── online_retail.csv           (generated, reproducible via fixed seed)
│   └── customer_archetypes.csv     (generated; external check only, never fitted on)
├── src/
│   ├── features.py
│   ├── autoresearch.py
│   └── train.py
└── artifacts/                      (generated by src/train.py)
    ├── best_pipeline.joblib
    ├── metrics.json
    ├── customer_segments.csv
    ├── segment_profile.csv
    ├── search_history.csv
    ├── grid_results.csv
    ├── cleaning_report.csv
    └── archetype_crosstab.csv
```
