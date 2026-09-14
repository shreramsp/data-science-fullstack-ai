# CRISP-DM — Master's Data Science Platform

## 1. Business Understanding

A retail analytics team wants five things from its transaction history:
customer segments for marketing, flagged suspicious bulk-order invoices,
churn-risk scores per customer, product-bundle recommendations, and fast
similarity search over a large product catalog. Those five needs map 1:1 onto
the five techniques covered here (clustering, anomaly
detection, supervised learning, association rule mining, sub-linear search),
so this project builds one coherent dataset and pipeline that exercises all
five honestly rather than five disconnected toy examples.

Success criteria, fixed before any modeling (checked for real in `app.py`'s
Evaluation tab): silhouette > 0.3 for clustering, recovery of the injected
bulk-order invoices for anomaly detection, ROC-AUC > 0.65 for churn
classification, at least one association rule with lift > 5, and a measurable
LSH speedup over brute force with reported recall.

## 2. Data Understanding

`data/generate_data.py` produces a **synthetic** online-retail transaction
log (numpy, seeded at 42) — not a scrape of any Kaggle file — sized so every
downstream technique has a real signal to find while staying small enough to
inspect by hand:

| Entity | Count |
|---|---|
| Line items | 7,083 |
| Invoices | 1,500 |
| Customers | 422 |
| Products | 300 (8 categories) |
| Date range | 2024-01-01 → 2024-12-30 |

Five customer archetypes (champion/loyal/regular/at-risk/dormant) drive
purchase frequency and recency bias, so RFM clustering has real structure to
recover. ~1% of invoices are generated as bulk orders (25-45 line items,
15-40 units each) for anomaly detection to find. Two products per category
are wired as a "bundle" pair with a 65% co-purchase reinforcement rate, so
association rule mining has genuine high-lift pairs to surface.

EDA (`app.py`, Phase 2 tab): revenue by category, monthly revenue trend, and
a right-skewed customer monetary-value histogram — the skew is the reason
RFM features are standardized before clustering/anomaly detection.

## 3. Data Preparation

`src/features.py` runs the cleaning checks a real dataset would need (drop
nulls in key columns, drop non-positive quantity/price, drop duplicates —
this synthetic set has none injected, so 0 rows are dropped; the report is
still generated and shown) and engineers two feature tables:

- **Customer features** (one row/customer): recency_days, frequency,
  monetary, avg_basket_value, product_diversity, category_diversity — feeds
  clustering and supervised learning.
- **Invoice features** (one row/invoice): n_lines, total_quantity,
  invoice_value, avg_unit_qty — feeds anomaly detection.

Baskets (`build_baskets`) are the per-invoice product-ID lists used by
association rule mining. All RFM/invoice features are standardized
(`StandardScaler`) before any distance-based model.

## 4. Modeling

**4a. Clustering (unsupervised)** — `src/clustering.py`. KMeans fit for
k=2..6 on standardized RFM features, model selected by silhouette score.

**4b. Anomaly detection** — `src/anomaly.py`. `IsolationForest`
(contamination=0.02) scored on the four invoice features, no labels used.

**4c. Supervised learning** — `src/supervised.py`. `RandomForestClassifier`
predicts `churn_risk` (1 if recency > population median). Recency is
excluded from the feature set — it defines the label, so including it would
let the model trivially re-derive rather than learn behavioral signal.
75/25 stratified train/test split.

**4d. Association rule mining** — `src/association.py`. `mlxtend`'s Apriori
(min_support=0.003) → `association_rules` filtered to lift ≥ 1.5.

**4e. Sub-linear search (LSH)** — `src/lsh.py`. Real 300-product co-purchase
vectors (invoice-indicator, one-hot over 1,500 invoices) are random-projected
to 32 dimensions, then expanded into a 20,100-item synthetic corpus (66 noisy
variants per real product) so a speedup is visible at a realistic catalog
scale. A random-hyperplane (SimHash) multi-table LSH index (8 tables × 8
hyperplanes) is benchmarked against brute-force cosine search over 40 query
products.

## 5. Evaluation

All numbers below are read directly from `artifacts/summary.json`, produced
by one run of `python3.13 src/pipeline.py` — nothing here is asserted without
a corresponding pipeline step:

| Technique | Metric | Result | Success criterion met? |
|---|---|---|---|
| Clustering | best silhouette (k) | 0.500 (k=2) | Yes (>0.3) |
| Anomaly detection | invoices flagged | 30 / 1,500 (2.0%, ~ the ~1% injection rate) | Yes |
| Supervised (churn-risk) | ROC-AUC / accuracy / F1 | 0.783 / 0.689 / 0.680 | Yes (>0.65) |
| Association rules | rules found / top lift | 82 rules, top lift ≈ 35 | Yes (>5) |
| LSH search | speedup / recall@5 / candidate fraction | 8.2x / 0.64 / 4.5% of corpus | Yes (measurable speedup, recall reported) |

**Honest limitations:**
- Silhouette peaks at the coarse k=2 split; k=4-5 give more actionable
  marketing segments at a modest silhouette cost (0.42-0.46) — this
  trade-off is shown in the dashboard's silhouette curve, not resolved away.
- The LSH benchmark's 20K-item corpus is a synthetic expansion around real
  product embeddings, built specifically to make the sub-linear speedup
  visible; it is not a performance claim about the 300-product catalog on
  its own (which is far too small for brute force to be a bottleneck).
- Recall@5 of 0.64 at 8.2x speedup is a real, tunable trade-off in this LSH
  implementation (more tables/fewer hyperplanes per table raises recall and
  lowers speedup) — the exact numbers were chosen honestly from a small
  parameter sweep, not fabricated.
- Synthetic data means no true seasonality, promotions, or return behavior —
  patterns a real retail log would also contain.

## 6. Deployment

`app.py` is a nine-page Streamlit dashboard (one page per CRISP-DM
sub-phase, in textbook order) that reads every number and table from
`artifacts/` — it computes nothing itself, so the app and the pipeline can
never silently disagree. Each phase page ends with a short, static
conceptual quiz (`src/quizzes.py`) on that phase's core idea. For a
compact local project, the working, reproducible, interactive dashboard is
the deployment artifact.
