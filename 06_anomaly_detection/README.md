# 06 — Autonomous Anomaly Detection Platform

Unsupervised anomaly detection on a credit-card-fraud-shaped transaction stream,
built end to end under CRISP-DM: cleaning → five classic detectors → an
**autoresearch hill-climbing search** over the whole pipeline space → honest
held-out evaluation → a Streamlit **data-science admin dashboard** that serves
the saved model live.

This implementation uses an original detection pipeline, interface,
documentation, metrics, and reported results.

## What was built

| File | Role |
|---|---|
| `data/load_data.py` | Dataset resolution: real Kaggle `creditcard.csv` if present, else a seeded synthetic stand-in with the same schema. |
| `src/features.py` | Cleaning rules with a reported ledger, derived features, stratified 60/20/20 split, the three feature sets and four scalers. |
| `src/detectors.py` | Five classic detectors behind one `fit`/`score` interface (higher = more anomalous). |
| `src/autoresearch.py` | Multi-restart steepest-ascent hill climbing over the joint pipeline space; `LITERATURE` table backing every design decision. |
| `src/train.py` | The CRISP-DM run: clean → baseline sweep → search → operating point → one test evaluation → artifacts. |
| `app.py` | Six-tab Streamlit admin dashboard: Overview, Data, Methods, AutoResearch, Evaluation, Live scoring. |
| `CRISP_DM.md` | The six phases mapped to this project's concrete decisions. |

## Setup

```bash
cd 06_anomaly_detection
python3.13 -m venv .venv          # 3.13 recommended; sklearn/pandas wheels
source .venv/bin/activate         # were not yet available for 3.14
pip install -r requirements.txt
```

## Run

```bash
# 1. Generate the dataset (train.py also does this automatically if missing)
python data/load_data.py

# 2. Full CRISP-DM run: baselines, hill climbing, evaluation, artifacts (~2.5 min)
python src/train.py

# 3. Launch the dashboard
streamlit run app.py
```

Step 2 is required before step 3 — the dashboard reads `artifacts/` and refuses
to invent numbers, so on a fresh clone it will tell you to run the pipeline
first. The run is deterministic: from the fixed seed it reproduces the table
below bit-for-bit.

Then open the printed URL (default `http://localhost:8501`).

To run everything against the **real** Kaggle data instead, download
`creditcard.csv` from the ULB "Credit Card Fraud Detection" dataset, drop it in
`data/`, and re-run step 2 — every panel regenerates against it, and the
dashboard header changes from `synthetic-fallback` to `kaggle-creditcard`.

## Approach

Unsupervised detection, because fraud labels arrive late or never. Labels are
used to stratify the split, to *score* candidates during the search, and to
evaluate — **never to fit a detector**.

**Objective: average precision, not ROC-AUC.** At 0.5% prevalence ROC-AUC is
dominated by the negative class. This run demonstrates the problem concretely:
all five methods land between 0.934 and 0.961 test ROC-AUC while their average
precision spans 0.130 to 0.267 — a 2x quality spread that the ROC number erases.
(Davis & Goadrich, ICML 2006; Saito & Rehmsmeier, PLOS ONE 2015.)

**AutoResearch.** Hill climbing over *feature set × scaler × detection method ×
that method's hyperparameters* — 3 × 4 × 5 methods with their own grids, so the
detection method is itself a search dimension. One move changes exactly one
decision; steepest ascent; random restarts to escape local optima. Every
evaluation is logged including rejected moves.

**Method pool and sources** (all real, all classic): Isolation Forest (Liu,
Ting & Zhou 2008), Local Outlier Factor (Breunig et al. 2000), One-Class SVM
(Schölkopf et al. 2001), Elliptic Envelope / MCD (Rousseeuw & Van Driessen
1999), PCA reconstruction error (Shyu et al. 2003). The citations are to the
original papers for the *methods*; **every number in this project comes from its
own run**, never from a paper's reported results.

## Honest results

From the last `python src/train.py` run (synthetic fallback, seed 20260913,
runtime 151.2 s). Data: 60,090 raw rows → 60,000 after cleaning (90 duplicates
dropped, 40 negative amounts corrected, 120 missing amounts imputed), 300
anomalies = 0.500% prevalence; split 36,000 / 12,000 / 12,000.

**Baseline sweep — all five detectors at default settings:**

| Detector | Valid AP | Test AP | Test ROC-AUC | Fit+score |
|---|---|---|---|---|
| Elliptic Envelope | 0.2719 | 0.2672 | 0.961 | 5.3 s |
| One-Class SVM | 0.2639 | 0.2523 | 0.959 | 0.4 s |
| PCA reconstruction | 0.2421 | 0.2423 | 0.934 | 0.03 s |
| Local Outlier Factor | 0.1586 | 0.1639 | 0.959 | 6.2 s |
| Isolation Forest | 0.1169 | 0.1304 | 0.942 | 0.8 s |

**AutoResearch:** 3 restarts, 140 evaluations logged, 69 distinct pipelines
fitted. Winner: `elliptic[support_fraction=1.0] | components_amount_time |
standard` — validation AP 0.2819.

**Held-out test split (scored once):**

| Metric | Value |
|---|---|
| Average precision (PR-AUC) | **0.2816** |
| ROC-AUC | 0.9733 |
| Precision @ 0.5% alert budget | 0.250 |
| Recall @ 0.5% alert budget | 0.250 |

At the deployed operating point — threshold 149.13, fixed on validation and
applied unchanged to test — the model raised 40 alerts on 12,000 transactions
(0.33% alert rate): 12 true positives, 28 false alarms, 48 anomalies missed.
Precision 30.0%, recall 20.0%.

**Limitations, stated plainly:**

- **The data is synthetic.** These metrics describe the generated fallback
  dataset, not the real ULB benchmark, and are not comparable to any published
  credit-card-fraud result. The generator's difficulty was tuned once (to avoid
  a trivially separable or hopeless problem) and then left alone.
- **The autoresearch lift is small and honestly so.** Best baseline → searched
  pipeline is +0.0144 test AP (0.2672 → 0.2816). All three restarts converged on
  essentially the same pipeline, which means the landscape here is easy, not
  that the search is powerful. A search that found a large lift on this data
  would be more suspicious, not less.
- **The threshold under-delivered on transfer.** Set at the 0.5% budget on
  validation, it raised only 0.33% alerts on test — the score scale shifted
  slightly between splits. That is a real, visible instance of the threshold
  drift an operations team has to monitor, so it is reported rather than papered
  over by re-fitting the threshold on test.
- **Validation numbers are optimistically biased** by the search that selected
  on them. Quote the test row.
- **Absolute precision is low in the way this problem always is**: at 0.5%
  prevalence, most alerts are false and most anomalies are missed. The
  Evaluation tab's threshold explorer exists to make that trade explicit rather
  than hide it.
- **The cost model is a decision aid**, driven entirely by user-entered costs.
  It is not a measured or claimed saving.
- One-Class SVM is fitted on a 6,000-row subsample for tractability; this is
  documented in code, not hidden.
- No Docker, authentication, database, or cloud deployment; the current
  implementation runs locally with Streamlit.

## Dashboard

Six tabs, all reading `artifacts/` — the dashboard computes no metric of its own
beyond the curves and threshold sweep derived from saved test scores.

- **Overview** — headline KPIs, selected pipeline, test confusion matrix, and an
  explicit note on which numbers are biased and which are not.
- **Data** — cleaning ledger with row counts, split plan, per-feature separation
  diagnostic with its one-sided-AUC caveat.
- **Methods** — the five-detector baseline leaderboard and the baseline →
  autoresearch lift, including why ROC-AUC and AP disagree.
- **AutoResearch** — the full search trajectory (accepted *and* rejected moves),
  the accepted-move climb, and the literature table behind each design decision.
- **Evaluation** — PR curve vs the prevalence floor, ROC curve, log-scaled score
  distribution by class, and an interactive threshold/cost explorer.
- **Live scoring** — loads `best_pipeline.joblib` and scores a real held-out
  transaction with editable amount and hour, showing the alert decision against
  ground truth.

Chart colours use a CVD-validated categorical palette; series identity is never
carried by colour alone (legends plus mark shape on the search trajectory), and
the score distribution is log-scaled because a linear axis erases a 0.5% class.

## Files

```
06_anomaly_detection/
├── README.md
├── CRISP_DM.md
├── requirements.txt
├── app.py
├── data/
│   ├── load_data.py
│   └── transactions.csv          (generated; git-ignored, 32 MB, reproducible from the seed)
├── src/
│   ├── features.py
│   ├── detectors.py
│   ├── autoresearch.py
│   └── train.py
└── artifacts/                    (generated by src/train.py)
    ├── metrics.json              ┐
    ├── baselines.csv            │ committed — the evidence behind
    ├── search_history.csv       │ every number quoted above
    ├── cleaning_report.csv      │
    ├── feature_diagnostics.csv  ┘
    ├── test_scores.csv           ┐ git-ignored; regenerated by step 2,
    └── best_pipeline.joblib      ┘ and the dashboard needs them
```
