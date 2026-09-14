# CRISP-DM in the Skills Mastery Lab

Every installed skill is demonstrated and mapped to CRISP-DM. Rather than
bolting a process diagram onto a
pile of demos, every one of the 46 skills is registered with the CRISP-DM phase
it actually serves (`skills/registry.py`), and the app's **CRISP-DM** tab builds
its phase map from that registry — so the documentation cannot drift away from
the code.

| Phase | Skills | Where the work happens |
|---|---|---|
| 1. Business Understanding | 6 | `analysis-planning`, `stakeholder-requirements-gathering`, `semantic-model-builder`, `business-metrics-calculator`, `sql-to-business-logic`, `analysis-assumptions-log` |
| 2. Data Understanding | 8 | `exploratory-data-analysis`, `programmatic-eda`, `data-quality-audit`, `data-catalog-entry`, `cohort-analysis`, `funnel-analysis`, `root-cause-investigation`, `time-series-analysis` |
| 3. Data Preparation | 5 | `data-cleaning`, `feature-engineering`, `pandas-patterns`, `query-validation`, `schema-mapper` |
| 4. Modeling | 7 | `sklearn-pipelines`, `hyperparameter-tuning`, `imbalanced-data`, `experiment-tracking`, `segmentation-analysis`, `pytorch-training-loop`, `llm-finetuning` |
| 5. Evaluation | 13 | `model-evaluation`, `ml-debugging`, `reproducible-ml`, `metric-reconciliation`, `analysis-qa-checklist`, `peer-review-template`, `insight-synthesis`, `methodology-explainer`, `ab-test-analysis`, `data-narrative-builder`, `visualization-builder`, `analysis-documentation`, `analysis-retrospective` |
| 6. Deployment | 7 | `model-serving`, `rag-pipeline`, `impact-quantification`, `executive-summary-generator`, `dashboard-specification`, `technical-to-business-translator`, `context-packager` |

## 1. Business Understanding

**Decision the project serves:** which telecom subscribers receive a retention
offer in the next campaign cycle. That framing fixes the deliverable as a ranked
scoring list rather than a report, and it makes the decision threshold — not the
model family — the most consequential choice in the project.

Requirements intake recorded one unanswered question (campaign budget), which is
why every dollar figure in the lab is labelled as resting on a stated,
unmeasured assumption. The metric layer (`churn_rate`, `arpu`, `mrr_active`,
`avg_tenure_months`, `addon_attach_rate`) is defined here and evaluated live, so
a definition that does not resolve fails loudly instead of silently.

## 2. Data Understanding

7,043 subscribers, 21 source columns, 26.54% churn. Quality scoring across five
dimensions returns 99.9% overall; the one genuine defect is 11 blank
`TotalCharges` values, all belonging to tenure-0 accounts that have never been
billed. Timeliness is not assessable at all — a static extract carries no load
timestamp — and is flagged as such rather than scored as clean.

Contribution decomposition (`root-cause-investigation`) attributes the churn
rate to segments: month-to-month contracts alone contribute +8.90 percentage
points of excess over the base rate.

## 3. Data Preparation

Cleaning is a single documented function (`src/data_prep.py: clean`) that
returns both the cleaned frame and an issue log with row counts and the action
taken. Blank `TotalCharges` are imputed as `tenure x MonthlyCharges` rather than
dropped — dropping them would have removed the newest customers, the
highest-risk segment, and biased every downstream skill.

Seven features are engineered. Their measured effect on 3-fold CV ROC-AUC is
+0.0014, which is reported as-is: their real contribution is interpretability,
not score.

## 4. Modeling

One `ColumnTransformer` pipeline (median/mode imputation → standard scaling →
one-hot encoding) feeding logistic regression, fitted on a 75% stratified split
with `random_state=42`. Everything transformative lives inside the pipeline, so
no preprocessing statistic is ever fitted on the test split.

Grid search over 8 configurations moves CV ROC-AUC by less than 0.002. The
threshold sweep moves F1 from 0.570 to 0.634. That comparison is the modeling
phase's main finding: on this dataset the decision threshold matters far more
than hyperparameters.

## 5. Evaluation

Held-out results on 1,761 unseen customers:

| Metric | Value |
|---|---|
| ROC-AUC | 0.849 |
| PR-AUC | 0.654 |
| Accuracy | 0.798 |
| Precision | 0.654 |
| Recall | 0.505 |
| F1 | 0.570 |
| Brier | 0.135 |

PR-AUC is reported alongside ROC-AUC because a 26.5% base rate makes accuracy
misleading — predicting "nobody churns" scores 73.5% accuracy and is worthless.

Validation is adversarial by design: the leakage scan runs with an artificial
leaky column injected as a positive control (it is flagged; no real feature is),
the learning curve shows a +0.006 AUC train-validation gap, the QA checklist
asserts against live objects including train/test index overlap, and the peer
review records three standing concerns against this project's own work.

## 6. Deployment

The Streamlit app is the deployment. **Live inference** calls the same fitted
pipeline object the reported metrics come from, so training and serving
preprocessing cannot diverge. Measured in-process latency is p50 ≈ 3 ms,
p95 ≈ 4 ms per single-row request; batching is roughly two orders of magnitude
cheaper per row.

The deployment phase also produces the artifacts a real handoff needs: an
executive summary, a dashboard specification, a technical-to-business
translation table, and a context package pinning the dataset hash, seed, split
sizes and the single command that regenerates everything.
