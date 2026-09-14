"""The skill catalog: all 46 skills from the two installed packs.

Skill names, categories and pack membership mirror the public structure of
`param087/agent-ml-skills` (15 skills) and `nimrodfisher/data-analytics-skills`
(31 unique skills across 6 categories). The executors are implemented locally
for this project.
"""
from __future__ import annotations

from skills import analytics_skills as A
from skills import ml_skills as M
from skills.base import Skill

ML_PACK = "agent-ml-skills"
DA_PACK = "data-analytics-skills"

BD, DU, DP, MO, EV, DE = (
    "1. Business Understanding", "2. Data Understanding", "3. Data Preparation",
    "4. Modeling", "5. Evaluation", "6. Deployment",
)

SKILLS: list[Skill] = [
    # ---------------- agent-ml-skills (15) ----------------
    Skill("exploratory-data-analysis", "Exploratory Data Analysis", ML_PACK,
          "ml-workflow", DU,
          "Profile the target, distributions and churn drivers before modeling.", M.eda),
    Skill("data-cleaning", "Data Cleaning", ML_PACK, "ml-workflow", DP,
          "Detect, decide and log every quality repair with row counts.", M.data_cleaning),
    Skill("feature-engineering", "Feature Engineering", ML_PACK, "ml-workflow", DP,
          "Derive features and measure whether they actually help.", M.feature_engineering),
    Skill("sklearn-pipelines", "Scikit-learn Pipelines", ML_PACK, "ml-workflow", MO,
          "Compose leakage-safe preprocessing and estimation in one object.",
          M.sklearn_pipelines),
    Skill("model-evaluation", "Model Evaluation", ML_PACK, "ml-workflow", EV,
          "Held-out metrics, ROC/PR curves, confusion matrix and calibration.",
          M.model_evaluation),
    Skill("hyperparameter-tuning", "Hyperparameter Tuning", ML_PACK, "ml-workflow", MO,
          "Cross-validated grid search with an honest held-out confirmation.",
          M.hyperparameter_tuning),
    Skill("imbalanced-data", "Imbalanced Data", ML_PACK, "ml-workflow", MO,
          "Class weighting and threshold moving under a 26.5% positive rate.",
          M.imbalanced_data),
    Skill("experiment-tracking", "Experiment Tracking", ML_PACK, "ml-workflow", MO,
          "Log every run with its seed, split, params and metrics.",
          M.experiment_tracking),
    Skill("ml-debugging", "ML Debugging", ML_PACK, "ml-workflow", EV,
          "Leakage scan with a positive control, learning curve, duplicate check.",
          M.ml_debugging),
    Skill("reproducible-ml", "Reproducible ML", ML_PACK, "ml-workflow", EV,
          "Seed pinning, data fingerprinting and a verified identical re-run.",
          M.reproducible_ml),
    Skill("pandas-patterns", "Pandas Patterns", ML_PACK, "ml-workflow", DP,
          "Idiomatic chaining, named aggregation and measured vectorization gains.",
          M.pandas_patterns),
    Skill("model-serving", "Model Serving", ML_PACK, "ml-workflow", DE,
          "Serving contract and measured p50/p95/p99 scoring latency.",
          M.model_serving),
    Skill("pytorch-training-loop", "PyTorch Training Loop", ML_PACK, "ml-workflow", MO,
          "The forward/loss/backward/step loop, implemented and run in NumPy.",
          M.pytorch_training_loop),
    Skill("llm-finetuning", "LLM Fine-tuning", ML_PACK, "ml-workflow", MO,
          "Build and profile an SFT corpus from real rows; no fine-tune executed.",
          M.llm_finetuning),
    Skill("rag-pipeline", "RAG Pipeline", ML_PACK, "ml-workflow", DE,
          "Real TF-IDF retrieval over the lab's findings with cited chunks.",
          M.rag_pipeline),

    # ------- data-analytics-skills / 01 data quality & validation -------
    Skill("data-quality-audit", "Data Quality Audit", DA_PACK,
          "01 data quality & validation", DU,
          "Score completeness, validity, uniqueness, consistency and timeliness.",
          A.data_quality_audit),
    Skill("programmatic-eda", "Programmatic EDA", DA_PACK,
          "01 data quality & validation", DU,
          "Automated per-column profile: types, missingness, cardinality, outliers.",
          A.programmatic_eda),
    Skill("metric-reconciliation", "Metric Reconciliation", DA_PACK,
          "01 data quality & validation (also 02)", EV,
          "Reconcile four defensible 'churn rate' figures to one canonical definition.",
          A.metric_reconciliation),
    Skill("query-validation", "Query Validation", DA_PACK,
          "01 data quality & validation", DP,
          "Lint a query for anti-patterns, then prove the rewrite returns the same answer.",
          A.query_validation),
    Skill("schema-mapper", "Schema Mapper", DA_PACK,
          "01 data quality & validation (also 02)", DP,
          "Map every source column to the analytical model with type and role.",
          A.schema_mapper),

    # ------- 02 documentation & knowledge -------
    Skill("analysis-assumptions-log", "Analysis Assumptions Log", DA_PACK,
          "02 documentation & knowledge", BD,
          "Register every assumption with impact, evidence and validation status.",
          A.analysis_assumptions_log),
    Skill("analysis-documentation", "Analysis Documentation", DA_PACK,
          "02 documentation & knowledge", EV,
          "Generate the analysis document with every figure interpolated live.",
          A.analysis_documentation),
    Skill("data-catalog-entry", "Data Catalog Entry", DA_PACK,
          "02 documentation & knowledge", DU,
          "Catalog the dataset: grain, key, provenance, PII and caveats.",
          A.data_catalog_entry),
    Skill("semantic-model-builder", "Semantic Model Builder", DA_PACK,
          "02 documentation & knowledge", BD,
          "Define the metric and dimension layer, then evaluate it to prove it resolves.",
          A.semantic_model_builder),
    Skill("sql-to-business-logic", "SQL to Business Logic", DA_PACK,
          "02 documentation & knowledge", BD,
          "Extract the business rules a query encodes and verify them by execution.",
          A.sql_to_business_logic),

    # ------- 03 data analysis & investigation -------
    Skill("ab-test-analysis", "A/B Test Analysis", DA_PACK,
          "03 data analysis & investigation", EV,
          "Two-proportion z-test with CI — and why this one is not causal.",
          A.ab_test_analysis),
    Skill("business-metrics-calculator", "Business Metrics Calculator", DA_PACK,
          "03 data analysis & investigation", BD,
          "MRR, ARPU, revenue churn and an LTV estimate with its caveat.",
          A.business_metrics_calculator),
    Skill("cohort-analysis", "Cohort Analysis", DA_PACK,
          "03 data analysis & investigation", DU,
          "Quarterly signup cohorts and a lifecycle-stage retention matrix.",
          A.cohort_analysis),
    Skill("funnel-analysis", "Funnel Analysis", DA_PACK,
          "03 data analysis & investigation", DU,
          "Service-adoption funnel with step conversion and largest leak.",
          A.funnel_analysis),
    Skill("root-cause-investigation", "Root Cause Investigation", DA_PACK,
          "03 data analysis & investigation", DU,
          "Decompose the overall churn rate into per-segment contributions.",
          A.root_cause_investigation),
    Skill("segmentation-analysis", "Segmentation Analysis", DA_PACK,
          "03 data analysis & investigation", MO,
          "k-means segments profiled by churn rate, tenure, spend and MRR.",
          A.segmentation_analysis),
    Skill("time-series-analysis", "Time Series Analysis", DA_PACK,
          "03 data analysis & investigation", DU,
          "Monthly cohort series, moving average and trend — with survivorship flagged.",
          A.time_series_analysis),

    # ------- 04 storytelling & visualization -------
    Skill("insight-synthesis", "Insight Synthesis", DA_PACK,
          "04 storytelling & visualization", EV,
          "Rank every candidate finding by reach x effect size.", A.insight_synthesis),
    Skill("data-narrative-builder", "Data Narrative Builder", DA_PACK,
          "04 storytelling & visualization", EV,
          "Assemble an SCQA narrative from computed figures only.",
          A.data_narrative_builder),
    Skill("executive-summary-generator", "Executive Summary Generator", DA_PACK,
          "04 storytelling & visualization", DE,
          "One page: bottom line, prediction, action, caveats.",
          A.executive_summary_generator),
    Skill("visualization-builder", "Visualization Builder", DA_PACK,
          "04 storytelling & visualization", EV,
          "Build four charts and justify each encoding choice.", A.visualization_builder),
    Skill("dashboard-specification", "Dashboard Specification", DA_PACK,
          "04 storytelling & visualization", DE,
          "Spec the monitoring dashboard: audience, zones, metrics, out-of-scope.",
          A.dashboard_specification),

    # ------- 05 stakeholder communication -------
    Skill("analysis-qa-checklist", "Analysis QA Checklist", DA_PACK,
          "05 stakeholder communication", EV,
          "Ten pre-delivery checks executed against live objects, not asserted.",
          A.analysis_qa_checklist),
    Skill("impact-quantification", "Impact Quantification", DA_PACK,
          "05 stakeholder communication", DE,
          "Campaign cost, revenue saved and net value across thresholds.",
          A.impact_quantification),
    Skill("methodology-explainer", "Methodology Explainer", DA_PACK,
          "05 stakeholder communication", EV,
          "Six modeling concepts in plain language, anchored to real numbers.",
          A.methodology_explainer),
    Skill("stakeholder-requirements-gathering", "Stakeholder Requirements Gathering",
          DA_PACK, "05 stakeholder communication", BD,
          "Intake questions, answers and the analytical consequence of each.",
          A.stakeholder_requirements_gathering),
    Skill("technical-to-business-translator", "Technical to Business Translator",
          DA_PACK, "05 stakeholder communication", DE,
          "Restate every metric as a decision, cost or missed opportunity.",
          A.technical_to_business_translator),

    # ------- 06 workflow optimization -------
    Skill("analysis-planning", "Analysis Planning", DA_PACK,
          "06 workflow optimization", BD,
          "Sequence the work, size the effort, name what is out of scope.",
          A.analysis_planning),
    Skill("analysis-retrospective", "Analysis Retrospective", DA_PACK,
          "06 workflow optimization", EV,
          "What this lab run should keep doing and what it should change.",
          A.analysis_retrospective),
    Skill("context-packager", "Context Packager", DA_PACK,
          "06 workflow optimization", DE,
          "Assemble the handoff bundle: data hash, split, metrics, open questions.",
          A.context_packager),
    Skill("peer-review-template", "Peer Review Template", DA_PACK,
          "06 workflow optimization", EV,
          "Review this project against ten areas and record the concerns found.",
          A.peer_review_template),
]

BY_ID = {s.id: s for s in SKILLS}
CRISP_PHASES = [BD, DU, DP, MO, EV, DE]


def by_pack(pack: str) -> list[Skill]:
    return [s for s in SKILLS if s.pack == pack]


def by_phase(phase: str) -> list[Skill]:
    return [s for s in SKILLS if s.crisp_dm == phase]


def categories(pack: str) -> list[str]:
    seen: list[str] = []
    for s in by_pack(pack):
        if s.category not in seen:
            seen.append(s.category)
    return seen
