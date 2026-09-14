"""Static conceptual quizzes, one small set per CRISP-DM phase / technique."""
from __future__ import annotations

QUIZZES: dict[str, list[dict]] = {
    "business_understanding": [
        {
            "question": "In CRISP-DM, what is the primary goal of the Business Understanding phase?",
            "options": [
                "Train the first candidate model",
                "Translate a business problem into data science objectives and success criteria",
                "Clean missing values in the raw data",
                "Deploy the model to production",
            ],
            "answer": 1,
            "explanation": "Business Understanding sets the objective and success metric *before* any "
                            "data work — everything downstream is judged against it.",
        },
        {
            "question": "Which is a well-formed data science success criterion for this project?",
            "options": [
                "\"Make the dashboard look nice\"",
                "\"Use scikit-learn\"",
                "\"Flag high-risk-of-churn customers with recall ≥ 0.6 on held-out data\"",
                "\"Collect more data\"",
            ],
            "answer": 2,
            "explanation": "A good criterion is measurable, tied to the business need, and evaluated on "
                            "data the model didn't train on.",
        },
    ],
    "data_understanding": [
        {
            "question": "Why compute per-customer Recency, Frequency, and Monetary (RFM) values in Data Understanding?",
            "options": [
                "They are required by scikit-learn",
                "They compress raw transaction history into signals that summarize customer behavior",
                "They remove the need for a train/test split",
                "They guarantee the model won't overfit",
            ],
            "answer": 1,
            "explanation": "RFM is a classic feature-engineering pattern in retail analytics: three numbers "
                            "that capture most of what matters about a customer's purchase behavior.",
        },
        {
            "question": "What does a right-skewed monetary-value histogram suggest for later modeling?",
            "options": [
                "Nothing — histograms don't matter",
                "A few high-spend customers dominate the scale; consider scaling or log-transforming",
                "The data must be duplicated",
                "Clustering will be impossible",
            ],
            "answer": 1,
            "explanation": "Skewed monetary distributions are common in retail; standardizing or "
                            "log-transforming keeps distance-based models (KMeans, IsolationForest) from "
                            "being dominated by a few whales.",
        },
    ],
    "data_preparation": [
        {
            "question": "Why is invoice-level total quantity, not raw line items, used as an anomaly-detection feature?",
            "options": [
                "Raw line items can't be loaded into pandas",
                "It aggregates behavior to the unit (invoice) that actually represents one transaction",
                "It removes the need for feature scaling",
                "It guarantees zero false positives",
            ],
            "answer": 1,
            "explanation": "Anomaly detection needs to compare like units — one invoice's overall shape "
                            "against another's, not individual product rows.",
        },
        {
            "question": "What is a preprocessing leakage risk this project explicitly avoids?",
            "options": [
                "Using pandas instead of NumPy",
                "Deriving a churn-risk label from recency, then also feeding recency into the model as a feature",
                "Splitting data into train and test at all",
                "Using StandardScaler",
            ],
            "answer": 1,
            "explanation": "If the label is a direct function of a feature, the model 'cheats' by "
                            "re-deriving the label instead of learning real behavioral signal — recency is "
                            "excluded from the churn model's feature set for exactly this reason.",
        },
    ],
    "clustering": [
        {
            "question": "Why try several values of k and compare silhouette scores instead of picking k arbitrarily?",
            "options": [
                "Silhouette score is required by scikit-learn's API",
                "It gives a data-driven, comparable measure of how well-separated clusters are at each k",
                "More clusters is always better",
                "It removes the need to scale features",
            ],
            "answer": 1,
            "explanation": "Silhouette score balances intra-cluster cohesion against inter-cluster "
                            "separation, giving an objective way to compare candidate values of k.",
        },
        {
            "question": "Why standardize RFM features before running KMeans?",
            "options": [
                "KMeans requires categorical input",
                "KMeans uses Euclidean distance, so features on larger raw scales (e.g. monetary) would dominate",
                "Standardization guarantees a unique global optimum",
                "It's not necessary for KMeans",
            ],
            "answer": 1,
            "explanation": "Without scaling, a feature measured in dollars would swamp one measured in "
                            "single-digit counts, distorting distance-based clustering.",
        },
    ],
    "anomaly_detection": [
        {
            "question": "How does Isolation Forest decide a point is anomalous?",
            "options": [
                "It measures distance to the nearest centroid",
                "Anomalies are isolated by fewer random feature splits than normal points, on average",
                "It fits a regression line and looks at residuals",
                "It counts missing values per row",
            ],
            "answer": 1,
            "explanation": "Isolation Forest's core idea: outliers are 'few and different', so random "
                            "partitioning trees separate them from the rest of the data in fewer splits.",
        },
        {
            "question": "What does the `contamination` parameter control?",
            "options": [
                "The train/test split ratio",
                "The assumed proportion of anomalies in the data, which sets the decision threshold",
                "The number of trees in the forest",
                "The learning rate",
            ],
            "answer": 1,
            "explanation": "`contamination` is a prior on how much of the data is expected to be "
                            "anomalous; it directly sets where the anomaly-score cutoff falls.",
        },
    ],
    "supervised_learning": [
        {
            "question": "Why report precision, recall, and F1 in addition to accuracy for churn-risk classification?",
            "options": [
                "They are required by RandomForestClassifier",
                "Accuracy alone can look good on imbalanced classes even when the model misses the minority class",
                "They are the same number reported three ways",
                "F1 is only used for regression",
            ],
            "answer": 1,
            "explanation": "If churn-risk is imbalanced, a model predicting the majority class every time "
                            "gets high accuracy but zero recall on the class that actually matters.",
        },
        {
            "question": "What does ROC-AUC measure that a single accuracy score does not?",
            "options": [
                "The model's speed",
                "The model's ability to rank positives above negatives across all decision thresholds",
                "The number of features used",
                "Memory usage during training",
            ],
            "answer": 1,
            "explanation": "ROC-AUC is threshold-independent — it evaluates ranking quality, which is why "
                            "it's a common companion metric to accuracy for classification.",
        },
    ],
    "association_rules": [
        {
            "question": "In association rule mining, what does 'lift' greater than 1 mean for a rule A -> B?",
            "options": [
                "A and B are never bought together",
                "B is bought more often when A is in the basket than by chance alone",
                "A causes B",
                "The rule has 100% confidence",
            ],
            "answer": 1,
            "explanation": "Lift compares observed co-occurrence to the co-occurrence expected under "
                            "independence; lift > 1 means a positive association, not causation.",
        },
        {
            "question": "Why filter frequent itemsets by a minimum support threshold before generating rules?",
            "options": [
                "To guarantee causal relationships",
                "To discard itemsets too rare to be statistically or commercially meaningful",
                "Support has no effect on Apriori's output",
                "To increase the number of rules artificially",
            ],
            "answer": 1,
            "explanation": "Minimum support prunes the search space to itemsets that actually occur often "
                            "enough to matter and keeps Apriori computationally tractable.",
        },
    ],
    "lsh": [
        {
            "question": "What makes a similarity search 'sub-linear' via LSH?",
            "options": [
                "It always returns the exact nearest neighbor",
                "Hashing groups similar items into the same buckets so a query only compares against a small candidate set, not the whole corpus",
                "It skips computing similarity entirely",
                "It requires no indexing step",
            ],
            "answer": 1,
            "explanation": "LSH trades a small amount of accuracy (recall) for speed: bucketing prunes "
                            "the search space so query time scales with candidate-set size, not corpus size.",
        },
        {
            "question": "What is the main trade-off of approximate nearest-neighbor search versus brute force?",
            "options": [
                "There is no trade-off — LSH is always exact and faster",
                "Speed for recall — LSH can miss some true nearest neighbors in exchange for sub-linear query time",
                "Memory for accuracy — LSH uses more memory to be more accurate",
                "LSH only works on text data",
            ],
            "answer": 1,
            "explanation": "This project's own benchmark reports both the speedup and the recall@k drop, "
                            "which is exactly this trade-off measured honestly.",
        },
    ],
    "evaluation_deployment": [
        {
            "question": "What is the purpose of the CRISP-DM Evaluation phase, distinct from Modeling?",
            "options": [
                "Retraining the model with more data",
                "Judging whether the modeling results actually meet the business objectives set at the start",
                "Writing the final report only",
                "Choosing which programming language to use",
            ],
            "answer": 1,
            "explanation": "Evaluation loops back to Business Understanding: technically good metrics are "
                            "worthless if they don't satisfy the original success criteria.",
        },
        {
            "question": "In a compact local project like this one, what does 'Deployment' honestly mean?",
            "options": [
                "A production Kubernetes cluster with autoscaling",
                "A working interactive interface (this Streamlit app) that lets a stakeholder use the results",
                "Nothing — deployment is always out of scope for a student project",
                "A mobile app release",
            ],
            "answer": 1,
            "explanation": "Deployment scales to the project's context — here, a runnable, honest dashboard "
                            "counts as deployment; nothing about that claims production infrastructure.",
        },
    ],
}
