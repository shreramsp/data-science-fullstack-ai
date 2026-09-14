"""The detector zoo: five classic unsupervised anomaly-detection methods.

Every detector is wrapped in one interface -- fit on unlabelled training data,
then emit a score where **higher means more anomalous**. That sign convention
is the only thing the rest of the platform needs to know about them, which is
what lets the autoresearch loop treat the method itself as a search dimension.

Methods and their sources (all real, all classic):

* ``iforest``  -- Isolation Forest. Liu, Ting & Zhou, "Isolation Forest",
  ICDM 2008. Isolates points by random axis-aligned splits; anomalies need
  fewer splits, so short average path length = anomalous.
* ``lof``      -- Local Outlier Factor. Breunig, Kriegel, Ng & Sander, SIGMOD
  2000. Compares a point's local density to its neighbours'; catches
  anomalies that are only anomalous relative to their own neighbourhood.
* ``ocsvm``    -- One-Class SVM. Scholkopf et al., "Estimating the Support of
  a High-Dimensional Distribution", Neural Computation 2001. Fits a kernel
  boundary around the bulk of the data. O(n^2)-ish, so it is fitted on a
  bounded subsample (``OCSVM_FIT_CAP``) and that cap is reported, not hidden.
* ``elliptic`` -- Elliptic Envelope / Minimum Covariance Determinant.
  Rousseeuw & Van Driessen, Technometrics 1999. Robust Gaussian fit;
  Mahalanobis distance to the robust centre is the score. Strong when the
  inliers really are one elliptical cloud, weak otherwise.
* ``pca_recon``-- PCA reconstruction error. Shyu et al., "A Novel Anomaly
  Detection Scheme Based on Principal Component Classifier", 2003. Projects
  onto the top components and scores the residual left behind.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

SEED = 20260913
OCSVM_FIT_CAP = 6_000  # rows used to fit the kernel; scoring still covers all rows

DETECTORS = ("iforest", "lof", "ocsvm", "elliptic", "pca_recon")

# Per-detector tunables the autoresearch loop is allowed to move.
DETECTOR_PARAMS: dict[str, dict[str, list]] = {
    "iforest": {"n_estimators": [100, 300, 600], "max_samples": [256, 1024, 4096],
                "max_features": [0.5, 0.8, 1.0]},
    "lof": {"n_neighbors": [10, 20, 35, 60], "metric": ["euclidean", "manhattan"]},
    "ocsvm": {"nu": [0.01, 0.05, 0.1], "gamma": ["scale", 0.01, 0.05]},
    "elliptic": {"support_fraction": [0.7, 0.85, 1.0]},
    "pca_recon": {"n_components": [3, 6, 10, 15]},
}


class Detector:
    """Uniform fit/score wrapper. ``score(X)``: higher = more anomalous."""

    def __init__(self, kind: str, params: dict):
        self.kind = kind
        self.params = dict(params)
        self.model = None
        self._pca_mean = None

    def fit(self, X: np.ndarray) -> "Detector":
        p = self.params
        if self.kind == "iforest":
            self.model = IsolationForest(
                n_estimators=p.get("n_estimators", 300),
                max_samples=min(p.get("max_samples", 1024), len(X)),
                max_features=p.get("max_features", 1.0),
                random_state=SEED, n_jobs=-1,
            ).fit(X)
        elif self.kind == "lof":
            self.model = LocalOutlierFactor(
                n_neighbors=p.get("n_neighbors", 20),
                metric=p.get("metric", "euclidean"),
                novelty=True, n_jobs=-1,
            ).fit(X)
        elif self.kind == "ocsvm":
            rng = np.random.default_rng(SEED)
            sample = X if len(X) <= OCSVM_FIT_CAP else X[rng.choice(len(X), OCSVM_FIT_CAP, replace=False)]
            self.model = OneClassSVM(
                kernel="rbf", nu=p.get("nu", 0.05), gamma=p.get("gamma", "scale"),
            ).fit(sample)
        elif self.kind == "elliptic":
            sf = p.get("support_fraction", 0.85)
            self.model = EllipticEnvelope(
                support_fraction=None if sf >= 1.0 else sf,
                contamination=0.01, random_state=SEED,
            ).fit(X)
        elif self.kind == "pca_recon":
            n = min(p.get("n_components", 10), X.shape[1])
            self.model = PCA(n_components=n, random_state=SEED).fit(X)
        else:
            raise ValueError(f"unknown detector: {self.kind}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("detector not fitted")
        if self.kind == "pca_recon":
            recon = self.model.inverse_transform(self.model.transform(X))
            return np.sqrt(((X - recon) ** 2).sum(axis=1))
        # sklearn's convention: score_samples is higher for inliers.
        return -self.model.score_samples(X)


def default_params(kind: str) -> dict:
    """Mid-of-the-road defaults -- the baseline every method is measured from."""
    return {k: v[len(v) // 2] for k, v in DETECTOR_PARAMS[kind].items()}


def build(kind: str, params: dict | None = None) -> Detector:
    merged = default_params(kind)
    merged.update(params or {})
    return Detector(kind, merged)
