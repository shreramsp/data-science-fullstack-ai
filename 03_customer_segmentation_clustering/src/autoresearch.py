"""AutoResearch: multi-restart steepest-ascent hill climbing over clustering pipelines.

This module implements an "autoresearch" loop that hill-climbs while keeping
dashboard details grounded in the research literature. Every quantity it
module optimises is a published internal validation index, and every design
choice is attributed in ``LITERATURE`` (rendered in the dashboard). No result or
citation here is invented -- the citations are the original sources of the
indices and algorithms, and all reported numbers come from the run itself.

Search space (one candidate = one full preprocessing + clustering pipeline):

    feature_set   rfm | rfm_extended
    log_transform False | True
    scaler        standard | robust | minmax
    algorithm     kmeans | agglomerative | gmm
    k             4 .. 8

Objective: a bounded composite of three internal indices (see ``composite``),
subject to an actionability constraint (see ``MIN_CLUSTER_SHARE``).
"""
from __future__ import annotations

import itertools
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from features import FEATURE_SETS, matrix

SEED = 20260913

# --- actionability constraint -------------------------------------------------
# Two requirements are declared BEFORE the search runs, so that neither is a
# post-hoc justification of a result we happened to like:
#
#  1. k in [4, 8]. Internal indices reward few, well-separated blobs, and they
#     score k=2 and k=3 ABOVE anything in this band -- see the
#     "unconstrained_reference" block train.py writes into metrics.json. A
#     two-way split of the customer base is a valid geometric optimum and not a
#     segmentation; k=3 scores well too, but the RFM quintile tiers (Hughes 1994)
#     already rank customers good/average/poor for free, so a clustering has to
#     resolve more than that to earn its place. The upper bound is operational:
#     campaign tracks a marketing team can actually staff.
#  2. No segment smaller than 3% of the base -- below that a segment is an
#     outlier pocket, not an audience.
#
# Infeasible candidates keep a graded penalty rather than -inf so hill climbing
# retains a gradient back into the feasible region.
MIN_K = 4
MAX_K = 8
MIN_CLUSTER_SHARE = 0.03
INFEASIBLE_PENALTY = 1.0

SEARCH_SPACE: dict[str, list] = {
    "feature_set": list(FEATURE_SETS),
    "log_transform": [False, True],
    "scaler": ["standard", "robust", "minmax"],
    "algorithm": ["kmeans", "agglomerative", "gmm"],
    "k": list(range(MIN_K, MAX_K + 1)),
}

# Composite weights. Silhouette carries the most weight because it is defined
# per-observation and is the index the segmentation literature reports most
# often; the other two act as tie-breakers on compactness/separation.
WEIGHTS = {"silhouette": 0.50, "davies_bouldin": 0.25, "calinski_harabasz": 0.25}
CH_SQUASH = 500.0  # constant in CH/(CH+c); monotone, keeps the term in [0,1)

LITERATURE: list[dict[str, str]] = [
    {"element": "Silhouette coefficient",
     "role": "Primary objective term; per-point cohesion vs. separation in [-1, 1].",
     "source": "Rousseeuw, P.J. (1987). Silhouettes: a graphical aid to the interpretation "
               "and validation of cluster analysis. J. Comput. Appl. Math. 20, 53-65."},
    {"element": "Calinski-Harabasz index",
     "role": "Objective term; between/within dispersion ratio, higher is better.",
     "source": "Calinski, T. & Harabasz, J. (1974). A dendrite method for cluster analysis. "
               "Communications in Statistics 3(1), 1-27."},
    {"element": "Davies-Bouldin index",
     "role": "Objective term; mean worst-case cluster similarity, lower is better.",
     "source": "Davies, D.L. & Bouldin, D.W. (1979). A cluster separation measure. "
               "IEEE Trans. Pattern Anal. Mach. Intell. PAMI-1(2), 224-227."},
    {"element": "RFM feature construction",
     "role": "Recency / Frequency / Monetary customer representation.",
     "source": "Hughes, A.M. (1994). Strategic Database Marketing. Probus Publishing."},
    {"element": "k-means++ seeding",
     "role": "Initialisation used by the kmeans candidate (scikit-learn default).",
     "source": "Arthur, D. & Vassilvitskii, S. (2007). k-means++: the advantages of careful "
               "seeding. SODA '07, 1027-1035."},
    {"element": "Ward linkage",
     "role": "Linkage rule for the agglomerative candidate.",
     "source": "Ward, J.H. (1963). Hierarchical grouping to optimize an objective function. "
               "J. Am. Stat. Assoc. 58(301), 236-244."},
    {"element": "Gaussian mixture / EM",
     "role": "Soft-assignment candidate in the search space.",
     "source": "Dempster, A.P., Laird, N.M. & Rubin, D.B. (1977). Maximum likelihood from "
               "incomplete data via the EM algorithm. JRSS-B 39(1), 1-38."},
    {"element": "Steepest-ascent hill climbing with random restarts",
     "role": "The AutoResearch search strategy itself.",
     "source": "Russell, S. & Norvig, P. Artificial Intelligence: A Modern Approach, "
               "ch. 4 'Beyond Classical Search' (local search)."},
    {"element": "Elbow / within-cluster sum of squares curve",
     "role": "Diagnostic shown alongside the search, not used as the objective.",
     "source": "Thorndike, R.L. (1953). Who belongs in the family? Psychometrika 18(4), 267-276."},
]


@dataclass(frozen=True)
class Config:
    feature_set: str
    log_transform: bool
    scaler: str
    algorithm: str
    k: int

    def key(self) -> tuple:
        return (self.feature_set, self.log_transform, self.scaler, self.algorithm, self.k)


@dataclass
class Result:
    """One evaluated pipeline.

    ``composite`` is the raw literature-index blend; ``objective`` is what hill
    climbing actually maximises (composite minus the feasibility penalty). They
    are equal for every feasible candidate.
    """
    config: Config
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    composite: float
    objective: float
    inertia: float | None
    min_cluster_share: float = 0.0
    feasible: bool = False
    cluster_sizes: list[int] = field(default_factory=list)
    error: str | None = None


def make_scaler(name: str):
    return {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}[name]()


def make_estimator(name: str, k: int, seed: int = SEED):
    if name == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=seed)
    if name == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward")
    if name == "gmm":
        return GaussianMixture(n_components=k, covariance_type="full",
                               n_init=3, random_state=seed)
    raise ValueError(f"unknown algorithm {name!r}")


def composite(sil: float, db: float, ch: float) -> float:
    """Bounded, monotone blend of the three indices -> roughly [0, 1]."""
    return (WEIGHTS["silhouette"] * (sil + 1.0) / 2.0
            + WEIGHTS["davies_bouldin"] * (1.0 / (1.0 + db))
            + WEIGHTS["calinski_harabasz"] * (ch / (ch + CH_SQUASH)))


def penalise(comp: float, min_share: float, n_clusters: int) -> tuple[float, bool]:
    """Apply the actionability constraint, returning (objective, feasible)."""
    deficit = max(0.0, MIN_CLUSTER_SHARE - min_share) + max(0, MIN_K - n_clusters) * 0.05
    if deficit <= 0.0:
        return comp, True
    # every infeasible candidate scores below every feasible one, but the
    # penalty shrinks as the candidate approaches feasibility -> usable gradient
    return comp - INFEASIBLE_PENALTY - deficit, False


def evaluate(cfg: Config, feat, seed: int = SEED) -> Result:
    """Fit one candidate pipeline and score it with the three indices."""
    X = matrix(feat, cfg.feature_set, cfg.log_transform)
    Xs = make_scaler(cfg.scaler).fit_transform(X)
    est = make_estimator(cfg.algorithm, cfg.k, seed)
    try:
        labels = est.fit_predict(Xs)
    except Exception as exc:                                   # degenerate candidate
        return Result(cfg, -1.0, np.inf, 0.0, 0.0, -np.inf, None, error=str(exc))

    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) < 2:
        return Result(cfg, -1.0, np.inf, 0.0, 0.0, -np.inf, None, 0.0, False,
                      counts.tolist(), error="collapsed to a single cluster")

    sil = float(silhouette_score(Xs, labels))
    db = float(davies_bouldin_score(Xs, labels))
    ch = float(calinski_harabasz_score(Xs, labels))
    comp = composite(sil, db, ch)
    min_share = float(counts.min() / counts.sum())
    obj, feasible = penalise(comp, min_share, len(uniq))
    inertia = float(getattr(est, "inertia_", np.nan))
    return Result(cfg, sil, db, ch, comp, obj,
                  None if np.isnan(inertia) else inertia, min_share, feasible,
                  counts.tolist(),
                  None if feasible else f"infeasible: smallest cluster {min_share:.1%}")


def neighbours(cfg: Config) -> Iterable[Config]:
    """All configs one coordinate away (k moves by +-1 only)."""
    d = asdict(cfg)
    for dim, values in SEARCH_SPACE.items():
        if dim == "k":
            for step in (-1, 1):
                nk = cfg.k + step
                if nk in values:
                    yield Config(**{**d, "k": nk})
        else:
            for v in values:
                if v != d[dim]:
                    yield Config(**{**d, dim: v})


def space_size() -> int:
    return int(np.prod([len(v) for v in SEARCH_SPACE.values()]))


def hill_climb(feat, restarts: int = 4, max_steps: int = 25,
               seed: int = SEED, verbose: bool = True) -> tuple[Result, list[dict]]:
    """Multi-restart steepest-ascent hill climbing.

    Each restart starts from a random config, repeatedly moves to the best
    strictly-improving neighbour, and stops at a local optimum. Every fit is
    cached so the reported evaluation count is the true model-fitting cost.
    """
    rng = random.Random(seed)
    cache: dict[tuple, Result] = {}
    history: list[dict] = []
    best: Result | None = None
    t0 = time.perf_counter()

    def score(cfg: Config) -> Result:
        if cfg.key() not in cache:
            cache[cfg.key()] = evaluate(cfg, feat, seed)
        return cache[cfg.key()]

    for restart in range(restarts):
        if restart == 0:
            # a deterministic, literature-standard starting point
            current = Config("rfm", True, "standard", "kmeans", 4)
        else:
            current = Config(**{d: rng.choice(v) for d, v in SEARCH_SPACE.items()})
        cur = score(current)

        for step in range(max_steps):
            cand = [score(n) for n in neighbours(current)]
            winner = max(cand, key=lambda r: r.objective)
            improved = winner.objective > cur.objective + 1e-9
            history.append(dict(
                restart=restart, step=step, n_evaluated=len(cache),
                moved=bool(improved),
                **{f"cur_{k}": v for k, v in asdict(current).items()},
                cur_objective=cur.objective, cur_composite=cur.composite,
                cur_silhouette=cur.silhouette,
                cur_davies_bouldin=cur.davies_bouldin,
                cur_calinski_harabasz=cur.calinski_harabasz,
                cur_feasible=cur.feasible,
                best_objective=max(cur.objective, best.objective if best else -np.inf),
            ))
            if not improved:
                break
            current, cur = winner.config, winner

        if best is None or cur.objective > best.objective:
            best = cur
        if verbose:
            print(f"  restart {restart}: local optimum {cur.config.key()} "
                  f"objective={cur.objective:.4f} silhouette={cur.silhouette:.4f} "
                  f"{'feasible' if cur.feasible else 'INFEASIBLE'}")

    assert best is not None
    for row in history:
        row["elapsed_s"] = None
    history[-1]["elapsed_s"] = round(time.perf_counter() - t0, 2)
    history_full = [dict(r, **{}) for r in history]
    return best, history_full


def exhaustive(feat, seed: int = SEED) -> list[Result]:
    """Full grid -- used once, offline, to report how close hill climbing got."""
    combos = itertools.product(*SEARCH_SPACE.values())
    return [evaluate(Config(*c), feat, seed) for c in combos]
