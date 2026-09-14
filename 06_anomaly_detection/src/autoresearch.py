"""AutoResearch: multi-restart steepest-ascent hill climbing over detector pipelines.

This module implements a hill-climbing autoresearch loop and supplies a
dashboard whose details are grounded in the research literature. It
supplies both halves:

* ``hill_climb`` searches the joint space of *feature set x scaler x detection
  method x that method's hyperparameters* -- the method itself is a search
  dimension, not a fixed choice.
* ``LITERATURE`` records the published source behind every design decision the
  dashboard displays. The citations are the genuine original papers; every
  number shown anywhere in this project comes from the run itself, never from
  a paper's reported results.

Objective: **average precision** (area under the precision-recall curve) on the
validation split. Under ~0.5% prevalence, ROC-AUC is dominated by the huge
negative class and stays flatteringly high for weak detectors -- Davis &
Goadrich (ICML 2006) and Saito & Rehmsmeier (PLOS ONE 2015) are the standard
references for preferring PR over ROC in exactly this regime. ROC-AUC is still
recorded for every candidate, it just does not steer the search.

Labels are used only to *score* a candidate, never to fit one. Doing so means
the reported validation numbers are optimistically biased by the search itself,
which is precisely why the untouched test split exists.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from detectors import DETECTOR_PARAMS, DETECTORS, build, default_params
from features import FEATURE_SETS, SCALERS, make_scaler, matrix

SEED = 20260913
ALERT_BUDGET = 0.005  # analysts review the top 0.5% of transactions per day

LITERATURE: list[dict[str, str]] = [
    {"topic": "Isolation Forest",
     "source": "Liu, Ting & Zhou (ICDM 2008)",
     "used_for": "Random-split isolation depth as an anomaly score; one of the five candidate detectors."},
    {"topic": "Local Outlier Factor",
     "source": "Breunig, Kriegel, Ng & Sander (SIGMOD 2000)",
     "used_for": "Local-density-ratio scoring, for anomalies that are only anomalous relative to their neighbourhood."},
    {"topic": "One-Class SVM",
     "source": "Scholkopf, Platt, Shawe-Taylor, Smola & Williamson (Neural Computation 2001)",
     "used_for": "Kernel support estimation of the inlier region; fitted on a bounded subsample for tractability."},
    {"topic": "Minimum Covariance Determinant",
     "source": "Rousseeuw & Van Driessen (Technometrics 1999)",
     "used_for": "Robust Mahalanobis distance (Elliptic Envelope) that is not dragged by the contamination itself."},
    {"topic": "PCA reconstruction error",
     "source": "Shyu, Chen, Sarinnapakorn & Chang (2003)",
     "used_for": "Residual off the principal subspace as a score; the cheapest detector in the pool."},
    {"topic": "PR over ROC under class imbalance",
     "source": "Davis & Goadrich (ICML 2006); Saito & Rehmsmeier (PLOS ONE 2015)",
     "used_for": "Average precision is the hill-climbing objective; ROC-AUC is reported but not optimised."},
    {"topic": "Credit-card fraud benchmark",
     "source": "Dal Pozzolo, Caelen, Johnson & Bontempi (IEEE CIDM 2015)",
     "used_for": "Origin of the ULB dataset this platform targets, and of the practice of evaluating at a fixed alert budget."},
    {"topic": "Hill climbing with random restarts",
     "source": "Russell & Norvig, AIMA, ch. 4",
     "used_for": "The search strategy: steepest-ascent moves, restarted to escape local optima."},
]

SHARED_SPACE = {
    "feature_set": list(FEATURE_SETS),
    "scaler": list(SCALERS),
    "detector": list(DETECTORS),
}


@dataclass
class Candidate:
    feature_set: str
    scaler: str
    detector: str
    params: dict = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.feature_set, self.scaler, self.detector,
                tuple(sorted((k, str(v)) for k, v in self.params.items())))

    def label(self) -> str:
        ps = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.detector}[{ps}] | {self.feature_set} | {self.scaler}"

    def as_dict(self) -> dict:
        return {"feature_set": self.feature_set, "scaler": self.scaler,
                "detector": self.detector, **{f"param_{k}": v for k, v in self.params.items()}}


def alert_metrics(y_true: np.ndarray, scores: np.ndarray, budget: float = ALERT_BUDGET) -> dict:
    """Operating-point metrics at a fixed analyst alert budget.

    A fraud team can only work so many alerts a day, so "precision@budget" is
    the number an operations owner actually signs off on.
    """
    k = max(1, int(round(len(scores) * budget)))
    top = np.argsort(scores)[::-1][:k]
    hits = int(y_true[top].sum())
    total = int(y_true.sum())
    return {
        "alerts": k,
        "precision_at_budget": hits / k,
        "recall_at_budget": hits / total if total else 0.0,
        "caught": hits,
        "missed": total - hits,
    }


class Evaluator:
    """Fits and scores one candidate; caches by candidate key."""

    def __init__(self, train, valid):
        self.train, self.valid = train, valid
        self.y_valid = valid["Class"].to_numpy()
        self.cache: dict[tuple, dict] = {}
        self.n_fits = 0

    def __call__(self, cand: Candidate) -> dict:
        key = cand.key()
        if key in self.cache:
            return self.cache[key]
        t0 = time.perf_counter()
        scores = self.score_valid(cand)
        result = {
            "average_precision": float(average_precision_score(self.y_valid, scores)),
            "roc_auc": float(roc_auc_score(self.y_valid, scores)),
            "seconds": time.perf_counter() - t0,
            **alert_metrics(self.y_valid, scores),
        }
        self.cache[key] = result
        self.n_fits += 1
        return result

    def score_valid(self, cand: Candidate) -> np.ndarray:
        Xtr = matrix(self.train, cand.feature_set)
        Xva = matrix(self.valid, cand.feature_set)
        scaler = make_scaler(cand.scaler).fit(Xtr)
        det = build(cand.detector, cand.params).fit(scaler.transform(Xtr))
        return det.score(scaler.transform(Xva))


def neighbours(cand: Candidate) -> list[Candidate]:
    """One-move neighbourhood: change exactly one decision.

    Switching the detector resets its hyperparameters to that method's
    defaults, since the parameter names are not shared across methods.
    """
    out: list[Candidate] = []
    for fs in SHARED_SPACE["feature_set"]:
        if fs != cand.feature_set:
            out.append(Candidate(fs, cand.scaler, cand.detector, dict(cand.params)))
    for sc in SHARED_SPACE["scaler"]:
        if sc != cand.scaler:
            out.append(Candidate(cand.feature_set, sc, cand.detector, dict(cand.params)))
    for det in SHARED_SPACE["detector"]:
        if det != cand.detector:
            out.append(Candidate(cand.feature_set, cand.scaler, det, default_params(det)))
    for name, values in DETECTOR_PARAMS[cand.detector].items():
        for value in values:
            if value != cand.params.get(name):
                params = dict(cand.params)
                params[name] = value
                out.append(Candidate(cand.feature_set, cand.scaler, cand.detector, params))
    return out


def random_candidate(rng: random.Random) -> Candidate:
    det = rng.choice(SHARED_SPACE["detector"])
    params = {k: rng.choice(v) for k, v in DETECTOR_PARAMS[det].items()}
    return Candidate(rng.choice(SHARED_SPACE["feature_set"]),
                     rng.choice(SHARED_SPACE["scaler"]), det, params)


def hill_climb(train, valid, restarts: int = 3, budget: int = 140,
               seed: int = SEED, verbose: bool = True):
    """Steepest-ascent hill climbing with random restarts.

    Returns ``(best_candidate, best_result, history)``. ``history`` is one row
    per *evaluation*, which is what the dashboard plots -- including the
    rejected moves, because a search trajectory that only shows improvements
    tells you nothing about how hard the landscape was.
    """
    rng = random.Random(seed)
    evaluate = Evaluator(train, valid)
    history: list[dict] = []
    best_overall: tuple[Candidate, dict] | None = None

    for restart in range(restarts):
        # Restart 0 starts from a fixed, defensible default so the run is
        # reproducible and comparable to the baseline sweep.
        current = (Candidate("components_amount", "standard", "iforest", default_params("iforest"))
                   if restart == 0 else random_candidate(rng))
        current_res = evaluate(current)
        history.append(_row(len(history), restart, 0, current, current_res, "start", True,
                            best_overall[1]["average_precision"] if best_overall else current_res["average_precision"]))
        if best_overall is None or current_res["average_precision"] > best_overall[1]["average_precision"]:
            best_overall = (current, current_res)

        step = 0
        while evaluate.n_fits < budget:
            step += 1
            candidates = neighbours(current)
            rng.shuffle(candidates)
            best_move: tuple[Candidate, dict] | None = None
            for cand in candidates:
                if evaluate.n_fits >= budget:
                    break
                res = evaluate(cand)
                improved = res["average_precision"] > current_res["average_precision"]
                is_best_move = best_move is None or res["average_precision"] > best_move[1]["average_precision"]
                if improved and is_best_move:
                    best_move = (cand, res)
                history.append(_row(len(history), restart, step, cand, res, "neighbour", False,
                                    best_overall[1]["average_precision"]))
            if best_move is None:
                if verbose:
                    print(f"  restart {restart}: local optimum after {step} steps "
                          f"AP={current_res['average_precision']:.4f} -> {current.label()}")
                break
            current, current_res = best_move
            history.append(_row(len(history), restart, step, current, current_res, "accepted", True,
                                max(best_overall[1]["average_precision"], current_res["average_precision"])))
            if current_res["average_precision"] > best_overall[1]["average_precision"]:
                best_overall = (current, current_res)
            if verbose:
                print(f"  restart {restart} step {step}: AP={current_res['average_precision']:.4f} "
                      f"<- {current.label()}")
        if evaluate.n_fits >= budget:
            if verbose:
                print(f"  evaluation budget ({budget}) exhausted during restart {restart}")
            break

    return best_overall[0], best_overall[1], history


def _row(i, restart, step, cand, res, kind, accepted, running_best) -> dict:
    return {
        "eval": i, "restart": restart, "step": step, "move": kind, "accepted": accepted,
        "label": cand.label(), **cand.as_dict(),
        "average_precision": res["average_precision"], "roc_auc": res["roc_auc"],
        "precision_at_budget": res["precision_at_budget"],
        "recall_at_budget": res["recall_at_budget"], "seconds": res["seconds"],
        "running_best_ap": max(running_best, res["average_precision"]),
    }
