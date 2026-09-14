"""AutoResearch: steepest-ascent hill climbing over AutoGluon's stacking
architecture, i.e. how many bagged folds each base model sees and how many
stacked layers sit on top of them.

Literature this design leans on
--------------------------------
- Wolpert, D. (1992). "Stacked Generalization." Neural Networks 5(2).
  The original stacking idea: train a layer of base learners, then train
  another learner on their out-of-fold predictions.
- Breiman, L. (1996). "Bagging Predictors." Machine Learning 24(2).
  Bagging as variance reduction -- the ``num_bag_folds`` knob.
- Caruana, R., et al. (2004). "Ensemble Selection from Libraries of Models."
  ICML. AutoGluon's ``WeightedEnsemble`` at each stack level is a greedy
  forward selection over the layer's models, in this lineage.
- Erickson, N., et al. (2020). "AutoGluon-Tabular: Robust and Accurate AutoML
  for Structured Data." arXiv:2003.06505. Combines the above into repeated
  k-fold bagging plus multi-layer stacking as the core AutoGluon recipe this
  project illustrates. Reports that quality keeps improving through several
  stack layers before flattening out -- which is exactly the curve the hill
  climb below is tracing out for each task, on data of our own choosing.

Search space
------------
``num_bag_folds``    in {0, 2, 3, 5, 8}   (0 = no bagging, single hold-in fit)
``num_stack_levels`` in {0, 1, 2}         (only meaningful when folds > 0)

Each candidate is scored by AutoGluon's own out-of-fold validation score
(``score_val`` of the run's best model) on the training split only -- the
held-out test split is never touched here. A move is accepted only if it
improves on the current best; the climb stops at a local optimum or when the
evaluation budget runs out.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

FOLD_OPTIONS = [0, 2, 3, 5, 8]
STACK_OPTIONS = [0, 1, 2]

LITERATURE = [
    {"citation": "Wolpert (1992), Stacked Generalization", "informs": "stacking a learner on out-of-fold predictions"},
    {"citation": "Breiman (1996), Bagging Predictors", "informs": "num_bag_folds as a variance-reduction knob"},
    {"citation": "Caruana et al. (2004), Ensemble Selection from Libraries of Models", "informs": "the greedy WeightedEnsemble at each stack level"},
    {"citation": "Erickson et al. (2020), AutoGluon-Tabular, arXiv:2003.06505", "informs": "combining bagging + multi-layer stacking as one recipe"},
]


@dataclass(frozen=True)
class Candidate:
    num_bag_folds: int
    num_stack_levels: int

    @property
    def label(self) -> str:
        return f"folds={self.num_bag_folds}, stack_levels={self.num_stack_levels}"


def neighbors(c: Candidate) -> list[Candidate]:
    out = []
    fi = FOLD_OPTIONS.index(c.num_bag_folds)
    for di in (-1, 1):
        ni = fi + di
        if 0 <= ni < len(FOLD_OPTIONS):
            out.append(Candidate(FOLD_OPTIONS[ni], c.num_stack_levels if FOLD_OPTIONS[ni] > 0 else 0))
    if c.num_bag_folds > 0:
        si = STACK_OPTIONS.index(c.num_stack_levels)
        for di in (-1, 1):
            ni = si + di
            if 0 <= ni < len(STACK_OPTIONS):
                out.append(Candidate(c.num_bag_folds, STACK_OPTIONS[ni]))
    # de-duplicate while preserving order
    seen, uniq = set(), []
    for n in out:
        if (n.num_bag_folds, n.num_stack_levels) not in seen:
            seen.add((n.num_bag_folds, n.num_stack_levels))
            uniq.append(n)
    return uniq


def hill_climb(
    evaluate: Callable[[Candidate], float],
    start: Candidate,
    max_evals: int = 7,
) -> tuple[Candidate, float, list[dict]]:
    """Steepest-ascent hill climbing. Returns (best_candidate, best_score, trace)."""
    trace: list[dict] = []
    visited: dict[tuple[int, int], float] = {}

    def score_of(cand: Candidate) -> float:
        key = (cand.num_bag_folds, cand.num_stack_levels)
        if key in visited:
            return visited[key]
        t0 = time.time()
        val = evaluate(cand)
        visited[key] = val
        trace.append({
            "step": len(trace),
            "num_bag_folds": cand.num_bag_folds,
            "num_stack_levels": cand.num_stack_levels,
            "score_val": val,
            "seconds": round(time.time() - t0, 2),
        })
        return val

    current = start
    current_score = score_of(current)
    evals_used = 1

    improved = True
    while improved and evals_used < max_evals:
        improved = False
        candidates = neighbors(current)
        best_next, best_next_score = current, current_score
        for cand in candidates:
            if evals_used >= max_evals:
                break
            s = score_of(cand)
            evals_used += 1
            if s > best_next_score:
                best_next, best_next_score = cand, s
        if best_next_score > current_score:
            current, current_score = best_next, best_next_score
            improved = True

    return current, current_score, trace


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
