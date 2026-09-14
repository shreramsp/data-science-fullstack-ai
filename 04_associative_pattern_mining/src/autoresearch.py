"""Autoresearch: hill-climbing search over association-mining hyperparameters.

This module applies hill climbing to automate parameter research. Concretely:
choosing ``min_support`` / ``min_confidence`` / ``min_lift`` / ``max_len`` /
algorithm by hand is the part of market-basket analysis that is usually done
by trial and error, so this module automates it as a discrete local search.

Algorithm — steepest-ascent hill climbing with random restarts:

1. Start from a configuration (the first restart uses a fixed sensible seed
   config; later restarts start from random grid points).
2. Enumerate the neighbourhood: every configuration reachable by moving
   exactly one hyperparameter one step along its grid axis.
3. Evaluate each neighbour with ``mining.objective`` (mean lift + basket
   coverage − a penalty for unusable rule-set sizes).
4. Move to the best neighbour if it strictly improves; otherwise this restart
   has hit a local optimum and stops.
5. Repeat from a new random start; keep the best configuration seen overall.

Random restarts are the standard remedy for hill climbing's local-optimum
problem (Russell & Norvig, *AIAI*, ch. 4). Every configuration evaluated is
cached and logged, so the dashboard can show the whole search trace rather
than just the winner.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .mining import (
    MiningConfig,
    clean_raw,
    evaluate,
    load_raw,
    mine_rules,
    objective,
    to_basket_matrix,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"

# Discrete search grid. Hill climbing moves one step along one axis at a time.
GRID: dict[str, list] = {
    "algorithm": ["apriori", "fpgrowth"],
    "min_support": [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
    "min_confidence": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    "min_lift": [1.0, 1.1, 1.25, 1.5, 2.0, 3.0],
    "max_len": [2, 3, 4],
}

SEED_CONFIG = MiningConfig(
    algorithm="fpgrowth", min_support=0.02, min_confidence=0.30, min_lift=1.1, max_len=3
)


def neighbours(config: MiningConfig) -> list[MiningConfig]:
    """Every config one grid step away along exactly one axis."""
    out: list[MiningConfig] = []
    for axis, values in GRID.items():
        current = getattr(config, axis)
        idx = values.index(current)
        for step in (-1, 1):
            j = idx + step
            if 0 <= j < len(values):
                out.append(replace(config, **{axis: values[j]}))
    return out


def random_config(rng: random.Random) -> MiningConfig:
    return MiningConfig(**{axis: rng.choice(values) for axis, values in GRID.items()})


def hill_climb(
    matrix: pd.DataFrame,
    restarts: int = 3,
    max_steps: int = 12,
    seed: int = 7,
) -> tuple[MiningConfig, dict, pd.DataFrame]:
    """Run the search. Returns (best_config, best_metrics, history)."""
    rng = random.Random(seed)
    cache: dict[MiningConfig, dict] = {}
    history: list[dict] = []

    def score_of(config: MiningConfig, restart: int, step: int, kind: str) -> dict:
        if config not in cache:
            itemsets, rules = mine_rules(matrix, config)
            metrics = evaluate(matrix, itemsets, rules)
            metrics["score"] = objective(metrics)
            cache[config] = metrics
            evaluated = True
        else:
            metrics = cache[config]
            evaluated = False
        history.append(
            {
                "restart": restart,
                "step": step,
                "kind": kind,
                "evaluated": evaluated,
                **config.as_dict(),
                **metrics,
            }
        )
        return metrics

    best_config: MiningConfig | None = None
    best_metrics: dict = {"score": -1.0}

    for restart in range(restarts):
        current = SEED_CONFIG if restart == 0 else random_config(rng)
        current_metrics = score_of(current, restart, 0, "start")

        for step in range(1, max_steps + 1):
            candidates = [
                (score_of(n, restart, step, "neighbour"), n) for n in neighbours(current)
            ]
            best_neighbour_metrics, best_neighbour = max(
                candidates, key=lambda pair: pair[0]["score"]
            )
            if best_neighbour_metrics["score"] <= current_metrics["score"]:
                break  # local optimum for this restart
            current, current_metrics = best_neighbour, best_neighbour_metrics
            history.append(
                {
                    "restart": restart,
                    "step": step,
                    "kind": "accepted",
                    "evaluated": False,
                    **current.as_dict(),
                    **current_metrics,
                }
            )

        if current_metrics["score"] > best_metrics["score"]:
            best_config, best_metrics = current, current_metrics

    assert best_config is not None
    history_df = pd.DataFrame(history)
    # Running best, for the search-trajectory chart.
    history_df["best_so_far"] = history_df["score"].cummax()
    history_df["trial"] = range(1, len(history_df) + 1)
    return best_config, best_metrics, history_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    clean, _ = clean_raw(load_raw())
    matrix = to_basket_matrix(clean)

    best_config, best_metrics, history = hill_climb(
        matrix, restarts=args.restarts, max_steps=args.max_steps, seed=args.seed
    )

    ARTIFACTS.mkdir(exist_ok=True)
    history.to_csv(ARTIFACTS / "autoresearch_history.csv", index=False)
    (ARTIFACTS / "autoresearch_best.json").write_text(
        json.dumps({"config": best_config.as_dict(), "metrics": best_metrics}, indent=2)
    )

    n_evaluated = int(history["evaluated"].sum())
    print(f"Configurations evaluated: {n_evaluated} (from {len(history)} logged trials)")
    print(f"Best config : {best_config.as_dict()}")
    print(f"Best metrics: { {k: round(v, 4) if isinstance(v, float) else v for k, v in best_metrics.items()} }")
    print(f"Wrote {ARTIFACTS/'autoresearch_history.csv'} and {ARTIFACTS/'autoresearch_best.json'}")


if __name__ == "__main__":
    main()
