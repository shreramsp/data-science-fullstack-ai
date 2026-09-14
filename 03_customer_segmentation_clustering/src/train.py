"""CRISP-DM pipeline: prepare -> AutoResearch hill climbing -> refit -> evaluate -> persist.

Run:  python src/train.py [--restarts 4] [--skip-exhaustive]
Writes every artifact the dashboard reads into ``artifacts/``.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autoresearch as ar                                        # noqa: E402
from features import DATA_DIR, matrix, prepare                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

# Marketing archetypes in normalised (recency-desirability, frequency, monetary)
# space, used only to give clusters readable names. Recency desirability = 1
# means "bought very recently".
NAME_TEMPLATES: list[tuple[str, tuple[float, float, float], str]] = [
    ("Champions",        (1.0, 1.0, 1.0), "Reward, early access, referral asks."),
    ("Loyal Regulars",   (0.8, 0.7, 0.5), "Subscription / replenishment offers."),
    ("Big-Ticket Buyers",(0.6, 0.2, 0.9), "High-margin bundles, concierge service."),
    ("Promising / New",  (0.9, 0.1, 0.2), "Onboarding series, second-purchase nudge."),
    ("At Risk",          (0.2, 0.7, 0.6), "Win-back discount before churn completes."),
    ("Hibernating",      (0.0, 0.1, 0.1), "Low-cost reactivation; suppress if unresponsive."),
]


def name_segments(profile: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Assign a distinct marketing name to each cluster by optimal template match."""
    def unit_rank(s: pd.Series, invert: bool = False) -> np.ndarray:
        r = s.rank(method="average").to_numpy(dtype=float)
        r = (r - 1) / max(len(r) - 1, 1)
        return 1.0 - r if invert else r

    pts = np.column_stack([
        unit_rank(profile["recency_days"], invert=True),   # recent = high
        unit_rank(profile["frequency"]),
        unit_rank(profile["monetary"]),
    ])
    tmpl = np.array([t[1] for t in NAME_TEMPLATES])
    cost = np.linalg.norm(pts[:, None, :] - tmpl[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)

    names: dict[int, tuple[str, str]] = {}
    for r, c in zip(rows, cols):
        cluster = int(profile.index[r])
        names[cluster] = (NAME_TEMPLATES[c][0], NAME_TEMPLATES[c][2])
    for i, cluster in enumerate(profile.index):            # k > len(templates)
        names.setdefault(int(cluster), (f"Segment {int(cluster)}", "No template match."))
    return names


def fit_labels(cfg: ar.Config, feat: pd.DataFrame, seed: int = ar.SEED) -> np.ndarray:
    X = matrix(feat, cfg.feature_set, cfg.log_transform)
    Xs = ar.make_scaler(cfg.scaler).fit_transform(X)
    return ar.make_estimator(cfg.algorithm, cfg.k, seed).fit_predict(Xs)


def stability(cfg: ar.Config, feat: pd.DataFrame, base_labels: np.ndarray,
              n_trials: int = 10, frac: float = 0.8) -> dict[str, float]:
    """Subsample stability: refit on 80% samples, compare with the full-data
    labels on the shared customers (adjusted Rand index)."""
    rng = np.random.default_rng(ar.SEED)
    base = pd.Series(base_labels, index=feat.index)
    scores = []
    for t in range(n_trials):
        idx = rng.choice(len(feat), size=int(frac * len(feat)), replace=False)
        sub = feat.iloc[np.sort(idx)]
        lab = fit_labels(cfg, sub, seed=ar.SEED + t)
        scores.append(adjusted_rand_score(base.loc[sub.index].to_numpy(), lab))
    return {"ari_mean": float(np.mean(scores)), "ari_std": float(np.std(scores)),
            "ari_min": float(np.min(scores)), "n_trials": n_trials, "subsample_frac": frac}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--skip-exhaustive", action="store_true",
                    help="skip the full-grid reference run (it is only used to "
                         "report the hill-climbing optimality gap)")
    args = ap.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    t_start = time.perf_counter()

    # --- CRISP-DM 2/3: Data understanding & preparation ---------------------
    print("[1/6] preparing data ...")
    clean, feat, report = prepare()
    print(f"      {report.rows_raw:,} raw lines -> {report.rows_final:,} clean lines "
          f"-> {len(feat):,} customers")

    # --- CRISP-DM 4: Modeling via AutoResearch hill climbing ----------------
    print(f"[2/6] hill climbing over {ar.space_size()} candidate pipelines "
          f"({args.restarts} restarts) ...")
    best, history = ar.hill_climb(feat, restarts=args.restarts,
                                  max_steps=args.max_steps)
    hist_df = pd.DataFrame(history)
    n_fits = int(hist_df["n_evaluated"].max())
    print(f"      best {best.config.key()} objective={best.objective:.4f} "
          f"after {n_fits} distinct fits")

    # --- honest optimality gap against the full grid ------------------------
    grid_summary = None
    if not args.skip_exhaustive:
        print(f"[3/6] exhaustive reference run over all {ar.space_size()} configs ...")
        t0 = time.perf_counter()
        all_res = ar.exhaustive(feat)
        ok = [r for r in all_res if r.feasible]
        gbest = max(ok, key=lambda r: r.objective)
        grid_summary = {
            "n_configs": len(all_res),
            "n_feasible": len(ok),
            "n_infeasible": len(all_res) - len(ok),
            "grid_best_objective": gbest.objective,
            "grid_best_config": asdict(gbest.config),
            "hill_climb_found_global_optimum": bool(gbest.config.key() == best.config.key()),
            "optimality_gap": float(gbest.objective - best.objective),
            "fits_saved_vs_grid": len(all_res) - n_fits,
            "grid_seconds": round(time.perf_counter() - t0, 2),
        }
        pd.DataFrame([{**asdict(r.config), "silhouette": r.silhouette,
                       "davies_bouldin": r.davies_bouldin,
                       "calinski_harabasz": r.calinski_harabasz,
                       "composite": r.composite, "objective": r.objective,
                       "min_cluster_share": r.min_cluster_share,
                       "feasible": r.feasible, "error": r.error}
                      for r in all_res]).to_csv(ARTIFACTS / "grid_results.csv", index=False)
        print(f"      grid best objective={gbest.objective:.4f} "
              f"({len(ok)}/{len(all_res)} feasible, gap {grid_summary['optimality_gap']:+.4f}, "
              f"global optimum reached: {grid_summary['hill_climb_found_global_optimum']})")
    else:
        print("[3/6] exhaustive reference run skipped")

    # --- CRISP-DM 5: Evaluation --------------------------------------------
    print("[4/6] refitting winner and profiling segments ...")
    cfg = best.config
    labels = fit_labels(cfg, feat)
    feat = feat.copy()
    feat["cluster"] = labels

    X = matrix(feat, cfg.feature_set, cfg.log_transform)
    Xs = ar.make_scaler(cfg.scaler).fit_transform(X)
    feat["silhouette"] = silhouette_samples(Xs, labels)

    profile = feat.groupby("cluster").agg(
        customers=("monetary", "size"),
        recency_days=("recency_days", "median"),
        frequency=("frequency", "median"),
        monetary=("monetary", "median"),
        avg_order_value=("avg_order_value", "median"),
        avg_basket_size=("avg_basket_size", "median"),
        tenure_days=("tenure_days", "median"),
        return_rate=("return_rate", "mean"),
        rfm_score=("RFM_score", "median"),
        revenue=("monetary", "sum"),
        mean_silhouette=("silhouette", "mean"),
    ).round(3)
    profile["revenue_share"] = (profile["revenue"] / profile["revenue"].sum()).round(4)
    names = name_segments(profile)
    profile["segment"] = [names[int(c)][0] for c in profile.index]
    profile["recommended_action"] = [names[int(c)][1] for c in profile.index]
    feat["segment"] = feat["cluster"].map(lambda c: names[int(c)][0])

    print("[5/6] stability + external checks ...")
    stab = stability(cfg, feat, labels)

    # external sanity check against the generator's latent archetypes
    external = None
    truth_path = DATA_DIR / "customer_archetypes.csv"
    if truth_path.exists():
        truth = pd.read_csv(truth_path).set_index("CustomerID")["archetype"]
        common = feat.index.intersection(truth.index)
        y = truth.loc[common].astype("category").cat.codes.to_numpy()
        yhat = feat.loc[common, "cluster"].to_numpy()
        external = {
            "n_customers_matched": int(len(common)),
            "adjusted_rand_index": float(adjusted_rand_score(y, yhat)),
            "normalized_mutual_info": float(normalized_mutual_info_score(y, yhat)),
            "note": "Latent archetypes are a generator-side sanity check only; they "
                    "overlap heavily in RFM space, so high agreement is not expected "
                    "and was never optimised for.",
        }
        ct = pd.crosstab(truth.loc[common], feat.loc[common, "segment"])
        ct.to_csv(ARTIFACTS / "archetype_crosstab.csv")

    # agreement with the interpretable RFM-quintile baseline
    rfm_tier = pd.cut(feat["RFM_score"], bins=[2, 6, 9, 12, 15],
                      labels=["Bronze", "Silver", "Gold", "Platinum"], include_lowest=True)
    baseline = {
        "adjusted_rand_index_vs_rfm_quintiles": float(
            adjusted_rand_score(rfm_tier.cat.codes.to_numpy(), labels)),
        "note": "Agreement with the classic 1-5 RFM quintile tiers (Hughes 1994). "
                "Moderate agreement is expected: the tiers are axis-aligned cuts, "
                "the clusters are not.",
    }

    # elbow curve (diagnostic only -- not part of the objective)
    # elbow / index curve (diagnostic only -- not part of the objective). It spans
    # k=2 as well, which is outside the feasible region, so the honest record of
    # what the unconstrained indices prefer stays visible in the artifacts.
    elbow = []
    for k in range(2, 11):
        r = ar.evaluate(ar.Config(cfg.feature_set, cfg.log_transform, cfg.scaler, "kmeans", k), feat)
        elbow.append({"k": k, "inertia": r.inertia, "silhouette": r.silhouette,
                      "composite": r.composite, "min_cluster_share": r.min_cluster_share,
                      "feasible": r.feasible})

    # Why the constraint exists, as a number rather than an assertion: the best
    # composite reachable at each k below the feasible band, over the whole
    # preprocessing/algorithm space. Cheap (a few dozen extra fits) and it makes the
    # claim in README/CRISP_DM.md checkable from the shipped artifacts.
    print("[6/6] unconstrained reference (k below the feasible band) ...")
    unconstrained = []
    for k in range(2, ar.MIN_K + 1):
        cands = [ar.evaluate(ar.Config(fs, lg, sc, alg, k), feat)
                 for fs in ar.SEARCH_SPACE["feature_set"]
                 for lg in ar.SEARCH_SPACE["log_transform"]
                 for sc in ar.SEARCH_SPACE["scaler"]
                 for alg in ar.SEARCH_SPACE["algorithm"]]
        top = max(cands, key=lambda r: r.composite)
        unconstrained.append({
            "k": k, "best_composite": top.composite, "silhouette": top.silhouette,
            "config": asdict(top.config), "min_cluster_share": top.min_cluster_share,
            "beats_constrained_winner": bool(top.composite > best.composite)})
        print(f"      k={k}: best composite {top.composite:.4f} "
              f"({top.config.algorithm}, {top.config.scaler}, {top.config.feature_set}, "
              f"smallest segment {top.min_cluster_share:.1%})")

    print("      writing artifacts ...")
    import joblib
    from features import FEATURE_SETS

    # Persist a scoring path that works for every algorithm in the space:
    # transform -> nearest centroid in the scaled feature space. (Agglomerative
    # has no predict(), so centroids rather than the estimator are the portable
    # artifact.)
    fitted_scaler = ar.make_scaler(cfg.scaler).fit(matrix(feat, cfg.feature_set, cfg.log_transform))
    centroids = np.vstack([Xs[labels == c].mean(axis=0) for c in sorted(np.unique(labels))])
    joblib.dump({"config": asdict(cfg),
                 "feature_columns": FEATURE_SETS[cfg.feature_set],
                 "log_transform": cfg.log_transform,
                 "scaler": fitted_scaler,
                 "centroids": centroids,
                 "centroid_labels": sorted(int(c) for c in np.unique(labels)),
                 "segment_names": {int(k): v for k, v in names.items()}},
                ARTIFACTS / "best_pipeline.joblib")

    hist_df.to_csv(ARTIFACTS / "search_history.csv", index=False)
    feat.reset_index().to_csv(ARTIFACTS / "customer_segments.csv", index=False)
    profile.reset_index().to_csv(ARTIFACTS / "segment_profile.csv", index=False)
    report.to_frame().to_csv(ARTIFACTS / "cleaning_report.csv", index=False)

    metrics = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": ar.SEED,
        "runtime_seconds": round(time.perf_counter() - t_start, 2),
        "data": {
            "raw_lines": report.rows_raw,
            "clean_lines": report.rows_final,
            "customers": int(len(feat)),
            "snapshot_date": str(clean["InvoiceDate"].max().date()),
            "total_revenue": round(float(feat["monetary"].sum()), 2),
        },
        "search": {
            "strategy": "multi-restart steepest-ascent hill climbing",
            "space_size": ar.space_size(),
            "restarts": args.restarts,
            "distinct_pipelines_fitted": n_fits,
            "objective": {"weights": ar.WEIGHTS, "ch_squash_constant": ar.CH_SQUASH,
                          "definition": "0.5*(sil+1)/2 + 0.25*1/(1+DB) + 0.25*CH/(CH+500)"},
            "constraint": {"min_k": ar.MIN_K, "max_k": ar.MAX_K,
                           "min_cluster_share": ar.MIN_CLUSTER_SHARE,
                           "why": "Unconstrained internal indices score k=2 and k=3 above "
                                  "anything in the feasible band (see "
                                  "search.unconstrained_reference). A two-way split is not "
                                  "a segmentation, and k=3 adds little over the free RFM "
                                  "quintile tiers. The feasible region was therefore "
                                  "declared before the search ran, not chosen after "
                                  "seeing results."},
            "grid_reference": grid_summary,
            "unconstrained_reference": unconstrained,
        },
        "best": {
            "config": asdict(cfg),
            "silhouette": best.silhouette,
            "davies_bouldin": best.davies_bouldin,
            "calinski_harabasz": best.calinski_harabasz,
            "composite": best.composite,
            "objective": best.objective,
            "min_cluster_share": best.min_cluster_share,
            "feasible": best.feasible,
            "cluster_sizes": best.cluster_sizes,
        },
        "stability": stab,
        "external_check": external,
        "baseline_agreement": baseline,
        "elbow": elbow,
        "literature": ar.LITERATURE,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== best pipeline ===")
    for key, val in asdict(cfg).items():
        print(f"  {key:<14}: {val}")
    print(f"  silhouette    : {best.silhouette:.4f}")
    print(f"  davies_bouldin: {best.davies_bouldin:.4f}")
    print(f"  calinski_har. : {best.calinski_harabasz:.1f}")
    print(f"  composite     : {best.composite:.4f}")
    print(f"  objective     : {best.objective:.4f} "
          f"(min cluster share {best.min_cluster_share:.1%})")
    print(f"  stability ARI : {stab['ari_mean']:.3f} +/- {stab['ari_std']:.3f}")
    print("\n=== segments ===")
    print(profile[["segment", "customers", "recency_days", "frequency",
                   "monetary", "revenue_share"]].to_string())
    print(f"\nartifacts -> {ARTIFACTS}")


if __name__ == "__main__":
    main()
