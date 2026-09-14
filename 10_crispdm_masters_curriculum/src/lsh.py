"""CRISP-DM Modeling phase — sub-linear similarity search via Locality-Sensitive
Hashing (random-hyperplane / SimHash), benchmarked honestly against brute force.

The 300-product catalog is too small to show a real speedup on its own, so we
build a larger synthetic corpus by drawing noisy variants around each real
product's co-purchase embedding (i.e. "more products like this one"). This
keeps the benchmark honest: real product vectors are the queries, the
enlarged corpus is clearly synthetic, and every reported number (candidate
set size, recall, timing) comes from an actual run, not an assumption.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

RNG_SEED = 42


def build_product_vectors(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Binary co-purchase vectors: rows = products, cols = invoices."""
    pivot = pd.crosstab(df["product_id"], df["invoice_id"])
    pivot = (pivot > 0).astype(float)
    return pivot.to_numpy(), pivot.index.tolist()


def reduce_dims(X: np.ndarray, target_dim: int = 32, seed: int = RNG_SEED) -> np.ndarray:
    """Johnson-Lindenstrauss style random projection to a manageable dimension."""
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(X.shape[1], target_dim)) / np.sqrt(target_dim)
    reduced = X @ projection
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return reduced / norms


def build_synthetic_corpus(
    anchors: np.ndarray, variants_per_anchor: int = 66, noise: float = 0.35, seed: int = RNG_SEED
) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    n_anchors, dim = anchors.shape
    noisy = anchors[:, None, :] + rng.normal(scale=noise, size=(n_anchors, variants_per_anchor, dim))
    noisy = noisy.reshape(-1, dim)
    corpus = np.vstack([anchors, noisy])
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return corpus / norms


class SimHashLSH:
    """Multi-table random-hyperplane LSH for cosine similarity."""

    def __init__(self, dim: int, n_tables: int = 8, n_hyperplanes: int = 8, seed: int = RNG_SEED):
        rng = np.random.default_rng(seed + 2)
        self.n_tables = n_tables
        self.n_hyperplanes = n_hyperplanes
        self.hyperplanes = [rng.normal(size=(dim, n_hyperplanes)) for _ in range(n_tables)]
        self.tables: list[dict[tuple, list[int]]] = [dict() for _ in range(n_tables)]

    def _hash(self, table_idx: int, vectors: np.ndarray) -> np.ndarray:
        proj = vectors @ self.hyperplanes[table_idx]
        bits = (proj > 0).astype(int)
        return bits

    def index(self, corpus: np.ndarray) -> None:
        self.corpus = corpus
        for t in range(self.n_tables):
            bits = self._hash(t, corpus)
            for i, row in enumerate(bits):
                key = tuple(row.tolist())
                self.tables[t].setdefault(key, []).append(i)

    def candidates(self, query: np.ndarray) -> np.ndarray:
        found: set[int] = set()
        for t in range(self.n_tables):
            bits = self._hash(t, query.reshape(1, -1))[0]
            key = tuple(bits.tolist())
            found.update(self.tables[t].get(key, []))
        return np.fromiter(found, dtype=int)


def brute_force_topk(corpus: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    sims = corpus @ query
    return np.argsort(-sims)[:k]


def lsh_topk(lsh: SimHashLSH, corpus: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    cand_idx = lsh.candidates(query)
    if len(cand_idx) == 0:
        return np.array([], dtype=int), 0
    sims = corpus[cand_idx] @ query
    order = np.argsort(-sims)[:k]
    return cand_idx[order], len(cand_idx)


def run_benchmark(df: pd.DataFrame, k: int = 5, n_queries: int = 40) -> dict:
    raw_vectors, product_ids = build_product_vectors(df)
    anchors = reduce_dims(raw_vectors, target_dim=32)
    corpus = build_synthetic_corpus(anchors)
    n_anchors = anchors.shape[0]

    lsh = SimHashLSH(dim=anchors.shape[1])
    lsh.index(corpus)

    rng = np.random.default_rng(RNG_SEED + 3)
    query_idx = rng.choice(n_anchors, size=min(n_queries, n_anchors), replace=False)

    recalls, cand_sizes = [], []
    brute_times, lsh_times = [], []

    for qi in query_idx:
        query = anchors[qi]

        t0 = time.perf_counter()
        true_top = brute_force_topk(corpus, query, k)
        brute_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        approx_top, cand_size = lsh_topk(lsh, corpus, query, k)
        lsh_times.append(time.perf_counter() - t0)

        cand_sizes.append(cand_size)
        overlap = len(set(true_top.tolist()) & set(approx_top.tolist()))
        recalls.append(overlap / k)

    return {
        "corpus_size": int(corpus.shape[0]),
        "n_queries": int(len(query_idx)),
        "k": k,
        "avg_recall_at_k": round(float(np.mean(recalls)), 4),
        "avg_candidate_set_size": round(float(np.mean(cand_sizes)), 1),
        "candidate_fraction_of_corpus": round(float(np.mean(cand_sizes)) / corpus.shape[0], 5),
        "avg_brute_force_ms": round(float(np.mean(brute_times)) * 1000, 4),
        "avg_lsh_ms": round(float(np.mean(lsh_times)) * 1000, 4),
        "speedup_x": round(float(np.mean(brute_times)) / max(float(np.mean(lsh_times)), 1e-9), 2),
    }
