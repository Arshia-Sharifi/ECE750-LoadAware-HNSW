"""Synthetic datasets (IMPLEMENTATION_PLAN.md §5, P1.4).

Follows the Ada-ef make_blobs recipe: uniformly drawn cluster centres, Gaussian clusters with one
shared standard deviation, and optionally Zipf-distributed cluster sizes. Everything is seeded and
written as float32, so 2M x 100 rows take about 0.8 GB.
"""
from __future__ import annotations

import numpy as np

from .registry import DatasetSpec


def equal_counts(n_clusters: int, total: int) -> np.ndarray:
    """Cluster sizes that differ by at most one row."""
    counts = np.full(n_clusters, total // n_clusters, dtype=np.int64)
    counts[: total % n_clusters] += 1
    return counts


def zipf_counts(n_clusters: int, total: int, s: float, base: float, rng: np.random.Generator) -> np.ndarray:
    """Cluster sizes with P(rank r) proportional to (r + base)^(-s), for r = 1..n_clusters."""
    ranks = np.arange(1, n_clusters + 1, dtype=np.float64)
    p = (ranks + base) ** (-s)
    return rng.multinomial(total, p / p.sum()).astype(np.int64)


def make_blobs(n: int, d: int, counts: np.ndarray, cluster_std: float, center_low: float,
               center_high: float, rng: np.random.Generator) -> np.ndarray:
    """One Gaussian cluster per entry of `counts`; the rows are shuffled before returning."""
    if int(counts.sum()) != n:
        raise ValueError("cluster sizes must sum to n")
    centers = rng.uniform(center_low, center_high, size=(len(counts), d))
    X = np.empty((n, d), dtype=np.float32)
    pos = 0
    for c, cnt in enumerate(counts):
        if cnt:
            X[pos:pos + cnt] = centers[c] + cluster_std * rng.standard_normal((cnt, d))
            pos += cnt
    return X[rng.permutation(n)]


def generate_synthetic(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (base, queries) for a synthetic dataset.

    Queries are sampled from the generated rows and removed from the base (held out), as the
    datasets.yaml comment says, so no query matches itself in the ground truth. spec.n_base counts
    rows before the split; the base keeps n_base - n_query rows.
    """
    p = spec.synthetic
    kind = p.get("kind", "blobs")
    rng = np.random.default_rng(spec.seed)
    if kind == "blobs":
        counts = equal_counts(int(p.get("n_clusters", 200)), spec.n_base)
        lo, hi = p.get("center_box", [-1.0, 1.0])
    elif kind == "zipf_blobs":
        counts = zipf_counts(int(p.get("n_clusters", 5000)), spec.n_base,
                             float(p.get("zipf_s", 1.0)), float(p.get("zipf_base", 100)), rng)
        lo, hi = p.get("center_low", -0.1), p.get("center_high", 0.1)
    else:
        raise ValueError(f"{spec.id}: unknown synthetic kind {kind!r}")

    X = make_blobs(spec.n_base, spec.dim, counts, float(p.get("cluster_std", 0.05)),
                   float(lo), float(hi), rng)
    q_rows = np.sort(rng.choice(spec.n_base, size=spec.n_query, replace=False))
    keep = np.ones(spec.n_base, dtype=bool)
    keep[q_rows] = False
    return X[keep], X[q_rows]
