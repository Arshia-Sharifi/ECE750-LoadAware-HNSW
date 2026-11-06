"""Exact k-NN ground truth, streamed over base chunks (IMPLEMENTATION_PLAN.md §7, P1.6).

Distances follow hnswlib/Ada-ef definitions so GT matches what the index is scored against:
    cd  (cosine distance):        both sides L2-normalised as v / (||v|| + 1e-30), d = 1 - q.x
    ipd (inner-product distance): raw vectors,                                    d = 1 - q.x
Results are sorted by ascending distance; ids are canonical base row ids (int32).

Backend: torch (cuda or cpu) when importable, else numpy (CPU-only, for tests/small data).
Memory: one [q_batch, base_chunk] similarity block at a time, plus a second block of the same size for
1 - similarity. The defaults (q_batch 1024, base_chunk 65536) make each block 256 MiB, and a 1536-d base
chunk 384 MiB, so they fit on a 16 GB T4. The earlier defaults (4096 x 262144) needed two 4 GiB blocks.
"""
from __future__ import annotations

import numpy as np

from .sources import VectorSource, as_source

EPS = 1e-30  # identical to Ada-ef normalize_matrix


def normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x * (1.0 / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)).astype(np.float32)


def _check_metric(metric: str) -> None:
    if metric not in ("cd", "ipd"):
        raise ValueError(f"Ada-ef supports only 'cd' and 'ipd', got {metric!r}")


def exact_knn(
    queries: np.ndarray,
    base,
    k: int,
    metric: str,
    device: str = "cuda",
    q_batch: int = 1024,
    base_chunk: int = 65_536,
    backend: str = "auto",
    fp16_rerank: bool = False,
    rerank_margin: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids int32 [Q, k], dists float32 [Q, k]) sorted by ascending distance.

    base: np.ndarray or VectorSource. backend: 'auto' | 'torch' | 'numpy'.
    fp16_rerank (torch only): select top-(k+margin) per chunk in fp16, re-score exactly in fp32.
    """
    _check_metric(metric)
    src: VectorSource = as_source(base)
    if k > src.n:
        raise ValueError(f"k={k} > n_base={src.n}")
    q = np.asarray(queries, dtype=np.float32)
    if metric == "cd":
        q = normalize_np(q)

    if backend == "auto":
        try:
            import torch  # noqa: F401

            backend = "torch"
        except ImportError:
            backend = "numpy"
    if backend == "numpy":
        return _knn_numpy(q, src, k, metric, q_batch, base_chunk)
    return _knn_torch(q, src, k, metric, device, q_batch, base_chunk, fp16_rerank, rerank_margin)


def _knn_numpy(q, src, k, metric, q_batch, base_chunk):
    Q = q.shape[0]
    best_d = np.full((Q, k), np.inf, dtype=np.float32)
    best_i = np.full((Q, k), -1, dtype=np.int64)
    for start, xc in src.iter_chunks(base_chunk):
        if metric == "cd":
            xc = normalize_np(xc)
        for qs in range(0, Q, q_batch):
            qe = min(qs + q_batch, Q)
            dist = 1.0 - q[qs:qe] @ xc.T  # [qb, chunk]
            kk = min(k, dist.shape[1])
            part = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
            cd = np.take_along_axis(dist, part, axis=1)
            ci = part + start
            md = np.concatenate([best_d[qs:qe], cd], axis=1)
            mi = np.concatenate([best_i[qs:qe], ci], axis=1)
            sel = np.argpartition(md, k - 1, axis=1)[:, :k]
            best_d[qs:qe] = np.take_along_axis(md, sel, axis=1)
            best_i[qs:qe] = np.take_along_axis(mi, sel, axis=1)
    order = np.argsort(best_d, axis=1, kind="stable")
    return (np.take_along_axis(best_i, order, 1).astype(np.int32),
            np.take_along_axis(best_d, order, 1).astype(np.float32))


def _knn_torch(q, src, k, metric, device, q_batch, base_chunk, fp16_rerank, margin):
    import torch

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    Q = q.shape[0]
    qt = torch.from_numpy(q).to(dev)
    best_d = torch.full((Q, k), float("inf"), device=dev, dtype=torch.float32)
    best_i = torch.full((Q, k), -1, device=dev, dtype=torch.int64)
    pin = dev.type == "cuda"
    for start, xc in src.iter_chunks(base_chunk):
        xt = torch.from_numpy(np.ascontiguousarray(xc, dtype=np.float32))
        if pin:
            xt = xt.pin_memory()
        xt = xt.to(dev, non_blocking=True)
        if metric == "cd":
            xt = xt * (1.0 / (torch.linalg.norm(xt, dim=1, keepdim=True) + EPS))
        xt_h = xt.half() if fp16_rerank else None
        for qs in range(0, Q, q_batch):
            qe = min(qs + q_batch, Q)
            qb = qt[qs:qe]
            if fp16_rerank:
                sim_h = qb.half() @ xt_h.T
                kk = min(k + margin, sim_h.shape[1])
                cand = torch.topk(sim_h, kk, dim=1).indices  # approximate candidates
                exact = torch.einsum("qd,qkd->qk", qb, xt[cand])  # fp32 re-score
                cd = 1.0 - exact
                kk2 = min(k, kk)
                cd, pos = torch.topk(cd, kk2, dim=1, largest=False)
                ci = torch.gather(cand, 1, pos) + start
            else:
                dist = 1.0 - qb @ xt.T
                kk2 = min(k, dist.shape[1])
                cd, ci = torch.topk(dist, kk2, dim=1, largest=False)
                ci = ci + start
            md = torch.cat([best_d[qs:qe], cd], dim=1)
            mi = torch.cat([best_i[qs:qe], ci], dim=1)
            nd, sel = torch.topk(md, k, dim=1, largest=False)
            best_d[qs:qe] = nd
            best_i[qs:qe] = torch.gather(mi, 1, sel)
    best_d, order = torch.sort(best_d, dim=1, stable=True)
    best_i = torch.gather(best_i, 1, order)
    return best_i.cpu().numpy().astype(np.int32), best_d.cpu().numpy().astype(np.float32)


def agreement(provided_ids: np.ndarray, exact_ids: np.ndarray, k: int) -> float:
    """Mean recall@k of provided GT against our exact GT (used by --verify-provided)."""
    hits = [len(set(a[:k]) & set(b[:k])) / k for a, b in zip(provided_ids, exact_ids)]
    return float(np.mean(hits))
