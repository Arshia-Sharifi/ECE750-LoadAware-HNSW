"""Dataset statistics for Ada-ef's Estimator (IMPLEMENTATION_PLAN.md §7, P1.8).

Replaces hnswdis::init_estimator (CPU, slow for d=1536). Streams base chunks and accumulates
sum (d) and X^T X (d x d) in float64, then
    mean = sum / n,   cov = (X^T X - n * mean mean^T) / (n - 1)      (unbiased, like Ada-ef)
For metric 'cd' the vectors are L2-normalised first (Ada-ef normalises before building stats).
"""
from __future__ import annotations

import numpy as np

from .adaef_formats import write_estimator
from .ground_truth import EPS, _check_metric
from .sources import as_source


def compute_stats(base, metric: str, device: str = "cuda", chunk_rows: int = 262_144,
                  backend: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Return (mean float64 [d], cov float64 [d, d])."""
    _check_metric(metric)
    src = as_source(base)
    if src.n < 2:
        raise ValueError("need at least 2 vectors")
    if backend == "auto":
        try:
            import torch  # noqa: F401

            backend = "torch"
        except ImportError:
            backend = "numpy"

    if backend == "numpy":
        s = np.zeros(src.d, dtype=np.float64)
        xtx = np.zeros((src.d, src.d), dtype=np.float64)
        for _, xc in src.iter_chunks(chunk_rows):
            x = xc.astype(np.float32)
            if metric == "cd":
                x = x * (1.0 / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)).astype(np.float32)
            x = x.astype(np.float64)
            s += x.sum(0)
            xtx += x.T @ x
    else:
        import torch

        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        s_t = torch.zeros(src.d, dtype=torch.float64, device=dev)
        xtx_t = torch.zeros((src.d, src.d), dtype=torch.float64, device=dev)
        for _, xc in src.iter_chunks(chunk_rows):
            x = torch.from_numpy(np.ascontiguousarray(xc, dtype=np.float32)).to(dev)
            if metric == "cd":
                x = x * (1.0 / (torch.linalg.norm(x, dim=1, keepdim=True) + EPS))
            x = x.double()
            s_t += x.sum(0)
            xtx_t += x.T @ x
        s, xtx = s_t.cpu().numpy(), xtx_t.cpu().numpy()

    n = src.n
    mean = s / n
    cov = (xtx - n * np.outer(mean, mean)) / (n - 1)
    cov = 0.5 * (cov + cov.T)  # exact symmetry
    return mean, cov


def write_stats(path, base, metric: str, **kw) -> None:
    mean, cov = compute_stats(base, metric, **kw)
    write_estimator(path, metric, mean.astype(np.float32), cov.astype(np.float32))
