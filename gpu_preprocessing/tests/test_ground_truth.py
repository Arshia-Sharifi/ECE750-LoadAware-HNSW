"""CPU tests for exact_knn and estimator stats (IMPLEMENTATION_PLAN.md §13.2)."""
import importlib.util

import numpy as np
import pytest

from prep.estimator_stats import compute_stats
from prep.ground_truth import agreement, exact_knn, normalize_np
from prep.sources import ArraySource

HAS_TORCH = importlib.util.find_spec("torch") is not None
BACKENDS = ["numpy"] + (["torch"] if HAS_TORCH else [])


def brute(q, x, k, metric):
    if metric == "cd":
        q, x = normalize_np(q), normalize_np(x)
    d = 1.0 - q.astype(np.float64) @ x.astype(np.float64).T
    ids = np.argsort(d, axis=1, kind="stable")[:, :k]
    return ids, np.take_along_axis(d, ids, 1)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((5003, 32)).astype(np.float32), rng.standard_normal((97, 32)).astype(np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("metric", ["cd", "ipd"])
@pytest.mark.parametrize("q_batch,chunk", [(16, 1000), (97, 5003), (40, 777)])  # incl. non-dividing chunks
def test_exact_matches_bruteforce(data, backend, metric, q_batch, chunk):
    x, q = data
    k = 10
    ids, dists = exact_knn(q, x, k, metric, device="cpu", q_batch=q_batch, base_chunk=chunk, backend=backend)
    ref_ids, ref_d = brute(q, x, k, metric)
    assert ids.dtype == np.int32 and ids.shape == (97, k)
    assert agreement(ref_ids, ids, k) == 1.0
    np.testing.assert_allclose(dists, ref_d, atol=1e-4)
    assert np.all(np.diff(dists, axis=1) >= -1e-6)  # ascending


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_fp16_rerank_close(data):
    x, q = data
    ids, _ = exact_knn(q, x, 10, "cd", device="cpu", backend="torch", fp16_rerank=True, base_chunk=1000)
    ref_ids, _ = brute(q, x, 10, "cd")
    assert agreement(ref_ids, ids, 10) >= 0.99


def test_source_chunks_cover_rows(data):
    x, _ = data
    rows = [s for s, _ in ArraySource(x).iter_chunks(777)]
    assert rows[0] == 0 and rows[-1] == 777 * (len(rows) - 1) and len(rows) == -(-x.shape[0] // 777)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("metric", ["cd", "ipd"])
def test_stats_match_numpy_cov(data, backend, metric):
    x, _ = data
    mean, cov = compute_stats(x, metric, device="cpu", chunk_rows=700, backend=backend)
    ref = normalize_np(x) if metric == "cd" else x
    np.testing.assert_allclose(mean, ref.astype(np.float64).mean(0), atol=1e-6)
    np.testing.assert_allclose(cov, np.cov(ref.astype(np.float64), rowvar=False, ddof=1), atol=1e-5)
