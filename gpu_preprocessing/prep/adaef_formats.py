"""Byte-exact writers/readers for the Ada-ef binary formats (IMPLEMENTATION_PLAN.md §4.1).

Ada-ef's hnswdis::MatrixXf / MatrixXi are ROW-MAJOR Eigen matrices, written as
    int32 rows, int32 cols, <dtype>[rows*cols]
All values little-endian.

samplings.bin  (hnswdis::serialize_samplings):
    matrix<float32> queries, matrix<int32> ground_truth
estimator.bin  (hnswdis::save_estimator_to_file + Estimator::serialize):
    uint64 type_len, char[type_len] type,
    matrix<float32> covariance (d x d),
    int32 d, float32[d] means,
    int32 d, float32[d] variances (= diag(covariance))
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# metric -> Estimator type name expected by hnswdis::load_estimator_from_file
ESTIMATOR_TYPE = {"cd": "CosineDistanceEstimator", "ipd": "InnerProductEstimator"}


def _write_matrix(f, a: np.ndarray, dtype) -> None:
    a = np.ascontiguousarray(a, dtype=dtype)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {a.shape}")
    f.write(struct.pack("<ii", a.shape[0], a.shape[1]))
    f.write(a.astype(np.dtype(dtype).newbyteorder("<"), copy=False).tobytes(order="C"))


def _read_matrix(f, dtype) -> np.ndarray:
    rows, cols = struct.unpack("<ii", f.read(8))
    n = rows * cols
    buf = f.read(n * np.dtype(dtype).itemsize)
    if len(buf) != n * np.dtype(dtype).itemsize:
        raise ValueError("truncated matrix")
    return np.frombuffer(buf, dtype=np.dtype(dtype).newbyteorder("<")).astype(dtype).reshape(rows, cols)


def _write_vector(f, v: np.ndarray) -> None:
    v = np.ascontiguousarray(v, dtype=np.float32).ravel()
    f.write(struct.pack("<i", v.shape[0]))
    f.write(v.astype("<f4", copy=False).tobytes())


def _read_vector(f) -> np.ndarray:
    (n,) = struct.unpack("<i", f.read(4))
    return np.frombuffer(f.read(4 * n), dtype="<f4").astype(np.float32)


def write_samplings(path: str | Path, queries: np.ndarray, ground_truth: np.ndarray) -> None:
    if queries.shape[0] != ground_truth.shape[0]:
        raise ValueError("queries and ground_truth must have the same number of rows")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        _write_matrix(f, queries, np.float32)
        _write_matrix(f, ground_truth, np.int32)


def read_samplings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        q = _read_matrix(f, np.float32)
        gt = _read_matrix(f, np.int32)
    return q, gt


def write_estimator(path: str | Path, metric: str, means: np.ndarray, covariance: np.ndarray) -> None:
    """means: (d,), covariance: (d, d) unbiased (n-1). variances are taken from the diagonal."""
    if metric not in ESTIMATOR_TYPE:
        raise ValueError(f"Ada-ef supports only metrics {list(ESTIMATOR_TYPE)}; got {metric!r}")
    d = means.shape[0]
    if covariance.shape != (d, d):
        raise ValueError(f"covariance shape {covariance.shape} != ({d}, {d})")
    name = ESTIMATOR_TYPE[metric].encode("ascii")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(name)))  # size_t on 64-bit Linux/GCC
        f.write(name)
        _write_matrix(f, covariance, np.float32)
        _write_vector(f, means)
        _write_vector(f, np.diag(covariance))


def read_estimator(path: str | Path) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (type_name, means, covariance, variances)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        name = f.read(n).decode("ascii")
        cov = _read_matrix(f, np.float32)
        means = _read_vector(f)
        var = _read_vector(f)
    return name, means, cov, var
