"""Chunked vector sources (IMPLEMENTATION_PLAN.md §7, P1.3).

NpySource reads our canonical matrices (train.npy, test.npy) through a memory map.
AnnHdf5Source reads the raw ann-benchmarks files, which are HDF5 upstream; it is used only by the
convert step. ParquetSource reads the raw dbpedia shards. Consumers read through iter_chunks
(streaming) or take_by_index (small random samples). Row ids are the canonical base ids (§4.2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - only the raw ann-benchmarks conversion needs h5py
    h5py = None


class VectorSource(Protocol):
    """Structural type: n, d, iter_chunks(chunk_rows), take_by_index(idx)."""

    n: int
    d: int

    def iter_chunks(self, chunk_rows: int) -> Iterator[tuple[int, np.ndarray]]: ...

    def take_by_index(self, idx) -> np.ndarray: ...


def _check_indices(idx, n: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).ravel()
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise IndexError(f"row index out of range [0, {n})")
    return idx


class ArraySource:
    """An in-memory array (synthetic data, tests)."""

    def __init__(self, arr: np.ndarray):
        self.arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.n, self.d = self.arr.shape

    def iter_chunks(self, chunk_rows: int) -> Iterator[tuple[int, np.ndarray]]:
        for start in range(0, self.n, chunk_rows):
            yield start, self.arr[start:start + chunk_rows]

    def take_by_index(self, idx) -> np.ndarray:
        return self.arr[_check_indices(idx, self.n)].copy()


class NpySource:
    """A float32 matrix saved with numpy (train.npy, test.npy), read through a memory map."""

    def __init__(self, path):
        self.path = str(path)
        self._arr = np.load(self.path, mmap_mode="r")
        if self._arr.ndim != 2:
            raise ValueError(f"{self.path}: expected a 2-D matrix")
        if self._arr.dtype != np.float32:
            raise ValueError(f"{self.path}: expected float32, found {self._arr.dtype}")
        self.n, self.d = self._arr.shape

    def iter_chunks(self, chunk_rows: int) -> Iterator[tuple[int, np.ndarray]]:
        for start in range(0, self.n, chunk_rows):
            yield start, np.asarray(self._arr[start:start + chunk_rows])

    def take_by_index(self, idx) -> np.ndarray:
        idx = _check_indices(idx, self.n)
        if idx.size == 0:
            return np.empty((0, self.d), dtype=np.float32)
        return np.array(self._arr[idx], dtype=np.float32)  # fancy indexing copies; any order works


class AnnHdf5Source:
    """Rows of a raw ann-benchmarks HDF5 file (dataset 'train'). Used only by the convert step."""

    def __init__(self, path, key: str = "train"):
        if h5py is None:
            raise RuntimeError("h5py is needed to read the raw ann-benchmarks file")
        self.path, self.key = str(path), key
        with h5py.File(self.path, "r") as f:
            self.n, self.d = f[key].shape

    def iter_chunks(self, chunk_rows: int) -> Iterator[tuple[int, np.ndarray]]:
        with h5py.File(self.path, "r") as f:
            ds = f[self.key]
            for start in range(0, self.n, chunk_rows):
                stop = min(start + chunk_rows, self.n)
                yield start, np.asarray(ds[start:stop], dtype=np.float32)

    def take_by_index(self, idx) -> np.ndarray:
        idx = _check_indices(idx, self.n)
        if idx.size == 0:
            return np.empty((0, self.d), dtype=np.float32)
        uniq, inv = np.unique(idx, return_inverse=True)  # h5py needs increasing indices
        with h5py.File(self.path, "r") as f:
            return np.asarray(f[self.key][uniq], dtype=np.float32)[inv]


class ParquetSource:
    """A list-valued embedding column spread over parquet files (e.g. dbpedia-openai-1m).

    Batches go through pyarrow's flatten/to_numpy, so no per-element Python objects are created.
    """

    def __init__(self, files, column: str):
        import pyarrow.parquet as pq

        self.files = [str(f) for f in files]
        if not self.files:
            raise ValueError("ParquetSource: no parquet files given")
        self.column = column
        self.n = sum(pq.ParquetFile(f).metadata.num_rows for f in self.files)
        first = next(pq.ParquetFile(self.files[0]).iter_batches(batch_size=1, columns=[column]))
        self.d = len(first.column(0)[0].as_py())

    def _to_matrix(self, arr) -> np.ndarray:
        if arr.null_count:
            raise ValueError(f"column {self.column!r} has null embeddings")
        values = np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
        if values.size != len(arr) * self.d:
            raise ValueError(f"column {self.column!r}: every row must have dimension {self.d}")
        return values.reshape(len(arr), self.d)

    def iter_chunks(self, chunk_rows: int) -> Iterator[tuple[int, np.ndarray]]:
        import pyarrow.parquet as pq

        start = 0
        for f in self.files:  # shard order is row order
            for batch in pq.ParquetFile(f).iter_batches(batch_size=chunk_rows, columns=[self.column]):
                block = self._to_matrix(batch.column(0))
                yield start, block
                start += len(block)

    def take_by_index(self, idx) -> np.ndarray:
        idx = _check_indices(idx, self.n)
        if idx.size == 0:
            return np.empty((0, self.d), dtype=np.float32)
        uniq = np.unique(idx)
        pieces = []
        for start, block in self.iter_chunks(65_536):
            lo, hi = np.searchsorted(uniq, [start, start + len(block)])
            if hi > lo:
                pieces.append(block[uniq[lo:hi] - start])
        rows_sorted = np.concatenate(pieces, axis=0)
        return rows_sorted[np.searchsorted(uniq, idx)]


def as_source(obj):
    """Wrap an ndarray, a .npy path, or pass through an existing source."""
    if isinstance(obj, np.ndarray):
        return ArraySource(obj)
    if isinstance(obj, (str, Path)) and str(obj).endswith(".npy"):
        return NpySource(obj)
    if hasattr(obj, "iter_chunks") and hasattr(obj, "n") and hasattr(obj, "d"):
        return obj
    raise TypeError(f"cannot use {type(obj).__name__} as a vector source")
