"""LA-ef: raw files -> canonical .npy dataset (+ provided ground truth). IMPLEMENTATION_PLAN.md §4, §7 (P1.5, §7.1).

The dataset folder (local disk, large) holds train.npy, test.npy, holdout.npy (holdout datasets only) and
meta.json. Small artifacts (gt/, calibration/, stats/, manifest.json) go to out_dir. Matrices are written
chunk by chunk through numpy's memmap, so the full base is never held in memory twice.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py  # the raw ann-benchmarks download is HDF5 upstream; this is the only place h5py is used
import numpy as np

from .download import raw_file
from .registry import DatasetSpec
from .sources import AnnHdf5Source, ParquetSource
from .synthetic import generate_synthetic

WRITE_CHUNK = 65_536


def _base_source(spec: DatasetSpec, raw_root: Path):
    if spec.source_type == "ann_hdf5":
        return AnnHdf5Source(raw_file(spec, raw_root), "train")
    if spec.source_type == "hf_parquet":
        if not spec.column:
            raise ValueError(f"{spec.id}: hf_parquet needs a 'column' entry in datasets.yaml")
        files = sorted((raw_root / spec.id).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"{spec.id}: no parquet files in {raw_root / spec.id}; run the download step")
        return ParquetSource(files, spec.column)
    raise ValueError(f"{spec.id}: no file-based source for source_type {spec.source_type!r}")


def save_npy(path: Path, array: np.ndarray) -> None:
    """np.save through a temporary file, so an interrupted run never leaves a half-written .npy behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def write_gt(gt_dir: Path, ids: np.ndarray, dists: np.ndarray | None, gt_source: str, metric: str) -> None:
    """gt/neighbors.npy (Q x K int32), optional gt/distances.npy (Q x K float32), gt/gt_meta.json."""
    gt_dir.mkdir(parents=True, exist_ok=True)
    save_npy(gt_dir / "neighbors.npy", np.asarray(ids, dtype=np.int32))
    if dists is not None:
        save_npy(gt_dir / "distances.npy", np.asarray(dists, dtype=np.float32))
    meta = {"gt_source": gt_source, "K": int(np.shape(ids)[1]), "metric": metric}
    (gt_dir / "gt_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def copy_provided_gt(spec: DatasetSpec, raw_root: Path, out_dir: Path, k_store: int) -> dict[str, Any]:
    """Copy the dataset's own top-K ground truth (first k_store columns) into out_dir/gt/."""
    with h5py.File(raw_file(spec, raw_root), "r") as f:
        if "neighbors" not in f:
            raise KeyError(f"{spec.id}: raw file has no 'neighbors'; cannot use provided ground truth")
        ids = np.asarray(f["neighbors"][:, :k_store], dtype=np.int32)
        dists = np.asarray(f["distances"][:, :k_store], dtype=np.float32) if "distances" in f else None
    write_gt(out_dir / "gt", ids, dists, "provided", spec.metric)
    return {"gt_source": "provided", "K": int(ids.shape[1])}


def _read_provided_queries(spec: DatasetSpec, raw_root: Path) -> np.ndarray:
    with h5py.File(raw_file(spec, raw_root), "r") as f:
        key = "test" if "test" in f else "queries"
        if key not in f:
            raise KeyError(f"{spec.id}: the raw file has neither 'test' nor 'queries'")
        return np.asarray(f[key][...], dtype=np.float32)


def convert_dataset(spec: DatasetSpec, raw_root: Path, local_dir: Path, out_dir: Path,
                    k_store: int = 100) -> dict[str, Any]:
    """Write local_dir/{train.npy, test.npy, holdout.npy?, meta.json}; provided GT goes to out_dir/gt/."""
    local_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    mats: dict[str, np.memmap] = {}

    if spec.source_type == "synthetic":
        base, queries = generate_synthetic(spec)
        n_base, n_query, d = base.shape[0], queries.shape[0], base.shape[1]
        holdout = None
        mats["train"] = np.lib.format.open_memmap(local_dir / "train.npy.tmp", mode="w+", dtype="<f4", shape=(n_base, d))
        mats["train"][:] = base
        mats["test"] = np.lib.format.open_memmap(local_dir / "test.npy.tmp", mode="w+", dtype="<f4", shape=(n_query, d))
        mats["test"][:] = queries
    else:
        src = _base_source(spec, raw_root)
        d = src.d
        if spec.query_source == "holdout":
            n_query = spec.n_query
            hold = np.sort(np.random.default_rng(spec.holdout_seed).choice(src.n, size=n_query, replace=False))
            n_base = src.n - n_query
            is_hold = np.zeros(src.n, dtype=bool)
            is_hold[hold] = True
            mats["train"] = np.lib.format.open_memmap(local_dir / "train.npy.tmp", mode="w+", dtype="<f4", shape=(n_base, d))
            mats["test"] = np.lib.format.open_memmap(local_dir / "test.npy.tmp", mode="w+", dtype="<f4", shape=(n_query, d))
            pb = pq = 0
            for start, block in src.iter_chunks(WRITE_CHUNK):  # one streaming pass, rows keep their order
                m = is_hold[start:start + len(block)]
                b, q = block[~m], block[m]
                mats["train"][pb:pb + len(b)] = b
                mats["test"][pq:pq + len(q)] = q
                pb += len(b)
                pq += len(q)
            if pb != n_base or pq != n_query:
                raise RuntimeError(f"holdout split wrote {pb}/{n_base} base and {pq}/{n_query} query rows")
            holdout = hold.astype("<i8")
        elif spec.query_source == "provided":
            queries = _read_provided_queries(spec, raw_root)
            if queries.shape[1] != d:
                raise ValueError(f"{spec.id}: query dim {queries.shape[1]} != base dim {d}")
            n_base, n_query = src.n, len(queries)
            mats["train"] = np.lib.format.open_memmap(local_dir / "train.npy.tmp", mode="w+", dtype="<f4", shape=(n_base, d))
            pb = 0
            for _, block in src.iter_chunks(WRITE_CHUNK):
                mats["train"][pb:pb + len(block)] = block
                pb += len(block)
            mats["test"] = np.lib.format.open_memmap(local_dir / "test.npy.tmp", mode="w+", dtype="<f4", shape=(n_query, d))
            mats["test"][:] = queries
            holdout = None
        else:
            raise ValueError(f"{spec.id}: query_source {spec.query_source!r} is not valid for {spec.source_type}")

    if spec.dim and d != spec.dim:
        raise ValueError(f"{spec.id}: expected dim {spec.dim}, found {d}")

    # Flush and release the memmaps, then move the finished files into place.
    for m in mats.values():
        m.flush()
    mats.clear()
    for name in ("train", "test"):
        os.replace(local_dir / f"{name}.npy.tmp", local_dir / f"{name}.npy")
    if holdout is not None:
        save_npy(local_dir / "holdout.npy", holdout)

    meta = {"dataset_id": spec.id, "metric": spec.metric, "dim": int(d), "n_base": int(n_base),
            "n_query": int(n_query), "seed": int(spec.seed), "holdout_seed": int(spec.holdout_seed),
            "query_source": spec.query_source, "source": spec.id}
    (local_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    gt_meta: dict[str, Any] = {}
    if spec.gt_provided_valid and spec.source_type == "ann_hdf5":
        gt_meta = copy_provided_gt(spec, raw_root, out_dir, k_store)
    return {"dataset_id": spec.id, "metric": spec.metric, "dim": int(d), "n_base": int(n_base),
            "n_query": int(n_query), "query_source": spec.query_source, "gt": gt_meta}
