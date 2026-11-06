"""LA-ef: fingerprints and manifest.json (IMPLEMENTATION_PLAN.md §7, P1.9).

The serving machine regenerates train.npy/test.npy and compares them with the manifest fingerprint
before using the Colab outputs. The large matrices stay on the local disk, so they are verified by
content (fingerprint), not by file hash. The small artifacts are hashed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .registry import DatasetSpec

SMALL_ARTIFACTS = ("gt/neighbors.npy", "gt/gt_meta.json", "calibration/samplings.bin",
                   "calibration/calib_meta.json", "stats/estimator.bin")


def file_sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(local_dir: Path) -> str:
    """sha256 over the shapes, the full test matrix, and base rows at stride max(1, n // 1024)."""
    train = np.load(local_dir / "train.npy", mmap_mode="r")
    test = np.load(local_dir / "test.npy", mmap_mode="r")
    n = train.shape[0]
    h = hashlib.sha256()
    h.update(f"{n}x{train.shape[1]}|{test.shape[0]}x{test.shape[1]}".encode())
    h.update(np.ascontiguousarray(test[...], dtype="<f4").tobytes())
    h.update(np.ascontiguousarray(train[0:n:max(1, n // 1024)], dtype="<f4").tobytes())
    return h.hexdigest()


def holdout_sha256(local_dir: Path) -> str | None:
    """sha256 of the holdout row indices (int64), or None when the dataset has no holdout."""
    path = local_dir / "holdout.npy"
    if not path.exists():
        return None
    return hashlib.sha256(np.ascontiguousarray(np.load(path), dtype="<i8").tobytes()).hexdigest()


def _torch_versions() -> dict[str, Any]:
    try:
        import torch

        return {"torch": torch.__version__, "cuda": torch.version.cuda if torch.cuda.is_available() else None}
    except Exception:  # torch is optional on CPU-only machines
        return {"torch": None, "cuda": None}


def make_manifest(out_dir: Path, local_dir: Path, spec: DatasetSpec) -> dict[str, Any]:
    """Write out_dir/manifest.json. Call after every other artifact exists."""
    meta = json.loads((local_dir / "meta.json").read_text(encoding="utf-8"))
    man: dict[str, Any] = {
        "dataset_id": spec.id,
        "metric": spec.metric,
        "dim": meta["dim"],
        "n_base": meta["n_base"],
        "n_query": meta["n_query"],
        "seed": spec.seed,
        "holdout_seed": spec.holdout_seed,
        "holdout_sha256": holdout_sha256(local_dir),
        "fingerprint": fingerprint(local_dir),
        "train_test_npy": "verified by fingerprint (stays on the local disk; not byte-hashed)",
        "files": {},
        "torch": _torch_versions(),
    }
    for rel in SMALL_ARTIFACTS:
        p = out_dir / rel
        if p.exists():
            man["files"][rel] = {"sha256": file_sha256(p), "bytes": p.stat().st_size}
    gt_meta = out_dir / "gt" / "gt_meta.json"
    if gt_meta.exists():
        man["gt_source"] = json.loads(gt_meta.read_text(encoding="utf-8")).get("gt_source", "unknown")
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def verify_manifest(manifest_path: Path, local_dir: Path) -> list[str]:
    """Return a list of problems; an empty list means the local data matches the manifest."""
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not (local_dir / "train.npy").exists() or not (local_dir / "test.npy").exists():
        errors.append("train.npy or test.npy is missing")
    else:
        if fingerprint(local_dir) != man.get("fingerprint"):
            errors.append("fingerprint mismatch: the regenerated train.npy/test.npy differ from the manifest")
        if holdout_sha256(local_dir) != man.get("holdout_sha256"):
            errors.append("holdout indices differ from the manifest")
    for rel, info in man.get("files", {}).items():
        p = manifest_path.parent / rel
        if not p.exists():
            errors.append(f"{rel}: missing")
        elif file_sha256(p) != info["sha256"]:
            errors.append(f"{rel}: sha256 mismatch")
    return errors
