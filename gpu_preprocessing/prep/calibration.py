"""LA-ef: calibration queries + exact ground truth -> calibration/samplings.bin (IMPLEMENTATION_PLAN.md §7, P1.7).

base_sample (the Ada-ef default): queries are base vectors, so each one is its own rank-1 neighbour.
query_holdout: queries are test vectors; their rows are listed as excluded from evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .adaef_formats import write_samplings
from .ground_truth import exact_knn
from .registry import DatasetSpec
from .sources import NpySource


def make_calibration(spec: DatasetSpec, local_dir: Path, out_dir: Path, n: int = 1000,
                     source: str = "base_sample", k: int = 10, seed: int = 42,
                     device: str = "cuda") -> dict[str, Any]:
    """Sample n calibration queries, compute their exact top-k, write samplings.bin and calib_meta.json."""
    train_path, test_path = local_dir / "train.npy", local_dir / "test.npy"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found; run the convert step first")
    base = NpySource(train_path)
    rng = np.random.default_rng(seed)

    if source == "base_sample":
        rows = np.sort(rng.choice(base.n, size=n, replace=False))
        queries = base.take_by_index(rows)
        meta = {"base_rows": rows.tolist(), "test_rows": [], "excluded_test_rows": []}
    elif source == "query_holdout":
        test = NpySource(test_path)
        if test.n < n:
            raise ValueError(f"query_holdout needs at least {n} test queries, the dataset has {test.n}")
        rows = np.sort(rng.choice(test.n, size=n, replace=False))
        queries = test.take_by_index(rows)
        meta = {"base_rows": [], "test_rows": rows.tolist(), "excluded_test_rows": rows.tolist()}
    else:
        raise ValueError(f"calibration.source must be base_sample or query_holdout, got {source!r}")

    ids, _ = exact_knn(queries, base, k, spec.metric, device=device)
    cal_dir = out_dir / "calibration"
    write_samplings(cal_dir / "samplings.bin", queries, ids)
    meta.update({
        "source": source, "n": int(n), "k": int(k), "seed": int(seed),
        "note": ("each query is a base vector and its own rank-1 neighbour (Ada-ef default)"
                 if source == "base_sample" else "query rows are excluded from evaluation"),
    })
    (cal_dir / "calib_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
