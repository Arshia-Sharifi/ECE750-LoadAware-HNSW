#!/usr/bin/env python
"""LA-ef preprocessing CLI (IMPLEMENTATION_PLAN.md §7.1).

    python run_preprocess.py --dataset <id> --steps download,convert,gt,calib,stats,manifest \
        --data-root /content/data --out /content/drive/MyDrive/laef_artifacts \
        --registry ../configs/datasets.yaml --device cuda --k 10 --k-store 100 \
        --calib-n 1000 --calib-source base_sample [--verify-provided 500] [--force]

Locations:
  <data-root>/raw/<id>/                 raw downloads (ann-benchmarks files are HDF5 upstream)
  <data-root>/artifacts/<id>/           train.npy, test.npy, holdout.npy, meta.json (large, local disk)
  <out>/<id>/                           gt/, calibration/, stats/, manifest.json (small)
                                        (default --out: <data-root>/artifacts, i.e. the same folder)

Every step is idempotent: it is skipped when its output already exists. --force reruns the selected
steps. Raw downloads are never deleted. Exit code 0 on success, 1 on any failure.

The gt, calib, stats and manifest steps need a CUDA GPU, so they run on Colab only. On the serving machine
run only download,convert (scripts/prepare_dataset.sh does this). The CLI refuses the GPU steps without CUDA.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prep import calibration, convert, download, estimator_stats, ground_truth, manifest, registry  # noqa: E402
from prep.sources import NpySource  # noqa: E402

KNOWN_STEPS = ("download", "convert", "gt", "calib", "stats", "manifest")
GPU_STEPS = ("gt", "calib", "stats", "manifest")  # need a CUDA GPU (Colab)
VERIFY_THRESHOLD = 0.999


def step_download(spec, raw_root: Path) -> None:
    if spec.source_type == "synthetic":
        return
    print(f"[download] {spec.id}", flush=True)
    download.download_raw(spec, raw_root)


def step_convert(spec, raw_root: Path, local_dir: Path, out_dir: Path, k_store: int, force: bool) -> None:
    if (local_dir / "train.npy").exists() and not force:
        print(f"[convert] {spec.id}: train.npy exists, skipping", flush=True)
        return
    print(f"[convert] {spec.id}", flush=True)
    meta = convert.convert_dataset(spec, raw_root, local_dir, out_dir, k_store=k_store)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "convert_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def verify_provided(spec, local_dir: Path, out_dir: Path, n_verify: int, k: int, device: str) -> None:
    """Recompute exact top-k for a random sample of test queries and compare with the provided GT."""
    base, test = NpySource(local_dir / "train.npy"), NpySource(local_dir / "test.npy")
    rows = np.sort(np.random.default_rng(12345).choice(test.n, size=min(n_verify, test.n), replace=False))
    ids_exact, _ = ground_truth.exact_knn(test.take_by_index(rows), base, k, spec.metric, device=device)
    ids_provided = np.load(out_dir / "gt" / "neighbors.npy", mmap_mode="r")[rows, :k]
    agree = ground_truth.agreement(np.asarray(ids_provided), ids_exact, k)
    print(f"[gt] {spec.id}: recall@{k} of provided vs exact on {len(rows)} queries = {agree:.4f}", flush=True)
    if agree < VERIFY_THRESHOLD:
        raise RuntimeError(f"provided ground truth disagrees with exact search: {agree:.4f} < {VERIFY_THRESHOLD}")


def step_gt(spec, raw_root: Path, local_dir: Path, out_dir: Path, k_store: int, device: str,
            verify: int, force: bool) -> None:
    gt_dir = out_dir / "gt"
    if (gt_dir / "neighbors.npy").exists() and not force:
        print(f"[gt] {spec.id}: exists, skipping", flush=True)
    elif spec.gt_provided_valid:
        print(f"[gt] {spec.id}: copying the provided ground truth", flush=True)
        convert.copy_provided_gt(spec, raw_root, out_dir, k_store)
    else:
        print(f"[gt] {spec.id}: exact k-NN, k={k_store}, device={device}", flush=True)
        queries = NpySource(local_dir / "test.npy")
        ids, dists = ground_truth.exact_knn(queries.take_by_index(np.arange(queries.n)),
                                            NpySource(local_dir / "train.npy"), k_store, spec.metric, device=device)
        convert.write_gt(gt_dir, ids, dists, "generated", spec.metric)
    if verify and spec.gt_provided_valid:
        verify_provided(spec, local_dir, out_dir, verify, 10, device)


def step_calib(spec, local_dir: Path, out_dir: Path, n: int, source: str, k: int, seed: int,
               device: str, force: bool) -> dict:
    cal_dir = out_dir / "calibration"
    if (cal_dir / "samplings.bin").exists() and not force:
        print(f"[calib] {spec.id}: exists, skipping", flush=True)
        return json.loads((cal_dir / "calib_meta.json").read_text(encoding="utf-8"))
    print(f"[calib] {spec.id}: {n} queries, source={source}, k={k}", flush=True)
    return calibration.make_calibration(spec, local_dir, out_dir, n=n, source=source, k=k, seed=seed, device=device)


def step_stats(spec, local_dir: Path, out_dir: Path, device: str, force: bool) -> None:
    path = out_dir / "stats" / "estimator.bin"
    if path.exists() and not force:
        print(f"[stats] {spec.id}: exists, skipping", flush=True)
        return
    print(f"[stats] {spec.id}: mean and covariance", flush=True)
    estimator_stats.write_stats(path, NpySource(local_dir / "train.npy"), spec.metric, device=device)


def step_manifest(spec, local_dir: Path, out_dir: Path, force: bool) -> dict:
    path = out_dir / "manifest.json"
    if path.exists() and not force:
        print(f"[manifest] {spec.id}: exists, skipping", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"[manifest] {spec.id}", flush=True)
    return manifest.make_manifest(out_dir, local_dir, spec)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LA-ef GPU/CPU preprocessing (plan §7.1)")
    ap.add_argument("--dataset", required=True, help="dataset id from configs/datasets.yaml")
    ap.add_argument("--steps", default=",".join(KNOWN_STEPS))
    ap.add_argument("--data-root", default="./data", help="raw/ and artifacts/ live here")
    ap.add_argument("--out", default=None, help="folder for small artifacts (default: <data-root>/artifacts)")
    ap.add_argument("--registry", default=None, help="path to datasets.yaml")
    ap.add_argument("--device", default="cuda", choices=["cuda"], help="device for the GPU steps (Colab)")
    ap.add_argument("--k", type=int, default=10, help="serving k (calibration ground truth width)")
    ap.add_argument("--k-store", type=int, default=100, help="ground truth columns stored")
    ap.add_argument("--calib-n", type=int, default=1000)
    ap.add_argument("--calib-source", default="base_sample", choices=["base_sample", "query_holdout"])
    ap.add_argument("--calib-seed", type=int, default=None, help="default: the dataset seed")
    ap.add_argument("--verify-provided", type=int, default=0, help="re-check N provided GT rows")
    ap.add_argument("--force", action="store_true", help="rerun the selected steps")
    args = ap.parse_args(argv)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in KNOWN_STEPS]
    if unknown:
        ap.error(f"unknown steps: {unknown}; choose from {list(KNOWN_STEPS)}")
    gpu_steps = [s for s in steps if s in GPU_STEPS]
    if gpu_steps and not _cuda_available():
        ap.error(f"steps {gpu_steps} need a CUDA GPU and run on Colab; on the serving machine use --steps download,convert")

    try:
        spec = registry.get_dataset(args.dataset, args.registry)
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    data_root = Path(args.data_root).resolve()
    raw_root = data_root / "raw"
    local_dir = data_root / "artifacts" / spec.id
    out_dir = Path(args.out).resolve() / spec.id if args.out else local_dir
    calib_seed = args.calib_seed if args.calib_seed is not None else spec.seed

    t0 = time.time()
    try:
        if "download" in steps:
            step_download(spec, raw_root)
        if "convert" in steps:
            step_convert(spec, raw_root, local_dir, out_dir, args.k_store, args.force)
        if any(s in steps for s in ("gt", "calib", "stats", "manifest")) and not (local_dir / "train.npy").exists():
            raise FileNotFoundError(f"{local_dir / 'train.npy'} is missing; run the convert step first")
        if "gt" in steps:
            step_gt(spec, raw_root, local_dir, out_dir, args.k_store, args.device, args.verify_provided, args.force)
        if "calib" in steps:
            step_calib(spec, local_dir, out_dir, args.calib_n, args.calib_source, args.k, calib_seed,
                       args.device, args.force)
        if "stats" in steps:
            step_stats(spec, local_dir, out_dir, args.device, args.force)
        if "manifest" in steps:
            step_manifest(spec, local_dir, out_dir, args.force)
    except Exception as e:  # report every failure; the notebook moves on to the next dataset
        print(f"[error] {spec.id}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(f"[done] {spec.id} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
