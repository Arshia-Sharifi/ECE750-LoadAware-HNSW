from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from . import config as C
from .shell import run

LOG = C.RESULTS / "pipeline.log"


def _bin(app: str) -> Path:
    p = C.BIN / app
    if not p.exists():
        raise RuntimeError(f"{p} is missing. Run the build section first.")
    return p


def build_index(dataset: str, jobs: int | None = None, force: bool = False) -> Path:
    d = C.ARTIFACTS / dataset
    out = d / "index" / f"hnsw_M{C.M}_efc{C.EFC}.bin"
    if out.exists() and not force:
        print(f"[index] {dataset}: exists")
        return out
    run([_bin("laef_build_index"), "--dataset-dir", d, "--M", C.M, "--efc", C.EFC,
         "--threads", jobs or os.cpu_count() or 1], log=LOG, quiet=True)
    return out


def calibrate(dataset: str, jobs: int | None = None, force: bool = False) -> Path:
    d = C.ARTIFACTS / dataset
    out = d / "tables" / "probe_menu.json"
    if out.exists() and not force:
        print(f"[calibrate] {dataset}: exists")
        return out
    run([_bin("laef_calibrate"), "--dataset-dir", d, "--M", C.M, "--efc", C.EFC,
         "--probe-lengths", C.PROBE_LENGTHS, "--threads", jobs or os.cpu_count() or 1],
        log=LOG, quiet=True)
    return out


def profile(dataset: str, threads: int, sat_seconds: float = 120.0, force: bool = False) -> Path:
    d = C.ARTIFACTS / dataset
    out = d / "profile" / f"profile_C{threads}.json"
    if out.exists() and not force:
        print(f"[profile] {dataset} C={threads}: exists")
        return out
    print(f"[profile] {dataset} C={threads}", flush=True)
    run([_bin("laef_profile"), "--dataset-dir", d, "--threads", threads,
         "--M", C.M, "--efc", C.EFC, "--sat-seconds", sat_seconds], log=LOG, quiet=True)
    return out


def prepare(datasets, threads, sat_seconds: float = 120.0, force: bool = False,
            build_jobs: int | None = None) -> pd.DataFrame:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        build_index(ds, build_jobs, force)
        calibrate(ds, build_jobs, force)
        for t in threads:
            profile(ds, t, sat_seconds, force)
    return profiles(datasets, threads)


def probe_menu(dataset: str, tables: Path | None = None) -> pd.DataFrame:
    p = (tables or C.ARTIFACTS / dataset / "tables") / "probe_menu.json"
    if not p.exists():
        return pd.DataFrame()
    j = json.loads(p.read_text(encoding="utf-8"))
    kept = [{"dataset": dataset, "status": "kept", "probe_len": o.get("probe_len"),
             "nominal_work_dc": o.get("model_nominal_work"), "floor_work_dc": o.get("model_floor_work"),
             "nominal_recall": o.get("model_nominal_recall"), "ef_pref_mean": o.get("ef_pref_mean"),
             "note": ""}
            for o in j.get("options", [])]
    dropped = [{"dataset": dataset, "status": "dropped", "probe_len": d.get("probe_len"),
                "nominal_work_dc": None, "floor_work_dc": None, "nominal_recall": None,
                "ef_pref_mean": None, "note": d.get("reason", "")}
               for d in j.get("dropped", [])]
    return pd.DataFrame(kept + dropped)


def profiles(datasets, threads) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for t in threads:
            p = C.ARTIFACTS / ds / "profile" / f"profile_C{t}.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text(encoding="utf-8"))
            sat, idle = j.get("saturation", {}), j.get("idle", {})
            rows.append({
                "dataset": ds, "threads": t,
                "mu_sat_qps": sat.get("mu_sat_qps"),
                "s_p95_ms": sat.get("s_p95_s", 0) * 1e3,
                "mean_service_ms": sat.get("mean_service_s", 0) * 1e3,
                "u_ns_per_dc": sat.get("u_s_per_dc", 0) * 1e9,
                "idle_mean_ms": idle.get("mean_s", 0) * 1e3,
                "idle_p95_ms": idle.get("p95_s", 0) * 1e3,
                "profile_s": sat.get("seconds"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["dataset", "threads"]).reset_index(drop=True)
    g = df.groupby("dataset")
    df["speedup"] = df.mu_sat_qps / g.mu_sat_qps.transform("first")
    df["efficiency"] = df.speedup / (df.threads / g.threads.transform("first"))
    return df
