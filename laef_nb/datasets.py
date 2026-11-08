from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import config as C
from .shell import run

LOCAL_STEPS = "download,convert,gt,calib,stats,manifest"


def registry() -> dict[str, dict]:
    spec = yaml.safe_load(C.REGISTRY.read_text(encoding="utf-8"))
    return {d["id"]: d for d in spec["datasets"]}


def catalogue() -> pd.DataFrame:
    rows = []
    for d in registry().values():
        rows.append({"dataset": d["id"], "source": d["source_type"], "dim": d.get("dim"),
                     "n_base": d.get("n_base"), "n_query": d.get("n_query"),
                     "gpu_needed": d["source_type"] in ("hf_text_encode",) or not d.get("gt_provided_valid", False),
                     "role": d.get("role", "")})
    return pd.DataFrame(rows).sort_values("n_base").reset_index(drop=True)


def status(datasets, threads=()) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        base = C.ARTIFACTS / ds
        missing = [rel for rel in C.DATASET_INPUTS if not (base / rel).exists()]
        profiles = {t: (base / "profile" / f"profile_C{t}.json").exists() for t in threads}
        rows.append({
            "dataset": ds,
            "artifacts": "ok" if not missing else f"missing {len(missing)}",
            "missing": ", ".join(missing[:4]) + ("..." if len(missing) > 4 else ""),
            "profiles": ",".join(f"C{t}{'+' if ok else '-'}" for t, ok in profiles.items()),
            "ready": not missing and all(profiles.values()),
        })
    return pd.DataFrame(rows)


def prepare_local(dataset: str, device: str = "cpu", force: bool = False) -> None:
    cmd = [sys.executable, str(C.PREP / "run_preprocess.py"), "--dataset", dataset,
           "--steps", LOCAL_STEPS, "--data-root", str(C.DATA), "--out", str(C.ARTIFACTS),
           "--registry", str(C.REGISTRY), "--device", device]
    if force:
        cmd.append("--force")
    run(cmd)


def import_colab(dataset: str, colab_dir: str | Path) -> None:
    run(["bash", str(C.SCRIPTS / "prepare_dataset.sh"), dataset, str(colab_dir)],
        env={"PYTHON": sys.executable})


def meta(dataset: str) -> dict:
    p = C.ARTIFACTS / dataset / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
