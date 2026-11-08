from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .config import Study

KEY = ["dataset", "threads", "rho", "repeat", "profile"]
NUMERIC = [
    "threads", "rho", "repeat", "slo_fraction", "profile_p95_ms", "slo_ms", "n_queries",
    "offered_qps", "achieved_qps", "goodput_qps", "p50_ms", "p95_ms", "p99_ms",
    "mean_wait_ms", "mean_service_ms", "slo_attainment", "latency_excess_ms",
    "mean_recall", "p5_recall", "p1_recall", "frac_target_recall", "mean_ef_ratio",
    "mean_ndis", "forced_floor_rate", "floor_mode_query_rate", "mean_probe_len",
    "frac_full_probe", "frac_windows_violation", "frac_windows_headroom", "frac_windows_comfort",
    "guard_fires", "floor_mode_windows", "p95_window_violations",
    "controller_update_us", "generator_lag_p99_us", "wall_s",
]


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summary(study: Study) -> pd.DataFrame:
    rows = []
    for d in sorted(p.parent for p in study.dir.rglob("run_meta.json")):
        s = d / "summary.json"
        if not s.exists():
            continue
        rows.append({**json.loads((d / "run_meta.json").read_text(encoding="utf-8")),
                     **json.loads(s.read_text(encoding="utf-8")), "run_dir": str(d)})
    if not rows:
        return pd.DataFrame()
    df = _numeric(pd.DataFrame(rows))
    df["label"] = df.policy.map(C.POLICY_LABEL).fillna(df.policy)
    return df.sort_values(["dataset", "threads", "rho", "policy"]).reset_index(drop=True)


def cut_diagnostics(df: pd.DataFrame, near_one: float = 0.98) -> pd.DataFrame:
    if df.empty:
        return df
    d = df[df.policy != C.BASELINE_POLICY]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["dataset", "threads", "policy"]).agg(
        runs=("run_id", "count"),
        ef_ratio_min=("mean_ef_ratio", "min"),
        ef_ratio_mean=("mean_ef_ratio", "mean"),
        frac_runs_uncut=("mean_ef_ratio", lambda s: float((s >= near_one).mean())),
        forced_floor=("forced_floor_rate", "mean"),
        floor_mode=("floor_mode_query_rate", "mean"),
        frac_full_probe=("frac_full_probe", "mean"),
        mean_ndis=("mean_ndis", "mean"),
    ).reset_index()
    g["verdict"] = np.where(g.frac_runs_uncut >= 0.5, "allocator barely cutting", "cutting")
    return g


def paired(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = df[df.policy == C.BASELINE_POLICY]
    dup = base.duplicated(KEY).sum()
    if dup:
        raise ValueError(f"{dup} duplicate {C.BASELINE_POLICY} runs for the same {KEY}. "
                         "Two studies were written into one results folder.")
    cols = KEY + ["mean_recall", "p5_recall", "p1_recall", "p95_ms", "slo_attainment",
                  "achieved_qps", "goodput_qps", "mean_ndis"]
    base = base[cols].rename(columns={c: f"ada_{c}" for c in cols if c not in KEY})
    la = df[df.policy != C.BASELINE_POLICY]
    out = la.merge(base, on=KEY, how="left")
    missing = out[out.ada_mean_recall.isna()]
    if not missing.empty:
        keys = missing[KEY].drop_duplicates().to_dict("records")
        print(f"warning: {len(missing)} LA-ef runs have no {C.BASELINE_POLICY} baseline and their "
              f"loss columns are NaN: {keys[:6]}{' ...' if len(keys) > 6 else ''}", flush=True)
    out["recall_loss_pts"] = (out.ada_mean_recall - out.mean_recall) * 100
    out["p5_recall_loss_pts"] = (out.ada_p5_recall - out.p5_recall) * 100
    out["p1_recall_loss_pts"] = (out.ada_p1_recall - out.p1_recall) * 100
    out["p95_ratio_vs_ada"] = out.p95_ms / out.ada_p95_ms
    out["p95_over_slo"] = out.p95_ms / out.slo_ms
    out["meets_slo"] = out.p95_ms <= out.slo_ms
    out["goodput_gain"] = out.goodput_qps - out.ada_goodput_qps
    return out


def allocator_delta(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = KEY + ["slo_fraction"]
    wf = df[df.policy == "laef_wf"].set_index(key)
    un = df[df.policy == "laef_uniform"].set_index(key)
    for name, frame in (("laef_wf", wf), ("laef_uniform", un)):
        if not frame.index.is_unique:
            raise ValueError(f"duplicate {name} runs for the same {key}. "
                             "Two studies were written into one results folder.")
    common = wf.index.intersection(un.index)
    if len(common) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(index=common)
    for col in ["mean_recall", "p5_recall", "p1_recall", "slo_attainment", "p95_ms",
                "goodput_qps", "mean_ef_ratio"]:
        out[f"d_{col}"] = wf.loc[common, col] - un.loc[common, col]
    out["wf_mean_recall"] = wf.loc[common, "mean_recall"]
    out["uniform_mean_recall"] = un.loc[common, "mean_recall"]
    return out.reset_index()


def intervals(study: Study, run_ids=None) -> pd.DataFrame:
    frames = []
    for d in sorted(p.parent for p in study.dir.rglob("intervals.csv")):
        if run_ids and d.name not in run_ids:
            continue
        meta_p = d / "run_meta.json"
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        f = pd.read_csv(d / "intervals.csv")
        for k, v in meta.items():
            f[k] = v
        frames.append(f)
    return _numeric(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def timeseries(run_dir: str | Path) -> pd.DataFrame:
    p = Path(run_dir) / "timeseries.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def export(study: Study, df: pd.DataFrame, pair: pd.DataFrame) -> dict[str, Path]:
    study.dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, frame in (("summary_tidy.csv", df), ("comparison_tidy.csv", pair)):
        if frame is not None and not frame.empty:
            p = study.dir / name
            frame.to_csv(p, index=False)
            out[name] = p
    return out
