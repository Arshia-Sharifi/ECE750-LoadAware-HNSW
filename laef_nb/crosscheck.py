from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CHECKED = ["p50_ms", "p95_ms", "p99_ms", "slo_attainment", "mean_recall", "p5_recall",
           "p1_recall", "achieved_qps", "goodput_qps", "mean_ef_ratio", "mean_ndis"]
# queries.csv stores latency truncated to whole microseconds, so a query sitting within 1 us of
# T_SLO can land on either side of the threshold here but not in the engine.
BOUNDARY = ("slo_attainment", "goodput_qps")


def from_queries(q: pd.DataFrame, slo_ms: float) -> dict:
    q = q[~q.warmup.astype(bool)] if "warmup" in q else q
    if q.empty:
        return {}
    lat = q.latency_us / 1e3
    rec = q.recall.dropna()
    wall = ((q.t_arrival_us + q.latency_us).max() - q.t_arrival_us.min()) / 1e6
    within = int((lat <= slo_ms).sum())
    return {
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "slo_attainment": within / len(q),
        "mean_recall": float(rec.mean()),
        "p5_recall": float(np.percentile(rec, 5)),
        "p1_recall": float(np.percentile(rec, 1)),
        "achieved_qps": len(q) / wall if wall > 0 else np.nan,
        "goodput_qps": within / wall if wall > 0 else np.nan,
        "mean_ef_ratio": float((q.ef_used / q.ef_pref.clip(lower=1)).mean()),
        "mean_ndis": float(q.ndis.mean()),
    }


def check_run(run_dir: str | Path, tol: float = 0.02) -> pd.DataFrame:
    """Recompute the summary from queries.csv. Needs a study launched with keep_queries."""
    run_dir = Path(run_dir)
    queries = run_dir / "queries.csv"
    if not queries.exists():
        return pd.DataFrame()
    reported = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    q = pd.read_csv(queries)
    slo_ms = float(reported["slo_ms"])
    mine = from_queries(q, slo_ms)
    kept = q[~q.warmup.astype(bool)] if "warmup" in q else q
    lat = kept.latency_us / 1e3
    n = max(len(kept), 1)
    boundary_frac = float(((lat - slo_ms).abs() <= 1e-3).sum()) / n
    allow = {"slo_attainment": boundary_frac,
             "goodput_qps": boundary_frac * mine.get("achieved_qps", 0.0)}
    rows = []
    for k in CHECKED:
        a, b = reported.get(k), mine.get(k)
        if a is None or b is None:
            continue
        slack = allow.get(k, 0.0)
        rows.append({"metric": k, "summary_json": a, "recomputed": b,
                     "rel_diff": abs(a - b) / max(abs(a), 1e-12),
                     "boundary_slack": slack,
                     "ok": abs(a - b) <= tol * abs(a) + slack})
    return pd.DataFrame(rows)


def check_study(study, tol: float = 0.02) -> pd.DataFrame:
    frames = []
    for d in sorted(p.parent for p in study.dir.rglob("queries.csv")):
        f = check_run(d, tol)
        if not f.empty:
            f["run_id"] = d.name
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
