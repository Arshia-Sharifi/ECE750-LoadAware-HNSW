#!/usr/bin/env python
"""LA-ef live statistics over time (IMPLEMENTATION_PLAN.md §11.3, §11.4).

    python scripts/interval_stats.py results/<experiment> [--interval 1.0]

For every run folder in results/<experiment>/ that has run_meta.json, it reads queries.csv (one row per query),
timeseries.csv (one row per controller window), config.json and run_config.json, and writes:

  <run>/intervals.csv     one row per fixed time interval over the whole run: arrivals and completions,
                          p50 and p95 latency of the completions, SLO attainment, requests in the system at
                          the end of the interval, ef ratio, recall, floor-mode fractions, and the controller
                          state (budget B, debt D, zone, guard fires) averaged over the windows in the interval.
  dynamics_phases.csv     one row per run and workload phase (phases come from run_config.json).
  dynamics_runs.csv       one row per run: peak interval p95, peak requests in the system, and the time for the
                          p95 to fall back under the SLO after the last load drop (step profile only).

Warm-up queries are included, so the start-up transient is visible. summary.csv keeps the final metrics.
Queries are read as numbers with numpy, so runs with millions of queries stay within memory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

VARIANT = {
    "static_ef": "static ef (no controller)",
    "ada_ef": "Ada-ef (no controller)",
    "laef_uniform": "LA-ef naive cut (uniform_cut)",
    "laef_wf": "LA-ef proportional cut (water_filling)",
}
NAN = math.nan
# queries.csv columns (0-based) that are read: t_arrival_us, latency_us, ef_pref, ef_used, forced_floor,
# floor_mode, ndis, recall
QUERY_COLS = (3, 6, 10, 11, 12, 13, 14, 17)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else NAN


def mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else NAN


def nanmax(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return max(vals) if vals else NAN


def phases_of(run_cfg: dict) -> list[dict]:
    wl = run_cfg.get("workload", {})
    if wl.get("profile") == "step" and wl.get("phases"):
        out, start = [], 0.0
        for i, ph in enumerate(wl["phases"]):
            end = start + float(ph["duration_s"])
            out.append({"phase": i, "start": start, "end": end, "qps": float(ph["qps"])})
            start = end
        return out
    return [{"phase": 0, "start": 0.0, "end": math.inf, "qps": float(wl.get("qps", NAN))}]


def phase_index(t: float, phases: list[dict]) -> int:
    for p in phases:
        if p["start"] <= t < p["end"]:
            return p["phase"]
    return phases[-1]["phase"]


def recovery_s(intervals: list[dict], t_drop: float, t_slo_ms: float) -> float:
    """Seconds from t_drop until two consecutive intervals have a p95 at or below the SLO."""
    after = [iv for iv in intervals if iv["t_start_s"] >= t_drop - 1e-9]
    for a, b in zip(after, after[1:]):
        pa, pb = a["p95_ms"], b["p95_ms"]
        if not (math.isnan(pa) or math.isnan(pb)) and pa <= t_slo_ms and pb <= t_slo_ms:
            return a["t_start_s"] - t_drop
    return NAN


def load_queries(path: Path) -> dict[str, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=QUERY_COLS, ndmin=2)
    arr_us, lat_us, ef_pref, ef_used, forced, floor, ndis, recall = data.T
    return {"arr": arr_us * 1e-6, "lat_ms": lat_us * 1e-3, "ef_pref": ef_pref, "ef_used": ef_used,
            "forced": forced, "floor": floor, "ndis": ndis, "recall": recall}


def analyse_run(run_dir: Path, interval: float) -> tuple[list[dict], list[dict], dict]:
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    t_slo_ms = float(cfg["t_slo_s"]) * 1e3
    phases = phases_of(run_cfg)

    q = load_queries(run_dir / "queries.csv")
    arr, lat_ms = q["arr"], q["lat_ms"]
    end = arr + lat_ms * 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        ef_ratio = np.where(q["ef_pref"] > 0, q["ef_used"] / np.maximum(q["ef_pref"], 1.0), np.nan)

    # Sorted copies: arrival times for counting, completions (sorted by end time) for the per-interval figures.
    arr_sorted = np.sort(arr)
    order = np.argsort(end, kind="stable")
    end_sorted = end[order]
    lat_e, efr_e, rec_e = lat_ms[order], ef_ratio[order], q["recall"][order]
    ndis_e, forced_e, floor_e = q["ndis"][order], q["forced"][order], q["floor"][order]

    ts = read_rows(run_dir / "timeseries.csv")
    ts_t = np.array([float(r["t_s"]) for r in ts])
    ts_B = np.array([float(r["B"]) for r in ts])
    ts_D = np.array([float(r["D"]) for r in ts])
    ts_zone = [r["zone"] for r in ts]
    ts_guard = np.array([float(r["guard_fired"]) for r in ts])
    ts_floor = np.array([float(r["floor_mode"]) for r in ts])

    horizon = max(float(end_sorted[-1]) if len(end_sorted) else 0.0, float(ts_t.max()) if len(ts_t) else 0.0)
    edges = np.arange(0.0, horizon + interval, interval)

    intervals = []
    for i in range(len(edges) - 1):
        t0, t1 = float(edges[i]), float(edges[i + 1])
        a_lo, a_hi = np.searchsorted(arr_sorted, [t0, t1])
        c_lo, c_hi = np.searchsorted(end_sorted, [t0, t1])
        lc = lat_e[c_lo:c_hi]
        m = (ts_t >= t0) & (ts_t < t1)
        intervals.append({
            "t_start_s": round(t0, 4), "t_end_s": round(t1, 4), "phase": phase_index(t0, phases),
            "arrivals": int(a_hi - a_lo), "completions": int(c_hi - c_lo),
            "offered_qps": round((a_hi - a_lo) / interval, 1), "served_qps": round((c_hi - c_lo) / interval, 1),
            "p50_ms": pct(lc, 50), "p95_ms": pct(lc, 95),
            "slo_attain": mean(lc <= t_slo_ms) if len(lc) else NAN,
            "in_system_end": int(np.searchsorted(arr_sorted, t1) - np.searchsorted(end_sorted, t1)),
            "ef_ratio_mean": float(np.nanmean(efr_e[c_lo:c_hi])) if len(lc) else NAN,
            "recall_mean": mean(rec_e[c_lo:c_hi]), "ndis_mean": mean(ndis_e[c_lo:c_hi]),
            "forced_floor_frac": mean(forced_e[c_lo:c_hi]), "floor_mode_frac": mean(floor_e[c_lo:c_hi]),
            "ctrl_windows": int(m.sum()), "B_mean": mean(ts_B[m]), "D_mean": mean(ts_D[m]),
            "zone_last": ts_zone[int(np.where(m)[0][-1])] if m.any() else "",
            "guard_fires": int(ts_guard[m].sum()), "ctrl_floor_frac": mean(ts_floor[m]),
        })

    phase_rows = []
    for p in phases:
        end_eff = min(p["end"], horizon)
        am = (arr >= p["start"]) & (arr < p["end"])
        cm = (end >= p["start"]) & (end < p["end"])
        wm = (ts_t >= p["start"]) & (ts_t < p["end"])
        in_phase = [iv for iv in intervals if iv["phase"] == p["phase"]]
        lat_a = lat_ms[am]
        phase_rows.append({
            **meta_fields(meta),
            "phase": p["phase"], "phase_start_s": round(p["start"], 3),
            "phase_end_s": round(end_eff, 3), "phase_qps": round(p["qps"], 1),
            "arrivals": int(am.sum()), "served_qps": round(cm.sum() / max(end_eff - p["start"], 1e-9), 1),
            "p95_ms": pct(lat_a, 95), "mean_ms": mean(lat_a),
            "slo_attain": mean(lat_a <= t_slo_ms) if am.any() else NAN,
            "ef_ratio_mean": float(np.nanmean(ef_ratio[am])) if am.any() else NAN,
            "recall_mean": mean(q["recall"][am]), "forced_floor_frac": mean(q["forced"][am]),
            "floor_mode_frac": mean(q["floor"][am]),
            "max_in_system": nanmax([float(iv["in_system_end"]) for iv in in_phase]),
            "peak_interval_p95_ms": nanmax([iv["p95_ms"] for iv in in_phase]),
            "ctrl_windows": int(wm.sum()), "B_mean": mean(ts_B[wm]), "D_mean": mean(ts_D[wm]),
            "violation_window_frac": float(np.mean([z == "violation" for z, w in zip(ts_zone, wm) if w]))
            if wm.any() else NAN,
        })

    drops = [p for prev, p in zip(phases, phases[1:]) if p["qps"] < prev["qps"]]
    drop_start = drops[-1]["start"] if drops else NAN
    run_row = {
        **meta_fields(meta),
        "t_slo_ms": round(t_slo_ms, 4),
        "peak_interval_p95_ms": nanmax([iv["p95_ms"] for iv in intervals]),
        "max_in_system": nanmax([float(iv["in_system_end"]) for iv in intervals]),
        "last_load_drop_s": drop_start,
        "recovery_s": recovery_s(intervals, drop_start, t_slo_ms) if drops else NAN,
    }
    return intervals, phase_rows, run_row


def meta_fields(meta: dict) -> dict:
    return {
        "run_id": meta["run_id"], "dataset": meta["dataset"], "policy": meta["policy"],
        "variant": VARIANT.get(meta["policy"], meta["policy"]),
        "workload": meta["profile"], "rho_label": meta["rho"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LA-ef live statistics per fixed time interval")
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--interval", type=float, default=1.0, help="interval length in seconds")
    args = ap.parse_args(argv)

    run_dirs = sorted(p for p in args.experiment_dir.iterdir() if (p / "run_meta.json").exists())
    phase_rows, run_rows = [], []
    for run_dir in run_dirs:
        intervals, phases, run_row = analyse_run(run_dir, args.interval)
        write_csv(run_dir / "intervals.csv", intervals)
        phase_rows += phases
        run_rows.append(run_row)
    write_csv(args.experiment_dir / "dynamics_phases.csv", phase_rows)
    write_csv(args.experiment_dir / "dynamics_runs.csv", run_rows)
    print(f"{len(run_dirs)} runs: intervals.csv per run, dynamics_phases.csv, dynamics_runs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
