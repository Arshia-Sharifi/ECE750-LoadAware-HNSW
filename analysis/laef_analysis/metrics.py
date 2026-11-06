"""LA-ef: metric computation (re-computes from queries.csv)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(queries: pd.DataFrame, t_slo_ms: float, offered_qps: float) -> dict:
    """Re-compute summary metrics from queries.csv."""
    # Filter out warmup
    q = queries[~queries.warmup.astype(bool)] if "warmup" in queries else queries.copy()

    n = len(q)
    if n == 0:
        return {"n_queries": 0}

    # Latency percentiles
    lat = q.latency_us / 1000.0 if "latency_us" in q else pd.Series([0.0])
    lat_sorted = lat.sort_values().values

    p50 = np.percentile(lat_sorted, 50)
    p95 = np.percentile(lat_sorted, 95)
    p99 = np.percentile(lat_sorted, 99)

    # SLO attainment
    within_slo = (lat <= t_slo_ms).sum()

    # Recall
    recall = q.recall if "recall" in q else pd.Series([0.0])
    mean_recall = recall.mean() if not recall.empty else 0.0
    p5_recall = np.percentile(recall.dropna().values, 5) if not recall.dropna().empty else 0.0
    p1_recall = np.percentile(recall.dropna().values, 1) if not recall.dropna().empty else 0.0

    # Throughput: first arrival to last completion of the measured queries (same definition as the C++ summary).
    if "t_arrival_us" in q and "latency_us" in q:
        wall_s = ((q.t_arrival_us + q.latency_us).max() - q.t_arrival_us.min()) / 1e6
    else:
        wall_s = 0.0
    achieved_qps = n / wall_s if wall_s > 0 else 0.0
    goodput_qps = within_slo / wall_s if wall_s > 0 else 0.0

    # Effort
    ef_ratio = (q.ef_used / q.ef_pref.clip(lower=1)) if "ef_used" in q else pd.Series([0.0])
    mean_ef_ratio = ef_ratio.mean()
    mean_ndis = q.ndis.mean() if "ndis" in q else 0.0
    forced_floor_rate = q.forced_floor.mean() if "forced_floor" in q else 0.0
    floor_mode_rate = q.floor_mode.mean() if "floor_mode" in q else 0.0

    return {
        "n_queries": n,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "slo_ms": t_slo_ms,
        "slo_attainment": within_slo / n,
        "latency_excess_ms": max(0.0, p95 - t_slo_ms),
        "mean_recall": mean_recall,
        "p5_recall": p5_recall,
        "p1_recall": p1_recall,
        "offered_qps": offered_qps,
        "achieved_qps": achieved_qps,
        "goodput_qps": goodput_qps,
        "mean_ef_ratio": mean_ef_ratio,
        "mean_ndis": mean_ndis,
        "forced_floor_rate": forced_floor_rate,
        "floor_mode_query_rate": floor_mode_rate,
    }
