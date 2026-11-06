"""LA-ef: I/O helpers for analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_run(run_dir: str | Path) -> dict:
    """Load all files from a run directory."""
    run_dir = Path(run_dir)
    result = {"dir": run_dir, "id": run_dir.name}

    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        result["config"] = json.loads(cfg_path.read_text())

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        result["summary"] = json.loads(summary_path.read_text())

    queries_path = run_dir / "queries.csv"
    if queries_path.exists():
        result["queries"] = pd.read_csv(queries_path)

    ts_path = run_dir / "timeseries.csv"
    if ts_path.exists():
        result["timeseries"] = pd.read_csv(ts_path)

    return result


def load_summary_csv(exp_dir: str | Path) -> pd.DataFrame:
    """Load the aggregated summary.csv for an experiment."""
    summary_path = Path(exp_dir) / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found")
    return pd.read_csv(summary_path)


def list_runs(exp_dir: str | Path) -> list[Path]:
    """List all run directories in an experiment."""
    exp_dir = Path(exp_dir)
    return sorted(p for p in exp_dir.iterdir() if (p / "summary.json").exists())
