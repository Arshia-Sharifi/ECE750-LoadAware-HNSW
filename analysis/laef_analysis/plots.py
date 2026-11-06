"""LA-ef: plotting helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "static_ef": "tab:gray",
    "ada_ef": "tab:blue",
    "laef_uniform": "tab:orange",
    "laef_wf": "tab:green",
}


def plot_recall_vs_rho(summary: pd.DataFrame, out_dir: Path, dataset: str | None = None) -> None:
    """Plot mean and P5 recall vs load for each policy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for (ds, m), d in summary.groupby(["dataset", "slo_mult"]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for pol in sorted(d.policy.unique()):
            s = d[d.policy == pol].sort_values("rho")
            if s.empty:
                continue
            c = COLORS.get(pol, "k")
            ax[0].plot(s.rho, s.mean_recall, "-o", ms=3, color=c, label=pol)
            ax[1].plot(s.rho, s.p5_recall, "-o", ms=3, color=c)

        ax[0].set(xlabel="load ρ", ylabel="mean recall", title=f"{ds}, m={m:g}")
        ax[1].set(xlabel="load ρ", ylabel="P5 recall", title="tail recall")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}_m{m:g}_recall_vs_rho.png", dpi=110)
        plt.close(fig)


def plot_latency_vs_rho(summary: pd.DataFrame, out_dir: Path) -> None:
    """Plot p95 latency / T_SLO vs load."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for (ds, m), d in summary.groupby(["dataset", "slo_mult"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        for pol in sorted(d.policy.unique()):
            s = d[d.policy == pol].sort_values("rho")
            if s.empty:
                continue
            c = COLORS.get(pol, "k")
            ax.plot(s.rho, s.p95_ms / s.slo_ms, "-o", ms=3, color=c, label=pol)

        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set(xlabel="load ρ", ylabel="p95 / T_SLO", yscale="log", title=f"{ds}, m={m:g}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}_m{m:g}_latency_vs_rho.png", dpi=110)
        plt.close(fig)


def plot_timeseries(run_dir: Path, out_dir: Path) -> None:
    """Plot B(t), p95(t), Q(t), floor_mode(t) for one run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts_path = run_dir / "timeseries.csv"
    if not ts_path.exists():
        return
    ts = pd.read_csv(ts_path)

    fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    ax[0].plot(ts.t_s, ts.B / 1e6)
    ax[0].plot(ts.t_s, ts.B_max / 1e6, ":", lw=0.8)
    ax[0].plot(ts.t_s, ts.B_min / 1e6, "--", lw=0.8)
    ax[0].set(ylabel="B [M dc/s]")

    ax[1].plot(ts.t_s, ts.p95_ms)
    ax[1].set(ylabel="p95 [ms]", yscale="log")

    ax[2].plot(ts.t_s, ts.Q)
    ax[2].set(ylabel="queue depth Q")

    ax[3].step(ts.t_s, ts.floor_mode.astype(int), where="post")
    ax[3].set(ylabel="floor mode", xlabel="time [s]")

    fig.suptitle(run_dir.name, y=1.0)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_dir.name}_timeseries.png", dpi=110)
    plt.close(fig)
