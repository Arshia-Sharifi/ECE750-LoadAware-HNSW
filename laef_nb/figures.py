from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

POLICY_COLOR = {"ada_ef": "#1f77b4", "laef_uniform": "#ff7f0e", "laef_wf": "#2ca02c",
                "static_ef": "#7f7f7f"}
POLICY_MARKER = {"ada_ef": "o", "laef_uniform": "s", "laef_wf": "^", "static_ef": "x"}
THREAD_CMAP = plt.get_cmap("viridis")

STYLE = {
    "figure.dpi": 120, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.6, "lines.markersize": 4.5,
}


def use_style() -> None:
    mpl.rcParams.update(STYLE)


def slug(s) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))


def outdir(study) -> Path:
    d = study.dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, study, name: str) -> Path:
    p = outdir(study) / f"{name}.png"
    fig.savefig(p)
    return p


def _grid(n: int, ncols: int = 3, w: float = 3.6, h: float = 2.9, sharey=True, sharex=True):
    n = max(1, int(n))
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                             sharey=sharey, sharex=sharex, squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def _log2_axis(ax, threads):
    ts = sorted({int(t) for t in threads})
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ts))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))


def _thread_color(t, threads):
    ts = sorted(set(threads))
    return THREAD_CMAP(ts.index(t) / max(1, len(ts) - 1)) if len(ts) > 1 else THREAD_CMAP(0.5)


_FRACTION_NOTED: set = set()


def pick_fraction(df: pd.DataFrame, slo_fraction: float | None = None) -> tuple[pd.DataFrame, float | None]:
    """Keep one SLO fraction. Averaging over fractions mixes different deadlines on one line."""
    if "slo_fraction" not in df:
        return df, None
    la = df[df.policy != C.BASELINE_POLICY]
    vals = sorted(la.slo_fraction.dropna().unique())
    if len(vals) <= 1:
        return df, (vals[0] if vals else None)
    if slo_fraction is None:
        slo_fraction = max(vals)
        key = tuple(vals)
        if key not in _FRACTION_NOTED:
            _FRACTION_NOTED.add(key)
            shown = ", ".join(f"{v:g}" for v in vals)
            print(f"note: {len(vals)} SLO fractions present [{shown}]; figures show {slo_fraction:g}. "
                  "Pass slo_fraction= to choose another.", flush=True)
    keep = (df.policy == C.BASELINE_POLICY) | np.isclose(df.slo_fraction, slo_fraction)
    return df[keep], slo_fraction


def _suffix(frac: float | None) -> str:
    return "" if frac is None else f", T_SLO = {frac:g} x profile p95"


def _policy_lines(ax, d: pd.DataFrame, x: str, y: str):
    for pol, s in d.groupby("policy"):
        s = s.groupby(x, as_index=False)[y].mean().sort_values(x)
        ax.plot(s[x], s[y], marker=POLICY_MARKER.get(pol, "o"),
                color=POLICY_COLOR.get(pol, "k"), label=C.POLICY_LABEL.get(pol, pol))


def capacity_scaling(prof: pd.DataFrame, study=None):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for ds, d in prof.groupby("dataset"):
        d = d.sort_values("threads")
        ax[0].plot(d.threads, d.mu_sat_qps, "-o", label=ds)
        ideal = d.mu_sat_qps.iloc[0] * d.threads / d.threads.iloc[0]
        ax[0].plot(d.threads, ideal, ":", color="0.6", lw=1)
        ax[1].plot(d.threads, d.efficiency, "-o", label=ds)
        ax[2].plot(d.threads, d.s_p95_ms, "-o", label=ds)
    ax[1].axhline(1.0, color="0.4", ls="--", lw=1)
    ax[0].set(xlabel="worker threads C", ylabel="mu_sat [queries/s]", title="measured capacity")
    ax[1].set(xlabel="worker threads C", ylabel="speedup / ideal", title="parallel efficiency")
    ax[2].set(xlabel="worker threads C", ylabel="saturation p95 service [ms]",
              title="service time at saturation")
    for a in ax:
        _log2_axis(a, prof.threads)
    ax[0].legend()
    fig.tight_layout()
    return (fig, _save(fig, study, "01_capacity_scaling")) if study else (fig, None)


def nearest_rho(df: pd.DataFrame, target: float = 1.0) -> float:
    vals = sorted(df.rho.dropna().unique())
    return min(vals, key=lambda r: abs(r - target)) if vals else target


def throughput_vs_threads(df: pd.DataFrame, study=None, rho: float | None = None,
                          slo_fraction: float | None = None):
    df, frac = pick_fraction(df, slo_fraction)
    rho = nearest_rho(df, 1.0) if rho is None else rho
    d = df[np.isclose(df.rho, rho)]
    if d.empty:
        d, rho = df, None
    datasets = sorted(d.dataset.unique())
    fig, axes = _grid(len(datasets), sharey=False)
    for ax, ds in zip(axes, datasets):
        sub = d[d.dataset == ds]
        _policy_lines(ax, sub, "threads", "achieved_qps")
        ax.set(xlabel="worker threads C", ylabel="achieved [queries/s]", title=ds)
        _log2_axis(ax, d.threads)
    axes[0].legend()
    fig.suptitle(f"served throughput against worker count, rho = "
                 f"{rho if rho is not None else 'all'}{_suffix(frac)}", y=1.02)
    fig.tight_layout()
    return (fig, _save(fig, study, "02_throughput_vs_threads")) if study else (fig, None)


def latency_vs_load(df: pd.DataFrame, study=None, slo_fraction: float | None = None):
    df, frac = pick_fraction(df, slo_fraction)
    keys = sorted(df.groupby(["dataset", "threads"]).groups)
    fig, axes = _grid(len(keys), ncols=len(df.threads.unique()) or 3, sharey=True)
    for ax, (ds, t) in zip(axes, keys):
        sub = df[(df.dataset == ds) & (df.threads == t)]
        _policy_lines(ax, sub, "rho", "p95_over_slo" if "p95_over_slo" in sub else "p95_ms")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set(xlabel="offered load rho", ylabel="p95 / T_SLO", title=f"{ds}, C={t}", yscale="log")
    axes[0].legend()
    fig.suptitle(f"tail latency against offered load{_suffix(frac)}", y=1.02)
    fig.tight_layout()
    return (fig, _save(fig, study, "03_latency_vs_load")) if study else (fig, None)


def recall_vs_load(df: pd.DataFrame, study=None, metric: str = "mean_recall",
                   slo_fraction: float | None = None):
    df, frac = pick_fraction(df, slo_fraction)
    keys = sorted(df.groupby(["dataset", "threads"]).groups)
    fig, axes = _grid(len(keys), ncols=len(df.threads.unique()) or 3, sharey=True)
    for ax, (ds, t) in zip(axes, keys):
        sub = df[(df.dataset == ds) & (df.threads == t)]
        _policy_lines(ax, sub, "rho", metric)
        ax.set(xlabel="offered load rho", ylabel=metric.replace("_", " "), title=f"{ds}, C={t}")
    axes[0].legend()
    fig.suptitle(f"{metric.replace('_', ' ')} against offered load{_suffix(frac)}", y=1.02)
    fig.tight_layout()
    return (fig, _save(fig, study, f"04_{metric}_vs_load")) if study else (fig, None)


def recall_vs_threads(df: pd.DataFrame, study=None, slo_fraction: float | None = None):
    df, frac = pick_fraction(df, slo_fraction)
    keys = sorted(df.groupby(["dataset"]).groups)
    rhos = sorted(df.rho.unique())
    fig, axes = _grid(len(keys) * len(rhos), ncols=len(rhos), sharey=False)
    i = 0
    for ds in keys:
        for rho in rhos:
            ax = axes[i]
            i += 1
            sub = df[(df.dataset == ds) & (np.isclose(df.rho, rho))]
            _policy_lines(ax, sub, "threads", "mean_recall")
            ax.set(xlabel="worker threads C", ylabel="mean recall", title=f"{ds}, rho={rho:g}")
            _log2_axis(ax, df.threads)
    axes[0].legend()
    fig.suptitle(f"recall against worker count at matched relative load{_suffix(frac)}", y=1.02)
    fig.tight_layout()
    return (fig, _save(fig, study, "05_recall_vs_threads")) if study else (fig, None)


def _heat(ax, piv, cmap, norm, fmt="{:.2f}"):
    im = ax.imshow(piv.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"C{r}" for r in piv.index])
    ax.grid(False)
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            v = piv.values[r, c]
            if np.isnan(v):
                continue
            rgb = im.cmap(im.norm(v))[:3]
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=7,
                    color="w" if lum < 0.45 else "k")
    return im


def _shared_norm(pivots, vcenter=None):
    vals = np.concatenate([p.values.ravel() for p in pivots]) if pivots else np.array([np.nan])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    lo, hi = float(vals.min()), float(vals.max())
    if vcenter is None:
        return mpl.colors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
    lim = max(abs(hi - vcenter), abs(lo - vcenter)) or 1.0
    return mpl.colors.TwoSlopeNorm(vmin=vcenter - lim, vcenter=vcenter, vmax=vcenter + lim)


def _policies_in(d: pd.DataFrame) -> list[str]:
    return sorted(d.policy.dropna().unique(), key=lambda p: (p != C.BASELINE_POLICY, p))


def _panel_heatmap(d, value, cmap, vcenter, label, title, fig_name, study):
    pols = _policies_in(d)
    pivots = [d[d.policy == p].pivot_table(index="threads", columns="rho", values=value, aggfunc="mean")
              for p in pols]
    keep = [(p, v) for p, v in zip(pols, pivots) if not v.empty]
    if not keep:
        return None, None
    pols, pivots = [p for p, _ in keep], [v for _, v in keep]
    norm = _shared_norm(pivots, vcenter=vcenter)
    fig, axes = _grid(len(pols), ncols=len(pols), w=3.4, h=2.9, sharey=False, sharex=False)
    for ax, pol, piv in zip(axes, pols, pivots):
        im = _heat(ax, piv, cmap, norm)
        ax.set(title=C.POLICY_LABEL.get(pol, pol), xlabel="rho")
    fig.colorbar(im, ax=list(axes[:len(pols)]), shrink=0.85, label=label)
    fig.suptitle(title, y=1.03)
    return (fig, _save(fig, study, fig_name)) if study else (fig, None)


def slo_heatmap(df: pd.DataFrame, study=None, dataset: str | None = None,
                slo_fraction: float | None = None):
    df, frac = pick_fraction(df, slo_fraction)
    d = df if dataset is None else df[df.dataset == dataset]
    name = f"06_slo_heatmap_{slug(dataset)}" if dataset else "06_slo_heatmap"
    return _panel_heatmap(d, "slo_attainment", "RdYlGn", None, "SLO attainment",
                          f"SLO attainment, {dataset or 'all datasets'}{_suffix(frac)}", name, study)


def recall_loss_heatmap(pair: pd.DataFrame, study=None, dataset: str | None = None,
                        value: str = "recall_loss_pts", slo_fraction: float | None = None):
    pair, frac = pick_fraction(pair, slo_fraction)
    d = pair if dataset is None else pair[pair.dataset == dataset]
    name = f"07_{value}_heatmap_{slug(dataset)}" if dataset else f"07_{value}_heatmap"
    return _panel_heatmap(d, value, "RdBu_r", 0.0, f"{value} vs stock Ada-ef",
                          f"recall given up against stock Ada-ef, "
                          f"{dataset or 'all datasets'}{_suffix(frac)}", name, study)


def allocator_gap(delta: pd.DataFrame, study=None, value: str = "d_mean_recall",
                  slo_fraction: float | None = None):
    if "slo_fraction" in delta and delta.slo_fraction.nunique() > 1:
        frac = slo_fraction if slo_fraction is not None else delta.slo_fraction.max()
        delta = delta[np.isclose(delta.slo_fraction, frac)]
    else:
        frac = delta.slo_fraction.iloc[0] if "slo_fraction" in delta and len(delta) else None
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    threads = sorted(delta.threads.unique())
    rhos = sorted(delta.rho.unique())
    width = 0.8 / max(1, len(threads))
    for i, t in enumerate(threads):
        d = delta[delta.threads == t].groupby("rho", as_index=False)[value].mean()
        x = np.arange(len(rhos)) + i * width - 0.4 + width / 2
        y = [float(d.loc[np.isclose(d.rho, r), value].mean()) if np.isclose(d.rho, r).any() else np.nan
             for r in rhos]
        ax.bar(x, y, width=width * 0.92, color=_thread_color(t, threads), label=f"C={t}")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(rhos)), [f"{r:g}" for r in rhos])
    ax.set(xlabel="offered load rho", ylabel=value.replace("d_", "delta "),
           title=f"proportional cut minus naive cut{_suffix(frac)}")
    ax.legend(ncol=max(1, len(threads)))
    fig.tight_layout()
    return (fig, _save(fig, study, "08_allocator_gap")) if study else (fig, None)


def operating_points(df: pd.DataFrame, study=None):
    datasets = sorted(df.dataset.unique())
    fig, axes = _grid(len(datasets), sharey=False, sharex=False)
    for ax, ds in zip(axes, datasets):
        sub = df[df.dataset == ds]
        for pol, s in sub.groupby("policy"):
            ax.scatter(s.p95_ms / s.slo_ms, s.mean_recall, s=18 + 4 * np.log2(s.threads.clip(lower=1)),
                       color=POLICY_COLOR.get(pol, "k"), marker=POLICY_MARKER.get(pol, "o"),
                       alpha=0.85, label=C.POLICY_LABEL.get(pol, pol))
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set(xlabel="p95 / T_SLO", ylabel="mean recall", title=ds, xscale="log")
    axes[0].legend()
    fig.suptitle("operating points, marker size grows with the worker count", y=1.02)
    fig.tight_layout()
    return (fig, _save(fig, study, "09_operating_points")) if study else (fig, None)


def generator_health(df: pd.DataFrame, study=None, limit_us: float = 500.0):
    """Validity check. A large generator lag means the arrival process, not the server, is the bottleneck."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    threads = sorted(df.threads.unique())
    for t in threads:
        s = df[df.threads == t]
        ax[0].scatter(s.offered_qps, s.generator_lag_p99_us, s=16,
                      color=_thread_color(t, threads), label=f"C={t}")
        ax[1].scatter(s.offered_qps, s.achieved_qps / s.offered_qps, s=16,
                      color=_thread_color(t, threads))
    ax[0].axhline(limit_us, color="r", ls="--", lw=1)
    ax[0].set(xlabel="offered [queries/s]", ylabel="generator lag p99 [us]",
              yscale="log", title="arrival generator lag")
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set(xlabel="offered [queries/s]", ylabel="achieved / offered", title="rate tracking")
    ax[0].legend(ncol=2)
    fig.tight_layout()
    return (fig, _save(fig, study, "10_generator_health")) if study else (fig, None)


def run_timeseries(iv: pd.DataFrame, run_id: str, study=None):
    d = iv[iv.run_id == run_id].sort_values("t_start_s")
    fig, ax = plt.subplots(5, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(d.t_start_s, d.offered_qps, label="offered")
    ax[0].plot(d.t_start_s, d.served_qps, label="served")
    ax[0].set(ylabel="queries/s")
    ax[0].legend(ncol=2)
    ax[1].plot(d.t_start_s, d.p95_ms, color="#d62728")
    ax[1].set(ylabel="p95 [ms]", yscale="log")
    ax[2].plot(d.t_start_s, d.recall_mean, color="#2ca02c")
    ax[2].set(ylabel="mean recall")
    ax[3].plot(d.t_start_s, d.ef_ratio_mean, color="#9467bd")
    ax[3].set(ylabel="ef used / preferred")
    ax[4].step(d.t_start_s, d.floor_mode_frac, where="post", color="#8c564b")
    ax[4].set(ylabel="floor mode fraction", xlabel="time [s]")
    fig.suptitle(run_id, y=1.0, fontsize=9)
    fig.tight_layout()
    return (fig, _save(fig, study, f"11_timeseries_{run_id[:60]}")) if study else (fig, None)


def all_figures(prof, df, pair, delta, study, iv=None, trace_run=None,
                slo_fraction: float | None = None) -> list[Path]:
    use_style()
    paths = []
    if prof is not None and not prof.empty:
        paths.append(capacity_scaling(prof, study)[1])
    if df is not None and not df.empty:
        paths += [throughput_vs_threads(df, study, slo_fraction=slo_fraction)[1],
                  latency_vs_load(df.assign(p95_over_slo=df.p95_ms / df.slo_ms), study,
                                  slo_fraction=slo_fraction)[1],
                  recall_vs_load(df, study, "mean_recall", slo_fraction=slo_fraction)[1],
                  recall_vs_load(df, study, "p5_recall", slo_fraction=slo_fraction)[1],
                  recall_vs_threads(df, study, slo_fraction=slo_fraction)[1],
                  operating_points(df, study)[1],
                  generator_health(df, study)[1]]
        paths += [slo_heatmap(df, study, dataset=ds, slo_fraction=slo_fraction)[1]
                  for ds in sorted(df.dataset.unique())]
    if pair is not None and not pair.empty:
        paths += [recall_loss_heatmap(pair, study, dataset=ds, slo_fraction=slo_fraction)[1]
                  for ds in sorted(pair.dataset.unique())]
    if delta is not None and not delta.empty:
        paths.append(allocator_gap(delta, study, slo_fraction=slo_fraction)[1])
    if iv is not None and not iv.empty and trace_run:
        paths.append(run_timeseries(iv, trace_run, study)[1])
    plt.close("all")
    return [p for p in paths if p]
