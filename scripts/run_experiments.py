#!/usr/bin/env python
"""LA-ef experiment runner (IMPLEMENTATION_PLAN.md §12 P6.4, §14.1, §11.4).

    python scripts/run_experiments.py configs/experiments/slo_study_120s.yaml [--force] [--dry-run]

Expands the YAML grid into laef_bench runs, runs the ones that have no summary.json yet, and writes
results/<experiment>/summary.csv with one row per run (the §11.4 columns). Every top-level section
that has a 'datasets' key is an experiment. Its output folder is the section's 'name', or its key
(for example slo_study_120s/ada_synthetic, which gives results/slo_study_120s/ada_synthetic/). Each run needs data/artifacts/<dataset>/ with the outputs of
laef_build_index, laef_calibrate and laef_profile at the run's thread count.

The ladder axis (calibration.ladder_ratio) runs laef_calibrate once per ratio, writes the probe menu and its per-length
tables to results/<experiment>/_tables/<dataset>/ladder_<ratio>/, and passes that folder to laef_bench with --tables-dir.
The default ratio uses data/artifacts/<dataset>/tables/.

SLO: the engine takes T_SLO in ms only. `slo: {ms: X}` is passed as X.
`slo: {fraction: [f, ...]}` sets T_SLO = f x the dataset's saturation p95: `saturation.s_p95_s` in
data/artifacts/<dataset>/profile/profile_C{threads}.json, the p95 service time of stock Ada-ef at mu_sat. The runner
turns it into ms before the engine starts (the same way rho becomes qps). That p95 is a closed-loop service time: it
has no queue wait in it.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]  # the Implementation folder
DEFAULT_BENCH = ROOT / "build" / "bin" / "laef_bench"
# calibration.ladder_ratio changes only the price list. The runner builds one price list per ratio.
# The other calibration axes change the samplings, which the runner does not regenerate.
LADDER_AXIS = "calibration.ladder_ratio"
DEFAULT_LADDER_RATIO = 2.0  # configs/defaults.yaml; those runs use <dataset>/tables/
RECALIBRATION_AXES = ("calibration.",)
_built_tables: set[Path] = set()  # ladder tables built during this invocation
META_COLUMNS = ["run_id", "experiment", "dataset", "policy", "threads", "slo_fraction", "profile_p95_ms", "profile", "rho",
                "repeat", "axis", "axis_value"]


def as_list(v: Any) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def slug(v: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(v))


def dot(x: float) -> str:
    """A number as a run-id label: 0.95 -> 0p95, 1.5 -> 1p5."""
    return f"{x:g}".replace(".", "p")


def set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


@dataclass
class Run:
    experiment: str
    run_id: str
    dataset: str
    policy: str
    threads: int
    profile: str
    rho: float
    repeat: int
    axis: str
    axis_value: str
    config: dict[str, Any] = field(default_factory=dict)
    slo_fraction: float | None = None  # slo.fraction sections: T_SLO = slo_fraction x the profile p95
    profile_p95_ms: float | None = None  # that profile p95 [ms]; set by run_bench

    def meta(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "experiment": self.experiment, "dataset": self.dataset,
                "policy": self.policy, "threads": self.threads,
                "profile": self.profile, "rho": self.rho, "repeat": self.repeat,
                "axis": self.axis, "axis_value": self.axis_value, "slo_fraction": self.slo_fraction,
                "profile_p95_ms": self.profile_p95_ms}


def workload_variants(wl: dict) -> list[tuple[dict, str, float]]:
    """Expand one 'workloads' entry into (bench workload config, profile name, rho label) tuples."""
    profile = wl.get("profile", "poisson")
    passthrough = [k for k in ("n_queries", "duration_s", "warmup_queries") if k in wl]
    if profile in ("poisson", "constant"):  # same parameters; constant has evenly spaced arrivals
        out = []
        for r in as_list(wl.get("rho", 0.8)):
            cfg = {"profile": profile, "rho": float(r), **{k: wl[k] for k in passthrough}}
            out.append((cfg, profile, float(r)))
        return out
    if profile == "step":
        phases = [{"rho": float(p["rho"]), "duration_s": float(p["duration_s"])} for p in wl["phases"]]
        return [({"profile": "step", "phases": phases}, "step", max(p["rho"] for p in phases))]
    if profile == "mmpp2":
        cfg = {"profile": "mmpp2", **{k: float(wl[k]) for k in ("rho_lo", "rho_hi", "mean_lo_s", "mean_hi_s")},
               **{k: wl[k] for k in passthrough}}
        return [(cfg, "mmpp2", float(wl["rho_hi"]))]
    raise ValueError(f"unknown workload profile {profile!r}")


def build_runs(section: str, sec: dict) -> list[Run]:
    out_name = sec.get("name", section)
    datasets = as_list(sec["datasets"])
    if "policies" in sec:
        policies = as_list(sec["policies"])
    elif "policy" in sec:
        policies = [sec["policy"]]
    else:
        policies = [sec.get("base_policy", "laef_wf")]
    threads = [int(t) for t in as_list(sec.get("threads", 4))]
    vary_threads = len(threads) > 1 and "axes" not in sec  # E5: the thread count is the axis

    workloads = sec.get("workloads", [{"profile": "poisson", "rho": 0.8, "n_queries": 20000}])
    wl_points = [point for wl in workloads for point in workload_variants(wl)]
    slo = sec.get("slo") or {}
    if "fraction" in slo:  # T_SLO = fraction x the dataset's profile p95, resolved in run_bench
        slo_points = [({"fraction": float(f)}, float(f)) for f in as_list(slo["fraction"])]
    elif slo.get("ms"):
        slo_points = [({"ms": float(slo["ms"])}, None)]
    else:
        raise ValueError("every section needs slo: {ms: ...} or slo: {fraction: [...]}")
    repeats = int(sec.get("repeats", 1))

    axis_points: list[tuple[str, str, dict]] = [("base", "", {})]
    for key, values in (sec.get("axes") or {}).items():
        if key.startswith(RECALIBRATION_AXES) and key != LADDER_AXIS:
            print(f"[skip] axis {key}: needs new samplings; not automated", file=sys.stderr)
            continue
        axis_points += [(key, str(v), {key: v}) for v in as_list(values)]

    runs = []
    for ds, pol, n_threads, (slo_cfg, frac), (wl_cfg, profile, rho), rep, (axis, av, overrides) in itertools.product(
            datasets, policies, threads, slo_points, wl_points, range(repeats), axis_points):
        cfg: dict[str, Any] = {"policy": pol, "dataset": ds, "engine": {"threads": n_threads},
                               "slo": slo_cfg, "workload": dict(wl_cfg)}
        for key, value in overrides.items():
            set_dotted(cfg, key, value)
        if vary_threads:
            axis, av = "threads", str(n_threads)
        slo_label = f"f{dot(frac)}" if frac is not None else "fixed"
        run_id = "__".join([slug(ds), slug(pol), f"c{n_threads}", slo_label, f"{profile}-rho{dot(rho)}",
                            "base" if axis == "base" else f"{slug(axis)}-{slug(av)}", f"r{rep}"])
        runs.append(Run(out_name, run_id, ds, pol, n_threads, profile, rho, rep, axis, av, cfg, frac))
    return runs


def ladder_tables(calibrate: Path, run: Run, out_dir: Path, ratio: float, force: bool,
                  dry_run: bool) -> tuple[Path | None, str]:
    """Ladder axis: one probe menu per ladder ratio, kept under results/<experiment>/_tables/<dataset>/ladder_<ratio>/."""
    tables = out_dir / "_tables" / run.dataset / f"ladder_{slug(ratio)}"
    menu = tables / "probe_menu.json"
    if tables in _built_tables or (menu.exists() and not force):
        return tables, "ladder tables reused"
    cmd = [str(calibrate), "--dataset-dir", str(ROOT / "data" / "artifacts" / run.dataset),
           "--tables-dir", str(tables), "--ladder-ratio", str(ratio), "--threads", str(run.threads)]
    if dry_run:
        print("  " + " ".join(cmd))
        return tables, "dry run"
    if not calibrate.exists():
        return None, f"failed: {calibrate} not found; build it first"
    tables.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        return None, f"failed: laef_calibrate exit {proc.returncode}"
    _built_tables.add(tables)
    return tables, "ladder tables built"


def to_absolute_rates(config: dict[str, Any], mu_sat_qps: float) -> dict[str, Any]:
    """Experiments write load as rho, a multiple of the dataset's measured mu_sat(C). The engine takes absolute
    arrival rates in queries/s, so the conversion happens here, once per run, and run_config.json holds qps."""
    cfg = json.loads(json.dumps(config))
    wl = cfg.get("workload", {})
    if "rho" in wl:
        wl["qps"] = float(wl.pop("rho")) * mu_sat_qps
    for ph in wl.get("phases", []):
        if "rho" in ph:
            ph["qps"] = float(ph.pop("rho")) * mu_sat_qps
    for end in ("lo", "hi"):
        if f"rho_{end}" in wl:
            wl[f"qps_{end}"] = float(wl.pop(f"rho_{end}")) * mu_sat_qps
    return cfg


def resolve_slo(config: dict[str, Any], profile_p95_ms: float) -> dict[str, Any]:
    """The engine takes T_SLO in ms only. `slo: {fraction: f}` becomes ms = f x the dataset's profile p95."""
    cfg = json.loads(json.dumps(config))
    slo = cfg.get("slo") or {}
    if "fraction" in slo:
        cfg["slo"] = {"ms": float(slo["fraction"]) * profile_p95_ms}
    return cfg


def run_bench(bench: Path, calibrate: Path, run: Run, out_dir: Path, force: bool, dry_run: bool) -> str:
    run_dir = out_dir / run.run_id
    if (run_dir / "summary.json").exists() and not force:
        return "skipped (summary exists)"
    dataset_dir = ROOT / "data" / "artifacts" / run.dataset
    if not (dataset_dir / "train.npy").exists():
        return f"skipped (no {dataset_dir / 'train.npy'})"
    extra: list[str] = []
    ratio = run.config.get("calibration", {}).get("ladder_ratio")
    if ratio is not None and float(ratio) != DEFAULT_LADDER_RATIO:
        tables, status = ladder_tables(calibrate, run, out_dir, float(ratio), force, dry_run)
        if tables is None:
            return status
        extra = ["--tables-dir", str(tables)]
    profile_path = dataset_dir / "profile" / f"profile_C{run.threads}.json"
    if not profile_path.exists():
        return f"skipped (no {profile_path}; run laef_profile first)"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    mu_sat = float(profile["saturation"]["mu_sat_qps"])
    run.profile_p95_ms = float(profile["saturation"]["s_p95_s"]) * 1e3
    cfg_path = run_dir / "run_config.json"
    cmd = [str(bench), "--dataset-dir", str(dataset_dir), "--config", str(cfg_path), "--out", str(run_dir),
           "--threads", str(run.threads), "--repeat", str(run.repeat), *extra]
    if dry_run:  # print only: a dry run writes no files
        print("  " + " ".join(cmd))
        return "dry run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(resolve_slo(to_absolute_rates(run.config, mu_sat), run.profile_p95_ms), indent=2),
                        encoding="utf-8")
    (run_dir / "run_meta.json").write_text(json.dumps(run.meta(), indent=2), encoding="utf-8")
    proc = subprocess.run(cmd, check=False)
    return "done" if proc.returncode == 0 else f"failed (exit {proc.returncode})"


def aggregate(out_dir: Path) -> int:
    rows = []
    for run_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()) if out_dir.exists() else []:
        meta_p, sum_p = run_dir / "run_meta.json", run_dir / "summary.json"
        if meta_p.exists() and sum_p.exists():
            rows.append({**json.loads(meta_p.read_text(encoding="utf-8")),
                         **json.loads(sum_p.read_text(encoding="utf-8"))})
    if not rows:
        return 0
    fieldnames = list(META_COLUMNS)
    for row in rows:
        fieldnames += [k for k in row if k not in fieldnames]
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LA-ef experiment runner")
    ap.add_argument("experiment", type=Path, help="experiment YAML, e.g. configs/experiments/slo_study_120s.yaml")
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH, help="path to laef_bench")
    ap.add_argument("--force", action="store_true", help="rerun runs that already have a summary.json")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--aggregate-only", action="store_true", help="only rebuild summary.csv")
    args = ap.parse_args(argv)
    calibrate = args.bench.with_name("laef_calibrate")  # built next to laef_bench (ladder axis)

    spec_path = args.experiment if args.experiment.is_absolute() else ROOT / args.experiment
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    sections: list[tuple[str, dict]] = []
    if "datasets" in spec:
        sections.append((spec_path.stem, spec))
    sections += [(key, val) for key, val in spec.items() if isinstance(val, dict) and "datasets" in val]

    if not (args.dry_run or args.aggregate_only) and not args.bench.exists():
        print(f"error: {args.bench} not found; build first: cmake -S cpp -B build && cmake --build build", file=sys.stderr)
        return 1

    failures = 0
    for section, sec in sections:
        try:
            runs = build_runs(section, sec)
        except (OSError, ValueError) as e:  # an invalid section
            print(f"[error] {section}: {e}", file=sys.stderr, flush=True)
            failures += 1
            continue
        out_dir = ROOT / "results" / sec.get("name", section)
        print(f"== {out_dir.relative_to(ROOT)}: {len(runs)} runs", flush=True)
        if not args.aggregate_only:
            for i, run in enumerate(runs, 1):
                status = run_bench(args.bench, calibrate, run, out_dir, args.force, args.dry_run)
                failures += status.startswith("failed")
                print(f"[{i}/{len(runs)}] {run.run_id}: {status}", flush=True)
        if not args.dry_run:
            n = aggregate(out_dir)
            print(f"summary.csv: {n} runs", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
