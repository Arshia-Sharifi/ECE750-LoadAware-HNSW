#!/usr/bin/env python
"""SLO study: run configs/experiments/slo_study_120s.yaml, then save the analytics next to its results.

    .venv/bin/python scripts/run_slo_study.py

The experiment file holds 81 runs of 120 s with constant-rate arrivals: 3 datasets x 3 loads (0.5, 1.0, 1.5 x mu_sat),
with stock Ada-ef (one run per load) and LA-ef (naive cut laef_uniform and proportional cut laef_wf, P-AIMD controller,
four SLO fractions 0.95, 0.9, 0.8 and 0.75). T_SLO = fraction x P, where P is the dataset's saturation p95 from
data/artifacts/<dataset>/profile/profile_C4.json (saturation.s_p95_s). P is a closed-loop service time: it has no queue
wait in it, so a run can miss T_SLO because of queueing alone.

This script does three things:
1. Runs the file with scripts/run_experiments.py and shows a progress bar (percent, runs done, elapsed time, ETA). The
   engine output goes to results/slo_study_120s/run.log; errors and warnings are also shown on screen. A run that
   already has summary.json is skipped, so running the same command again finishes an interrupted study.
2. Writes intervals.csv into every run folder: one row per time bin (1 s by default) with arrivals, completions, p50 and
   p95 latency, SLO attainment, requests in the system, ef ratio, recall and controller state. Then it deletes
   queries.csv, unless --keep-queries is given.
3. Writes summary_all.csv (every run) and comparison.csv (each LA-ef run next to the stock Ada-ef run at the same
   dataset and load) into results/slo_study_120s/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import study_common as sc

ROOT = sc.ROOT
DEFAULT_EXPERIMENT = ROOT / "configs" / "experiments" / "slo_study_large_120s.yaml"
BASELINE = "ada_ef"  # stock Ada-ef: the reference for every LA-ef run
LA_ALLOCATOR = {"laef_uniform": "naive cut", "laef_wf": "proportional cut"}


def comparison(rows: list[dict]) -> list[dict]:
    """Each LA-ef run next to the stock Ada-ef run at the same dataset and load."""
    base = {}
    for r in rows:
        if r["policy"] == BASELINE:
            base[(r["dataset"], round(float(r["rho"]), 3))] = r
    out = []
    for r in rows:
        if r["policy"] not in LA_ALLOCATOR:
            continue
        ds, rho = r["dataset"], round(float(r["rho"]), 3)
        b = base.get((ds, rho))
        if b is None:
            print(f"warning: no stock Ada-ef run for {ds} at rho {rho:g}; {r['run_id']} is not compared", flush=True)
            continue
        slo, p95 = float(r["slo_ms"]), float(r["p95_ms"])
        rec_ref, rec = float(b["mean_recall"]), float(r["mean_recall"])
        out.append({
            "dataset": ds, "rho": rho, "allocator": LA_ALLOCATOR[r["policy"]], "policy": r["policy"],
            "fraction": round(float(r["slo_fraction"]), 3), "slo_ms": round(slo, 4),
            "profile_p95_ms": round(float(r["profile_p95_ms"]), 4),
            "ada_p95_ms": round(float(b["p95_ms"]), 3), "la_p95_ms": round(p95, 3),
            "p95_over_slo": round(p95 / slo, 3), "meets_slo": p95 <= slo,
            "slo_attainment": round(float(r["slo_attainment"]), 4),
            "ada_slo_attainment_at_p": round(float(b["slo_attainment"]), 4),
            "ada_recall": round(rec_ref, 5), "la_recall": round(rec, 5),
            "recall_loss_pts": round((rec_ref - rec) * 100, 3),
            "la_ef_ratio": round(float(r["mean_ef_ratio"]), 4),
            "la_mean_probe_len": round(float(r["mean_probe_len"]), 1) if r.get("mean_probe_len") else "",
            "la_frac_full_probe": round(float(r["frac_full_probe"]), 4) if r.get("frac_full_probe") else "",
            "offered_qps": round(float(r["offered_qps"]), 1), "achieved_qps": round(float(r["achieved_qps"]), 1),
            "goodput_qps": round(float(r["goodput_qps"]), 1), "n_queries": int(float(r["n_queries"])),
            "generator_lag_p99_us": round(float(r["generator_lag_p99_us"]), 1), "run_id": r["run_id"],
        })
    out.sort(key=lambda x: (x["dataset"], x["rho"], -x["fraction"], x["policy"]))
    return out


def print_table(rows: list[dict]) -> None:
    print(f"\n{'dataset':20s} {'rho':>4s} {'allocator':16s} {'frac':>5s} {'slo_ms':>9s} {'la_p95_ms':>10s} "
          f"{'meets':>5s} {'attain':>7s} {'loss_pts':>9s} {'ef_ratio':>8s}")
    for r in rows:
        print(f"{r['dataset']:20s} {r['rho']:>4g} {r['allocator']:16s} {r['fraction']:>5.2f} "
              f"{r['slo_ms']:>9.4f} {r['la_p95_ms']:>10.3f} {'yes' if r['meets_slo'] else 'no':>5s} "
              f"{r['slo_attainment']:>7.3f} {r['recall_loss_pts']:>9.3f} {r['la_ef_ratio']:>8.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the SLO study and save its analytics")
    ap.add_argument(
        "--experiment", "--config",
        dest="experiment",
        default=str(DEFAULT_EXPERIMENT),
        help="path to experiment YAML (default: configs/experiments/slo_study_large_120s.yaml)",
    )
    ap.add_argument("--interval", type=float, default=1.0, help="length of each saved time bin in seconds")
    ap.add_argument("--keep-queries", action="store_true", help="keep each run's queries.csv (large)")
    args = ap.parse_args(argv)

    exp_path = Path(args.experiment)
    if not exp_path.is_absolute():
        exp_path = ROOT / exp_path

    study = exp_path.stem
    study_dir = ROOT / "results" / study
    sections = sc.load_sections(exp_path, study)
    sc.check_inputs(sections)

    total, pending = sc.count_runs(sections)
    print(f"{total} runs in {exp_path.name}, {pending} still to run. Details go to results/{study}/run.log", flush=True)
    status = sc.run_with_progress([sys.executable, "-u", str(sc.RUNNER), str(exp_path)], total, pending,
                                  study_dir / "run.log")
    if status == 130:  # interrupted: skip the summary step; the same command continues the study
        return 130

    n = sc.write_intervals(study_dir, args.interval, args.keep_queries)
    print(f"intervals.csv written for {n} runs", flush=True)

    rows = sc.gather(study_dir)
    comp = comparison(rows)
    sc.write_csv(study_dir / "summary_all.csv", rows)
    sc.write_csv(study_dir / "comparison.csv", comp)
    print_table(comp)
    print(f"\nwrote results/{study}/summary_all.csv ({len(rows)} runs) and comparison.csv ({len(comp)} rows)",
          flush=True)

    bad = sc.incomplete_runs(study_dir)
    if status or bad:
        print(f"warning: runs without summary.json: {bad}. Run the same command again to retry them.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
