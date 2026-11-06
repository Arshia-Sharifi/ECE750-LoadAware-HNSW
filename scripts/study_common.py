#!/usr/bin/env python
"""Helpers for scripts/run_slo_study.py.

The script reads one experiment file from configs/experiments/, runs it with scripts/run_experiments.py, and saves the
analytics under results/<study>/, where <study> is the file name without .yaml.
"""
from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]  # the Implementation folder
SCRIPTS = ROOT / "scripts"
RUNNER = SCRIPTS / "run_experiments.py"
sys.path.insert(0, str(SCRIPTS))
import interval_stats  # noqa: E402  (scripts/interval_stats.py)

# Files that every dataset needs before a run: the files laef_bench opens. The index name matches the defaults in
# configs/defaults.yaml (hnsw M 16, ef_construction 500). The profile is checked for each thread count. The probe menu
# lists the other probe lengths, and laef_bench checks those tables itself. The full-probe table (L1025) is checked here.
DATASET_INPUTS = ["train.npy", "test.npy", "meta.json", "gt/neighbors.npy", "index/hnsw_M16_efc500.bin",
                  "stats/estimator.bin", "tables/probe_menu.json", "tables/L1025/ada_ef_adapter.bin",
                  "tables/L1025/price_list.json"]


def as_list(v) -> list:
    return list(v) if isinstance(v, (list, tuple)) else [v]


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_sections(path: Path, study: str) -> list[dict]:
    """The sections of an experiment file (the dicts with a 'datasets' key). Each one must be named <study>/<section>,
    so that its outputs land in results/<study>/."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    sections = [v for v in spec.values() if isinstance(v, dict) and "datasets" in v]
    if not sections:
        raise SystemExit(f"error: {path} has no sections with a 'datasets' key")
    for sec in sections:
        if not str(sec.get("name", "")).startswith(f"{study}/"):
            raise SystemExit(f"error: every section needs name: {study}/<section> (see {path})")
    return sections


def check_inputs(sections: list[dict]) -> None:
    """Stops before the first run when an input is missing. The runner skips runs whose inputs are missing, so without
    this check a study would finish with holes in it instead of an error."""
    threads = sorted({int(t) for sec in sections for t in as_list(sec.get("threads", 4))})
    for ds in sorted({d for sec in sections for d in as_list(sec["datasets"])}):
        base = ROOT / "data" / "artifacts" / ds
        needed = DATASET_INPUTS + [f"profile/profile_C{t}.json" for t in threads]
        missing = [rel for rel in needed if not (base / rel).exists()]
        if missing:
            raise SystemExit(f"error: data/artifacts/{ds} is missing {missing}; build, calibrate and profile it first")
        for t in threads:
            sat = json.loads((base / f"profile/profile_C{t}.json").read_text(encoding="utf-8"))["saturation"]
            print(f"{ds}: mu_sat({t} threads) {sat['mu_sat_qps']:.1f} qps, profile run {sat['seconds']:.0f} s",
                  flush=True)


def count_runs(sections: list[dict]) -> tuple[int, int]:
    """(total runs, runs still to do) for the sections. A run is done when its folder has a summary.json."""
    import run_experiments as rx  # scripts/run_experiments.py

    total = pending = 0
    for sec in sections:
        try:
            runs = rx.build_runs(sec["name"], sec)
        except (OSError, ValueError) as e:
            print(f"warning: {sec['name']}: {e}", flush=True)
            continue
        total += len(runs)
        pending += sum(not (ROOT / "results" / sec["name"] / r.run_id / "summary.json").exists() for r in runs)
    return total, pending


RUN_LINE = re.compile(r"^\[(\d+)/(\d+)\] (\S+): (.*)$")  # the runner prints this after each run
SHOW_LINE = re.compile(r"error|failed|warning|traceback|\[skip\]", re.IGNORECASE)


def _fmt_time(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def run_with_progress(cmd: list[str], total: int, pending: int, log_path: Path) -> int:
    """Runs the runner and shows one progress line: a bar, percent, runs done, elapsed time, ETA and the time spent in
    the current run. Everything the runner and the engine print goes to log_path. Lines that mention an error, a failure
    or a warning are also shown on screen. Returns the runner's exit code.

    On a terminal the line is redrawn in place. Otherwise (a redirected log) one line is printed per finished run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tty = sys.stdout.isatty()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)  # own process group: Ctrl-C is handled here, then the group is stopped
    interrupted = False
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    start = last_mark = last_beat = time.time()
    done = executed = 0
    exec_time = 0.0
    last_run = ""

    def bar_text() -> str:
        now = time.time()
        frac = min(1.0, done / total) if total else 1.0
        filled = int(30 * frac)
        left = max(0, pending - executed)
        eta = _fmt_time(left * exec_time / executed) if executed else "?"
        return (f"[{'#' * filled}{'-' * (30 - filled)}] {frac * 100:5.1f}%  {done}/{total} runs  "
                f"elapsed {_fmt_time(now - start)}  ETA {eta}  current run {int(now - last_mark)} s")

    def draw() -> None:
        if tty:
            width = shutil.get_terminal_size((100, 20)).columns - 1
            sys.stdout.write("\r" + bar_text()[:width].ljust(width))
            sys.stdout.flush()

    def message(text: str) -> None:
        if tty:
            sys.stdout.write("\r" + " " * (shutil.get_terminal_size((100, 20)).columns - 1) + "\r")
        print(text, flush=True)

    try:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            draw()
            while True:
                try:
                    line = lines.get(timeout=1.0)
                except queue.Empty:
                    draw()
                    if not tty and time.time() - last_beat >= 30:  # a heartbeat, so a redirected log shows life
                        last_beat = time.time()
                        print(f"{bar_text()}  (running)", flush=True)
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                text = line.rstrip()
                m = RUN_LINE.match(text)
                if m:
                    done += 1
                    status = m.group(4)
                    now = time.time()
                    if status.startswith(("done", "failed")):
                        executed += 1
                        exec_time += now - last_mark
                    last_mark = now
                    last_beat = now
                    last_run = f"{m.group(3)} {status}"
                    if not tty:
                        print(f"{bar_text()}  last: {last_run}", flush=True)
                    elif status.startswith("failed"):
                        message(f"failed: {m.group(3)} ({status})")
                elif SHOW_LINE.search(text) and not text.startswith("["):
                    message(text)
                draw()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for _ in range(2):  # a second signal (Ctrl-C twice, or one sent to the whole process group) must not crash
            try:
                if interrupted and proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)  # the runner and the engine it started
                    except ProcessLookupError:
                        pass
                proc.wait()
                break
            except KeyboardInterrupt:
                interrupted = True
    if tty:
        sys.stdout.write("\n")
    if interrupted:
        print("interrupted: the runner was stopped. Run the same command again to continue; "
              "finished runs are kept.", flush=True)
        return 130
    print(f"finished in {_fmt_time(time.time() - start)}; details in {log_path}", flush=True)
    return proc.returncode


def run_dirs(study_dir: Path) -> list[Path]:
    """Run folders: every folder under study_dir that has a run_meta.json."""
    return sorted(p.parent for p in study_dir.rglob("run_meta.json"))


def incomplete_runs(study_dir: Path) -> list[str]:
    """Run folders with a run_meta.json but no summary.json: the run failed or was interrupted."""
    return [d.name for d in run_dirs(study_dir) if not (d / "summary.json").exists()]


def write_intervals(study_dir: Path, interval: float, keep_queries: bool) -> int:
    """Builds intervals.csv in each run folder from its queries.csv, then deletes queries.csv unless keep_queries is
    set. A run whose statistics fail keeps its queries.csv, so no data is lost."""
    n = 0
    for run_dir in run_dirs(study_dir):
        queries = run_dir / "queries.csv"
        if not queries.exists():
            continue
        try:
            intervals, _, _ = interval_stats.analyse_run(run_dir, interval)
        except (OSError, ValueError, KeyError) as e:
            print(f"warning: {run_dir.name}: intervals.csv not written ({e}); queries.csv kept", flush=True)
            continue
        interval_stats.write_csv(run_dir / "intervals.csv", intervals)
        if not keep_queries:
            queries.unlink()
        n += 1
    return n


def gather(study_dir: Path) -> list[dict]:
    """Every row of every summary.csv under study_dir (one row per run)."""
    rows = []
    for path in sorted(study_dir.rglob("summary.csv")):
        rows += read_rows(path)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
