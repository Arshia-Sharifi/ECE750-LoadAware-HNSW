from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

import pandas as pd

from .shell import capture


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def cpu_info() -> dict:
    fields = {}
    for line in capture(["lscpu"]).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    logical = os.cpu_count() or 1
    try:
        per_core = int(fields.get("Thread(s) per core", "1"))
    except ValueError:
        per_core = 1
    per_core = max(1, per_core)
    return {
        "model": fields.get("Model name", platform.processor() or "unknown"),
        "logical_cpus": logical,
        "physical_cores": logical // per_core,
        "threads_per_core": per_core,
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or "unknown",
        "wsl": "microsoft" in platform.release().lower(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def show(info: dict) -> None:
    for k, v in info.items():
        print(f"{k:16s} {v}")


def check_threads(threads, info: dict, pin_cores: bool = False) -> pd.DataFrame:
    logical, physical = info["logical_cpus"], info["physical_cores"]
    rows = []
    for t in sorted(threads):
        notes, level = [], "ok"
        if t > logical:
            level = "block"
            notes.append(f"{t} workers on {logical} logical CPUs: mu_sat is not a capacity measurement")
        elif t > physical:
            level = "warn"
            notes.append(f"above {physical} physical cores, so these workers share SMT siblings")
        if pin_cores and t + 2 > logical:
            level = "block" if level != "block" else level
            notes.append(f"pin_cores needs {t + 2} cores and the generator would collide with a worker")
        rows.append({"threads": t, "level": level, "note": "; ".join(notes) or "within physical cores"})
    return pd.DataFrame(rows)


def environment_warnings(info: dict) -> list[str]:
    out = []
    if info["governor"] not in ("performance", "unknown"):
        out.append(f"cpufreq governor is {info['governor']}, so clock changes add latency noise. "
                   "Set it to performance before a measurement run.")
    if info["governor"] == "unknown":
        out.append("cpufreq governor is not readable, so the clock policy cannot be checked from here. "
                   "This is expected under WSL2 and inside containers.")
    if info["wsl"]:
        out.append("Running under WSL2. Timing jitter is higher than on native Linux, so repeat each "
                   "point at least 3 times before trusting a tail-latency difference.")
    if shutil.which("cmake") is None:
        out.append("cmake is not on PATH.")
    return out


def usable_threads(threads, info: dict) -> list[int]:
    return [t for t in sorted(threads) if t <= info["logical_cpus"]]
