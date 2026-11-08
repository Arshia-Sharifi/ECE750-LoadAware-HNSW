from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .config import ROOT


def run(cmd, cwd: Path = ROOT, env: dict | None = None, echo: bool = True,
        check: bool = True, log: Path | None = None, quiet: bool = False) -> int:
    cmd = [str(c) for c in cmd]
    if echo:
        print("$ " + " ".join(cmd), flush=True)
    e = {**os.environ, **(env or {})}
    t0 = time.time()
    sink = open(log, "a", encoding="utf-8") if log else None
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=e, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    for line in proc.stdout:
        if sink:
            sink.write(line)
        if not quiet:
            sys.stdout.write(line)
        tail.append(line)
        del tail[:-40]
    proc.wait()
    if sink:
        sink.close()
    if quiet and proc.returncode != 0:
        sys.stdout.write("".join(tail))
    if echo:
        print(f"  exit {proc.returncode} in {time.time() - t0:.1f}s", flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def capture(cmd, cwd: Path = ROOT) -> str:
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd), capture_output=True,
                          text=True).stdout.strip()
