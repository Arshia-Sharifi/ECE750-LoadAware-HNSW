from __future__ import annotations

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from . import config as C
from .shell import capture, run


def _cmake() -> str:
    exe = shutil.which("cmake")
    if exe:
        return exe
    local = Path.home() / ".local" / "bin" / "cmake"
    if local.exists():
        return str(local)
    raise RuntimeError("cmake not found. Install it with apt, or run build.install_cmake().")


def install_cmake() -> str:
    run(["pip", "install", "--quiet", "cmake", "ninja"], quiet=True)
    os.environ["PATH"] = f"{Path.home() / '.local' / 'bin'}:{os.environ['PATH']}"
    return _cmake()


def ensure_adaef(force: bool = False) -> Path:
    if force and C.ADAEF_DIR.exists():
        shutil.rmtree(C.ADAEF_DIR)
    if not (C.ADAEF_DIR / "hnswlib").exists():
        C.ADAEF_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", C.ADAEF_REPO, str(C.ADAEF_DIR)])
    head = capture(["git", "-C", str(C.ADAEF_DIR), "rev-parse", "HEAD"])
    if head != C.ADAEF_SHA:
        run(["git", "-C", str(C.ADAEF_DIR), "fetch", "--quiet", "origin", C.ADAEF_SHA], check=False)
        run(["git", "-C", str(C.ADAEF_DIR), "checkout", "--quiet", C.ADAEF_SHA])
    return C.ADAEF_DIR


def find_eigen() -> str | None:
    vendored = C.VENDOR / "eigen-3.4.0"
    if (vendored / "Eigen" / "Core").exists():
        return str(vendored)
    for p in ("/usr/include/eigen3", "/usr/local/include/eigen3"):
        if Path(p, "Eigen", "Core").exists():
            return p
    return None


def find_boost_math() -> str | None:
    vendored = C.VENDOR / "math" / "include"
    if (vendored / "boost" / "math" / "distributions" / "normal.hpp").exists():
        return str(vendored)
    for p in ("/usr/include", "/usr/local/include"):
        if Path(p, "boost", "math", "distributions", "normal.hpp").exists():
            return p
    return None


def vendor_headers() -> tuple[str, str]:
    C.VENDOR.mkdir(parents=True, exist_ok=True)
    if find_eigen() is None:
        tgz = C.VENDOR / "eigen.tar.gz"
        urllib.request.urlretrieve(C.EIGEN_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(C.VENDOR)
        tgz.unlink()
    if find_boost_math() is None:
        run(["git", "clone", "--quiet", "--depth", "1", "--branch", C.BOOST_MATH_TAG,
             C.BOOST_MATH_REPO, str(C.VENDOR / "math")])
    eigen, boost = find_eigen(), find_boost_math()
    if not eigen or not boost:
        raise RuntimeError("could not provide Eigen and Boost.Math headers")
    return eigen, boost


def configure(native: bool = True, extra: list[str] | None = None) -> None:
    ensure_adaef()
    eigen, boost = find_eigen(), find_boost_math()
    if not eigen or not boost:
        eigen, boost = vendor_headers()
    cmd = [_cmake(), "-S", str(C.ROOT / "cpp"), "-B", str(C.BUILD),
           "-DCMAKE_BUILD_TYPE=Release",
           f"-DLAEF_EIGEN_INCLUDE={eigen}", f"-DLAEF_BOOST_INCLUDE={boost}",
           f"-DLAEF_NATIVE={'ON' if native else 'OFF'}"]
    run(cmd + (extra or []), quiet=True)


def compile_(jobs: int | None = None) -> None:
    jobs = jobs or os.cpu_count() or 1
    run([_cmake(), "--build", str(C.BUILD), "-j", str(jobs)], quiet=True)


def ctest(fast: bool = True, jobs: int = 1) -> int:
    ctest_exe = str(Path(_cmake()).with_name("ctest"))
    cmd = [ctest_exe, "--test-dir", str(C.BUILD), "--output-on-failure", "-j", str(jobs)]
    if fast:
        cmd += ["-E", "engine_qps"]
    return run(cmd, check=False, quiet=True)


def binaries() -> dict[str, bool]:
    return {app: (C.BIN / app).exists() for app in C.APPS}


def ready() -> bool:
    return all(binaries().values())
