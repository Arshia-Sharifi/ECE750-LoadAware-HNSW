from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
BIN = BUILD / "bin"
DATA = ROOT / "data"
ARTIFACTS = DATA / "artifacts"
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "scripts"
EXPERIMENTS = ROOT / "configs" / "experiments"
REGISTRY = ROOT / "configs" / "datasets.yaml"
PREP = ROOT / "gpu_preprocessing"

THIRD_PARTY = ROOT / "cpp" / "third_party"
ADAEF_DIR = THIRD_PARTY / "hnsw-ada-ef"
ADAEF_REPO = "https://github.com/chaozhang-cs/hnsw-ada-ef"
ADAEF_SHA = "ed463f9993868f7ecc7c103920644e7f94abb377"
VENDOR = THIRD_PARTY / "vendor"
EIGEN_URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
BOOST_MATH_REPO = "https://github.com/boostorg/math.git"
BOOST_MATH_TAG = "boost-1.84.0"

APPS = ("laef_build_index", "laef_calibrate", "laef_profile", "laef_bench")
M = 16
EFC = 500
PROBE_LENGTHS = "1025,512,256,128"

BASELINE_POLICY = "ada_ef"
LA_POLICIES = ("laef_uniform", "laef_wf")
ARRIVALS = ("constant", "poisson")
POLICY_LABEL = {
    "ada_ef": "stock Ada-ef",
    "laef_uniform": "LA-ef naive cut",
    "laef_wf": "LA-ef proportional cut",
    "static_ef": "static ef",
}

DATASET_INPUTS = (
    "train.npy", "test.npy", "meta.json", "gt/neighbors.npy",
    f"index/hnsw_M{M}_efc{EFC}.bin", "stats/estimator.bin",
    "tables/probe_menu.json", "tables/L1025/ada_ef_adapter.bin", "tables/L1025/price_list.json",
)


@dataclass(frozen=True)
class Study:
    name: str = "thread_scaling"
    datasets: tuple[str, ...] = ("synthetic-small",)
    threads: tuple[int, ...] = (2, 4, 8, 16, 24)
    la_policies: tuple[str, ...] = LA_POLICIES
    rhos: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.3)
    slo_fractions: tuple[float, ...] = (0.9,)
    duration_s: float = 60.0
    repeats: int = 1
    arrival: str = "constant"
    pin_cores: bool = False

    def with_(self, **kw) -> "Study":
        return replace(self, **kw)

    @property
    def dir(self) -> Path:
        return RESULTS / self.name

    @property
    def yaml_path(self) -> Path:
        return EXPERIMENTS / f"{self.name}.yaml"

    def thread_dir(self, t: int) -> Path:
        return self.dir / f"c{t}"

    @property
    def runs_per_thread(self) -> int:
        n = len(self.datasets) * len(self.rhos) * self.repeats
        return n + n * len(self.la_policies) * len(self.slo_fractions)

    @property
    def n_runs(self) -> int:
        return self.runs_per_thread * len(self.threads)

    @property
    def bench_hours(self) -> float:
        return self.n_runs * self.duration_s / 3600.0
