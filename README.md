# LA-ef: Load-Aware Ada-ef

LA-ef adds a load-aware latency controller to Ada-ef's HNSW search. The full spec is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Design background: `../Main Flow.md` and `../Budget B calculations.md`.

**Status:** the code is written but has not been compiled or run. The first job on WSL Ubuntu is to build and test it. The plan's "Implementation status" table lists what was verified and what was not.

## Pipeline

1. **Colab T4** (`gpu_preprocessing/`): exact ground truth, calibration samples, estimator statistics and `manifest.json`.
2. **Serving machine** (WSL2 Ubuntu or Linux, C++20):
   - `prepare_dataset.sh` rebuilds `train.npy` and `test.npy` from the raw download, checks the fingerprints, and copies the Colab outputs.
   - `laef_build_index` builds the HNSW index (M = 16, efc = 500).
   - `laef_calibrate` builds the Ada-ef adapter, the ladder sweep and the price list (Step A).
   - `laef_profile` measures idle and saturation latency (auto SLO and mu_sat).
   - `laef_bench` runs one experiment under load (Steps B to F). No query is dropped.
3. `scripts/run_experiments.py` expands an experiment YAML into `laef_bench` runs and writes `results/<experiment>/summary.csv`. The notebooks in `analysis/notebooks/` plot the results.

## Dataset files

Datasets are NumPy `.npy` files with JSON metadata. HDF5 is not used (plan section 4.3).

```
data/artifacts/<id>/
  train.npy, test.npy, holdout.npy, meta.json   rebuilt on the serving machine (large)
  gt/neighbors.npy, gt/gt_meta.json             from Colab
  calibration/samplings.bin, calib_meta.json    from Colab
  stats/estimator.bin, manifest.json            from Colab
  index/, tables/, profile/                     built on the serving machine
```

## Quick start (WSL2 Ubuntu)

Run the scripts with `bash`. A Windows checkout does not keep the executable bit, so `bash scripts/...` always works.

```bash
bash scripts/setup_wsl_ubuntu.sh                         # apt packages, uv, tool version checks
uv venv --python 3.11 .venv
uv pip install -p .venv numpy h5py pyyaml pytest pandas matplotlib
.venv/bin/python -m pytest gpu_preprocessing/tests -q    # Python tests (CPU)
git clone https://github.com/chaozhang-cs/hnsw-ada-ef cpp/third_party/hnsw-ada-ef
git -C cpp/third_party/hnsw-ada-ef checkout ed463f9993868f7ecc7c103920644e7f94abb377
bash scripts/build.sh                                    # configure, build and ctest
bash scripts/prepare_dataset.sh glove-100-angular /path/to/laef_artifacts/glove-100-angular
bash scripts/run_pipeline.sh glove-100-angular 4        # build_index -> calibrate -> profile, C = 4
.venv/bin/python scripts/run_slo_study.py                # SLO study: stock Ada-ef and LA-ef at fractions of the profile p95 (about 3 hours)
```

`scripts/build.sh` passes extra arguments to CMake, for example `bash scripts/build.sh -DLAEF_WITH_ADAEF=OFF` for a build without the Ada-ef backend.
