# LA-ef: Load-Aware Ada-ef

LA-ef adds a load-aware latency controller and a per-query effort allocator to Ada-ef's HNSW search.
Under overload the server cannot give every query its preferred `efSearch`, so a budget is split
across queries. The question the project answers is whether spending that budget where it buys the
most recall per unit of work preserves more retrieval quality than cutting every query by the same
amount at the same latency deadline.

The full specification is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Start here

Open [LAEF.ipynb](LAEF.ipynb). It is the single entry point. It checks the machine, builds the C++,
prepares datasets, runs the index, calibration and profiling steps, launches the thread-scaling
study and draws every figure. Each step is skipped when its output already exists, so the notebook
is safe to rerun and an interrupted study continues where it stopped.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab LAEF.ipynb
```

The notebook calls into `laef_nb/`, which is documented in [laef_nb/README.md](laef_nb/README.md).
Nothing else needs to be run by hand. The scripts in `scripts/` are still the underlying entry
points and can be used directly for a one-off run.

The harness has its own regression suite, which exercises the full intended grid of 3 datasets, 5
thread counts, 5 loads, 4 SLO fractions and 3 repeats without running the engine.

```bash
.venv/bin/python -m pytest laef_nb/tests -q
```

## Pipeline

1. **Preprocessing.** Exact ground truth, calibration samplings and estimator statistics. This is
   GPU work for anything large and runs on Colab through `gpu_preprocessing/colab_preprocess.ipynb`.
   Small datasets can be prepared locally on CPU from the notebook.
2. **Index and calibration.** `laef_build_index` builds HNSW at M 16 and ef_construction 500.
   `laef_calibrate` builds the Ada-ef adapter, the ladder sweep and the price list. Neither depends
   on the worker count, so both are built once per dataset.
3. **Profile.** `laef_profile` measures idle and saturation service time at one worker count and
   writes `profile/profile_C<threads>.json`. Every worker count needs its own profile, because the
   saturation throughput, the unit cost per distance computation and the saturation p95 that defines
   `T_SLO` are all measured at that worker count.
4. **Bench.** `laef_bench` runs one policy on one dataset under one arrival process. No query is
   dropped.
5. **Study.** `scripts/run_experiments.py` expands an experiment YAML into bench runs.
   `scripts/run_slo_study.py` wraps it with a progress bar, per-run interval statistics and
   aggregation. The notebook generates the YAML and calls both.

## Results layout

```
results/<study>/
  c<threads>/ada/<run-id>/     stock Ada-ef at this worker count
  c<threads>/laef/<run-id>/    LA-ef naive cut and proportional cut
  summary_all.csv              one row per run
  comparison.csv               each LA-ef run next to its stock Ada-ef baseline
  summary_tidy.csv             the same runs with the derived columns used by the figures
  comparison_tidy.csv
  figures/                     every plot
  run.log
```

Load is expressed as rho, a multiple of that worker count's own measured saturation throughput, so
each thread count is driven to the same relative pressure. Absolute arrival rates differ between
thread counts by design, and the baseline for a comparison is always the stock Ada-ef run at the
same dataset, worker count, load and repeat.

## Dataset files

Datasets are NumPy `.npy` files with JSON metadata. HDF5 is read only when downloading from
ann-benchmarks.

```
data/artifacts/<id>/
  train.npy, test.npy, holdout.npy, meta.json   rebuilt on the serving machine
  gt/neighbors.npy, gt/gt_meta.json             exact ground truth
  calibration/samplings.bin, calib_meta.json
  stats/estimator.bin, manifest.json
  index/, tables/, profile/                     built on the serving machine
```

## Dependencies

C++20, CMake 3.22 or newer, Eigen 3.4 and Boost.Math. The notebook downloads Eigen and Boost.Math as
headers into `cpp/third_party/vendor/` when the system does not provide them, and clones
`hnsw-ada-ef` at the pinned commit `ed463f9` into `cpp/third_party/`. Python dependencies are in
`requirements.txt`.

The engine tests are timing sensitive. Run them one at a time. Running them in parallel on a small
machine produces failures that are contention rather than defects.
