# Verification notes: items from the previous pass

Status: findings recorded, fixes in progress. Nothing has been compiled or run (per the user's earlier instruction). Each fix is checked by reading the code against the plan.

## Findings

### Python (gpu_preprocessing)
- F1 `sources.py`: `as_source` calls `isinstance` on a Protocol without `@runtime_checkable`, which raises TypeError. This breaks `tests/test_ground_truth.py`. FIX.
- F2 `sources.py`: `ParquetSource.take` is O(n^2), returns rows in file order instead of request order, and builds Python lists per element. FIX (iter_batches + flatten, sorted take).
- F3 `Hdf5Source.take_by_index` with unsorted indices: h5py rejects non-increasing index lists, and calibration `base_sample` draws unsorted indices. FIX.
- F4 `synthetic.generate_synthetic_small` allocates a (100k x 101k) float32 array (~40 GB) and ignores `datasets.yaml` (blobs, 200 clusters, std 0.05, box [-1, 1]). FIX.
- F5 `registry.py` ignores the yaml keys `column`, `hf_tree_api`, `hf_resolve_prefix`, `synthetic`. `download.py` lists `.../tree/main/train`, but the shards live under `data/`. FIX.
- F6 `convert.py`: holdout split copies the full base (`base[mask]`), so large datasets peak near 12 GB against Colab's ~12.7 GB RAM. FIX: stream chunks into the h5 file.
- F7 Wrong call signatures: `write_samplings(path, q, gt)` and `write_estimator(path, metric, mean, cov)` take the path first. `ground_truth.recall_at_k` does not exist (use `agreement`). `np` is not imported at module scope, so `step_stats` raises NameError. FIX.
- F8 `run_preprocess`: after the download step `raw_dir` is `raw/<id>`, and convert appends `<id>` again. FIX.
- F9 `run_preprocess`: `dataset.h5` is written under `--out` (Google Drive), but plan §7.1 says it stays in `data_root/artifacts`. FIX.
- F10 `step_gt` skips generation when provided GT is missing, so GT can be silently absent. FIX.
- F11 `manifest.py`: fingerprint omits the query bytes, holdout hash is missing, and rows are read one at a time. FIX.
- F12 `download.py`: no retries and no size check (plan P1.2). FIX.

### C++ (cpp/)
- C1 `price_list.cpp` writes private members that do not exist in `price_list.h`. `PriceList::load`, `save`, and `groups()` are not declared. It does not compile. FIX: new `price_list_io.h`, rewrite of `price_list.cpp`.
- C2 `workload.h` is an empty stub. `workload.cpp` treats `rho` as qps (placeholder), starts step phases at the last arrival instead of the phase boundary, returns fewer than n_queries from `pick_queries` when n_eval < n_queries (out of bounds in the engine), and throws for `trace`. FIX.
- C3 `calibration.cpp` is a stub that returns an empty result, so `laef_calibrate` writes nothing. FIX: full Step A driver.
- C4 `engine.cpp` calls `SensorWindow::add/should_close/p95/s_p95/mean_service` and `Telemetry` methods that do not exist. The planner is a passthrough, with no reservation or bucket. Lambda-hat is never updated. Controller inputs are incomplete (t_slo, floor_work, C). The initial snapshot is never stored. Workers overwrite the warm-up flags (race). Static-ef queries are marked probe_incomplete. FIX: rewrite on the real APIs (`Planner`, `TokenBucket`, `SnapshotStore`, `SensorWindow::push/ready/close`).
- C5 `results_writer`: `results_writer.h` uses nlohmann without including it. `TimeSeriesRow` is incomplete in the .cpp. `frac_target_recall` hard-codes 0.95. Wall time includes warm-up. `controller_update_us` and `generator_lag_p99_us` are always 0. `recovery_time_s` is always null. FIX.
- C6 `dataset_io.cpp` calls `getDimension(i)`, which is not a HighFive call. Use `getDimensions()`. FIX.
- C7 `laef_profile` writes zeros and never measures anything. Its unit cost is computed from the wrong quantity, and the probe is double-counted. FIX: rewrite.
- C8 `laef_bench` is a stub. It overrides `floor_mode` with the CLI default (true) even when the policy says off. FIX: rewrite.
- C9 `laef_calibrate` is a stub. FIX (with C3).
- C10 `laef_build_index` does not create `index/`. FIX.
- C11 `calibration.cpp` lacks `<map>`. `fake_backend.cpp` lacks `<algorithm>`. `make_fake_backend` has no declaration. FIX.
- C12 CMake: `Dependencies.cmake` does not define HighFive, nlohmann_json, or CLI11, but `CMakeLists.txt` uses them. nlohmann is linked PRIVATE, so apps get no include path for it. Apps build into `build/`, but `run_experiments.py` looks in `build/apps/`. FIX.

### Experiment runner and analysis
- R1 `run_experiments.py` does not match the YAML format. The YAML uses `slo.multiplier` lists, `workloads` lists, `repeats`, `base_policy` plus `axes`, and an `e5_threads` section. Run ids omit the dataset, so runs collide across datasets. `profile` and `rho` never reach `laef_bench`. `run_config.json` is not a RunConfig. Every run is tagged `axis=base`. FIX: rewrite.
- A1 `metrics.py` defines wall time as the arrival span, while C++ uses end-to-end time. FIX (small).

### Plan document
- P1 The status table marks stubs as done (checkmarks). FIX: correct the rows.

## Known gaps (not fixed in this pass; report to the user)
- Plan finding 5 (query_holdout zero-recall filter before `build_ef_adapter`) is not implemented. The default `base_sample` is unaffected.
- The E4 `calibration.ladder_ratio` axis needs a new calibration per ratio. The runner does not automate that.
- `scripts/build.sh` is still a stub.
- Unverified API details that need a WSL build: HighFive `read_raw`, CLI11 option types, nlohmann `value()`.

## Decisions to confirm with the user
- `synthetic_sample` queries are removed from the base (this matches the `datasets.yaml` comment).
- `dataset.h5` is verified by fingerprint, not by file hash, because it stays on local disk and may not be byte-identical across machines.

## Fix order
1. Python prep: F1-F12.
2. C++ core: C1, C2, C6, C11 (workload, price_list_io, dataset_io).
3. Calibration driver and app: C3, C9.
4. Profile, engine, bench: C4, C7, C8.
5. Results writer, build index, CMake: C5, C10, C12.
6. Experiment runner and metrics: R1, A1.
7. Plan status table: P1.

## Final status
- Verification pass complete by static review. Nothing was compiled, tested, or run (per the user's instruction).
- Plan status table and checklist corrected (12 rows, items 2, 5, 8, 9).
- Open items: query_holdout zero-recall filter; `calibration.*` axes need a new calibration per value; `scripts/build.sh` still a stub; `test_engine_qps` not written; HighFive `read_raw`, CLI11, and nlohmann usage need a WSL build check.
- Decisions to confirm: synthetic_sample queries are held out of the base; `dataset.h5` is verified by fingerprint, not by file hash; holdout indices are stored as the `/holdout` dataset.
- Step 1 (Python prep), written, not run:
  - F1, F2, F3: `prep/sources.py` rewritten. `as_source` no longer uses isinstance on a Protocol. Parquet reads go through `iter_batches`. Sorted, de-duplicated takes.
  - F4: `prep/synthetic.py` rewritten. Blobs and Zipf sizes come from `datasets.yaml`. Synthetic queries are held out of the base.
  - F5, F12: `prep/registry.py` reads `column`, `hf_tree_api`, `hf_resolve_prefix`, `synthetic`. `prep/download.py` uses the shard listing under `data/`, retries with resume, and checks sizes. Raw files are never deleted.
  - F6: `prep/convert.py` streams writes. Holdout split is a single pass. Writes `/holdout` as a dataset (not an attribute, which is too large). Provided GT is copied to `out_dir`.
  - F7, F3: `prep/calibration.py` uses `write_samplings(path, q, gt)` and sorted takes.
  - F11: `prep/manifest.py` fingerprint includes the queries and the holdout hash. Adds `verify_manifest`.
  - F7, F8, F9, F10: `run_preprocess.py` uses the right signatures and paths. `dataset.h5` stays under `data_root/artifacts`. Missing provided GT is copied, not silently skipped. Verification uses `agreement`.
- Step 1 check: grep of the Python tree shows no stale references to removed names or old signatures.
- Step 2 (C++ core), written, not compiled:
  - C2: `include/laef/workload.h` (was an empty stub) and `src/workload.cpp` rewritten. `rho` is resolved against mu_sat (no placeholder). Step phases start at their boundary. `pick_queries` always returns n_queries rows, sampled with replacement from a pool. Trace profile implemented. Added `burst_end_s` for recovery time.
  - C1: `include/laef/price_list_io.h` (new) and `src/price_list.cpp` rewritten on the real `PriceList` API (public `groups`, `finalize()`). The steps are derived, not stored.
  - C6: `include/laef/dataset_io.h` and `src/dataset_io.cpp`: `getDimensions()`. Added `read_dataset_meta`, `load_queries` (no base vectors), and `eval_pool` (honours the query_holdout exclusions).
- Step 3 (Step A driver), written, not compiled:
  - C3: `include/laef/calibration.h` and `src/calibration.cpp` rewritten. The sweep runs the preference pass, then every ladder rung, on `threads` workers. Grouping, envelopes, and steps reuse `envelope_inplace` and `make_groups` from `price_list.h`. Added `read_samplings` (plan §4.1 format) and `make_ladder`.
  - C9: `apps/laef_calibrate.cpp` rewritten. It builds the CPU estimator and the Ada-ef adapter only when they are missing. It writes `tables/price_list.json` and `tables/calibration_sweep.csv`.
- Step 4 (engine, results, profile, bench), written, not compiled:
  - C4: `include/laef/engine.h` and `src/engine.cpp` rewritten on the real APIs. The generator reserves at arrival, workers settle in the probe hook through `Planner`, the controller publishes complete snapshots via `make_snapshot`, and lambda-hat is updated each tick. Warm-up flags are set by the generator only. Static-ef queries are `Ok`. Pinning is Linux-only.
  - C5: `include/laef/results_writer.h` and `src/results_writer.cpp`. `frac_target_recall` uses the configured target. Wall time covers measured queries only. Controller cost and generator lag are measured. `recovery_time_s` is computed for step profiles.
  - C7: `apps/laef_profile.cpp` measures the idle and saturation runs and writes `profile/profile_C{C}.json`.
  - C8: `apps/laef_bench.cpp` loads the profile, price list, backend, and ground truth, resolves rates from mu_sat, runs the engine, and writes config.json, queries.csv, timeseries.csv, summary.json. `--floor-mode` only overrides when given.
- Step 5 (build layer, runner, metrics), written, not compiled or run:
  - C10: `apps/laef_build_index.cpp` creates `index/` and writes `index_meta.json`.
  - C11: `include/laef/fake_backend.h` (new) and `src/fake_backend.cpp`: missing includes added, score mapped into [0, 100), factory declared.
  - C12: `cmake/Dependencies.cmake` now fetches HighFive, nlohmann_json, and CLI11, and finds HDF5. `CMakeLists.txt` lists the real sources, links nlohmann publicly, puts apps in `build/bin`, and drops the merged `test_formats` target.
  - R1: `scripts/run_experiments.py` rewritten for the real YAML layout (`slo.multiplier`, `workloads`, `repeats`, `base_policy` + `axes`, the `e5_threads` section). Run ids include the dataset. Each run writes `run_config.json` (a RunConfig overlay) and `run_meta.json`. `summary.csv` is built from the metadata plus summary keys. `calibration.*` axes are skipped with a message, because they need a new calibration.
  - A1: `analysis/laef_analysis/metrics.py` wall time is end-to-end, matching C++.
- Remaining: update the plan status table and checklist (P1), then the final report.


## Pass 2 (after the steering that asked to drop HDF5 for our dataset)

Steering steer-160c279f: "If HDF5 is creating a lot of implementation complexity then remove this dataset entirely and replace it with something that is easier to work and is actually helpful for our analysis."

Interpretation: the canonical dataset and ground-truth files (`dataset.h5`, `gt/test_gt.h5`) are replaced. HDF5 disappears from the serving code, including CMake, HighFive and the HDF5 C library. The raw ann-benchmarks downloads are HDF5 upstream; only Python (h5py) reads them, once, during conversion.

Decision: NumPy `.npy` plus JSON.
- Why: `np.load(..., mmap_mode="r")` gives analysis code direct access with no custom code. The C++ reader is about 100 lines with no library. The header is self-describing (dtype and shape).
- Dataset folder `data/artifacts/<id>/`: `train.npy` (float32 N x d), `test.npy` (float32 Q x d), `holdout.npy` (int64, holdout datasets only), `meta.json` (metric, seed, source, query_source, holdout_seed).
- GT folder `gt/`: `neighbors.npy` (int32 Q x K), `distances.npy` (float32, optional), `gt_meta.json` (gt_source, K, metric).
- Unchanged: `calibration/samplings.bin` and `stats/estimator.bin`, which are Ada-ef binary formats.

Files to change, in order:
1. C++: `include/laef/dataset_io.h` (API takes the dataset folder), `src/dataset_io.cpp` (NPY reader), call sites in `apps/laef_build_index.cpp`, `apps/laef_calibrate.cpp`, `apps/laef_profile.cpp`, `apps/laef_bench.cpp`.
2. Build: `cpp/CMakeLists.txt` and `cmake/Dependencies.cmake` (remove HDF5 and HighFive).
3. Python: `prep/sources.py` (NpySource; AnnHdf5Source only for the raw download), `prep/convert.py`, `prep/calibration.py`, `prep/manifest.py`, `run_preprocess.py`.
4. Colab notebook: check cell and markdown (reads `gt/test_gt.h5` with h5py).
5. Plan: §2, §3, §4, §6 P0.3, §7 (P1.3, P1.5, P1.6, §7.1), §8.1, §8.2 P2.4, §17 and status table.
6. Still open from pass 1: query_holdout zero-recall filter (finding 5), E4 ladder-ratio axis (needs a per-value calibration), scripts (build.sh, setup_wsl_ubuntu.sh, run_pipeline.sh, prepare_dataset.sh), test files (test_engine_qps, test_fixed_aimd, test_sensor_window, test_allocators), README files.

Status: step 1 (dataset_io) in progress.


## Migration progress (HDF5 to NumPy)
- Done: `cpp/include/laef/dataset_io.h` and `cpp/src/dataset_io.cpp` (folder-based readers for `.npy` and JSON, no library). `cpp/cmake/Dependencies.cmake` and `cpp/CMakeLists.txt` no longer use HDF5 or HighFive. `gpu_preprocessing/prep/sources.py` now has `NpySource` and keeps `AnnHdf5Source` for the raw ann-benchmarks files. Call sites in `laef_build_index`, `laef_profile` and `laef_bench` use the folder API. Ada-ef sample filter (`usable_calibration_rows`), `select_rows`, `write_samplings`, and the price-list fields `ladder_ratio` and `n_dropped_zero_recall` are added.
- Not done yet: `apps/laef_calibrate.cpp` still calls the old file paths (full rewrite pending: tables dir, valid samplings, sample filter). Python `prep/convert.py`, `prep/calibration.py`, `prep/manifest.py`, `run_preprocess.py` still write and read `dataset.h5` and `gt/test_gt.h5`. Colab check cell still uses h5py on `gt/test_gt.h5`. `configs/datasets.yaml` comment and the plan document still mention HDF5.
- Open decision for the user: the two raw ann-benchmarks downloads (`glove-100-angular.hdf5`, `deep-image-96-angular.hdf5`) are HDF5 upstream. The plan keeps h5py for converting them once, in Python only.
- Nothing has been compiled or run.

## Pass 3 (HDF5 to NumPy plan update and the remaining stale items)

Static review only. Nothing was compiled or run (the user asked for no code execution; WSL build is the next step).

Plan (IMPLEMENTATION_PLAN.md)
- Every HDF5 reference now describes the `.npy` layout: §1.3 row 12, §3 tree, §4 tree and conventions, new §4.3 decision, §5 note on the two ann-benchmarks HDF5 downloads, §6 P0.1/P0.3, §7 P1.3/P1.5/P1.6/P1.9 and §7.1, §8.1, P2.4, §11.2.
- P0.1 apt list: `python3-venv` plus uv (uv provides Python 3.11). The old `python3.11-venv` package is not in Ubuntu 22.04.
- §8.1: the fake backend is `100 + ef` dc with no jitter, as implemented.
- Status rows updated for sources, CMake, apps, engine, run_experiments, shell scripts and the §13.1/§13.3 tests.

Code
- `apps/laef_calibrate.cpp` rewritten on the folder API. It writes `calibration/samplings_valid.bin` from `usable_calibration_rows` (finding 5), builds the adapter and runs the sweep from that file, takes `--tables-dir`, and records `ladder_ratio` and `n_dropped` in the price list.
- `apps/laef_bench.cpp`: `--price-list`. `apps/laef_profile.cpp`: `--n` alias. `apps/laef_build_index.cpp`: help text.
- `cpp/include/laef/ann_backend.h`: the comments for `usable_calibration_rows` and `build_ef_adapter` were attached to the wrong declarations. Fixed.
- `scripts/run_experiments.py`: `calibration.ladder_ratio` builds one price list per ratio under `results/<exp>/_tables/<dataset>/` (cached per invocation). Other `calibration.*` axes are still skipped. Dataset check uses `train.npy`.
- `scripts/setup_wsl_ubuntu.sh`, `build.sh`, `prepare_dataset.sh`, `run_pipeline.sh` were stubs and are now written. `prepare_dataset.sh` verifies `manifest.json` before it copies anything.
- `gpu_preprocessing/colab_preprocess.ipynb`: check cell reads `gt/neighbors.npy` (numpy) and `gt/gt_meta.json`. Markdown and settings mention `.npy`. JSON parses.
- `configs/datasets.yaml`, `gpu_preprocessing/requirements.txt`: comments updated (h5py only for the raw downloads).
- READMEs: root and `gpu_preprocessing/README.md` rewritten (ASCII, `bash scripts/...`). `cpp/third_party/README.md` rule on `hnswlib` includes corrected. Placeholder docstrings in `prep/__init__.py` and `analysis/laef_analysis/__init__.py` fixed. `cpp/tests/test_formats.cpp` is a placeholder note (merged into `test_backend_parity.cpp`).

Tests (written by a sub-agent, reviewed against the headers by grep)
- `cpp/tests/test_sensor_window.cpp`, `test_fixed_aimd.cpp`, `test_allocators.cpp`, `test_engine_qps.cpp`. CMake: `test_engine_qps` target linked to `laef_core laef_fake`.
- Identifiers and field orders checked against the headers. The expected values in `test_fixed_aimd` and `test_allocators` were re-derived by hand from the header formulas.
- `test_engine_qps`: k = 40 (not in `qps_matrix.yaml`), analytic mu_sat, documented timing tolerances. The five real-dataset cases skip (the real engine run is not wired into the test).

Open
- Not compiled. Timing-sensitive assertions (light-load p95, overload p95, step window fractions, generator lag) may need tolerance changes on WSL.
- Decision for the user: keep h5py for the two ann-benchmarks HDF5 downloads (current plan), or find `.npy` copies.

## Pass 4 (dataset selection: sets that need no extra processing)

Source check: the Ada-ef paper (arXiv 2512.06636, section 7.1) uses GloVe-100, DeepImage, MS MARCO V1/V2.1, LAION I2I/T2I (30.6M vectors each, CLIP) and synthetic cluster sets. ann-benchmarks (`datasets.py`, README table) lists the rest.

Added to `configs/datasets.yaml` (existing `ann_hdf5` path, provided ground truth, metric cd, no code change):
- `coco-i2i-512-angular`: CLIP image queries, 113,287 x 512, 10,000 queries. In-distribution control; small analogue of LAION-I2I.
- `coco-t2i-512-angular`: CLIP caption queries against the same kind of base, 113,287 x 512, 10,000 queries. Cross-modal shift; small analogue of LAION-T2I.
- `lastfm-64-dot`: recommender factors, 292,385 x 65, 50,000 queries. Stored as angular upstream.

Checked on the downloaded HDF5 files (WSL scratch folder, outside the project):
- Keys `train`, `test`, `neighbors`, `distances`; attribute `distance` = angular. COCO train/test are float16; the reader casts chunks to float32.
- Neighbour ids in range; distances ascending in every row; no NaN; no zero-norm vectors (COCO and Last.fm).
- `prep.registry.load_registry` loads all 8 ids and builds the three new specs.

Rejected: `nytimes-256-angular`. It has 239 zero-norm training vectors, 9 zero-norm queries and 900 NaN ground-truth distances, so it needs cleaning or GPU ground truth.

Plan and other files updated: §5 table and text, §5 "not included" paragraph (reasons), §7 P1.10 list, §4.3 and §1.3 row 12 and tech stack wording (HDF5 downloads), `colab_preprocess.ipynb` (DATASETS and PROVIDED_GT), `gpu_preprocessing/README.md`, `requirements.txt` comment, `datasets.yaml` header.

Not changed: experiment YAMLs and `configs/tests/qps_matrix.yaml` (they choose runs; the C++ real-dataset QPS cases still skip). Not verified yet: convert, ground truth and calibration on these files (not run).

## Pass 5 (new datasets in the test matrix and analysis)

Files updated for `coco-i2i-512-angular`, `coco-t2i-512-angular` and `lastfm-64-dot`:
- `configs/tests/qps_matrix.yaml`: three real-dataset rows (threads 4, 10,000 queries, rho 0.5 and 1.5). The file now has 10 rows.
- `cpp/tests/test_engine_qps.cpp`: three real-dataset cases, and the comment now says rows 3-10. All 8 real-dataset cases are registered. They skip because the artifacts are not built, and the target rebuilds cleanly.
- `IMPLEMENTATION_PLAN.md` §13.3 matrix block mirrors the YAML. §14.1 E1 row now says 7 datasets.
- `configs/experiments/e1_load_sweep.yaml`: E1 covers all analysis datasets, so the sweep grows from 576 to 1008 runs. E2, E3 and E4 keep their targeted subsets.
- `analysis/notebooks/01_calibration.ipynb`: `DATASETS` includes the three new sets. Datasets without calibration tables are skipped.

Verified: the QPS matrix YAML parses to 10 rows, the calibration notebook JSON is valid, and the QPS test target builds.

## Pass 6 (rho only for experiments; window and arrival-model review)

Decision: the engine takes absolute arrival rates (qps) only. Experiments may write `rho` (a multiple of the profiled saturation throughput mu_sat(C)).

Code:
- `WorkloadCfg` and `PhaseCfg` no longer have `rho`, `rho_lo` or `rho_hi`. `resolve_rates` is replaced by `check_rates`, which rejects missing or invalid rates. `laef_bench --rho` is removed.
- `scripts/run_experiments.py`: `to_absolute_rates` converts `rho` (and step-phase and mmpp2 `rho_lo`/`rho_hi`) to `qps` with `saturation.mu_sat_qps` from `profile_C{threads}.json`. The converted config is written as `run_config.json`, and `run_meta.json` keeps `rho` as the experiment label. Checked: rho 0.5 at mu 4000 gives 2000 qps; step 0.3 and 1.5 give 1200 and 6000; mmpp2 0.4 and 1.6 give 1600 and 6400.
- `configs/defaults.yaml`: `workload.rho` removed and `workload.qps` required. Window fields documented against Main Flow §3. `config.h` header corrected: nothing parses defaults.yaml.

Window (no engine change): Main Flow §3 and Budget B §3 both say "every ~500 completed queries, or every 20 ms provided at least 50 new completions". The p95 is over the last 500 completions. `sensor_window.h` implements exactly this. The 20 ms trigger only fires above 2,500 QPS with the defaults.

Fixed in Main Flow §8 (md and html, at the user's request). The sentence said recovery takes about 26 windows, "about half a second". At the 600 QPS of that example each window is about 0.83 s, so recovery takes about 22 s. Half a second holds only above 2,500 QPS.

Arrival model: Main Flow and Budget B name no arrival distribution. `poisson` (exponential gaps at rate qps) is recorded as an assumption in plan §11.1. It is the testing profile for QPS for now, to be discussed later.

Tests:
- Build passes.
- Full run with one test at a time: 88 of 89 pass. `FakeInfeasibleStep.FloorModeClearsAfterTheLoadDropsAndBudgetRecovers` fails: the recovery tail is too short for B to climb back.
- Under four parallel test processes, timing checks fail from CPU contention. `LightLoadNeverFloorsAndKeepsPreferredEf/0.3`, `FeasibleOverloadKeepsBudgetInsideTheBandInEveryWindow/1.5` and `FeasibleOverloadCutsEfAndKeepsP95NearSlo/1.5` change between runs. Recommendation: run `test_engine_qps` with `RUN_SERIAL`.
## Pass 7 (COCO ground truth: provided file replaced by generated GT)

Finding. The Colab verification (`--verify-provided 500`) failed on both COCO sets. Recall of the provided GT against exact search was 0.9888 (I2I) and 0.9938 (T2I). The threshold is 0.999, which allows at most 5 wrong neighbours in 5,000 ids.

Checks (raw HDF5 files, with the Colab GPU result checked alongside):
- The Colab GPU result matches exact float64 cosine search on all 500 sampled queries (recall 1.0 on I2I). The GPU path is correct, so the disagreement is in the provided file.
- The provided file ranks by cosine. Ranking by raw dot product (0.41 on I2I, 0.48 on T2I) or raw L2 (0.84 on I2I, 0.07 on T2I) agrees far less.
- No query is an exact base row, so "query inside the base" does not explain the gap.
- Rounding the vectors to float16 gives the same recall (0.9888 and 0.9938), so vector precision is not the cause.
- The provided distances differ from exact cosine distances by up to 7.7e-4 (I2I) and 3.3e-4 (T2I). The swapped neighbours sit within about 5e-4 of each other, so the order in the file is wrong at that scale. On I2I, 35 of the 56 wrong neighbours are closer to the query than the worst provided neighbour, so the provided list is not an exact top-100.
- The exact cause of the error in the provided file is not known. This project did not produce that file.

Decision: COCO-I2I and COCO-T2I use generated GT (`gt_provided_valid: false`). It is computed on the GPU with the cosine metric the engine uses. The provided top-100 stays in `data/raw/` for reference. The cost is that recall numbers are not directly comparable with runs that used the provided file.

Changed files:
- `configs/datasets.yaml`: `gt_provided_valid: false` for both COCO sets.
- `gpu_preprocessing/colab_preprocess.ipynb`: COCO removed from `PROVIDED_GT`, DATASETS comments and section 5 notes updated. On Colab, delete `laef_artifacts/coco-*/gt/` on Drive before rerunning, because the gt step skips datasets whose `neighbors.npy` already exists.
- `IMPLEMENTATION_PLAN.md` §5: intro sentence and the two COCO rows.

Not yet verified: glove and deep-image (`--verify-provided 500`). LastFM passed at recall 1.0000.
## Pass 8 (local work limited to the split; GPU steps on Colab only)

Decision: the serving machine rebuilds only `train.npy`, `test.npy` and `holdout.npy` (download and convert), then checks Colab's fingerprints and copies the small files. The `gt`, `calib`, `stats` and `manifest` steps need a CUDA GPU and run on Colab only.

Changes:
- `gpu_preprocessing/run_preprocess.py`: `--device` accepts only `cuda`. The four GPU steps refuse to run without CUDA (`_cuda_available`) and point to `--steps download,convert`. The functions are unchanged, because Colab uses them.
- `scripts/prepare_dataset.sh`: runs `--steps download,convert` with no device flag.
- `gpu_preprocessing/colab_preprocess.ipynb`: `synthetic-small` added to `DATASETS`, so the GPU generates it like the other synthetic set.
- `gpu_preprocessing/README.md`, `IMPLEMENTATION_PLAN.md` §5 and the GPU-preprocessing row: wording updated (five generated sets, GPU steps on Colab only).

Provenance: the `gt`, `calibration`, `stats` and `manifest` files of `synthetic-small` in `data/artifacts/synthetic-small/` were made on the CPU before this change. They are kept, not regenerated. A later Colab run would make a separate copy in `laef_artifacts/synthetic-small/`.

Checks:
- Unit tests: 14 passed, 1 skipped (the torch test, because torch is not installed locally), the same as before.
- Guard: `run_preprocess.py --steps gt` on this machine stops with the expected message.
- `prepare_dataset.sh` with the new command on lastfm-64-dot: manifest OK, files copied.
## Pass 9 (the engine takes T_SLO in ms only)

Decision: the engine never computes T_SLO from a multiplier. `laef_bench` takes `slo.ms` (or `--slo-ms`) and stops with a message if it is not set. The multiplier stays in the experiment files and in `defaults.yaml`. `run_experiments.py` turns it into ms from the profile's idle S_p95 before the run (`resolve_slo`), the same way rho becomes qps. The Ada-ef reference comparison sets `ms` directly from the reference p95.

Changes:
- `cpp/include/laef/config.h`: `SloCfg` holds `ms` only. `mode` and `multiplier` are no longer read or written.
- `cpp/apps/laef_bench.cpp`: `--slo-mult` removed. `t_slo_s = slo.ms * 1e-3`. The idle S_p95 is still read, but only to start the controller's service-time estimate, not for T_SLO.
- `scripts/run_experiments.py`: `resolve_slo` converts a multiplier to ms; `run_config.json` holds only `ms`.
- `configs/defaults.yaml`: `slo.ms` is required; `multiplier` is documented as experiment-only.
- `IMPLEMENTATION_PLAN.md`: the SLO decisions row, §14.1 defaults and the §15 table.

Checks:
- The C++ tree builds with no errors (all targets).
- `laef_bench` without an SLO exits with status 1 and the message "T_SLO is required".
- `laef_bench --slo-ms 4.0893` runs; `config.json` holds `slo: {ms: 4.0893}` and `t_slo_s = 0.0040893`.
- `run_experiments.py` with `slo: {multiplier: [5]}` on synthetic-small writes `ms = 4.089298` (5 × idle S_p95 0.81786 ms), the same value the engine uses.
- The full C++ test suite was not run in this pass.
## Pass 10 (one SLO study; saved analytics are 1 s time bins)

Decision: the experiments are one study, `scripts/run_slo_study.py`. Stage 1 runs stock Ada-ef at loads 0.5, 1.0 and 1.5 x mu_sat for 120 s. Stage 2 runs LA-ef with the naive cut (`laef_uniform`) and the proportional cut (`laef_wf`), P-AIMD controller, at T_SLO = 0.95, 0.9, 0.8 and 0.75 x that baseline p95. It covers synthetic-small, lastfm-64-dot and coco-i2i-512-angular, the three with 120 s profiles.

Removed: the E1–E4 grids, `smoke.yaml`, `pilot_small3.yaml`, `ref_ada_ef.yaml`, `slo_vs_ada.yaml`, `make_slo_vs_ada_config.py` and `compare_to_ada.py`, together with every result folder from earlier runs. The plan's §14.1 is marked superseded.

Saved analytics: one `intervals.csv` per run with 1 s time bins. Each bin holds arrivals, completions, p50 and p95, SLO attainment, requests in the system, ef ratio, recall, floor-mode fractions and the controller state. Time bins rather than batches of queries, so every run has the same time axis, and the bins stay regular under overload. A 1 s bin holds about qps completions, so the batch size follows the load automatically. At the lowest load in the study, there are still more than a thousand completions per bin, enough for a stable p95.

Per-query records (`queries.csv`) are read once to build the bins and then deleted. `--keep-queries` keeps them. The engine still holds the per-query records in memory during each run, but they are no longer saved in bulk. Peak disk use during the study is about the size of all the per-query files together, because they are deleted only after stage 2, roughly 6 to 7 GB.

Checks: a 5-second test on synthetic-small (27 runs) finished. No `queries.csv` was left, 27 `intervals.csv` files were written, and `comparison.csv` and `summary_all.csv` were built. The full study has not been run. Expected time is about 2.5 to 3 hours.

## Pass 11 (two steps: stock Ada-ef reference first, then LA-ef; static review only)

Decision: the stock Ada-ef p95 at each load is its own experiment, and it runs first. Pass 10 described one command that ran both stages, with the reference inside the study file. That design is superseded.

- `configs/experiments/ada_reference_120s.yaml`: stock Ada-ef at rho 0.5, 1.0 and 1.5 x mu_sat, 120 s, on synthetic-small, lastfm-64-dot and coco-i2i-512-angular (9 runs). `scripts/run_ada_reference.py` runs it and writes `results/ada_reference_120s/ada_p95.csv`: one row per dataset and load, with the p95, the throughput, the p95 of each half of the run, and the backlog when the arrivals stop.
- `configs/experiments/slo_study_120s.yaml`: LA-ef only, 72 runs (3 datasets x 3 loads x {laef_uniform, laef_wf} x fractions 0.95, 0.9, 0.8, 0.75). Each section names its reference folder in `slo.reference`, and `run_experiments.py` reads that p95 to set T_SLO. `scripts/run_slo_study.py` runs it and stops with a message if the reference summaries are missing.
- `scripts/study_common.py`: input checks, per-run interval statistics, the half-run p95, and the summary writer. Both scripts use it.
- `scripts/run_experiments.py`: `--dry-run` no longer writes run_config.json or run_meta.json. The missing-reference message names `run_ada_reference.py`.
- Removed: `configs/experiments/slo_check_5s.yaml`. Its reference ran inside the same file, which the two-step flow no longer allows.
- Docs: `commands.md` lists the two commands in order. The README and the plan's directory tree no longer name deleted experiment files.

Status: no runs were made for this pass, as the user asked. The scripts were checked by reading only, so the first real run may show bugs.

## Pass 12 (the SLO multiplier is removed)

Decision (user request): no experiment uses a multiplier. The engine already read T_SLO in ms only, so the change is in the runner and the configs. Pass 9's rule that the multiplier stays in the experiment files is superseded.

Changes:
- `configs/defaults.yaml`: `slo.multiplier` removed. The `slo.ms` comment says experiments set it per run.
- `scripts/run_experiments.py`: `resolve_slo` and the `idle_p95_s` read are removed. A section needs `slo: {ms: X}` or `slo: {reference: ..., fraction: [...]}`. A section with neither is an error, and it no longer defaults to a multiplier of 5. `slo_mult` is gone from `Run`, `run_meta.json` and `summary.csv`. Run ids use `f<fraction>` for reference sections and `fixed` otherwise. `run_config.json` holds `ms` and qps only.
- `scripts/interval_stats.py`: `meta_fields` no longer reads `slo_mult`.
- `configs/experiments/ada_reference_120s.yaml`: `slo: {ms: 10}` as a placeholder. Stock Ada-ef has no controller and ignores T_SLO. The value only sets slo_attainment and goodput in that run's summary, and the study does not use either.
- `IMPLEMENTATION_PLAN.md`: the §11.4 column list, the laef_bench CLI line, the status row and the SLO decision row no longer mention the multiplier.

Not changed: the analysis notebooks in `analysis/notebooks/` still read `slo_mult` and the deleted experiments. They are stale and need a rewrite.

Status: nothing was run. The code was checked by reading only.

## Pass 13 (constant-rate arrivals; T_SLO from the profile p95; one study command)

Decision (user request): the SLO study uses evenly spaced arrivals and takes T_SLO as a fraction of each dataset's profile p95. Pass 11 (two steps, with a per-load Ada-ef reference) is superseded.

Changes:
- `cpp/src/workload.cpp`: new `constant` profile, arrival i due at i / qps, with `n_queries` or `duration_s`. `check_rates` rejects qps <= 0. Declared in `workload.h`, `config.h` and `defaults.yaml`. No unit test was added.
- `scripts/run_experiments.py`: `workloads` accepts `profile: constant`. `slo: {fraction: [...]}` sets T_SLO = fraction x `saturation.s_p95_s` of the dataset's `profile_C{threads}.json`, converted to ms in `resolve_slo` before the engine starts. `slo.reference` and `reference_slo_points` are removed. `run_meta.json` and `summary.csv` gain `profile_p95_ms`.
- `configs/experiments/slo_study_120s.yaml`: 81 runs. Per dataset, stock Ada-ef at three loads (fraction 1.0) and LA-ef (2 allocators x 4 fractions x 3 loads). Ada-ef ignores T_SLO, so it runs once per load.
- `scripts/run_slo_study.py`: one command. `comparison.csv` pairs each LA-ef run with the Ada-ef run at the same dataset and load, and carries `generator_lag_p99_us`. `scripts/study_common.py` keeps the shared helpers.
- Removed: `configs/experiments/ada_reference_120s.yaml`, `scripts/run_ada_reference.py` and `results/slo_check_5s/`.

Progress display: `run_slo_study.py` counts the runs first (81, minus any that already have `summary.json`), then shows a bar with percent, runs done, elapsed time, an ETA and the time in the current run. The ETA is the average time of the runs executed so far times the runs left, so it reads `?` until the first run finishes. The runner and engine output goes to `results/slo_study_120s/run.log`. Lines with an error, failure or warning also appear on screen. On a terminal the line redraws in place. In a redirected log, one line is printed per finished run. Checked with a stand-in child process that prints runner-format lines, in both modes, and the run count against the real file. No engine ran.

What the profile p95 is: the p95 of the service time in the closed-loop saturation run. It holds no queue wait, no generator thread and no queue lock. The engine's open-loop latency includes wait, so a run can miss T_SLO from queueing alone. In an earlier 5 s synthetic-small check, stock Ada-ef at 0.5 x mu_sat had a mean wait of 54 ms and a mean service time of 0.55 ms. The profile p95 values are 0.44 ms (synthetic-small), 0.33 ms (lastfm-64-dot) and 1.20 ms (coco-i2i-512-angular).

Generator check (standalone, queue and pacing loop copied from `engine.cpp`, workers spin only): one generator thread pushed 26,490 qps with the mean rate held. Lag p99 was 30-80 µs. In a real LA-ef run at 26,486 qps on lastfm-64-dot (10 s) the lag p99 was 2.65 ms, against 165 µs for stock Ada-ef. Mean lag, maximum lag and the generator finish time are not recorded, so the mean rate for LA-ef at that load is inferred, not measured. Every `comparison.csv` row carries `generator_lag_p99_us`.

Open: on lastfm-64-dot the preferred ef was 10 (= k) for all 265,203 queries in that LA-ef run, so the planner has nothing to cut and the ef ratio is 1.0. The cause is not traced.

Status: the C++ was built and the Python was checked by reading and a dry run. No study run was made.

## Pass 14 (stale artifacts: all 81 runs failed with `key 'adaef' not found`) — superseded by Pass 15

Symptom: the first study run on WSL failed every run in seconds with `[json.exception.out_of_range.403] key 'adaef' not found`.

Cause: the engine sources (`laef_profile.cpp`, `laef_calibrate.cpp`, `price_list.cpp`, `engine.cpp`, `planner.h`, `token_bucket.h`, `adaef_backend.cpp` and the new `artifact_metadata.h`) were edited on 3 Nov by another agent, not in this pass. `laef_bench` now validates compatibility metadata: `prof.at("adaef")` in the profile, `tables/ada_ef_adapter.meta.json`, and `quantile_step` and `ef_upper_bound` in the price list. The artifacts of the three study datasets were made on 1-2 Nov and have none of these.

Fix: run `laef_calibrate` and then `laef_profile --sat-seconds 120` for each dataset (`commands.md`, step 1). The calibration rebuilds the adapter by itself when the metadata file is missing. The profile is measured again, so m_sat and the profile p95 change slightly, and so do T_SLO and the load in qps.

Change: `study_common.check_inputs` now stops before the first run when the adapter metadata file is missing, when the price list lacks `quantile_step` or `ef_upper_bound`, or when the profile has no `adaef` block. The message gives the two commands. Checked once against the current artifacts: it stopped at the coco dataset with that message and started no engine. The failed runs left `run_config.json` and `run_meta.json` without `summary.json`, so the next study run redoes them.

Caveat: earlier findings in this file about the engine (ef never cut, generator lag, service time of LA-ef) were measured on a build from before these edits. They need to be re-checked on the current build.

## Pass 15 (the artifact metadata checks are removed)

Decision (user request): the metadata checks that another agent added to the engine (see Pass 14) are not needed for the study, so they are removed together with the guard that Pass 14 added. They only compared k, M, target recall, quantile step, probe size and ef cap between the artifacts and the run. All study artifacts were built with the defaults. The checks' only effect was to block runs of existing artifacts.

Removed:
- `cpp/include/laef/artifact_metadata.h` (deleted).
- `laef_bench.cpp`: the `prof.at("adaef")` check, the price-list metadata check and the `ada_ef_adapter.meta.json` requirement and check. The price list is loaded without the meta argument. The check that the price list's k equals the run's k stays.
- `laef_profile.cpp`: the sidecar requirement and check, and the `adaef` block in `profile_C{C}.json`.
- `laef_calibrate.cpp`: the sidecar check and write. The adapter is built only when `ada_ef_adapter.bin` is missing. `effective_statics` is computed inline, because the price list still records it.
- `scripts/study_common.py`: the Pass 14 guard (`_regenerate_hint`, the price-list and profile checks, the sidecar in the input list).
- `commands.md`: the step that re-ran calibration and profiling. The plan lines about the sidecar.

Kept: `PriceListMeta` and its fields in `price_list.json`, which the loader reads with defaults, so old price lists load. The existing calibration tables and 120 s profiles work again, and nothing needs regenerating.

Risk accepted: the engine no longer detects an adapter, price list or profile that was made with a different k, M, target recall or quantile step. If those parameters are changed, rebuild the artifacts by hand.

Status: the engine was rebuilt. No study run was made.

## Pass 16 (progress display checked for real; stop handling fixed)

Request: the user ran `run_slo_study.py` and saw no progress. The log showed a run started at 15:23, and no study process was alive afterwards. Run for 8 s on a pseudo-terminal (`script`), 8 s redirected to a file, and 35 s redirected, each stopped with SIGINT via `timeout`.

Results:
- Terminal: the bar is on screen from the first second and redraws every second.
- Redirected: output stayed silent for the whole length of a run (about 2 minutes or more). Added a heartbeat line every 30 s, `... (running)`, besides the line per finished run.
- Stop handling: SIGINT crashed the redirected case with a traceback and left the engine child behind. On a terminal it carried on into the summary step. Fixed: `interrupted` flag, the runner starts in its own session (`start_new_session=True`) and the whole group gets SIGTERM, `run_with_progress` returns 130, and `run_slo_study.py` then skips the summary step. After each test no `run_slo_study`, `run_experiments` or `laef_bench` process was left, and no `summary.json` was written.

Not verified: what the user's own terminal shows. If it does not redraw a line in place, use the `setsid nohup ... > slo_study.log` command and `tail -f slo_study.log`.

Cleanup: `results/slo_study_120s/` held no finished run (0 `summary.json`), only the run and meta files of the failed attempt and its log, so it was removed.
