# LA-ef (Load-Aware Ada-ef): Implementation Plan

**Audience:** a coding agent implementing this project end-to-end.
**Source designs:** `../Main Flow.md` (Steps A–F) and `../Budget B calculations.md` (Step D, P-AIMD thermostat). Read both before starting. Section numbers like "MF §5" and "BB §1" refer to them. Both docs were **revised on 2026-11-01** to match this plan, and each starts with a revision note. If the docs and this plan disagree, this plan wins.
**Base code:** the Ada-ef authors' fork of hnswlib, <https://github.com/chaozhang-cs/hnsw-ada-ef> (SIGMOD '26, arXiv 2512.06636).

> **Design revision 2026-11-04 (read §1A first).** Measurements showed that the post-probe ef cut has almost nothing to cut: on synthetic-small, lastfm-64-dot and coco-i2i, Ada-ef chose `ef_pref = k` for every calibration query, and the 1,025-distance probe was about 99.6% of each query's work. LA-ef v2 therefore makes the **probe length L** the main budget lever, with a separate Ada-ef table for each L, and keeps the existing post-probe ef cut as a second stage. §1A is the specification and the change list for the implementing agent. Where §1A and an older section disagree, §1A wins.

---

## 0. Rules for the implementing agent

1. Implement the phases in order (§17 has the checklist). Every task lists its **files**, **spec** and **acceptance** criteria. A task is done only when its acceptance criteria pass.
2. **Never edit `cpp/third_party/hnsw-ada-ef/`.** It is a pinned git submodule (commit `ed463f9993868f7ecc7c103920644e7f94abb377`). Everything we need is written in our own files, and that works because every member of `hnswlib::HierarchicalNSW` is `public`.
3. **Single-translation-unit rule (important).** Ada-ef's headers (`hnswlib/hnswalg.h` → `distribution.h`, `adaptive_ef.h`) define non-`inline` free functions. If two `.cpp` files in one binary include them, linking fails with "multiple definition" errors. Therefore **only `cpp/src/adaef_backend.cpp` may include any `hnswlib/*.h` header.** All other code talks to it through the `laef::AnnBackend` interface (§8.1). The only exception is `tests/test_backend_parity.cpp`, which `#include`s `adaef_backend.cpp` as a unity TU so it can call the authors' functions directly.
4. Every tunable comes from config (§15) and can be overridden on the CLI. Do not hard-code SLOs, thread counts, QPS or dataset paths.
5. Use the units consistently: time in **seconds (`double`)** inside controllers, **nanoseconds (`int64_t`, `steady_clock`)** for timestamps, work in **distance computations (dc)**, and B in **dc/s**.
6. Python is used only for data preparation, GPU work, experiment orchestration and analysis. Everything on the serving hot path is C++20.

---

## Implementation status (updated as work lands)

**Legend:** ✅ done and tested · 🟡 partial · ⬜ not started. Re-run the listed tests after any change.

| Item | Status | Files | Verified by |
|---|---|---|---|
| Ada-ef binary formats (samplings.bin, estimator.bin) | ✅ | `gpu_preprocessing/prep/adaef_formats.py` | `test_formats.py`; and the **authors'** C++ deserializers read the Python fixtures (`AdaEfFormats.*` in `test_backend_parity`) |
| Exact k-NN GT, streamed, torch (cuda/cpu) + numpy fallback, fp16-rerank option | ✅ | `prep/ground_truth.py` | `test_ground_truth.py` (cd/ipd, odd chunk sizes, both backends) |
| Estimator stats (mean/cov, float64, streamed) | ✅ | `prep/estimator_stats.py` | `test_ground_truth.py::test_stats_*` vs `np.cov` |
| VectorSource protocol, ArraySource, NpySource, AnnHdf5Source, ParquetSource | 🟡 rewritten, not run | `prep/sources.py` | sorted takes (h5py needs increasing indices); `NpySource` reads our `.npy` files through a memmap; `AnnHdf5Source` is only for the ann-benchmarks downloads; Parquet read in record batches; no `isinstance` on a Protocol |
| registry / download / convert / synthetic / calibration / manifest / CLI | 🟡 rewritten, not run | `prep/*.py`, `run_preprocess.py` | rewritten in the verification pass: path layout (train.npy and test.npy stay local), signatures, streaming writes, provided-GT handling; re-run `pytest` and a small dataset on WSL |
| **Colab notebook** | ✅ written, not run | `gpu_preprocessing/colab_preprocess.ipynb` | drives `run_preprocess.py` with the flags in §7.1; checks every output file |
| **Analysis notebooks E0–E5** | ✅ written, not run | `analysis/notebooks/01_calibration … 04_ablations.ipynb` | self-contained (pandas + matplotlib); read only the files in §11.4; save figures to `results/figures/<exp>/` |
| Controllers: P-AIMD, fixed-AIMD, static-capacity (+ shared limits/clamp) | ✅ | `include/laef/controllers/*.h` | `test_p_aimd` (incl. MF §8 Phase 3 → B = 2.4M, floor mode) |
| controller_factory / allocator_factory | ✅ | `controllers/controller_factory.h`, `allocators/allocator_factory.h` | take the config structs from `config.h`; `make_controller` returns `nullptr` for `none` |
| `config.h` (RunConfig + JSON merge/serialize + policy presets) | ✅ | `include/laef/config.h` | mirrors `defaults.yaml`; `merge()` applies a `"policy"` preset first, then explicit keys |
| `types.h`, `telemetry.h`, `floor_mode.h` | ✅ | `include/laef/` | QueryRecord / `now_ns` / `recall_at_k`; EWMA telemetry seeded from the profile; `floor_mode_required()` predicate |
| Token bucket (cap 0, −D floor, debt forgiveness, single-lock settle) | ✅ core-tested | `token_bucket.h`, `planner.h`, `engine.cpp` | `test_token_bucket` covers clipped reservations and the MF §8 grace period |
| Sensor window (exact nearest-rank p95, close rules) | ✅ | `sensor_window.h` | `test_token_bucket` (SensorWindow.*) |
| Price list: envelope, grouping, interpolation, steps, water level | ✅ | `price_list.h` | `test_price_list` |
| price_list.json load/save | 🟡 written, not compiled | `include/laef/price_list_io.h`, `src/price_list.cpp` | rewritten on the real `PriceList` API (public groups, `finalize()`); steps are derived, not stored |
| dataset_io, workload, results_writer, fake_backend | 🟡 written, not compiled | `src/*.cpp`, `include/laef/{dataset_io,workload,results_writer,fake_backend}.h` | `workload.h` was an empty stub before; `rho` is converted to qps by `run_experiments.py`, so the engine takes qps only; step phases start at their boundary; `pick_queries` always returns n_queries rows |
| Step A driver: preference pass, ladder sweep, groups, envelopes, static ef | 🟡 written, not compiled | `include/laef/calibration.h`, `src/calibration.cpp`, `apps/laef_calibrate.cpp` | the query_holdout zero-recall filter (finding 5) is in `usable_calibration_rows`, and `laef_calibrate` writes `calibration/samplings_valid.bin` from it; base_sample is unaffected |
| apps (build_index, calibrate, profile, bench) | 🟡 syntax-checked, not linked | `apps/*.cpp`, `include/laef/config.h` | saturation tail seeding is implemented; full link requires Eigen/Boost |
| engine (generator, workers, controller) | 🟡 core-built, Ada-ef integration pending | `include/laef/engine.h`, `src/engine.cpp` | core-only build and fake QPS tests run; real backend still needs full Ada-ef build |
| `analysis/laef_analysis/*.py` | ✅ | `analysis/laef_analysis/` | io, metrics, plots |
| `scripts/run_experiments.py` | 🟡 rewritten, not run | `scripts/` | reads the real YAML layout (`workloads`, `repeats`, `slo.ms` or `slo.fraction`, `base_policy` + `axes`); `calibration.ladder_ratio` builds one price list per ratio under `results/<exp>/_tables/`; other `calibration.*` axes are skipped because they need new samplings |
| Shell scripts (setup, build, prepare_dataset, run_pipeline) | 🟡 written, not run | `scripts/*.sh` | `prepare_dataset.sh` checks the manifest before it copies anything; run them with `bash scripts/<name>.sh` (the files carry no executable bit from Windows) |
| Allocators (passthrough, uniform_cut, water_filling, shared bucket check) | ✅ | `allocators/*.h` | `test_price_list` (Allocators.*) |
| Plan snapshot + store, Planner glue (reservation, settle, floor mode) | ✅ core-tested | `plan_snapshot.h`, `planner.h`, `engine.cpp` | `test_price_list` covers complete and incomplete settlement; `test_token_bucket` covers clipped reservations |
| AnnBackend: planned search (probe + hook), fixed search, build_index, build_ef_adapter, CPU estimator | 🟡 changed, not full-built | `ann_backend.h`, `src/adaef_backend.cpp`, `src/engine.cpp` | incomplete probes now expose measured work; full Ada-ef compile/parity test is blocked by missing Eigen/Boost |
| CMake (header-only core + backend + tests; offline dep overrides) | 🟡 written, not configured | `cpp/CMakeLists.txt`, `cmake/Dependencies.cmake` | fetches nlohmann_json and CLI11 (were missing) and GoogleTest; no HDF5 or HighFive; apps and binaries go to `build/bin` |
| Unit tests for §13.1 and §13.3: test_sensor_window, test_fixed_aimd, test_allocators, test_engine_qps | 🟡 written, not compiled | `cpp/tests/*.cpp`, `cpp/CMakeLists.txt` | fake-backend QPS cases plus the window, controller and allocator checks; the five real-dataset QPS cases skip because the real engine run is not wired into the test yet |
| **v2: probe-length lever (§1A)** | ⬜ not started | see §1A.3 (C1–C13) | §1A.8 (minimal checks); checklist §17.1 |

**Status legend addition:** "written, not compiled/run" means the code exists but was never built or executed. The first step for the next agent is to build and test it on WSL, then fix it as needed.

**Build and test on WSL / Linux (the only target).** All Windows-only tooling used earlier was removed: the Windows venv, the temporary dependency clones and build folders in `%TEMP%`, and the MSYS2 HDF5 package. The tests marked ✅ passed before that cleanup, with GCC 13. Re-run them on WSL with system packages:
```bash
scripts/setup_wsl_ubuntu.sh                     # apt: build-essential cmake libeigen3-dev libboost-dev; uv
uv venv .venv && uv pip install -p .venv numpy h5py pyyaml pytest
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
.venv/bin/python -m pytest gpu_preprocessing/tests -q                          # expect 23 passed
git clone https://github.com/chaozhang-cs/hnsw-ada-ef cpp/third_party/hnsw-ada-ef && git -C cpp/third_party/hnsw-ada-ef checkout ed463f99
cmake -S cpp -B build && cmake --build build -j && ctest --test-dir build      # expect 26 core + 5 parity tests
```
Header-only Eigen / Boost.Math checkouts also work instead of apt (`-DLAEF_EIGEN_INCLUDE=…`, `-DLAEF_BOOST_INCLUDE=…/include`), as does `-DLAEF_WITH_ADAEF=OFF` for the core-only build.

### Findings during implementation (already reflected in the sections below)

1. **The backend TU has no HDF5.** `build_index` and `compute_estimator_cpu` take a row-major `float*` (n × d). The app loads `train.npy` through `dataset_io` and passes the pointer (§8.1).
2. **Testing the backend against the authors' code needs a unity TU.** `test_backend_parity.cpp` `#include`s `src/adaef_backend.cpp` and does **not** link `laef_adaef`, so the Ada-ef headers appear exactly once. That test also replaces the planned standalone `test_formats.cpp` (§13.1).
3. **`ndis` counts actual distance calls**, including +1 for the entry point and +1 for the layer-0 restart, and skips visited nodes. Ada-ef's `metric_distance_computations` adds whole neighbour-list sizes instead; it is a shared atomic and also overcounts.
4. **The probe often is the whole search at small ef.** On the 6k × 32 parity fixture, `ndis == probe_ndis` for many queries at ef = k = 10: the ~1025-dc probe already exhausts the search. Consequences:
   - c̄(k) can be close to w̄_nom for easy queries, so the budget's control range [B_min, B_max] shrinks.
   - E0 must report `probe_ndis / ndis` per dataset.
   - If the range is too narrow on real data, the lever is `adaef.statics_length`. The authors ablate it in `ablation_sampling_size/`; it is a config key, not a code change.
5. **There is a bug in the authors' `RecallEstimator::init`.** Queries with recall < 1e-3 are removed from `recalls`, but the loop then indexes `recalls[i]` with the *original* query index. That misaligns the group statistics (and reads out of range) whenever any calibration query has zero recall at ef = k or 1.5k. `base_sample` calibration includes the query itself as its own neighbour, so recall is ≥ 1/k and the bug is never hit. With `calibration.source = query_holdout`, check that every calibration query has non-zero recall at ef = k, and drop the ones that don't before calling `build_ef_adapter`.
6. **`SnapshotStore` uses a mutex plus `shared_ptr`**, not `std::atomic<std::shared_ptr>`, which requires libstdc++ from GCC 12 (WSL Ubuntu 22.04 ships GCC 11). The minimum is now GCC 11 and CMake 3.22.
7. **Both allocators choose ef from ladder rungs** (plus ef_pref itself as the top rung), so uniform_cut and water_filling are quantised identically. Ladder density is the E4 `ladder_ratio` ablation.
8. **The water level uses the model demand** Σ_g p_g·c̄_g(e*_g) (`PriceList::model_nominal_work`), and the arrival-time reservation is the same number, so the two stay consistent with each other. The measured `nominal_work_mean` is kept for reporting only.
9. **Floor mode runs each query at its group's floor rung** (`PriceList::floor_ef(g)`, always ≥ k). This equals k unless the k rung was dominated (same work as a higher rung) and removed by the envelope.
10. **Windows:** MSVC is unsupported, but MSYS2 MinGW-w64 GCC compiles the authors' code, because GCC accepts VLAs. Final latency numbers should still come from Linux or WSL2.
11. **Python environment:** use `uv` (§2). Installing torch from the PyTorch CPU index alone fails, because its dependencies are not all on that index; add PyPI as an extra index as shown above.
12. **Test and dataset artifacts are `.npy`, not HDF5.** `test_engine_qps` and its real-dataset rows check `train.npy` and `test.npy` (NumPy `.npy`, §4.3). The C++ tests never read or require HDF5. The two ann-benchmarks HDF5 downloads are converted to `.npy` once, by h5py, in the convert step (§7, P1.5).

---

## 1. What we are building (one paragraph)

Ada-ef chooses a per-query `ef` that reaches a declared recall (for example 0.95) on an idle machine. LA-ef adds a closed control loop so that a p95 latency SLO holds under load:
- **Step A (offline):** calibration tables and a "price list" of recall lost per dc saved.
- **Step E (per query):** token-bucket planning in two stages (§1A): choose the probe length L before the search, then trim `ef` after the probe using that L's price list.
- **Step F:** C worker threads execute the searches.
- **Steps B/C:** EWMA telemetry and a sensor window over the last 500 completions.
- **Step D:** the P-AIMD thermostat sets the global work budget B (dc/s). When even ef = k cannot meet the SLO it switches on **floor mode**.

**Hard rule: no query is ever dropped.** Under any load every query is admitted and served. When the budget cannot be met even at ef = k, every query runs at ef = k and the SLO is allowed to deteriorate. Admission control (Direction 4 in the original docs) has been removed.

### 1.1 How the user's decisions are applied

| Decision | How this plan applies it |
|---|---|
| "Naive equal cut vs intelligent proportional cut" | Implemented as two **allocators** (Step E): `uniform_cut` gives every query the same fraction of its preferred work, and `water_filling` cuts the cheapest recall-per-dc steps first (price list). Both share the same bucket, controller and floor mode, so only the cut policy differs. A naive `fixed_aimd` **controller** (constant 10% cut) is also included as a controller ablation against P-AIMD. |
| "Keep scope open for other ways to compute the total budget" | The `BudgetController` interface plus a factory (§10.4). It ships with `p_aimd` (the design doc), `fixed_aimd` and `static_capacity`. Adding a method means one header plus one factory line. |
| SLO 10 ms is only an example | The engine takes T_SLO in ms only (`slo.ms` or `--slo-ms`, required). For a comparison, T_SLO is a fraction of the stock Ada-ef p95 in the dataset's profile (`saturation.s_p95_s`, the closed-loop service-time p95 at μ_sat; `slo.fraction` in an experiment file, resolved to ms by `run_experiments.py`). That p95 has no queue wait in it. Several SLOs can be swept. |
| CPU cores configurable | `--threads C` on every app. Optional `--pin-cores`. |
| GPU preprocessing on Colab T4 | Self-contained folder `gpu_preprocessing/`. It produces the same artifacts whether the ground truth was provided or generated (§4). |
| Use the authors' Ada-ef code | Their `Estimator`, `ApproximatedScoreCalculator`, `EfAdapter` and `Sketch` classes produce the per-query score and preferred ef. Our search function reproduces their `adaptiveSearchBaseLayerST` line for line, but replaces the `sketch->estimate_ef2(score)` call with a planning hook. |
| Few datasets, enough for a good analysis | 4 analysis datasets plus 1 tiny synthetic set for tests (§5). |

### 1.2 Key design decision: where planning happens

> **v2 (§1A):** planning now has a stage before the search: the worker chooses the probe length L from the snapshot just before calling `search_planned`. The post-probe hook described below is stage 2 and uses the price list of the chosen L.

In Ada-ef, the difficulty probe is **not** a separate step. It is the first `statics_length` (≈1025 for M=16) distance computations of the layer-0 search itself. A probe "at arrival" would run that work twice or need a separate thread pool. This plan does the following instead:

- **At arrival (generator thread):** always admit, reserve the nominal work w̄_nom from the bucket, and retain the **actual amount charged** after the `−D` clamp. Arrival bursts therefore drain the bucket at the arrival rate, which is what the MF §8 Phase-2 walkthrough requires. If the clamp forgives part of a reservation, that forgiven part must not be refunded later.
- **At the end of the probe (worker thread, inside the search):** the hook receives Ada-ef's score and its preferred `ef`. It refunds only the actual charged reservation, runs the allocator and bucket check, debits the planned work, and returns the `ef` the rest of the search will use.

As a result the latency identity T = W_q + S holds exactly (the probe is inside S), and no work is duplicated. MF Case 3 (floor work unaffordable) no longer drops the query: it runs at the group's floor rung (or strict `k` if the implementation chooses that convention) with `forced_floor = true`.

### 1.3 Corrections made to the design docs (and why)

| # | Original statement | Problem | Change (in the docs and in this plan) |
|---|---|---|---|
| 1 | MF §0/§5: queries are "classified at arrival (microseconds)" by the Ada-ef probe | In the authors' code the probe is the first ≈1025 dc *of the search itself* (`adaptiveSearchBaseLayerST`) | Reserve at arrival, plan in a hook when the probe completes (§1.2, §10.8) |
| 2 | MF §1: ladder {k…32k}; e*_g = lowest rung with r̄ ≥ 0.95 | Ada-ef's ef for hard queries goes far beyond 32k (cap 5000); the per-query ef should come from the authors' table | e*_i = `Sketch::estimate_ef2(score)`; geometric ladder up to max e*_i (§9) |
| 3 | MF §5 Case 2: "always take the step with the least recall per dc" for one query | A single query can only walk its own ladder, so the global sort has no effect | Water level λ* from the sorted global price list against w_target (§9.5, §10.6) |
| 4 | MF §5.1: per-query share w_avg = (B/B_max)·w̄_nom | Circular derivation; equals the rate balance only at saturation; contradicts the MF §8 walkthrough (which uses B/λ̂) | w_target = B/λ̂ (§10.3) |
| 5 | Bucket upper cap unspecified | MF §8 Phase 2's 6 ms grace equals exactly D, which implies a cap of 0 | `bucket.cap_pos = 0` (§10.5) |
| 6 | Window "500 completions or 20 ms" | At low QPS this gives p95 from ~5 samples | `window.min_samples = 50` (§10.2) |
| 7 | t-digest / P² for p95 | Unnecessary for 500 samples | Exact `nth_element` (§10.2) |
| 8 | Admission Control drops queries (`trigger_admission_control()`, MF Case 3) | User requirement: never drop a query | **Floor mode**: every query runs at ef = k and the SLO deteriorates; bucket debt beyond −D is forgiven (§10.5, §10.7) |
| 9 | BB §2.2 guard: B × room/S_p95 fixes the tail in one step | Water-filling protects hard queries, so S_p95 shrinks more slowly than B | Documented caveat; measured by the guard and allocator ablations (E4) |
| 10 | BB §3: "+2% recovers a 10% cut in ~120 ms" | True only for 20 ms windows | Stated as ≈5.3 windows |
| 11 | Implicit: any metric | Ada-ef models cosine/IP only | Datasets restricted to `cd`/`ipd` (§4.2, §5) |
| 12 | Earlier version of this plan (§4, §7): datasets and ground truth stored as HDF5 (`dataset.h5`, `gt/test_gt.h5`) | Needs the libhdf5 C library and a C++ wrapper whose API changes between versions; attributes are awkward; no gain for whole-matrix and sequential reads | `.npy` matrices with JSON metadata (`train.npy`, `test.npy`, `gt/neighbors.npy`, `gt/gt_meta.json`). h5py only converts the HDF5 ann-benchmarks-format downloads (§4.3) |
| 13 | §1.2 (v1): the only lever is the ef chosen after the probe | Measured on three datasets: `ef_pref = k` for 100% of calibration queries, and the probe is ~99.6% of the work (§1A.1) | The probe length L becomes the main lever, with one Ada-ef table per L; the ef cut stays as stage 2 (§1A) |

---

## 1A. Design revision (2026-11-04): the probe length is the budget lever

This section is the specification for LA-ef v2. The controller (Step D), sensor window (Step C), telemetry (Step B), token bucket, floor-mode rule, workload generator and output writers are **unchanged**. What changes is what the budget buys: first the probe length, then (as before) the ef after the probe.

### 1A.1 Why the design changed

Measured with the v1 engine (`tables/calibration_sweep.csv` and a 10 s lastfm overload run):

| Dataset | `ef_pref = k` | Probe work | Total work | Work after the probe |
|---|---|---|---|---|
| synthetic-small | 100% | 1,124 dc | 1,127 dc | 4 dc (0.3%) |
| lastfm-64-dot | 100% | 1,104 dc | 1,109 dc | 5 dc (0.4%) |
| coco-i2i-512-angular | 100% | 1,119 dc | 1,124 dc | 5 dc (0.4%) |

- Every price list had one rung (`ladder: [10]`) and `steps_sorted: []`, so neither allocator could cut anything.
- Under overload the controller entered floor mode, but the floor ef was the ef already in use, so no query did less work.

**Mechanism.** During the probe, ef is unlimited: every new neighbour is kept until `statics_length = 1 + 2M + (2M−1)·2M = 1025` distances have been collected (a two-hop neighbourhood for M = 16). The probe is sized by the graph degree, not by query difficulty. After it, the result list is trimmed to `ef`, and the stopping distance becomes the ef-th best of ~1,025 points. With ef = k that distance is already tight, so the search stops after a few more distances. With a large ef (hard data, larger k, higher target recall) the post-probe search grows, so the probe share is high only when Ada-ef picks a small ef. Our port matches the authors' code (`test_backend_parity`), so this is the authors' design, not a bug.

### 1A.2 The design in short

1. **Offline:** pick a menu of probe lengths, `adaef.probe_lengths`, default `[1025, 512, 256, 128]`. For **each** L, build the authors' `EfAdapter` with `statics_length = L` (its own score-to-ef table), run the existing preference pass and ladder sweep with that probe length, and write a price list for that L. The work in every price list already includes the probe of that length.
2. **Online, stage 1 (worker, before the search starts):** choose L for this query from the current snapshot. The controller has already computed, for each L, the expected recall if that L's allocator cuts ef until the expected work reaches `w_target = B/λ̂`. It picks the L with the **highest expected recall among those whose expected work fits**. At light load this is always the full L, so LA-ef behaves exactly like stock Ada-ef.
3. **Online, stage 2 (planner hook, after the probe):** unchanged v1 logic, but using **the price list of the L the query actually ran**: refund the reservation, water-filling or uniform cut, bucket check, debit. Stage 2 handles what stage 1 cannot know: the query's own difficulty (hard queries may cost more than the average), difficulty-aware cuts, and the gap between two L values.
4. **Floor mode** runs every query at the smallest L and that L's floor rung. The controller's floor work `c̄(k)` becomes the floor work of the smallest L, which widens the range between `B_min` and `B_max`.

Why stage 2 does not cut twice: stage 1 and stage 2 use the same `w_target` and the same per-L price list. If the chosen L already fits at the preferred ef, its water level is 0 and only the per-query bucket check can act. If stage 1 kept a longer L because ef cuts would close the gap, stage 2 applies exactly those cuts.

### 1A.3 Change summary for the implementing agent

| # | Area | Files | Change |
|---|---|---|---|
| C1 | Config | `include/laef/config.h`, `configs/defaults.yaml` | Add `adaef.probe_lengths` (default `[1025, 512, 256, 128]`) and `probe.enabled` (default `true`). Serialize both. `probe.enabled = false` restricts the menu to the full L (the v1 ef-only ablation). |
| C2 | Backend interface | `include/laef/ann_backend.h` | `search_planned` gains a `uint32_t probe_len` argument. `SearchStats` gains `probe_len`. `BackendConfig` gains `std::vector<ProbeAdapter> probes` (`{probe_len, adapter_path}`); `adapter_path`/`statics_length` remain for the single-L helpers. |
| C3 | Backend | `src/adaef_backend.cpp` | Load one `EfAdapter` + `Sketch` per L (keep each adapter alive while its sketch lives). `planned_search` uses `statics_length = probe_len` and that L's sketch. The score calculator is shared (it takes the number of collected distances). Throw if an unknown L is requested. `build_ef_adapter` is unchanged; calibration calls it once per L. |
| C4 | Fake backend | `src/fake_backend.cpp`, `include/laef/fake_backend.h` | Probe cost = `round(100 · L / 1025)` dc, so the existing tests (L = 1025, probe 100 dc) are unchanged. |
| C5 | Probe menu | new `include/laef/probe_menu.h`, `src/price_list.cpp` | `ProbeMenu` holds one `PriceList` per L (descending L), plus `choose(w_target, kind)`. `PriceList` gains `model_nominal_recall()` and `expected_at(w_target, kind) -> {work, recall}` (§1A.5). Load/save `tables/probe_menu.json`. |
| C6 | Calibration | `src/calibration.cpp`, `include/laef/calibration.h`, `apps/laef_calibrate.cpp` | Loop over `probe_lengths`: per L, build the adapter, sweep, build the price list, write `tables/L{L}/…`. Write `tables/probe_menu.json`, dropping dominated L values (§1A.4). `run_sweep` takes the probe length. |
| C7 | Snapshot | `include/laef/plan_snapshot.h` | Add `probe_idx` (chosen L), `water_level_by_L` and `uniform_fraction_by_L` (one value per menu entry). `reservation` = expected work of the chosen L. `make_snapshot` takes the `ProbeMenu` and the allocator kind. Floor mode sets `probe_idx` to the smallest L. |
| C8 | Planner | `include/laef/planner.h` | New `choose_probe(QueryCtx&)` (stage 1; returns the full L for passthrough/static policies). `QueryCtx` gains `probe_idx`. `on_probe` uses the chosen L's price list and builds a local snapshot copy with that L's `water_level`/`uniform_fraction` before calling the allocator, so **allocators need no change**. `on_incomplete` unchanged. |
| C9 | Engine | `src/engine.cpp`, `include/laef/engine.h` | `EngineInputs` takes `const ProbeMenu*` instead of `const PriceList*`. Worker: `L = planner.choose_probe(qc)` just before `search_planned(q, k, L, …)`. Controller input `floor_work` = smallest L's `model_floor_work()`. Record `probe_len` per query and per window. |
| C10 | Outputs | `src/results_writer.cpp`, `include/laef/types.h` | Append column `probe_len` to `queries.csv` and `timeseries.csv` (at the end, so existing readers keep working). Add `mean_probe_len` and `frac_full_probe` to `summary.json`. |
| C11 | Apps | `apps/laef_bench.cpp`, `apps/laef_profile.cpp` | Bench loads `tables/probe_menu.json`, validates each per-L price list and adapter metadata (`statics_length = L`), passes all adapters to the backend. Profile is still stock Ada-ef at the full L and reads `tables/L{full}/ada_ef_adapter.bin`. |
| C12 | Scripts | `scripts/run_pipeline.sh` | The calibrate skip check uses `tables/probe_menu.json`. Pass `--probe-lengths` through from an env var `PROBE_LENGTHS` (default `1025,512,256,128`). |
| C13 | Tests | see §1A.8 | Kept minimal; correctness is reviewed by logic. |

The old single-L files `tables/ada_ef_adapter.bin`, `tables/price_list.json` and `tables/calibration_sweep.csv` are no longer read. `commands.md` deletes them before recalibrating.

### 1A.4 Calibration per probe length (`laef_calibrate`)

1. Validate `probe_lengths`: sorted descending, unique, every value ≥ 64, and the largest equal to the full length `effective_statics_length(adaef.statics_length, M)` (1,025 for M = 16). The full L is required: it serves stock Ada-ef and light load.
2. Build `calibration/samplings_valid.bin` once (finding 5 filter; it does not depend on L).
3. For each L:
   - Build `tables/L{L}/ada_ef_adapter.bin` with `build_ef_adapter(cfg with statics_length = L, samplings_valid.bin, ef_upper_bound, …)`, and write `tables/L{L}/ada_ef_adapter.meta.json` with `statics_length = L`. Rebuild when the metadata does not match, as today.
   - Run the preference pass and the ladder sweep with probe length L. The ladder goes from k up to the largest `ef_pref` seen **at this L** (capped at `ef_upper_bound`).
   - Write `tables/L{L}/calibration_sweep.csv` and `tables/L{L}/price_list.json` (metadata `statics_length = L`).
   - If the adapter build or the sweep fails at some L (for example too few distances for a stable score), log the reason and drop that L. Never drop the full L.
4. Write `tables/probe_menu.json`:
   ```json
   {"dataset":"lastfm-64-dot","k":10,"M":16,"target_recall":0.95,"full_probe_len":1025,
    "options":[{"probe_len":1025,"price_list":"L1025/price_list.json","adapter":"L1025/ada_ef_adapter.bin",
                "model_nominal_work":1109,"model_floor_work":1109,"model_nominal_recall":0.964,
                "calib_mean_recall_at_pref":0.964,"mean_probe_ndis":1104,"ef_pref_mean":10}],
    "dropped":[{"probe_len":128,"reason":"dominated: nominal work not below L=256"}]}
   ```
5. **Drop dominated L values** in descending order: keep an L only if both its `model_nominal_work` and its `model_floor_work` are strictly below those of the last kept (longer) L. A shorter probe whose table asks for so much ef that it costs as much as a longer probe has no use.

Calibration time grows with the number of L values (one adapter build and one sweep per L).

### 1A.5 Stage 1: choosing L (snapshot-time, controller thread)

For every menu entry `j` and the current `w_target`, `PriceList::expected_at(w_target, kind)` returns the expected `(work, recall)` per query, using group shares `p_g` and the group-median `ef_pref` (the same model as `model_nominal_work`):

- **`water_filling`:** start at `model_nominal_work` and `model_nominal_recall = Σ p_g r̄_g(e*_g)`. If `model_nominal_work ≤ w_target`, return them. Otherwise walk `steps_sorted`, adding `p_g·d_work` to the saved work and `p_g·d_recall` to the lost recall, until `nominal − saved ≤ w_target`. If all steps are taken, return the all-floor point.
- **`uniform_cut`:** `f = min(1, w_target / model_nominal_work)`. For each group, take the largest rung with `work ≤ f·work(g, e*_g)` (else the floor rung), and return `Σ p_g·work` and `Σ p_g·recall`.

`ProbeMenu::choose(w_target, kind)`:
1. If `w_target ≥` the full L's `model_nominal_work`, return the full L (light load: stock Ada-ef).
2. Otherwise, the candidates are the entries whose expected work ≤ `w_target · 1.001`. Return the one with the highest expected recall. On ties (difference < 1e-4), prefer the longer L.
3. If no entry fits, return the smallest L. The bucket check and floor mode handle the rest.

`make_snapshot` stores `probe_idx`, the chosen entry's expected work as `reservation`, and **for every entry** its `water_level(w_target)` and `uniform_fraction`. Floor mode sets `probe_idx` to the smallest L. Policies without a bucket (`ada_ef`, `static_ef`) always use the full L.

### 1A.6 Stage 2: after the probe (planner hook)

Unchanged from §10.8, with two differences:
1. `on_probe` uses `menu[qc.probe_idx].pl`: the query's group, rungs, work and floor all come from the price list of the L it ran with.
2. It reads `water_level_by_L[qc.probe_idx]` and `uniform_fraction_by_L[qc.probe_idx]` from the **current** snapshot. A query that chose L under an older snapshot still uses its own L's price list, with the newest water level for that L.

The `ef_pref` passed to `on_probe` comes from the Ada-ef table for that L, so short probes leave room above k where the table asks for it.

### 1A.7 Engine, worker order

```text
generator: on_arrival → reserve snapshot.reservation (charged amount kept in the task)
worker:    L = planner.choose_probe(qc)                    # stage 1, reads snapshot.probe_idx
           backend.search_planned(q, k, L, hook, ids, &st)  # probe of length L, then hook = stage 2
           if !st.probe_complete: planner.on_incomplete(qc, st.ndis, now)
controller: ControllerInputs.floor_work = menu.smallest().model_floor_work()
            snapshot = make_snapshot(output, menu, kind, λ̂, …)
```

### 1A.8 Validation (kept minimal)

Correctness is reviewed by reading the logic. Only these checks are required:
1. **Build:** core and Ada-ef targets compile on WSL (needs Eigen and Boost: `scripts/setup_wsl_ubuntu.sh`).
2. **`test_backend_parity`** (existing, updated to pass `probe_len = 1025`): proves the full-L path is still identical to the authors' `adaptiveSearchKnn`.
3. **One new unit test** in `test_price_list.cpp`, `ProbeMenu.Choose`: a hand-built two-L menu returns the full L at light load, the shorter L under overload, and the smallest L when nothing fits.
4. **Existing tests** keep compiling and passing after the interface change (fake backend at L = 1025).
5. **One real calibration** (lastfm): `probe_menu.json` lists at least two L values, and the shorter ones have lower `model_nominal_work`.
6. **One short bench run** at ρ = 1.5 with `laef_wf`: `mean_probe_len < 1025` and `mean_ndis` below the stock `ada_ef` run at the same load; at ρ = 0.5, `frac_full_probe = 1` and `ndis` equals the `ada_ef` run.

### 1A.9 Risks specific to v2

| Risk | Mitigation |
|---|---|
| The Ada-ef score is noisier with fewer distances, so the table at short L may be less reliable | The table is calibrated at that L, so its recall is measured, not assumed. The dominance filter drops useless L values. Report recall per L from `probe_menu.json` |
| A short L makes the table pick a large ef, so the total saving is small | The price list work includes the post-probe search; dominated L values are dropped |
| Stage 1 is uniform: every query in a window gets the same L | Stage 2 is difficulty-aware; a cheap pre-search predictor is a possible extension, not in scope |
| `ef_pref = k` at every L on easy data | Stage 2 does nothing there, and stage 1 alone carries the savings. That is expected |

---

## 2. Tech stack and platform

| Layer | Choice |
|---|---|
| Serving engine, calibration, index build | **C++20**, GCC ≥ 11, CMake ≥ 3.22, OpenMP |
| Ada-ef dependencies | Eigen 3.4, Boost (headers; Boost.Math standalone is enough) |
| Our C++ dependencies (via FetchContent) | nlohmann/json (reads `meta.json`), CLI11, GoogleTest. `.npy` files are read by a small parser in `dataset_io.cpp`, so no HDF5 library is needed |
| GPU preprocessing | Python 3.11+, PyTorch with CUDA (Colab T4), numpy, pyarrow, pyyaml, tqdm, and h5py (only to read the HDF5 ann-benchmarks-format downloads, §4.3). The `gt`, `calib`, `stats` and `manifest` steps run on Colab only and refuse to run without CUDA. The serving machine runs only `download,convert`. The numpy fallback in `exact_knn` stays for the unit tests. **Use `uv`** for environments and installs (`uv venv`, `uv pip install -r requirements.txt`). |
| Orchestration and analysis | Python 3.11+, pandas, matplotlib, pyyaml (installed with `uv`); Jupyter notebooks only for plotting |
| OS | **Linux or WSL2 Ubuntu** for experiments. MSVC is not supported because Ada-ef uses variable-length arrays (`dist_t distances[statics_limit]`, `float bin_thresholds[num_bins]`), a GCC extension. MSYS2 MinGW-w64 GCC works for development and tests. |

---

## 3. Repository layout

The skeleton files already exist. Each one has a header comment that points back to its section in this plan.

```
Implementation/
├── IMPLEMENTATION_PLAN.md          this file
├── README.md                       quick start
├── .gitignore                      ignores data/, results/, build/
├── configs/
│   ├── datasets.yaml               dataset registry (§5)
│   ├── defaults.yaml               every tunable with its default (§15)
│   ├── experiments/
│   │   └── slo_study_120s.yaml     SLO study: stock Ada-ef and LA-ef at 0.95, 0.9, 0.8, 0.75 x the profile p95, constant-rate arrivals (120 s)
│   └── tests/qps_matrix.yaml       per-dataset QPS × N-queries test matrix (§13.3)
├── gpu_preprocessing/              ── RUN ON COLAB T4 (self-contained) ──
│   ├── README.md                   Colab instructions
│   ├── requirements.txt
│   ├── colab_preprocess.ipynb      thin driver notebook
│   ├── run_preprocess.py           CLI: --dataset --steps --out --device
│   ├── prep/
│   │   ├── __init__.py
│   │   ├── registry.py             reads configs/datasets.yaml
│   │   ├── download.py             fetch raw files (resumable)
│   │   ├── sources.py              chunked VectorSource readers per source type
│   │   ├── synthetic.py            synthetic-small, cluster-zipf generators
│   │   ├── convert.py              raw → canonical train.npy, test.npy (+ provided GT → gt/neighbors.npy)
│   │   ├── ground_truth.py         exact k-NN on GPU (torch), streamed in chunks
│   │   ├── calibration.py          sample calibration queries + exact GT
│   │   ├── estimator_stats.py      mean/covariance in Ada-ef Estimator format
│   │   ├── adaef_formats.py        writers/readers for Ada-ef binary formats
│   │   └── manifest.py             fingerprints + manifest.json
│   └── tests/
│       ├── test_ground_truth.py
│       └── test_formats.py
├── cpp/
│   ├── CMakeLists.txt
│   ├── cmake/Dependencies.cmake
│   ├── third_party/README.md       how to add the hnsw-ada-ef submodule
│   ├── include/laef/
│   │   ├── types.h                 QueryRecord, SearchStats, clocks
│   │   ├── config.h                RunConfig structs + JSON (de)serialization
│   │   ├── dataset_io.h            `.npy` + JSON readers (no HDF5)
│   │   ├── ann_backend.h           AnnBackend interface (the ONLY door to hnswlib)
│   │   ├── price_list.h            Step A tables, interpolation, water level
│   │   ├── calibration.h           Step A driver API
│   │   ├── telemetry.h             EWMA (Step B)
│   │   ├── sensor_window.h         p95 / S_p95 window (Step C)
│   │   ├── token_bucket.h          Step E bucket
│   │   ├── floor_mode.h            floor-mode logic (replaces admission control; never drops)
│   │   ├── plan_snapshot.h         shared state published by the controller
│   │   ├── controllers/
│   │   │   ├── budget_controller.h interface + ControllerInputs/Output
│   │   │   ├── p_aimd.h            design doc (BB §1)
│   │   │   ├── fixed_aimd.h        naive fixed-cut baseline
│   │   │   ├── static_capacity.h   open-loop B = ρ_target × B_max
│   │   │   └── controller_factory.h
│   │   ├── allocators/
│   │   │   ├── allocator.h         interface
│   │   │   ├── passthrough.h       Ada-ef (no cuts) / static-ef baselines
│   │   │   ├── uniform_cut.h       naive equal cut
│   │   │   ├── water_filling.h     price-list cut
│   │   │   └── allocator_factory.h
│   │   ├── planner.h               glue: bucket + allocator + snapshot (hook body)
│   │   ├── workload.h              arrival schedules
│   │   ├── engine.h                generator, dispatcher queue, workers, controller thread
│   │   └── results_writer.h        CSV/JSON outputs
│   ├── src/
│   │   ├── adaef_backend.cpp       ONLY TU including hnswlib/Ada-ef headers
│   │   ├── dataset_io.cpp
│   │   ├── price_list.cpp
│   │   ├── calibration.cpp
│   │   ├── workload.cpp
│   │   ├── engine.cpp
│   │   ├── results_writer.cpp
│   │   └── fake_backend.cpp        deterministic backend for engine tests
│   ├── apps/
│   │   ├── laef_build_index.cpp
│   │   ├── laef_calibrate.cpp      Step A
│   │   ├── laef_profile.cpp        idle + saturation profile → auto SLO / capacity
│   │   └── laef_bench.cpp          one experiment run
│   └── tests/                      GoogleTest (§13.1)
├── scripts/
│   ├── setup_wsl_ubuntu.sh         apt packages
│   ├── build.sh                    configure + build + ctest
│   ├── prepare_dataset.sh          serving machine: download+convert+verify manifest
│   ├── run_pipeline.sh             index → calibrate → profile for one dataset
│   └── run_experiments.py          expands experiment YAML → laef_bench runs
├── analysis/
│   ├── requirements.txt
│   ├── laef_analysis/{__init__,io,metrics,plots}.py
│   └── notebooks/01_calibration.ipynb … 04_ablations.ipynb
├── data/                           (gitignored) raw/ + artifacts/<dataset>/
└── results/                        (gitignored) <experiment>/<run>/
```

---

## 4. Data and artifact contract (the most important interface)

Every downstream step reads **only** these files. Whether the ground truth was provided by the dataset or generated on the GPU is visible only in an attribute (`gt_source`). Nothing downstream branches on it.

```
data/artifacts/<dataset_id>/
├── train.npy                  float32 [N×d] raw base vectors (NumPy .npy)
├── test.npy                   float32 [Q×d] raw query vectors
├── holdout.npy                int64 [Q] sorted base row ids of the held-out queries; only when query_source = holdout
├── meta.json                  dataset_id, metric, dim, n_base, n_query, seed, holdout_seed, query_source
├── gt/neighbors.npy           int32 [Q×K] exact top-K ids, ascending distance
├── gt/distances.npy           float32 [Q×K] distances (optional)
├── gt/gt_meta.json            gt_source = provided|generated, K, metric
├── calibration/samplings.bin  Ada-ef samplings format (queries + GT), see §4.1
├── calibration/calib_meta.json source, n, seed, row indices
├── stats/estimator.bin        Ada-ef Estimator format, see §4.1
├── manifest.json              fingerprints + provenance of every file above
├── index/hnsw_M{M}_efc{efc}.bin                  (CPU, serving machine)
├── tables/ada_ef_adapter.bin                     (v1 only; no longer read, see §1A.3)
├── tables/L{L}/ada_ef_adapter.bin                (CPU, v2) EfAdapter::serialize at statics_length = L, one folder per probe length
├── tables/L{L}/ada_ef_adapter.meta.json          (v2) compatibility metadata, statics_length = L
├── tables/L{L}/calibration_sweep.csv             (v2) Step A measurements at probe length L
├── tables/L{L}/price_list.json                   (v2) price list at probe length L (work includes the probe)
├── tables/probe_menu.json                        (v2) kept probe lengths and their model work/recall, §1A.4
├── tables/calibration_sweep.csv                  (CPU) raw Step A measurements
├── tables/price_list.json                        (CPU) Step A output, §9.3
└── profile/profile_C{C}.json                     (CPU) idle + saturation profile
```

**Machine split.** The Colab step produces `gt/`, `calibration/`, `stats/` and `manifest.json` (all small) and copies them to Google Drive. The serving machine runs `prepare_dataset.sh`, which regenerates `train.npy` and `test.npy` deterministically and verifies the fingerprints in `manifest.json` before copying in the Colab outputs.

### 4.1 Binary formats (must match Ada-ef exactly; little-endian)

- **`samplings.bin`** (`hnswdis::serialize_samplings`). Ada-ef's `MatrixXf`/`MatrixXi` are **row-major**:
  `int32 rows, int32 cols, float32[rows*cols] queries, int32 rows, int32 cols, int32[rows*cols] gt`
- **`estimator.bin`** (`hnswdis::save_estimator_to_file`):
  `uint64 type_len, char[type_len] type, int32 d, int32 d, float32[d*d] covariance, int32 d, float32[d] means, int32 d, float32[d] variances`
  - `type` is `"CosineDistanceEstimator"` for metric `cd` and `"InnerProductEstimator"` for `ipd`.
  - The covariance is **unbiased (divide by n−1)**, computed over the base vectors **after normalization when the metric is `cd`**. `variances` is the diagonal of the covariance.
  - Accumulate in float64 on the GPU and cast to float32 at the end.

### 4.2 Conventions

- Metrics: `cd` = cosine distance (vectors L2-normalized with `v / (‖v‖ + 1e-30)`, the same as Ada-ef's `normalize_matrix`), and `ipd` = inner-product distance `1 − q·x`. **Ada-ef only supports these two metrics**, so L2 datasets are not used.
- `train.npy` and `test.npy` store the raw (unnormalized) vectors. Normalization happens at load time in both Python and C++.
- GT always stores K = 100 neighbours (`gt.k_store`), which allows k ∈ {1, 10, 100}.
- Holdout queries (for datasets without a query set): `idx = sort(np.random.default_rng(holdout_seed).choice(N_raw, n_q, replace=False))`, where `holdout_seed` defaults to the dataset seed. The base is the remaining rows in their original order, and a base vector's ID is its row in `train.npy`.

### 4.3 Decision: NumPy `.npy` instead of HDF5

The datasets, query matrices and ground truth are stored as `.npy` files with JSON metadata. HDF5 is not used for them.

- **Why.** Every read in this project is a whole matrix or a sequential scan (base vectors, queries, ground truth). `.npy` serves that directly. Python opens the files with `np.load(..., mmap_mode="r")`. C++ reads them with a small header parser in `dataset_io.cpp`, so the build needs no HDF5 library and no CMake discovery. The attributes HDF5 would hold (metric, K, `gt_source`) live in JSON files (`meta.json`, `gt/gt_meta.json`).
- **Rejected.** Keeping HDF5 meant linking libhdf5 and a C++ wrapper (HighFive) on the serving machine. The wrapper's API changed between major versions, and we could not confirm the read calls for the version we would pin. String attributes were also awkward. None of this bought anything for our access pattern.
- **Where HDF5 remains.** The ann-benchmarks-format downloads (for example `glove-100-angular.hdf5` and `coco-t2i-512-angular.hdf5`) are HDF5 upstream. The convert step reads them once with h5py and writes `.npy` files (§7, P1.3 and P1.5). Nothing downstream reads HDF5.

---

## 5. Datasets (kept small on purpose)

Six analysis datasets cover the behaviours that matter for LA-ef: a hub-heavy word-embedding set, image descriptors where Ada-ef's tail loss is printed in the paper, synthetic cluster-size skew, CLIP image and caption queries against one image base (COCO I2I and T2I), and a recommender workload (Last.fm). Three have provided ground truth (glove, deep-image, Last.fm) and four get GT generated on the GPU (cluster-zipf, synthetic-small and the two COCO sets). The COCO provided files are not used: they disagree with exact cosine search on about 1% of neighbours (VERIFICATION_NOTES, Pass 7). Both GT paths still get exercised.

| id | Role in the analysis | N × d | Metric | GT | Source / download | Disk / RAM (approx.) |
|---|---|---|---|---|---|---|
| `synthetic-small` | **Tests and smoke runs only** | 100k × 64 (Gaussian blobs) | cd | generated on the GPU (Colab) | `prep/synthetic.py` (seeded). The first copy was made on the CPU and kept as it is | 25 MB / <1 GB |
| `glove-100-angular` | Hub-heavy; worst static-ef recall in the Ada-ef paper; fast iteration | 1.18M × 100 | cd | **provided** (top-100) | <http://ann-benchmarks.com/glove-100-angular.hdf5> (463 MB) | 0.5 GB / ~2 GB |
| `deep-image-96-angular` | Ada-ef's printed tail gap (P1 = 0.81); 10M scale for memory contention | 9.99M × 96 | cd | **provided** (top-100) | <http://ann-benchmarks.com/deep-image-96-angular.hdf5> (3.6 GB) | 3.7 GB / ~10 GB at build, ~6 GB serving |
| `coco-i2i-512-angular` | CLIP image queries against an image base: the in-distribution control, and the small analogue of Ada-ef's Laion-I2I | 113k x 512 | cd | **generated** (exact, GPU; provided top-100 kept for reference only, Pass 7) | <https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-i2i-512-angular.hdf5> (136 MB) | 0.3 GB / <1 GB |
| `coco-t2i-512-angular` | CLIP caption queries against the same kind of image base: a cross-modal shift, and the small analogue of Ada-ef's Laion-T2I | 113k x 512 | cd | **generated** (exact, GPU; provided top-100 kept for reference only, Pass 7) | same release, `coco-t2i-512-angular.hdf5` (136 MB) | 0.3 GB / <1 GB |
| `lastfm-64-dot` | Matrix-factorization recommender retrieval with item-popularity hubs. Stored as angular upstream (the inner-product to cosine step is done there), so no extra step is needed here | 292k x 65 | cd | **provided** (top-100) | <https://ann-benchmarks.com/lastfm-64-dot.hdf5> (135 MB) | 0.1 GB / <1 GB |
| `cluster-zipf-100d` | Cluster-size skew; "all methods struggle" in the Ada-ef paper | default **2M** × 100 (paper uses 10M; set `n_base`) | cd | **generated** | `prep/synthetic.py`, reproducing Ada-ef's `data_prep.ipynb` (5000 clusters, std 0.01, centers U(−0.1, 0.1), Zipf s=1, base=100; queries = 10k sampled base rows) | 0.8 GB / ~3 GB |

Not included, because each adds processing beyond the ann-benchmarks path: MS MARCO V1/V2.1 and the full LAION I2I/T2I (30.6M vectors each in the Ada-ef paper; MS MARCO and LAION each need roughly 35–110 GB downloads and more than 64 GB of RAM); NYTimes (239 zero-norm training vectors and NaN ground-truth distances, so it needs cleaning first); GIST, SIFT, MNIST and Fashion-MNIST (Euclidean, which Ada-ef does not support); and Kosarak and MovieLens (Jaccard). No native inner-product (ipd) dataset is included: Last.fm is stored as angular, and a native inner-product set would need a new reader for its binary format. The registry format supports adding them later (source types `hf_npy_shards` and `laion_npy`), and their URLs are in Ada-ef's `experiments_driver/data_prep.ipynb`. The authors also publish a prepared data and index bundle (<https://drive.google.com/uc?id=1K1yeVxLe6L2ZMjnwLxHcJOPoFy9uNR-_>), which is optional and not required by this plan.

The ann-benchmarks-format downloads are HDF5 files upstream. The convert step reads them once with h5py and writes `.npy` files (§4.3).

`configs/datasets.yaml` holds one entry per dataset with these fields: `id, role, source_type (ann_hdf5|hf_parquet|synthetic), urls, metric, dim, n_base, n_query, query_source (provided|holdout|synthetic_sample), gt_provided_valid (bool), seed, synthetic params`.

---

## 6. Phase 0: Environment and build skeleton

**P0.1 Setup script.** Files: `scripts/setup_wsl_ubuntu.sh`.
- Spec: `apt install build-essential cmake ninja-build libeigen3-dev libboost-dev python3-venv git curl`, then installs uv (uv provides the Python 3.11 that the project venv uses). Verify that `gcc --version` is ≥ 11 and `cmake --version` is ≥ 3.22.
- Acceptance: the script is idempotent and prints the tool versions.

**P0.2 Submodule.** Files: `cpp/third_party/README.md`.
- Spec: `git submodule add https://github.com/chaozhang-cs/hnsw-ada-ef cpp/third_party/hnsw-ada-ef`, pinned to a commit hash recorded in the README.
- Spec: do **not** use their `CMakeLists.txt` (it hard-codes local paths). We only add `cpp/third_party/hnsw-ada-ef` to the include path, so the headers are reached as `hnswlib/...`.

**P0.3 CMake.** Files: `cpp/CMakeLists.txt`, `cpp/cmake/Dependencies.cmake`.
- Spec: `CMAKE_CXX_STANDARD 20`. Release flags `-O3 -march=native -fopenmp -DNDEBUG`. Option `LAEF_NATIVE` (ON by default) controls `-march=native`.
- Spec: `find_package(Eigen3 3.4 REQUIRED)`, `find_package(Boost REQUIRED)`, `find_package(OpenMP REQUIRED)`. FetchContent for nlohmann_json, CLI11 and googletest. No HDF5 library is needed.
- Targets:
  - `laef_core` (static library): every file in `src/` except `adaef_backend.cpp` and `fake_backend.cpp`.
  - `laef_adaef` (static library): only `adaef_backend.cpp`.
  - `laef_fake` (static library): only `fake_backend.cpp`.
  - Apps link `laef_core` and `laef_adaef`. Tests link whichever they need.
- Acceptance: `scripts/build.sh` configures, builds empty apps, and `ctest` runs (0 tests is fine at this stage).

---

## 7. Phase 1: GPU preprocessing (`gpu_preprocessing/`, Colab T4)

The CLI is `python run_preprocess.py --dataset <id> --steps download,convert,gt,calib,stats,manifest --data-root <dir> --out <dir> [--device cuda|cpu] [--k-store 100] [--calib-n 1000]`. Steps are idempotent: a step skips itself when its output exists and the manifest hash matches, and `--force` reruns it.

**P1.1 `registry.py`.** Loads `configs/datasets.yaml`. By default it looks two folders up; `--registry` overrides the path, so the folder can be copied to Colab on its own. Returns a `DatasetSpec` dataclass.

**P1.2 `download.py`.** Streams HTTP to `<data-root>/raw/<id>/`, with resume (`Range` header), retries and a size check. For `hf_parquet` it lists the shards through `https://huggingface.co/api/datasets/<repo>/tree/main/data` and downloads each `.../resolve/main/<path>`.

**P1.3 `sources.py`.** Defines `class VectorSource: n, d; iter_chunks(chunk_rows) -> Iterator[(start_row, np.ndarray float32)]; take_by_index(idx) -> np.ndarray`. Implementations:
- `NpySource(path)`: one of our `.npy` matrices (`train.npy`, `test.npy`), read through a memory map. The GT, calibration, stats and manifest steps read the data through it.
- `AnnHdf5Source(path, key)`: an h5py dataset read in slices. It is used only for the ann-benchmarks-format HDF5 downloads.
- `ParquetSource(files, column)`: iterates pyarrow row groups and stacks each list column into a float32 `[n, d]` array. It applies the holdout mask so that the IDs follow §4.2.
- `ArraySource(np.ndarray | np.memmap)`: used for synthetic data.

The GT and stats steps read **directly from these sources**, so Colab never needs to write a second full copy of the dataset.

**P1.4 `synthetic.py`.** Generators for `synthetic-small` and `cluster-zipf-100d` that reproduce Ada-ef's `make_blobs` recipe and Zipf counts (`zipf_samples(N=5000, total=n_base, s=1, base=100)`), seeded. Write them to an `.npy` memmap so 2M–10M rows fit in Colab's RAM.

**P1.5 `convert.py`.** Writes the canonical `train.npy` and `test.npy` (§4) from a `VectorSource`, plus the query matrix (provided, holdout or sampled), `holdout.npy` when `query_source = holdout`, and `meta.json`. If `gt_provided_valid` is true, it copies the source's `neighbors`/`distances` (the first K columns) into `gt/neighbors.npy` (and `gt/distances.npy` when the source has distances) with `gt_source="provided"` in `gt/gt_meta.json`. This step runs on **both** machines and must be byte-deterministic.

**P1.6 `ground_truth.py`.** This is the core GPU job and the only one that is truly expensive. Spec:

```python
def exact_knn(queries: np.ndarray, base: VectorSource, k: int, metric: str,
              device="cuda", q_batch=4096, base_chunk=262_144, dtype=torch.float32):
    # 1. normalize queries (and every base chunk) if metric == "cd"
    # 2. for each query batch on the GPU: keep running best (dist[Qb,k], ids[Qb,k]) = +inf / -1
    #    for each base chunk (streamed from disk, pinned memory, moved with non_blocking):
    #        sim  = Qb @ Xc.T                  # [Qb, chunk]
    #        dist = 1 - sim                     # identical for cd and ipd (hnswlib definitions)
    #        cand = topk(-dist, k) on chunk; merge with running best via torch.topk over the 2k candidates
    #        candidate ids += chunk start_row
    # 3. return ids int32 [Q,k] sorted by ascending distance, dists float32
```

- fp32 is the default (exact). `--fp16-rerank` is optional: take the top `k+64` candidates in fp16, then re-score them in fp32.
- The T4 has 16 GB of VRAM. With 4096 queries and a 262k-row chunk, the similarity matrix takes 4.3 GB in fp32, which fits. Drop `base_chunk` to 65,536 for d = 1536.
- `--verify-provided N`: for datasets with provided GT, recompute exact GT for N random queries and report agreement (recall@10 of provided vs ours). It must be ≥ 0.999, otherwise fail. This validates both our code and the provided GT.
- Output: `gt/neighbors.npy` (int32 [Q×K]), `gt/distances.npy` and `gt/gt_meta.json` with `gt_source="generated"`.

**P1.7 `calibration.py`.** Samples the calibration queries and computes their exact GT with `exact_knn`.
- `calibration.source = base_sample` is the authors' default (`compute_samplings`). It draws rows from the base without replacement, seeded. The query's own vector is then its rank-1 neighbour, exactly as in Ada-ef; document this in `calib_meta.json`.
- `query_holdout` takes a disjoint slice of the test queries, which are then excluded from evaluation through `calib_meta.json`.
- GT width = `k` (the serving k, default 10). This is what `EfAdapter` reads, and it must be ≥ k.
- Default `n = 1000`. Ada-ef uses 200; the larger value gives more queries per difficulty group for the price list.
- Output: `calibration/samplings.bin` (§4.1).

**P1.8 `estimator_stats.py`.** Streams base chunks, accumulating `sum` (d) and `X^T X` (d×d) in float64 on the GPU. Then mean = sum/n and cov = (XᵀX − n·μμᵀ)/(n−1). Writes `stats/estimator.bin`. This replaces Ada-ef's CPU `init_estimator`, which is slow for d = 1536.

**P1.9 `manifest.py`.** Records:
- the dataset spec, seed, n, d and holdout indices hash;
- `fingerprint` = sha256 over the shapes, the full test matrix and the base rows at stride `max(1, n//1024)`;
- sha256 and `gt_source` of every output file;
- the torch/CUDA versions.

`prepare_dataset.sh` recomputes the fingerprint on the serving machine and refuses to continue if it differs.

**P1.10 `colab_preprocess.ipynb`** (thin). Cells:
1. `!nvidia-smi`
2. Mount Drive
3. Copy or clone `gpu_preprocessing/` and `configs/datasets.yaml`
4. `pip install -r requirements.txt`
5. A loop over `["glove-100-angular", "deep-image-96-angular", "cluster-zipf-100d", "coco-i2i-512-angular", "coco-t2i-512-angular", "lastfm-64-dot"]` calling `run_preprocess.py ... --data-root /content/data --out /content/drive/MyDrive/laef_artifacts`
6. Print each manifest

Note in the notebook that Colab local disk is about 100 GB and RAM about 12.7 GB. Everything streams, so the only per-dataset requirement is disk space for the raw download.

**Acceptance (Phase 1):** `pytest gpu_preprocessing/tests` passes on CPU (§13.2). On Colab, all 4 datasets produce artifacts, and `--verify-provided 500` reports ≥ 0.999 for GloVe and Deep.

### 7.1 CLI contract used by `colab_preprocess.ipynb` (✅ notebook written)

The notebook runs, from inside `gpu_preprocessing/`:
```
python run_preprocess.py --dataset <id> --steps download,convert,gt,calib,stats,manifest
       --data-root /content/data --out /content/drive/MyDrive/laef_artifacts --registry ../configs/datasets.yaml
       --device cuda --k 10 --k-store 100 --calib-n 1000 --calib-source base_sample [--verify-provided 500] [--force]
```
`run_preprocess.py` must accept exactly these flags and lay files out as follows:
- raw downloads go to `<data-root>/raw/<id>/`;
- the large `train.npy`, `test.npy`, `holdout.npy` (holdout datasets only) and `meta.json` always go to `<data-root>/artifacts/<id>/` (Colab local disk). Later steps read the matrices with `NpySource` through a memory map; this is simpler than streaming from the raw files and costs only local disk;
- small artifacts go to `<out>/<id>/`: `gt/neighbors.npy`, `gt/distances.npy` (when present), `gt/gt_meta.json`, `calibration/samplings.bin`, `calibration/calib_meta.json`, `stats/estimator.bin`, `manifest.json`, and `convert_meta.json` (what the convert step wrote). On the serving machine `--out` defaults to `<data-root>/artifacts`, so everything ends up in one folder;
- the exit code is non-zero on failure (the notebook continues with the next dataset);
- the notebook checks these fields: `gt/neighbors.npy` is an int32 matrix and `gt/gt_meta.json` has `gt_source`, `K` and `metric`; `calib_meta.json` is a flat JSON object (list-valued keys such as `base_rows`, `test_rows` and `excluded_test_rows` are allowed); `manifest.json` has the keys `fingerprint` and `files`.

---

## 8. Phase 2: C++ Ada-ef backend

### 8.1 Interface (`include/laef/ann_backend.h`)

```cpp
namespace laef {
struct SearchStats {           // filled per query
  uint32_t ndis = 0;           // all distance computations (upper layers + layer 0)
  uint32_t probe_ndis = 0;     // dc spent before the hook fired
  float    score = -1.f;       // Ada-ef score (0..100); -1 if probe never completed
  uint32_t ef_pref = 0;        // Ada-ef's ef (Sketch::estimate_ef2(score), floored at k)
  uint32_t ef_used = 0;        // ef returned by the planner
  bool     probe_complete = false;
};
// Called exactly once per query, on the worker thread, when the probe completes.
// Returns the ef to use for the remainder of the search (>= k).
using EfPlannerFn = std::function<uint32_t(float score, uint32_t ef_pref, uint32_t probe_ndis)>;

struct BackendConfig { std::string index_path, metric /*cd|ipd*/, estimator_path, adapter_path;
                       uint32_t dim, M, k; float target_recall; float quantile_step = 1e-3f;
                       size_t statics_length = 0; /*0 => 1 + 2M + (2M-1)*2M  (=1025 for M=16)*/ };

class AnnBackend {
 public:
  virtual ~AnnBackend() = default;
  static std::unique_ptr<AnnBackend> open(const BackendConfig&);      // in adaef_backend.cpp
  virtual uint32_t dim() const = 0;
  // Ada-ef probe + planner hook (LA-ef, Ada-ef baseline, calibration sweep)
  virtual void search_planned(const float* q, uint32_t k, const EfPlannerFn& planner,
                              uint32_t* out_ids, SearchStats* st) const = 0;
  // plain HNSW with explicit ef (static-ef baseline); thread-safe, no setEf()
  virtual void search_fixed(const float* q, uint32_t k, uint32_t ef,
                            uint32_t* out_ids, SearchStats* st) const = 0;
};
// Offline helpers (also in adaef_backend.cpp). Data pointers are row-major float32 [n x d], raw;
// they are normalised internally for cd. The backend TU never touches HDF5: apps load `train.npy` via `dataset_io`.
void build_index(const float* data, size_t n, size_t d, const std::string& metric, uint32_t M,
                 uint32_t ef_construction, uint32_t threads, const std::string& out_index, size_t seed = 100);
void build_ef_adapter(const BackendConfig&, const std::string& samplings_bin,
                      uint32_t ef_upper_bound, const std::string& out_adapter_bin);
void compute_estimator_cpu(const float* data, size_t n, size_t d, const std::string& metric,
                           const std::string& out_estimator_bin);   // fallback if GPU stats missing
}
```
✅ **Implemented as specified in `cpp/include/laef/ann_backend.h` / `cpp/src/adaef_backend.cpp`.**

> **v2 change (§1A.3 C2–C3):** `search_planned(q, k, probe_len, planner, out_ids, st)`; the backend loads one adapter and sketch per probe length; `SearchStats::probe_len` records the L used.

`fake_backend.cpp` implements the same interface for engine tests:
- work(ef) = `100 + ef` dc (100 dc probe), with no jitter, so the engine tests are exact;
- service time = work × `u_fake`, spent busy-waiting;
- the planner is called once per query with a score drawn from a seeded distribution.

### 8.2 `src/adaef_backend.cpp` (the only TU that includes `hnswlib/hnswlib.h` and `hnswlib/adaptive_ef.h`)

**P2.1 Loading.**
- Create the space for the metric: for `cd` and `ipd` use `hnswlib::InnerProductSpace(dim)`, which is the same as the authors' `init_space`.
- Load the index with `HierarchicalNSW<float>(space, index_path)`.
- Load `hnswdis::load_estimator_from_file(estimator_path)` and build `hnswdis::ApproximatedScoreCalculator(estimator, quantile_step)` with the paper's fixed exponential decay and five bins. These are implementation constants, not runtime configuration.
- Load `hnswdis::EfAdapter(adapter_path)` and build `hnswdis::Sketch(adapter.get_ef_recall_estimators(), target_recall)`. Keep the `EfAdapter` alive for as long as the `Sketch` exists, because the `Sketch` stores a reference to it.
- For `cd`, normalize every query before searching, as the authors' loader does.

**P2.2 `search_planned`.** Write a free function template `planned_search(const HierarchicalNSW<float>& h, const void* q, size_t k, size_t statics_length, const ApproximatedScoreCalculator&, const Sketch&, const EfPlannerFn&, SearchStats*)`. It copies `adaptiveSearchKnn` and `adaptiveSearchBaseLayerST<bare_bone_search=true>` from `hnswalg.h` (lines ≈1335–1600 at the pinned commit) with exactly these changes:

1. Replace `std::max(ef_, k)` with `k`, so the search never reads the shared `ef_` member. This makes it thread-safe.
2. Replace the VLA `dist_t distances[statics_limit]` with a `thread_local std::vector<float>` resized to `statics_length`.
3. Count `ndis += size` for every neighbour list scanned, in the upper layers and in layer 0. Count it locally instead of using the shared atomic `metric_distance_computations`.
4. At the point where the original code does `ef = sketch->estimate_ef2(score); if (ef < ef_copy) ef = ef_copy;`, do this instead:
   ```cpp
   uint32_t ef_pref = std::max<uint32_t>(sketch.estimate_ef2(score), k);
   st->probe_ndis = ndis; st->score = score; st->ef_pref = ef_pref; st->probe_complete = true;
   ef = std::max<uint32_t>(planner(score, ef_pref, ndis), k);
   st->ef_used = ef;
   ```
   Then continue exactly as the original: shrink `top_candidates` to `ef` and update `lowerBound`.
5. If the loop ends before `size_distances == statics_limit` (only possible on tiny graphs), set `probe_complete = false` and `ef_used = 0`, and do not call the planner. The engine counts these queries separately.
6. Copy the final top-k labels (closest first) into `out_ids`.

The deleted-element and filter branches (`bare_bone_search=false`) are not needed.

**P2.3 `search_fixed`.** Copy `searchKnn` + `searchBaseLayerST<true>` with an explicit `ef` parameter and the local `ndis` counter.

**P2.4 `build_index`.** Reads `train.npy` through `dataset_io`, normalizes for `cd`, then calls `HierarchicalNSW<float>(space, N, M, efc)` and `addPoint` in parallel (copy the authors' `ParallelFor` pattern), and finally `saveIndex`. Defaults are M = 16 and efc = 500, as the authors use.

**P2.5 `build_ef_adapter`.**
- `deserialize_samplings(samplings_bin, Q, GT)`, normalize Q for `cd`.
- Construct `hnswdis::EfAdapter(alg_hnsw, data_vectors, k, metric, target_recall, quantile_step, statics_length, Q_ptr, GT_ptr, estimator_ptr, ef_upper_bound)` (the constructor overload that takes queries, GT and an `Estimator` pointer). At the pinned commit, `data_vectors` is passed down to `hnsw_search_and_score` but never read. Verify this in the source: if it is unused, pass an empty `std::make_shared<hnswdis::MatrixXf>()`; otherwise load the base vectors.
- Then `serialize(out_adapter_bin)`.
- Note that `EfAdapter::init` calls `alg_hnsw->setEf(...)`, which is not thread-safe. Calibration is single-threaded, so this is acceptable.

**Acceptance (Phase 2):** `test_backend_parity` (§13.1) passes:
- `search_planned` with `planner = [](s, ef_pref, n){ return ef_pref; }` returns the same IDs as the authors' `adaptiveSearchKnn(q, k, statics_length, score_cal, &sketch)` with `setEf(k)`, for 200 queries.
- `search_fixed(ef)` returns the same IDs as `setEf(ef); searchKnn(q, k)`.

---

## 9. Phase 3: Step A calibration and price list (`laef_calibrate`)

> **v2 (§1A.4):** the steps below run once **per probe length** L, with `statics_length = L`, writing into `tables/L{L}/`, and the run ends by writing `tables/probe_menu.json`.

**P3.1 Driver (`calibration.cpp`, app `laef_calibrate`).** Steps:
1. Ensure `tables/ada_ef_adapter.bin` exists; if it is missing, build it from `calibration/samplings_valid.bin`. If `stats/estimator.bin` is missing, call `compute_estimator_cpu`; if `calibration/samplings.bin` is missing, fail with a clear message telling the user to run `gpu_preprocessing ... --steps calib` on the GPU (Colab).
2. Open the backend. For every calibration query i, run `search_planned` with a planner that **records** (score_i, ef_pref_i) and returns `ef_pref_i`. Record ndis and recall at ef_pref (Ada-ef's own operating point).
3. **Ladder sweep.** Build the ladder L = geometric from k with ratio `calibration.ladder_ratio` (default 2): {k, 2k, 4k, …}. Stop at the first rung ≥ `max_i ef_pref_i`, capped at `ef_upper_bound`. For each query i and each rung e ∈ L, run `search_planned` with a planner that returns `e` (so the probe is identical to live serving), and record `(i, score_i, e, recall, ndis, service_us)`. This uses `--threads` worker threads, one query per task. ndis and recall do not depend on parallelism, and `service_us` is informational only.
4. Write `tables/calibration_sweep.csv` with columns `query, score, ef_pref, ef, recall, ndis, probe_ndis, service_us, is_pref`.

**P3.2 Grouping.**
- Group g is a contiguous range of integer Ada-ef score keys (`int(score)`, 0..100).
- Start from one bucket per score key. Greedily merge adjacent buckets until every bucket has ≥ `calibration.min_group_size` (default 30) queries. Then, if there are still more than `calibration.max_groups` (default 8), merge the smallest adjacent pair until the count fits.
- Store `score_lo/score_hi` and `p_g` (the group's share of queries).
- Live lookup is a binary search on `score_lo`.

**P3.3 Tables (`price_list.cpp`).** For each group g and rung e:
- r̄_g(e) = mean recall; c̄_g(e) = mean ndis; t̄_g(e) = mean service time.
- Enforce monotonicity with a running max along e, for both r̄ and c̄.
- Compute the **upper concave envelope** of (c̄, r̄) per group, dropping rungs below it. This makes the loss ratios non-decreasing as `ef` drops, so the greedy cut order is optimal.
- e*_g = median of ef_pref over the queries in g.
- Global values:
  - c̄(k) = Σ p_g c̄_g(k) (the floor work)
  - w̄_nom = mean ndis at ef_pref
  - `static_ef_for_target` = smallest rung whose overall mean recall is ≥ the target (for the static-ef baseline)

Steps: for each group and each pair of adjacent kept rungs e_j → e_{j−1}, compute Δw = c̄(e_j) − c̄(e_{j−1}), Δr = r̄(e_j) − r̄(e_{j−1}), and ℓ = Δr/Δw. `steps_sorted` is all steps across groups sorted ascending by ℓ (MF §1 "Price List").

**P3.4 `price_list.json` schema:**
```json
{"dataset":"glove-100-angular","metric":"cd","k":10,"target_recall":0.95,"quantile_step":0.001,"M":16,"statics_length":1025,"ef_upper_bound":5000,
 "ladder":[10,20,40,80,160,320],"static_ef_for_target":80,
 "floor_work_mean":1650.0,"nominal_work_mean":3900.0,"calib_mean_recall_at_pref":0.951,
 "groups":[{"id":0,"score_lo":0,"score_hi":7,"p":0.18,"ef_pref_median":160,
            "ladder":[10,20,40,80,160],"recall":[...],"work":[...],"service_us":[...]}],
 "steps_sorted":[{"group":3,"from":40,"to":20,"d_recall":0.001,"d_work":700,"loss":1.4e-6}]}
```

**P3.5 `PriceList` C++ API (`price_list.h`):**
```cpp
class PriceList {
 public:
  static PriceList load(const std::string& json_path);
  int    group_of(float score) const;                    // binary search
  double work(int g, double ef) const;                   // piecewise-linear interpolation in ef; clamp at the ends
  double recall(int g, double ef) const;
  const std::vector<uint32_t>& rungs(int g) const;       // ascending, kept (concave) rungs
  double floor_work(int g) const;                        // work(g, k)
  double floor_work_mean() const;  double nominal_work_mean() const;
  // Water level (MF §1 price list + §5 Case 2), computed on expected demand:
  //   if nominal_work_mean <= w_target -> 0 (no cuts)
  //   else walk steps_sorted accumulating saved += p_g * d_work until nominal - saved <= w_target;
  //   return loss of the last step taken; +inf if all steps are taken.
  double water_level(double w_target) const;
};
```

**Acceptance (Phase 3):** `test_price_list` passes. On `glove-100-angular`, `calib_mean_recall_at_pref` is within ±0.02 of 0.95 (Ada-ef's claim), and r̄_g is monotone after the envelope.

---

## 10. Phase 4: Online components (header-only, no hnswlib)

### 10.1 Telemetry, Step B (`telemetry.h`)
```cpp
struct Ewma { double alpha, value; bool init=false;
  void add(double x){ value = init ? alpha*x + (1-alpha)*value : x; init = true; } };
struct Telemetry {                       // guarded by one mutex; updated by workers on completion
  Ewma unit_cost{0.05};                  // u   = EWMA(S_i / ndis_i)        [s/dc]
  Ewma mean_service{0.05};               // E[S]= EWMA(S_i)                  [s]
  Ewma arrival_rate{0.2};                // λ̂   = EWMA(arrivals per tick)/tick  (tick = 10 ms, updated by the controller thread)
};
```

### 10.2 Sensor window, Step C (`sensor_window.h`)

A ring buffer holding the last `window.n` (default 500) completed `(T_i, S_i)` pairs.
- It "closes" when `window.n` new completions have arrived since the last close, **or** when `window.max_ms` (20 ms) has elapsed **and** at least `window.min_samples` (50) new completions exist. Without the minimum-sample guard, slow datasets would compute p95 from about 5 samples.
- Cadence follows throughput (Main Flow §3, Budget B §3). The 500-completion trigger fires every 500 / λ seconds: about 0.83 s at 600 QPS and 0.25 s at 2,000 QPS. The 20 ms trigger can fire only when at least `min_samples` completions arrive within `max_ms`, which is above 2,500 QPS with the defaults. Recovery is therefore counted in windows, and its duration in seconds depends on λ.
- p95 and S_p95 are **exact**, using `std::nth_element` on a copy of the buffer (500 elements, a few µs). This replaces the t-digest from MF §3 at no real cost.
- Q is read from the dispatcher's atomic queue size.
- Ŵ_q = Q·E[S]/C.

### 10.3 Plan snapshot (`plan_snapshot.h`)

The controller thread publishes an immutable `PlanSnapshot` through `SnapshotStore` (a mutex-guarded `std::shared_ptr`, portable to GCC 11). Readers copy the pointer, so the lock is held only for a pointer copy. ✅ Implemented, including `make_snapshot()`, which derives D, w_target, water_level, uniform_fraction and reservation from one controller output.
```cpp
struct PlanSnapshot { double B, B_max, B_min, D, T_slo, S_floor, mean_service, w_target,
                      water_level, uniform_fraction; bool floor_mode; uint64_t version; };
```
Where:
- w_target = B/max(λ̂, ε)
- D = B·T_base, with T_base = T_SLO − E[S] (or T_SLO − S_p95 when `bucket.t_base = tail`), floored at 0
- water_level = `price_list.water_level(w_target)`
- uniform_fraction = min(1, w_target / w̄_nom)

### 10.4 Budget controllers, Step D (`controllers/`)
```cpp
struct ControllerInputs { double p95, s_p95, queue_depth, mean_service, unit_cost,
                          arrival_rate, floor_work, t_slo; int C; };
struct ControllerOutput { double B, B_max, B_min, S_floor; bool floor_mode; const char* zone; };
class BudgetController { public: virtual ~BudgetController() = default;
  virtual ControllerOutput update(const ControllerInputs&) = 0;
  virtual void reset(double B0) = 0; virtual std::string name() const = 0; };
std::unique_ptr<BudgetController> make_controller(const ControllerCfg& cfg); // controller_factory.h (nullptr for "none")
```
- **`p_aimd`**: a line-for-line port of BB §1 `update_budget`. The parameters come from config: `cut_floor 0.5`, `headroom 0.85`, `growth 1.02`, `guard_enabled true`. `enter_floor_mode` becomes `floor_mode = true` in the output: it is true when `room < S_floor` **or** `B_min > B_max`, and false otherwise. It is re-evaluated every window, so it is never sticky. `zone` ∈ {violation, headroom, comfort} plus guard/floor flags, for logging.
- **`fixed_aimd`** (naive): on violation `B *= fixed_cut` (0.9), on headroom `B *= growth`, otherwise hold. The guard is switchable (default off, to keep it naive). Same clamp as `p_aimd`.
- **`static_capacity`**: `B = rho_target × B_max` (default 0.9) with no feedback. It is the open-loop "different computation method" and shows the pattern for adding more.
- A **`none`** mode, used by the Ada-ef and static-ef baselines, means no controller and no bucket at all. The engine handles it (`controller.type = none`).

To add a new budget method: create `controllers/<name>.h` implementing `BudgetController`, then add one `if (type == "<name>")` branch in `controller_factory.h`.

### 10.5 Token bucket, Step E (`token_bucket.h`)
```cpp
class TokenBucket {                      // one mutex; reservation + settlement are lock-protected
 public:
  // Lazily refill at B; cap at +cap_pos (default 0 => total burst credit = D).
  // Return the actual amount charged after the -D clamp; a later settlement refunds only this amount.
  double reserve(double nominal_dc, int64_t now_ns, double B, double D);
  // Refund charged_reservation, compute affordable = balance + D, let plan choose work, and debit it
  // under the same lock.
  template <class PlanFn>
  double settle(double charged_reservation, int64_t now_ns, double B, double D, PlanFn&& plan);
 private: std::mutex m; double balance = 0; int64_t last_ns = 0; double cap_pos = 0;
};
```

### 10.6 Allocators (`allocators/`)
```cpp
struct AllocRequest { int group; uint32_t ef_pref; uint32_t k; double affordable; };
struct AllocResult  { uint32_t ef; double planned_work; bool forced_floor; };
class Allocator { public: virtual ~Allocator() = default;
  virtual AllocResult plan(const AllocRequest&, const PlanSnapshot&, const PriceList&) const = 0;
  virtual std::string name() const = 0; };
```
The ladder for query i is the kept rungs of its group that are below ef_pref_i, plus ef_pref_i itself as the top rung. Work and recall at ef_pref_i are interpolated.

- **`passthrough`**: `ef = ef_pref` (Ada-ef baseline). With `static_ef` set, `ef = static_ef_for_target`, and the engine calls `search_fixed` for the static baseline.
- **`uniform_cut`** (naive): target = `uniform_fraction × work(g, ef_pref)`. Choose the largest ladder ef with `work ≤ target`, or k if none qualifies. Option `uniform_cut.mode = ef` scales ef instead of work.
- **`water_filling`** (intelligent): starting at ef_pref, step down the ladder while the next step's ℓ ≤ `water_level`. Cheap cuts are taken first and consistently across groups. Hard queries, whose top steps are expensive, keep their ef.
- **The bucket check is common to all allocators except passthrough.** While `work(g, ef) > affordable` and `ef > k`, step down one rung. If the work at k still exceeds `affordable`, run at k with `forced_floor = true`. Queries are never dropped.

### 10.7 Floor mode (`floor_mode.h`), which replaces admission control

**No query is ever dropped, rejected or aborted, under any load.** There is no admission gate and no drop path in the engine. When the budget cannot be met, quality goes down to the group's lowest retained rung (never below `k`), and after that latency rises.
- **Floor mode on** (`snapshot.floor_mode`, set by the controller when `room < S_floor` or `B_min > B_max`): the planner returns the group's floor rung (`PriceList::floor_ef(g)`, always `>= k`) for every query, skipping the allocator. It still debits the modeled floor work so that the bucket stays accurate. If the design requires strict `ef = k`, the price-list envelope and controller floor-work definition must use strict `k` as well.
- **Per-query Case 3** (floor mode off, but this query cannot afford its floor work): run at the group's floor rung with `forced_floor = true`.
- **Bucket debt forgiveness:** `debit()` clamps `balance = max(balance − w, −D)`. Without the clamp, sustained overload would build unbounded debt and keep forcing cuts long after the load falls. With it, the bucket is full again within T_base once the load drops.
- SLO deterioration is measured, not prevented. The metrics report SLO attainment, time in floor mode, and the latency excess (p95 − T_SLO) for infeasible loads (§14.2).

### 10.8 Planner glue (`planner.h`)

> **v2 (§1A.6):** this hook is stage 2. It uses `menu[qc.probe_idx].pl` instead of a single `price_list`, and that L's `water_level`/`uniform_fraction` from the current snapshot. Stage 1 (`choose_probe`) runs on the worker before the search (§1A.5, §1A.7).

This is the body of the `EfPlannerFn` the engine passes to `search_planned`:
```cpp
uint32_t plan(float score, uint32_t ef_pref, uint32_t /*probe_ndis*/, QueryCtx& qc) {
  auto snap = snapshot.load();
  const int g = price_list.group_of(score);
  AllocResult r;
  bucket.settle(qc.reservation, now_ns(), snap->B, snap->D, [&](double affordable) {
    if (snap->floor_mode) {                         // never drop; use the group's floor rung
      r.ef = price_list.floor_ef(g);
      r.planned_work = price_list.floor_work(g);
      r.forced_floor = r.planned_work > affordable;
    } else {
      r = allocator.plan({g, ef_pref, k, affordable}, *snap, price_list);
    }
    return r.planned_work;
  });
  qc.group = g; qc.forced_floor = r.forced_floor; qc.floor_mode = snap->floor_mode;
  qc.planned_work = r.planned_work;
  return std::max(k, r.ef);
}
```
The reservation refund, affordability check, and planned-work debit happen under **one** bucket lock. The charged reservation is the amount actually removed at arrival after the `-D` clamp. An incomplete probe uses the same settlement lock to refund its charged reservation and debit its measured `ndis`.

**Acceptance (Phase 4):** the unit tests in §13.1 pass, including the MF §8 walkthrough numbers.

---

## 11. Phase 5: Engine and workload

### 11.1 Workload (`workload.h`, `workload.cpp`)

`std::vector<int64_t> make_arrivals(const WorkloadCfg&, uint64_t seed)` returns arrival offsets in ns, and `pick_queries(n, n_eval, seed)` chooses evaluation query indices with replacement. Profiles:
- `poisson {qps}`
- `constant {qps}` (evenly spaced arrivals, gap = 1/qps; used by the SLO study)
- `step {phases: [{qps, duration_s}]}` (the burst pattern of MF §8, in queries per second)
- `mmpp2 {qps_lo, qps_hi, mean_lo_s, mean_hi_s}`
- `trace {csv}`

The `poisson` profile draws exponential gaps between arrivals, so arrivals form a Poisson process with rate λ. The design documents (Main Flow, Budget B) do not name an arrival distribution, so this is our assumption. A perfectly regular schedule would understate queueing and tail latency. The `step` profile reproduces the piecewise-constant rates of MF §8. `mmpp2` (E3) and `trace` are experiment-only profiles that the design documents do not describe.

Rates are absolute arrival rates λ in queries per second (`qps`). The engine never sees a load factor. Experiments write `rho`, a multiple of the profiled saturation throughput μ_sat(C) from `profile_C{C}.json`, and `scripts/run_experiments.py` converts it to `qps` once per run, so every `run_config.json` holds absolute rates. Lengths are given as `n_queries` or `duration_s`. Warm-up queries (`workload.warmup_queries`, default 1000) run but are excluded from the metrics.

### 11.2 Engine (`engine.h`, `engine.cpp`)

Threads: 1 generator, C workers, 1 controller. With `--pin-cores`, pin them to cores 0 (generator), 1 (controller) and 2..C+1 (workers), and warn if there aren't enough cores.

- **Generator.**
  - Open-loop. For each scheduled arrival, `sleep_until` when the gap is > 200 µs, otherwise spin.
  - Stamp `t_arrival` with the **scheduled** time, which avoids coordinated omission. Log the generator lag separately.
  - Count the arrival for λ̂ and reserve nominal work, retaining the actual charged reservation after the `−D` clamp (always; no admission check exists).
  - Push `{qid, eval_idx, t_arrival, charged_reservation}` onto the dispatcher queue (`std::deque` + mutex + condvar, with an atomic size for Q).
- **Worker.**
  - Pop a query. **v2:** take `L = planner.choose_probe(qc)` from the current snapshot (§1A.7).
  - Set `t_start`. Call `backend.search_planned(q, k, L, planner, ids, &st)` for LA-ef and Ada-ef (Ada-ef always uses the full L), or `search_fixed` for static-ef. Set `t_end`.
  - If the backend reports `probe_incomplete`, call the planner's measured-work settlement path before recording the query, so the charged reservation is refunded and actual `ndis` is debited.
  - Record S_i = t_end − t_start, W_q = t_start − t_arrival and T_i = t_end − t_arrival.
  - Update telemetry: u with S_i/ndis_i, E[S] with S_i. Push (T_i, S_i) into the sensor window. Write the `QueryRecord` into its preallocated slot and the IDs into `ids[qid*k ...]`.
- **Controller thread.**
  - Every 1 ms, check whether the window closed. Every 10 ms, update λ̂.
  - When the window closes: build `ControllerInputs`, call `controller.update`, compute D, w_target, water_level and uniform_fraction, publish a new `PlanSnapshot`, and append one row to the time series.
  - The initial snapshot uses B0 = B_max from the profile's u at C threads.
- **Shutdown.** After the last arrival, wait until the queue is empty and all workers are idle, then join.
- **Post-run.** Compute recall@k per completed query against `gt/neighbors.npy` (first k columns). Write the outputs (§11.3).

### 11.3 Outputs (`results_writer.cpp`), in `results/<experiment>/<run_id>/`
- `config.json`: the fully resolved config, including the auto SLO and the absolute rates.
- `queries.csv`: `qid, eval_idx, warmup, t_arrival_us, wait_us, service_us, latency_us, status (ok|probe_incomplete), score, group, ef_pref, ef_used, forced_floor, floor_mode, ndis, probe_ndis, planned_work, recall`
- `timeseries.csv`: one row per window: `t_s, B, B_max, B_min, D, p95_ms, s_p95_ms, Q, W_hat_ms, u_ns, ES_ms, lambda_hat, w_target, water_level, uniform_fraction, zone, guard_fired, floor_mode, slo_violated`
- `summary.json`: §14.2 metrics, already computed in C++ so the analysis can be done with pandas alone.
- **v2 (§1A.3 C10):** `queries.csv` and `timeseries.csv` gain a final column `probe_len`; `summary.json` gains `mean_probe_len` and `frac_full_probe`. The columns are appended at the end, so the §11.4 readers keep working.

### 11.4 Exact output contract read by the analysis notebooks (✅ notebooks written against it)

The notebooks are self-contained and read only these files, so the C++ and `run_experiments.py` code must produce exactly these names. Booleans in CSV files are written as 0/1. Times are in the units given by the column suffix.

| File | Required content |
|---|---|
| `tables/calibration_sweep.csv` | `query, score, ef_pref, ef, recall, ndis, probe_ndis, service_us, is_pref`. **`probe_ndis` is a new column**, needed for the E0 probe-share plot (finding 4). Preference-pass rows have `is_pref = 1`; ladder-sweep rows have `is_pref = 0`. |
| `tables/price_list.json` | The §9.4 schema. Keys used: `k, target_recall, ladder, static_ef_for_target, floor_work_mean, nominal_work_mean, calib_mean_recall_at_pref, groups[id, score_lo, score_hi, p, ef_pref_median, ladder, recall, work], steps_sorted[group, from, to, d_recall, d_work, loss]`. |
| `profile/profile_C{C}.json` | `{"threads": C, "k": k, "M": M, "adaef": {"dataset", "metric", "k", "M", "target_recall", "quantile_step", "statics_length", "ef_upper_bound"}, "idle": {"mean_s","p50_s","p95_s","p99_s","u_s_per_dc","mean_ndis"}, "saturation": {"mu_sat_qps","u_s_per_dc","mean_service_s","s_p95_s"}}` |
| `<run>/config.json` | The resolved `RunConfig` (`to_json` in `config.h`) **plus** top-level `t_slo_s`, `repeat`, `dataset` and `policy`. `workload.phases[*].qps` must hold the **absolute** rate after ρ is resolved. |
| `<run>/queries.csv` | `qid, eval_idx, warmup, t_arrival_us, wait_us, service_us, latency_us, status, score, group, ef_pref, ef_used, forced_floor, floor_mode, ndis, probe_ndis, planned_work, recall`. **`warmup` and `probe_ndis` are new columns.** `status` ∈ {ok, probe_incomplete}. `group = -1` and `score = -1` for static-ef. Every query has a recall value. |
| `<run>/timeseries.csv` | Columns as in §11.3: `t_s, B, B_max, B_min, D, p95_ms, s_p95_ms, Q, W_hat_ms, u_ns, ES_ms, lambda_hat, w_target, water_level, uniform_fraction, zone, guard_fired, floor_mode, slo_violated`. Rows are written for **every** policy, including `controller.type = none`, so baselines get p95/Q time series too. |
| `<run>/summary.json` | Keys: `n_queries, n_ok, n_probe_incomplete, offered_qps, achieved_qps, goodput_qps, slo_ms, p50_ms, p95_ms, p99_ms, mean_wait_ms, mean_service_ms, slo_attainment, latency_excess_ms, mean_recall, p5_recall, p1_recall, frac_target_recall, mean_ef_ratio, mean_ndis, forced_floor_rate, floor_mode_query_rate, n_windows, frac_windows_violation, frac_windows_headroom, frac_windows_comfort, guard_fires, floor_mode_windows, p95_window_violations, recovery_time_s (null unless step profile), controller_update_us, generator_lag_p99_us, wall_s`. All are computed over non-warmup queries. `p5_recall` / `p1_recall` are the 5th / 1st percentiles of per-query recall. |
| `results/<exp>/summary.csv` | One row per run: `run_id, experiment, dataset, policy, threads, slo_fraction, profile, rho, repeat, axis, axis_value` followed by every `summary.json` key. `run_id` is the run's directory name under `results/<exp>/`. For step and MMPP profiles, `rho` is the peak ρ. For E1–E3, `axis = base`. For E4, the shared baseline runs have `axis = base`, and each varied run has `axis` = the dotted config key (e.g. `controller.guard`) and `axis_value = str(python value)` (e.g. `True`, `500`, `2.0`, `p_aimd`). E5 runs go to `results/e5_threads/summary.csv` with `axis = threads`. |

Figures are written to `results/figures/<experiment>/`.

---

## 12. Phase 6: Apps, configuration and scripts

**P6.1 `config.h`.** Defines `RunConfig` with nested structs mirroring `configs/defaults.yaml` (§15), serialized with nlohmann::json. Precedence: `defaults.yaml` (converted to JSON by `run_experiments.py`), then `--config run.json`, then CLI flags.

**P6.2 Apps (CLI11).**

| App | Key flags |
|---|---|
| `laef_build_index` | `--dataset-dir data/artifacts/<id> --M 16 --efc 500 --threads N` |
| `laef_calibrate` | `--dataset-dir … --k 10 --target-recall 0.95 --quantile-step 1e-3 --threads N [--ladder-ratio 2]`. Rebuilds or validates the adapter compatibility sidecar. |
| `laef_profile` | `--dataset-dir … --threads C [--n 2000] [--sat-seconds 5] [--k 10 --target-recall 0.95 --quantile-step 1e-3 --statics-length 0 --ef-upper-bound 5000]`. (1) Idle: 1 thread, sequential, Ada-ef passthrough → S mean/p50/p95/p99, u_idle, mean ndis. (2) Saturation: C threads, closed loop, Ada-ef passthrough → μ_sat(C), u_sat(C). Writes `profile_C{C}.json` with compatibility metadata. |
| `laef_bench` | `--dataset-dir … --config run.json --threads C [--slo-ms X] [--qps Q] [--n-queries N \| --duration-s S] [--policy static_ef\|ada_ef\|laef] [--allocator uniform_cut\|water_filling] [--controller p_aimd\|fixed_aimd\|static_capacity\|none] [--floor-mode on\|off] [--pin-cores] --out results/<exp>/<run>` |

`laef_bench` refuses to run without `profile_C{C}.json` for the requested C, and prints the command that creates it.

**P6.3 Policies (presets that expand to allocator, controller and floor mode). No policy ever drops a query:**

| Policy | Allocator | Controller | Bucket | Floor mode |
|---|---|---|---|---|
| `static_ef` | passthrough (static ef) | none | no | off |
| `ada_ef` | passthrough | none | no | off |
| `laef_uniform` | uniform_cut | p_aimd | yes | on |
| `laef_wf` (main) | water_filling | p_aimd | yes | on |

`--floor-mode off` exists only as an ablation. With it off, the controller still clamps B at B_min, but queries keep going through the allocator instead of being forced to ef = k.

The individual flags override the preset, which is how the ablations are built.

**P6.4 Scripts.**
- `prepare_dataset.sh <id>`: runs `run_preprocess.py --steps download,convert --device cpu`, verifies the fingerprint against the Colab `manifest.json`, and copies `gt/`, `calibration/` and `stats/` from a given path.
- `run_pipeline.sh <id> <C>`: runs build_index, then calibrate, then profile.
- `run_experiments.py <experiment.yaml>`: expands the grid (datasets × policies × loads × SLOs × C × repeats), skips runs whose `summary.json` exists, invokes `laef_bench`, and concatenates all summaries into `results/<exp>/summary.csv`.

---

## 13. Phase 7: Tests

Run all tests with `scripts/build.sh && ctest --output-on-failure` and `pytest gpu_preprocessing/tests`.

### 13.1 C++ unit tests (GoogleTest, `cpp/tests/`)

| File | What it checks |
|---|---|
| `test_p_aimd.cpp` | **MF §8 walkthrough**: C=4, u=1 µs, T_SLO=10 ms, B=4M, λ̂=2000, c̄(k)=1200. Phase 3 inputs: p95=10.5 ms, Q=16, E[S]=2 ms, S_p95=4 ms. Expect the proportional cut to give 3.8095M, then the guard ×0.5 to give 1.905M, then the clamp to give **B = 2.4M**, with `floor_mode = false` (room 2 ms ≥ S_floor 1.2 ms). Also: headroom → ×1.02 capped at B_max; comfort band → unchanged; p95 = 25 ms → ×0.5 floor; room < S_floor → `floor_mode = true` and B = min(B_max, B_min); B_min > B_max → B = B_max and `floor_mode = true`; floor mode clears in the next window once the room recovers. |
| `test_fixed_aimd.cpp` | Fixed 10% cut regardless of severity; same clamps. |
| `test_token_bucket.cpp` | Refill at B, overdraft floor −D (debit clamps, so debt beyond −D is forgiven), cap 0. **MF §8 Phase 2:** B = 4M, D = 24,000, a demand of 8M dc/s drains the credit in ≈ 6 ms (±5%). After sustained overload, the bucket is full again within T_base of the demand stopping. |
| `test_price_list.cpp` | JSON load; group lookup boundaries; interpolation; concave envelope; `water_level` = 0 when not overloaded, the correct step for a hand-built 2-group example, and +inf when even the floor is too much. |
| `test_allocators.cpp` | uniform gives the same work fraction to every group; water_filling cuts the cheapest-ℓ group first and leaves the hard group's ef alone in the hand-built example; neither goes below k; the bucket check forces a step down and sets `forced_floor`. |
| `test_sensor_window.cpp` | Exact p95 against a sorted reference; the close-on-n / close-on-time-with-min-samples rules. |
| `test_formats.cpp` | **Merged into `test_backend_parity.cpp`** (`AdaEfFormats.PythonFixturesReadByAuthorsCode`, ✅). It reads the `cpp/tests/fixtures/` files written by `test_formats.py` using the authors' deserializers. |
| `test_backend_parity.cpp` | ✅ A unity TU (`#include "../src/adaef_backend.cpp"`, not linked with `laef_adaef`). On a 6k × 32 clustered fixture it checks: `search_planned` with a passthrough planner gives the same IDs **and score** as the authors' `adaptiveSearchKnn` (with `setEf(k)`); `search_fixed` gives the same IDs as `setEf + searchKnn` for ef ∈ {10, 40, 160}; the planner's ef is honoured (clamped ≥ k) and the probe is identical for every plan; 8 threads give identical results and ndis. |

### 13.2 Python tests (`gpu_preprocessing/tests/`, CPU only)
- `test_ground_truth.py`: `exact_knn` matches a numpy brute force for `cd` and `ipd` on 5k × 32 random data. The results are invariant to `q_batch`/`base_chunk` (including chunks that don't divide n). The holdout ID mapping is checked.
- `test_formats.py`: write/read round-trips for `samplings.bin` and `estimator.bin`. The covariance matches `np.cov(X, rowvar=False, ddof=1)`. It also writes the fixtures for `test_formats.cpp`.

### 13.3 QPS × query-count tests per dataset (`cpp/tests/test_engine_qps.cpp` + `configs/tests/qps_matrix.yaml`)

The cases are parameterized from `qps_matrix.yaml`. The fake-backend cases always run. A real-dataset case calls `GTEST_SKIP()` when its `data/artifacts/<id>/` files or profile are missing. The dataset files are NumPy `.npy` (`train.npy`, `test.npy`, §4.3), not HDF5. The skip check also requires `index/hnsw_M16_efc500.bin`, `tables/price_list.json` and `profile/profile_C4.json`.

```yaml
# configs/tests/qps_matrix.yaml
- {dataset: fake,              threads: 4, n_queries: 20000, rho: [0.3, 1.0, 1.5]}
- {dataset: fake,              threads: 4, n_queries: 20000, infeasible_step: [3.0, 0.3]}   # floor mode, no drops
- {dataset: synthetic-small,   threads: 4, n_queries: 5000,  rho: [0.3, 1.0, 1.5]}
- {dataset: glove-100-angular, threads: 4, n_queries: 10000, rho: [0.5, 1.5]}
- {dataset: deep-image-96-angular, threads: 4, n_queries: 10000, rho: [0.5, 1.5]}
- {dataset: cluster-zipf-100d, threads: 4, n_queries: 10000, rho: [0.5, 1.5]}
- {dataset: coco-i2i-512-angular, threads: 4, n_queries: 10000, rho: [0.5, 1.5]}
- {dataset: coco-t2i-512-angular, threads: 4, n_queries: 10000, rho: [0.5, 1.5]}
- {dataset: lastfm-64-dot, threads: 4, n_queries: 10000, rho: [0.5, 1.5]}
```

Assertions for each case, with policy `laef_wf` and slo `auto`:
1. **Accounting (no drops):** `ok + probe_incomplete == n_queries`, every query has a completion timestamp, and `queries.csv` has `n_queries` rows.
2. **Offered QPS:** n_queries / (last scheduled arrival − first) is within ±5% of the target. The generator lag p99 is < 1 ms.
3. **Light load (ρ ≤ 0.5):** p95 ≤ T_SLO; floor mode never on; ≥ 99% of queries have `ef_used == ef_pref`, meaning LA-ef behaves exactly like Ada-ef when it is not overloaded.
4. **Feasible overload (ρ = 1.5, with B_min < B_max in the run):** B_min ≤ B ≤ B_max in every window; mean `ef_used` < mean `ef_pref`; p95 ≤ 1.2·T_SLO after the first 10 windows. The run terminates (no deadlock) within 3× its scheduled duration.
5. **Infeasible overload (fake backend only, ρ chosen so that λ·c̄(k) > B_max):** all `n_queries` complete (no drops); floor mode is on in ≥ 90% of the windows after warm-up; ≥ 95% of queries run at `ef_used == k`; p95 > T_SLO is **allowed** and is reported; once the load stops, floor mode clears and B recovers (checked by a step profile 3.0 → 0.3 ρ).
6. For each run, `summary.json` exists and recall ∈ [0, 1].

---

## 14. Phase 8: Experiments and analysis

### 14.1 Experiment matrix (`configs/experiments/*.yaml`)

> **Superseded (Pass 10):** the E1–E4 grids below are no longer configured. The experiments are now the SLO study in `scripts/run_slo_study.py`. It saves one row per 1 s time bin for every run, and it deletes the per-query file after building those rows.

Defaults: k = 10, target recall 0.95, C = `--threads` (record the core count of the machine), 3 repeats, SLO multipliers m ∈ {3, 5} over the idle single-thread S_p95, converted to ms by `run_experiments.py` (the engine only sees ms).

| Exp | Question | Grid |
|---|---|---|
| **E0 Calibration** | Does Ada-ef hit its target at idle? What do the price lists look like? | 4 datasets; `laef_calibrate` outputs only. Plots of r̄_g(e), c̄_g(e), ℓ distribution, p_g. |
| **E1 Load sweep** | How do recall and latency change as load grows? | 7 datasets × {static_ef, ada_ef, laef_uniform, laef_wf} × Poisson ρ ∈ {0.5, 0.8, 1.0, 1.2, 1.5, 2.0} × m ∈ {3, 5} |
| **E2 Step burst** | Transient behaviour (MF §8) | glove + deep × {ada_ef, laef_uniform, laef_wf} × step 0.3 → 1.5 → 0.3 ρ (5 s / 5 s / 5 s), m = 5 |
| **E3 Bursty traffic** | Realistic bursts | glove + deep × same policies × MMPP-2 (ρ_lo 0.4, ρ_hi 1.6, mean 1 s / 0.3 s) |
| **E4 Ablations** | Which parts matter? | glove, ρ ∈ {1.2, 2.0}: controller {p_aimd, fixed_aimd, static_capacity}; guard on/off; floor mode on/off; window n ∈ {250, 500, 1000}; growth ∈ {1.01, 1.02, 1.05}; ladder ratio {√2, 2} |
| **E5 Threads** | Hardware contention | glove + deep × laef_wf × C ∈ {1, 2, 4, …, cores}, ρ = 1.2. Reports u(C) and B_max(C). |

### 14.2 Metrics (computed in `summary.json`; definitions in `analysis/laef_analysis/metrics.py`)
- **Latency:** p50, p95 and p99 of T over all queries (every query is served); SLO attainment = fraction of queries with T ≤ T_SLO; latency excess p95 − T_SLO when it is positive (infeasible loads).
- **Throughput:** goodput = queries per second completed within the SLO; achieved throughput.
- **Quality:** mean recall; P5 and P1 recall (the hardest 5% and 1%, the Ada-ef paper's tail metric); fraction of queries reaching the target recall.
- **Effort:** mean ef_used/ef_pref, mean ndis, forced-floor rate.
- **Control:** time spent in each zone; guard-fire count; recovery time after a burst (time from the burst ending until B ≥ 0.95·B_max); number of p95 window violations; controller overhead (µs per update).
- Headline comparison: **laef_wf vs laef_uniform at matched SLO attainment**, comparing effective recall and P5/P1 recall. This tests the claim that water-filling protects recall better than equal cuts.

### 14.3 Analysis (`analysis/`)
- `io.py`: loads a run directory, or concatenates `summary.csv` files.
- `metrics.py`: recomputes metrics from `queries.csv`, as a cross-check of the C++ numbers.
- `plots.py`:
  - recall-vs-ρ and p95-vs-ρ per dataset (E1);
  - B(t), p95(t), Q(t) and floor_mode(t) time series with the burst phases shaded (E2/E3);
  - per-group ef_used histograms (shows who absorbs the cuts);
  - ablation bar charts (E4);
  - u(C) curve (E5).

The notebooks `01`–`04` (✅ written) are **self-contained**: each one reads the §11.4 files directly with pandas and matplotlib, so they work without `laef_analysis`. `laef_analysis` is still useful for scripted cross-checks (e.g. `metrics.py` recomputing `summary.json` from `queries.csv`), but the notebooks do not import it.

| Notebook | Experiments | Main outputs |
|---|---|---|
| `01_calibration.ipynb` | E0 | summary table (incl. ±0.02 target check, floor/nominal, probe share), per-group recall/work/envelope curves vs raw sweep means, loss-ratio histograms, demand-vs-water-level curve, group shares, Ada-ef operating point (ef_pref, recall P5/P1), static-ef curve |
| `02_load_sweep.ipynb` | E1 | recall / P5 / p95÷T_SLO / attainment / ef ratio vs ρ per (dataset, m); goodput vs offered QPS; floor-mode rate; **headline wf-vs-uniform table at matched attainment** (`MATCH_TOL = 0.02`); recall–attainment frontier |
| `03_burst_timeseries.ipynb` | E2, E3 | per-run panels of arrivals, B (with B_max/B_min), window p95, Q, water level and floor mode, with the burst shaded; rolling per-query p95 and recall; per-group ef_used/ef_pref and recall during the burst; summary over repeats (incl. recovery time) |
| `04_ablations.ipynb` | E4, E5 | bar grid per ablation axis (baseline marked ★); effect sizes vs baseline; B(t)/p95(t) per controller; u(C), B_max(C), μ_sat(C) from the profiles; LA-ef metrics vs C |

---

## 15. Configuration reference (`configs/defaults.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `k` | 10 | neighbours per query |
| `target_recall` | 0.95 | Ada-ef declared recall |
| `hnsw.M` / `hnsw.ef_construction` | 16 / 500 | the authors' build settings |
| `adaef.quantile_step` | 1e-3 | the paper's score quantile step; decay mode and bin count are fixed to the paper implementation |
| `adaef.statics_length` | 0 (auto: 1 + 2M + (2M−1)·2M = 1025) | full probe size (stock Ada-ef) |
| `adaef.probe_lengths` | [1025, 512, 256, 128] | v2 probe-length menu (§1A.4). Descending, each ≥ 64; the largest must equal the full probe size |
| `probe.enabled` | true | v2 stage 1. `false` restricts the menu to the full L (the v1 ef-only ablation) |
| `adaef.ef_upper_bound` | 5000 | the authors' cap |
| `calibration.n` / `source` / `seed` | 1000 / base_sample / 42 | calibration queries |
| `calibration.ladder_ratio` | 2.0 | ef ladder spacing |
| `calibration.min_group_size` / `max_groups` | 30 / 8 | difficulty grouping |
| `engine.threads` | required (`--threads`) | C |
| `engine.pin_cores` | false | |
| `slo.ms` / `slo.multiplier` | null / 5 | the engine takes `ms` only (required). `multiplier` is turned into ms by `run_experiments.py` from the idle S_p95 |
| `window.n` / `max_ms` / `min_samples` | 500 / 20 / 50 | sensor window |
| `telemetry.alpha_u` / `alpha_es` / `alpha_lambda` / `lambda_tick_ms` | 0.05 / 0.05 / 0.2 / 10 | EWMAs |
| `controller.type` | p_aimd | p_aimd \| fixed_aimd \| static_capacity \| none |
| `controller.cut_floor` / `headroom` / `growth` / `guard` | 0.5 / 0.85 / 1.02 / true | P-AIMD (BB §3) |
| `controller.fixed_cut` | 0.9 | fixed_aimd |
| `controller.rho_target` | 0.9 | static_capacity |
| `bucket.t_base` / `cap_pos` | mean / 0 | T_base = T_SLO − E[S] (mean) or − S_p95 (tail) |
| `allocator.type` | water_filling | water_filling \| uniform_cut \| passthrough |
| `allocator.uniform_mode` | work | work \| ef |
| `floor_mode.enabled` | true | Never drops; set to false only for the ablation |
| `workload.profile` / `qps` / `n_queries` / `warmup_queries` / `seed` | poisson / required / 20000 / 1000 / 7 | |

---

## 16. Risks and decisions to keep in mind

| Risk | Mitigation |
|---|---|
| Probe cost is large relative to the floor (≈1025 dc of the probe sits inside c̄(k)) | Expected and correct: B_min and S_floor include it. Report `probe_ndis / ndis` in E0. |
| The guard's one-step factor assumes S_p95 ∝ B, but water-filling protects hard queries (BB §2.2 caveat) | Safe because of the B_min clamp and the bucket check. Measured in E4 (guard on/off × allocator). A possible refinement is to use the measured ∂S_p95/∂B. |
| p95 is noisy on slow queries (few completions per 20 ms) | `window.min_samples`; E4 sweeps `window.n`. |
| WSL2 timing jitter | Use `--pin-cores`; report the machine and kernel; repeat 3×; prefer native Linux for the final numbers if one is available. |
| `EfAdapter` construction calls `setEf` (not thread-safe) | Calibration builds the adapter single-threaded. Serving never calls `setEf` (§8.2, change 1). |
| ann-benchmarks GT could disagree with our cosine definition | `--verify-provided` (P1.6) fails loudly if it does. |
| Colab disk or session limits | Streaming GT/stats; idempotent steps; outputs written straight to Drive. |
| Holdout/ID mismatch between Colab and the serving machine | Manifest fingerprint check in `prepare_dataset.sh`. |

---

## 16.1 Scoped core-engine audit (2026-11-03)

**Scope:** static review of `Implementation/cpp/{include,src,apps,tests}` against `Main Flow.md` and `Budget B calculations.md`. No other project folders were inspected. The P-AIMD equations, physical clamps, water-level calculation, complete-probe allocator path, queue synchronization, and normal shutdown path were internally consistent. The following fixes were applied in the C++ core; full Ada-ef compilation and real-dataset validation remain pending because the WSL environment lacks Eigen and Boost.

### Applied fixes

1. **Fixed reservation accounting at the overdraft floor.**
   `TokenBucket::reserve()` now returns the amount that actually reduced the balance. `Planner` and the dispatcher task carry that charged amount, and settlement refunds only that amount. Debt forgiveness at `-D` remains unchanged. `test_token_bucket` now covers multiple clipped reservations and passes.

2. **Fixed incomplete-probe settlement.**
   An incomplete backend probe now reports its measured `ndis`, and the engine calls `Planner::on_incomplete()`. That path refunds the charged reservation and debits the measured search work under the bucket lock. The query remains marked `probe_incomplete`; it is never silently treated as a planned query. `test_price_list` now covers the settlement path.

3. **Fixed configuration scope and enforced artifact compatibility.**
   The implementation uses the Ada-ef paper's fixed decay mode and five-bin score calculation; `num_bins` and `decay` were removed from `RunConfig` and are no longer accepted as tunables. The remaining score/search settings are recorded in adapter metadata, price-list metadata, and profile metadata. Calibration rebuilds or rejects an adapter with incompatible metadata, and `laef_bench` fails instead of warning when profile, price-list, adapter, dataset, or run settings disagree.
   - Keep the paper constants internal to the backend.
   - Validate dataset, metric, `M`, `k`, target recall, quantile step, effective `statics_length`, and `ef_upper_bound` before serving.
   - Keep the adapter sidecar metadata next to `ada_ef_adapter.bin`.

4. **Fixed tail-profile initialization.**
   `laef_bench` now uses `profile.saturation.s_p95_s` for the initial tail-based bucket slack. Idle p95 remains separate and is used only when deriving an SLO externally.

### Documentation consistency

The implementation intentionally uses `PriceList::floor_ef(g)`, which can be above `k` when the `k` rung is dominated on the envelope. Update §10.7 and the acceptance text to call this the **group floor rung** (always `>= k`), or change the code and model to use strict `k` everywhere. The budget model, actual planner, and documentation must choose the same floor definition.

### Remaining validation

The core-only WSL build and focused token-bucket, allocator, planner, and fake-engine tests pass. The full Ada-ef build still requires Eigen and Boost in WSL. After installing those dependencies, compile `laef_adaef` and the applications, run `test_backend_parity`, and exercise one real dataset through `laef_profile` and `laef_bench`. The current real-dataset QPS tests still skip before running the engine.

---

## 17. Implementation checklist (do these in order)

1. [ ] P0.1–P0.3 environment, submodule, CMake builds
2. [ ] P1.1–P1.9 `gpu_preprocessing` rewritten in the verification pass (not run); re-run `pytest gpu_preprocessing/tests` on WSL
3. [ ] P1.10 Colab notebook; artifacts for the 4 datasets on Drive (🟡 notebook written; needs `run_preprocess.py` per §7.1, then a Colab run)
4. [x] P2.1–P2.5 backend + `test_backend_parity` (includes the format check) green
5. [ ] `prepare_dataset.sh` + `laef_build_index` for synthetic-small and glove (laef_build_index written, not compiled or run)
6. [ ] P3.1–P3.5 `laef_calibrate` + `test_price_list` green; E0 sanity (recall ≈ 0.95)
7. [x] §10 online components + their unit tests (including the MF §8 numbers) green
8. [ ] §11 engine + fake backend; reservation clipping and incomplete-probe settlement are implemented and core-tested; run the full fake/synthetic matrix after the Ada-ef build is available
9. [ ] §12 `laef_profile`, `laef_bench`, `run_experiments.py`; metadata validation and saturation-tail initialization are implemented; build/link the apps and run `configs/experiments/smoke.yaml` end to end
10. [ ] Remaining datasets: index → calibrate → profile; the real-dataset rows of `test_engine_qps` green
11. [ ] E1–E5 runs; `analysis` notebooks produce all plots (🟡 notebooks written against §11.4; need real results)

### 17.1 v2 checklist (probe-length lever, §1A). Do these in order

12. [ ] C1–C4: config keys, backend interface and per-L adapters, fake backend. Update `test_backend_parity` to pass `probe_len = 1025`; it must still match the authors' code.
13. [ ] C5: `ProbeMenu`, `expected_at`, `probe_menu.json` load/save; add the single `ProbeMenu.Choose` test (§1A.8).
14. [ ] C6: per-L calibration and the dominance filter. Calibrate lastfm and check `probe_menu.json` (§1A.8 item 5).
15. [ ] C7–C10: snapshot, two-stage planner, engine, outputs.
16. [ ] C11–C12: bench, profile and `run_pipeline.sh`. One short ρ = 0.5 and ρ = 1.5 bench run on lastfm (§1A.8 item 6).
17. [ ] Run `commands.md` in order for all datasets, then the SLO study.
