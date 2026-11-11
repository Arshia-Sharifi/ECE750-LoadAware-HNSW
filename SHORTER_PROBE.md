# How LA-ef chooses and uses the shorter probe

Status: **multi-probe retention ported to the `Arshia` working tree.** Source: `main` at `325a361`; notebook branch base: `4587eec`. There are two separate decisions: **offline calibration decides what can be used; runtime planning decides what to use now.**

## 1. What a shorter probe does

The probe is the beginning of the HNSW base-layer traversal. It collects distances for Ada-ef’s query difficulty score. The same traversal then continues at the selected `ef`, using the candidates already found. A shorter probe lets the planner choose `ef` earlier and can leave more of the work allowance for the rest of the search. It does not cap total work at the probe length: scoring earlier changes the sample, the preferred `ef`, and the traversal, so total work and recall can move either way.

With the default graph degree `M = 16`, the full probe is 1025 samples. Both branches default to candidates `1025,512,256,128`. [Defaults](configs/defaults.yaml#L14) · [Full-length formula](cpp/include/laef/ann_backend.h#L42) · [Traversal](cpp/src/adaef_backend.cpp#L88).

## 2. Offline: calibrate candidates and retain every qualifying shorter probe

Calibration builds a separate Ada-ef adapter for each candidate length, then runs a preference pass and sweeps `ef` rungs on sample queries. The resulting price list estimates recall and **total** distance computations, including the probe, by difficulty group and `ef`. The aggregate model weights groups by their calibration share and uses each group’s median preferred `ef` for nominal demand. The floor is each group’s lowest retained rung, which is at least `k`. A failed shorter calibration can be dropped; a failed full calibration aborts. [Per-length adapter and sweep](cpp/apps/laef_calibrate.cpp#L145) · [Sweep implementation](cpp/src/calibration.cpp#L93) · [Aggregate model](cpp/include/laef/price_list.h#L125).

Current `main` always keeps the full probe, then visits shorter candidates in descending order. It keeps a candidate only if **both** its modeled nominal work and its modeled floor work are strictly lower than those of the **last retained** probe:

```cpp
// Core retention test; rejection logging omitted.
for (auto& o : options) {
  if (kept.empty()) {
    kept.push_back(std::move(o));
    continue;
  }
  const double nom = o.price_list.model_nominal_work();
  const double flo = o.price_list.model_floor_work();
  const auto& last = kept.back();
  if (nom < last.price_list.model_nominal_work() &&
      flo < last.price_list.model_floor_work()) {
    kept.push_back(std::move(o));
  }
}
```

If 512 passes but 256 fails, 128 is compared with 512. The menu can contain zero, one, or several shorter probes. This filter compares work only; it does not establish that rejected probes are dominated across the full recall–work curve. The retained options and rejection reasons go into `tables/probe_menu.json`; files for a rejected length can still exist on disk without being used. [Retention rule](cpp/apps/laef_calibrate.cpp#L231) · [Menu load](cpp/src/price_list.cpp#L133).

Before the port, the notebook branch evaluated all shorter candidates at one budget (the full probe’s modeled floor), used water filling to estimate work and recall there, and kept at most one shorter probe. The port replaces that single-anchor test.

## 3. Online: choose a length when the controller publishes its plan

The controller publishes a plan with `w_target = B / estimated_arrival_rate`, an average work allowance in distance computations per query. For each retained length, `expected_at(w_target, kind)` predicts work and recall using the **active policy’s** model. `ProbeMenu::choose()` ignores options exceeding the allowance by more than 0.1%, takes the highest predicted recall among those that fit, and favors the longer probe on near ties. If none fits, it picks the shortest retained option. A shorter probe can win even when the full probe is affordable. [Budget and plan](cpp/include/laef/plan_snapshot.h#L45) · [Runtime selection](cpp/include/laef/probe_menu.h#L55) · [Policy models](cpp/include/laef/price_list.h#L194).

```cpp
if (kind == CutKind::Passthrough) s.probe_idx = 0;
else if (o.floor_mode) s.probe_idx = 0;
else s.probe_idx = menu.choose(s.w_target, kind);
```

Index `0` is the full probe. **The no-fit fallback and floor mode are different:** no-fit selection tries the shortest retained probe; floor mode sends newly starting searches to the **full probe** and uses the applicable group’s floor `ef`. P-AIMD bases its floor check on full-probe floor work: it can enter floor mode if the estimated remaining time cannot cover that work or minimum demand exceeds estimated capacity. That does not prove a shorter probe would also be infeasible. Stock Ada-ef uses the full probe; the fixed-`ef` baseline bypasses the Ada-ef probe. Setting `probe.enabled=false` leaves just the full probe for an `ef`-only ablation. [Plan override](cpp/include/laef/plan_snapshot.h#L57) · [Floor inputs](cpp/src/engine.cpp#L294) · [Floor triggers](cpp/include/laef/controllers/p_aimd.h#L39) · [Ablation switch](cpp/apps/laef_bench.cpp#L104).

## 4. Per query: run the selected probe, then choose ef

1. As a worker starts a query, it reads the current plan and saves the selected probe index in the query context. Queries starting under the same plan share a length. This happens **before** their difficulty is known. [Worker call site](cpp/src/engine.cpp#L225) · [Probe selection](cpp/include/laef/planner.h#L42).
2. The backend uses the adapter corresponding to that length. Once the sample is collected, it computes the score and looks up Ada-ef’s preferred `ef`, then calls the planner hook. [Adapter lookup](cpp/src/adaef_backend.cpp#L243) · [Probe-to-ef transition](cpp/src/adaef_backend.cpp#L138).
3. The hook uses the price list for the probe the query **actually ran**, and reads the latest plan’s cut parameters for that length. Uniform cutting targets a common fraction of preferred estimated work; water filling takes cuts below the shared recall-loss-per-work threshold. The token bucket can force further cuts. If floor mode turns on while a shorter probe is already running, the hook can apply the floor rung for that shorter probe; it does not restart the probe. If traversal ends before collecting the sample, the engine records an incomplete probe and settles its measured work. [Planner hook and settlement](cpp/include/laef/planner.h#L48) · [Uniform](cpp/include/laef/allocators/uniform_cut.h) · [Water filling](cpp/include/laef/allocators/water_filling.h) · [Incomplete probe](cpp/src/adaef_backend.cpp#L161).

## 5. Verify what happened in a run

- **Available options:** open `data/artifacts/<dataset>/tables/probe_menu.json`. On the notebook branch, Section 5 displays its `kept` and `dropped` rows. The notebook reuses an existing menu by default, so regenerate it before testing a different retention rule.
- **Selected lengths:** `summary.json` reports `mean_probe_len` and `frac_full_probe` for measured queries with a probe. These count the **selected** length, including incomplete probes. A fraction of 1 means all such queries selected the full length; it does not mean every query collected the full sample. [Summary calculation](cpp/src/results_writer.cpp#L44).
- **When it switched:** `timeseries.csv` logs the plan’s probe length, budget, and floor flag. For per-query decisions, retain `queries.csv` using `KEEP_QUERIES=True` and inspect `probe_len`, `probe_ndis`, `ndis`, `ef_pref`, `ef_used`, `status`, and `recall`. `probe_ndis` is measured work, which can differ from the sample length; `ndis` is total work. [Query output](cpp/src/results_writer.cpp#L130) · [Plan output](cpp/src/results_writer.cpp#L146).
- **What the tests cover:** `ProbeMenu.Choose` checks the runtime decision with hand-built curves. It does not tell you which candidate survives calibration or whether the shorter probe improves a particular dataset. For that, compare adaptive and full-probe-only runs with the same allocator, load, and latency target. [Selector test](cpp/tests/test_price_list.cpp#L191).

## 6. Handoff for the notebook port

- **C++ change ported:** `git show 325a361 -- cpp/apps/laef_calibrate.cpp` is the reference diff. The calibrator now matches that commit. Runtime probe selection, full-probe floor behavior, per-query `ef` selection, and the menu format are unchanged.
- **Notebook and harness:** Section 5 now explains the retention rule. Its existing display shows all retained lengths, and `laef_nb/pipeline.py` already passes the candidate lengths to the C++ calibrator. No notebook code cells or harness code changed.
- **Regenerate before comparing:** Rebuild the calibrator, explicitly recalibrate each test dataset, and use a new study name to avoid reusing completed runs. The notebook skips calibration when the menu already exists. For a fair comparison, keep the index, workload, SLO and controller settings fixed.
- **Validation to add after porting:** Check a menu where several shorter candidates qualify, one where a rejected candidate does not become the comparison reference, and one where only the full probe remains. For real data, inspect retained options and actual choices; neither multiple retained lengths nor switching at a specific load is guaranteed.
- **Existing documentation to align:** `IMPLEMENTATION_PLAN.md` §1A.4 already describes multi-probe retention; §1A.2, C7/C9 in §1A.3, §1A.5, §1A.7 and §1A.8 still include older runtime assumptions (full at light load, shortest in floor mode, or fixed outcomes at specific loads). `commands.md` steps 5 and 7 should treat retained lengths and load-specific choices as observations, not expected guarantees.

The port is an uncommitted working-tree change. Source parity and notebook syntax were checked; the C++ build, calibration, and benchmarks were not run here.
