# laef_nb

Helper package behind `LAEF.ipynb`. The notebook holds the decisions. This package holds the
plumbing, so that nothing in the notebook is longer than a few lines.

## Files

`config.py` is the single source of truth for paths, binary names, HNSW parameters, policy labels
and the `Study` dataclass. A `Study` describes one grid of bench runs and derives its own experiment
file path, results folder, run count and expected wall-clock cost.

`shell.py` runs a subprocess, streams its output, optionally tees it to a log file and raises on a
non-zero exit.

`system.py` reports the CPU, the physical and logical core counts, the cpufreq governor and whether
the host is WSL. `check_threads` grades each requested worker count. A count above the number of
logical CPUs is marked `block`, because the measured saturation throughput at that point is a
queueing artefact and not a capacity measurement.

`build.py` clones `hnsw-ada-ef` at the pinned commit, finds or downloads Eigen and Boost.Math as
headers, then configures, compiles and tests the C++ tree.

`datasets.py` lists the registry, reports which artifacts each dataset is missing and prepares a
dataset either locally on CPU or by importing a folder produced on Colab.

`pipeline.py` builds the HNSW index, runs the Ada-ef calibration and writes one profile per worker
count. It also reads the profiles back as a table with speedup and parallel efficiency.

`experiments.py` turns a `Study` into an experiment YAML, checks that every input exists, reports
what is still pending and launches the run through `scripts/run_slo_study.py`.

`tests/test_harness.py` is a regression suite over the whole intended grid of 3 datasets, 5 thread
counts, 5 loads, 4 SLO fractions and 3 repeats. Run it with `python -m pytest laef_nb/tests -q`.

`collect.py` loads every finished run into a tidy frame, pairs each LA-ef run with the stock Ada-ef
run at the same dataset, worker count, load and repeat, and computes the proportional cut minus
naive cut difference at matched settings. `cut_diagnostics` answers whether the allocator reduced
effort at all, which has to hold before the recall comparison means anything.

`figures.py` draws the figure set and writes it to `results/<study>/figures/`.

## Why the thread count is the organising axis

The worker count changes the measurement, not only the speed. The saturation throughput, the unit
cost per distance computation and the saturation p95 that defines `T_SLO` are all measured per
worker count and stored in `profile_C<threads>.json`. Load is then expressed as rho, a multiple of
that thread count's own saturation throughput, so each worker count is driven to the same relative
pressure. Absolute arrival rates differ between thread counts by design.

For that reason the generated experiment file emits one pair of sections per thread count and
results land in `results/<study>/c<threads>/ada/` and `results/<study>/c<threads>/laef/`. The index
and the calibration tables are shared across worker counts, so they are built once per dataset.

## Two traps worth knowing

A finished run is reused as it is. `run_experiments.py` builds the run id from dataset, policy,
thread count, SLO fraction, arrival profile, load and repeat. The run duration is not in that id, so
raising `duration_s` from 10 to 120 leaves every old run in place and the aggregated frame then
mixes run lengths. `experiments.stale` lists those runs and `launch(force=True)` reruns them.

Figures never average across SLO fractions. A `Study` with four fractions has four different
deadlines, so averaging `p95 / T_SLO` across them puts a single point on a plot whose whole purpose
is to show which side of 1 the run falls on. Each figure picks one fraction and names it in the
title.

## Arrival processes

The engine supports `constant` and `poisson` only. The `step`, `mmpp2` and `trace` profiles were
removed in commit ca82c38, together with `burst_end_s` and the `recovery_time_s` summary field and
the per-phase rows of `interval_stats.py`. `experiments.spec` rejects anything other than the two
remaining profiles rather than letting the engine fail mid-study.

## Adding a study

Copy the `Study` in the notebook, change the fields and rerun the launch cell. Nothing else needs to
change. The run folders, the aggregation and the figures all key off the same dataclass.
