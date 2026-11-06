#pragma once
// LA-ef: arrival schedules and evaluation-query selection. IMPLEMENTATION_PLAN.md §11.1.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "laef/config.h"

namespace laef {

// Throws std::invalid_argument unless the arrival rates are absolute and usable (qps, queries/s).
// Load factors (rho) belong to the experiment files: scripts/run_experiments.py converts them to qps
// with the dataset's measured mu_sat(C) before a run.
void check_rates(const WorkloadCfg& w);

// Scheduled arrival offsets in ns from the run start, sorted. All rates must be absolute.
//   poisson : n_queries arrivals, or arrivals until duration_s when duration_s > 0
//   constant: evenly spaced arrivals, gap = 1/qps; n_queries arrivals, or until duration_s when duration_s > 0
//   step    : phases back to back; each phase has its own qps and duration_s (qps 0 = silence)
//   mmpp2   : two-state Poisson process that starts in the low state; duration_s, or n_queries
//   trace   : one arrival time [s] per line of trace_csv (first arrival moved to 0)
std::vector<int64_t> make_arrivals(const WorkloadCfg& w, uint64_t seed);

// n_queries evaluation rows, drawn with replacement from pool (pool must not be empty).
std::vector<uint32_t> pick_queries(size_t n_queries, const std::vector<uint32_t>& pool, uint64_t seed);

// Seconds at which the last highest-rate phase of a step profile ends. Recovery is measured from
// here. Returns -1 for other profiles. Call after check_rates.
double burst_end_s(const WorkloadCfg& w);

}  // namespace laef
