#pragma once
// LA-ef: engine (generator, C workers, controller thread). IMPLEMENTATION_PLAN.md §11.2.
#include <cstdint>
#include <vector>

#include "laef/config.h"
#include "laef/types.h"

namespace laef {

class AnnBackend;
class ProbeMenu;

// One row per controller window (timeseries.csv, §11.3). Baselines without a controller get rows too,
// with zone "none" and zero budget fields.
struct TimeSeriesRow {
  double t_s = 0;
  double B = 0, B_max = 0, B_min = 0, D = 0;
  double p95_ms = 0, s_p95_ms = 0;
  double Q = 0;
  double W_hat_ms = 0, u_ns = 0, ES_ms = 0, lambda_hat = 0;
  double w_target = 0, water_level = 0, uniform_fraction = 1;
  const char* zone = "none";
  bool guard_fired = false, floor_mode = false, slo_violated = false;
  uint32_t probe_len = 0;  // probe length the window's snapshot assigns to new queries (v2)
};

struct EngineInputs {
  const RunConfig* cfg = nullptr;
  const AnnBackend* backend = nullptr;
  const ProbeMenu* probe_menu = nullptr;  // finalized menu of probe lengths; option 0 is the full probe
  const float* queries = nullptr;          // [n_query x dim], raw (the backend normalises cosine queries)
  uint32_t dim = 0;
  const int32_t* gt = nullptr;             // [n_query x gt_k] exact top-K for recall; may be null
  uint32_t gt_k = 0;
  std::vector<int64_t> arrivals_ns;        // scheduled arrival offsets from the run start, sorted
  std::vector<uint32_t> eval_idx;          // query row used by each arrival
  uint64_t warmup = 0;                     // the first `warmup` arrivals are excluded from the metrics
  double t_slo_s = 0;                      // T_SLO [s]
  double u_s_per_dc = 0;                   // profile: unit cost at `threads` (initial u, and B_max = C / u)
  double mean_service_s = 0;               // profile: initial E[S]
  double s_p95_s = 0;                      // profile: idle S_p95 (initial S for the bucket when t_base = tail)
  int threads = 1;                         // C workers
  bool pin_cores = false;                  // generator core 0, controller core 1, workers 2..C+1
};

struct EngineResult {
  std::vector<QueryRecord> records;        // one per arrival, in qid order
  std::vector<uint32_t> result_ids;        // [n x cfg.k], closest first
  std::vector<TimeSeriesRow> timeseries;
  double controller_update_us = 0;         // mean cost of one controller update
  double generator_lag_p99_us = 0;         // p99 of (actual dispatch time - scheduled time)
};

// Runs the workload end to end. Every arrival is served; nothing is dropped.
EngineResult run_engine(const EngineInputs& in);

}  // namespace laef
