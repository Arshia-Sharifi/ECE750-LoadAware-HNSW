#pragma once
// LA-ef: output files and summary metrics (IMPLEMENTATION_PLAN.md §11.3, §11.4, §14.2).
// config.json and summary.json are written with write_json_file() from config.h.
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "laef/engine.h"
#include "laef/types.h"

namespace laef {

struct SummaryParams {
  double t_slo_ms = 0;
  double offered_qps = 0;
  double target_recall = 0.95;
  double burst_end_s = -1;           // step profile: recovery is measured from here (-1 = not a step profile)
  double controller_update_us = 0;
  double generator_lag_p99_us = 0;
  uint32_t full_probe_len = 0;       // full probe L: frac_full_probe counts queries that ran with it
};

// Metrics over the non-warm-up queries (§11.4 keys). Percentiles are linear interpolation, like numpy.
nlohmann::json compute_summary(const std::vector<QueryRecord>& queries, const std::vector<TimeSeriesRow>& ts,
                               const SummaryParams& p);

void write_queries_csv(const std::string& path, const std::vector<QueryRecord>& queries);
void write_timeseries_csv(const std::string& path, const std::vector<TimeSeriesRow>& ts);

}  // namespace laef
