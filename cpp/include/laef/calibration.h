#pragma once
// LA-ef: Step A driver (IMPLEMENTATION_PLAN.md §9). Runs the preference pass and the ladder sweep,
// then builds the price list. Functions here do no file I/O except the explicit read/write helpers.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "laef/price_list.h"

namespace laef {

class AnnBackend;

// calibration/samplings.bin (§4.1): int32 rows, int32 cols, float32[rows*cols] queries (row-major);
// then int32 rows, int32 cols, int32[rows*cols] ground truth (row-major, ascending distance).
struct Samplings {
  uint32_t n = 0, dim = 0;
  uint32_t gt_k = 0;
  std::vector<float> queries;  // [n x dim] raw (the backend normalises cosine queries itself)
  std::vector<int32_t> gt;     // [n x gt_k]
};
Samplings read_samplings(const std::string& path);

// One measurement of the sweep. is_pref marks Ada-ef's own operating point (preference pass).
struct SweepRow {
  uint32_t query = 0;
  float score = -1.f;
  uint32_t ef_pref = 0;
  uint32_t ef = 0;
  float recall = 0.f;
  uint32_t ndis = 0;
  uint32_t probe_ndis = 0;
  double service_us = 0;
  bool is_pref = false;
};

struct CalibrationSettings {
  uint32_t k = 10;
  float target_recall = 0.95f;
  double ladder_ratio = 2.0;
  uint32_t ef_upper_bound = 5000;
  size_t min_group_size = 30;
  size_t max_groups = 8;
  uint32_t threads = 1;
};

// Ladder {k, ceil(k*ratio), ...} that stops at the first rung >= max_pref, capped at ef_upper_bound.
std::vector<uint32_t> make_ladder(uint32_t k, double ratio, uint32_t max_pref, uint32_t ef_upper_bound);

struct SweepResult {
  std::vector<uint32_t> ladder;  // global ladder, ascending
  std::vector<SweepRow> rows;    // preference rows first, then one row per (query, rung)
};

// Preference pass: Ada-ef's own ef for every query. Ladder sweep: every query at every rung. Both run with the probe
// length `probe_len` through the same probe and planner hook that live serving uses. Threads only change the timing.
SweepResult run_sweep(const AnnBackend& backend, const Samplings& samplings, const CalibrationSettings& cfg,
                      uint32_t probe_len);

struct CalibrationResult {
  PriceList price_list;
  size_t n_queries = 0;
  double calib_mean_recall_at_pref = 0;  // recall at Ada-ef's ef, averaged over the calibration queries
  double nominal_work_mean = 0;          // measured mean ndis at ef_pref
};

// Groups queries by integer score key (§9.2), builds per-group curves on the global ladder with their
// concave envelopes (§9.3), sets static_ef_for_target, and finalizes the price list.
CalibrationResult build_price_list(const std::vector<SweepRow>& rows, size_t n_queries,
                                   const std::vector<uint32_t>& ladder, const CalibrationSettings& cfg);

// Keeps the rows listed in `rows`, in that order.
Samplings select_rows(const Samplings& s, const std::vector<uint32_t>& rows);
// Writes calibration/samplings.bin in the §4.1 layout (the inverse of read_samplings).
void write_samplings(const std::string& path, const Samplings& s);

void write_sweep_csv(const std::string& path, const std::vector<SweepRow>& rows);

}  // namespace laef
