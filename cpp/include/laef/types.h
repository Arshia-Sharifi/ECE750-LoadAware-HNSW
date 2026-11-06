#pragma once
// LA-ef: per-query record, clock helpers and recall@k (IMPLEMENTATION_PLAN.md §8.1, §11.2, §11.3).
#include <chrono>
#include <cstdint>

namespace laef {

using Clock = std::chrono::steady_clock;

inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

enum class QueryStatus : uint8_t { Ok, ProbeIncomplete };

inline const char* to_string(QueryStatus s) { return s == QueryStatus::Ok ? "ok" : "probe_incomplete"; }

// One served query. Every scheduled query produces exactly one record: there is no drop path.
// Times are nanoseconds relative to the run start; t_arrival_ns is the SCHEDULED arrival time
// (avoids coordinated omission).
struct QueryRecord {
  uint32_t qid = 0;
  uint32_t eval_idx = 0;       // row of the test-query matrix
  bool warmup = false;         // served but excluded from metrics
  int64_t t_arrival_ns = 0;
  int64_t t_start_ns = 0;
  int64_t t_end_ns = 0;
  QueryStatus status = QueryStatus::Ok;
  float score = -1.f;          // Ada-ef score (-1 for static-ef or incomplete probe)
  int group = -1;
  uint32_t ef_pref = 0, ef_used = 0;
  uint32_t ndis = 0, probe_ndis = 0;
  uint32_t probe_len = 0;      // L: probe length the query ran with (0 = no probe, static ef)
  bool forced_floor = false;   // bucket could not pay even the floor work
  bool floor_mode = false;     // planned while floor mode was on
  double planned_work = 0;
  float recall = -1.f;         // filled after the run (-1 = unknown)

  double wait_s() const { return (t_start_ns - t_arrival_ns) * 1e-9; }
  double service_s() const { return (t_end_ns - t_start_ns) * 1e-9; }
  double latency_s() const { return (t_end_ns - t_arrival_ns) * 1e-9; }
};

// |found ∩ gt[0..k)| / k. Missing results (UINT32_MAX) never match.
inline float recall_at_k(const uint32_t* found, const int32_t* gt, uint32_t k) {
  uint32_t hits = 0;
  for (uint32_t i = 0; i < k; ++i)
    for (uint32_t j = 0; j < k; ++j)
      if (gt[j] >= 0 && found[i] == static_cast<uint32_t>(gt[j])) { ++hits; break; }
  return k ? static_cast<float>(hits) / static_cast<float>(k) : 0.f;
}

}  // namespace laef
