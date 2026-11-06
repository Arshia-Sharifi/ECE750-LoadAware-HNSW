// LA-ef: output files and summary metrics (IMPLEMENTATION_PLAN.md §11.3, §11.4, §14.2).
#include "laef/results_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace laef {
namespace {

// numpy's default percentile: linear interpolation between order statistics. `sorted` must not be empty.
double percentile_linear(const std::vector<double>& sorted, double q) {
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(sorted.size() - 1, lo + 1);
  return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

}  // namespace

nlohmann::json compute_summary(const std::vector<QueryRecord>& queries, const std::vector<TimeSeriesRow>& ts,
                               const SummaryParams& p) {
  std::vector<const QueryRecord*> kept;
  for (const auto& r : queries)
    if (!r.warmup) kept.push_back(&r);

  std::vector<double> latency_ms, recalls;
  double sum_wait_ms = 0, sum_service_ms = 0, sum_ef_ratio = 0, sum_ndis = 0;
  uint64_t n_ok = 0, n_within = 0, n_forced = 0, n_floor = 0, n_ef = 0;
  double sum_probe_len = 0;
  uint64_t n_probe = 0, n_full_probe = 0;
  int64_t first_arrival = std::numeric_limits<int64_t>::max(), last_end = std::numeric_limits<int64_t>::min();

  for (const QueryRecord* r : kept) {
    const double T = r->latency_s() * 1e3;
    latency_ms.push_back(T);
    sum_wait_ms += r->wait_s() * 1e3;
    sum_service_ms += r->service_s() * 1e3;
    if (r->status == QueryStatus::Ok) ++n_ok;
    if (T <= p.t_slo_ms) ++n_within;
    if (r->recall >= 0) recalls.push_back(r->recall);
    if (r->ef_pref > 0) {
      sum_ef_ratio += static_cast<double>(r->ef_used) / r->ef_pref;
      ++n_ef;
    }
    if (r->probe_len > 0) {  // static ef runs no probe, so it is left out of the probe statistics
      sum_probe_len += r->probe_len;
      ++n_probe;
      if (r->probe_len == p.full_probe_len) ++n_full_probe;
    }
    sum_ndis += r->ndis;
    if (r->forced_floor) ++n_forced;
    if (r->floor_mode) ++n_floor;
    first_arrival = std::min(first_arrival, r->t_arrival_ns);
    last_end = std::max(last_end, r->t_end_ns);
  }

  nlohmann::json s;
  const size_t n = kept.size();
  s["n_queries"] = n;
  s["n_ok"] = n_ok;
  s["n_probe_incomplete"] = n - n_ok;
  s["slo_ms"] = p.t_slo_ms;
  s["offered_qps"] = p.offered_qps;
  s["n_windows"] = ts.size();

  // Wall time: first arrival to last completion of the measured queries.
  const double wall_s = (n > 0 && last_end > first_arrival) ? (last_end - first_arrival) * 1e-9 : 0.0;
  s["wall_s"] = wall_s;
  s["achieved_qps"] = wall_s > 0 ? n / wall_s : 0.0;
  s["goodput_qps"] = wall_s > 0 ? n_within / wall_s : 0.0;
  s["controller_update_us"] = p.controller_update_us;
  s["generator_lag_p99_us"] = p.generator_lag_p99_us;

  if (n > 0) {
    std::sort(latency_ms.begin(), latency_ms.end());
    const double p95 = percentile_linear(latency_ms, 0.95);
    s["p50_ms"] = percentile_linear(latency_ms, 0.50);
    s["p95_ms"] = p95;
    s["p99_ms"] = percentile_linear(latency_ms, 0.99);
    s["mean_wait_ms"] = sum_wait_ms / n;
    s["mean_service_ms"] = sum_service_ms / n;
    s["slo_attainment"] = static_cast<double>(n_within) / n;
    s["latency_excess_ms"] = std::max(0.0, p95 - p.t_slo_ms);
    s["mean_ef_ratio"] = n_ef ? sum_ef_ratio / n_ef : 0.0;
    s["mean_ndis"] = sum_ndis / n;
    s["forced_floor_rate"] = static_cast<double>(n_forced) / n;
    s["floor_mode_query_rate"] = static_cast<double>(n_floor) / n;
    s["mean_probe_len"] = n_probe ? sum_probe_len / n_probe : 0.0;
    s["frac_full_probe"] = n_probe ? static_cast<double>(n_full_probe) / n_probe : 0.0;
  }

  if (!recalls.empty()) {
    std::sort(recalls.begin(), recalls.end());
    double sum = 0;
    uint64_t at_target = 0;
    for (double r : recalls) {
      sum += r;
      if (r >= p.target_recall - 1e-6) ++at_target;
    }
    s["mean_recall"] = sum / recalls.size();
    s["p5_recall"] = percentile_linear(recalls, 0.05);
    s["p1_recall"] = percentile_linear(recalls, 0.01);
    s["frac_target_recall"] = static_cast<double>(at_target) / recalls.size();
  }

  // Control: window statistics and recovery after a burst.
  uint64_t violation = 0, headroom = 0, comfort = 0, guard = 0, floor_windows = 0, p95_bad = 0;
  double recovery = -1;
  for (const auto& w : ts) {
    if (std::string(w.zone) == "violation") ++violation;
    else if (std::string(w.zone) == "headroom") ++headroom;
    else if (std::string(w.zone) == "comfort") ++comfort;
    if (w.guard_fired) ++guard;
    if (w.floor_mode) ++floor_windows;
    if (w.p95_ms > p.t_slo_ms) ++p95_bad;
    if (recovery < 0 && p.burst_end_s >= 0 && w.t_s >= p.burst_end_s && w.B_max > 0 && w.B >= 0.95 * w.B_max)
      recovery = w.t_s - p.burst_end_s;
  }
  const double nw = static_cast<double>(ts.size());
  s["frac_windows_violation"] = ts.empty() ? 0.0 : violation / nw;
  s["frac_windows_headroom"] = ts.empty() ? 0.0 : headroom / nw;
  s["frac_windows_comfort"] = ts.empty() ? 0.0 : comfort / nw;
  s["guard_fires"] = guard;
  s["floor_mode_windows"] = floor_windows;
  s["p95_window_violations"] = p95_bad;
  s["recovery_time_s"] = (p.burst_end_s >= 0 && recovery >= 0) ? nlohmann::json(recovery) : nlohmann::json(nullptr);
  return s;
}

void write_queries_csv(const std::string& path, const std::vector<QueryRecord>& queries) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "qid,eval_idx,warmup,t_arrival_us,wait_us,service_us,latency_us,status,score,group,ef_pref,ef_used,"
       "forced_floor,floor_mode,ndis,probe_ndis,planned_work,recall,probe_len\n";
  for (const auto& r : queries) {
    f << r.qid << ',' << r.eval_idx << ',' << (r.warmup ? 1 : 0) << ',' << r.t_arrival_ns / 1000 << ','
      << static_cast<int64_t>(r.wait_s() * 1e6) << ',' << static_cast<int64_t>(r.service_s() * 1e6) << ','
      << static_cast<int64_t>(r.latency_s() * 1e6) << ',' << to_string(r.status) << ','
      << std::fixed << std::setprecision(4) << r.score << ',' << r.group << ',' << r.ef_pref << ',' << r.ef_used << ','
      << (r.forced_floor ? 1 : 0) << ',' << (r.floor_mode ? 1 : 0) << ',' << r.ndis << ',' << r.probe_ndis << ','
      << std::setprecision(1) << r.planned_work << ',' << std::setprecision(4) << r.recall << ',' << r.probe_len << '\n';
  }
}

void write_timeseries_csv(const std::string& path, const std::vector<TimeSeriesRow>& ts) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "t_s,B,B_max,B_min,D,p95_ms,s_p95_ms,Q,W_hat_ms,u_ns,ES_ms,lambda_hat,w_target,water_level,"
       "uniform_fraction,zone,guard_fired,floor_mode,slo_violated,probe_len\n";
  for (const auto& r : ts) {
    f << std::fixed << std::setprecision(3) << r.t_s << ',' << std::setprecision(1) << r.B << ',' << r.B_max << ','
      << r.B_min << ',' << r.D << ',' << std::setprecision(3) << r.p95_ms << ',' << r.s_p95_ms << ','
      << std::setprecision(1) << r.Q << ',' << std::setprecision(3) << r.W_hat_ms << ',' << std::setprecision(1)
      << r.u_ns << ',' << std::setprecision(3) << r.ES_ms << ',' << std::setprecision(2) << r.lambda_hat << ','
      << std::setprecision(1) << r.w_target << ',' << std::scientific << std::setprecision(6) << r.water_level << ','
      << std::fixed << std::setprecision(4) << r.uniform_fraction << ',' << r.zone << ',' << (r.guard_fired ? 1 : 0)
      << ',' << (r.floor_mode ? 1 : 0) << ',' << (r.slo_violated ? 1 : 0) << ',' << r.probe_len << '\n';
  }
}

}  // namespace laef
