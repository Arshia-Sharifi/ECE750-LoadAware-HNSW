// LA-ef: arrival schedules and evaluation-query selection. IMPLEMENTATION_PLAN.md §11.1.
#include "laef/workload.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace laef {
namespace {

constexpr double kNsPerS = 1e9;

int64_t to_ns(double seconds) { return static_cast<int64_t>(seconds * kNsPerS); }

// Poisson arrivals with rate qps in [start_ns, end_ns). A rate of 0 means silence.
void poisson_until(std::vector<int64_t>& out, std::mt19937_64& rng, double qps, int64_t start_ns, int64_t end_ns) {
  if (!(qps > 0)) return;
  std::exponential_distribution<double> gap(qps / kNsPerS);  // rate per ns
  int64_t t = start_ns;
  for (;;) {
    t += std::max<int64_t>(1, static_cast<int64_t>(gap(rng)));
    if (t >= end_ns) break;
    out.push_back(t);
  }
}

// Exactly n Poisson arrivals starting at start_ns.
void poisson_count(std::vector<int64_t>& out, std::mt19937_64& rng, double qps, int64_t start_ns, uint64_t n) {
  if (!(qps > 0)) throw std::invalid_argument("poisson workload needs qps > 0");
  std::exponential_distribution<double> gap(qps / kNsPerS);
  int64_t t = start_ns;
  for (uint64_t i = 0; i < n; ++i) {
    t += std::max<int64_t>(1, static_cast<int64_t>(gap(rng)));
    out.push_back(t);
  }
}

// Evenly spaced arrivals: gap = 1/qps, arrival i is due at start + i x gap (i = 1, 2, ...), until end_ns.
void constant_until(std::vector<int64_t>& out, double qps, int64_t start_ns, int64_t end_ns) {
  if (!(qps > 0)) return;
  const double gap_ns = kNsPerS / qps;
  for (uint64_t i = 1;; ++i) {
    const int64_t t = start_ns + static_cast<int64_t>(static_cast<double>(i) * gap_ns);
    if (t >= end_ns) break;
    out.push_back(t);
  }
}

// Exactly n evenly spaced arrivals starting at start_ns.
void constant_count(std::vector<int64_t>& out, double qps, int64_t start_ns, uint64_t n) {
  if (!(qps > 0)) throw std::invalid_argument("constant workload needs qps > 0");
  const double gap_ns = kNsPerS / qps;
  for (uint64_t i = 1; i <= n; ++i)
    out.push_back(start_ns + static_cast<int64_t>(static_cast<double>(i) * gap_ns));
}

// Arrival times from a CSV whose first field is seconds. Header and comment lines are skipped.
std::vector<int64_t> read_trace(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open trace_csv: " + path);
  std::vector<int64_t> out;
  std::string line;
  while (std::getline(in, line)) {
    const std::string field = line.substr(0, line.find(','));
    try {
      out.push_back(to_ns(std::stod(field)));
    } catch (const std::exception&) {
      // not a number (header or comment): skip the line
    }
  }
  std::sort(out.begin(), out.end());
  if (!out.empty()) {
    const int64_t first = out.front();
    for (auto& t : out) t -= first;
  }
  return out;
}

}  // namespace

void check_rates(const WorkloadCfg& w) {
  if (w.profile == "poisson" && !(w.qps > 0)) throw std::invalid_argument("poisson workload needs qps > 0");
  if (w.profile == "constant" && !(w.qps > 0)) throw std::invalid_argument("constant workload needs qps > 0");
  if (w.profile == "step") {
    if (w.phases.empty()) throw std::invalid_argument("step workload needs at least one phase");
    for (const auto& ph : w.phases)
      if (!(ph.qps >= 0)) throw std::invalid_argument("step phase qps must be >= 0");
  }
  if (w.profile == "mmpp2" && !(w.qps_lo > 0 || w.qps_hi > 0)) throw std::invalid_argument("mmpp2 needs a positive rate");
}

std::vector<int64_t> make_arrivals(const WorkloadCfg& w, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<int64_t> out;

  if (w.profile == "poisson") {
    if (w.duration_s > 0) poisson_until(out, rng, w.qps, 0, to_ns(w.duration_s));
    else poisson_count(out, rng, w.qps, 0, w.n_queries);

  } else if (w.profile == "constant") {
    if (w.duration_s > 0) constant_until(out, w.qps, 0, to_ns(w.duration_s));
    else constant_count(out, w.qps, 0, w.n_queries);

  } else if (w.profile == "step") {
    if (w.phases.empty()) throw std::invalid_argument("step workload needs at least one phase");
    int64_t start = 0;
    for (const auto& ph : w.phases) {
      if (!(ph.duration_s > 0)) throw std::invalid_argument("every step phase needs duration_s > 0");
      const int64_t end = start + to_ns(ph.duration_s);
      poisson_until(out, rng, ph.qps, start, end);  // phases begin at their boundary, not at the last arrival
      start = end;
    }

  } else if (w.profile == "mmpp2") {
    if (!(w.mean_lo_s > 0 && w.mean_hi_s > 0)) throw std::invalid_argument("mmpp2 needs mean_lo_s and mean_hi_s > 0");
    if (!(w.qps_lo > 0 || w.qps_hi > 0)) throw std::invalid_argument("mmpp2 needs a positive rate");
    const bool timed = w.duration_s > 0;
    if (!timed && w.n_queries == 0) throw std::invalid_argument("mmpp2 needs duration_s or n_queries");
    const int64_t end_ns = timed ? to_ns(w.duration_s) : std::numeric_limits<int64_t>::max();
    std::exponential_distribution<double> hold_lo(1.0 / w.mean_lo_s), hold_hi(1.0 / w.mean_hi_s);
    int64_t t = 0;
    bool high = false;
    while (t < end_ns && (timed || out.size() < w.n_queries)) {
      const double hold = high ? hold_hi(rng) : hold_lo(rng);
      const int64_t state_end = std::min(end_ns, t + to_ns(hold));
      poisson_until(out, rng, high ? w.qps_hi : w.qps_lo, t, state_end);
      t = state_end;
      high = !high;
    }
    if (!timed && out.size() > w.n_queries) out.resize(w.n_queries);

  } else if (w.profile == "trace") {
    out = read_trace(w.trace_csv);

  } else {
    throw std::invalid_argument("unknown workload profile: " + w.profile);
  }

  if (out.empty()) throw std::invalid_argument("workload '" + w.profile + "' produced no arrivals");
  return out;
}

std::vector<uint32_t> pick_queries(size_t n_queries, const std::vector<uint32_t>& pool, uint64_t seed) {
  if (pool.empty()) throw std::invalid_argument("evaluation query pool is empty");
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
  std::vector<uint32_t> out(n_queries);
  for (auto& o : out) o = pool[pick(rng)];
  return out;
}

double burst_end_s(const WorkloadCfg& w) {
  if (w.profile != "step" || w.phases.empty()) return -1.0;
  double t = 0, peak = -1.0, burst_end = -1.0;
  for (const auto& ph : w.phases) {
    t += ph.duration_s;
    const double rate = ph.qps;
    if (rate >= peak) {  // the last phase at the highest rate ends the burst
      peak = rate;
      burst_end = t;
    }
  }
  return burst_end;
}

}  // namespace laef
