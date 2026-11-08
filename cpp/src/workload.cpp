// LA-ef: arrival schedules and evaluation-query selection. IMPLEMENTATION_PLAN.md §11.1.
#include "laef/workload.h"

#include <algorithm>
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

}  // namespace

void check_rates(const WorkloadCfg& w) {
  if (w.profile == "poisson") {
    if (!(w.qps > 0)) throw std::invalid_argument("poisson workload needs qps > 0");
  } else if (w.profile == "constant") {
    if (!(w.qps > 0)) throw std::invalid_argument("constant workload needs qps > 0");
  } else {
    throw std::invalid_argument("unknown workload profile: " + w.profile);
  }
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

}  // namespace laef
