// LA-ef: deterministic fake backend (IMPLEMENTATION_PLAN.md §8.1). See fake_backend.h.
#include "laef/fake_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace laef {
namespace {

constexpr uint32_t kFullProbeLen = 1025;  // the authors' full probe length for M = 16
constexpr uint32_t kFullProbeNdis = 100;  // the fake's probe cost at the full probe length

// Probe cost for probe length L, scaled linearly from the full probe (v2: probe-length menu).
uint32_t probe_dc(uint32_t probe_len) {
  return std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(double(kFullProbeNdis) * probe_len / kFullProbeLen)));
}

void spin_for_ns(double ns) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::nanoseconds(static_cast<int64_t>(ns));
  while (std::chrono::steady_clock::now() < end) {
  }
}

class FakeBackend final : public AnnBackend {
 public:
  FakeBackend(uint32_t dim, double u_ns) : dim_(dim), u_ns_(u_ns) {}

  uint32_t dim() const override { return dim_; }

  void search_planned(const float* q, uint32_t k, uint32_t probe_len, const EfPlannerFn& planner,
                      uint32_t* out_ids, SearchStats* st) const override {
    const float score = score_of(q);
    const uint32_t probe = probe_dc(probe_len);
    const uint32_t ef_pref = k * (1 + static_cast<uint32_t>(score) % 8);
    const uint32_t ef_used = std::max<uint32_t>(k, planner(score, ef_pref, probe));
    const uint32_t ndis = probe + ef_used;
    spin_for_ns(ndis * u_ns_);
    fill_results(out_ids, k);
    SearchStats s;
    s.ndis = ndis;
    s.probe_ndis = probe;
    s.probe_len = probe_len;
    s.score = score;
    s.ef_pref = ef_pref;
    s.ef_used = ef_used;
    s.probe_complete = true;
    if (st) *st = s;
  }

  void search_fixed(const float* q, uint32_t k, uint32_t ef, uint32_t* out_ids, SearchStats* st) const override {
    const uint32_t ef_used = std::max<uint32_t>(k, ef);
    const uint32_t ndis = kFullProbeNdis + ef_used;
    spin_for_ns(ndis * u_ns_);
    fill_results(out_ids, k);
    SearchStats s;
    s.ndis = ndis;
    s.ef_pref = ef_used;
    s.ef_used = ef_used;
    (void)q;
    if (st) *st = s;  // probe_complete stays false and probe_len stays 0: no probe in plain HNSW
  }

 private:
  // FNV-1a over the query bits, mapped to [0, 100).
  float score_of(const float* q) const {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < dim_; ++i) {
      uint32_t bits = 0;
      std::memcpy(&bits, q + i, sizeof bits);
      h = (h ^ bits) * 1099511628211ull;
    }
    return static_cast<float>(h % 10000) / 100.0f;
  }

  static void fill_results(uint32_t* out_ids, uint32_t k) {
    for (uint32_t i = 0; i < k; ++i) out_ids[i] = i;
  }

  uint32_t dim_;
  double u_ns_;
};

}  // namespace

std::unique_ptr<AnnBackend> make_fake_backend(uint32_t dim, double u_ns) {
  return std::make_unique<FakeBackend>(dim, u_ns);
}

}  // namespace laef
