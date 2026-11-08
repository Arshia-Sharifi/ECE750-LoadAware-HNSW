#pragma once
// LA-ef v2: the probe-length menu (IMPLEMENTATION_PLAN.md §1A.2, §1A.4, §1A.5).
// One Ada-ef table and one price list per probe length L. Stage 1 (choose) picks the L a query runs with from the
// expected work and recall of every option at the current water budget. JSON I/O lives in price_list_io.h.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laef/price_list.h"

namespace laef {

struct ProbeOption {
  uint32_t probe_len = 0;                 // L: distance computations collected before the planner runs
  PriceList price_list;                   // curves at this L; every work value includes the L-probe
  std::string adapter_path;               // tables/L{L}/ada_ef_adapter.bin
  std::string price_list_path;            // tables/L{L}/price_list.json
  double calib_mean_recall_at_pref = 0;   // recall at Ada-ef's own ef, at this L (calibration)
  double mean_probe_ndis = 0;             // mean probe distance computations, calibration preference pass
  double ef_pref_mean = 0;                // mean Ada-ef ef, calibration preference pass
};

// Options in strictly descending probe length. Option 0 is the full probe.
class ProbeMenu {
 public:
  ProbeMenu() = default;  // empty until assigned; every user must check size() first
  explicit ProbeMenu(std::vector<ProbeOption> options) : opts_(std::move(options)) {
    if (opts_.empty()) throw std::invalid_argument("ProbeMenu: no probe lengths");
    for (size_t i = 0; i < opts_.size(); ++i) {
      if (opts_[i].probe_len == 0) throw std::invalid_argument("ProbeMenu: probe length must be positive");
      if (i > 0 && opts_[i].probe_len >= opts_[i - 1].probe_len)
        throw std::invalid_argument("ProbeMenu: probe lengths must be strictly descending");
    }
  }

  size_t size() const { return opts_.size(); }
  const std::vector<ProbeOption>& options() const { return opts_; }
  const ProbeOption& at(size_t i) const { return opts_.at(i); }
  const ProbeOption& full() const { return opts_.at(0); }
  size_t smallest_index() const { return opts_.empty() ? 0 : opts_.size() - 1; }
  const ProbeOption& smallest() const { return opts_.at(smallest_index()); }

  // probe.enabled = false (the v1 ef-only ablation): keep only the full probe.
  void keep_full_only() {
    if (!opts_.empty()) opts_.resize(1);
  }

  // Stage 1 (§1A.5). Index of the probe a query runs with, given the water budget w_target [dc per arrival].
  //  - among the options whose expected work fits (within 0.1%), the one with the highest expected recall.
  //    Options are scanned from the longest probe, so a tie (difference below 1e-4) keeps the longer one.
  //  - nothing fits: the shortest probe. The bucket check and floor mode handle the rest.
  size_t choose(double w_target, CutKind kind) const {
    if (opts_.empty()) throw std::logic_error("ProbeMenu::choose on an empty menu");
    size_t best = opts_.size();
    double best_recall = 0;
    for (size_t j = 0; j < opts_.size(); ++j) {
      const ExpectedPoint p = opts_[j].price_list.expected_at(w_target, kind);
      if (p.work > w_target * 1.001) continue;
      if (best == opts_.size() || p.recall > best_recall + 1e-4) {
        best = j;
        best_recall = p.recall;
      }
    }
    return best == opts_.size() ? smallest_index() : best;
  }

 private:
  std::vector<ProbeOption> opts_;
};

}  // namespace laef
