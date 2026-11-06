#pragma once
// LA-ef: immutable plan published by the controller thread each window (IMPLEMENTATION_PLAN.md §10.3, §1A.5-§1A.6).
// SnapshotStore uses a mutex-guarded shared_ptr (portable to GCC 11; std::atomic<shared_ptr> needs GCC 12+).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "laef/controllers/budget_controller.h"
#include "laef/price_list.h"
#include "laef/probe_menu.h"

namespace laef {

struct PlanSnapshot {
  double B = 0, B_max = 0, B_min = 0, D = 0;
  double T_slo = 0, S_floor = 0, mean_service = 0;
  double w_target = 0;          // B / lambda-hat          [dc per arrival]
  double water_level = 0;       // lambda* of the chosen probe  [recall per dc]
  double uniform_fraction = 1;  // min(1, w_target / W_nom) of the chosen probe
  double reservation = 0;       // expected work of the chosen probe at w_target, reserved at arrival [dc]
  bool floor_mode = false;
  uint64_t version = 0;
  // Probe menu (v2). The probe the next queries run with (stage 1), and the cut parameters of every option (stage 2).
  size_t probe_idx = 0;
  std::vector<double> water_level_by_L;       // one entry per menu option
  std::vector<double> uniform_fraction_by_L;  // one entry per menu option
};

enum class TBase { Mean, Tail };

// Builds the snapshot from one controller update (Main Flow §5, §5.1; plan §1A.5). Floor mode runs the shortest probe.
// Policies without a cut (kind Passthrough) always run the full probe.
inline PlanSnapshot make_snapshot(const ControllerOutput& o, const ProbeMenu& menu, CutKind kind, double lambda_hat,
                                  double mean_service, double s_p95, double t_slo, TBase tb, uint64_t version) {
  if (menu.size() == 0) throw std::invalid_argument("make_snapshot: empty probe menu");
  PlanSnapshot s;
  s.B = o.B; s.B_max = o.B_max; s.B_min = o.B_min; s.S_floor = o.S_floor;
  s.T_slo = t_slo; s.mean_service = mean_service; s.floor_mode = o.floor_mode; s.version = version;
  const double t_base = std::max(0.0, t_slo - (tb == TBase::Mean ? mean_service : s_p95));
  s.D = o.B * t_base;
  s.w_target = o.B / std::max(lambda_hat, 1e-9);

  const size_t n = menu.size();
  s.water_level_by_L.resize(n);
  s.uniform_fraction_by_L.resize(n);
  for (size_t j = 0; j < n; ++j) {
    const PriceList& pl = menu.at(j).price_list;
    s.water_level_by_L[j] = pl.water_level(s.w_target);
    const double wn = pl.model_nominal_work();
    s.uniform_fraction_by_L[j] = wn > 0 ? std::min(1.0, s.w_target / wn) : 1.0;
  }

  if (kind == CutKind::Passthrough) s.probe_idx = 0;
  else if (o.floor_mode) s.probe_idx = menu.smallest_index();
  else s.probe_idx = menu.choose(s.w_target, kind);
  s.water_level = s.water_level_by_L[s.probe_idx];
  s.uniform_fraction = s.uniform_fraction_by_L[s.probe_idx];
  // In floor mode every query runs the shortest probe at its floor rung, so reserve that work.
  s.reservation = (kind != CutKind::Passthrough && o.floor_mode)
                      ? menu.smallest().price_list.model_floor_work()
                      : menu.at(s.probe_idx).price_list.expected_at(s.w_target, kind).work;
  return s;
}

class SnapshotStore {
 public:
  explicit SnapshotStore(PlanSnapshot s0 = {}) : p_(std::make_shared<const PlanSnapshot>(s0)) {}
  std::shared_ptr<const PlanSnapshot> load() const { std::lock_guard<std::mutex> g(m_); return p_; }
  void store(const PlanSnapshot& s) {
    auto n = std::make_shared<const PlanSnapshot>(s);
    std::lock_guard<std::mutex> g(m_);
    p_ = std::move(n);
  }

 private:
  mutable std::mutex m_;
  std::shared_ptr<const PlanSnapshot> p_;
};

}  // namespace laef
