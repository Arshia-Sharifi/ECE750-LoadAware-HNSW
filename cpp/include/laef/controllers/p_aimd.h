#pragma once
// LA-ef: Proportional-AIMD thermostat. Line-for-line port of Budget B calculations.md §1
// (IMPLEMENTATION_PLAN.md §10.4). Admission control is replaced by floor mode: nothing is dropped.
#include <algorithm>
#include <string>

#include "laef/controllers/budget_controller.h"

namespace laef {

struct PAimdParams {
  double cut_floor = 0.50;  // never cut more than half at once
  double headroom = 0.95;   // p95 < headroom*T_SLO -> grow
  double growth = 1.02;     // +2% per window
  bool guard = true;        // tail-aware queue guard
};

class PAimdController final : public BudgetController {
 public:
  explicit PAimdController(PAimdParams p = {}, double B0 = 0) : p_(p), B_(B0) {}

  ControllerOutput update(const ControllerInputs& in) override {
    const PhysicalLimits l = physical_limits(in);
    ControllerOutput o;
    if (!(B_ > 0)) B_ = l.B_max;  // first window: start at the ceiling

    // 2. proportional latency feedback
    if (in.p95 > in.t_slo) {
      B_ *= std::max(p_.cut_floor, in.t_slo / in.p95);
      o.zone = "violation";
    } else if (in.p95 < p_.headroom * in.t_slo) {
      B_ = std::min(l.B_max, B_ * p_.growth);
      o.zone = "headroom";
    } else {
      o.zone = "comfort";
    }

    // 3. tail-aware queue guard (instant; p95 lags)
    if (l.room < l.S_floor) {
      o.floor_mode = true;                 // infeasible: run everything at ef = k, SLO may slip
      B_ = std::min(l.B_max, l.B_min);
    } else if (p_.guard && in.s_p95 > l.room) {
      B_ *= std::max(p_.cut_floor, l.room / in.s_p95);
      o.guard_fired = true;
    }

    // 4. final clamp; B_min > B_max means even all-floor exceeds hardware -> floor mode
    if (l.B_min > l.B_max) o.floor_mode = true;
    B_ = clamp_budget(B_, l);
    fill_common(o, l, B_);
    return o;
  }

  void reset(double B0) override { B_ = B0; }
  double budget() const override { return B_; }
  std::string name() const override { return "p_aimd"; }

 private:
  PAimdParams p_;
  double B_;
};

}  // namespace laef
