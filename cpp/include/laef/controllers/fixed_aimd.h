#pragma once
// LA-ef: naive fixed-step AIMD baseline (IMPLEMENTATION_PLAN.md §10.4, BB §5).
// Same clamps and floor-mode rule as P-AIMD; cut size ignores severity. Guard optional (default off).
#include <algorithm>
#include <string>

#include "laef/controllers/budget_controller.h"

namespace laef {

struct FixedAimdParams {
  double fixed_cut = 0.90;
  double headroom = 0.85;
  double growth = 1.02;
  bool guard = false;
  double cut_floor = 0.50;  // only used by the optional guard
};

class FixedAimdController final : public BudgetController {
 public:
  explicit FixedAimdController(FixedAimdParams p = {}, double B0 = 0) : p_(p), B_(B0) {}

  ControllerOutput update(const ControllerInputs& in) override {
    const PhysicalLimits l = physical_limits(in);
    ControllerOutput o;
    if (!(B_ > 0)) B_ = l.B_max;
    if (in.p95 > in.t_slo) { B_ *= p_.fixed_cut; o.zone = "violation"; }
    else if (in.p95 < p_.headroom * in.t_slo) { B_ = std::min(l.B_max, B_ * p_.growth); o.zone = "headroom"; }
    else { o.zone = "comfort"; }

    if (l.room < l.S_floor) { o.floor_mode = true; B_ = std::min(l.B_max, l.B_min); }
    else if (p_.guard && in.s_p95 > l.room) { B_ *= std::max(p_.cut_floor, l.room / in.s_p95); o.guard_fired = true; }

    if (l.B_min > l.B_max) o.floor_mode = true;
    B_ = clamp_budget(B_, l);
    fill_common(o, l, B_);
    return o;
  }

  void reset(double B0) override { B_ = B0; }
  double budget() const override { return B_; }
  std::string name() const override { return "fixed_aimd"; }

 private:
  FixedAimdParams p_;
  double B_;
};

}  // namespace laef
