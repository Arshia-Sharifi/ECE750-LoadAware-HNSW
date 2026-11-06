#pragma once
// LA-ef: open-loop budget B = rho_target * B_max, no feedback (IMPLEMENTATION_PLAN.md §10.4, BB §5).
// Template for adding model-based budget methods. Floor mode uses the same infeasibility rule.
#include <string>

#include "laef/controllers/budget_controller.h"

namespace laef {

class StaticCapacityController final : public BudgetController {
 public:
  explicit StaticCapacityController(double rho_target = 0.9) : rho_(rho_target) {}

  ControllerOutput update(const ControllerInputs& in) override {
    const PhysicalLimits l = physical_limits(in);
    ControllerOutput o;
    o.zone = "static";
    o.floor_mode = (l.room < l.S_floor) || (l.B_min > l.B_max);
    B_ = clamp_budget(rho_ * l.B_max, l);
    fill_common(o, l, B_);
    return o;
  }

  void reset(double B0) override { B_ = B0; }
  double budget() const override { return B_; }
  std::string name() const override { return "static_capacity"; }

 private:
  double rho_;
  double B_ = 0;
};

}  // namespace laef
