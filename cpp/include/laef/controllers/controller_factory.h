#pragma once
// LA-ef: budget-controller factory (IMPLEMENTATION_PLAN.md §10.4).
// To add a budget method: write controllers/<name>.h implementing BudgetController and add one branch here.
#include <memory>
#include <stdexcept>

#include "laef/config.h"
#include "laef/controllers/fixed_aimd.h"
#include "laef/controllers/p_aimd.h"
#include "laef/controllers/static_capacity.h"

namespace laef {

// Returns nullptr for type "none" (Ada-ef / static-ef baselines: the plan never changes).
inline std::unique_ptr<BudgetController> make_controller(const ControllerCfg& c) {
  if (c.type == "none") return nullptr;
  if (c.type == "p_aimd") return std::make_unique<PAimdController>(PAimdParams{c.cut_floor, c.headroom, c.growth, c.guard});
  if (c.type == "fixed_aimd")
    return std::make_unique<FixedAimdController>(FixedAimdParams{c.fixed_cut, c.headroom, c.growth, c.fixed_guard, c.cut_floor});
  if (c.type == "static_capacity") return std::make_unique<StaticCapacityController>(c.rho_target);
  throw std::invalid_argument("unknown controller.type: " + c.type);
}

}  // namespace laef
