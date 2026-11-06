#pragma once
// LA-ef: floor mode, which replaces admission control (IMPLEMENTATION_PLAN.md §10.7).
//
// NO QUERY IS EVER DROPPED. When even ef = k cannot meet the SLO, every query runs at its group's
// floor rung and the SLO is allowed to deteriorate. The pieces live where they act:
//   * decision  : controllers set ControllerOutput::floor_mode using floor_mode_required() below,
//                 re-evaluated every window (never sticky);
//   * action    : Planner::on_probe returns PriceList::floor_ef(g) while the snapshot says floor mode;
//   * per query : bucket_check() sets forced_floor when one query cannot afford its floor work;
//   * recovery  : TokenBucket clamps debt at -D, so it is full again within T_base after overload.
#include "laef/controllers/budget_controller.h"

namespace laef {

// Infeasible: the queue leaves less room than a floor-speed query needs, or floor demand exceeds hardware.
inline bool floor_mode_required(const PhysicalLimits& l) { return l.room < l.S_floor || l.B_min > l.B_max; }

}  // namespace laef
