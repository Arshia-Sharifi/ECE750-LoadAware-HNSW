#pragma once
// LA-ef: BudgetController interface (IMPLEMENTATION_PLAN.md §10.4, Budget B calculations.md §5).
// Units: seconds, dc (distance computations), dc/s. Never drops queries: infeasibility -> floor_mode.
#include <algorithm>
#include <limits>
#include <string>

namespace laef {

struct ControllerInputs {
  double p95 = 0;           // end-to-end p95 latency of the window [s]
  double s_p95 = 0;         // p95 service time of the window [s]
  double queue_depth = 0;   // Q (instantaneous)
  double mean_service = 0;  // E[S] EWMA [s]
  double unit_cost = 0;     // u EWMA [s/dc]
  double arrival_rate = 0;  // lambda-hat EWMA [queries/s]
  double floor_work = 0;    // c(k) [dc/query]
  double t_slo = 0;         // T_SLO [s]
  int C = 1;                // worker threads
};

struct ControllerOutput {
  double B = 0, B_max = 0, B_min = 0, S_floor = 0;
  double W_hat = 0, room = 0;   // estimated queue wait, T_SLO - W_hat [s]
  bool floor_mode = false;      // room < S_floor or B_min > B_max (re-evaluated every window)
  bool guard_fired = false;
  const char* zone = "init";    // violation | headroom | comfort | static
};

class BudgetController {
 public:
  virtual ~BudgetController() = default;
  virtual ControllerOutput update(const ControllerInputs&) = 0;
  virtual void reset(double B0) = 0;
  virtual double budget() const = 0;
  virtual std::string name() const = 0;
};

// Shared physical quantities (BB §1 step 1, §3).
struct PhysicalLimits { double B_max, B_min, S_floor, W_hat, room; };

inline PhysicalLimits physical_limits(const ControllerInputs& in) {
  const double inf = std::numeric_limits<double>::infinity();
  PhysicalLimits l{};
  l.B_max = in.unit_cost > 0 ? in.C / in.unit_cost : inf;
  l.B_min = std::max(0.0, in.arrival_rate) * in.floor_work;
  l.S_floor = in.floor_work * in.unit_cost;
  l.W_hat = in.C > 0 ? in.queue_depth * in.mean_service / in.C : 0.0;
  l.room = in.t_slo - l.W_hat;
  return l;
}

// Final clamp: the ceiling always beats the floor (BB §1 step 4).
inline double clamp_budget(double B, const PhysicalLimits& l) {
  return std::min(l.B_max, std::max(B, l.B_min));
}

inline void fill_common(ControllerOutput& o, const PhysicalLimits& l, double B) {
  o.B = B; o.B_max = l.B_max; o.B_min = l.B_min; o.S_floor = l.S_floor; o.W_hat = l.W_hat; o.room = l.room;
}

}  // namespace laef
