#pragma once
// LA-ef: EWMA telemetry, Step B (IMPLEMENTATION_PLAN.md §10.1, Main Flow.md §2).
//   u    = EWMA(S_i / ndis_i)   [s/dc]       updated by workers on completion
//   E[S] = EWMA(S_i)            [s]          updated by workers on completion
//   λ̂    = EWMA(arrivals/tick)  [queries/s]  updated by the controller thread every tick
#include <cstdint>
#include <mutex>

namespace laef {

struct Ewma {
  double alpha;
  double value = 0;
  bool init = false;

  explicit Ewma(double a) : alpha(a) {}
  void seed(double x) { value = x; init = true; }
  void add(double x) {
    value = init ? alpha * x + (1 - alpha) * value : x;
    init = true;
  }
  double get(double fallback = 0) const { return init ? value : fallback; }
};

class Telemetry {
 public:
  struct View { double unit_cost, mean_service, arrival_rate; };

  // u0 / es0 seed the EWMAs (from the profile) so the first controller window has sane inputs.
  Telemetry(double alpha_u, double alpha_es, double alpha_lambda, double u0, double es0)
      : u_(alpha_u), es_(alpha_es), lambda_(alpha_lambda) {
    if (u0 > 0) u_.seed(u0);
    if (es0 > 0) es_.seed(es0);
  }

  void on_completion(double service_s, uint32_t ndis) {
    std::lock_guard<std::mutex> g(m_);
    if (ndis > 0) u_.add(service_s / ndis);
    es_.add(service_s);
  }

  void on_lambda_tick(double arrivals_per_s) {
    std::lock_guard<std::mutex> g(m_);
    lambda_.add(arrivals_per_s);
  }

  View view() const {
    std::lock_guard<std::mutex> g(m_);
    return {u_.get(), es_.get(), lambda_.get(0)};
  }

 private:
  mutable std::mutex m_;
  Ewma u_, es_, lambda_;
};

}  // namespace laef
