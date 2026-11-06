#pragma once
// LA-ef: shared work balance (Step E). IMPLEMENTATION_PLAN.md §10.5 / §10.7, Main Flow.md §5.
//   - refills lazily at B [dc/s], capped at +cap_pos (default 0 => total burst credit = D)
//   - overdraft floor -D; debits CLAMP at -D (debt beyond -D is forgiven; queries are never refused)
//   - all multi-step operations happen under ONE lock (settle()).
#include <algorithm>
#include <cstdint>
#include <mutex>

namespace laef {

class TokenBucket {
 public:
  explicit TokenBucket(double cap_pos = 0.0) : cap_pos_(cap_pos), balance_(cap_pos) {}

  // Arrival-time reservation of nominal work. Returns the amount that actually reduced the balance.
  // If the debit was clipped at -D, the forgiven portion must not be refunded later.
  double reserve(double w, int64_t now_ns, double B, double D) {
    std::lock_guard<std::mutex> g(m_);
    refill_locked(now_ns, B, D);
    const double before = balance_;
    debit_locked(w, D);
    return std::max(0.0, before - balance_);
  }

  // Probe-time settlement: refund the amount actually charged, compute affordable = balance + D,
  // let `plan` choose the work to spend (double plan(double affordable)), and debit it.
  template <class PlanFn>
  double settle(double refund, int64_t now_ns, double B, double D, PlanFn&& plan) {
    std::lock_guard<std::mutex> g(m_);
    refill_locked(now_ns, B, D);
    balance_ = std::min(cap_pos_, balance_ + refund);
    const double w = plan(balance_ + D);
    debit_locked(w, D);
    return w;
  }

  double affordable(int64_t now_ns, double B, double D) {
    std::lock_guard<std::mutex> g(m_);
    refill_locked(now_ns, B, D);
    return balance_ + D;
  }

  double balance() const { std::lock_guard<std::mutex> g(m_); return balance_; }

 private:
  void refill_locked(int64_t now_ns, double B, double D) {
    if (started_) {
      const double dt = std::max<int64_t>(0, now_ns - last_ns_) * 1e-9;
      balance_ = std::min(cap_pos_, balance_ + B * dt);
    }
    started_ = true;
    last_ns_ = std::max(last_ns_, now_ns);
    balance_ = std::max(balance_, -D);  // D shrinks when B is cut: re-apply the floor
  }
  void debit_locked(double w, double D) { balance_ = std::max(balance_ - w, -D); }

  mutable std::mutex m_;
  double cap_pos_;
  double balance_;
  int64_t last_ns_ = 0;
  bool started_ = false;
};

}  // namespace laef
