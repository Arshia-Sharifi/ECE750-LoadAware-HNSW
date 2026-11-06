#pragma once
// LA-ef: sensor window (Step C). IMPLEMENTATION_PLAN.md §10.2, Main Flow.md §3.
// Ring buffer of the last `n` completions (T_i, S_i). Closes when `n` new completions arrived,
// or when `max_ms` elapsed AND at least `min_samples` new completions exist.
// Percentiles are EXACT (nearest-rank: element at index ceil(q*m)-1 of the sorted m samples).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

namespace laef {

inline double nearest_rank(std::vector<double>& v, double q) {
  if (v.empty()) return 0.0;
  size_t idx = static_cast<size_t>(std::ceil(q * v.size()));
  idx = std::min(v.size(), std::max<size_t>(1, idx)) - 1;
  std::nth_element(v.begin(), v.begin() + idx, v.end());
  return v[idx];
}

struct WindowStats { double p95 = 0, s_p95 = 0; size_t n_new = 0, n_buffer = 0; };

class SensorWindow {
 public:
  SensorWindow(size_t n = 500, double max_ms = 20, size_t min_samples = 50)
      : n_(n), max_ns_(static_cast<int64_t>(max_ms * 1e6)), min_samples_(min_samples), T_(n), S_(n) {}

  void push(double T, double S) {  // called by workers
    std::lock_guard<std::mutex> g(m_);
    T_[head_] = T; S_[head_] = S;
    head_ = (head_ + 1) % n_;
    size_ = std::min(size_ + 1, n_);
    ++new_;
  }

  bool ready(int64_t now_ns) const {
    std::lock_guard<std::mutex> g(m_);
    if (new_ >= n_) return true;
    return (now_ns - last_close_ns_) >= max_ns_ && new_ >= min_samples_;
  }

  // Computes exact p95 of T and S over the buffer and resets the "new" counter.
  WindowStats close(int64_t now_ns) {
    std::vector<double> t, s;
    WindowStats w;
    {
      std::lock_guard<std::mutex> g(m_);
      t.assign(T_.begin(), T_.begin() + size_);
      s.assign(S_.begin(), S_.begin() + size_);
      w.n_new = new_; w.n_buffer = size_;
      new_ = 0;
      last_close_ns_ = now_ns;
    }
    w.p95 = nearest_rank(t, 0.95);
    w.s_p95 = nearest_rank(s, 0.95);
    return w;
  }

  void start(int64_t now_ns) { std::lock_guard<std::mutex> g(m_); last_close_ns_ = now_ns; }

 private:
  size_t n_;
  int64_t max_ns_;
  size_t min_samples_;
  std::vector<double> T_, S_;
  size_t head_ = 0, size_ = 0, new_ = 0;
  int64_t last_close_ns_ = 0;
  mutable std::mutex m_;
};

}  // namespace laef
