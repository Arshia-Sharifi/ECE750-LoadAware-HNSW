// LA-ef: sensor window tests (IMPLEMENTATION_PLAN.md §13.1, row test_sensor_window.cpp).
// Exact nearest-rank p95 against a sorted reference, and the close rules of sensor_window.h:
// the window closes on n completions, or on max_ms elapsed AND at least min_samples new completions.
// Config defaults used here (config.h WindowCfg): n = 500, max_ms = 20, min_samples = 50.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "laef/sensor_window.h"

using namespace laef;

namespace {
constexpr int64_t MS = 1'000'000;  // nanoseconds per millisecond

// Reference nearest-rank q-quantile (q = 0.95) over the last min(total, n) values:
// sort them and take the element at index ceil(q * m) - 1.
double reference_p95(const std::vector<double>& all, size_t n) {
  const size_t m = std::min(all.size(), n);
  std::vector<double> last(all.end() - static_cast<std::ptrdiff_t>(m), all.end());
  std::sort(last.begin(), last.end());
  const size_t idx = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(m))) - 1;
  return last[idx];
}
}  // namespace

TEST(SensorWindowExact, P95MatchesSortedReferenceOverRandomData) {
  std::mt19937 rng(1);  // fixed seed: the data are reproducible
  std::uniform_real_distribution<double> U(0.0, 1.0);
  SensorWindow w(500, 20, 50);
  std::vector<double> T_all, S_all;
  size_t prev = 0;
  // Checkpoints: partly filled buffer, exactly full buffer, and two wrap-arounds of the ring.
  for (size_t target : {137u, 500u, 1234u}) {
    while (T_all.size() < target) {
      const double T = U(rng), S = 0.1 * U(rng);  // independent S values, so s_p95 gets its own reference
      w.push(T, S);
      T_all.push_back(T);
      S_all.push_back(S);
    }
    const WindowStats st = w.close(0);
    EXPECT_DOUBLE_EQ(st.p95, reference_p95(T_all, 500)) << "after " << target << " completions";
    EXPECT_DOUBLE_EQ(st.s_p95, reference_p95(S_all, 500)) << "after " << target << " completions";
    EXPECT_EQ(st.n_buffer, std::min<size_t>(target, 500));
    EXPECT_EQ(st.n_new, target - prev);
    prev = target;
  }
}

TEST(SensorWindowExact, NearestRankEdgeSizes) {
  // m = 1: ceil(0.95) = 1, the only sample.
  SensorWindow one(500, 20, 50);
  one.push(3.0, 0.3);
  EXPECT_DOUBLE_EQ(one.close(0).p95, 3.0);

  // m = 2: ceil(1.9) = 2, so the larger value is the p95.
  SensorWindow two(500, 20, 50);
  two.push(7.0, 0);
  two.push(2.0, 0);
  EXPECT_DOUBLE_EQ(two.close(0).p95, 7.0);

  // m = 21: ceil(19.95) = 20, so the 20th smallest of 1..21 (value 20), not the maximum.
  SensorWindow twenty_one(500, 20, 50);
  for (int v = 21; v >= 1; --v) twenty_one.push(v, 0);
  EXPECT_DOUBLE_EQ(twenty_one.close(0).p95, 20.0);
}

TEST(SensorWindowClose, ClosesOnNCompletionsWithoutWaitingForTime) {
  SensorWindow w(500, 20, 50);
  w.start(0);
  for (int i = 0; i < 499; ++i) w.push(1, 1);
  EXPECT_FALSE(w.ready(0));  // 499 < n and no time has elapsed
  w.push(1, 1);
  EXPECT_TRUE(w.ready(0));   // the 500th completion closes the window at once
}

TEST(SensorWindowClose, TimeRuleNeedsMinSamples) {
  SensorWindow w(500, 20, 50);
  w.start(0);
  for (int i = 0; i < 10; ++i) w.push(1, 1);
  EXPECT_FALSE(w.ready(1000 * MS));  // a whole second elapsed, but only 10 new completions
  for (int i = 0; i < 39; ++i) w.push(1, 1);
  EXPECT_FALSE(w.ready(25 * MS));    // 25 ms elapsed, 49 new completions (< 50)
  w.push(1, 1);
  EXPECT_TRUE(w.ready(25 * MS));     // 25 ms elapsed, 50 new completions
}

TEST(SensorWindowClose, TimeRuleNeedsMaxMsElapsed) {
  SensorWindow w(500, 20, 50);
  w.start(0);
  for (int i = 0; i < 200; ++i) w.push(1, 1);  // plenty of samples
  EXPECT_FALSE(w.ready(20 * MS - 1));          // one nanosecond short of max_ms
  EXPECT_TRUE(w.ready(20 * MS));               // exactly max_ms
}

TEST(SensorWindowClose, CloseResetsCountersAndRestartsTheTimer) {
  SensorWindow w(500, 20, 50);
  w.start(0);
  for (int i = 0; i < 50; ++i) w.push(1, 1);
  ASSERT_TRUE(w.ready(20 * MS));
  const WindowStats st = w.close(20 * MS);
  EXPECT_EQ(st.n_new, 50u);
  EXPECT_FALSE(w.ready(30 * MS));  // no completions since the close
  for (int i = 0; i < 50; ++i) w.push(1, 1);
  EXPECT_FALSE(w.ready(30 * MS));  // only 10 ms since the close
  EXPECT_TRUE(w.ready(40 * MS));   // 20 ms since the close and 50 new completions
}
