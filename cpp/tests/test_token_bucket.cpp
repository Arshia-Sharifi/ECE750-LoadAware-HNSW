// LA-ef: token bucket + sensor window tests (IMPLEMENTATION_PLAN.md §13.1).
#include <gtest/gtest.h>

#include <random>

#include "laef/sensor_window.h"
#include "laef/token_bucket.h"

using namespace laef;

constexpr int64_t MS = 1'000'000;

TEST(TokenBucket, Section8Phase2GracePeriod) {
  // B = 4M dc/s, D = B * T_base = 4M * 6ms = 24,000 dc. Demand 2000 q/s x 4000 dc = 8M dc/s.
  TokenBucket b;
  const double B = 4e6, D = 24000;
  int64_t t = 0;
  b.reserve(0, t, B, D);
  int served = 0;
  while (b.affordable(t, B, D) >= 4000) {  // a query can still pay its preferred work
    b.reserve(4000, t, B, D);
    t += MS / 2;  // 2000 q/s
    ++served;
  }
  const double grace_ms = t / 1e6;
  EXPECT_NEAR(grace_ms, 6.0, 0.6);  // ~6 ms (discretisation of 0.5 ms arrivals)
}

TEST(TokenBucket, CapAndOverdraftClamp) {
  TokenBucket b;  // cap 0
  b.reserve(0, 0, 1e6, 1000);
  EXPECT_DOUBLE_EQ(b.balance(), 0);
  b.reserve(5000, 0, 1e6, 1000);   // debt beyond -D is forgiven
  EXPECT_DOUBLE_EQ(b.balance(), -1000);
  b.reserve(0, 10 * MS, 1e6, 1000);  // refill 10,000 dc but capped at 0
  EXPECT_DOUBLE_EQ(b.balance(), 0);
}

TEST(TokenBucket, ClippedReservationRefundsOnlyChargedAmount) {
  TokenBucket b;
  const double B = 0, D = 1000;
  const double first = b.reserve(800, 0, B, D);
  const double second = b.reserve(800, 0, B, D);  // only 200 can reduce the balance to -D
  EXPECT_DOUBLE_EQ(first, 800);
  EXPECT_DOUBLE_EQ(second, 200);
  EXPECT_DOUBLE_EQ(b.balance(), -1000);

  b.settle(first, 0, B, D, [](double) { return 800.0; });
  double affordable = -1;
  b.settle(second, 0, B, D, [&](double a) {
    affordable = a;
    return 0.0;
  });
  EXPECT_DOUBLE_EQ(affordable, 200);
  EXPECT_DOUBLE_EQ(b.balance(), -800);
}

TEST(TokenBucket, RecoversWithinTbaseAfterSustainedOverload) {
  TokenBucket b;
  const double B = 4e6, D = 24000;  // T_base = 6 ms
  int64_t t = 0;
  for (int i = 0; i < 10000; ++i, t += MS / 10) b.reserve(4000, t, B, D);  // 40M dc/s for 1 s
  EXPECT_DOUBLE_EQ(b.balance(), -D);
  EXPECT_DOUBLE_EQ(b.affordable(t + 6 * MS, B, D), D);  // full again after T_base
}

TEST(TokenBucket, SettleRefundsThenDebitsUnderOneLock) {
  TokenBucket b;
  b.reserve(3000, 0, 0, 10000);  // balance -3000
  double seen = -1;
  double w = b.settle(3000, 0, 0, 10000, [&](double aff) { seen = aff; return 1200.0; });
  EXPECT_DOUBLE_EQ(seen, 10000);  // refunded to 0, + D
  EXPECT_DOUBLE_EQ(w, 1200);
  EXPECT_DOUBLE_EQ(b.balance(), -1200);
}

TEST(TokenBucket, ShrinkingDReappliesFloor) {
  TokenBucket b;
  b.reserve(9000, 0, 0, 10000);
  EXPECT_DOUBLE_EQ(b.affordable(0, 0, 2000), 0);  // balance clamped to new -D
}

TEST(SensorWindow, ExactP95MatchesSortedReference) {
  SensorWindow w(500, 20, 50);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> U(0, 1);
  std::vector<double> T;
  for (int i = 0; i < 700; ++i) { double x = U(rng); w.push(x, x / 2); T.push_back(x); }
  std::vector<double> last(T.end() - 500, T.end());
  std::sort(last.begin(), last.end());
  auto s = w.close(0);
  EXPECT_DOUBLE_EQ(s.p95, last[474]);  // ceil(0.95*500)-1
  EXPECT_DOUBLE_EQ(s.s_p95, last[474] / 2);
  EXPECT_EQ(s.n_buffer, 500u);
}

TEST(SensorWindow, CloseRules) {
  SensorWindow w(500, 20, 50);
  w.start(0);
  for (int i = 0; i < 40; ++i) w.push(1, 1);
  EXPECT_FALSE(w.ready(25 * MS));  // time elapsed but < min_samples
  for (int i = 0; i < 10; ++i) w.push(1, 1);
  EXPECT_FALSE(w.ready(10 * MS));  // enough samples but not enough time
  EXPECT_TRUE(w.ready(25 * MS));
  w.close(25 * MS);
  for (int i = 0; i < 500; ++i) w.push(1, 1);
  EXPECT_TRUE(w.ready(26 * MS));   // n completions closes immediately
}
