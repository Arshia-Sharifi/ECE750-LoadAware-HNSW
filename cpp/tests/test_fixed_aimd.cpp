// LA-ef: fixed-AIMD baseline tests (IMPLEMENTATION_PLAN.md §13.1, row test_fixed_aimd.cpp).
// Plan row: "Fixed 10% cut regardless of severity; same clamps."
// Inputs mirror test_p_aimd.cpp: C=4, u=1 us/dc, T_SLO=10 ms, lambda=2000/s, c(k)=1200 dc.
//   B_max = C/u = 4e6 dc/s,  B_min = lambda*c(k) = 2.4e6 dc/s,  S_floor = c(k)*u = 1.2 ms.
// Defaults (fixed_aimd.h FixedAimdParams): fixed_cut 0.90, headroom 0.85, growth 1.02, guard off.
#include <gtest/gtest.h>

#include <algorithm>

#include "laef/controllers/fixed_aimd.h"

using namespace laef;

namespace {
constexpr double kBmax = 4e6, kBmin = 2.4e6;

ControllerInputs base() {
  ControllerInputs in;
  in.C = 4; in.unit_cost = 1e-6; in.t_slo = 10e-3; in.arrival_rate = 2000; in.floor_work = 1200;
  in.mean_service = 2e-3; in.s_p95 = 4e-3; in.queue_depth = 0; in.p95 = 5e-3;
  return in;
}
}  // namespace

TEST(FixedAimdSpec, CutIsTenPercentForMildAndSevereViolations) {
  // The same 10% cut for every violation, whatever the size of the overshoot.
  for (double p95_ms : {10.1, 10.5, 15.0, 25.0, 100.0}) {
    FixedAimdController c({}, kBmax);
    auto in = base();
    in.p95 = p95_ms * 1e-3;
    const ControllerOutput o = c.update(in);
    EXPECT_NEAR(o.B, 0.9 * kBmax, 1) << "p95 = " << p95_ms << " ms";
    EXPECT_STREQ(o.zone, "violation") << "p95 = " << p95_ms << " ms";
    EXPECT_FALSE(o.floor_mode);
  }
}

TEST(FixedAimdSpec, RepeatedViolationsCompoundAndStopAtBMin) {
  FixedAimdController c({}, kBmax);
  auto in = base();
  in.p95 = 25e-3;  // severe, every window
  double expect = kBmax;
  for (int i = 0; i < 6; ++i) {
    expect = std::max(0.9 * expect, kBmin);  // the clamp applies to the stored budget too
    EXPECT_NEAR(c.update(in).B, expect, 1) << "window " << i;
  }
  EXPECT_NEAR(c.budget(), kBmin, 1);  // 0.9^5 * B_max is below B_min, so the floor clamp holds
}

TEST(FixedAimdSpec, HeadroomGrowsByTwoPercentAndCapsAtBMax) {
  FixedAimdController c({}, 3e6);
  auto in = base();  // p95 = 5 ms < 0.85 * T_SLO
  EXPECT_NEAR(c.update(in).B, 3.06e6, 1);
  c.reset(3.99e6);
  EXPECT_NEAR(c.update(in).B, kBmax, 1);  // 3.99e6 * 1.02 is capped at B_max
  for (int i = 0; i < 50; ++i) EXPECT_NEAR(c.update(in).B, kBmax, 1) << "window " << i;
}

TEST(FixedAimdSpec, ComfortBandHoldsTheBudget) {
  FixedAimdController c({}, 3e6);
  auto in = base();
  in.p95 = 9e-3;  // between 0.85 * T_SLO and T_SLO
  const ControllerOutput o = c.update(in);
  EXPECT_NEAR(o.B, 3e6, 1);
  EXPECT_STREQ(o.zone, "comfort");
}

TEST(FixedAimdSpec, FirstWindowStartsAtCeiling) {
  FixedAimdController c;  // B0 = 0: the first update starts from B_max
  auto in = base();
  in.p95 = 9e-3;
  EXPECT_NEAR(c.update(in).B, kBmax, 1);
}

TEST(FixedAimdSpec, InfeasibleLoadPinsCeilingAndSetsFloorMode) {
  FixedAimdController c({}, kBmax);
  auto in = base();
  in.arrival_rate = 5000;  // B_min = 6e6 > B_max = 4e6: even the all-floor demand exceeds the hardware
  const ControllerOutput o = c.update(in);
  EXPECT_NEAR(o.B, kBmax, 1);
  EXPECT_TRUE(o.floor_mode);
}

TEST(FixedAimdSpec, RoomBelowFloorSetsFloorModeAndClearsNextWindow) {
  FixedAimdController c({}, kBmax);
  auto in = base();
  in.queue_depth = 18;  // W_hat = 18 * 2 ms / 4 = 9 ms, room = 1 ms < S_floor = 1.2 ms
  ControllerOutput o = c.update(in);
  EXPECT_TRUE(o.floor_mode);
  EXPECT_NEAR(o.B, kBmin, 1);  // min(B_max, B_min)
  in.queue_depth = 0;          // room recovers: floor mode is re-evaluated every window, not sticky
  o = c.update(in);
  EXPECT_FALSE(o.floor_mode);
  EXPECT_NEAR(o.B, kBmin * 1.02, 1);  // headroom growth from the floor value
}

TEST(FixedAimdSpec, GuardIsOffByDefault) {
  // s_p95 (12 ms) is above the room (10 ms). The default baseline ignores that; only the cut applies.
  FixedAimdController c({}, kBmax);
  auto in = base();
  in.p95 = 10.5e-3;
  in.s_p95 = 12e-3;
  const ControllerOutput o = c.update(in);
  EXPECT_FALSE(o.guard_fired);
  EXPECT_NEAR(o.B, 0.9 * kBmax, 1);

  // Switching the guard on adds max(cut_floor, room / s_p95) = 10/12 on top of the fixed cut.
  FixedAimdParams with_guard;
  with_guard.guard = true;
  FixedAimdController g(with_guard, kBmax);
  const ControllerOutput og = g.update(in);
  EXPECT_TRUE(og.guard_fired);
  EXPECT_NEAR(og.B, 0.9 * kBmax * (10.0 / 12.0), 1);
}
