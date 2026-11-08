// LA-ef: P-AIMD / fixed-AIMD / static controller tests (IMPLEMENTATION_PLAN.md §13.1).
// Numbers from the Main Flow.md §8 walkthrough: C=4, u=1us, T_SLO=10ms, lambda=2000/s, c(k)=1200 dc.
#include <gtest/gtest.h>

#include "laef/controllers/fixed_aimd.h"
#include "laef/controllers/p_aimd.h"
#include "laef/controllers/static_capacity.h"

using namespace laef;

namespace {
ControllerInputs base() {
  ControllerInputs in;
  in.C = 4; in.unit_cost = 1e-6; in.t_slo = 10e-3; in.arrival_rate = 2000; in.floor_work = 1200;
  in.mean_service = 2e-3; in.s_p95 = 4e-3; in.queue_depth = 0; in.p95 = 5e-3;
  return in;
}
}  // namespace

TEST(PAimd, Section8Phase3Walkthrough) {
  PAimdController c({}, 4e6);
  auto in = base();
  in.p95 = 10.5e-3; in.queue_depth = 16;  // W_hat = 16*2ms/4 = 8 ms, room = 2 ms
  auto o = c.update(in);
  // 4M * 10/10.5 = 3.8095M ; guard x max(0.5, 2/4) = 1.905M ; clamp to B_min = 2000*1200 = 2.4M
  EXPECT_NEAR(o.B, 2.4e6, 1);
  EXPECT_NEAR(o.B_max, 4e6, 1);
  EXPECT_NEAR(o.B_min, 2.4e6, 1);
  EXPECT_NEAR(o.W_hat, 8e-3, 1e-12);
  EXPECT_NEAR(o.room, 2e-3, 1e-12);
  EXPECT_NEAR(o.S_floor, 1.2e-3, 1e-12);
  EXPECT_TRUE(o.guard_fired);
  EXPECT_FALSE(o.floor_mode);  // room 2 ms >= S_floor 1.2 ms
  EXPECT_STREQ(o.zone, "violation");
}

TEST(PAimd, ProportionalCutWithoutGuard) {
  PAimdController c({}, 4e6);
  auto in = base();
  in.p95 = 10.5e-3; in.arrival_rate = 100;  // B_min small, no queue
  EXPECT_NEAR(c.update(in).B, 4e6 * 10.0 / 10.5, 1);
}

TEST(PAimd, HeadroomGrowsAndCapsAtBmax) {
  PAimdController c({}, 3e6);
  auto in = base();  // p95 = 5ms < 9.5ms
  EXPECT_NEAR(c.update(in).B, 3.06e6, 1);
  c.reset(3.99e6);
  EXPECT_NEAR(c.update(in).B, 4e6, 1);
}

TEST(PAimd, ComfortBandHolds) {
  PAimdController c({}, 3e6);
  auto in = base();
  in.p95 = 9.6e-3;
  auto o = c.update(in);
  EXPECT_NEAR(o.B, 3e6, 1);
  EXPECT_STREQ(o.zone, "comfort");
}

TEST(PAimd, SevereViolationCutsAtMostHalf) {
  PAimdController c({}, 4e6);
  auto in = base();
  in.p95 = 25e-3; in.arrival_rate = 100;
  EXPECT_NEAR(c.update(in).B, 2e6, 1);
}

TEST(PAimd, RoomBelowFloorEntersFloorModeNoDrop) {
  PAimdController c({}, 4e6);
  auto in = base();
  in.queue_depth = 18;  // W_hat = 9 ms, room = 1 ms < 1.2 ms
  auto o = c.update(in);
  EXPECT_TRUE(o.floor_mode);
  EXPECT_NEAR(o.B, 2.4e6, 1);  // min(B_max, B_min)
  in.queue_depth = 0;          // next window: room recovered -> floor mode clears
  EXPECT_FALSE(c.update(in).floor_mode);
}

TEST(PAimd, InfeasibleLoadCeilingWins) {
  PAimdController c({}, 4e6);
  auto in = base();
  in.arrival_rate = 5000;  // B_min = 6M > B_max = 4M
  auto o = c.update(in);
  EXPECT_NEAR(o.B, 4e6, 1);
  EXPECT_TRUE(o.floor_mode);
}

TEST(PAimd, FirstWindowStartsAtCeiling) {
  PAimdController c;
  auto in = base();
  in.p95 = 9e-3;
  EXPECT_NEAR(c.update(in).B, 4e6, 1);
}

TEST(FixedAimd, CutIgnoresSeverity) {
  FixedAimdController c({}, 4e6);
  auto in = base();
  in.arrival_rate = 100; in.p95 = 25e-3;
  EXPECT_NEAR(c.update(in).B, 3.6e6, 1);
  c.reset(4e6); in.p95 = 10.1e-3;
  EXPECT_NEAR(c.update(in).B, 3.6e6, 1);
}

TEST(StaticCapacity, FractionOfCeiling) {
  StaticCapacityController c(0.9);
  auto in = base();
  in.arrival_rate = 100;
  EXPECT_NEAR(c.update(in).B, 3.6e6, 1);
}
