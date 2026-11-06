// LA-ef: allocator tests on a hand-built two-group price list (IMPLEMENTATION_PLAN.md §13.1, row test_allocators.cpp).
// Plan row: "uniform gives the same work fraction to every group; water_filling cuts the cheapest-ell group
// first and leaves the hard group's ef alone in the hand-built example; neither goes below k; the bucket
// check forces a step down and sets forced_floor."
// Suite names use the prefix AllocatorSpec so they do not collide with the Allocators.* tests in test_price_list.cpp.
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>

#include "laef/allocators/allocator.h"
#include "laef/allocators/passthrough.h"
#include "laef/allocators/uniform_cut.h"
#include "laef/allocators/water_filling.h"
#include "laef/plan_snapshot.h"
#include "laef/price_list.h"

using namespace laef;

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

// k = 10, ladder {10, 20, 40, 80} for both groups. Ladders lie on the concave envelope, so finalize() accepts them.
//   easy (score 0..49, p = 0.5): work {1000, 1500, 2500, 4500}, recall {0.90, 0.97, 0.98, 0.985}, ef_pref 40
//   hard (score 50..100, p = 0.5): work {1000, 2000, 4000, 8000}, recall {0.50, 0.75, 0.90, 0.96}, ef_pref 80
// Step losses (d_recall / d_work): easy 40->20 = 1e-5, hard 80->40 = 1.5e-5, hard 40->20 = 7.5e-5,
// easy 20->10 = 1.4e-4, hard 20->10 = 2.5e-4. Model nominal work = 0.5*2500 + 0.5*8000 = 5250 dc.
PriceList two_groups() {
  PriceList pl;
  pl.k = 10;
  GroupCurve e{0, 0, 49, 0.5, 40, {10, 20, 40, 80}, {0.90, 0.97, 0.98, 0.985}, {1000, 1500, 2500, 4500}, {}};
  GroupCurve h{1, 50, 100, 0.5, 80, {10, 20, 40, 80}, {0.50, 0.75, 0.90, 0.96}, {1000, 2000, 4000, 8000}, {}};
  pl.groups = {e, h};
  pl.finalize();
  return pl;
}

// Snapshot for an overload target w_target (dc per arrival), derived the way make_snapshot() derives it.
PlanSnapshot snap_for(const PriceList& pl, double w_target) {
  PlanSnapshot s;
  s.w_target = w_target;
  s.water_level = pl.water_level(w_target);
  s.uniform_fraction = std::min(1.0, w_target / pl.model_nominal_work());
  return s;
}
}  // namespace

TEST(AllocatorSpec, UniformAppliesTheSameFractionToEveryGroupAndPicksTheLargestFittingRung) {
  const PriceList pl = two_groups();
  const UniformCutAllocator uc{};
  for (double f : {0.6, 0.7, 0.9}) {
    PlanSnapshot s;
    s.uniform_fraction = f;
    for (int g = 0; g < 2; ++g) {
      const uint32_t pref = g == 0 ? 40u : 80u;
      const AllocResult r = uc.plan({g, pref, 10, kInf}, s, pl);
      const double limit = f * pl.work(g, pref);  // the same f, applied to this group's own preferred work
      EXPECT_LE(r.planned_work, limit + 1e-6) << "group " << g << " f " << f;
      // The rung is discrete, so the achieved fraction can fall short of f. No larger rung below ef_pref may fit.
      for (uint32_t rung : pl.rungs(g))
        if (rung < pref && rung > r.ef) EXPECT_GT(pl.work(g, rung), limit + 1e-9) << "group " << g << " f " << f;
    }
  }
  // f = 0.6: easy 0.6 * 2500 = 1500 is exactly rung 20; hard 0.6 * 8000 = 4800 falls back to rung 40 (4000).
  PlanSnapshot s;
  s.uniform_fraction = 0.6;
  EXPECT_EQ(uc.plan({0, 40, 10, kInf}, s, pl).ef, 20u);
  EXPECT_EQ(uc.plan({1, 80, 10, kInf}, s, pl).ef, 40u);
}

TEST(AllocatorSpec, UniformWithFractionOneKeepsEveryPreferredEf) {
  const PriceList pl = two_groups();
  const UniformCutAllocator uc{};
  PlanSnapshot s;
  s.uniform_fraction = 1.0;
  EXPECT_EQ(uc.plan({0, 40, 10, kInf}, s, pl).ef, 40u);
  EXPECT_EQ(uc.plan({1, 80, 10, kInf}, s, pl).ef, 80u);
}

TEST(AllocatorSpec, UniformEfModeScalesEfAndStillStopsAtK) {
  const PriceList pl = two_groups();
  const UniformCutAllocator uc(UniformCutAllocator::Mode::Ef);
  EXPECT_EQ(uc.name(), "uniform_cut_ef");
  PlanSnapshot s;
  s.uniform_fraction = 0.5;
  EXPECT_EQ(uc.plan({1, 80, 10, kInf}, s, pl).ef, 40u);  // 0.5 * 80
  EXPECT_EQ(uc.plan({0, 40, 10, kInf}, s, pl).ef, 20u);  // 0.5 * 40
  s.uniform_fraction = 0.01;
  EXPECT_EQ(uc.plan({1, 80, 10, kInf}, s, pl).ef, 10u);  // max(k, 0.8) = k, never lower
}

TEST(AllocatorSpec, WaterFillingTakesTheCheapestLossFirstAndLeavesTheHardEfAlone) {
  const PriceList pl = two_groups();
  const WaterFillingAllocator wf{};
  // Overload 5000 dc: the water level is 1e-5, the easy 40->20 step. The hard query's first step (1.5e-5) is
  // dearer, so the hard query keeps ef 80.
  const PlanSnapshot s5 = snap_for(pl, 5000.0);
  EXPECT_NEAR(s5.water_level, 1e-5, 1e-12);
  EXPECT_EQ(wf.plan({0, 40, 10, kInf}, s5, pl).ef, 20u);
  EXPECT_EQ(wf.plan({1, 80, 10, kInf}, s5, pl).ef, 80u);

  // Overload 4000 dc: 1250 dc must go. The easy step saves 500 and the hard 80->40 step saves 2000 (after
  // weighting by p), so the level rises to 1.5e-5 and the hard group takes its first step.
  const PlanSnapshot s4 = snap_for(pl, 4000.0);
  EXPECT_NEAR(s4.water_level, 1.5e-5, 1e-12);
  EXPECT_EQ(wf.plan({0, 40, 10, kInf}, s4, pl).ef, 20u);
  EXPECT_EQ(wf.plan({1, 80, 10, kInf}, s4, pl).ef, 40u);
}

TEST(AllocatorSpec, NoCutWithoutOverload) {
  const PriceList pl = two_groups();
  const WaterFillingAllocator wf{};
  const PlanSnapshot none = snap_for(pl, 1e9);  // water level 0
  EXPECT_EQ(wf.plan({1, 80, 10, kInf}, none, pl).ef, 80u);
  EXPECT_EQ(wf.plan({0, 40, 10, kInf}, none, pl).ef, 40u);
}

TEST(AllocatorSpec, NothingGoesBelowKForAnyWaterLevelFractionOrBudget) {
  const PriceList pl = two_groups();
  const WaterFillingAllocator wf{};
  const UniformCutAllocator uc{};
  const PassthroughAllocator pa{};
  const PlanSnapshot all_floor = snap_for(pl, 1.0);  // water level +inf: every step is taken
  PlanSnapshot zero_fraction;
  zero_fraction.uniform_fraction = 0.0;
  for (int g = 0; g < 2; ++g) {
    for (uint32_t pref : {5u, 10u, 40u, 80u}) {  // 5 < k: the allocator clamps its own output
      EXPECT_EQ(wf.plan({g, pref, 10, kInf}, all_floor, pl).ef, 10u) << "wf g " << g << " pref " << pref;
      EXPECT_EQ(wf.plan({g, pref, 10, 0.0}, PlanSnapshot{}, pl).ef, 10u) << "wf zero budget g " << g;
      EXPECT_EQ(uc.plan({g, pref, 10, kInf}, zero_fraction, pl).ef, 10u) << "uc g " << g << " pref " << pref;
      EXPECT_GE(pa.plan({g, pref, 10, kInf}, all_floor, pl).ef, 10u) << "pa g " << g << " pref " << pref;
    }
  }
}

TEST(AllocatorSpec, BucketCheckStepsDownOneRungAtATimeUntilAffordable) {
  const PriceList pl = two_groups();
  const WaterFillingAllocator wf{};
  const PlanSnapshot none = snap_for(pl, 1e9);  // no price-list cuts, so only the bucket acts
  // Hard query, ef_pref 80 (work 8000 dc).
  AllocResult r = wf.plan({1, 80, 10, 4000.0}, none, pl);  // rung 40 costs exactly 4000: affordable
  EXPECT_EQ(r.ef, 40u);
  EXPECT_FALSE(r.forced_floor);
  EXPECT_DOUBLE_EQ(r.planned_work, 4000.0);
  r = wf.plan({1, 80, 10, 3999.0}, none, pl);               // 40 is 1 dc too dear: one rung lower
  EXPECT_EQ(r.ef, 20u);
  EXPECT_FALSE(r.forced_floor);
  EXPECT_DOUBLE_EQ(r.planned_work, 2000.0);
  r = wf.plan({1, 80, 10, 999.0}, none, pl);                // not even rung 10 (1000 dc) is affordable
  EXPECT_EQ(r.ef, 10u);
  EXPECT_TRUE(r.forced_floor);                              // still served, never dropped
  EXPECT_DOUBLE_EQ(r.planned_work, 1000.0);
  r = wf.plan({1, 80, 10, 1000.0}, none, pl);               // exactly the floor work: not forced
  EXPECT_EQ(r.ef, 10u);
  EXPECT_FALSE(r.forced_floor);

  // Off-ladder ef_pref 33 (work 2150): 1800 dc affords rung 20 (1500).
  r = wf.plan({0, 33, 10, 1800.0}, none, pl);
  EXPECT_EQ(r.ef, 20u);
  EXPECT_FALSE(r.forced_floor);
}

TEST(AllocatorSpec, BucketCheckIsSharedByUniformButPassthroughIgnoresIt) {
  const PriceList pl = two_groups();
  PlanSnapshot s;
  s.uniform_fraction = 1.0;  // no uniform cut: only the bucket acts
  const UniformCutAllocator uc{};
  AllocResult r = uc.plan({1, 80, 10, 999.0}, s, pl);
  EXPECT_EQ(r.ef, 10u);
  EXPECT_TRUE(r.forced_floor);

  const PassthroughAllocator pa{};
  EXPECT_FALSE(pa.uses_bucket());
  r = pa.plan({1, 80, 10, 0.0}, s, pl);  // the affordable value is not consulted
  EXPECT_EQ(r.ef, 80u);
  EXPECT_FALSE(r.forced_floor);

  const PassthroughAllocator static_ef(40);
  EXPECT_EQ(static_ef.name(), "static_ef");
  EXPECT_EQ(static_ef.plan({1, 80, 10, kInf}, s, pl).ef, 40u);
  EXPECT_TRUE(WaterFillingAllocator().uses_bucket());
  EXPECT_TRUE(UniformCutAllocator().uses_bucket());
}
