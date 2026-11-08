// LA-ef: price list, envelope, grouping, water level, allocators, planner, probe menu (IMPLEMENTATION_PLAN.md §13.1, §1A.8).
#include <gtest/gtest.h>

#include <cmath>

#include "laef/allocators/passthrough.h"
#include "laef/allocators/uniform_cut.h"
#include "laef/allocators/water_filling.h"
#include "laef/planner.h"
#include "laef/price_list.h"
#include "laef/probe_menu.h"

using namespace laef;

namespace {
// Two groups, k = 10, ladder {10, 20, 40, 80}.
//  easy (p=0.5): recall saturates early -> cuts are cheap        e* = 40
//  hard (p=0.5): recall keeps climbing  -> cuts are expensive    e* = 80
PriceList two_groups() {
  PriceList pl;
  pl.k = 10;
  GroupCurve e{0, 0, 49, 0.5, 40, {10, 20, 40, 80}, {0.90, 0.97, 0.98, 0.985}, {1000, 1500, 2500, 4500}, {}};
  GroupCurve h{1, 50, 100, 0.5, 80, {10, 20, 40, 80}, {0.50, 0.75, 0.90, 0.96}, {1000, 2000, 4000, 8000}, {}};
  pl.groups = {e, h};
  pl.finalize();
  return pl;
}
// The snapshot the controller would publish for an overload target w_target (dc per arrival), with the per-probe
// arrays filled for a one-option menu.
PlanSnapshot snap_for(const PriceList& pl, double w_target) {
  PlanSnapshot s;
  s.w_target = w_target;
  s.water_level = pl.water_level(w_target);
  s.uniform_fraction = std::min(1.0, w_target / pl.model_nominal_work());
  s.water_level_by_L = {s.water_level};
  s.uniform_fraction_by_L = {s.uniform_fraction};
  return s;
}
ProbeMenu one_probe(const PriceList& pl) {
  ProbeOption o;
  o.probe_len = 1025;
  o.price_list = pl;
  return ProbeMenu({o});
}
// One group (p = 1) on the ladder {10, 20, 40}, with the given work and recall per rung.
PriceList one_group(std::vector<double> work, std::vector<double> recall) {
  PriceList pl;
  pl.k = 10;
  GroupCurve g{0, 0, 100, 1.0, 40, {10, 20, 40}, std::move(recall), std::move(work), {}};
  pl.groups = {g};
  pl.finalize();
  return pl;
}
}  // namespace

TEST(Envelope, MonotoneDedupeConcave) {
  std::vector<uint32_t> ef{10, 20, 40, 80, 160};
  std::vector<double> r{0.5, 0.49, 0.9, 0.91, 0.99};   // dip at 20, non-concave at 80
  std::vector<double> w{1000, 1000, 3000, 5000, 9000}; // 10 and 20 equal work
  envelope_inplace(ef, r, w);
  EXPECT_EQ(ef, (std::vector<uint32_t>{20, 40, 160}));  // 20 dominates 10; 80 under chord 40->160
  for (size_t i = 2; i < ef.size(); ++i) {
    double s1 = (r[i - 1] - r[i - 2]) / (w[i - 1] - w[i - 2]), s2 = (r[i] - r[i - 1]) / (w[i] - w[i - 1]);
    EXPECT_GE(s1, s2);
  }
}

TEST(Groups, MergeToMinSizeAndCoverRange) {
  std::vector<size_t> c(101, 0);
  c[3] = 5; c[4] = 40; c[10] = 31; c[60] = 2; c[61] = 50;
  auto g = make_groups(c, 30, 8);
  ASSERT_EQ(g.size(), 3u);
  EXPECT_EQ(g.front().first, 0);
  EXPECT_EQ(g.back().second, 100);
  for (size_t i = 1; i < g.size(); ++i) EXPECT_EQ(g[i].first, g[i - 1].second + 1);
  auto g2 = make_groups(c, 1, 2);
  EXPECT_EQ(g2.size(), 2u);
}

TEST(PriceList, LookupInterpolation) {
  auto pl = two_groups();
  EXPECT_EQ(pl.group_of(12.7f), 0);
  EXPECT_EQ(pl.group_of(50.0f), 1);
  EXPECT_EQ(pl.group_of(150.0f), 1);
  EXPECT_EQ(pl.group_of(-3.0f), 0);
  EXPECT_DOUBLE_EQ(pl.work(0, 30), 2000);
  EXPECT_DOUBLE_EQ(pl.work(0, 5), 1000);
  EXPECT_DOUBLE_EQ(pl.work(0, 999), 4500);
  EXPECT_DOUBLE_EQ(pl.model_nominal_work(), 0.5 * 2500 + 0.5 * 8000);
}

TEST(PriceList, StepsSortedAndWaterLevel) {
  auto pl = two_groups();
  const auto& s = pl.steps_sorted();
  for (size_t i = 1; i < s.size(); ++i) EXPECT_LE(s[i - 1].loss, s[i].loss);
  EXPECT_EQ(s.front().group, 0);  // cheapest cut is in the easy group (40->20: 0.01/1000)
  EXPECT_DOUBLE_EQ(pl.water_level(1e9), 0.0);
  EXPECT_TRUE(std::isinf(pl.water_level(1.0)));  // even all-floor (1000) is too much
  // Need to save 250 per query on average: the easy 40->20 step saves 0.5*1000 = 500 -> enough.
  EXPECT_NEAR(pl.water_level(5000), 0.01 / 1000, 1e-12);
}

TEST(Allocators, WaterFillingCutsEasyFirstUniformCutsEveryone) {
  auto pl = two_groups();
  auto snap = snap_for(pl, 5000);  // overloaded: nominal 5250
  WaterFillingAllocator wf;
  UniformCutAllocator uc;
  const double inf = 1e18;
  EXPECT_EQ(wf.plan({0, 40, 10, inf}, snap, pl).ef, 20u);  // easy query cut
  EXPECT_EQ(wf.plan({1, 80, 10, inf}, snap, pl).ef, 80u);  // hard query keeps its ef
  // uniform f = 5000/5250: easy 2500*f=2381 -> 20 (1500); hard 8000*f=7619 -> 40 (4000)
  EXPECT_EQ(uc.plan({0, 40, 10, inf}, snap, pl).ef, 20u);
  EXPECT_EQ(uc.plan({1, 80, 10, inf}, snap, pl).ef, 40u);
}

TEST(Allocators, NoOverloadNoCut) {
  auto pl = two_groups();
  auto snap = snap_for(pl, 1e9);
  WaterFillingAllocator wf;
  UniformCutAllocator uc;
  EXPECT_EQ(wf.plan({1, 80, 10, 1e18}, snap, pl).ef, 80u);
  EXPECT_EQ(uc.plan({1, 80, 10, 1e18}, snap, pl).ef, 80u);
  EXPECT_EQ(wf.plan({0, 33, 10, 1e18}, snap, pl).ef, 33u);  // off-ladder ef_pref is kept as-is
}

TEST(Allocators, BucketCheckStepsDownAndForcesFloor) {
  auto pl = two_groups();
  auto snap = snap_for(pl, 1e9);
  WaterFillingAllocator wf;
  auto r = wf.plan({1, 80, 10, 2500}, snap, pl);
  EXPECT_EQ(r.ef, 20u);  // 8000 > 2500 -> 40 (4000) -> 20 (2000)
  EXPECT_FALSE(r.forced_floor);
  r = wf.plan({1, 80, 10, 500}, snap, pl);
  EXPECT_EQ(r.ef, 10u);
  EXPECT_TRUE(r.forced_floor);  // still served, never dropped
}

TEST(Planner, FloorModeRunsAtFloorAndKeepsBucketConsistent) {
  auto pl = two_groups();
  const ProbeMenu menu = one_probe(pl);
  WaterFillingAllocator wf;
  TokenBucket bucket;
  PlanSnapshot s = snap_for(pl, 1e9);
  s.B = 1e6; s.D = 1e4; s.reservation = pl.model_nominal_work(); s.floor_mode = true;
  SnapshotStore store(s);
  Planner p(menu, wf, bucket, store, 10);
  QueryCtx qc;
  p.on_arrival(qc, 0);
  EXPECT_EQ(p.choose_probe(qc), 1025u);
  EXPECT_EQ(p.on_probe(80.0f, 80, qc, 0), 10u);
  EXPECT_TRUE(qc.floor_mode);
  EXPECT_DOUBLE_EQ(qc.planned_work, 1000);
  EXPECT_DOUBLE_EQ(bucket.balance(), -1000);  // reservation refunded, floor work debited
}

TEST(Planner, IncompleteProbeSettlesMeasuredWork) {
  auto pl = two_groups();
  const ProbeMenu menu = one_probe(pl);
  WaterFillingAllocator wf;
  TokenBucket bucket;
  PlanSnapshot s = snap_for(pl, 1e9);
  s.B = 1e6; s.D = 1e4; s.reservation = pl.model_nominal_work();
  SnapshotStore store(s);
  Planner p(menu, wf, bucket, store, 10);
  QueryCtx qc;
  p.on_arrival(qc, 0);
  p.on_incomplete(qc, 700, 0);
  EXPECT_DOUBLE_EQ(qc.planned_work, 700);
  EXPECT_FALSE(qc.forced_floor);
  EXPECT_DOUBLE_EQ(bucket.balance(), -700);
}

TEST(Planner, PassthroughIgnoresBucket) {
  auto pl = two_groups();
  const ProbeMenu menu = one_probe(pl);
  PassthroughAllocator pa;
  TokenBucket bucket;
  SnapshotStore store(snap_for(pl, 1.0));
  Planner p(menu, pa, bucket, store, 10);
  QueryCtx qc;
  p.on_arrival(qc, 0);
  EXPECT_EQ(p.choose_probe(qc), 1025u);
  EXPECT_EQ(p.on_probe(70.0f, 80, qc, 0), 80u);
  EXPECT_DOUBLE_EQ(bucket.balance(), 0);
}

// Stage 1 (§1A.5). Full probe: work {2000, 3000, 5000}, recall {0.90, 0.95, 0.97} on the ladder {10, 20, 40}.
// Short probe 256: work {500, 1000, 1700}, recall {0.80, 0.90, 0.93}. Water filling expected points (by hand):
//   full:  nominal 5000 -> one cut (40->20) gives 3000 at 0.95; two cuts give the floor 2000 at 0.90
//   short: nominal 1700 at 0.93; one cut gives 1000 at 0.90; two cuts give the floor 500 at 0.80
TEST(ProbeMenu, Choose) {
  ProbeOption full;
  full.probe_len = 1025;
  full.price_list = one_group({2000, 3000, 5000}, {0.90, 0.95, 0.97});
  ProbeOption shorter;
  shorter.probe_len = 256;
  shorter.price_list = one_group({500, 1000, 1700}, {0.80, 0.90, 0.93});
  const ProbeMenu menu({full, shorter});
  const CutKind wf = CutKind::WaterFilling;

  EXPECT_EQ(menu.choose(1e9, wf), 0u);    // both at nominal; the full probe has the higher recall
  EXPECT_EQ(menu.choose(4000, wf), 0u);   // both fit; full at 3000 / 0.95 beats the short probe at 1700 / 0.93
  EXPECT_EQ(menu.choose(2100, wf), 1u);   // full needs both cuts (2000 / 0.90); short fits at 1700 / 0.93
  EXPECT_EQ(menu.choose(1000, wf), 1u);   // full cannot fit (floor 2000); short cuts to 1000 / 0.90
  EXPECT_EQ(menu.choose(100, wf), 1u);    // nothing fits: the shortest probe

  // A tie keeps the longer probe: the same curves under a shorter length give the same expected point.
  ProbeOption tied = full;
  tied.probe_len = 512;
  EXPECT_EQ(ProbeMenu({full, tied}).choose(2100, wf), 0u);
}
