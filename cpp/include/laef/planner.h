#pragma once
// LA-ef: Step E glue (IMPLEMENTATION_PLAN.md §1.2, §1A.5-§1A.7, §10.7, §10.8).
//   on_arrival    : reserve the snapshot's expected work and retain the amount actually charged (always admitted; no drop path)
//   choose_probe  : stage 1, before the search: the probe length (a menu option) from the current snapshot
//   on_probe      : stage 2, from the AnnBackend hook when the probe completes: refund the reservation, floor mode or the
//                   allocator on that probe's price list, bucket check, debit
//   on_incomplete : settle a search whose probe never completed, at its measured distance work
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "laef/allocators/allocator.h"
#include "laef/plan_snapshot.h"
#include "laef/price_list.h"
#include "laef/probe_menu.h"
#include "laef/token_bucket.h"

namespace laef {

struct QueryCtx {
  double reservation = 0;      // charged amount of the arrival reservation [dc]
  size_t probe_idx = 0;        // menu option the query runs with (stage 1)
  int group = -1;
  double planned_work = 0;
  bool forced_floor = false;
  bool floor_mode = false;
};

class Planner {
 public:
  Planner(const ProbeMenu& menu, const Allocator& alloc, TokenBucket& bucket, const SnapshotStore& snaps,
          uint32_t k, bool floor_mode_enabled = true)
      : menu_(menu), alloc_(alloc), bucket_(bucket), snaps_(snaps), k_(k), floor_enabled_(floor_mode_enabled) {}

  void on_arrival(QueryCtx& qc, int64_t now_ns) {
    if (!alloc_.uses_bucket()) return;
    auto s = snaps_.load();
    qc.reservation = bucket_.reserve(s->reservation, now_ns, s->B, s->D);
  }

  // Stage 1. Returns the probe length the search must use. Policies without a bucket always use the full probe.
  uint32_t choose_probe(QueryCtx& qc) const {
    qc.probe_idx = alloc_.uses_bucket() ? std::min(snaps_.load()->probe_idx, menu_.size() - 1) : 0;
    return menu_.at(qc.probe_idx).probe_len;
  }

  // Stage 2. Called by the backend hook with the score and Ada-ef's ef of the probe that was just completed.
  uint32_t on_probe(float score, uint32_t ef_pref, QueryCtx& qc, int64_t now_ns) {
    auto s = snaps_.load();
    const PriceList& pl = menu_.at(qc.probe_idx).price_list;  // the table of the probe this query actually ran
    const int g = pl.group_of(score);
    qc.group = g;
    ef_pref = std::max(ef_pref, k_);
    if (!alloc_.uses_bucket()) {  // Ada-ef / static-ef baselines
      AllocResult r = alloc_.plan({g, ef_pref, k_, 0.0}, *s, pl);
      qc.planned_work = r.planned_work;
      return r.ef;
    }
    // The allocators read the water level and uniform fraction of this probe (plan §1A.6).
    PlanSnapshot here = *s;
    here.water_level = s->water_level_by_L.at(qc.probe_idx);
    here.uniform_fraction = s->uniform_fraction_by_L.at(qc.probe_idx);

    AllocResult r;
    const bool floor = floor_enabled_ && s->floor_mode;
    bucket_.settle(qc.reservation, now_ns, s->B, s->D, [&](double affordable) {
      if (floor) {  // §10.7: everything at the probe's floor rung, never dropped
        r.ef = pl.floor_ef(g);
        r.planned_work = pl.floor_work(g);
        r.forced_floor = r.planned_work > affordable;
      } else {
        r = alloc_.plan({g, ef_pref, k_, affordable}, here, pl);
      }
      return r.planned_work;
    });
    qc.planned_work = r.planned_work;
    qc.forced_floor = r.forced_floor;
    qc.floor_mode = floor;
    return std::max(k_, r.ef);
  }

  // A probe can finish without collecting statics_length samples on a small or sparse graph.
  // The search has already done its actual work, so settle that measured cost instead of leaking
  // the arrival reservation or pretending that a planner decision was made.
  void on_incomplete(QueryCtx& qc, uint32_t actual_ndis, int64_t now_ns) {
    const double actual_work = static_cast<double>(actual_ndis);
    if (!alloc_.uses_bucket()) {
      qc.planned_work = actual_work;
      return;
    }
    auto s = snaps_.load();
    const bool floor = floor_enabled_ && s->floor_mode;
    bucket_.settle(qc.reservation, now_ns, s->B, s->D, [&](double affordable) {
      qc.forced_floor = actual_work > affordable;
      qc.floor_mode = floor;
      return actual_work;
    });
    qc.planned_work = actual_work;
  }

 private:
  const ProbeMenu& menu_;
  const Allocator& alloc_;
  TokenBucket& bucket_;
  const SnapshotStore& snaps_;
  uint32_t k_;
  bool floor_enabled_;
};

}  // namespace laef
