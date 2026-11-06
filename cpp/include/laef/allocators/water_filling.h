#pragma once
// LA-ef: price-list (water-filling) cut (IMPLEMENTATION_PLAN.md §10.6, Main Flow §5 step 5).
// From ef_pref, take every cut whose loss ratio <= snapshot.water_level (cheap cuts first; the
// shared water level makes the cut order consistent across groups), then the bucket check.
#include "laef/allocators/allocator.h"

namespace laef {

class WaterFillingAllocator final : public Allocator {
 public:
  AllocResult plan(const AllocRequest& rq, const PlanSnapshot& s, const PriceList& pl) const override {
    double ef = rq.ef_pref;
    if (s.water_level > 0) {
      for (const auto& st : pl.steps_from(rq.group, ef)) {  // descending ef, non-decreasing loss
        if (st.loss <= s.water_level) ef = st.to; else break;
      }
    }
    return bucket_check(pl, rq.group, ef, rq.k, rq.affordable);
  }
  std::string name() const override { return "water_filling"; }
};

}  // namespace laef
