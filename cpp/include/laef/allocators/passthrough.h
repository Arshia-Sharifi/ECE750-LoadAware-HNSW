#pragma once
// LA-ef: Ada-ef baseline (ef = ef_pref) and static-ef baseline. No bucket. IMPLEMENTATION_PLAN.md §10.6.
#include <algorithm>

#include "laef/allocators/allocator.h"

namespace laef {

class PassthroughAllocator final : public Allocator {
 public:
  explicit PassthroughAllocator(uint32_t static_ef = 0) : static_ef_(static_ef) {}
  AllocResult plan(const AllocRequest& rq, const PlanSnapshot&, const PriceList& pl) const override {
    AllocResult r;
    r.ef = std::max(rq.k, static_ef_ ? static_ef_ : rq.ef_pref);
    r.planned_work = pl.work(rq.group, r.ef);
    return r;
  }
  bool uses_bucket() const override { return false; }
  std::string name() const override { return static_ef_ ? "static_ef" : "passthrough"; }

 private:
  uint32_t static_ef_;
};

}  // namespace laef
