#pragma once
// LA-ef: allocator interface + the bucket check shared by all cutting allocators (IMPLEMENTATION_PLAN.md §10.6).
// Allocators choose ef on ladder rungs (plus ef_pref itself as the top rung) so uniform_cut and
// water_filling are quantised identically (fair comparison; ladder density is ablated in E4).
#include <cstdint>
#include <string>

#include "laef/plan_snapshot.h"
#include "laef/price_list.h"

namespace laef {

struct AllocRequest { int group = 0; uint32_t ef_pref = 0; uint32_t k = 10; double affordable = 0; };
struct AllocResult  { uint32_t ef = 0; double planned_work = 0; bool forced_floor = false; };

class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual AllocResult plan(const AllocRequest&, const PlanSnapshot&, const PriceList&) const = 0;
  virtual bool uses_bucket() const { return true; }
  virtual std::string name() const = 0;
};

// Step down one rung at a time while the planned work exceeds `affordable` (Main Flow §5 step 6).
// Never below the group's floor rung (>= k). If even the floor is unaffordable: forced_floor.
inline AllocResult bucket_check(const PriceList& pl, int g, double ef, uint32_t k, double affordable) {
  const auto& L = pl.rungs(g);
  while (pl.work(g, ef) > affordable) {
    double next = -1;
    for (size_t i = L.size(); i-- > 0;)
      if (L[i] < ef) { next = L[i]; break; }
    if (next < 0) break;
    ef = next;
  }
  AllocResult r;
  r.ef = std::max<uint32_t>(k, static_cast<uint32_t>(ef));
  r.planned_work = pl.work(g, ef);
  r.forced_floor = r.planned_work > affordable;
  return r;
}

}  // namespace laef
