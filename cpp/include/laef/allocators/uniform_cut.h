#pragma once
// LA-ef: naive equal cut (IMPLEMENTATION_PLAN.md §10.6, Main Flow §5 "Naive comparison allocator").
// Every query gets the same fraction f = snapshot.uniform_fraction of its preferred work (mode Work)
// or of its preferred ef (mode Ef); then the shared bucket check.
#include <algorithm>
#include <cmath>

#include "laef/allocators/allocator.h"

namespace laef {

class UniformCutAllocator final : public Allocator {
 public:
  enum class Mode { Work, Ef };
  explicit UniformCutAllocator(Mode m = Mode::Work) : mode_(m) {}

  AllocResult plan(const AllocRequest& rq, const PlanSnapshot& s, const PriceList& pl) const override {
    const int g = rq.group;
    const double f = std::clamp(s.uniform_fraction, 0.0, 1.0);
    double ef = rq.ef_pref;
    if (f < 1.0) {
      const auto& L = pl.rungs(g);
      auto fits = [&](double e) {
        return mode_ == Mode::Work ? pl.work(g, e) <= f * pl.work(g, rq.ef_pref) + 1e-9
                                   : e <= std::max<double>(rq.k, f * rq.ef_pref) + 1e-9;
      };
      if (!fits(ef)) {
        double pick = L.front();  // nothing fits -> group floor
        for (size_t i = L.size(); i-- > 0;)
          if (L[i] < rq.ef_pref && fits(L[i])) { pick = L[i]; break; }
        ef = pick;
      }
    }
    return bucket_check(pl, g, ef, rq.k, rq.affordable);
  }
  std::string name() const override { return mode_ == Mode::Work ? "uniform_cut" : "uniform_cut_ef"; }

 private:
  Mode mode_;
};

}  // namespace laef
