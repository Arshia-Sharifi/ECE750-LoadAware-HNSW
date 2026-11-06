#pragma once
// LA-ef: allocator factory (IMPLEMENTATION_PLAN.md §10.6).
// "static_ef" is a passthrough allocator pinned to the price list's static_ef_for_target; the engine
// then serves it with AnnBackend::search_fixed (EngineConfig::fixed_ef).
#include <memory>
#include <stdexcept>

#include "laef/allocators/passthrough.h"
#include "laef/allocators/uniform_cut.h"
#include "laef/allocators/water_filling.h"
#include "laef/config.h"

namespace laef {

inline std::unique_ptr<Allocator> make_allocator(const AllocatorCfg& c, const PriceList& pl) {
  if (c.type == "water_filling") return std::make_unique<WaterFillingAllocator>();
  if (c.type == "uniform_cut") {
    if (c.uniform_mode != "work" && c.uniform_mode != "ef") throw std::invalid_argument("allocator.uniform_mode: work | ef");
    return std::make_unique<UniformCutAllocator>(c.uniform_mode == "ef" ? UniformCutAllocator::Mode::Ef
                                                                         : UniformCutAllocator::Mode::Work);
  }
  if (c.type == "passthrough") return std::make_unique<PassthroughAllocator>();
  if (c.type == "static_ef") return std::make_unique<PassthroughAllocator>(pl.static_ef_for_target);
  throw std::invalid_argument("unknown allocator.type: " + c.type);
}

// The cut policy the snapshot models when it chooses the probe (plan §1A.5). Passthrough means no cut and no bucket.
inline CutKind cut_kind(const AllocatorCfg& c) {
  if (c.type == "water_filling") return CutKind::WaterFilling;
  if (c.type == "uniform_cut") return c.uniform_mode == "ef" ? CutKind::UniformEf : CutKind::UniformWork;
  return CutKind::Passthrough;
}

}  // namespace laef
