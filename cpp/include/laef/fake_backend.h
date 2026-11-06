#pragma once
// LA-ef: deterministic stand-in for the Ada-ef backend (IMPLEMENTATION_PLAN.md §8.1, §13.3).
// Used by engine tests so that load behaviour can be checked without an index.
#include <cstdint>
#include <memory>

#include "laef/ann_backend.h"

namespace laef {

// Per query: the probe costs round(100 * L / 1025) dc for probe length L (100 dc at the full probe), and the planner
// sees a score in [0, 100) derived from the query vector. The planner's ef is honoured, and the total work is probe + ef
// dc. Service time is that work times u_ns, spent busy-waiting. Static searches (search_fixed) report probe_complete = false.
std::unique_ptr<AnnBackend> make_fake_backend(uint32_t dim, double u_ns);

}  // namespace laef
