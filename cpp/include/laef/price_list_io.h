#pragma once
// LA-ef: price_list.json and probe_menu.json readers and writers (IMPLEMENTATION_PLAN.md §9.4, §1A.4).
// Kept apart from price_list.h so the algorithms stay header-only and free of JSON.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "laef/price_list.h"
#include "laef/probe_menu.h"

namespace laef {

// Metadata stored alongside the curves. The algorithms do not use it.
struct PriceListMeta {
  std::string dataset, metric = "cd";
  double target_recall = 0.95;
  double quantile_step = 1e-3;
  uint32_t M = 16;
  size_t statics_length = 0;
  uint32_t ef_upper_bound = 5000;
  std::vector<uint32_t> ladder;          // global ef ladder of the sweep
  size_t n_calib = 0;
  double ladder_ratio = 2.0;             // calibration.ladder_ratio used for the sweep
  size_t n_dropped = 0;                  // calibration queries removed for zero recall at ef = k or ceil(1.5k)
  double calib_mean_recall_at_pref = 0; // mean recall at Ada-ef's own ef over the calibration queries
};

// Writes the curves and the derived cut list. The steps come from pl.steps_sorted(), so they always
// match the curves.
void save_price_list(const std::string& path, const PriceList& pl, const PriceListMeta& meta);

// Reads the curves, rebuilds the cut list with finalize(). meta is optional.
PriceList load_price_list(const std::string& path, PriceListMeta* meta = nullptr);

// A probe length that the calibration dropped, and why (probe_menu.json "dropped").
struct DroppedProbe {
  uint32_t probe_len = 0;
  std::string reason;
};

struct ProbeMenuMeta {
  std::string dataset;
  uint32_t k = 10, M = 16;
  double target_recall = 0.95;
  uint32_t full_probe_len = 0;
  std::vector<DroppedProbe> dropped;
};

// Writes tables/probe_menu.json (§1A.4). Each option is written with its conventional paths relative to the menu's
// folder: L{L}/price_list.json and L{L}/ada_ef_adapter.bin.
void save_probe_menu(const std::string& path, const ProbeMenu& menu, const ProbeMenuMeta& meta);
// Reads probe_menu.json and every option's price list (paths are resolved against the menu's folder). Throws when an
// option's price list was built for another k or another probe length. meta is optional.
ProbeMenu load_probe_menu(const std::string& path, ProbeMenuMeta* meta = nullptr);

}  // namespace laef
