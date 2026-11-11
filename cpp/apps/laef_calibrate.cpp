// LA-ef: Step A for every probe length (IMPLEMENTATION_PLAN.md §9, §1A.4).
//   1. drops the calibration queries that the Ada-ef table cannot use (finding 5) -> calibration/samplings_valid.bin
//   2. for each probe length L: builds the Ada-ef table (tables/L{L}/ada_ef_adapter.bin), then runs the preference pass
//      and the ladder sweep at L and writes tables/L{L}/price_list.json and calibration_sweep.csv
//   3. drops the probe lengths that are dominated by a longer one, and writes tables/probe_menu.json
// The full probe length must succeed; a shorter one that fails is logged and dropped.
// The estimator and the filtered samplings live in the dataset folder. The menu and the per-L tables go to
// --tables-dir, so each ladder ratio (E4) can keep its own copy.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <CLI/CLI.hpp>

#include "laef/ann_backend.h"
#include "laef/calibration.h"
#include "laef/config.h"
#include "laef/dataset_io.h"
#include "laef/price_list_io.h"
#include "laef/probe_menu.h"

namespace fs = std::filesystem;

namespace {

std::string dataset_name(const std::string& dir) {
  fs::path p = fs::path(dir).lexically_normal();
  if (p.filename().empty()) p = p.parent_path();  // trailing slash
  return p.filename().string();
}

// "1025,512,256,128" -> {1025, 512, 256, 128}
std::vector<uint32_t> parse_lengths(const std::string& s) {
  std::vector<uint32_t> out;
  std::stringstream ss(s);
  for (std::string tok; std::getline(ss, tok, ',');) {
    if (tok.empty()) continue;
    size_t used = 0;
    const unsigned long v = std::stoul(tok, &used);
    if (used != tok.size()) throw std::invalid_argument("--probe-lengths: not a number: " + tok);
    out.push_back(static_cast<uint32_t>(v));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"LA-ef Step A: Ada-ef tables, ladder sweeps and the probe menu"};
  const laef::RunConfig defaults;  // defaults come from config.h, which mirrors configs/defaults.yaml

  std::string dataset_dir, tables_dir, probe_lengths_arg;
  uint32_t k = defaults.k, M = defaults.hnsw.M, efc = defaults.hnsw.ef_construction;
  uint32_t ef_upper_bound = defaults.adaef.ef_upper_bound, threads = 0;
  double target_recall = defaults.target_recall, ladder_ratio = defaults.calibration.ladder_ratio;
  double quantile_step = defaults.adaef.quantile_step;
  size_t min_group = defaults.calibration.min_group_size, max_groups = defaults.calibration.max_groups;
  size_t statics_length = defaults.adaef.statics_length;

  app.add_option("--dataset-dir", dataset_dir, "data/artifacts/<id>")->required();
  app.add_option("--tables-dir", tables_dir,
                 "where probe_menu.json and the L{L}/ folders are written (default: <dataset-dir>/tables)");
  app.add_option("--probe-lengths", probe_lengths_arg,
                 "probe menu, comma-separated and descending (default: adaef.probe_lengths = 1025,512,256,128)");
  app.add_option("-k,--k", k, "neighbours per query");
  app.add_option("--target-recall", target_recall, "Ada-ef declared recall");
  app.add_option("--quantile-step", quantile_step, "Ada-ef quantile step (paper default: 1e-3)");
  app.add_option("-M,--M", M, "HNSW M (must match the built index)");
  app.add_option("--efc,--ef-construction", efc, "HNSW ef_construction (must match the built index)");
  app.add_option("-j,--threads", threads, "sweep threads (0 = all cores)");
  app.add_option("--ladder-ratio", ladder_ratio, "ratio between ladder rungs");
  app.add_option("--ef-upper-bound", ef_upper_bound, "largest ef the Ada-ef table may return");
  app.add_option("--min-group-size", min_group, "minimum calibration queries per difficulty group");
  app.add_option("--max-groups", max_groups, "maximum difficulty groups");
  app.add_option("--statics-length", statics_length, "full probe length (0 = 1 + 2M + (2M-1)*2M, i.e. 1025 for M = 16)");
  CLI11_PARSE(app, argc, argv);

  try {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    const std::string dir = dataset_dir;
    const std::string tables = tables_dir.empty() ? dir + "/tables" : tables_dir;
    const laef::DatasetMeta dm = laef::read_dataset_meta(dir);

    const uint32_t full_L = static_cast<uint32_t>(statics_length ? statics_length : laef::default_statics_length(M));
    const std::vector<uint32_t> lengths =
        probe_lengths_arg.empty() ? defaults.adaef.probe_lengths : parse_lengths(probe_lengths_arg);
    if (lengths.empty()) throw std::invalid_argument("the probe menu is empty");
    for (size_t i = 0; i < lengths.size(); ++i) {
      if (lengths[i] < 64) throw std::invalid_argument("probe lengths must be at least 64");
      if (i > 0 && lengths[i] >= lengths[i - 1]) throw std::invalid_argument("probe lengths must be strictly descending");
    }
    if (lengths.front() != full_L)
      throw std::invalid_argument("the largest probe length must be the full probe length " + std::to_string(full_L));

    const std::string index_path = dir + "/index/hnsw_M" + std::to_string(M) + "_efc" + std::to_string(efc) + ".bin";
    const std::string est_path = dir + "/stats/estimator.bin";
    const std::string samplings_path = dir + "/calibration/samplings.bin";
    const std::string valid_path = dir + "/calibration/samplings_valid.bin";
    if (!fs::exists(index_path)) throw std::runtime_error("missing " + index_path + " (run laef_build_index first)");
    if (!fs::exists(samplings_path))
      throw std::runtime_error("missing " + samplings_path + " (run gpu_preprocessing with --steps calib)");
    fs::create_directories(dir + "/calibration");
    fs::create_directories(dir + "/stats");
    fs::create_directories(tables);

    laef::BackendConfig bc;
    bc.index_path = index_path;
    bc.metric = dm.metric;
    bc.estimator_path = est_path;
    bc.dim = dm.dim;
    bc.M = M;
    bc.k = k;
    bc.target_recall = static_cast<float>(target_recall);
    bc.quantile_step = static_cast<float>(quantile_step);
    bc.statics_length = full_L;

    if (!fs::exists(est_path)) {
      std::cout << "stats/estimator.bin missing: computing it on the CPU" << std::endl;
      const laef::Dataset ds = laef::load_dataset(dir);
      laef::compute_estimator_cpu(ds.vectors.data(), ds.n, ds.dim, dm.metric, est_path);
    }

    // Finding 5: the Ada-ef table cannot use a calibration query whose recall is zero at ef = k or ceil(1.5k).
    // The usable rows do not depend on the probe length, so they are chosen once and written for every table.
    // Always rebuilt: calibration keeps no metadata, so a stale file could not be detected.
    const laef::Samplings all = laef::read_samplings(samplings_path);
    const std::vector<uint32_t> usable = laef::usable_calibration_rows(bc, samplings_path);
    if (usable.empty())
      throw std::runtime_error("no calibration query has non-zero recall at ef = k; check the index and samplings");
    const laef::Samplings valid = laef::select_rows(all, usable);
    laef::write_samplings(valid_path, valid);
    const size_t n_dropped = all.n - valid.n;
    if (n_dropped > 0)
      std::cout << "dropped " << n_dropped << " of " << all.n << " calibration queries with zero recall (finding 5)\n";

    // ---- step 1: one Ada-ef table per probe length (always rebuilt) ----
    std::vector<laef::DroppedProbe> dropped;
    std::vector<uint32_t> built;
    for (uint32_t L : lengths) {
      const std::string ldir = tables + "/L" + std::to_string(L);
      fs::create_directories(ldir);
      laef::BackendConfig bl = bc;
      bl.statics_length = L;
      std::cout << "Ada-ef table for probe length " << L << " from " << valid_path << std::endl;
      try {
        laef::build_ef_adapter(bl, valid_path, ef_upper_bound, ldir + "/ada_ef_adapter.bin");
        built.push_back(L);
      } catch (const std::exception& e) {
        if (L == full_L) throw;
        std::cout << "  dropped: " << e.what() << "\n";
        dropped.push_back({L, std::string("Ada-ef table failed: ") + e.what()});
      }
    }

    // ---- step 2: the sweep and the price list for each probe length, with every table open in one backend ----
    for (uint32_t L : built) bc.probes.push_back({L, tables + "/L" + std::to_string(L) + "/ada_ef_adapter.bin"});
    auto backend = laef::AnnBackend::open(bc);

    std::vector<laef::ProbeOption> options;  // descending probe length, as built
    for (uint32_t L : built) {
      const std::string ldir = tables + "/L" + std::to_string(L);
      try {
        laef::CalibrationSettings cs;
        cs.k = k;
        cs.target_recall = bc.target_recall;
        cs.ladder_ratio = ladder_ratio;
        cs.ef_upper_bound = ef_upper_bound;
        cs.min_group_size = min_group;
        cs.max_groups = max_groups;
        cs.threads = threads;

        std::cout << "probe length " << L << ": ladder sweep over " << valid.n << " calibration queries on " << threads
                  << " threads" << std::endl;
        const laef::SweepResult sweep = laef::run_sweep(*backend, valid, cs, L);
        const laef::CalibrationResult res = laef::build_price_list(sweep.rows, valid.n, sweep.ladder, cs);
        laef::write_sweep_csv(ldir + "/calibration_sweep.csv", sweep.rows);

        laef::PriceListMeta pm;
        pm.dataset = dataset_name(dir);
        pm.metric = dm.metric;
        pm.target_recall = target_recall;
        pm.quantile_step = quantile_step;
        pm.M = M;
        pm.statics_length = L;
        pm.ef_upper_bound = ef_upper_bound;
        pm.ladder = sweep.ladder;
        pm.n_calib = valid.n;
        pm.ladder_ratio = ladder_ratio;
        pm.n_dropped = n_dropped;
        pm.calib_mean_recall_at_pref = res.calib_mean_recall_at_pref;
        laef::save_price_list(ldir + "/price_list.json", res.price_list, pm);

        laef::ProbeOption o;
        o.probe_len = L;
        o.price_list = res.price_list;
        o.adapter_path = ldir + "/ada_ef_adapter.bin";
        o.price_list_path = ldir + "/price_list.json";
        o.calib_mean_recall_at_pref = res.calib_mean_recall_at_pref;
        double sum_probe = 0, sum_ef = 0;
        size_t n_pref = 0;
        for (const auto& r : sweep.rows)
          if (r.is_pref) {
            sum_probe += r.probe_ndis;
            sum_ef += r.ef_pref;
            ++n_pref;
          }
        o.mean_probe_ndis = n_pref ? sum_probe / static_cast<double>(n_pref) : 0.0;
        o.ef_pref_mean = n_pref ? sum_ef / static_cast<double>(n_pref) : 0.0;

        std::cout << "  recall at Ada-ef's ef: " << std::fixed << std::setprecision(4) << res.calib_mean_recall_at_pref
                  << " (target " << target_recall << "), mean ef_pref " << std::setprecision(1) << o.ef_pref_mean
                  << ", probe " << std::setprecision(1) << o.mean_probe_ndis << " dc, model work "
                  << o.price_list.model_nominal_work() << " dc, floor " << o.price_list.model_floor_work() << " dc\n";
        options.push_back(std::move(o));
      } catch (const std::exception& e) {
        if (L == full_L) throw;
        std::cout << "  dropped: " << e.what() << "\n";
        dropped.push_back({L, std::string("calibration failed: ") + e.what()});
      }
    }

    // ---- step 3: drop dominated probe lengths, then write the menu (plan §1A.4 step 5) ----
    // A shorter probe is kept only if its model work and its model floor are both strictly below those of the last
    // kept (longer) probe. Otherwise it saves nothing that the longer probe does not already offer.
    std::vector<laef::ProbeOption> kept;
    for (auto& o : options) {
      if (kept.empty()) {
        kept.push_back(std::move(o));
        continue;
      }
      const double nom = o.price_list.model_nominal_work(), flo = o.price_list.model_floor_work();
      const laef::ProbeOption& last = kept.back();
      if (nom < last.price_list.model_nominal_work() && flo < last.price_list.model_floor_work()) {
        kept.push_back(std::move(o));
      } else {
        std::ostringstream why;
        why << "dominated: model work " << std::fixed << std::setprecision(1) << nom << " dc and floor " << flo
            << " dc are not both below L=" << last.probe_len << " (" << last.price_list.model_nominal_work() << " / "
            << last.price_list.model_floor_work() << ")";
        dropped.push_back({o.probe_len, why.str()});
      }
    }

    laef::ProbeMenuMeta mm;
    mm.dataset = dataset_name(dir);
    mm.k = k;
    mm.M = M;
    mm.target_recall = target_recall;
    mm.full_probe_len = full_L;
    mm.dropped = dropped;
    const laef::ProbeMenu menu(kept);
    laef::save_probe_menu(tables + "/probe_menu.json", menu, mm);

    std::cout << "probe menu: " << menu.size() << " probe length(s), written to " << tables << "/probe_menu.json\n";
    for (const auto& o : menu.options())
      std::cout << "  L=" << std::setw(5) << o.probe_len << "  model work " << std::setw(8) << std::fixed
                << std::setprecision(1) << o.price_list.model_nominal_work() << " dc  floor "
                << o.price_list.model_floor_work() << " dc  recall " << std::setprecision(4)
                << o.price_list.model_nominal_recall() << "\n";
    for (const auto& d : dropped) std::cout << "  dropped L=" << d.probe_len << ": " << d.reason << "\n";
    std::cout << "static_ef_for_target (full probe): " << menu.full().price_list.static_ef_for_target << "\n";
    if (std::fabs(menu.full().calib_mean_recall_at_pref - target_recall) > 0.02)
      std::cout << "warning: calibration recall is outside target +/- 0.02 (plan §9 acceptance)\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
