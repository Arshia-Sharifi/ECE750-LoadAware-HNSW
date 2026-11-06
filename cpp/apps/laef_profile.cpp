// LA-ef: profile of one dataset at C threads (IMPLEMENTATION_PLAN.md §12 P6.2, §11.4).
// Writes profile/profile_C{C}.json with two measurements, both stock Ada-ef (full probe, passthrough planner):
//   idle        1 thread, sequential queries     -> service-time percentiles (T_SLO base), u, mean ndis
//   saturation  C threads, closed loop           -> mu_sat(C) (queries/s), u_sat(C), mean service time
// The Ada-ef table is tables/L{full}/ada_ef_adapter.bin (laef_calibrate, full probe length).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "laef/ann_backend.h"
#include "laef/config.h"
#include "laef/dataset_io.h"
#include "laef/types.h"

namespace fs = std::filesystem;

namespace {

struct Sample {
  double service_s = 0;
  uint32_t ndis = 0;
  bool probe_complete = true;
};

// One query with Ada-ef's own ef (passthrough planner) on the full probe. Times the search only.
Sample search_once(const laef::AnnBackend& backend, uint32_t probe_len, const float* q, uint32_t k,
                   std::vector<uint32_t>& ids) {
  laef::SearchStats st;
  const int64_t t0 = laef::now_ns();
  backend.search_planned(q, k, probe_len, [](float, uint32_t ef_pref, uint32_t) { return ef_pref; }, ids.data(), &st);
  const int64_t t1 = laef::now_ns();
  return {(t1 - t0) * 1e-9, st.ndis, st.probe_complete};
}

// numpy-style linear percentile of a sorted, non-empty vector.
double percentile(const std::vector<double>& sorted, double q) {
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(sorted.size() - 1, lo + 1);
  return sorted[lo] + (pos - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

std::string dataset_name(const std::string& dir) {
  fs::path p = fs::path(dir).lexically_normal();
  if (p.filename().empty()) p = p.parent_path();
  return p.filename().string();
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"LA-ef profile: idle and saturation service times at C threads"};
  std::string dataset_dir;
  uint32_t threads = 1, k = 10, M = 16, efc = 500, ef_upper_bound = 5000;
  float quantile_step = 1e-3f;
  size_t statics_length = 0;
  uint64_t n_idle = 2000, warmup = 100;
  double sat_seconds = 5.0, target_recall = 0.95;
  app.add_option("--dataset-dir", dataset_dir, "data/artifacts/<id>")->required();
  app.add_option("-j,--threads", threads, "C: the worker count being profiled");
  app.add_option("-n,--n,--n-idle", n_idle, "sequential queries in the idle run");
  app.add_option("--sat-seconds", sat_seconds, "duration of the saturation run");
  app.add_option("-k,--k", k, "neighbours per query (must match calibration)");
  app.add_option("--target-recall", target_recall, "Ada-ef declared recall");
  app.add_option("--quantile-step", quantile_step, "Ada-ef quantile step (paper default: 1e-3)");
  app.add_option("--statics-length", statics_length, "full probe length (0 = paper default)");
  app.add_option("--ef-upper-bound", ef_upper_bound, "Ada-ef preferred-ef cap");
  app.add_option("-M,--M", M, "HNSW M (must match the built index)");
  app.add_option("--efc,--ef-construction", efc, "HNSW ef_construction (must match the built index)");
  CLI11_PARSE(app, argc, argv);

  try {
    if (threads == 0) throw std::invalid_argument("--threads must be at least 1");
    const std::string dir = dataset_dir;
    const laef::DatasetMeta dm = laef::read_dataset_meta(dir);
    const laef::Dataset ds = laef::load_queries(dir);
    const std::vector<uint32_t> rows = laef::eval_pool(dir, dm.n_query);

    const uint32_t full_L = static_cast<uint32_t>(statics_length ? statics_length : laef::default_statics_length(M));
    const std::string index_path = dir + "/index/hnsw_M" + std::to_string(M) + "_efc" + std::to_string(efc) + ".bin";
    const std::string adapter_path = dir + "/tables/L" + std::to_string(full_L) + "/ada_ef_adapter.bin";
    if (!fs::exists(index_path) || !fs::exists(adapter_path))
      throw std::runtime_error("missing index or the full-probe Ada-ef table " + adapter_path +
                               "; run laef_build_index and laef_calibrate first");

    laef::BackendConfig bc;
    bc.index_path = index_path;
    bc.metric = dm.metric;
    bc.estimator_path = dir + "/stats/estimator.bin";
    bc.adapter_path = adapter_path;
    bc.dim = dm.dim;
    bc.M = M;
    bc.k = k;
    bc.target_recall = static_cast<float>(target_recall);
    bc.quantile_step = quantile_step;
    bc.statics_length = full_L;
    auto backend = laef::AnnBackend::open(bc);

    // Idle: one thread, one query after another (after a short warm-up that is not recorded).
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<size_t> pick(0, rows.size() - 1);
    std::vector<uint32_t> ids(k);
    for (uint64_t i = 0; i < warmup; ++i) {
      const Sample x = search_once(*backend, full_L, ds.queries.data() + rows[pick(rng)] * dm.dim, k, ids);
      if (!x.probe_complete) throw std::runtime_error("Ada-ef probe incomplete during profile warm-up");
    }
    std::vector<double> idle_s;
    idle_s.reserve(n_idle);
    double idle_sum_s = 0, idle_sum_ndis = 0;
    std::cout << "idle run: " << n_idle << " queries on 1 thread" << std::endl;
    for (uint64_t i = 0; i < n_idle; ++i) {
      const Sample x = search_once(*backend, full_L, ds.queries.data() + rows[pick(rng)] * dm.dim, k, ids);
      if (!x.probe_complete) throw std::runtime_error("Ada-ef probe incomplete during idle profile");
      idle_s.push_back(x.service_s);
      idle_sum_s += x.service_s;
      idle_sum_ndis += x.ndis;
    }
    std::sort(idle_s.begin(), idle_s.end());

    // Saturation: C threads, each issuing the next query as soon as the previous one finishes.
    std::cout << "saturation run: " << threads << " threads for " << sat_seconds << " s" << std::endl;
    std::atomic<bool> stop{false};
    std::atomic<bool> profile_failed{false};
    std::mutex mu;
    std::vector<double> sat_s;
    double sat_sum_s = 0, sat_sum_ndis = 0;
    uint64_t sat_done = 0;
    auto closed_loop = [&](uint64_t seed) {
      std::mt19937_64 r(seed);
      std::uniform_int_distribution<size_t> p(0, rows.size() - 1);
      std::vector<uint32_t> out(k);
      std::vector<double> local_s;
      double local_sum_s = 0, local_sum_ndis = 0;
      uint64_t local_done = 0;
      while (!stop.load() && !profile_failed.load()) {
        const Sample x = search_once(*backend, full_L, ds.queries.data() + rows[p(r)] * dm.dim, k, out);
        if (!x.probe_complete) {
          profile_failed = true;
          stop = true;
          break;
        }
        local_s.push_back(x.service_s);
        local_sum_s += x.service_s;
        local_sum_ndis += x.ndis;
        ++local_done;
      }
      std::lock_guard<std::mutex> lk(mu);
      sat_s.insert(sat_s.end(), local_s.begin(), local_s.end());
      sat_sum_s += local_sum_s;
      sat_sum_ndis += local_sum_ndis;
      sat_done += local_done;
    };
    const auto t_start = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < threads; ++t) workers.emplace_back(closed_loop, 1000 + t);
    std::this_thread::sleep_for(std::chrono::duration<double>(sat_seconds));
    stop = true;
    for (auto& w : workers) w.join();
    if (profile_failed) throw std::runtime_error("Ada-ef probe incomplete during saturation profile");
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    std::sort(sat_s.begin(), sat_s.end());

    nlohmann::json idle = {{"mean_s", idle_sum_s / n_idle},
                           {"p50_s", percentile(idle_s, 0.50)},
                           {"p95_s", percentile(idle_s, 0.95)},
                           {"p99_s", percentile(idle_s, 0.99)},
                           {"u_s_per_dc", idle_sum_s / idle_sum_ndis},
                           {"mean_ndis", idle_sum_ndis / n_idle}};
    nlohmann::json sat = {{"mu_sat_qps", sat_done / elapsed},
                          {"u_s_per_dc", sat_sum_s / sat_sum_ndis},
                          {"mean_service_s", sat_sum_s / static_cast<double>(sat_s.size())},
                          {"s_p95_s", percentile(sat_s, 0.95)},
                          {"completed", sat_done},
                          {"seconds", elapsed}};
    nlohmann::json j = {{"dataset", dataset_name(dir)}, {"threads", threads}, {"k", k}, {"M", M},
                        {"idle", idle}, {"saturation", sat}};

    fs::create_directories(dir + "/profile");
    const std::string out_path = dir + "/profile/profile_C" + std::to_string(threads) + ".json";
    std::ofstream(out_path) << j.dump(2) << "\n";
    std::cout << "idle S_p95 " << idle["p95_s"].get<double>() * 1e3 << " ms, mu_sat(" << threads << ") "
              << sat["mu_sat_qps"].get<double>() << " qps\nwrote " << out_path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
