// LA-ef: one experiment run (IMPLEMENTATION_PLAN.md §11, §12 P6.2-P6.3).
// Writes config.json, queries.csv, timeseries.csv and summary.json into --out.
// Needs the outputs of laef_build_index, laef_calibrate and laef_profile (same thread count).
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "laef/ann_backend.h"
#include "laef/config.h"
#include "laef/dataset_io.h"
#include "laef/engine.h"
#include "laef/price_list_io.h"
#include "laef/results_writer.h"
#include "laef/workload.h"

namespace fs = std::filesystem;

namespace {

std::string dataset_name(const std::string& dir) {
  fs::path p = fs::path(dir).lexically_normal();
  if (p.filename().empty()) p = p.parent_path();
  return p.filename().string();
}

void require_file(const std::string& path, const std::string& hint) {
  if (!fs::exists(path)) throw std::runtime_error("missing " + path + "; " + hint);
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"LA-ef bench: one run of one policy on one dataset"};
  std::string dataset_dir, config_path, out_dir, policy, allocator, controller, floor_mode, tables_dir_arg;
  uint32_t threads = 0, repeat = 0;
  double slo_ms = 0, qps = 0, duration_s = 0;
  uint64_t n_queries = 0;
  bool pin_cores = false;

  app.add_option("--dataset-dir", dataset_dir, "data/artifacts/<id>")->required();
  app.add_option("--out", out_dir, "results/<experiment>/<run>")->required();
  app.add_option("--tables-dir", tables_dir_arg,
                 "folder with probe_menu.json and the L{L}/ tables from laef_calibrate (default: <dataset-dir>/tables)");
  app.add_option("--config", config_path, "partial RunConfig JSON, applied on top of the defaults");
  app.add_option("-j,--threads", threads, "C (overrides the config)");
  app.add_option("--policy", policy, "static_ef | ada_ef | laef_uniform | laef_wf (sets allocator, controller, floor mode)");
  app.add_option("--allocator", allocator, "water_filling | uniform_cut | passthrough | static_ef");
  app.add_option("--controller", controller, "p_aimd | fixed_aimd | static_capacity | none");
  app.add_option("--floor-mode", floor_mode, "on | off (ablation; off keeps queries on the allocator)");
  app.add_option("--slo-ms", slo_ms, "T_SLO in ms (required unless slo.ms is set in --config)");
  app.add_option("--qps", qps, "arrival rate lambda [queries/s] (poisson, constant)");
  app.add_option("-n,--n-queries", n_queries, "arrivals for the poisson and constant profiles");
  app.add_option("--duration-s", duration_s, "duration for the poisson and constant profiles");
  app.add_option("--repeat", repeat, "repeat index; added to the workload seed");
  app.add_flag("--pin-cores", pin_cores, "pin generator, controller and workers to cores");
  CLI11_PARSE(app, argc, argv);

  try {
    // ---- configuration: defaults -> --config -> flags ----
    laef::RunConfig cfg = config_path.empty() ? laef::RunConfig{} : laef::load_run_config(config_path);
    if (!policy.empty()) laef::apply_policy(cfg, policy);
    if (!allocator.empty()) cfg.allocator.type = allocator;
    if (!controller.empty()) cfg.controller.type = controller;
    if (floor_mode == "on") cfg.floor_mode.enabled = true;
    else if (floor_mode == "off") cfg.floor_mode.enabled = false;
    else if (!floor_mode.empty()) throw std::invalid_argument("--floor-mode takes on or off");
    if (slo_ms > 0) cfg.slo.ms = slo_ms;
    if (qps > 0) cfg.workload.qps = qps;
    if (n_queries > 0) cfg.workload.n_queries = n_queries;
    if (duration_s > 0) cfg.workload.duration_s = duration_s;
    if (threads > 0) cfg.engine.threads = static_cast<int>(threads);
    if (cfg.engine.threads <= 0) cfg.engine.threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    cfg.engine.pin_cores = cfg.engine.pin_cores || pin_cores;
    cfg.workload.seed += repeat;
    const uint32_t C = static_cast<uint32_t>(cfg.engine.threads);
    const std::string dir = dataset_dir;
    cfg.dataset = dataset_name(dir);
    const laef::DatasetMeta dm = laef::read_dataset_meta(dir);

    // ---- profile: saturation throughput and unit cost at this thread count ----
    const std::string profile_path = dir + "/profile/profile_C" + std::to_string(C) + ".json";
    require_file(profile_path, "run: laef_profile --dataset-dir " + dir + " --threads " + std::to_string(C));
    std::ifstream pin(profile_path);
    const nlohmann::json prof = nlohmann::json::parse(pin);
    if (prof.value("threads", 0u) != C)
      throw std::runtime_error("profile thread count does not match benchmark threads");
    const double u_sat = prof.at("saturation").at("u_s_per_dc").get<double>();
    const double mean_service = prof.at("saturation").at("mean_service_s").get<double>();
    const double saturation_p95 = prof.at("saturation").at("s_p95_s").get<double>();
    if (!(cfg.slo.ms > 0))
      throw std::invalid_argument("T_SLO is required: pass --slo-ms or set slo.ms (the Ada-ef p95 for this load)");
    const double t_slo_s = cfg.slo.ms * 1e-3;

    // ---- probe menu (Step A): one Ada-ef table and one price list per probe length (plan §1A.3 C11) ----
    const std::string tables = tables_dir_arg.empty() ? dir + "/tables" : tables_dir_arg;
    laef::ProbeMenuMeta mm;
    laef::ProbeMenu menu = laef::load_probe_menu(tables + "/probe_menu.json", &mm);
    if (mm.k != cfg.k)
      throw std::runtime_error("probe menu was built for k=" + std::to_string(mm.k) + ", config has k=" +
                               std::to_string(cfg.k));
    if (!cfg.probe.enabled) menu.keep_full_only();  // probe.enabled = false: the v1 ef-only ablation
    std::cout << "probe menu: L = ";
    for (size_t i = 0; i < menu.size(); ++i) std::cout << (i ? ", " : "") << menu.at(i).probe_len;
    std::cout << "\n";

    // ---- dataset and ground truth ----
    const laef::Dataset ds = laef::load_queries(dir);
    const laef::GroundTruth gt = laef::load_ground_truth(dir);
    if (gt.n != dm.n_query) throw std::runtime_error("ground truth rows do not match the test queries");
    if (gt.k < cfg.k) throw std::runtime_error("ground truth stores fewer than k neighbours");

    // ---- Ada-ef backend ----
    const std::string index_path = dir + "/index/hnsw_M" + std::to_string(cfg.hnsw.M) + "_efc" +
                                   std::to_string(cfg.hnsw.ef_construction) + ".bin";
    require_file(index_path, "run: laef_build_index --dataset-dir " + dir);
    require_file(dir + "/stats/estimator.bin", "run: laef_calibrate --dataset-dir " + dir);
    laef::BackendConfig bc;
    bc.index_path = index_path;
    bc.metric = dm.metric;
    bc.estimator_path = dir + "/stats/estimator.bin";
    bc.dim = dm.dim;
    bc.M = cfg.hnsw.M;
    bc.k = cfg.k;
    bc.target_recall = static_cast<float>(cfg.target_recall);
    bc.quantile_step = cfg.adaef.quantile_step;
    bc.statics_length = menu.full().probe_len;
    for (const auto& o : menu.options()) {
      require_file(o.adapter_path, "run: laef_calibrate --dataset-dir " + dir);
      bc.probes.push_back({o.probe_len, o.adapter_path});
    }
    auto backend = laef::AnnBackend::open(bc);

    // ---- workload: rates are absolute qps (checked here), then arrivals and evaluation rows ----
    laef::check_rates(cfg.workload);
    const std::vector<int64_t> arrivals = laef::make_arrivals(cfg.workload, cfg.workload.seed);
    const std::vector<uint32_t> pool = laef::eval_pool(dir, dm.n_query);
    const std::vector<uint32_t> eval_idx = laef::pick_queries(arrivals.size(), pool, cfg.workload.seed + 1);

    // ---- run ----
    laef::EngineInputs ei;
    ei.cfg = &cfg;
    ei.backend = backend.get();
    ei.probe_menu = &menu;
    ei.queries = ds.queries.data();
    ei.dim = dm.dim;
    ei.gt = gt.neighbors.data();
    ei.gt_k = static_cast<uint32_t>(gt.k);
    ei.arrivals_ns = arrivals;
    ei.eval_idx = eval_idx;
    ei.warmup = std::min<uint64_t>(cfg.workload.warmup_queries, arrivals.size());
    ei.t_slo_s = t_slo_s;
    ei.u_s_per_dc = u_sat;
    ei.mean_service_s = mean_service;
    ei.s_p95_s = saturation_p95;
    ei.threads = cfg.engine.threads;
    ei.pin_cores = cfg.engine.pin_cores;
    std::cout << "run " << cfg.policy << " on " << cfg.dataset << ": " << arrivals.size() << " arrivals, C=" << C
              << ", T_SLO=" << t_slo_s * 1e3 << " ms" << std::endl;
    const laef::EngineResult res = laef::run_engine(ei);

    // ---- outputs ----
    const double offered_qps =
        arrivals.size() > 1 ? (arrivals.size() - 1) * 1e9 / static_cast<double>(arrivals.back() - arrivals.front()) : 0.0;
    laef::SummaryParams sp;
    sp.t_slo_ms = t_slo_s * 1e3;
    sp.offered_qps = offered_qps;
    sp.target_recall = cfg.target_recall;
    sp.controller_update_us = res.controller_update_us;
    sp.generator_lag_p99_us = res.generator_lag_p99_us;
    sp.full_probe_len = menu.full().probe_len;
    const nlohmann::json summary = laef::compute_summary(res.records, res.timeseries, sp);

    fs::create_directories(out_dir);
    nlohmann::json config = laef::to_json(cfg);  // includes the resolved absolute rates
    config["t_slo_s"] = t_slo_s;
    config["repeat"] = repeat;
    config["dataset"] = cfg.dataset;
    config["policy"] = cfg.policy;
    laef::write_json_file(out_dir + "/config.json", config);
    laef::write_queries_csv(out_dir + "/queries.csv", res.records);
    laef::write_timeseries_csv(out_dir + "/timeseries.csv", res.timeseries);
    laef::write_json_file(out_dir + "/summary.json", summary);

    std::cout << "p95 " << summary.value("p95_ms", 0.0) << " ms (T_SLO " << sp.t_slo_ms << " ms), mean recall "
              << summary.value("mean_recall", 0.0) << ", SLO attainment " << summary.value("slo_attainment", 0.0)
              << "\nwrote " << out_dir << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
