#pragma once
// LA-ef: run configuration (IMPLEMENTATION_PLAN.md §12 P6.1, §15).
// The structs mirror configs/defaults.yaml (same keys, same defaults). Precedence:
//   built-in defaults -> --config run.json (partial JSON is fine; "policy" preset applied first)
//   -> CLI flags.  scripts/run_experiments.py writes each run.json from its experiment file.
// Nothing parses defaults.yaml: the defaults here are kept in step with it by hand.
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace laef {

using json = nlohmann::json;

namespace detail {
template <class T>
void get_if(const json& j, const char* key, T& v) {
  auto it = j.find(key);
  if (it != j.end() && !it->is_null()) v = it->get<T>();
}
}  // namespace detail

struct HnswCfg { uint32_t M = 16, ef_construction = 500; uint64_t seed = 100; };
struct AdaefCfg {
  double quantile_step = 1e-3;
  uint64_t statics_length = 0;  // full probe: 0 => 1 + 2M + (2M-1)*2M (1025 for M = 16)
  uint32_t ef_upper_bound = 5000;
  // Probe menu (plan §1A.2, §1A.4): one Ada-ef table per probe length. Descending; the largest is the full probe.
  std::vector<uint32_t> probe_lengths = {1025, 512, 256, 128};
};
struct ProbeCfg { bool enabled = true; };  // false: only the full probe is kept (the v1 ef-only ablation)
struct CalibCfg {
  uint64_t n = 1000;
  std::string source = "base_sample";
  uint64_t seed = 42;
  double ladder_ratio = 2.0;
  uint64_t min_group_size = 30, max_groups = 8;
};
struct EngineCfg { int threads = 0; bool pin_cores = false; };
struct SloCfg { double ms = 0; /* T_SLO in ms; required. The engine takes it as given (e.g. the Ada-ef p95 at this load) */ };
struct WindowCfg { uint64_t n = 500; double max_ms = 20; uint64_t min_samples = 50; };
struct TelemetryCfg { double alpha_u = 0.05, alpha_es = 0.05, alpha_lambda = 0.2, lambda_tick_ms = 10; };
struct ControllerCfg {
  std::string type = "p_aimd";  // p_aimd | fixed_aimd | static_capacity | none
  double cut_floor = 0.5, headroom = 0.95, growth = 1.02;
  bool guard = true;
  double fixed_cut = 0.9;
  bool fixed_guard = false;     // fixed_aimd keeps the guard off by default (naive baseline)
  double rho_target = 0.9;
};
struct BucketCfg { std::string t_base = "mean"; double cap_pos = 0; };
struct AllocatorCfg {
  std::string type = "water_filling";  // water_filling | uniform_cut | passthrough | static_ef
  std::string uniform_mode = "work";   // work | ef
};
struct FloorModeCfg { bool enabled = true; };
struct WorkloadCfg {
  std::string profile = "constant";  // arrival process: constant | poisson (workload.h)
  double qps = 0;                   // arrival rate lambda [queries/s]; required for poisson and constant
  uint64_t n_queries = 20000;
  double duration_s = 0;            // > 0 overrides n_queries (poisson, constant)
  uint64_t warmup_queries = 1000;
  uint64_t seed = 7;
};

struct RunConfig {
  uint32_t k = 10;
  double target_recall = 0.95;
  std::string policy;   // informational after apply_policy()
  std::string dataset;  // informational
  HnswCfg hnsw;
  AdaefCfg adaef;
  CalibCfg calibration;
  EngineCfg engine;
  SloCfg slo;
  WindowCfg window;
  TelemetryCfg telemetry;
  ControllerCfg controller;
  BucketCfg bucket;
  AllocatorCfg allocator;
  FloorModeCfg floor_mode;
  ProbeCfg probe;
  WorkloadCfg workload;
};

// Policy presets (plan §12 P6.3). Individual settings applied afterwards override the preset.
inline void apply_policy(RunConfig& c, const std::string& p) {
  if (p == "static_ef") {
    c.allocator.type = "static_ef"; c.controller.type = "none"; c.floor_mode.enabled = false;
  } else if (p == "ada_ef") {
    c.allocator.type = "passthrough"; c.controller.type = "none"; c.floor_mode.enabled = false;
  } else if (p == "laef_uniform") {
    c.allocator.type = "uniform_cut"; c.controller.type = "p_aimd"; c.floor_mode.enabled = true;
  } else if (p == "laef_wf" || p == "laef") {
    c.allocator.type = "water_filling"; c.controller.type = "p_aimd"; c.floor_mode.enabled = true;
  } else {
    throw std::invalid_argument("unknown policy: " + p + " (static_ef | ada_ef | laef_uniform | laef_wf)");
  }
  c.policy = p == "laef" ? "laef_wf" : p;
}

// ---------------- JSON <-> structs ----------------
inline void merge(RunConfig& c, const json& j) {
  using detail::get_if;
  if (j.contains("policy") && j["policy"].is_string() && !j["policy"].get<std::string>().empty())
    apply_policy(c, j["policy"].get<std::string>());
  get_if(j, "k", c.k); get_if(j, "target_recall", c.target_recall); get_if(j, "dataset", c.dataset);
  if (auto s = j.find("hnsw"); s != j.end()) {
    get_if(*s, "M", c.hnsw.M); get_if(*s, "ef_construction", c.hnsw.ef_construction); get_if(*s, "seed", c.hnsw.seed);
  }
  if (auto s = j.find("adaef"); s != j.end()) {
    get_if(*s, "quantile_step", c.adaef.quantile_step);
    get_if(*s, "statics_length", c.adaef.statics_length);
    get_if(*s, "ef_upper_bound", c.adaef.ef_upper_bound);
    get_if(*s, "probe_lengths", c.adaef.probe_lengths);
  }
  if (auto s = j.find("calibration"); s != j.end()) {
    get_if(*s, "n", c.calibration.n); get_if(*s, "source", c.calibration.source); get_if(*s, "seed", c.calibration.seed);
    get_if(*s, "ladder_ratio", c.calibration.ladder_ratio);
    get_if(*s, "min_group_size", c.calibration.min_group_size); get_if(*s, "max_groups", c.calibration.max_groups);
  }
  if (auto s = j.find("engine"); s != j.end()) { get_if(*s, "threads", c.engine.threads); get_if(*s, "pin_cores", c.engine.pin_cores); }
  if (auto s = j.find("slo"); s != j.end()) {
    get_if(*s, "ms", c.slo.ms);
  }
  if (auto s = j.find("window"); s != j.end()) {
    get_if(*s, "n", c.window.n); get_if(*s, "max_ms", c.window.max_ms); get_if(*s, "min_samples", c.window.min_samples);
  }
  if (auto s = j.find("telemetry"); s != j.end()) {
    get_if(*s, "alpha_u", c.telemetry.alpha_u); get_if(*s, "alpha_es", c.telemetry.alpha_es);
    get_if(*s, "alpha_lambda", c.telemetry.alpha_lambda); get_if(*s, "lambda_tick_ms", c.telemetry.lambda_tick_ms);
  }
  if (auto s = j.find("controller"); s != j.end()) {
    get_if(*s, "type", c.controller.type); get_if(*s, "cut_floor", c.controller.cut_floor);
    get_if(*s, "headroom", c.controller.headroom); get_if(*s, "growth", c.controller.growth);
    get_if(*s, "guard", c.controller.guard); get_if(*s, "fixed_cut", c.controller.fixed_cut);
    get_if(*s, "fixed_guard", c.controller.fixed_guard); get_if(*s, "rho_target", c.controller.rho_target);
  }
  if (auto s = j.find("bucket"); s != j.end()) { get_if(*s, "t_base", c.bucket.t_base); get_if(*s, "cap_pos", c.bucket.cap_pos); }
  if (auto s = j.find("allocator"); s != j.end()) {
    get_if(*s, "type", c.allocator.type); get_if(*s, "uniform_mode", c.allocator.uniform_mode);
  }
  if (auto s = j.find("floor_mode"); s != j.end()) get_if(*s, "enabled", c.floor_mode.enabled);
  if (auto s = j.find("probe"); s != j.end()) get_if(*s, "enabled", c.probe.enabled);
  if (auto s = j.find("workload"); s != j.end()) {
    auto& w = c.workload;
    get_if(*s, "profile", w.profile); get_if(*s, "qps", w.qps);
    get_if(*s, "n_queries", w.n_queries); get_if(*s, "duration_s", w.duration_s);
    get_if(*s, "warmup_queries", w.warmup_queries); get_if(*s, "seed", w.seed);
  }
}

inline json to_json(const RunConfig& c) {
  return {
      {"k", c.k}, {"target_recall", c.target_recall}, {"policy", c.policy}, {"dataset", c.dataset},
      {"hnsw", {{"M", c.hnsw.M}, {"ef_construction", c.hnsw.ef_construction}, {"seed", c.hnsw.seed}}},
      {"adaef", { {"quantile_step", c.adaef.quantile_step},
                   {"statics_length", c.adaef.statics_length},
                   {"ef_upper_bound", c.adaef.ef_upper_bound},
                   {"probe_lengths", c.adaef.probe_lengths} }},
      {"calibration", {{"n", c.calibration.n}, {"source", c.calibration.source}, {"seed", c.calibration.seed},
                       {"ladder_ratio", c.calibration.ladder_ratio}, {"min_group_size", c.calibration.min_group_size},
                       {"max_groups", c.calibration.max_groups}}},
      {"engine", {{"threads", c.engine.threads}, {"pin_cores", c.engine.pin_cores}}},
      {"slo", {{"ms", c.slo.ms}}},
      {"window", {{"n", c.window.n}, {"max_ms", c.window.max_ms}, {"min_samples", c.window.min_samples}}},
      {"telemetry", {{"alpha_u", c.telemetry.alpha_u}, {"alpha_es", c.telemetry.alpha_es},
                     {"alpha_lambda", c.telemetry.alpha_lambda}, {"lambda_tick_ms", c.telemetry.lambda_tick_ms}}},
      {"controller", {{"type", c.controller.type}, {"cut_floor", c.controller.cut_floor}, {"headroom", c.controller.headroom},
                      {"growth", c.controller.growth}, {"guard", c.controller.guard}, {"fixed_cut", c.controller.fixed_cut},
                      {"fixed_guard", c.controller.fixed_guard}, {"rho_target", c.controller.rho_target}}},
      {"bucket", {{"t_base", c.bucket.t_base}, {"cap_pos", c.bucket.cap_pos}}},
      {"allocator", {{"type", c.allocator.type}, {"uniform_mode", c.allocator.uniform_mode}}},
      {"floor_mode", {{"enabled", c.floor_mode.enabled}}},
      {"probe", {{"enabled", c.probe.enabled}}},
      {"workload", {{"profile", c.workload.profile}, {"qps", c.workload.qps},
                    {"n_queries", c.workload.n_queries}, {"duration_s", c.workload.duration_s},
                    {"warmup_queries", c.workload.warmup_queries}, {"seed", c.workload.seed}}},

  };
}

inline json read_json_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  return json::parse(f);
}

inline void write_json_file(const std::string& path, const json& j) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << j.dump(2) << "\n";
}

inline RunConfig load_run_config(const std::string& path) {
  RunConfig c;
  merge(c, read_json_file(path));
  return c;
}

}  // namespace laef
