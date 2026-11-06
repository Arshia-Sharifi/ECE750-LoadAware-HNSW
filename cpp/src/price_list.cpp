// LA-ef: price_list.json and probe_menu.json readers and writers (IMPLEMENTATION_PLAN.md §9.4, §1A.4).
// Algorithms: price_list.h and probe_menu.h.
#include "laef/price_list_io.h"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace laef {

void save_price_list(const std::string& path, const PriceList& pl, const PriceListMeta& meta) {
  nlohmann::json j;
  j["dataset"] = meta.dataset;
  j["metric"] = meta.metric;
  j["k"] = pl.k;
  j["target_recall"] = meta.target_recall;
  j["quantile_step"] = meta.quantile_step;
  j["M"] = meta.M;
  j["statics_length"] = meta.statics_length;
  j["ef_upper_bound"] = meta.ef_upper_bound;
  j["ladder"] = meta.ladder;
  j["n_calib"] = meta.n_calib;
  j["static_ef_for_target"] = pl.static_ef_for_target;
  j["floor_work_mean"] = pl.model_floor_work();
  j["nominal_work_mean"] = pl.nominal_work_mean_measured;
  j["model_nominal_work"] = pl.model_nominal_work();
  j["calib_mean_recall_at_pref"] = meta.calib_mean_recall_at_pref;
  j["ladder_ratio"] = meta.ladder_ratio;
  j["n_dropped_zero_recall"] = meta.n_dropped;

  j["groups"] = nlohmann::json::array();
  for (const auto& g : pl.groups) {
    nlohmann::json gj = {{"id", g.id},
                         {"score_lo", g.score_lo},
                         {"score_hi", g.score_hi},
                         {"p", g.p},
                         {"ef_pref_median", g.ef_pref_median},
                         {"ladder", g.ladder},
                         {"recall", g.recall},
                         {"work", g.work}};
    if (!g.service_us.empty()) gj["service_us"] = g.service_us;
    j["groups"].push_back(gj);
  }

  j["steps_sorted"] = nlohmann::json::array();
  for (const auto& s : pl.steps_sorted()) {
    j["steps_sorted"].push_back({{"group", s.group},
                                 {"from", s.from},
                                 {"to", s.to},
                                 {"d_recall", s.d_recall},
                                 {"d_work", s.d_work},
                                 {"loss", s.loss}});
  }

  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write price list: " + path);
  f << j.dump(2) << "\n";
}

PriceList load_price_list(const std::string& path, PriceListMeta* meta) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open price list: " + path + " (run laef_calibrate first)");
  const nlohmann::json j = nlohmann::json::parse(in);

  PriceList pl;
  pl.k = j.at("k").get<uint32_t>();
  pl.static_ef_for_target = j.at("static_ef_for_target").get<uint32_t>();
  pl.floor_work_mean_measured = j.value("floor_work_mean", 0.0);
  pl.nominal_work_mean_measured = j.value("nominal_work_mean", 0.0);
  for (const auto& gj : j.at("groups")) {
    GroupCurve g;
    g.id = gj.at("id").get<int>();
    g.score_lo = gj.at("score_lo").get<int>();
    g.score_hi = gj.at("score_hi").get<int>();
    g.p = gj.at("p").get<double>();
    g.ef_pref_median = gj.at("ef_pref_median").get<double>();
    g.ladder = gj.at("ladder").get<std::vector<uint32_t>>();
    g.recall = gj.at("recall").get<std::vector<double>>();
    g.work = gj.at("work").get<std::vector<double>>();
    if (gj.contains("service_us")) g.service_us = gj.at("service_us").get<std::vector<double>>();
    pl.groups.push_back(std::move(g));
  }
  pl.finalize();  // rebuilds the cut list and the model demand from the curves

  if (meta) {
    meta->dataset = j.value("dataset", std::string());
    meta->metric = j.value("metric", std::string("cd"));
    meta->target_recall = j.value("target_recall", 0.95);
    meta->quantile_step = j.value("quantile_step", 1e-3);
    meta->M = j.value("M", 16u);
    meta->statics_length = j.value("statics_length", size_t{0});
    meta->ef_upper_bound = j.value("ef_upper_bound", 5000u);
    meta->ladder = j.value("ladder", std::vector<uint32_t>{});
    meta->n_calib = j.value("n_calib", size_t{0});
    meta->calib_mean_recall_at_pref = j.value("calib_mean_recall_at_pref", 0.0);
    meta->ladder_ratio = j.value("ladder_ratio", 2.0);
    meta->n_dropped = j.value("n_dropped_zero_recall", size_t{0});
  }
  return pl;
}

void save_probe_menu(const std::string& path, const ProbeMenu& menu, const ProbeMenuMeta& meta) {
  nlohmann::json j;
  j["dataset"] = meta.dataset;
  j["k"] = meta.k;
  j["M"] = meta.M;
  j["target_recall"] = meta.target_recall;
  j["full_probe_len"] = meta.full_probe_len;
  j["options"] = nlohmann::json::array();
  for (const auto& o : menu.options()) {
    const std::string dir = "L" + std::to_string(o.probe_len);
    j["options"].push_back({{"probe_len", o.probe_len},
                            {"price_list", dir + "/price_list.json"},
                            {"adapter", dir + "/ada_ef_adapter.bin"},
                            {"model_nominal_work", o.price_list.model_nominal_work()},
                            {"model_floor_work", o.price_list.model_floor_work()},
                            {"model_nominal_recall", o.price_list.model_nominal_recall()},
                            {"calib_mean_recall_at_pref", o.calib_mean_recall_at_pref},
                            {"mean_probe_ndis", o.mean_probe_ndis},
                            {"ef_pref_mean", o.ef_pref_mean}});
  }
  j["dropped"] = nlohmann::json::array();
  for (const auto& d : meta.dropped) j["dropped"].push_back({{"probe_len", d.probe_len}, {"reason", d.reason}});

  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write probe menu: " + path);
  f << j.dump(2) << "\n";
}

ProbeMenu load_probe_menu(const std::string& path, ProbeMenuMeta* meta) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open probe menu: " + path + " (run laef_calibrate first)");
  const nlohmann::json j = nlohmann::json::parse(in);
  const std::filesystem::path base = std::filesystem::path(path).parent_path();

  ProbeMenuMeta m;
  m.dataset = j.value("dataset", std::string());
  m.k = j.at("k").get<uint32_t>();
  m.M = j.value("M", 16u);
  m.target_recall = j.value("target_recall", 0.95);
  m.full_probe_len = j.at("full_probe_len").get<uint32_t>();

  std::vector<ProbeOption> options;
  for (const auto& oj : j.at("options")) {
    ProbeOption o;
    o.probe_len = oj.at("probe_len").get<uint32_t>();
    o.price_list_path = (base / oj.at("price_list").get<std::string>()).string();
    o.adapter_path = (base / oj.at("adapter").get<std::string>()).string();
    PriceListMeta pm;
    o.price_list = load_price_list(o.price_list_path, &pm);
    if (o.price_list.k != m.k)
      throw std::runtime_error(o.price_list_path + ": built for k=" + std::to_string(o.price_list.k) +
                               ", the probe menu has k=" + std::to_string(m.k));
    if (pm.statics_length != o.probe_len)
      throw std::runtime_error(o.price_list_path + ": built for probe length " + std::to_string(pm.statics_length) +
                               ", the menu lists " + std::to_string(o.probe_len));
    o.calib_mean_recall_at_pref = oj.value("calib_mean_recall_at_pref", 0.0);
    o.mean_probe_ndis = oj.value("mean_probe_ndis", 0.0);
    o.ef_pref_mean = oj.value("ef_pref_mean", 0.0);
    options.push_back(std::move(o));
  }
  if (j.contains("dropped"))
    for (const auto& dj : j.at("dropped"))
      m.dropped.push_back({dj.at("probe_len").get<uint32_t>(), dj.at("reason").get<std::string>()});

  if (meta) *meta = m;
  return ProbeMenu(std::move(options));
}

}  // namespace laef
