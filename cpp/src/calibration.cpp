// LA-ef: Step A driver (IMPLEMENTATION_PLAN.md §9.1-§9.4).
#include "laef/calibration.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <istream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "laef/ann_backend.h"
#include "laef/types.h"

namespace laef {
namespace {

// Calls body(i) for every i in [0, n) on `threads` threads with dynamic scheduling.
// The first exception is rethrown after all threads have stopped.
template <class Fn>
void parallel_for(size_t n, uint32_t threads, Fn&& body) {
  std::atomic<size_t> next{0};
  std::exception_ptr first_error;
  std::mutex error_mu;
  auto worker = [&] {
    for (;;) {
      const size_t i = next.fetch_add(1);
      if (i >= n) break;
      try {
        body(i);
      } catch (...) {
        std::lock_guard<std::mutex> lk(error_mu);
        if (!first_error) first_error = std::current_exception();
        next.store(n);  // hand out no more work
      }
    }
  };
  std::vector<std::thread> pool;
  for (uint32_t t = 1; t < threads; ++t) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
  if (first_error) std::rethrow_exception(first_error);
}

int32_t read_i32(std::istream& in) {
  int32_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof v);
  return v;
}

double median_of(std::vector<double> v) {
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];  // upper median for an even count
}

}  // namespace

Samplings read_samplings(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path + " (generate it with gpu_preprocessing --steps calib)");
  Samplings s;
  const int32_t rows = read_i32(in), cols = read_i32(in);
  if (rows <= 0 || cols <= 0) throw std::runtime_error(path + ": bad query matrix header");
  s.n = static_cast<uint32_t>(rows);
  s.dim = static_cast<uint32_t>(cols);
  s.queries.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  in.read(reinterpret_cast<char*>(s.queries.data()), static_cast<std::streamsize>(s.queries.size() * sizeof(float)));
  const int32_t grows = read_i32(in), gcols = read_i32(in);
  if (grows != rows || gcols <= 0) throw std::runtime_error(path + ": bad ground-truth matrix header");
  s.gt_k = static_cast<uint32_t>(gcols);
  s.gt.resize(static_cast<size_t>(grows) * static_cast<size_t>(gcols));
  in.read(reinterpret_cast<char*>(s.gt.data()), static_cast<std::streamsize>(s.gt.size() * sizeof(int32_t)));
  if (!in) throw std::runtime_error(path + ": truncated file");
  return s;
}

std::vector<uint32_t> make_ladder(uint32_t k, double ratio, uint32_t max_pref, uint32_t ef_upper_bound) {
  if (!(ratio > 1.0)) throw std::invalid_argument("calibration.ladder_ratio must be greater than 1");
  std::vector<uint32_t> ladder{k};
  while (ladder.back() < max_pref && ladder.back() < ef_upper_bound) {
    const double next = std::ceil(ladder.back() * ratio);
    uint32_t e = next >= ef_upper_bound ? ef_upper_bound : static_cast<uint32_t>(next);
    if (e <= ladder.back()) e = ladder.back() + 1;  // ratios very close to 1 still make progress
    ladder.push_back(e);
  }
  return ladder;
}

SweepResult run_sweep(const AnnBackend& backend, const Samplings& s, const CalibrationSettings& cfg,
                      uint32_t probe_len) {
  if (s.dim != backend.dim()) throw std::runtime_error("samplings dimension does not match the index");
  if (s.gt_k < cfg.k) throw std::runtime_error("samplings ground truth has fewer than k columns");
  if (probe_len == 0) throw std::invalid_argument("run_sweep: probe length must be positive");
  const size_t n = s.n;

  // Preference pass: Ada-ef's own operating point for every calibration query.
  std::vector<SweepRow> pref(n);
  parallel_for(n, cfg.threads, [&](size_t i) {
    std::vector<uint32_t> ids(cfg.k);
    SearchStats st;
    const float* q = s.queries.data() + i * s.dim;
    const int64_t t0 = now_ns();
    backend.search_planned(q, cfg.k, probe_len, [](float, uint32_t ef_pref, uint32_t) { return ef_pref; }, ids.data(), &st);
    const int64_t t1 = now_ns();
    if (!st.probe_complete)
      throw std::runtime_error("calibration query " + std::to_string(i) + " finished before the Ada-ef probe");
    SweepRow& r = pref[i];
    r.query = static_cast<uint32_t>(i);
    r.score = st.score;
    r.ef_pref = st.ef_pref;
    r.ef = st.ef_used;
    r.recall = recall_at_k(ids.data(), s.gt.data() + i * s.gt_k, cfg.k);
    r.ndis = st.ndis;
    r.probe_ndis = st.probe_ndis;
    r.service_us = (t1 - t0) * 1e-3;
    r.is_pref = true;
  });

  uint32_t max_pref = cfg.k;
  for (const auto& r : pref) max_pref = std::max(max_pref, r.ef_pref);
  SweepResult out;
  out.ladder = make_ladder(cfg.k, cfg.ladder_ratio, max_pref, cfg.ef_upper_bound);
  const auto& L = out.ladder;

  // Ladder sweep: one task per (query, rung). The planner returns the rung, so the probe is the same as in serving.
  std::vector<SweepRow> sweep(n * L.size());
  parallel_for(sweep.size(), cfg.threads, [&](size_t t) {
    const size_t i = t / L.size();
    const uint32_t e = L[t % L.size()];
    std::vector<uint32_t> ids(cfg.k);
    SearchStats st;
    const float* q = s.queries.data() + i * s.dim;
    const int64_t t0 = now_ns();
    backend.search_planned(q, cfg.k, probe_len, [e](float, uint32_t, uint32_t) { return e; }, ids.data(), &st);
    const int64_t t1 = now_ns();
    SweepRow& r = sweep[t];
    r.query = static_cast<uint32_t>(i);
    r.score = pref[i].score;
    r.ef_pref = pref[i].ef_pref;
    r.ef = e;
    r.recall = recall_at_k(ids.data(), s.gt.data() + i * s.gt_k, cfg.k);
    r.ndis = st.ndis;
    r.probe_ndis = st.probe_ndis;
    r.service_us = (t1 - t0) * 1e-3;
    r.is_pref = false;
  });

  out.rows.reserve(pref.size() + sweep.size());
  out.rows.insert(out.rows.end(), pref.begin(), pref.end());
  out.rows.insert(out.rows.end(), sweep.begin(), sweep.end());
  return out;
}

CalibrationResult build_price_list(const std::vector<SweepRow>& rows, size_t n, const std::vector<uint32_t>& ladder,
                                   const CalibrationSettings& cfg) {
  const size_t R = ladder.size();
  if (n == 0 || R == 0) throw std::invalid_argument("build_price_list: no calibration data");

  // Dense tables: [query][rung] for the sweep, [query] for the preference pass.
  std::vector<int> key(n, 0);
  std::vector<uint32_t> pref(n, 0);
  std::vector<double> rec_pref(n, 0.0), ndis_pref(n, 0.0);
  std::vector<double> rec(n * R, 0.0), work(n * R, 0.0), svc(n * R, 0.0);
  std::map<uint32_t, size_t> rung_of;
  for (size_t j = 0; j < R; ++j) rung_of[ladder[j]] = j;

  for (const auto& r : rows) {
    if (r.is_pref) {
      key[r.query] = std::clamp(static_cast<int>(r.score), 0, kMaxScore);  // Ada-ef's group key: int(score)
      pref[r.query] = r.ef_pref;
      rec_pref[r.query] = r.recall;
      ndis_pref[r.query] = r.ndis;
    } else {
      const size_t idx = r.query * R + rung_of.at(r.ef);
      rec[idx] = r.recall;
      work[idx] = r.ndis;
      svc[idx] = r.service_us;
    }
  }

  std::vector<size_t> counts(kMaxScore + 1, 0);
  for (size_t i = 0; i < n; ++i) counts[key[i]]++;

  CalibrationResult res;
  res.n_queries = n;
  PriceList& pl = res.price_list;
  pl.k = cfg.k;

  // Overall recall per rung, for static_ef_for_target (the static-ef baseline).
  std::vector<double> recall_all(R, 0.0);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < R; ++j) recall_all[j] += rec[i * R + j] / static_cast<double>(n);

  for (const auto& [lo, hi] : make_groups(counts, cfg.min_group_size, cfg.max_groups)) {
    std::vector<size_t> members;
    for (size_t i = 0; i < n; ++i)
      if (key[i] >= lo && key[i] <= hi) members.push_back(i);
    const double m = static_cast<double>(members.size());

    std::vector<double> mr(R, 0.0), mw(R, 0.0), ms(R, 0.0);
    std::vector<double> prefs;
    for (size_t i : members) {
      for (size_t j = 0; j < R; ++j) {
        mr[j] += rec[i * R + j];
        mw[j] += work[i * R + j];
        ms[j] += svc[i * R + j];
      }
      prefs.push_back(pref[i]);
    }
    for (size_t j = 0; j < R; ++j) {
      mr[j] /= m;
      mw[j] /= m;
      ms[j] /= m;
    }

    std::vector<uint32_t> kept = ladder;
    envelope_inplace(kept, mr, mw, &ms);  // running max, then the upper concave envelope of (work, recall)

    GroupCurve g;
    g.id = static_cast<int>(pl.groups.size());
    g.score_lo = lo;
    g.score_hi = hi;
    g.p = m / static_cast<double>(n);
    g.ef_pref_median = median_of(prefs);
    g.ladder = std::move(kept);
    g.recall = std::move(mr);
    g.work = std::move(mw);
    g.service_us = std::move(ms);
    pl.groups.push_back(std::move(g));
  }

  pl.static_ef_for_target = ladder.back();
  for (size_t j = 0; j < R; ++j) {
    if (recall_all[j] >= cfg.target_recall - 1e-6) {
      pl.static_ef_for_target = ladder[j];
      break;
    }
  }

  pl.finalize();  // checks that score ranges are contiguous and builds the cut list

  double sum_rec = 0, sum_ndis = 0;
  for (size_t i = 0; i < n; ++i) {
    sum_rec += rec_pref[i];
    sum_ndis += ndis_pref[i];
  }
  res.calib_mean_recall_at_pref = sum_rec / static_cast<double>(n);
  res.nominal_work_mean = sum_ndis / static_cast<double>(n);
  pl.nominal_work_mean_measured = res.nominal_work_mean;
  return res;
}

Samplings select_rows(const Samplings& s, const std::vector<uint32_t>& rows) {
  Samplings out;
  out.n = static_cast<uint32_t>(rows.size());
  out.dim = s.dim;
  out.gt_k = s.gt_k;
  out.queries.reserve(rows.size() * s.dim);
  out.gt.reserve(rows.size() * s.gt_k);
  for (uint32_t r : rows) {
    if (r >= s.n) throw std::out_of_range("select_rows: row index out of range");
    const auto qb = s.queries.begin() + static_cast<std::ptrdiff_t>(r) * s.dim;
    out.queries.insert(out.queries.end(), qb, qb + s.dim);
    const auto gb = s.gt.begin() + static_cast<std::ptrdiff_t>(r) * s.gt_k;
    out.gt.insert(out.gt.end(), gb, gb + s.gt_k);
  }
  return out;
}

void write_samplings(const std::string& path, const Samplings& s) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  auto put_i32 = [&f](int32_t v) { f.write(reinterpret_cast<const char*>(&v), sizeof v); };
  put_i32(static_cast<int32_t>(s.n));
  put_i32(static_cast<int32_t>(s.dim));
  f.write(reinterpret_cast<const char*>(s.queries.data()), static_cast<std::streamsize>(s.queries.size() * sizeof(float)));
  put_i32(static_cast<int32_t>(s.n));
  put_i32(static_cast<int32_t>(s.gt_k));
  f.write(reinterpret_cast<const char*>(s.gt.data()), static_cast<std::streamsize>(s.gt.size() * sizeof(int32_t)));
  if (!f) throw std::runtime_error("write failed: " + path);
}

void write_sweep_csv(const std::string& path, const std::vector<SweepRow>& rows) {
  std::ofstream f(path);
  if (!f) throw std::runtime_error("cannot write " + path);
  f << "query,score,ef_pref,ef,recall,ndis,probe_ndis,service_us,is_pref\n";
  for (const auto& r : rows) {
    f << r.query << ',' << std::fixed << std::setprecision(4) << r.score << ',' << r.ef_pref << ',' << r.ef << ','
      << std::setprecision(6) << r.recall << ',' << r.ndis << ',' << r.probe_ndis << ','
      << std::setprecision(1) << r.service_us << ',' << (r.is_pref ? 1 : 0) << '\n';
  }
}

}  // namespace laef
