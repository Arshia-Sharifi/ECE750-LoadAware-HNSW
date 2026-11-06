// LA-ef: Ada-ef backend (IMPLEMENTATION_PLAN.md §8.2).
// THE ONLY TRANSLATION UNIT THAT MAY INCLUDE hnswlib/*.h (Ada-ef headers define non-inline functions).
//
// planned_search() is a line-for-line copy of the authors' HierarchicalNSW::adaptiveSearchKnn +
// adaptiveSearchBaseLayerST<bare_bone_search=true> (hnsw-ada-ef @ ed463f99, hnswalg.h ~L1335-1600) with:
//   1. no read of the shared ef_ member (floor is k)          -> thread-safe, setEf() never needed
//   2. VLA distances[statics_limit] -> thread_local vector
//   3. local ndis counter (actual distance calls), no shared atomics
//   4. sketch->estimate_ef2(score) replaced by the LA-ef planner hook
//   5. probe-incomplete reported instead of silently searching with ef = SIZE_MAX semantics
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hnswlib/hnswlib.h"
#include "hnswlib/adaptive_ef.h"
#include "laef/ann_backend.h"

namespace laef {
namespace {

using hnswlib::labeltype;
using hnswlib::linklistsizeint;
using hnswlib::tableint;
using HNSW = hnswlib::HierarchicalNSW<float>;
using MaxHeap = std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>, HNSW::CompareByFirst>;

void check_metric(const std::string& m) {
  if (m != "cd" && m != "ipd") throw std::invalid_argument("Ada-ef supports only metrics cd and ipd, got " + m);
}

// Ada-ef normalize_matrix: v * 1/(||v|| + 1e-30)
void normalize_into(const float* src, float* dst, size_t d) {
  double s = 0;
  for (size_t i = 0; i < d; ++i) s += double(src[i]) * src[i];
  const float inv = 1.0f / (static_cast<float>(std::sqrt(s)) + 1e-30f);
  for (size_t i = 0; i < d; ++i) dst[i] = src[i] * inv;
}

inline void prefetch(const void* p) {
#ifdef USE_SSE
  _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0);
#else
  (void)p;
#endif
}

// Greedy descent through the upper layers (identical to searchKnn / adaptiveSearchKnn).
tableint descend(const HNSW& h, const void* q, uint32_t& ndis) {
  tableint cur = h.enterpoint_node_;
  float curdist = h.fstdistfunc_(q, h.getDataByInternalId(cur), h.dist_func_param_);
  ++ndis;
  for (int level = h.maxlevel_; level > 0; level--) {
    bool changed = true;
    while (changed) {
      changed = false;
      unsigned int* data = (unsigned int*)h.get_linklist(cur, level);
      int size = h.getListCount(data);
      tableint* datal = (tableint*)(data + 1);
      for (int i = 0; i < size; i++) {
        tableint cand = datal[i];
        if (cand > h.max_elements_) throw std::runtime_error("cand error");
        float d = h.fstdistfunc_(q, h.getDataByInternalId(cand), h.dist_func_param_);
        ++ndis;
        if (d < curdist) { curdist = d; cur = cand; changed = true; }
      }
    }
  }
  return cur;
}

void emit(const HNSW& h, MaxHeap& top, uint32_t k, uint32_t* out) {
  while (top.size() > k) top.pop();
  for (uint32_t i = 0; i < k; ++i) out[i] = std::numeric_limits<uint32_t>::max();
  for (size_t i = top.size(); i-- > 0;) {
    out[i] = static_cast<uint32_t>(h.getExternalLabel(top.top().second));
    top.pop();
  }
}

template <class Planner>
void planned_search(const HNSW& h, const void* q, uint32_t k, size_t statics_length,
                    const hnswdis::ApproximatedScoreCalculator& sc, hnswdis::Sketch& sketch,
                    Planner&& planner, uint32_t* out, SearchStats* st) {
  SearchStats local;
  SearchStats& S = st ? *st : local;
  S = SearchStats{};
  S.probe_len = static_cast<uint32_t>(statics_length);
  uint32_t ndis = 0;
  if (h.cur_element_count == 0) { for (uint32_t i = 0; i < k; ++i) out[i] = UINT32_MAX; return; }
  const tableint ep = descend(h, q, ndis);

  size_t ef = std::numeric_limits<size_t>::max();  // probe phase: consider every candidate
  bool collecting = true;
  thread_local std::vector<float> distances;
  distances.resize(statics_length);
  size_t nd = 0;

  hnswlib::VisitedList* vl = h.visited_list_pool_->getFreeVisitedList();
  hnswlib::vl_type* visited = vl->mass;
  const hnswlib::vl_type tag = vl->curV;
  MaxHeap top, cand;

  float d0 = h.fstdistfunc_(q, h.getDataByInternalId(ep), h.dist_func_param_);
  ++ndis;
  float lowerBound = d0;
  top.emplace(d0, ep);
  distances[nd++] = d0;
  cand.emplace(-d0, ep);
  visited[ep] = tag;

  while (!cand.empty()) {
    auto cp = cand.top();
    if (-cp.first > lowerBound) break;  // bare-bone stop rule
    cand.pop();
    int* data = (int*)h.get_linklist0(cp.second);
    size_t size = h.getListCount((linklistsizeint*)data);
    prefetch(visited + *(data + 1));
    prefetch(h.data_level0_memory_ + (*(data + 1)) * h.size_data_per_element_ + h.offsetData_);
    for (size_t j = 1; j <= size; j++) {
      int cid = *(data + j);
      prefetch(visited + *(data + j + 1));
      prefetch(h.data_level0_memory_ + (*(data + j + 1)) * h.size_data_per_element_ + h.offsetData_);
      if (visited[cid] == tag) continue;
      visited[cid] = tag;
      char* obj = h.getDataByInternalId(cid);
      float d = h.fstdistfunc_(q, obj, h.dist_func_param_);
      ++ndis;
      bool consider;
      if (collecting) {
        consider = true;
        if (nd == statics_length) {  // probe complete: score -> Ada-ef ef -> planner decides
          collecting = false;
          const float score = sc.compute_score(q, *((size_t*)h.dist_func_param_), distances.data(), nd);
          const uint32_t ef_pref = std::max<uint32_t>(static_cast<uint32_t>(sketch.estimate_ef2(score)), k);
          S.score = score; S.ef_pref = ef_pref; S.probe_ndis = ndis; S.probe_complete = true;
          ef = std::max<uint32_t>(planner(score, ef_pref, ndis), k);
          S.ef_used = static_cast<uint32_t>(ef);
          while (top.size() > ef) top.pop();
          if (!top.empty()) lowerBound = top.top().first;
        }
      } else {
        consider = top.size() < ef || lowerBound > d;
      }
      if (consider) {
        cand.emplace(-d, cid);
        top.emplace(d, cid);
        if (collecting) distances[nd++] = d;
        while (top.size() > ef) top.pop();
        if (!top.empty()) lowerBound = top.top().first;
      }
    }
  }
  h.visited_list_pool_->releaseVisitedList(vl);
  if (collecting) {
    // The graph ended before the probe reached statics_length. The result is still valid,
    // but no score or planned ef exists; the engine settles the measured ndis separately.
    S.probe_ndis = ndis;
    S.ef_used = 0;
  }
  S.ndis = ndis;
  emit(h, top, k, out);
}

void fixed_search(const HNSW& h, const void* q, uint32_t k, size_t ef, uint32_t* out, SearchStats* st) {
  uint32_t ndis = 0;
  if (h.cur_element_count == 0) { for (uint32_t i = 0; i < k; ++i) out[i] = UINT32_MAX; return; }
  ef = std::max<size_t>(ef, k);
  const tableint ep = descend(h, q, ndis);
  hnswlib::VisitedList* vl = h.visited_list_pool_->getFreeVisitedList();
  hnswlib::vl_type* visited = vl->mass;
  const hnswlib::vl_type tag = vl->curV;
  MaxHeap top, cand;
  float d0 = h.fstdistfunc_(q, h.getDataByInternalId(ep), h.dist_func_param_);
  ++ndis;
  float lowerBound = d0;
  top.emplace(d0, ep);
  cand.emplace(-d0, ep);
  visited[ep] = tag;
  while (!cand.empty()) {
    auto cp = cand.top();
    if (-cp.first > lowerBound) break;
    cand.pop();
    int* data = (int*)h.get_linklist0(cp.second);
    size_t size = h.getListCount((linklistsizeint*)data);
    for (size_t j = 1; j <= size; j++) {
      int cid = *(data + j);
      prefetch(visited + *(data + j + 1));
      prefetch(h.data_level0_memory_ + (*(data + j + 1)) * h.size_data_per_element_ + h.offsetData_);
      if (visited[cid] == tag) continue;
      visited[cid] = tag;
      float d = h.fstdistfunc_(q, h.getDataByInternalId(cid), h.dist_func_param_);
      ++ndis;
      if (top.size() < ef || lowerBound > d) {
        cand.emplace(-d, cid);
        top.emplace(d, cid);
        if (top.size() > ef) top.pop();
        if (!top.empty()) lowerBound = top.top().first;
      }
    }
  }
  h.visited_list_pool_->releaseVisitedList(vl);
  if (st) { *st = SearchStats{}; st->ndis = ndis; st->ef_used = static_cast<uint32_t>(ef); st->ef_pref = st->ef_used; }
  emit(h, top, k, out);
}

// One Ada-ef table: the authors' EfAdapter for one probe length, and the Sketch that reads it.
struct ProbeTable {
  std::unique_ptr<hnswdis::EfAdapter> adapter;  // declared first: must outlive the Sketch, which holds a reference to it
  std::unique_ptr<hnswdis::Sketch> sketch;      // estimate_ef2 is read-only -> thread-safe
};

class AdaEfBackend final : public AnnBackend {
 public:
  explicit AdaEfBackend(const BackendConfig& c) : cfg_(c) {
    check_metric(c.metric);
    if (!cfg_.statics_length) cfg_.statics_length = default_statics_length(c.M);
    space_ = std::make_unique<hnswlib::InnerProductSpace>(c.dim);  // same as authors' init_space for cd/ipd
    hnsw_ = std::make_unique<HNSW>(space_.get(), c.index_path);
    est_ = hnswdis::load_estimator_from_file(c.estimator_path);
    sc_ = std::make_unique<hnswdis::ApproximatedScoreCalculator>(est_, c.quantile_step);
    // One table per probe length. Without an explicit menu, adapter_path is the only table, at statics_length.
    std::vector<ProbeAdapter> specs = c.probes;
    if (specs.empty()) specs.push_back({static_cast<uint32_t>(cfg_.statics_length), c.adapter_path});
    for (const auto& p : specs) {
      if (p.probe_len == 0) throw std::invalid_argument("probe length must be positive");
      if (tables_.count(p.probe_len)) throw std::invalid_argument("duplicate probe length " + std::to_string(p.probe_len));
      ProbeTable t;
      t.adapter = std::make_unique<hnswdis::EfAdapter>(p.adapter_path);
      t.sketch = std::make_unique<hnswdis::Sketch>(t.adapter->get_ef_recall_estimators(), c.target_recall);
      tables_.emplace(p.probe_len, std::move(t));
    }
  }

  uint32_t dim() const override { return cfg_.dim; }

  void search_planned(const float* q, uint32_t k, uint32_t probe_len, const EfPlannerFn& planner, uint32_t* out,
                      SearchStats* st) const override {
    const auto it = tables_.find(probe_len);
    if (it == tables_.end())
      throw std::invalid_argument("no Ada-ef table for probe length " + std::to_string(probe_len));
    planned_search(*hnsw_, prep(q), k, probe_len, *sc_, *it->second.sketch,
                   [&](float s, uint32_t e, uint32_t n) { return planner(s, e, n); }, out, st);
  }

  void search_fixed(const float* q, uint32_t k, uint32_t ef, uint32_t* out, SearchStats* st) const override {
    fixed_search(*hnsw_, prep(q), k, ef, out, st);
  }

 private:
  const float* prep(const float* q) const {
    if (cfg_.metric != "cd") return q;
    thread_local std::vector<float> buf;
    buf.resize(cfg_.dim);
    normalize_into(q, buf.data(), cfg_.dim);
    return buf.data();
  }

  BackendConfig cfg_;
  std::unique_ptr<hnswlib::InnerProductSpace> space_;
  std::unique_ptr<HNSW> hnsw_;
  std::shared_ptr<hnswdis::Estimator> est_;
  std::unique_ptr<hnswdis::ApproximatedScoreCalculator> sc_;
  std::map<uint32_t, ProbeTable> tables_;  // probe length -> table (read-only after construction)
};

hnswdis::MatrixXf to_matrix(const float* data, size_t n, size_t d, bool normalize) {
  hnswdis::MatrixXf m(n, d);
  for (size_t i = 0; i < n; ++i) {
    if (normalize) normalize_into(data + i * d, m.row(i).data(), d);
    else std::copy(data + i * d, data + (i + 1) * d, m.row(i).data());
  }
  return m;
}

}  // namespace

std::unique_ptr<AnnBackend> AnnBackend::open(const BackendConfig& c) { return std::make_unique<AdaEfBackend>(c); }

void build_index(const float* data, size_t n, size_t d, const std::string& metric, uint32_t M,
                 uint32_t efc, uint32_t threads, const std::string& out_index, size_t seed) {
  check_metric(metric);
  hnswlib::InnerProductSpace space(d);
  HNSW h(&space, n, M, efc, seed);
  const bool norm = metric == "cd";
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  std::exception_ptr err;
  std::mutex err_m;
  for (uint32_t t = 0; t < std::max(1u, threads); ++t)
    pool.emplace_back([&] {
      std::vector<float> buf(d);
      try {
        for (size_t i; (i = next.fetch_add(1)) < n;) {
          const float* v = data + i * d;
          if (norm) { normalize_into(v, buf.data(), d); v = buf.data(); }
          h.addPoint(v, i);
        }
      } catch (...) { std::lock_guard<std::mutex> g(err_m); err = std::current_exception(); }
    });
  for (auto& t : pool) t.join();
  if (err) std::rethrow_exception(err);
  h.saveIndex(out_index);
}

void build_ef_adapter(const BackendConfig& c, const std::string& samplings_bin, uint32_t ef_upper_bound,
                      const std::string& out_adapter_bin) {
  check_metric(c.metric);
  const size_t sl = c.statics_length ? c.statics_length : default_statics_length(c.M);
  hnswlib::InnerProductSpace space(c.dim);
  auto h = std::make_shared<HNSW>(&space, c.index_path);
  hnswdis::MatrixXf Q;
  hnswdis::MatrixXi GT;
  hnswdis::deserialize_samplings(samplings_bin, Q, GT);
  if (Q.cols() != static_cast<long>(c.dim)) throw std::runtime_error("samplings dim mismatch");
  if (GT.cols() < static_cast<long>(c.k)) throw std::runtime_error("samplings GT has fewer than k columns");
  if (c.metric == "cd")
    for (long i = 0; i < Q.rows(); ++i) normalize_into(Q.row(i).data(), Q.row(i).data(), c.dim);
  auto est = hnswdis::load_estimator_from_file(c.estimator_path);
  // data_vectors is dereferenced but never read by the authors' calibration path -> empty matrix.
  auto empty = std::make_shared<hnswdis::MatrixXf>();
  hnswdis::EfAdapter ad(h, empty, c.k, c.metric, c.target_recall, c.quantile_step, sl,
                        std::make_shared<hnswdis::MatrixXf>(Q), std::make_shared<hnswdis::MatrixXi>(GT), est,
                        static_cast<int>(ef_upper_bound));
  ad.serialize(out_adapter_bin);
}

// Calibration rows whose recall@k is non-zero at ef = k and at ef = ceil(1.5k). The authors'
// RecallEstimator::init indexes its statistics by the original query index after dropping zero-recall
// queries, so those rows must not reach the adapter (IMPLEMENTATION_PLAN.md finding 5).
// Uses our own search (fixed_search, parity-tested against searchKnn), so setEf() is never called.
std::vector<uint32_t> usable_calibration_rows(const BackendConfig& c, const std::string& samplings_bin) {
  check_metric(c.metric);
  hnswlib::InnerProductSpace space(c.dim);
  HNSW h(&space, c.index_path);
  hnswdis::MatrixXf Q;
  hnswdis::MatrixXi GT;
  hnswdis::deserialize_samplings(samplings_bin, Q, GT);
  if (Q.cols() != static_cast<long>(c.dim)) throw std::runtime_error("samplings dim mismatch");
  if (GT.cols() < static_cast<long>(c.k)) throw std::runtime_error("samplings GT has fewer than k columns");
  const uint32_t ef_hi = (3 * c.k + 1) / 2;  // ceil(1.5 k)

  std::vector<float> q(c.dim);
  std::vector<uint32_t> ids(c.k);
  // Number of ground-truth neighbours among the first k labels returned with this ef.
  auto hits = [&](uint32_t ef, long row) {
    fixed_search(h, q.data(), c.k, ef, ids.data(), nullptr);
    uint32_t found = 0;
    for (uint32_t i = 0; i < c.k; ++i)
      for (uint32_t j = 0; j < c.k; ++j)
        if (GT(row, j) >= 0 && ids[i] == static_cast<uint32_t>(GT(row, j))) {
          ++found;
          break;
        }
    return found;
  };

  std::vector<uint32_t> keep;
  for (long i = 0; i < Q.rows(); ++i) {
    if (c.metric == "cd") normalize_into(Q.row(i).data(), q.data(), c.dim);
    else std::copy(Q.row(i).data(), Q.row(i).data() + c.dim, q.data());
    if (hits(c.k, i) > 0 && hits(ef_hi, i) > 0) keep.push_back(static_cast<uint32_t>(i));
  }
  return keep;
}

void compute_estimator_cpu(const float* data, size_t n, size_t d, const std::string& metric,
                           const std::string& out_estimator_bin) {
  check_metric(metric);
  auto m = to_matrix(data, n, d, metric == "cd");
  auto est = hnswdis::init_estimator(metric, m);
  hnswdis::save_estimator_to_file(*est, out_estimator_bin);
}

}  // namespace laef
