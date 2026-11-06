#pragma once
// LA-ef: the ONLY interface to hnswlib / Ada-ef (IMPLEMENTATION_PLAN.md §8.1).
// Implemented in src/adaef_backend.cpp (the single TU allowed to include hnswlib headers).
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace laef {

struct SearchStats {
  uint32_t ndis = 0;          // distance computations actually performed (upper layers + layer 0)
  uint32_t probe_ndis = 0;    // dc performed when the planner hook fired
  float score = -1.f;         // Ada-ef score in [0,100]; -1 if the probe never completed
  uint32_t ef_pref = 0;       // Ada-ef's ef = max(k, Sketch::estimate_ef2(score))
  uint32_t ef_used = 0;       // ef returned by the planner (0 if probe incomplete)
  uint32_t probe_len = 0;     // L: the probe length this search ran with (0 for plain HNSW)
  bool probe_complete = false;
};

// Called once per query on the worker thread when the probe completes; returns ef (>= k is enforced).
using EfPlannerFn = std::function<uint32_t(float score, uint32_t ef_pref, uint32_t probe_ndis)>;

// One Ada-ef table: the score-to-ef table an EfAdapter built with statics_length = probe_len (plan §1A.3 C2).
struct ProbeAdapter {
  uint32_t probe_len = 0;
  std::string adapter_path;
};

struct BackendConfig {
  std::string index_path, metric /* cd | ipd */, estimator_path, adapter_path;
  uint32_t dim = 0, M = 16, k = 10;
  float target_recall = 0.95f;
  float quantile_step = 1e-3f;
  size_t statics_length = 0;  // 0 => 1 + 2M + (2M-1)*2M (= 1025 for M = 16, the authors' value)
  // Probe menu: one table per probe length. When empty, adapter_path is the only table, at statics_length.
  std::vector<ProbeAdapter> probes;
};

inline size_t default_statics_length(uint32_t M) { return 1 + 2 * M + (2 * M - 1) * 2 * M; }

class AnnBackend {
 public:
  virtual ~AnnBackend() = default;
  static std::unique_ptr<AnnBackend> open(const BackendConfig&);
  virtual uint32_t dim() const = 0;
  // Ada-ef probe of length probe_len + planner hook. Thread-safe. out_ids: k labels, closest first (missing -> UINT32_MAX).
  // probe_len must be one of the probes opened in BackendConfig (throws otherwise).
  virtual void search_planned(const float* q, uint32_t k, uint32_t probe_len, const EfPlannerFn& planner,
                              uint32_t* out_ids, SearchStats* st) const = 0;
  // Plain HNSW with explicit ef (static-ef baseline). Thread-safe (never touches setEf()).
  virtual void search_fixed(const float* q, uint32_t k, uint32_t ef, uint32_t* out_ids, SearchStats* st) const = 0;
};

// Offline helpers. Data pointers are row-major float32 [n x d], raw (normalised internally for cd).
void build_index(const float* data, size_t n, size_t d, const std::string& metric, uint32_t M,
                 uint32_t ef_construction, uint32_t threads, const std::string& out_index, size_t seed = 100);
// Calibration rows the Ada-ef table can use (plan finding 5): recall@k is non-zero at ef = k and at ef = ceil(1.5k).
// The authors' RecallEstimator::init indexes its statistics by the original query index after dropping
// zero-recall queries, so the other rows must not reach build_ef_adapter. Loads the index and the samplings only.
std::vector<uint32_t> usable_calibration_rows(const BackendConfig& cfg, const std::string& samplings_bin);
// Builds the authors' EfAdapter from samplings_bin and the estimator at cfg.estimator_path, then serializes it
// to out_adapter_bin.
void build_ef_adapter(const BackendConfig& cfg, const std::string& samplings_bin, uint32_t ef_upper_bound,
                      const std::string& out_adapter_bin);
// CPU fallback for stats/estimator.bin (hnswdis::init_estimator + save_estimator_to_file).
void compute_estimator_cpu(const float* data, size_t n, size_t d, const std::string& metric,
                           const std::string& out_estimator_bin);

}  // namespace laef
