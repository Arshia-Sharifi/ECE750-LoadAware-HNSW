#pragma once
// LA-ef: dataset readers for the canonical NumPy layout (IMPLEMENTATION_PLAN.md §4).
// Plain C++ with no library dependency: .npy files (format 1.0 or 2.0, little-endian float32 or int32)
// and JSON metadata. Every reader takes the dataset folder (data/artifacts/<id>).
//
//   <dataset>/train.npy          float32 [n x dim]       base vectors
//   <dataset>/test.npy           float32 [n_query x dim] query vectors
//   <dataset>/meta.json          metric, seed, ...       (holdout.npy is optional, not read here)
//   <dataset>/gt/neighbors.npy   int32 [n_query x K]     exact top-K, ascending distance
//   <dataset>/gt/gt_meta.json    gt_source, K, metric
#include <cstdint>
#include <string>
#include <vector>

namespace laef {

struct DatasetMeta {
  uint64_t n = 0;        // base rows (train.npy)
  uint64_t n_query = 0;  // query rows (test.npy)
  uint32_t dim = 0;
  std::string metric = "cd";
  uint64_t seed = 0;
};

struct Dataset : DatasetMeta {
  std::vector<float> vectors;  // [n x dim] raw; empty unless loaded with load_dataset
  std::vector<float> queries;  // [n_query x dim] raw
};

struct GroundTruth {
  uint64_t n = 0;
  uint32_t k = 0;                  // stored width K (>= the serving k)
  std::string source = "unknown";  // provided | generated
  std::string metric = "cd";
  std::vector<int32_t> neighbors;  // [n x k], ascending distance
};

DatasetMeta read_dataset_meta(const std::string& dataset_dir);  // meta.json and the shapes of train.npy, test.npy
Dataset load_dataset(const std::string& dataset_dir);           // base and queries (index build, estimator fallback)
Dataset load_queries(const std::string& dataset_dir);           // queries only (search apps)
GroundTruth load_ground_truth(const std::string& dataset_dir);  // gt/neighbors.npy and gt/gt_meta.json

// Test rows that may be used for evaluation: every row except calibration/calib_meta.json's
// excluded_test_rows (query_holdout calibration). Throws if nothing is left.
std::vector<uint32_t> eval_pool(const std::string& dataset_dir, uint64_t n_query);

}  // namespace laef
