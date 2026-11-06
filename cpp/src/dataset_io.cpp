// LA-ef: dataset readers for the NumPy layout (IMPLEMENTATION_PLAN.md §4). See dataset_io.h.
#include "laef/dataset_io.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace laef {
namespace {

// The part of a .npy header that the readers need.
struct NpyInfo {
  std::string descr;         // dtype, e.g. "<f4" or "<i4"
  std::vector<uint64_t> shape;
  uint64_t data_offset = 0;  // bytes from the start of the file to the first element
};

// Text that follows `key` in the header, up to the next `end` character.
std::string header_field(const std::string& header, const std::string& key, char end) {
  const size_t start = header.find(key);
  if (start == std::string::npos) throw std::runtime_error("npy header has no " + key);
  const size_t from = start + key.size();
  const size_t stop = header.find(end, from);
  if (stop == std::string::npos) throw std::runtime_error("malformed npy header");
  return header.substr(from, stop - from);
}

// Reads the magic string, the length prefix and the header dictionary (NumPy format 1.0, 2.0 or 3.0).
NpyInfo read_npy_info(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  unsigned char magic[8] = {};
  in.read(reinterpret_cast<char*>(magic), 8);
  if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error(path + " is not a .npy file");

  uint64_t header_len = 0, prefix = 0;
  if (magic[6] == 1) {  // 2-byte length
    uint16_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof len);
    header_len = len;
    prefix = 10;
  } else if (magic[6] == 2 || magic[6] == 3) {  // 4-byte length
    uint32_t len = 0;
    in.read(reinterpret_cast<char*>(&len), sizeof len);
    header_len = len;
    prefix = 12;
  } else {
    throw std::runtime_error(path + ": unsupported npy format version");
  }
  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!in) throw std::runtime_error(path + ": truncated npy header");

  NpyInfo info;
  info.descr = header_field(header, "'descr': '", '\'');
  if (header_field(header, "'fortran_order': ", ',') != "False")
    throw std::runtime_error(path + ": Fortran-ordered arrays are not supported");
  std::istringstream dims(header_field(header, "'shape': (", ')'));  // "(a, b)" or "(a,)"
  for (std::string item; std::getline(dims, item, ',');) {
    const size_t first = item.find_first_not_of(' ');
    if (first != std::string::npos) info.shape.push_back(std::stoull(item.substr(first)));
  }
  info.data_offset = prefix + header_len;
  return info;
}

// Reads a whole .npy array. The dtype must match `descr`; the shape is returned through `shape`.
template <class T>
std::vector<T> read_npy(const std::string& path, const char* descr, std::vector<uint64_t>& shape) {
  const NpyInfo info = read_npy_info(path);
  if (info.descr != descr) throw std::runtime_error(path + ": expected dtype " + descr + ", found " + info.descr);
  shape = info.shape;
  uint64_t count = 1;
  for (uint64_t d : shape) count *= d;
  std::vector<T> data(count);
  std::ifstream in(path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(info.data_offset));
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) throw std::runtime_error(path + ": truncated data");
  return data;
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  return nlohmann::json::parse(in);
}

}  // namespace

DatasetMeta read_dataset_meta(const std::string& dir) {
  const nlohmann::json meta = read_json(dir + "/meta.json");
  DatasetMeta m;
  m.metric = meta.value("metric", std::string("cd"));
  m.seed = meta.value("seed", uint64_t{0});

  const NpyInfo train = read_npy_info(dir + "/train.npy");
  if (train.shape.size() != 2) throw std::runtime_error(dir + "/train.npy must be a 2-D matrix");
  m.n = train.shape[0];
  m.dim = static_cast<uint32_t>(train.shape[1]);

  const NpyInfo test = read_npy_info(dir + "/test.npy");
  if (test.shape.size() != 2 || test.shape[1] != train.shape[1])
    throw std::runtime_error(dir + "/test.npy must be a Q x dim matrix matching train.npy");
  m.n_query = test.shape[0];
  return m;
}

Dataset load_queries(const std::string& dir) {
  Dataset d;
  static_cast<DatasetMeta&>(d) = read_dataset_meta(dir);
  std::vector<uint64_t> shape;
  d.queries = read_npy<float>(dir + "/test.npy", "<f4", shape);
  return d;
}

Dataset load_dataset(const std::string& dir) {
  Dataset d = load_queries(dir);
  std::vector<uint64_t> shape;
  d.vectors = read_npy<float>(dir + "/train.npy", "<f4", shape);
  if (d.vectors.size() != d.n * d.dim) throw std::runtime_error(dir + "/train.npy has an unexpected size");
  return d;
}

GroundTruth load_ground_truth(const std::string& dir) {
  const std::string gt_dir = dir + "/gt";
  std::vector<uint64_t> shape;
  GroundTruth g;
  g.neighbors = read_npy<int32_t>(gt_dir + "/neighbors.npy", "<i4", shape);
  if (shape.size() != 2) throw std::runtime_error(gt_dir + "/neighbors.npy must be a Q x K matrix");
  g.n = shape[0];
  g.k = static_cast<uint32_t>(shape[1]);
  const nlohmann::json meta = read_json(gt_dir + "/gt_meta.json");
  g.source = meta.value("gt_source", std::string("unknown"));
  g.metric = meta.value("metric", std::string("cd"));
  return g;
}

std::vector<uint32_t> eval_pool(const std::string& dir, uint64_t n_query) {
  std::vector<bool> excluded(n_query, false);
  std::ifstream in(dir + "/calibration/calib_meta.json");
  if (in) {
    const nlohmann::json meta = nlohmann::json::parse(in);
    if (meta.contains("excluded_test_rows")) {
      for (const auto& r : meta.at("excluded_test_rows")) {
        const uint64_t row = r.get<uint64_t>();
        if (row < n_query) excluded[row] = true;
      }
    }
  }
  std::vector<uint32_t> pool;
  for (uint64_t i = 0; i < n_query; ++i)
    if (!excluded[i]) pool.push_back(static_cast<uint32_t>(i));
  if (pool.empty()) throw std::runtime_error("no evaluation queries left after the calibration holdout");
  return pool;
}

}  // namespace laef
