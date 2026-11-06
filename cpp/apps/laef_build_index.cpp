// LA-ef: build the HNSW index (IMPLEMENTATION_PLAN.md §8.2 P2.4, §12 P6.2).
// Output: <dataset-dir>/index/hnsw_M{M}_efc{efc}.bin plus index_meta.json.
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include "laef/ann_backend.h"
#include "laef/dataset_io.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  CLI::App app{"LA-ef: build the HNSW index"};
  std::string dataset_dir;
  uint32_t M = 16, efc = 500, threads = 0;
  uint64_t seed = 100;

  app.add_option("--dataset-dir", dataset_dir, "data/artifacts/<id> (contains train.npy, test.npy)")->required();
  app.add_option("-M,--M", M, "HNSW M (the authors use 16)");
  app.add_option("--efc,--ef-construction", efc, "HNSW ef_construction (the authors use 500)");
  app.add_option("-j,--threads", threads, "build threads (0 = all cores)");
  app.add_option("--seed", seed, "HNSW level seed");
  CLI11_PARSE(app, argc, argv);

  try {
    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
    const laef::Dataset ds = laef::load_dataset(dataset_dir);
    const std::string index_dir = dataset_dir + "/index";
    fs::create_directories(index_dir);
    const std::string out = index_dir + "/hnsw_M" + std::to_string(M) + "_efc" + std::to_string(efc) + ".bin";

    std::cout << "building " << out << " from " << ds.n << " x " << ds.dim << " (" << ds.metric << ") on "
              << threads << " threads" << std::endl;
    laef::build_index(ds.vectors.data(), ds.n, ds.dim, ds.metric, M, efc, threads, out, seed);

    const nlohmann::json meta = {{"M", M},       {"ef_construction", efc}, {"threads", threads}, {"seed", seed},
                                 {"n", ds.n},    {"dim", ds.dim},          {"metric", ds.metric}};
    std::ofstream(index_dir + "/index_meta.json") << meta.dump(2) << "\n";
    std::cout << "wrote " << out << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
