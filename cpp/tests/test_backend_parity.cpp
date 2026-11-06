// LA-ef: backend parity with the AUTHORS' Ada-ef code + Ada-ef binary format check (IMPLEMENTATION_PLAN.md §13.1).
// Unity build: this single TU includes adaef_backend.cpp, so hnswlib headers appear in exactly one TU.
#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <thread>

#include "../src/adaef_backend.cpp"

namespace fs = std::filesystem;
using namespace laef;

namespace {
constexpr uint32_t kDim = 32, kN = 6000, kK = 10, kM = 16, kQ = 200;
const uint32_t kProbeLen = static_cast<uint32_t>(default_statics_length(kM));  // full probe: 1025 for M = 16

struct Fixture {
  fs::path dir;
  std::vector<float> data, queries;
  BackendConfig cfg;
};

std::vector<uint32_t> brute_knn_cd(const std::vector<float>& base, const float* q, uint32_t n, uint32_t d, uint32_t k) {
  std::vector<float> qn(d), xn(d);
  normalize_into(q, qn.data(), d);
  std::vector<std::pair<float, uint32_t>> all(n);
  for (uint32_t i = 0; i < n; ++i) {
    normalize_into(&base[size_t(i) * d], xn.data(), d);
    float dot = 0;
    for (uint32_t j = 0; j < d; ++j) dot += qn[j] * xn[j];
    all[i] = {1.f - dot, i};
  }
  std::partial_sort(all.begin(), all.begin() + k, all.end());
  std::vector<uint32_t> ids(k);
  for (uint32_t i = 0; i < k; ++i) ids[i] = all[i].second;
  return ids;
}

const Fixture& fixture() {
  static Fixture F = [] {
    Fixture f;
    f.dir = fs::temp_directory_path() / "laef_parity";
    fs::create_directories(f.dir);
    std::mt19937 rng(7);
    std::normal_distribution<float> N(0, 1);
    std::vector<float> centers(50 * kDim);
    for (auto& c : centers) c = N(rng);
    f.data.resize(size_t(kN) * kDim);
    for (uint32_t i = 0; i < kN; ++i)
      for (uint32_t j = 0; j < kDim; ++j) f.data[size_t(i) * kDim + j] = centers[(i % 50) * kDim + j] + 0.3f * N(rng);
    f.queries.resize(size_t(kQ) * kDim);
    for (auto& v : f.queries) v = N(rng);
    for (uint32_t i = 0; i < kQ; ++i)
      for (uint32_t j = 0; j < kDim; ++j) f.queries[size_t(i) * kDim + j] += centers[(i % 50) * kDim + j];

    f.cfg.metric = "cd"; f.cfg.dim = kDim; f.cfg.M = kM; f.cfg.k = kK; f.cfg.target_recall = 0.95f;
    f.cfg.index_path = (f.dir / "index.bin").string();
    f.cfg.estimator_path = (f.dir / "estimator.bin").string();
    f.cfg.adapter_path = (f.dir / "adapter.bin").string();
    build_index(f.data.data(), kN, kDim, "cd", kM, 100, 4, f.cfg.index_path);
    compute_estimator_cpu(f.data.data(), kN, kDim, "cd", f.cfg.estimator_path);
    // calibration samplings: 200 base rows (authors' base_sample default) + exact GT
    hnswdis::MatrixXf Q(200, kDim);
    hnswdis::MatrixXi GT(200, kK);
    for (int i = 0; i < 200; ++i) {
      const float* v = &f.data[size_t(i * 29) * kDim];
      std::copy(v, v + kDim, Q.row(i).data());
      auto ids = brute_knn_cd(f.data, v, kN, kDim, kK);
      for (uint32_t j = 0; j < kK; ++j) GT(i, j) = static_cast<int>(ids[j]);
    }
    const auto samp = (f.dir / "samplings.bin").string();
    hnswdis::serialize_samplings(samp, Q, GT);
    build_ef_adapter(f.cfg, samp, 5000, f.cfg.adapter_path);
    return f;
  }();
  return F;
}

struct Reference {  // the authors' objects, loaded from the same files
  hnswlib::InnerProductSpace space{kDim};
  std::unique_ptr<HNSW> h;
  std::shared_ptr<hnswdis::Estimator> est;
  std::unique_ptr<hnswdis::ApproximatedScoreCalculator> sc;
  std::unique_ptr<hnswdis::EfAdapter> ad;
  std::unique_ptr<hnswdis::Sketch> sk;
  explicit Reference(const BackendConfig& c) {
    h = std::make_unique<HNSW>(&space, c.index_path);
    est = hnswdis::load_estimator_from_file(c.estimator_path);
    sc = std::make_unique<hnswdis::ApproximatedScoreCalculator>(est, c.quantile_step);
    ad = std::make_unique<hnswdis::EfAdapter>(c.adapter_path);
    sk = std::make_unique<hnswdis::Sketch>(ad->get_ef_recall_estimators(), c.target_recall);
  }
};

std::vector<uint32_t> labels_closest_first(std::priority_queue<std::pair<float, labeltype>> pq) {
  std::vector<uint32_t> v(pq.size());
  for (size_t i = v.size(); i-- > 0;) { v[i] = static_cast<uint32_t>(pq.top().second); pq.pop(); }
  return v;
}
}  // namespace

TEST(BackendParity, PlannedSearchMatchesAuthorsAdaptiveSearch) {
  const auto& F = fixture();
  auto be = AnnBackend::open(F.cfg);
  Reference R(F.cfg);
  R.h->setEf(kK);  // authors' floor: ef_copy = max(ef_, k)
  const size_t sl = default_statics_length(kM);
  int with_probe = 0;
  for (uint32_t i = 0; i < kQ; ++i) {
    const float* q = &F.queries[size_t(i) * kDim];
    std::vector<float> qn(kDim);
    normalize_into(q, qn.data(), kDim);
    auto ref = R.h->adaptiveSearchKnn(qn.data(), kK, sl, *R.sc, R.sk.get());
    auto ref_ids = labels_closest_first(ref.first);
    std::vector<uint32_t> ids(kK);
    SearchStats st;
    be->search_planned(q, kK, kProbeLen, [](float, uint32_t ef_pref, uint32_t) { return ef_pref; }, ids.data(), &st);
    ids.resize(ref_ids.size());
    ASSERT_EQ(ids, ref_ids) << "query " << i;
    if (st.probe_complete) {
      ++with_probe;
      EXPECT_FLOAT_EQ(st.score, ref.second);
      EXPECT_GE(st.ndis, st.probe_ndis);  // often equal at small ef: the probe already covers the search
      EXPECT_GE(st.probe_ndis, sl);
      EXPECT_GE(st.ef_pref, kK);
    }
  }
  EXPECT_GT(with_probe, int(kQ * 0.9));
}

TEST(BackendParity, FixedSearchMatchesAuthorsSearchKnn) {
  const auto& F = fixture();
  auto be = AnnBackend::open(F.cfg);
  Reference R(F.cfg);
  for (uint32_t ef : {10u, 40u, 160u}) {
    R.h->setEf(ef);
    for (uint32_t i = 0; i < kQ; ++i) {
      const float* q = &F.queries[size_t(i) * kDim];
      std::vector<float> qn(kDim);
      normalize_into(q, qn.data(), kDim);
      auto ref_ids = labels_closest_first(R.h->searchKnn(qn.data(), kK));
      std::vector<uint32_t> ids(kK);
      SearchStats st;
      be->search_fixed(q, kK, ef, ids.data(), &st);
      ASSERT_EQ(ids, ref_ids) << "ef " << ef << " query " << i;
      EXPECT_GT(st.ndis, 0u);
    }
  }
}

TEST(BackendParity, PlannerEfIsHonouredAndMonotone) {
  const auto& F = fixture();
  auto be = AnnBackend::open(F.cfg);
  const float* q = &F.queries[0];
  std::vector<uint32_t> ids(kK);
  SearchStats lo, hi;
  be->search_planned(q, kK, kProbeLen, [](float, uint32_t, uint32_t) { return 1u; }, ids.data(), &lo);  // clamped to k
  be->search_planned(q, kK, kProbeLen, [](float, uint32_t, uint32_t) { return 400u; }, ids.data(), &hi);
  ASSERT_TRUE(lo.probe_complete && hi.probe_complete);
  EXPECT_EQ(lo.ef_used, kK);
  EXPECT_EQ(hi.ef_used, 400u);
  EXPECT_EQ(lo.probe_ndis, hi.probe_ndis);  // the probe is identical regardless of the plan
  EXPECT_LE(lo.ndis, hi.ndis);
}

TEST(BackendParity, ConcurrentSearchesAreDeterministic) {
  const auto& F = fixture();
  auto be = AnnBackend::open(F.cfg);
  auto plan = [](float, uint32_t ef_pref, uint32_t) { return ef_pref; };
  std::vector<uint32_t> serial(size_t(kQ) * kK), par(size_t(kQ) * kK);
  std::vector<uint32_t> nd_s(kQ), nd_p(kQ);
  for (uint32_t i = 0; i < kQ; ++i) {
    SearchStats st;
    be->search_planned(&F.queries[size_t(i) * kDim], kK, kProbeLen, plan, &serial[size_t(i) * kK], &st);
    nd_s[i] = st.ndis;
  }
  std::vector<std::thread> th;
  std::atomic<uint32_t> next{0};
  for (int t = 0; t < 8; ++t)
    th.emplace_back([&] {
      for (uint32_t i; (i = next++) < kQ;) {
        SearchStats st;
        be->search_planned(&F.queries[size_t(i) * kDim], kK, kProbeLen, plan, &par[size_t(i) * kK], &st);
        nd_p[i] = st.ndis;
      }
    });
  for (auto& t : th) t.join();
  EXPECT_EQ(serial, par);
  EXPECT_EQ(nd_s, nd_p);
}

// Python-written fixtures (gpu_preprocessing/tests/test_formats.py) read with the AUTHORS' deserializers.
TEST(AdaEfFormats, PythonFixturesReadByAuthorsCode) {
  const fs::path fx = fs::path(LAEF_FIXTURES_DIR);
  if (!fs::exists(fx / "samplings_3x4_k2.bin")) GTEST_SKIP() << "run pytest gpu_preprocessing/tests first";
  hnswdis::MatrixXf Q;
  hnswdis::MatrixXi GT;
  hnswdis::deserialize_samplings((fx / "samplings_3x4_k2.bin").string(), Q, GT);
  ASSERT_EQ(Q.rows(), 3); ASSERT_EQ(Q.cols(), 4); ASSERT_EQ(GT.cols(), 2);
  EXPECT_FLOAT_EQ(Q(1, 0), 2.0f);  // 0.5 * 4 (row-major)
  EXPECT_EQ(GT(2, 1), 105);
  auto est = hnswdis::load_estimator_from_file((fx / "estimator_cd_d3.bin").string());
  EXPECT_EQ(est->get_name(), "CosineDistanceEstimator");
  hnswdis::RowVectorXf q(3);
  q << 1, 0, 0;
  auto [mean, var] = est->get_practical_distribution(q);
  EXPECT_NEAR(mean, 1 - 0.1f, 1e-6);  // cosine distance mean = 1 - q.mu
  EXPECT_NEAR(var, 1.0f, 1e-6);       // q^T Sigma q
}
