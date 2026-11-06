#pragma once
// LA-ef: Step A tables + price list + water level. IMPLEMENTATION_PLAN.md §9.2-§9.5, Main Flow.md §1/§5.
// Header-only algorithms (no JSON here). JSON load/save lives in src/price_list.cpp.
//
// Invariants after finalize():
//   * each group's kept rungs are ascending in ef, strictly ascending in work, non-decreasing in
//     recall, and lie on the upper concave envelope of (work, recall)  =>  per-group loss ratios
//     are non-decreasing as ef goes DOWN, so greedy cutting is optimal.
//   * a group's ladder may start above k if the k rung was dominated (same work as a higher rung).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace laef {

constexpr int kMaxScore = 100;  // Ada-ef score keys are int(score) in [0, 100]

// The cut policy that runs after the probe. The probe menu uses it to model the work and recall of each probe length.
enum class CutKind { Passthrough, WaterFilling, UniformWork, UniformEf };

// Model prediction for one query at a given water budget: expected work [dc] and recall.
struct ExpectedPoint { double work = 0, recall = 0; };

struct GroupCurve {
  int id = 0;
  int score_lo = 0, score_hi = kMaxScore;  // inclusive integer score-key range
  double p = 0;                            // share of calibration queries
  double ef_pref_median = 0;               // median Ada-ef ef of the group (expected-demand only)
  std::vector<uint32_t> ladder;            // kept rungs (ascending)
  std::vector<double> recall, work, service_us;
};

struct CutStep {          // one cut  from -> to  (from > to)
  int group = 0;
  double from = 0, to = 0, d_recall = 0, d_work = 0, loss = 0;
};

// ---------- calibration-side builders (used by laef_calibrate, unit-tested) ----------

// Running max on recall and work, drop rungs with equal work (keep the higher-ef one),
// then keep only the upper concave envelope of (work, recall). Endpoints are always kept.
inline void envelope_inplace(std::vector<uint32_t>& ef, std::vector<double>& recall,
                             std::vector<double>& work, std::vector<double>* svc = nullptr) {
  const size_t n = ef.size();
  if (n == 0 || recall.size() != n || work.size() != n) throw std::invalid_argument("envelope: size mismatch");
  for (size_t i = 1; i < n; ++i) {
    if (ef[i] <= ef[i - 1]) throw std::invalid_argument("envelope: ef must be strictly ascending");
    recall[i] = std::max(recall[i], recall[i - 1]);
    work[i] = std::max(work[i], work[i - 1]);
  }
  std::vector<size_t> keep;
  for (size_t i = 0; i < n; ++i) {
    if (!keep.empty() && work[i] <= work[keep.back()]) keep.back() = i;  // equal work: higher ef dominates
    else keep.push_back(i);
  }
  std::vector<size_t> hull;
  for (size_t i : keep) {
    while (hull.size() >= 2) {
      const size_t a = hull[hull.size() - 2], b = hull.back();
      // pop b if it is on/below the chord a->i  (cross >= 0)
      const double cross = (work[b] - work[a]) * (recall[i] - recall[a]) - (recall[b] - recall[a]) * (work[i] - work[a]);
      if (cross >= 0) hull.pop_back(); else break;
    }
    hull.push_back(i);
  }
  auto pick = [&](auto& v) { std::decay_t<decltype(v)> o; for (size_t i : hull) o.push_back(v[i]); v.swap(o); };
  pick(ef); pick(recall); pick(work);
  if (svc && svc->size() == n) pick(*svc);
}

// Greedy grouping of integer score keys (§9.2). counts[s] = #calibration queries with int(score)==s.
// Returns inclusive [lo, hi] ranges covering 0..kMaxScore without gaps.
inline std::vector<std::pair<int, int>> make_groups(const std::vector<size_t>& counts, size_t min_size, size_t max_groups) {
  struct B { int lo, hi; size_t c; };
  std::vector<B> b;
  for (int s = 0; s < static_cast<int>(counts.size()) && s <= kMaxScore; ++s)
    if (counts[s] > 0) b.push_back({s, s, counts[s]});
  if (b.empty()) return {{0, kMaxScore}};
  auto merge = [&](size_t i) { b[i].hi = b[i + 1].hi; b[i].c += b[i + 1].c; b.erase(b.begin() + i + 1); };
  for (;;) {  // every bucket must reach min_size: merge the smallest undersized bucket into its smaller neighbour
    size_t worst = b.size();
    for (size_t i = 0; i < b.size(); ++i)
      if (b[i].c < min_size && (worst == b.size() || b[i].c < b[worst].c)) worst = i;
    if (worst == b.size() || b.size() == 1) break;
    if (worst == 0) merge(0);
    else if (worst == b.size() - 1) merge(worst - 1);
    else if (b[worst - 1].c <= b[worst + 1].c) merge(worst - 1);
    else merge(worst);
  }
  while (b.size() > std::max<size_t>(1, max_groups)) {  // merge the adjacent pair with the smallest total
    size_t best = 0;
    for (size_t i = 1; i + 1 < b.size(); ++i)
      if (b[i].c + b[i + 1].c < b[best].c + b[best + 1].c) best = i;
    merge(best);
  }
  std::vector<std::pair<int, int>> r;
  for (size_t i = 0; i < b.size(); ++i)
    r.push_back({i == 0 ? 0 : r.back().second + 1, i + 1 == b.size() ? kMaxScore : b[i + 1].lo - 1});
  return r;
}

// ---------- runtime price list ----------

class PriceList {
 public:
  uint32_t k = 10;
  double floor_work_mean_measured = 0;    // c(k) from the sweep (JSON floor_work_mean)
  double nominal_work_mean_measured = 0;  // mean ndis at ef_pref (JSON nominal_work_mean)
  uint32_t static_ef_for_target = 0;
  std::vector<GroupCurve> groups;

  // Call after filling `groups` (ladders must already be enveloped). Builds sorted cut list.
  void finalize() {
    if (groups.empty()) throw std::invalid_argument("PriceList: no groups");
    lo_.clear();
    for (size_t g = 0; g < groups.size(); ++g) {
      auto& G = groups[g];
      if (G.ladder.empty() || G.recall.size() != G.ladder.size() || G.work.size() != G.ladder.size())
        throw std::invalid_argument("PriceList: bad group curve");
      if (g > 0 && G.score_lo != groups[g - 1].score_hi + 1) throw std::invalid_argument("PriceList: score ranges must be contiguous");
      lo_.push_back(G.score_lo);
    }
    steps_.clear();
    model_nominal_ = model_floor_ = model_nominal_recall_ = model_floor_recall_ = 0;
    for (size_t g = 0; g < groups.size(); ++g) {
      const auto& G = groups[g];
      model_nominal_ += G.p * work(static_cast<int>(g), G.ef_pref_median);
      model_nominal_recall_ += G.p * recall(static_cast<int>(g), G.ef_pref_median);
      model_floor_ += G.p * floor_work(static_cast<int>(g));
      model_floor_recall_ += G.p * G.recall.front();  // floor rung = lowest kept rung
      for (auto& s : steps_from(static_cast<int>(g), G.ef_pref_median)) steps_.push_back(s);
    }
    std::stable_sort(steps_.begin(), steps_.end(), [](const CutStep& a, const CutStep& b) { return a.loss < b.loss; });
  }

  int group_of(float score) const {
    int s = static_cast<int>(score);  // Ada-ef uses static_cast<int>(score) as the key
    s = std::clamp(s, 0, kMaxScore);
    auto it = std::upper_bound(lo_.begin(), lo_.end(), s);
    return static_cast<int>(std::max<std::ptrdiff_t>(0, (it - lo_.begin()) - 1));
  }

  double work(int g, double ef) const { return interp(groups[g], groups[g].work, ef); }
  double recall(int g, double ef) const { return interp(groups[g], groups[g].recall, ef); }
  const std::vector<uint32_t>& rungs(int g) const { return groups[g].ladder; }
  double floor_work(int g) const { return groups[g].work.front(); }
  uint32_t floor_ef(int g) const { return std::max(k, groups[g].ladder.front()); }
  double model_nominal_work() const { return model_nominal_; }   // sum_g p_g c_g(e*_g)
  double model_floor_work() const { return model_floor_; }       // sum_g p_g c_g(floor)
  double model_nominal_recall() const { return model_nominal_recall_; }  // sum_g p_g r_g(e*_g)
  double model_floor_recall() const { return model_floor_recall_; }      // sum_g p_g r_g(floor)
  const std::vector<CutStep>& steps_sorted() const { return steps_; }

  // Cuts available to a query of group g currently at `ef_top` (descending order of ef).
  // The first step may start off-ladder (ef_top between rungs); its values are interpolated,
  // so it lies on the envelope segment and has that segment's loss ratio.
  std::vector<CutStep> steps_from(int g, double ef_top) const {
    std::vector<CutStep> out;
    const auto& L = groups[g].ladder;
    double cur = ef_top;
    for (size_t i = L.size(); i-- > 0;) {
      if (L[i] >= cur) continue;
      CutStep s;
      s.group = g; s.from = cur; s.to = L[i];
      s.d_work = work(g, cur) - work(g, L[i]);
      s.d_recall = recall(g, cur) - recall(g, L[i]);
      if (s.d_work <= 0) { cur = L[i]; continue; }  // free/degenerate step: treat as no-op
      s.loss = std::max(0.0, s.d_recall) / s.d_work;
      out.push_back(s);
      cur = L[i];
    }
    return out;
  }

  // Water level (Main Flow §5): smallest loss lambda* such that taking every cut with loss <= lambda*
  // brings the expected per-query demand down to w_target. 0 => no cuts; +inf => all at floor.
  double water_level(double w_target) const {
    if (model_nominal_ <= w_target) return 0.0;
    double saved = 0;
    for (const auto& s : steps_) {
      saved += groups[s.group].p * s.d_work;
      if (model_nominal_ - saved <= w_target) return s.loss;
    }
    return std::numeric_limits<double>::infinity();
  }

  // Expected per-query work and recall under one cut policy at water budget w_target (plan §1A.5). It uses the
  // group shares p_g and the group medians of ef_pref, and mirrors the allocators: water filling takes the cuts in
  // loss order; uniform cut gives every group the same fraction of its work (or of its ef). Work includes the probe.
  ExpectedPoint expected_at(double w_target, CutKind kind) const {
    if (kind == CutKind::Passthrough || model_nominal_ <= w_target) return {model_nominal_, model_nominal_recall_};
    if (kind == CutKind::WaterFilling) {
      double saved = 0, lost = 0;
      for (const auto& s : steps_) {
        saved += groups[s.group].p * s.d_work;
        lost += groups[s.group].p * s.d_recall;
        if (model_nominal_ - saved <= w_target) return {model_nominal_ - saved, model_nominal_recall_ - lost};
      }
      return {model_floor_, model_floor_recall_};  // every cut taken: all groups at their floor rung
    }
    // Uniform cut: f = w_target / nominal. A group keeps its preferred ef if that fits, else the largest lower rung
    // that fits, else its floor rung (the same order as UniformCutAllocator).
    const double f = std::clamp(w_target / model_nominal_, 0.0, 1.0);
    ExpectedPoint out;
    for (size_t g = 0; g < groups.size(); ++g) {
      const GroupCurve& G = groups[g];
      const int gi = static_cast<int>(g);
      const double e0 = G.ef_pref_median;
      auto fits = [&](double e) {
        return kind == CutKind::UniformWork ? work(gi, e) <= f * work(gi, e0) + 1e-9
                                            : e <= std::max<double>(k, f * e0) + 1e-9;
      };
      double pick = G.ladder.front();
      if (fits(e0)) {
        pick = e0;
      } else {
        for (size_t i = G.ladder.size(); i-- > 0;)
          if (G.ladder[i] < e0 && fits(G.ladder[i])) { pick = G.ladder[i]; break; }
      }
      out.work += G.p * work(gi, pick);
      out.recall += G.p * recall(gi, pick);
    }
    return out;
  }

 private:
  static double interp(const GroupCurve& G, const std::vector<double>& y, double ef) {
    const auto& L = G.ladder;
    if (ef <= L.front()) return y.front();
    if (ef >= L.back()) return y.back();
    const size_t j = static_cast<size_t>(std::upper_bound(L.begin(), L.end(), ef) - L.begin());  // L[j-1] < ef < L[j]
    const double t = (ef - L[j - 1]) / double(L[j] - L[j - 1]);
    return y[j - 1] + t * (y[j] - y[j - 1]);
  }

  std::vector<int> lo_;
  std::vector<CutStep> steps_;
  double model_nominal_ = 0, model_floor_ = 0;
  double model_nominal_recall_ = 0, model_floor_recall_ = 0;
};

}  // namespace laef
