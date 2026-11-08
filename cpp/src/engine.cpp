// LA-ef: engine (IMPLEMENTATION_PLAN.md §11.2, §1A.7). Generator thread, C worker threads, controller thread.
//
// Per query: the generator reserves the snapshot's expected work at arrival and retains the amount actually charged
// after the -D clamp. The worker first chooses the probe length from the current snapshot (stage 1), then runs the
// backend search. When that probe completes, the planner hook settles the reservation, picks ef and debits planned
// work (stage 2). Incomplete probes settle their measured work instead.
#include "laef/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

#include "laef/ann_backend.h"
#include "laef/allocators/allocator_factory.h"
#include "laef/allocators/passthrough.h"
#include "laef/controllers/budget_controller.h"
#include "laef/controllers/controller_factory.h"
#include "laef/plan_snapshot.h"
#include "laef/planner.h"
#include "laef/price_list.h"
#include "laef/probe_menu.h"
#include "laef/sensor_window.h"
#include "laef/telemetry.h"
#include "laef/token_bucket.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace laef {
namespace {

// The dispatcher queue between the generator and the workers. Its depth Q is read by the controller.
class TaskQueue {
 public:
  struct Task {
    uint32_t qid = 0;
    uint32_t eval_idx = 0;
    double reservation = 0;  // amount actually charged for the nominal reservation [dc]
  };

  void push(const Task& t) {
    {
      std::lock_guard<std::mutex> lk(m_);
      q_.push_back(t);
      depth_.store(q_.size());
    }
    cv_.notify_one();
  }

  // Blocks until a task is available. Returns false once the queue is closed and empty.
  bool pop(Task& t) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    t = q_.front();
    q_.pop_front();
    depth_.store(q_.size());
    return true;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lk(m_);
      closed_ = true;
    }
    cv_.notify_all();
  }

  size_t depth() const { return depth_.load(); }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Task> q_;
  bool closed_ = false;
  std::atomic<size_t> depth_{0};
};

struct Shared {
  std::atomic<uint64_t> arrived{0};
  std::atomic<uint64_t> completed{0};
  std::atomic<bool> stop_controller{false};
  std::atomic<bool> failed{false};
  std::mutex error_mu;
  std::exception_ptr error;

  void fail(std::exception_ptr e) {
    std::lock_guard<std::mutex> lk(error_mu);
    if (!error) error = e;
    failed = true;
  }
};

// Pins the calling thread to one core (Linux). Returns false where pinning is unavailable.
bool pin_current_thread(int core) {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
  (void)core;
  return false;
#endif
}

double p99_of(std::vector<double> v) {
  return v.empty() ? 0.0 : nearest_rank(v, 0.99);
}

}  // namespace

EngineResult run_engine(const EngineInputs& in) {
  if (!in.cfg || !in.backend || !in.probe_menu || in.probe_menu->size() == 0)
    throw std::invalid_argument("run_engine: cfg, backend and a non-empty probe_menu are required");
  const RunConfig& cfg = *in.cfg;
  const ProbeMenu& menu = *in.probe_menu;
  const PriceList& pl = menu.full().price_list;  // static ef target; the full probe's tables
  const uint32_t k = cfg.k;
  const size_t n = in.arrivals_ns.size();
  if (n == 0 || in.eval_idx.size() != n) throw std::invalid_argument("run_engine: arrivals and eval_idx must match and be non-empty");
  if (in.threads < 1) throw std::invalid_argument("run_engine: threads must be >= 1");
  if (!(in.t_slo_s > 0)) throw std::invalid_argument("run_engine: T_SLO must be positive");
  if (in.gt && in.gt_k < k) throw std::invalid_argument("run_engine: ground truth is narrower than k");

  // static_ef: fixed ef, plain HNSW. Otherwise the planner runs; without a controller (Ada-ef) it is passthrough.
  const bool fixed = cfg.allocator.type == "static_ef";
  const bool controlled = !fixed && cfg.controller.type != "none";
  std::unique_ptr<Allocator> alloc = controlled ? make_allocator(cfg.allocator, pl) : std::make_unique<PassthroughAllocator>();
  std::unique_ptr<BudgetController> ctrl = controlled ? make_controller(cfg.controller) : nullptr;
  if (controlled && !ctrl) throw std::invalid_argument("run_engine: controller type resolved to none");
  const CutKind kind = controlled ? cut_kind(cfg.allocator) : CutKind::Passthrough;
  const TBase tbase = cfg.bucket.t_base == "tail" ? TBase::Tail : TBase::Mean;
  const double t_slo = in.t_slo_s;

  // Initial plan: B = B_max = C / u (BB §1 start), so the first window starts from the profiled capacity.
  // The initial snapshot always runs the full probe (stock Ada-ef) until the first controller window.
  const double B_max0 = in.u_s_per_dc > 0 ? in.threads / in.u_s_per_dc : 0.0;
  ControllerOutput init;
  init.B = init.B_max = B_max0;
  init.zone = "init";
  if (ctrl) ctrl->reset(B_max0);
  SnapshotStore snaps(make_snapshot(init, menu, kind, 0.0, in.mean_service_s, in.s_p95_s, t_slo, tbase, 0));
  TokenBucket bucket(cfg.bucket.cap_pos);
  Planner planner(menu, *alloc, bucket, snaps, k, cfg.floor_mode.enabled);

  Telemetry tel(cfg.telemetry.alpha_u, cfg.telemetry.alpha_es, cfg.telemetry.alpha_lambda, in.u_s_per_dc,
                in.mean_service_s);
  SensorWindow window(cfg.window.n, cfg.window.max_ms, cfg.window.min_samples);

  EngineResult out;
  out.records.resize(n);
  out.result_ids.assign(n * k, UINT32_MAX);
  std::vector<double> lag_us;     // written by the generator only
  std::vector<double> ctrl_us;    // written by the controller only
  lag_us.reserve(n);

  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  if (in.pin_cores && static_cast<unsigned>(in.threads + 2) > hw)
    std::fprintf(stderr, "warning: --pin-cores with %d threads needs %d cores; this machine has %u\n", in.threads,
                 in.threads + 2, hw);
  auto core_for = [hw](int role) { return static_cast<int>(static_cast<unsigned>(role) % hw); };

  TaskQueue queue;
  Shared sh;
  const int64_t t0 = now_ns();  // run start: all record times are relative to this

  auto generator = [&] {
    try {
      if (in.pin_cores) pin_current_thread(core_for(0));
      for (size_t i = 0; i < n; ++i) {
        if (sh.failed) break;
        const int64_t sched = t0 + in.arrivals_ns[i];
        for (int64_t now = now_ns(); now < sched; now = now_ns()) {
          const int64_t gap = sched - now;
          if (gap > 200'000) std::this_thread::sleep_for(std::chrono::nanoseconds(gap - 100'000));
        }
        const int64_t actual = now_ns();
        lag_us.push_back((actual - sched) * 1e-3);

        QueryRecord& r = out.records[i];
        r.qid = static_cast<uint32_t>(i);
        r.eval_idx = in.eval_idx[i];
        r.warmup = i < in.warmup;
        r.t_arrival_ns = sched - t0;  // scheduled time, so a late generator does not hide latency

        QueryCtx qc;
        planner.on_arrival(qc, actual);  // reserves the expected work and returns the actual charged amount
        TaskQueue::Task task;
        task.qid = static_cast<uint32_t>(i);
        task.eval_idx = in.eval_idx[i];
        task.reservation = qc.reservation;
        sh.arrived.fetch_add(1);
        queue.push(task);
      }
    } catch (...) {
      sh.fail(std::current_exception());
    }
  };

  auto worker = [&](int idx) {
    try {
      if (in.pin_cores) pin_current_thread(core_for(2 + idx));
      TaskQueue::Task task;
      while (queue.pop(task)) {
        QueryRecord& r = out.records[task.qid];
        const float* q = in.queries + static_cast<size_t>(task.eval_idx) * in.dim;
        uint32_t* ids = out.result_ids.data() + static_cast<size_t>(task.qid) * k;
        SearchStats st{};
        QueryCtx qc;
        qc.reservation = task.reservation;

        // Stage 1 (§1A.7): the probe length comes from the snapshot that is current when the search starts.
        const uint32_t probe_len = fixed ? 0 : planner.choose_probe(qc);
        const int64_t t_start = now_ns();
        if (fixed) {
          in.backend->search_fixed(q, k, pl.static_ef_for_target, ids, &st);
        } else {
          // Stage 2: the hook runs when the probe of probe_len has completed.
          in.backend->search_planned(
              q, k, probe_len,
              [&](float score, uint32_t ef_pref, uint32_t) { return planner.on_probe(score, ef_pref, qc, now_ns()); },
              ids, &st);
          if (!st.probe_complete) planner.on_incomplete(qc, st.ndis, now_ns());
        }
        const int64_t t_end = now_ns();

        r.t_start_ns = t_start - t0;
        r.t_end_ns = t_end - t0;
        r.status = (fixed || st.probe_complete) ? QueryStatus::Ok : QueryStatus::ProbeIncomplete;
        r.score = fixed ? -1.f : st.score;
        r.group = qc.group;  // -1 when the planner was not called (static ef, probe incomplete)
        r.ef_pref = fixed ? pl.static_ef_for_target : st.ef_pref;
        r.ef_used = fixed ? pl.static_ef_for_target : st.ef_used;
        r.ndis = st.ndis;
        r.probe_ndis = st.probe_ndis;
        r.probe_len = probe_len;  // 0 for static ef (no probe)
        r.forced_floor = qc.forced_floor;
        r.floor_mode = qc.floor_mode;
        r.planned_work = qc.planned_work;

        tel.on_completion((t_end - t_start) * 1e-9, st.ndis);
        window.push(r.latency_s(), r.service_s());
        sh.completed.fetch_add(1);
      }
    } catch (...) {
      sh.fail(std::current_exception());
    }
  };

  auto controller = [&] {
    try {
      if (in.pin_cores) pin_current_thread(core_for(1));
      const int64_t tick_ns = static_cast<int64_t>(cfg.telemetry.lambda_tick_ms * 1e6);
      int64_t last_tick = now_ns();
      uint64_t last_arrived = 0;
      uint64_t version = 1;
      window.start(last_tick);
      while (!sh.stop_controller.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        const int64_t now = now_ns();
        if (now - last_tick >= tick_ns) {  // lambda-hat: arrivals per second over the last tick
          const uint64_t arrived = sh.arrived.load();
          tel.on_lambda_tick(static_cast<double>(arrived - last_arrived) * 1e9 / static_cast<double>(now - last_tick));
          last_arrived = arrived;
          last_tick = now;
        }
        if (!window.ready(now)) continue;

        const WindowStats ws = window.close(now);
        const Telemetry::View v = tel.view();
        TimeSeriesRow row;
        row.t_s = (now - t0) * 1e-9;
        row.p95_ms = ws.p95 * 1e3;
        row.s_p95_ms = ws.s_p95 * 1e3;
        row.Q = static_cast<double>(queue.depth());
        row.u_ns = v.unit_cost * 1e9;
        row.ES_ms = v.mean_service * 1e3;
        row.lambda_hat = v.arrival_rate;
        row.slo_violated = ws.p95 > t_slo;

        if (ctrl) {
          ControllerInputs ci;
          ci.p95 = ws.p95;
          ci.s_p95 = ws.s_p95;
          ci.queue_depth = row.Q;
          ci.mean_service = v.mean_service;
          ci.unit_cost = v.unit_cost;
          ci.arrival_rate = v.arrival_rate;
          ci.floor_work = menu.full().price_list.model_floor_work();  // B_min: the full probe at ef = k
          ci.t_slo = t_slo;
          ci.C = in.threads;
          const int64_t c0 = now_ns();
          const ControllerOutput o = ctrl->update(ci);
          ctrl_us.push_back((now_ns() - c0) * 1e-3);
          // floor_mode.enabled = false (ablation): the snapshot never forces the floor probe; the controller's
          // own floor flag is still logged below.
          ControllerOutput planned = o;
          planned.floor_mode = o.floor_mode && cfg.floor_mode.enabled;
          const PlanSnapshot snap =
              make_snapshot(planned, menu, kind, v.arrival_rate, v.mean_service, ws.s_p95, t_slo, tbase, version++);
          snaps.store(snap);
          row.B = o.B;
          row.B_max = o.B_max;
          row.B_min = o.B_min;
          row.D = snap.D;
          row.W_hat_ms = o.W_hat * 1e3;
          row.w_target = snap.w_target;
          row.water_level = snap.water_level;
          row.uniform_fraction = snap.uniform_fraction;
          row.zone = o.zone;
          row.guard_fired = o.guard_fired;
          row.floor_mode = o.floor_mode;
        }
        row.probe_len = menu.at(snaps.load()->probe_idx).probe_len;
        out.timeseries.push_back(row);
      }
    } catch (...) {
      sh.fail(std::current_exception());
    }
  };

  std::thread gen_thread(generator);
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(in.threads));
  for (int w = 0; w < in.threads; ++w) workers.emplace_back(worker, w);
  std::thread ctrl_thread(controller);

  gen_thread.join();  // every arrival is now queued
  while (sh.completed.load() < n && !sh.failed.load()) std::this_thread::sleep_for(std::chrono::microseconds(200));
  queue.close();
  for (auto& w : workers) w.join();
  sh.stop_controller = true;
  ctrl_thread.join();
  if (sh.error) std::rethrow_exception(sh.error);

  if (in.gt) {
    for (size_t i = 0; i < n; ++i) {
      QueryRecord& r = out.records[i];
      r.recall = recall_at_k(out.result_ids.data() + i * k, in.gt + static_cast<size_t>(r.eval_idx) * in.gt_k, k);
    }
  }
  out.controller_update_us = ctrl_us.empty() ? 0.0
                                             : std::accumulate(ctrl_us.begin(), ctrl_us.end(), 0.0) /
                                                   static_cast<double>(ctrl_us.size());
  out.generator_lag_p99_us = p99_of(lag_us);
  return out;
}

}  // namespace laef
