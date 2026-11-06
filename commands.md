# Commands

Run every command from the `Implementation/` folder on WSL, in the order below. The design is `IMPLEMENTATION_PLAN.md` §1A (LA-ef v2: the probe length is the budget lever). Steps 1 onward assume the v2 code (checklist §17.1) is in place; `laef_calibrate --probe-lengths` is part of that change (§1A.3 C6).

## What already exists (checked 2026-11-04)

| Dataset | Data, GT, calibration samplings, stats, manifest | HNSW index | v1 tables + profile |
|---|---|---|---|
| synthetic-small | yes | yes | yes (stale) |
| lastfm-64-dot | yes | yes | yes (stale) |
| coco-i2i-512-angular | yes | yes | yes (stale) |
| coco-t2i-512-angular | yes | **missing** | none |
| glove-100-angular | yes | **missing** | none |
| cluster-zipf-100d | yes | **missing** | none |
| deep-image-96-angular | yes | **missing** | none |
| dbpedia-openai-1m | yes | **missing** | none |

- No Colab run and no `scripts/prepare_dataset.sh` are needed: every dataset already has its `.npy` files and Colab outputs.
- No setup script is needed: `build/` is already configured with the Ada-ef backend (Eigen and Boost from `/home/f1coder/laef-deps`), and the submodule and `.venv` exist.
- The v1 tables and profiles are stale. v2 writes per-probe-length tables, and the old profiles lack the `adaef` metadata that `laef_bench` validates.

## 1. Rebuild and run the minimal checks

```bash
cmake --build build -j 12
ctest --test-dir build -R "parity|price_list" --output-on-failure    # §1A.8 items 2-4
```

## 2. Build the five missing indexes

Largest last. deep-image needs ~10 GB RAM and takes the longest; dbpedia needs ~13 GB RAM.

```bash
for ID in coco-t2i-512-angular glove-100-angular cluster-zipf-100d deep-image-96-angular dbpedia-openai-1m; do
  build/bin/laef_build_index --dataset-dir "data/artifacts/$ID" --M 16 --efc 500 --threads "$(nproc)"
done
```

## 3. Remove the stale v1 tables and profiles

The indexes are kept.

```bash
for ID in synthetic-small lastfm-64-dot coco-i2i-512-angular; do
  rm -rf "data/artifacts/$ID/tables" "data/artifacts/$ID/profile"
done
```

## 4. Calibrate every dataset per probe length

Writes `tables/L{L}/` for every probe length and `tables/probe_menu.json`. It takes roughly four times as long as the v1 calibration (one adapter build and one sweep per probe length).

```bash
for ID in synthetic-small lastfm-64-dot coco-i2i-512-angular coco-t2i-512-angular \
          glove-100-angular cluster-zipf-100d deep-image-96-angular dbpedia-openai-1m; do
  build/bin/laef_calibrate --dataset-dir "data/artifacts/$ID" --M 16 --efc 500 \
    --probe-lengths 1025,512,256,128 --threads "$(nproc)"
done
```

## 5. Check the probe menus

Do this before profiling. A dataset where only `L=1025` is kept cannot benefit from the probe lever.

```bash
.venv/bin/python - <<'EOF'
import json, pathlib
for d in sorted(pathlib.Path("data/artifacts").iterdir()):
    p = d / "tables" / "probe_menu.json"
    if not p.exists():
        print(d.name, ": no probe_menu.json"); continue
    m = json.loads(p.read_text())
    print(d.name)
    for o in m["options"]:
        print(f"  L={o['probe_len']:5d}  work={o['model_nominal_work']:8.0f}  floor={o['model_floor_work']:8.0f}"
              f"  recall={o['model_nominal_recall']:.3f}  mean ef_pref={o['ef_pref_mean']:.1f}")
    for o in m.get("dropped", []):
        print(f"  dropped L={o['probe_len']}: {o['reason']}")
EOF
```

What to look for: at least two probe lengths kept, work falling as L falls, and plausible recall at each L.

## 6. Profile stock Ada-ef at 4 threads (120 s, as before)

The SLO study derives T_SLO from `saturation.s_p95_s`, so keep the 120 s saturation run that the v1 profiles used. Each dataset takes about 2–3 minutes.

```bash
for ID in synthetic-small lastfm-64-dot coco-i2i-512-angular coco-t2i-512-angular \
          glove-100-angular cluster-zipf-100d deep-image-96-angular dbpedia-openai-1m; do
  build/bin/laef_profile --dataset-dir "data/artifacts/$ID" --M 16 --efc 500 --threads 4 --sat-seconds 120
done
```

## 7. Short bench check on lastfm (§1A.8 item 6)

```bash
D=data/artifacts/lastfm-64-dot
MU=$(.venv/bin/python -c "import json;print(json.load(open('$D/profile/profile_C4.json'))['saturation']['mu_sat_qps'])")
SLO=$(.venv/bin/python -c "import json;print(0.9e3*json.load(open('$D/profile/profile_C4.json'))['saturation']['s_p95_s'])")
for RHO in 0.5 1.5; do
  QPS=$(.venv/bin/python -c "print($RHO*$MU)")
  for P in ada_ef laef_wf; do
    build/bin/laef_bench --dataset-dir "$D" --threads 4 --policy "$P" --slo-ms "$SLO" \
      --qps "$QPS" --duration-s 20 --out "results/check/lastfm_${P}_rho${RHO}"
  done
done
grep -H -E '"(mean_probe_len|frac_full_probe|mean_ndis|p95_ms|mean_recall)"' results/check/*/summary.json
```

Expected: at ρ = 0.5 the `laef_wf` run has `frac_full_probe = 1` and the same `mean_ndis` as `ada_ef`. At ρ = 1.5 it has `mean_probe_len < 1025` and a lower `mean_ndis` than `ada_ef`.

## 8. SLO study

The study runs `configs/experiments/slo_study_120s.yaml`, which covers synthetic-small, lastfm-64-dot and coco-i2i-512-angular. To include another dataset, copy one `ada_*`/`laef_*` section pair and change the dataset name. T_SLO is 0.95, 0.9, 0.8 and 0.75 times each dataset's profile p95. Results go to `results/slo_study_120s/` (`summary_all.csv`, `comparison.csv`, 1 s time bins, `run.log`). The v1 results are not comparable, so move them away first:

```bash
mv results/slo_study_120s results/slo_study_120s_v1 2>/dev/null || true
.venv/bin/python scripts/run_slo_study.py
```

To keep it running after the terminal closes, and to watch it with `tail -f slo_study.log`:

```bash
setsid nohup .venv/bin/python scripts/run_slo_study.py > slo_study.log 2>&1 &
```

## 9. Remote Machine Migration & Pipeline Execution

Use these steps to transfer the codebase to a new remote machine via SSH, configure the environment, compile the C++ binaries, build indexes for the remaining 5 datasets, and generate calibration tables and profiles.

### 9.1 Copying the Implementation Folder via SSH

Run this from your **local machine** (source). 

This command uses `-W` (whole-file streaming), disables gzip compression (`-z`) which heavily throttles CPU on uncompressible float vector `.npy` arrays, and selects the hardware-accelerated `aes128-gcm` cipher for maximum transfer throughput:

```bash
rsync -avP -W \
  -e "ssh -T -c aes128-gcm@openssh.com -o Compression=no" \
  --exclude='.venv' \
  --exclude='.pytest_cache' \
  --exclude='build' \
  --exclude='results' \
  --exclude='data/raw' \
  /mnt/c/users/rizvi/Documents/ECE-750/Project/Implementation/ \
  USER@REMOTE_IP:~/Implementation/
```

- **Resumable**: If the SSH connection drops, simply re-running the exact same command will automatically resume from where it stopped.
- **Transfer Size**: ~11 GB (skips `data/raw` archives, virtualenv, builds, and results).
- **Estimated Transfer Time**: ~1.5–2 minutes on 1 Gbps LAN; ~10–15 minutes on 100 Mbps broadband.

---

### 9.2 Rootless Toolchain & Dependencies Setup (No Sudo Required)

SSH into your remote machine:
```bash
ssh USER@REMOTE_IP
cd ~/Implementation
```

#### Step A: Verify Compiler & CMake
Check if `g++` (>= 11) and `cmake` (>= 3.22) are already available:
```bash
g++ --version
cmake --version
```
*(On university / HPC clusters with environment modules, you may simply need `module load gcc/11 cmake`)*.

If `cmake` is missing or older than 3.22, install the official precompiled binary directly into `~/.local` in 10 seconds (no root needed):
```bash
mkdir -p ~/.local
curl -sSL https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz | tar -xz -C ~/.local --strip-components=1
export PATH="$HOME/.local/bin:$PATH"
```

#### Step B: Install Header-Only Dependencies (Eigen3 & Boost) without Sudo
Ada-ef only needs the C++ header files for Eigen3 and Boost.

- **Option 1 (Fastest - Copy directly from this machine in 10s)**:
  Run this from your local machine terminal:
  ```bash
  rsync -avP /home/f1coder/laef-deps USER@REMOTE_IP:~/
  ```

- **Option 2 (Download directly on the remote machine)**:
  Run this on the remote machine (uses the active official Boost archive):
  ```bash
  mkdir -p ~/laef-deps && cd ~/laef-deps

  # 1. Eigen 3.4 headers
  curl -sSL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar -xz

  # 2. Boost headers (official active mirror)
  curl -sSL https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz | tar -xz --wildcards "boost_1_82_0/boost"
  ```

---

### 9.3 Setup Python Virtual Environment (No Sudo Required)

Using `uv` (installs cleanly inside `~/.local/bin`, fetches standalone Python 3.11 automatically if needed without root):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/Implementation
uv venv --python 3.11 .venv
uv pip install -p .venv numpy h5py pyyaml pytest pandas matplotlib
```

---

### 9.4 Build the C++ Engine & Run Tests

Point CMake to your header directory:

```bash
cd ~/Implementation

# If you copied ~/laef-deps (Option 1):
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release \
  -DLAEF_EIGEN_INCLUDE="$HOME/laef-deps/usr/include/eigen3" \
  -DLAEF_BOOST_INCLUDE="$HOME/laef-deps/usr/include"

# If you downloaded directly on the remote machine (Option 2):
# cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release \
#   -DLAEF_EIGEN_INCLUDE="$HOME/laef-deps/eigen-3.4.0" \
#   -DLAEF_BOOST_INCLUDE="$HOME/laef-deps/boost_1_82_0"

cmake --build build -j "$(nproc)"

# Run backend parity and price list unit tests
ctest --test-dir build -R "parity|price_list" --output-on-failure
```
*(Note: If your cluster already has system Eigen3/Boost installed, standard `cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release` works directly).*

---

### 9.5 Build Indexes for Remaining Datasets

The 3 datasets already indexed (`coco-i2i-512-angular`, `lastfm-64-dot`, `synthetic-small`) are transferred ready to use. Build the indexes for the remaining 5 datasets (ordered smallest to largest; note that `deep-image` requires ~10 GB RAM, `dbpedia` requires ~13 GB RAM):

```bash
for ID in coco-t2i-512-angular glove-100-angular cluster-zipf-100d deep-image-96-angular dbpedia-openai-1m; do
  echo "=== Building Index: $ID ==="
  build/bin/laef_build_index --dataset-dir "data/artifacts/$ID" --M 16 --efc 500 --threads "$(nproc)"
done
```

---

### 9.6 Calibrate & Generate Water-Filling Tables

Generate per-probe-length calibration tables (`tables/L{L}/` and `tables/probe_menu.json`) for the 5 newly indexed datasets:

```bash
for ID in coco-t2i-512-angular glove-100-angular cluster-zipf-100d deep-image-96-angular dbpedia-openai-1m; do
  echo "=== Calibrating: $ID ==="
  build/bin/laef_calibrate --dataset-dir "data/artifacts/$ID" --M 16 --efc 500 \
    --probe-lengths 1025,512,256,128 --threads "$(nproc)"
done
```

Verify that the probe menus were generated properly:
```bash
.venv/bin/python - <<'EOF'
import json, pathlib
for d in sorted(pathlib.Path("data/artifacts").iterdir()):
    p = d / "tables" / "probe_menu.json"
    if not p.exists():
        continue
    m = json.loads(p.read_text())
    print(f"[{d.name}]")
    for o in m["options"]:
        print(f"  L={o['probe_len']:5d}  work={o['model_nominal_work']:8.0f}  floor={o['model_floor_work']:8.0f}"
              f"  recall={o['model_nominal_recall']:.3f}  mean ef_pref={o['ef_pref_mean']:.1f}")
    for o in m.get("dropped", []):
        print(f"  dropped L={o['probe_len']}: {o['reason']}")
EOF
```

---

### 9.7 Generate Latency Profiles

Measure baseline saturation and idle latency at 4 threads (120 s saturation run to establish baseline $P_{95}$ latency and capacity $\mu_{sat}$):

```bash
for ID in coco-t2i-512-angular glove-100-angular cluster-zipf-100d deep-image-96-angular dbpedia-openai-1m; do
  echo "=== Profiling: $ID ==="
  build/bin/laef_profile --dataset-dir "data/artifacts/$ID" --M 16 --efc 500 --threads 4 --sat-seconds 30
done
```

---

### 9.8 Running the SLO Study on the Larger Datasets

Once the indexes, tables, and profiles exist for the larger datasets, you can run the SLO study using the new configuration [`configs/experiments/slo_study_large_120s.yaml`](configs/experiments/slo_study_large_120s.yaml).

By default, `scripts/run_slo_study.py` now runs `slo_study_large_120s.yaml`:

```bash
# Launch in background with persistent logs (runs slo_study_large_120s.yaml by default):
setsid nohup .venv/bin/python scripts/run_slo_study.py > slo_study.log 2>&1 &

# Monitor execution:
tail -f slo_study.log
```

*(Note: To run the original 3-dataset study instead, simply pass `--config configs/experiments/slo_study_120s.yaml`).*


