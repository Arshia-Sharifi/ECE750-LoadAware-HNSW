#!/usr/bin/env bash
# LA-ef: configure, build and test the C++ tree (IMPLEMENTATION_PLAN.md §6 P0.3).
# Usage: bash scripts/build.sh [extra cmake options, e.g. -DLAEF_WITH_ADAEF=OFF]
# Environment: JOBS (parallel build jobs, default: all cores), BUILD_TYPE (default: Release).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${HERE}/build"
JOBS="${JOBS:-$(nproc)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

cmake -S "${HERE}/cpp" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "$@"
cmake --build "${BUILD_DIR}" -j "${JOBS}"
ctest --test-dir "${BUILD_DIR}" --output-on-failure
