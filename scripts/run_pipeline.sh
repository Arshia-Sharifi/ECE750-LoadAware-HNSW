#!/usr/bin/env bash
# LA-ef: build_index -> calibrate -> profile for one dataset and one thread count (IMPLEMENTATION_PLAN.md §12 P6.4, §1A.3 C12).
# Usage: bash scripts/run_pipeline.sh <dataset-id> <threads>
# Environment: PROBE_LENGTHS (default 1025,512,256,128) is the probe menu, descending; its first value is the full probe.
# A step is skipped when its output already exists. FORCE=1 reruns all three steps.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <dataset-id> <threads>" >&2
  exit 2
fi
ID="$1"
C="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${HERE}/build/bin"
DIR="${HERE}/data/artifacts/${ID}"
M=16
EFC=500
PROBE_LENGTHS="${PROBE_LENGTHS:-1025,512,256,128}"
FORCE="${FORCE:-0}"

if [ ! -f "${DIR}/train.npy" ]; then
  echo "error: ${DIR}/train.npy is missing; run scripts/prepare_dataset.sh first" >&2
  exit 1
fi

INDEX="${DIR}/index/hnsw_M${M}_efc${EFC}.bin"
if [ -f "${INDEX}" ] && [ "${FORCE}" != "1" ]; then
  echo "[index] exists: ${INDEX}"
else
  echo "[index] laef_build_index (M=${M}, efc=${EFC}, threads=${C})"
  "${BIN}/laef_build_index" --dataset-dir "${DIR}" --M "${M}" --efc "${EFC}" --threads "${C}"
fi

MENU="${DIR}/tables/probe_menu.json"
if [ -f "${MENU}" ] && [ "${FORCE}" != "1" ]; then
  echo "[calibrate] exists: ${MENU}"
else
  echo "[calibrate] laef_calibrate (probe lengths=${PROBE_LENGTHS}, threads=${C})"
  "${BIN}/laef_calibrate" --dataset-dir "${DIR}" --M "${M}" --efc "${EFC}" --threads "${C}" \
    --probe-lengths "${PROBE_LENGTHS}"
fi

PROFILE="${DIR}/profile/profile_C${C}.json"
if [ -f "${PROFILE}" ] && [ "${FORCE}" != "1" ]; then
  echo "[profile] exists: ${PROFILE}"
else
  echo "[profile] laef_profile (threads=${C})"
  "${BIN}/laef_profile" --dataset-dir "${DIR}" --M "${M}" --efc "${EFC}" --threads "${C}"
fi

echo "[done] ${ID}, C=${C}: ${INDEX}, ${MENU}, ${PROFILE}"
