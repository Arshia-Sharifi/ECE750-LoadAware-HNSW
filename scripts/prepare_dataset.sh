#!/usr/bin/env bash
# LA-ef: serving machine (IMPLEMENTATION_PLAN.md §4, §12 P6.4).
#   1. Rebuilds train.npy, test.npy and holdout.npy locally from the raw download (run_preprocess.py, download and convert only).
#   2. Checks the fingerprints in the manifest.json that Colab wrote, and the checksums of the small files.
#   3. Copies gt/, calibration/, stats/ and manifest.json into data/artifacts/<id>/.
# Verification runs before anything is copied, so a mismatch leaves data/artifacts/<id>/ unchanged.
# Usage: bash scripts/prepare_dataset.sh <dataset-id> <colab-output-dir>/<dataset-id>
# Environment: PYTHON (default: .venv/bin/python in the Implementation folder).
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <dataset-id> <colab-output-dir>/<dataset-id>" >&2
  exit 2
fi
ID="$1"
SRC="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${HERE}/data"
DIR="${DATA}/artifacts/${ID}"
PY="${PYTHON:-${HERE}/.venv/bin/python}"

for part in manifest.json gt calibration stats; do
  if [ ! -e "${SRC}/${part}" ]; then
    echo "error: ${SRC}/${part} is missing; copy the Colab output folder for ${ID} first" >&2
    exit 1
  fi
done

echo "[1/3] rebuilding train.npy and test.npy locally (download and convert only; GPU steps run on Colab)"
"${PY}" "${HERE}/gpu_preprocessing/run_preprocess.py" --dataset "${ID}" --steps download,convert \
  --data-root "${DATA}" --out "${DATA}/artifacts" --registry "${HERE}/configs/datasets.yaml"

echo "[2/3] checking the fingerprints against ${SRC}/manifest.json"
PYTHONPATH="${HERE}/gpu_preprocessing" "${PY}" - "${SRC}/manifest.json" "${DIR}" <<'PY'
import sys
from pathlib import Path

from prep.manifest import verify_manifest

errors = verify_manifest(Path(sys.argv[1]), Path(sys.argv[2]))
for error in errors:
    print("  " + error, file=sys.stderr)
print("  manifest OK" if not errors else f"  {len(errors)} problem(s); nothing was copied")
sys.exit(1 if errors else 0)
PY

echo "[3/3] copying gt/, calibration/, stats/ and manifest.json into ${DIR}"
mkdir -p "${DIR}"
cp -r "${SRC}/gt" "${SRC}/calibration" "${SRC}/stats" "${DIR}/"
cp "${SRC}/manifest.json" "${DIR}/manifest.json"
echo "prepared ${DIR}"
