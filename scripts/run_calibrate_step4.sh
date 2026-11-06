#!/usr/bin/env bash
# commands.md step 4 for the first three datasets (run from the Implementation/ folder).
cd "$(dirname "$0")/.."
for ID in synthetic-small lastfm-64-dot coco-i2i-512-angular; do
  echo "=== ${ID}"
  build/bin/laef_calibrate --dataset-dir "data/artifacts/${ID}" --M 16 --efc 500 \
    --probe-lengths 1025,512,256,128 --threads "$(nproc)" || echo "FAILED ${ID}"
done
echo FINISHED
