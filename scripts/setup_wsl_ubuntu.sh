#!/usr/bin/env bash
# LA-ef: install the toolchain on Ubuntu / WSL2 (IMPLEMENTATION_PLAN.md §2, §6 P0.1).
# Idempotent: apt skips installed packages, uv is installed only when missing, and the script ends by
# printing the tool versions and failing when a tool is older than the plan's minimum.
# Python: uv creates the project venv and fetches Python 3.11 when the system Python is older.
set -euo pipefail

MIN_GCC=11
MIN_CMAKE=3.22

if ! command -v apt-get >/dev/null 2>&1; then
  echo "error: apt-get not found; this script targets Ubuntu / WSL2" >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libeigen3-dev libboost-dev python3-venv git curl

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

gcc_major="$(gcc -dumpversion | cut -d. -f1)"
cmake_version="$(cmake --version | awk 'NR == 1 { print $3 }')"
uv_version="$(uv --version | awk '{ print $2 }')"

echo "gcc     $(gcc -dumpversion)   (minimum ${MIN_GCC})"
echo "cmake   ${cmake_version}   (minimum ${MIN_CMAKE})"
echo "uv      ${uv_version}"
echo "python3 $(python3 --version 2>&1 | awk '{ print $2 }')   (the project venv uses Python 3.11 via uv)"

if [ "${gcc_major}" -lt "${MIN_GCC}" ]; then
  echo "error: gcc ${gcc_major} is older than ${MIN_GCC}" >&2
  exit 1
fi
lowest="$(printf '%s\n%s\n' "${MIN_CMAKE}" "${cmake_version}" | sort -V | head -n 1)"
if [ "${lowest}" != "${MIN_CMAKE}" ]; then
  echo "error: cmake ${cmake_version} is older than ${MIN_CMAKE}" >&2
  exit 1
fi
echo "setup OK. Next: uv venv --python 3.11 .venv && uv pip install -p .venv numpy h5py pyyaml pytest pandas matplotlib"
