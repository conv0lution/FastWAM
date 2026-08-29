#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${ROUND4B_PYTHON:-${FASTWAM_PYTHON:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10}}"
OUTPUT_ROOT="${ROUND4B_OUTPUT_ROOT:-$PROJECT_ROOT/asre_results/round4b_subspace}"
GPU_IDS_RAW="${ROUND4B_GPU_IDS:-0 1 2 3}"
STAGGER="${ROUND4B_LAUNCH_STAGGER_SECONDS:-30}"
ROUND4A_ROOT="${ROUND4B_ROUND4A_ROOT:-$PROJECT_ROOT/asre_results/round4a/retry_20260829_configfix}"
read -r -a GPU_IDS <<< "$GPU_IDS_RAW"

exec "$PYTHON_BIN" -m experiments.asre_diagnosis.round4b.run_round4b \
  --output-root "$OUTPUT_ROOT" \
  --round4a-root "$ROUND4A_ROOT" \
  --python "$PYTHON_BIN" \
  --gpu-ids "${GPU_IDS[@]}" \
  --launch-stagger-seconds "$STAGGER"
