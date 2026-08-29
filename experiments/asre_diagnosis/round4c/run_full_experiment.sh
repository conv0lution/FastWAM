#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${ROUND4C_PYTHON:-${FASTWAM_PYTHON:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10}}"
OUTPUT_ROOT="${ROUND4C_OUTPUT_ROOT:-$PROJECT_ROOT/asre_results/round4c_energy_sufficiency}"
GPU_IDS_RAW="${ROUND4C_GPU_IDS:-0 1 2 3}"
STAGGER="${ROUND4C_LAUNCH_STAGGER_SECONDS:-30}"
ROUND4B_ROOT="${ROUND4C_ROUND4B_ROOT:-$PROJECT_ROOT/asre_results/round4b_subspace}"
read -r -a GPU_IDS <<< "$GPU_IDS_RAW"

exec "$PYTHON_BIN" -m experiments.asre_diagnosis.round4c.run_round4c \
  --output-root "$OUTPUT_ROOT" \
  --round4b-root "$ROUND4B_ROOT" \
  --python "$PYTHON_BIN" \
  --gpu-ids "${GPU_IDS[@]}" \
  --launch-stagger-seconds "$STAGGER"
