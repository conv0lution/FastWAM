#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${ROUND4B_ENERGY_PYTHON:-${FASTWAM_PYTHON:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10}}"
ROUND4B_ROOT="${ROUND4B_ENERGY_PARENT_ROOT:-$PROJECT_ROOT/asre_results/round4b_subspace}"
OUTPUT_ROOT="${ROUND4B_ENERGY_OUTPUT_ROOT:-$ROUND4B_ROOT/energy_curve_analysis}"
GPU_IDS_RAW="${ROUND4B_ENERGY_GPU_IDS:-0 1 2 3}"
STAGGER="${ROUND4B_ENERGY_LAUNCH_STAGGER_SECONDS:-30}"
read -r -a GPU_IDS <<< "$GPU_IDS_RAW"

exec "$PYTHON_BIN" -m experiments.asre_diagnosis.round4b.run_energy_curve_analysis \
  --round4b-root "$ROUND4B_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --python "$PYTHON_BIN" \
  --gpu-ids "${GPU_IDS[@]}" \
  --launch-stagger-seconds "$STAGGER"
