#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

LIBERO_ROOT="${LIBERO_ROOT:-$PROJECT_ROOT/../LIBERO}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$LIBERO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/fastwam-salvage-a-numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/fastwam-salvage-a-matplotlib-cache}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/local_home/zhaizicheng/fastwam_assets/checkpoints/wan_base}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

PYTHON_BIN="${SALVAGE_A_PYTHON:-${FASTWAM_PYTHON:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10}}"
OUTPUT_ROOT="${SALVAGE_A_OUTPUT_ROOT:-$PROJECT_ROOT/asre_results/salvage_a_action_sensitive}"
GPU_IDS_RAW="${SALVAGE_A_GPU_IDS:-0 1 2 3}"
STAGGER="${SALVAGE_A_LAUNCH_STAGGER_SECONDS:-70}"
ROUND4C_SUMMARY="${SALVAGE_A_ROUND4C_SUMMARY:-$PROJECT_ROOT/asre_results/round4c_energy_sufficiency/retry_20260829_donorcheckfix/aggregate/round4c_summary.json}"
VALID_MANIFEST="${SALVAGE_A_VALID_MANIFEST:-$PROJECT_ROOT/asre_results/round2/state_bank_valid_manifest.json}"
SOURCE_DONOR_ROOT="${SALVAGE_A_SOURCE_DONOR_ROOT:-$PROJECT_ROOT/asre_results/round3b/donors/online}"
read -r -a GPU_IDS <<< "$GPU_IDS_RAW"

exec "$PYTHON_BIN" -m experiments.asre_diagnosis.salvage_a.run_salvage_a \
  --output-root "$OUTPUT_ROOT" \
  --round4c-summary "$ROUND4C_SUMMARY" \
  --valid-manifest "$VALID_MANIFEST" \
  --source-donor-root "$SOURCE_DONOR_ROOT" \
  --python "$PYTHON_BIN" \
  --gpu-ids "${GPU_IDS[@]}" \
  --launch-stagger-seconds "$STAGGER"
