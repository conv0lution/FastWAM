#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${SALVAGE_B_NUMEQ_PYTHON:-${FASTWAM_PYTHON:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10}}"
SOURCE_ROOT="${SALVAGE_B_NUMEQ_SOURCE_ROOT:-$PROJECT_ROOT/asre_results/salvage_b_world_action_dissociation/retry_20260830_dtypegatefix}"
OUTPUT_ROOT="${SALVAGE_B_NUMEQ_OUTPUT_ROOT:-$PROJECT_ROOT/asre_results/salvage_b_numerical_equivalence}"
GPU_ID="${SALVAGE_B_NUMEQ_GPU_ID:-0}"
INCLUDE_FP16="${SALVAGE_B_NUMEQ_INCLUDE_FP16:-0}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/runtime"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$OUTPUT_ROOT/runtime/matplotlib_cache"
export TORCH_HOME="$OUTPUT_ROOT/runtime/torch_home"
export TORCHINDUCTOR_CACHE_DIR="$OUTPUT_ROOT/runtime/torchinductor"
export TRITON_CACHE_DIR="$OUTPUT_ROOT/runtime/triton"

SALVAGE_LOG="$OUTPUT_ROOT/logs/salvage_b_tests.log"
FULL_LOG="$OUTPUT_ROOT/logs/full_asre_tests.log"

"$PYTHON_BIN" -m pytest -q \
  experiments/asre_diagnosis/tests/test_salvage_b_*.py \
  2>&1 | tee "$SALVAGE_LOG"

"$PYTHON_BIN" -m pytest -q \
  experiments/asre_diagnosis/tests \
  2>&1 | tee "$FULL_LOG"

ARGS=(
  --source-root "$SOURCE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --runtime-work-dir "$OUTPUT_ROOT/runtime"
  --salvage-b-test-log "$SALVAGE_LOG"
  --full-asre-test-log "$FULL_LOG"
)
if [[ "$INCLUDE_FP16" == "1" ]]; then
  ARGS+=(--include-fp16)
fi

AUDIT_LOG="$OUTPUT_ROOT/logs/numerical_equivalence.log"
"$PYTHON_BIN" -m experiments.asre_diagnosis.salvage_b.numerical_equivalence "${ARGS[@]}" \
  2>&1 | tee "$AUDIT_LOG"
