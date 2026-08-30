#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_ROOT="${SALVAGE_B_V2_OUTPUT_ROOT:-$PROJECT_ROOT/asre_results/salvage_b_v2_native_shared_node}"
SOURCE_ROOT="${SALVAGE_B_V2_SOURCE_INPUT_ROOT:-$PROJECT_ROOT/asre_results/salvage_b_world_action_dissociation/retry_20260830_dtypegatefix}"
PYTHON_BIN="${SALVAGE_B_V2_PYTHON:-$(command -v python)}"
STAGGER="${SALVAGE_B_V2_LAUNCH_STAGGER_SECONDS:-70}"
read -r -a GPU_IDS <<< "${SALVAGE_B_V2_GPU_IDS:-4 5 6 7}"

exec "$PYTHON_BIN" -m experiments.asre_diagnosis.salvage_b_v2.run_salvage_b_v2 \
  --output-root "$OUTPUT_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --gpu-ids "${GPU_IDS[@]}" \
  --python "$PYTHON_BIN" \
  --launch-stagger-seconds "$STAGGER"
