#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${SALVAGE_B_PYTHON:-${FASTWAM_PYTHON:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10}}"
OUTPUT_ROOT="${SALVAGE_B_OUTPUT_ROOT:-$PROJECT_ROOT/asre_results/salvage_b_world_action_dissociation}"
GPU_IDS_RAW="${SALVAGE_B_GPU_IDS:-0 1 2 3}"
STAGGER="${SALVAGE_B_LAUNCH_STAGGER_SECONDS:-70}"
ROUND4B_ROOT="${SALVAGE_B_ROUND4B_ROOT:-$PROJECT_ROOT/asre_results/round4b_subspace}"
ROUND4C_ROOT="${SALVAGE_B_ROUND4C_ROOT:-$PROJECT_ROOT/asre_results/round4c_energy_sufficiency/retry_20260829_donorcheckfix}"
SALVAGE_A_SUMMARY="${SALVAGE_B_SALVAGE_A_SUMMARY:-$PROJECT_ROOT/asre_results/salvage_a_action_sensitive/aggregate/salvage_a_summary.json}"
VALID_MANIFEST="${SALVAGE_B_VALID_MANIFEST:-$PROJECT_ROOT/asre_results/round2/state_bank_valid_manifest.json}"
DONOR_ROOT="${SALVAGE_B_DONOR_ROOT:-$PROJECT_ROOT/asre_results/round3b/donors/online}"
DATASET_ROOT="${SALVAGE_B_DATASET_ROOT:-/local_home/zhaizicheng/fastwam_assets/datasets/LIBERO-fastwam/lerobot_v30/libero_spatial_no_noops_lerobot}"
read -r -a GPU_IDS <<< "$GPU_IDS_RAW"

exec "$PYTHON_BIN" -m experiments.asre_diagnosis.salvage_b.run_salvage_b \
  --output-root "$OUTPUT_ROOT" \
  --round4b-root "$ROUND4B_ROOT" \
  --round4c-root "$ROUND4C_ROOT" \
  --salvage-a-summary "$SALVAGE_A_SUMMARY" \
  --valid-manifest "$VALID_MANIFEST" \
  --donor-root "$DONOR_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --python "$PYTHON_BIN" \
  --gpu-ids "${GPU_IDS[@]}" \
  --launch-stagger-seconds "$STAGGER"
