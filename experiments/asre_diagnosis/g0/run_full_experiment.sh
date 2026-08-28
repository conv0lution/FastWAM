#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${FASTWAM_PYTHON:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10}"
OUTPUT_ROOT="${G0_OUTPUT_ROOT:-${PROJECT_ROOT}/asre_results/g0_cross_suite}"

cd "${PROJECT_ROOT}"

exec "${PYTHON_BIN}" \
  -m experiments.asre_diagnosis.g0.run_g0 \
  --checkpoint /local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --dataset-stats /local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --valid-manifest "${PROJECT_ROOT}/asre_results/round2/state_bank_valid_manifest.json" \
  --state-bank-dir "${PROJECT_ROOT}/asre_results/state_bank" \
  --output-root "${OUTPUT_ROOT}" \
  --python "${PYTHON_BIN}" \
  --gpu-ids 4 5 6 7 \
  --launch-stagger-seconds 70
