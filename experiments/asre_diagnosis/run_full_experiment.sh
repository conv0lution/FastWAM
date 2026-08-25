#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python}"
CHECKPOINT="${CHECKPOINT:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
OUTPUT_ROOT="${ASRE_OUTPUT_ROOT:-${REPO_ROOT}/asre_results}"
STATE_BANK_DIR="${OUTPUT_ROOT}/state_bank"
ONLINE_SMOKE_DIR="${OUTPUT_ROOT}/online_smoke"
ONLINE_FULL_DIR="${OUTPUT_ROOT}/online_full"
LOG_DIR="${OUTPUT_ROOT}/logs"

# State-bank collection intentionally uses one baseline model. Keep T5 and
# MuJoCo/EGL on another physical GPU so they do not compete with Wan 2.2.
STATE_MODEL_GPU="${STATE_MODEL_GPU:-0}"
STATE_AUX_GPU="${STATE_AUX_GPU:-1}"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python executable is unavailable: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -r "${CHECKPOINT}" ]]; then
  printf 'Checkpoint is unavailable: %s\n' "${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -r "${DATASET_STATS}" ]]; then
  printf 'Dataset stats are unavailable: %s\n' "${DATASET_STATS}" >&2
  exit 1
fi
if [[ "${STATE_MODEL_GPU}" == "${STATE_AUX_GPU}" ]]; then
  printf 'STATE_MODEL_GPU and STATE_AUX_GPU must be different.\n' >&2
  exit 1
fi

run_logged() {
  local step_name="$1"
  shift
  local log_path="${LOG_DIR}/${step_name}.log"
  printf '\n[%s] Starting %s\n' "$(date --iso-8601=seconds)" "${step_name}"
  printf '[%s] Log: %s\n' "$(date --iso-8601=seconds)" "${log_path}"
  "$@" > >(tee -a "${log_path}") 2>&1
  printf '[%s] Finished %s\n' "$(date --iso-8601=seconds)" "${step_name}"
}

run_logged collect_state_bank \
  env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${STATE_MODEL_GPU},${STATE_AUX_GPU}" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${STATE_AUX_GPU}" \
  "${PYTHON_BIN}" experiments/asre_diagnosis/collect_state_bank.py \
    task=libero_uncond_2cam224_1e-4 \
    ckpt="${CHECKPOINT}" \
    EVALUATION.device=cuda:0 \
    EVALUATION.text_encoder_device=cuda:1 \
    EVALUATION.task_suite_name=libero_spatial \
    EVALUATION.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
    EVALUATION.num_trials=10 \
    EVALUATION.dataset_stats_path="${DATASET_STATS}" \
    ASRE_DIAGNOSIS.state_bank_dir="${STATE_BANK_DIR}"

run_logged checkpoint_smoke_test \
  env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${STATE_MODEL_GPU}" \
  "${PYTHON_BIN}" experiments/asre_diagnosis/smoke_test.py \
    task=libero_uncond_2cam224_1e-4 \
    ckpt="${CHECKPOINT}" \
    model.load_text_encoder=false \
    EVALUATION.device=cuda:0 \
    EVALUATION.dataset_stats_path="${DATASET_STATS}" \
    ASRE_DIAGNOSIS.state_bank_dir="${STATE_BANK_DIR}"

run_logged online_smoke \
  "${PYTHON_BIN}" experiments/asre_diagnosis/launch_eight_gpu.py \
    --mode smoke \
    --python "${PYTHON_BIN}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-stats-path "${DATASET_STATS}" \
    --output-root "${ONLINE_SMOKE_DIR}"

run_logged online_full \
  "${PYTHON_BIN}" experiments/asre_diagnosis/launch_eight_gpu.py \
    --mode full \
    --python "${PYTHON_BIN}" \
    --checkpoint "${CHECKPOINT}" \
    --dataset-stats-path "${DATASET_STATS}" \
    --output-root "${ONLINE_FULL_DIR}" \
    --smoke-summary "${ONLINE_SMOKE_DIR}/launcher_summary.json" \
    --correctness-report "${STATE_BANK_DIR}/smoke_test_report.json"

printf '\nAll four ASRE diagnosis stages completed successfully.\n'
