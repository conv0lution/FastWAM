#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python}"
CHECKPOINT="${CHECKPOINT:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/local_home/zhaizicheng/fastwam_assets/checkpoints/wan_base}"
ROUND1_ROOT="${ASRE_ROUND1_OUTPUT_ROOT:-${REPO_ROOT}/asre_results}"
ROUND2_ROOT="${ASRE_ROUND2_OUTPUT_ROOT:-${ROUND1_ROOT}/round2}"
ROUND3B_ROOT="${ASRE_ROUND3B_OUTPUT_ROOT:-${ROUND1_ROOT}/round3b}"
STATE_BANK_DIR="${ROUND1_ROOT}/state_bank"
VALID_MANIFEST="${ROUND2_ROOT}/state_bank_valid_manifest.json"
OUTPUT_ROOT="${ROUND3B_ROOT}/offline"
DONOR_MAPPING="${ROUND3B_ROOT}/donors/offline_donor_mapping.json"
LOG_ROOT="${ROUND3B_ROOT}/logs/offline"
ROUND3B_GPU_IDS="${ROUND3B_GPU_IDS:-0,1,2}"
IFS=',' read -r -a physical_gpus <<<"${ROUND3B_GPU_IDS}"
if [[ "${#physical_gpus[@]}" -ne 3 || \
      ! "${physical_gpus[0]}" =~ ^[0-9]+$ || \
      ! "${physical_gpus[1]}" =~ ^[0-9]+$ || \
      ! "${physical_gpus[2]}" =~ ^[0-9]+$ || \
      "${physical_gpus[0]}" == "${physical_gpus[1]}" || \
      "${physical_gpus[0]}" == "${physical_gpus[2]}" || \
      "${physical_gpus[1]}" == "${physical_gpus[2]}" ]]; then
  printf 'ROUND3B_GPU_IDS must contain three distinct nonnegative integers: %s\n' \
    "${ROUND3B_GPU_IDS}" >&2
  exit 1
fi

LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/../LIBERO}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export DIFFSYNTH_MODEL_BASE_PATH
unset MUJOCO_EGL_DEVICE_ID MUJOCO_GL PYOPENGL_PLATFORM

for required in \
  "${PYTHON_BIN}" \
  "${CHECKPOINT}" \
  "${DATASET_STATS}" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
  "${STATE_BANK_DIR}/manifest.jsonl" \
  "${STATE_BANK_DIR}/run_metadata.json" \
  "${VALID_MANIFEST}"; do
  if [[ ! -r "${required}" ]]; then
    printf 'Required Round-3B offline input is unavailable: %s\n' "${required}" >&2
    exit 1
  fi
done

GIT_STATUS="$(git status --porcelain --untracked-files=normal -- . \
  ':(exclude)asre_results/round3b/**')"
if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n%s\n' \
    'Round-3B offline replay requires a clean code worktree.' \
    "${GIT_STATUS}" >&2
  exit 1
fi

mkdir -p "${ROUND3B_ROOT}"
exec {OFFLINE_LOCK_FD}>"${ROUND3B_ROOT}/.offline.lock"
if ! flock -n "${OFFLINE_LOCK_FD}"; then
  printf 'Another Round-3B offline replay owns %s/.offline.lock.\n' \
    "${ROUND3B_ROOT}" >&2
  exit 1
fi

"${PYTHON_BIN}" experiments/asre_diagnosis/round3b/prepare_offline_donors.py \
  --state-bank-dir "${STATE_BANK_DIR}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --output "${DONOR_MAPPING}"

conditions=(late_current_correct late_wrong_scene late_no_video)
enabled_layers=(
  '[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]'
  '[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]'
  '[]'
)
disabled_layers=(
  '[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]'
  '[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]'
  '[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]'
)
replacement_layers=(
  '[]'
  '[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]'
  '[]'
)

condition_complete() {
  local condition="$1"
  local index="$2"
  local physical_gpu="$3"
  "${PYTHON_BIN}" -c \
    'import json,pathlib,sys
root,condition,index,physical_gpu=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
path=pathlib.Path(root)/condition
try:
    meta=json.loads((path/"run_metadata.json").read_text(encoding="utf-8"))
    records=[json.loads(line) for line in (path/"per_sample.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ok=(meta.get("status")=="complete" and meta.get("condition_protocol")=="round3b_matched_kv_replacement"
        and meta.get("diagnosis_condition")==condition and meta.get("physical_gpu")==physical_gpu
        and len(records)==499 and (path/"summary.csv").stat().st_size>0
        and (path/"actions.npz").stat().st_size>0)
except (OSError,ValueError,TypeError,json.JSONDecodeError):
    ok=False
sys.exit(0 if ok else 1)' \
    "${OUTPUT_ROOT}" "${condition}" "${index}" "${physical_gpu}"
}

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
pids=()
launched_indices=()
for index in 0 1 2; do
  condition="${conditions[$index]}"
  physical_gpu="${physical_gpus[$index]}"
  if condition_complete "${condition}" "${index}" "${physical_gpu}"; then
    printf 'Skipping complete Round-3B offline condition: %s\n' "${condition}"
    continue
  fi
  if [[ -d "${OUTPUT_ROOT}/${condition}" && \
        -n "$(find "${OUTPUT_ROOT}/${condition}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'Partial/incompatible offline condition requires review; refusing overwrite: %s\n' \
      "${OUTPUT_ROOT}/${condition}" >&2
    exit 1
  fi
  log_path="${LOG_ROOT}/${condition}.log"
  if [[ -e "${log_path}" ]]; then
    printf 'Offline log already exists for an incomplete condition: %s\n' "${log_path}" >&2
    exit 1
  fi
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
  ASRE_ROUND3B_PHYSICAL_GPU="${physical_gpu}" \
  "${PYTHON_BIN}" experiments/asre_diagnosis/round3b/replay_state_bank.py \
    task=libero_uncond_2cam224_1e-4 \
    "ckpt=${CHECKPOINT}" \
    seed=42 \
    model.load_text_encoder=false \
    EVALUATION.device=cuda:0 \
    EVALUATION.text_encoder_device=null \
    "EVALUATION.dataset_stats_path=${DATASET_STATS}" \
    ASRE_DIAGNOSIS.enabled=true \
    ASRE_DIAGNOSIS.mode=replace_video_kv \
    ASRE_DIAGNOSIS.protocol=round3b_matched_kv_replacement \
    "ASRE_DIAGNOSIS.condition_name=${condition}" \
    "ASRE_DIAGNOSIS.condition_index=${index}" \
    "ASRE_DIAGNOSIS.enabled_video_retrieval_layers=${enabled_layers[$index]}" \
    "ASRE_DIAGNOSIS.disabled_video_layers=${disabled_layers[$index]}" \
    "ASRE_DIAGNOSIS.replacement_video_layers=${replacement_layers[$index]}" \
    "ASRE_DIAGNOSIS.state_bank_dir=${STATE_BANK_DIR}" \
    "ASRE_DIAGNOSIS.valid_state_bank_manifest_path=${VALID_MANIFEST}" \
    "ASRE_DIAGNOSIS.offline_output_dir=${OUTPUT_ROOT}" \
    "ASRE_DIAGNOSIS.offline_donor_mapping_path=${DONOR_MAPPING}" \
    ASRE_DIAGNOSIS.executed_prefix_length=10 \
    >"${log_path}" 2>&1 &
  pids+=("$!")
  launched_indices+=("${index}")
  printf 'Launched offline %s on physical GPU %s; log=%s\n' \
    "${condition}" "${physical_gpu}" "${log_path}"
done

status=0
for offset in "${!pids[@]}"; do
  pid="${pids[$offset]}"
  index="${launched_indices[$offset]}"
  if ! wait "${pid}"; then
    printf 'Offline condition failed: %s (GPU %s).\n' \
      "${conditions[$index]}" "${physical_gpus[$index]}" >&2
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  printf 'Round-3B offline replay failed; inspect %s.\n' "${LOG_ROOT}" >&2
  exit "${status}"
fi
for index in 0 1 2; do
  if ! condition_complete \
      "${conditions[$index]}" "${index}" "${physical_gpus[$index]}"; then
    printf 'Offline validation failed after replay: %s.\n' "${conditions[$index]}" >&2
    exit 1
  fi
done
printf 'Round-3B offline replay complete: %s\n' "${OUTPUT_ROOT}"
