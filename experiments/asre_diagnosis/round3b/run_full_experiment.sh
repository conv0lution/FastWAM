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
ROUND3A_ROOT="${ASRE_ROUND3A_OUTPUT_ROOT:-${ROUND1_ROOT}/round3a}"
ROUND3B_ROOT="${ASRE_ROUND3B_OUTPUT_ROOT:-${ROUND1_ROOT}/round3b}"
STATE_BANK_DIR="${ROUND1_ROOT}/state_bank"
VALID_MANIFEST="${ROUND2_ROOT}/state_bank_valid_manifest.json"
DONOR_ROOT="${ROUND3B_ROOT}/donors"
ONLINE_DONOR_ROOT="${DONOR_ROOT}/online"
DONOR_MAPPING="${ONLINE_DONOR_ROOT}/donor_mapping.json"
DONOR_MANIFEST="${ONLINE_DONOR_ROOT}/donor_observation_manifest.json"
OFFLINE_DONOR_MAPPING="${DONOR_ROOT}/offline_donor_mapping.json"
PREFLIGHT_REPORT="${ROUND3B_ROOT}/preflight_report.json"
SELF_REPORT="${ROUND3B_ROOT}/self_replacement/result.json"
SMOKE_ROOT="${ROUND3B_ROOT}/online_smoke"
FULL_ROOT="${ROUND3B_ROOT}/online_full"
OFFLINE_ROOT="${ROUND3B_ROOT}/offline"
AGGREGATE_ROOT="${ROUND3B_ROOT}/aggregate"
LOG_ROOT="${ROUND3B_ROOT}/logs"
ROUND3B_GPU_IDS="${ROUND3B_GPU_IDS:-0,1,2}"
ROUND3B_LAUNCH_STAGGER_SECONDS="${ROUND3B_LAUNCH_STAGGER_SECONDS:-0}"
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

for required in \
  "${PYTHON_BIN}" \
  "${CHECKPOINT}" \
  "${DATASET_STATS}" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
  "${VALID_MANIFEST}" \
  "${STATE_BANK_DIR}/manifest.jsonl" \
  "${ROUND3A_ROOT}/online_full/launcher_summary.json" \
  "${ROUND3A_ROOT}/aggregate/aggregate_metadata.json"; do
  if [[ ! -r "${required}" ]]; then
    printf 'Required Round-3B input is unavailable: %s\n' "${required}" >&2
    exit 1
  fi
done

GIT_STATUS="$(git status --porcelain --untracked-files=normal -- . \
  ':(exclude)asre_results/round3b/**')"
if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n%s\n' \
    'Round-3B requires a clean committed source worktree before any experiment.' \
    "${GIT_STATUS}" >&2
  exit 1
fi

mkdir -p "${ROUND3B_ROOT}" "${LOG_ROOT}"
exec {DRIVER_LOCK_FD}>"${ROUND3B_ROOT}/.driver.lock"
if ! flock -n "${DRIVER_LOCK_FD}"; then
  printf 'Another Round-3B driver owns %s/.driver.lock.\n' "${ROUND3B_ROOT}" >&2
  exit 1
fi

printf '1/9 Validating frozen Round-3A parent and 499-state provenance.\n'
"${PYTHON_BIN}" experiments/asre_diagnosis/round3b/preflight.py \
  --round3a-root "${ROUND3A_ROOT}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --parent-only

printf '2/9 Capturing or validating 100 fixed first-query donor observations.\n'
env \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID=0 \
  "${PYTHON_BIN}" experiments/asre_diagnosis/round3b/prepare_online_donors.py \
    --output-root "${ONLINE_DONOR_ROOT}" \
    --dataset-stats "${DATASET_STATS}" \
    --task-config libero_uncond_2cam224_1e-4 \
    --seed 42 \
    --num-steps-wait 30

"${PYTHON_BIN}" experiments/asre_diagnosis/round3b/prepare_offline_donors.py \
  --state-bank-dir "${STATE_BANK_DIR}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --output "${OFFLINE_DONOR_MAPPING}"

printf '3/9 Freezing the full donor-aware preflight report.\n'
"${PYTHON_BIN}" experiments/asre_diagnosis/round3b/preflight.py \
  --round3a-root "${ROUND3A_ROOT}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --donor-mapping "${DONOR_MAPPING}" \
  --donor-observation-manifest "${DONOR_MANIFEST}" \
  --donor-observation-root "${ONLINE_DONOR_ROOT}" \
  --output-report "${PREFLIGHT_REPORT}"

printf '4/9 Running the mandatory same-input replacement identity gate on GPU %s.\n' \
  "${physical_gpus[0]}"
mkdir -p "$(dirname -- "${SELF_REPORT}")"
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="${physical_gpus[0]}" \
"${PYTHON_BIN}" experiments/asre_diagnosis/round3b/self_replacement_test.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset-stats "${DATASET_STATS}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --state-bank-dir "${STATE_BANK_DIR}" \
  --output "${SELF_REPORT}" \
  --task-config libero_uncond_2cam224_1e-4

printf '5/9 Running the exact 499-state offline control on GPUs %s.\n' \
  "${ROUND3B_GPU_IDS}"
ASRE_ROUND3B_OUTPUT_ROOT="${ROUND3B_ROOT}" \
CHECKPOINT="${CHECKPOINT}" \
DATASET_STATS="${DATASET_STATS}" \
PYTHON_BIN="${PYTHON_BIN}" \
ROUND3B_GPU_IDS="${ROUND3B_GPU_IDS}" \
"${SCRIPT_DIR}/run_offline.sh"

launch_args=(
  --checkpoint "${CHECKPOINT}"
  --dataset-stats-path "${DATASET_STATS}"
  --valid-manifest "${VALID_MANIFEST}"
  --donor-mapping "${DONOR_MAPPING}"
  --donor-observation-manifest "${DONOR_MANIFEST}"
  --donor-observation-root "${ONLINE_DONOR_ROOT}"
  --preflight-report "${PREFLIGHT_REPORT}"
  --self-replacement-report "${SELF_REPORT}"
  --task-config libero_uncond_2cam224_1e-4
  --python "${PYTHON_BIN}"
  --gpu-ids "${physical_gpus[@]}"
  --launch-stagger-seconds "${ROUND3B_LAUNCH_STAGGER_SECONDS}"
)

printf '6/9 Running task-0 two-trial smoke for all three conditions.\n'
"${PYTHON_BIN}" experiments/asre_diagnosis/round3b/launch_three_gpu.py \
  --mode smoke \
  --output-root "${SMOKE_ROOT}" \
  "${launch_args[@]}"

printf '7/9 Running the three 100-episode paired online conditions.\n'
"${PYTHON_BIN}" experiments/asre_diagnosis/round3b/launch_three_gpu.py \
  --mode full \
  --output-root "${FULL_ROOT}" \
  --smoke-summary "${SMOKE_ROOT}/launcher_summary.json" \
  "${launch_args[@]}"

printf '8/9 Aggregating paired online, offline, donor, and K/V sanity results.\n'
"${PYTHON_BIN}" experiments/asre_diagnosis/round3b/aggregate_results.py \
  --online-root "${FULL_ROOT}" \
  --offline-root "${OFFLINE_ROOT}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --offline-donor-mapping "${OFFLINE_DONOR_MAPPING}" \
  --online-donor-mapping "${DONOR_MAPPING}" \
  --online-donor-manifest "${DONOR_MANIFEST}" \
  --self-replacement-result "${SELF_REPORT}" \
  --output-dir "${AGGREGATE_ROOT}" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 0

printf '9/9 Generating the five pre-specified figures and stopping.\n'
MPLCONFIGDIR="${ROUND3B_ROOT}/.matplotlib" \
"${PYTHON_BIN}" experiments/asre_diagnosis/round3b/plot_results.py \
  --aggregate-dir "${AGGREGATE_ROOT}"

printf '%s\n' \
  'Round-3B complete. Stop rule applied: no stale/cross-task/key-only/value-only,' \
  'patching, sparsity, pruning, additional-suite, or additional-checkpoint run was launched.'
