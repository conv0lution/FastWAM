#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/../LIBERO}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Fixed-state replay does not render. Do not inherit an EGL GPU chosen by an
# earlier online evaluation shell.
unset MUJOCO_EGL_DEVICE_ID MUJOCO_GL PYOPENGL_PLATFORM

PYTHON_BIN="${PYTHON_BIN:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python}"
CHECKPOINT="${CHECKPOINT:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
OUTPUT_ROOT="${ASRE_OUTPUT_ROOT:-${REPO_ROOT}/asre_results}"
STATE_BANK_DIR="${OUTPUT_ROOT}/state_bank"
OFFLINE_DIR="${OUTPUT_ROOT}/offline"
AGGREGATE_DIR="${OUTPUT_ROOT}/aggregate"
LOG_DIR="${OUTPUT_ROOT}/logs"

CONDITIONS=(
  baseline
  drop_00_04
  drop_05_09
  drop_10_14
  drop_15_19
  drop_20_24
  drop_25_29
  drop_all
)

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
if [[ ! -r "${STATE_BANK_DIR}/manifest.jsonl" ]]; then
  printf 'State-bank manifest is unavailable: %s\n' "${STATE_BANK_DIR}/manifest.jsonl" >&2
  exit 1
fi

EXPECTED_SAMPLES="$(awk 'NF { count += 1 } END { print count + 0 }' "${STATE_BANK_DIR}/manifest.jsonl")"
mkdir -p "${OFFLINE_DIR}" "${LOG_DIR}"

condition_complete() {
  local condition="$1"
  local summary_path="${OFFLINE_DIR}/${condition}/summary.csv"
  [[ -r "${summary_path}" ]] || return 1
  "${PYTHON_BIN}" -c \
    'import csv, sys; rows=list(csv.DictReader(open(sys.argv[1], encoding="utf-8"))); sys.exit(0 if len(rows)==1 and int(rows[0]["num_samples"])==int(sys.argv[2]) else 1)' \
    "${summary_path}" "${EXPECTED_SAMPLES}"
}

declare -a pids=()
declare -a launched_conditions=()
declare -a launched_logs=()

for condition_index in "${!CONDITIONS[@]}"; do
  condition="${CONDITIONS[$condition_index]}"
  condition_dir="${OFFLINE_DIR}/${condition}"
  log_path="${LOG_DIR}/offline_${condition_index}.log"

  if condition_complete "${condition}"; then
    printf 'Skipping completed %s (%s samples).\n' "${condition}" "${EXPECTED_SAMPLES}"
    continue
  fi
  if [[ -e "${condition_dir}/per_sample.jsonl" ]]; then
    printf 'Incomplete replay output exists; refusing to overwrite: %s\n' "${condition_dir}" >&2
    printf 'Move that condition directory aside, then rerun this script.\n' >&2
    exit 1
  fi

  printf 'Launching %s on physical GPU %s; log=%s\n' \
    "${condition}" "${condition_index}" "${log_path}"
  env \
    -u MUJOCO_EGL_DEVICE_ID \
    -u MUJOCO_GL \
    -u PYOPENGL_PLATFORM \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${condition_index}" \
    "${PYTHON_BIN}" experiments/asre_diagnosis/replay_state_bank.py \
      task=libero_uncond_2cam224_1e-4 \
      ckpt="${CHECKPOINT}" \
      model.load_text_encoder=false \
      EVALUATION.device=cuda:0 \
      EVALUATION.dataset_stats_path="${DATASET_STATS}" \
      ASRE_DIAGNOSIS.enabled=true \
      ASRE_DIAGNOSIS.condition_index="${condition_index}" \
      ASRE_DIAGNOSIS.state_bank_dir="${STATE_BANK_DIR}" \
      ASRE_DIAGNOSIS.offline_output_dir="${OFFLINE_DIR}" \
      > "${log_path}" 2>&1 &
  pids+=("$!")
  launched_conditions+=("${condition}")
  launched_logs+=("${log_path}")
done

failed=0
for process_index in "${!pids[@]}"; do
  pid="${pids[process_index]}"
  condition="${launched_conditions[process_index]}"
  if wait "${pid}"; then
    printf '%s: exit=0\n' "${condition}"
  else
    status="$?"
    printf '%s: exit=%s; log=%s\n' \
      "${condition}" "${status}" "${launched_logs[process_index]}" >&2
    failed=1
  fi
done

if (( failed != 0 )); then
  printf 'At least one offline replay failed; aggregation was not run.\n' >&2
  exit 1
fi

for condition in "${CONDITIONS[@]}"; do
  if ! condition_complete "${condition}"; then
    printf 'Missing or incomplete summary for %s.\n' "${condition}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" experiments/asre_diagnosis/aggregate_results.py \
  --online-root "${OUTPUT_ROOT}/online_full" \
  --offline-root "${OFFLINE_DIR}" \
  --output-dir "${AGGREGATE_DIR}"

"${PYTHON_BIN}" experiments/asre_diagnosis/plot_results.py \
  --aggregate-dir "${AGGREGATE_DIR}"

printf 'Offline replay, aggregation, and plotting completed successfully.\n'
printf 'Summary: %s\n' "${AGGREGATE_DIR}/summary.csv"
printf 'Plots: %s\n' "${AGGREGATE_DIR}/plots"
