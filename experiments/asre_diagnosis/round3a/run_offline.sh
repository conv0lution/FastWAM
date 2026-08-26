#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/../LIBERO}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
unset MUJOCO_EGL_DEVICE_ID MUJOCO_GL PYOPENGL_PLATFORM

PYTHON_BIN="${PYTHON_BIN:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python}"
CHECKPOINT="${CHECKPOINT:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
ROUND1_ROOT="${ASRE_ROUND1_OUTPUT_ROOT:-${REPO_ROOT}/asre_results}"
ROUND2_ROOT="${ASRE_ROUND2_OUTPUT_ROOT:-${ROUND1_ROOT}/round2}"
ROUND3A_ROOT="${ASRE_ROUND3A_OUTPUT_ROOT:-${ROUND1_ROOT}/round3a}"
STATE_BANK_DIR="${ROUND1_ROOT}/state_bank"
VALID_MANIFEST="${ROUND2_ROOT}/state_bank_valid_manifest.json"
ROUND2_REFERENCE_METADATA="${ROUND2_ROOT}/online_full/keep_15_29/run_metadata.json"
OFFLINE_DIR="${ROUND3A_ROOT}/offline"
LOG_DIR="${ROUND3A_ROOT}/logs"

CONDITIONS=(
  keep_none_late
  keep_20_24
  keep_15_24
)
ENABLED_SCHEDULES=(
  '[]'
  '[20,21,22,23,24]'
  '[15,16,17,18,19,20,21,22,23,24]'
)

for required_path in "${PYTHON_BIN}" "${CHECKPOINT}" "${DATASET_STATS}" \
  "${STATE_BANK_DIR}/manifest.jsonl" "${STATE_BANK_DIR}/run_metadata.json" \
  "${VALID_MANIFEST}" "${ROUND2_REFERENCE_METADATA}"; do
  if [[ ! -r "${required_path}" ]]; then
    printf 'Required input is unavailable: %s\n' "${required_path}" >&2
    exit 1
  fi
done

GIT_STATUS="$(git status --porcelain --untracked-files=normal -- . \
  ':(exclude)asre_results/round3a/aggregate/**')"
if [[ -n "${GIT_STATUS}" ]]; then
  printf '%s\n' \
    'ASRE Round 3A offline replay requires a clean code worktree.' \
    'Commit the Round-3A implementation first; this script will not commit or discard changes.' \
    "${GIT_STATUS}" >&2
  exit 1
fi

"${PYTHON_BIN}" -c \
  'import pathlib,sys
from experiments.asre_diagnosis.round3a.launch_three_gpu import _validate_round3a_output_scope
_validate_round3a_output_scope(pathlib.Path(sys.argv[1]),valid_manifest_path=pathlib.Path(sys.argv[2]))' \
  "${ROUND3A_ROOT}" "${VALID_MANIFEST}"

mkdir -p "${ROUND3A_ROOT}"
exec {OFFLINE_LOCK_FD}> "${ROUND3A_ROOT}/.offline.lock"
if ! flock -n "${OFFLINE_LOCK_FD}"; then
  printf 'Another Round-3A offline replay owns %s/.offline.lock; refusing concurrent launch.\n' \
    "${ROUND3A_ROOT}" >&2
  exit 1
fi

# Reuse the immutable Round-2 QC-valid manifest. This validates all declared
# paths and hashes once, including the multi-GB checkpoint, without regenerating
# or rewriting the bank or manifest.
"${PYTHON_BIN}" -c \
  'import pathlib,sys
from experiments.asre_diagnosis.round3a.launch_three_gpu import _load_provenance
p=_load_provenance(pathlib.Path(sys.argv[1]),sys.argv[2],sys.argv[3],pathlib.Path(sys.argv[4]))
print(f"Validated immutable Round-2 QC manifest: {p.valid_sample_count} samples")' \
  "${VALID_MANIFEST}" "${CHECKPOINT}" "${DATASET_STATS}" \
  "${ROUND2_REFERENCE_METADATA}"
export ASRE_ROUND3A_TRUSTED_PREFLIGHT_MANIFEST_SHA256="$("${PYTHON_BIN}" -c \
  'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "${VALID_MANIFEST}")"

mkdir -p "${OFFLINE_DIR}" "${LOG_DIR}"

EXPECTED_SAMPLES="$("${PYTHON_BIN}" -c \
  'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["valid_sample_ids"]))' \
  "${VALID_MANIFEST}")"
printf 'Fixed valid state population: %s samples.\n' "${EXPECTED_SAMPLES}"

condition_complete() {
  local condition="$1"
  local enabled_schedule="$2"
  local physical_gpu="$3"
  local summary_path="${OFFLINE_DIR}/${condition}/summary.csv"
  local metadata_path="${OFFLINE_DIR}/${condition}/run_metadata.json"
  local per_sample_path="${OFFLINE_DIR}/${condition}/per_sample.jsonl"
  [[ -r "${summary_path}" && -r "${metadata_path}" && \
     -r "${per_sample_path}" ]] || return 1
  "${PYTHON_BIN}" -c \
    'import csv,hashlib,json,math,pathlib,subprocess,sys
summary_path,metadata_path,per_sample_path,manifest_path,condition,enabled_json,physical_gpu,expected,checkpoint,stats,repo=sys.argv[1:]
try:
    rows=list(csv.DictReader(open(summary_path,encoding="utf-8")))
    meta=json.load(open(metadata_path,encoding="utf-8"))
    manifest=json.load(open(manifest_path,encoding="utf-8"))
    records=[json.loads(line) for line in open(per_sample_path,encoding="utf-8") if line.strip()]
    if not isinstance(meta,dict) or not isinstance(manifest,dict) or not all(isinstance(record,dict) for record in records):
        raise TypeError("Offline artifacts must contain JSON objects")
    enabled=[int(value) for value in json.loads(enabled_json)]
    disabled=[index for index in range(30) if index not in set(enabled)]
    valid_ids=[str(value) for value in manifest["valid_sample_ids"]]
    current_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    manifest_sha=hashlib.sha256(pathlib.Path(manifest_path).read_bytes()).hexdigest()
    summary=rows[0] if len(rows)==1 else {}
    summary_ok=(
        summary.get("condition")==condition
        and json.loads(summary.get("enabled_video_retrieval_layers","null"))==enabled
        and json.loads(summary.get("disabled_video_layers","null"))==disabled
        and int(summary.get("num_samples",-1))==int(expected)
        and int(summary.get("executed_prefix_length",-1))==10
    )
    metadata_expected={
        "status":"complete",
        "condition_protocol":"round3a_late_factorial",
        "diagnosis_condition":condition,
        "enabled_video_retrieval_layers":enabled,
        "disabled_video_layers":disabled,
        "num_model_layers":30,
        "git_commit_hash":current_commit,
        "checkpoint_path":str(pathlib.Path(checkpoint).resolve()),
        "checkpoint_sha256":manifest["checkpoint_sha256"],
        "dataset_stats_path":str(pathlib.Path(stats).resolve()),
        "dataset_stats_sha256":manifest["dataset_stats_sha256"],
        "state_bank_manifest_path":str(pathlib.Path(manifest["source_manifest_path"]).resolve()),
        "state_bank_manifest_sha256":manifest["source_manifest_sha256"],
        "valid_state_bank_manifest_path":str(pathlib.Path(manifest_path).resolve()),
        "valid_state_bank_manifest_sha256":manifest_sha,
        "prompt_context_cache_path":str(pathlib.Path(manifest["prompt_context_cache_path"]).resolve()),
        "prompt_context_cache_sha256":manifest["prompt_context_cache_sha256"],
        "num_valid_samples":int(expected),
        "num_samples":int(expected),
        "executed_prefix_length":10,
        "replan_steps":10,
        "condition_index":int(physical_gpu),
        "physical_gpu":int(physical_gpu),
        "cuda_visible_devices":str(physical_gpu),
        "model_device":"cuda:0",
    }
    metadata_ok=all(meta.get(key)==value for key,value in metadata_expected.items())
    scalar_metrics=(
        "executed_prefix_norm_rms",
        "norm_rms_h0",
        "norm_rms_h0_h1",
        "full_chunk_norm_rms_0_31",
        "round1_raw_output_full_chunk_rms",
        "executed_prefix_cosine_similarity",
        "executed_prefix_gripper_flip_rate",
        "full_horizon_gripper_flip_rate",
        "translation_norm_rms",
        "rotation_norm_rms",
    )
    dimension_metric="executed_prefix_norm_rms_by_dimension"
    dimension_names=[
        "delta_position_x","delta_position_y","delta_position_z",
        "delta_axis_angle_x","delta_axis_angle_y","delta_axis_angle_z",
    ]
    def finite(value):
        return math.isfinite(float(value))
    metric_schema_ok=all(
        all(metric in record and finite(record[metric]) for metric in scalar_metrics)
        and record.get("action_dimension_names")==dimension_names
        and isinstance(record.get(dimension_metric),list)
        and len(record[dimension_metric])==6
        and all(finite(value) for value in record[dimension_metric])
        for record in records
    )
    summary_metrics_ok=all(
        metric in summary
        and finite(summary[metric])
        and math.isclose(
            float(summary[metric]),
            math.fsum(float(record[metric]) for record in records)/len(records),
            rel_tol=1e-7,
            abs_tol=1e-12,
        )
        for metric in scalar_metrics
    )
    summary_dimensions=json.loads(summary.get(dimension_metric,"null"))
    summary_dimension_ok=(
        isinstance(summary_dimensions,list)
        and len(summary_dimensions)==6
        and all(finite(value) for value in summary_dimensions)
        and all(
            math.isclose(
                float(summary_dimensions[index]),
                math.fsum(float(record[dimension_metric][index]) for record in records)/len(records),
                rel_tol=1e-7,
                abs_tol=1e-12,
            )
            for index in range(6)
        )
    )
    records_ok=(
        len(records)==int(expected)
        and [str(record.get("sample_id","")) for record in records]==valid_ids
        and all(
            record.get("condition")==condition
            and record.get("enabled_video_retrieval_layers")==enabled
            and record.get("disabled_video_layers")==disabled
            and int(record.get("executed_prefix_length",-1))==10
            for record in records
        )
    )
    ok=(summary_ok and metadata_ok and records_ok and metric_schema_ok
        and summary_metrics_ok and summary_dimension_ok
        and len(valid_ids)==int(expected))
except (OSError,ValueError,TypeError,KeyError,AttributeError,ZeroDivisionError,json.JSONDecodeError,subprocess.SubprocessError):
    ok=False
sys.exit(0 if ok else 1)' \
    "${summary_path}" "${metadata_path}" "${per_sample_path}" \
    "${VALID_MANIFEST}" "${condition}" "${enabled_schedule}" \
    "${physical_gpu}" "${EXPECTED_SAMPLES}" "${CHECKPOINT}" \
    "${DATASET_STATS}" "${REPO_ROOT}"
}

declare -a needs_launch=()

# Complete the safety preflight for all three directories before starting any
# GPU process.  This avoids orphaning earlier jobs if a later directory is
# partial or belongs to stale provenance.
for condition_index in "${!CONDITIONS[@]}"; do
  condition="${CONDITIONS[$condition_index]}"
  enabled_schedule="${ENABLED_SCHEDULES[$condition_index]}"
  condition_dir="${OFFLINE_DIR}/${condition}"
  if condition_complete "${condition}" "${enabled_schedule}" "${condition_index}"; then
    printf 'Skipping completed %s (%s samples).\n' "${condition}" "${EXPECTED_SAMPLES}"
    needs_launch[$condition_index]=0
    continue
  fi
  if [[ -d "${condition_dir}" ]]; then
    if [[ -n "$(find "${condition_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      printf 'Partial/unknown offline directory exists; refusing overwrite: %s\n' \
        "${condition_dir}" >&2
      exit 1
    fi
  fi
  needs_launch[$condition_index]=1
done

declare -a pids=()
declare -a launched_conditions=()
declare -a launched_logs=()

terminate_children() {
  local pid
  local any_live
  local _attempt
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  for _attempt in {1..15}; do
    any_live=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_live=1
      fi
    done
    if (( any_live == 0 )); then
      break
    fi
    sleep 1
  done
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}

handle_signal() {
  local exit_status="$1"
  trap - EXIT INT TERM
  terminate_children
  exit "${exit_status}"
}

handle_exit() {
  local exit_status="$?"
  trap - EXIT INT TERM
  if (( exit_status != 0 )); then
    terminate_children
  fi
  exit "${exit_status}"
}

trap handle_exit EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

for condition_index in "${!CONDITIONS[@]}"; do
  if (( needs_launch[$condition_index] == 0 )); then
    continue
  fi
  condition="${CONDITIONS[$condition_index]}"
  enabled_schedule="${ENABLED_SCHEDULES[$condition_index]}"
  condition_dir="${OFFLINE_DIR}/${condition}"
  log_path="${LOG_DIR}/offline_${condition_index}_${condition}.log"
  printf 'Launching %s on physical GPU %s; log=%s\n' \
    "${condition}" "${condition_index}" "${log_path}"
  env \
    -u MUJOCO_EGL_DEVICE_ID \
    -u MUJOCO_GL \
    -u PYOPENGL_PLATFORM \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${condition_index}" \
    "${PYTHON_BIN}" experiments/asre_diagnosis/round3a/replay_state_bank.py \
      task=libero_uncond_2cam224_1e-4 \
      ckpt="${CHECKPOINT}" \
      model.load_text_encoder=false \
      EVALUATION.device=cuda:0 \
      EVALUATION.dataset_stats_path="${DATASET_STATS}" \
      ASRE_DIAGNOSIS.enabled=true \
      ASRE_DIAGNOSIS.protocol=round3a_late_factorial \
      ASRE_DIAGNOSIS.condition_name="${condition}" \
      ASRE_DIAGNOSIS.enabled_video_retrieval_layers="${enabled_schedule}" \
      ASRE_DIAGNOSIS.state_bank_dir="${STATE_BANK_DIR}" \
      ASRE_DIAGNOSIS.valid_state_bank_manifest_path="${VALID_MANIFEST}" \
      ASRE_DIAGNOSIS.offline_output_dir="${OFFLINE_DIR}" \
      > "${log_path}" 2>&1 &
  pids+=("$!")
  launched_conditions+=("${condition}")
  launched_logs+=("${log_path}")
done

failed=0
for process_index in "${!pids[@]}"; do
  process_pid="${pids[$process_index]}"
  if wait "${process_pid}"; then
    status=0
  else
    status="$?"
  fi
  # A reaped PID can be reused by the OS. Remove it immediately so the EXIT
  # trap can never signal an unrelated process after a later replay fails.
  unset "pids[$process_index]"
  if (( status == 0 )); then
    printf '%s: exit=0\n' "${launched_conditions[$process_index]}"
  else
    printf '%s: exit=%s; log=%s\n' \
      "${launched_conditions[$process_index]}" "${status}" \
      "${launched_logs[$process_index]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  printf 'At least one Round-3A offline replay failed.\n' >&2
  exit 1
fi
for condition_index in "${!CONDITIONS[@]}"; do
  condition="${CONDITIONS[$condition_index]}"
  if ! condition_complete "${condition}" "${ENABLED_SCHEDULES[$condition_index]}" \
    "${condition_index}"; then
    printf 'Missing/incomplete Round-3A offline condition: %s\n' "${condition}" >&2
    exit 1
  fi
done
trap - EXIT INT TERM
printf 'Round-3A immutable-manifest validation and all three offline replays completed successfully.\n'
