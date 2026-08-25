#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhaizicheng/miniconda3/envs/fastwam/bin/python}"
CHECKPOINT="${CHECKPOINT:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt}"
DATASET_STATS="${DATASET_STATS:-/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json}"
ROUND1_ROOT="${ASRE_ROUND1_OUTPUT_ROOT:-${REPO_ROOT}/asre_results}"
ROUND2_ROOT="${ASRE_ROUND2_OUTPUT_ROOT:-${ROUND1_ROOT}/round2}"
VALID_MANIFEST="${ROUND2_ROOT}/state_bank_valid_manifest.json"
SMOKE_ROOT="${ROUND2_ROOT}/online_smoke"
FULL_ROOT="${ROUND2_ROOT}/online_full"
OFFLINE_ROOT="${ROUND2_ROOT}/offline"
AGGREGATE_ROOT="${ROUND2_ROOT}/aggregate"
ROUND1_SUMMARY="${ASRE_ROUND1_SUMMARY:-${ROUND1_ROOT}/aggregate/summary.csv}"

LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/../LIBERO}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${ROUND2_ROOT}/.matplotlib"
export PYTHON_BIN CHECKPOINT DATASET_STATS LIBERO_ROOT
export ASRE_ROUND1_OUTPUT_ROOT="${ROUND1_ROOT}"
export ASRE_ROUND2_OUTPUT_ROOT="${ROUND2_ROOT}"

if [[ ! -r "${ROUND1_SUMMARY}" ]]; then
  printf 'Round-1 aggregate summary is required for joint interpretation: %s\n' \
    "${ROUND1_SUMMARY}" >&2
  exit 1
fi

"${SCRIPT_DIR}/run_offline.sh"

"${PYTHON_BIN}" experiments/asre_diagnosis/round2/launch_eight_gpu.py \
  --mode smoke \
  --python "${PYTHON_BIN}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-stats-path "${DATASET_STATS}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --output-root "${SMOKE_ROOT}"

"${PYTHON_BIN}" experiments/asre_diagnosis/round2/launch_eight_gpu.py \
  --mode full \
  --python "${PYTHON_BIN}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-stats-path "${DATASET_STATS}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --output-root "${FULL_ROOT}" \
  --smoke-summary "${SMOKE_ROOT}/launcher_summary.json"

directory_nonempty() {
  local path="$1"
  [[ -d "${path}" && -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

aggregate_complete() {
  "${PYTHON_BIN}" -c \
    'import json,pathlib,subprocess,sys
root,repo,online,offline,round1=map(pathlib.Path,sys.argv[1:])
required=("summary.csv","summary.json","paired_transitions.csv","task_success_rates.csv","task_success_delta.csv","offline_action_dimensions.csv","replan_stage_metrics.csv","planned_comparisons.csv","round1_round2_joint_summary.csv","diagnostic_summary.json","diagnostic_summary.md","aggregate_metadata.json")
try:
    meta=json.loads((root/"aggregate_metadata.json").read_text(encoding="utf-8"))
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    ok=(meta.get("status")=="complete" and meta.get("analysis_git_commit_hash")==commit and pathlib.Path(meta.get("online_root","")).resolve()==online.resolve() and pathlib.Path(meta.get("offline_root","")).resolve()==offline.resolve() and pathlib.Path(meta.get("round1_summary_path","")).resolve()==round1.resolve() and all((root/name).is_file() and (root/name).stat().st_size>0 for name in required))
except (OSError,ValueError,TypeError,json.JSONDecodeError,subprocess.SubprocessError):
    ok=False
sys.exit(0 if ok else 1)' \
    "${AGGREGATE_ROOT}" "${REPO_ROOT}" "${FULL_ROOT}" "${OFFLINE_ROOT}" \
    "${ROUND1_SUMMARY}"
}

if directory_nonempty "${AGGREGATE_ROOT}"; then
  if ! aggregate_complete; then
    printf 'Existing aggregate directory is partial or incompatible; refusing overwrite: %s\n' \
      "${AGGREGATE_ROOT}" >&2
    exit 1
  fi
  printf 'Skipping immutable completed Round-2 aggregate: %s\n' "${AGGREGATE_ROOT}"
else
  "${PYTHON_BIN}" experiments/asre_diagnosis/round2/aggregate_results.py \
    --online-root "${FULL_ROOT}" \
    --offline-root "${OFFLINE_ROOT}" \
    --round1-summary "${ROUND1_SUMMARY}" \
    --output-dir "${AGGREGATE_ROOT}"
fi

PLOT_ROOT="${AGGREGATE_ROOT}/plots"
if directory_nonempty "${PLOT_ROOT}"; then
  expected_plots=(
    figure_1_retrieval_schedule_vs_behavioral_performance.png
    figure_2_retrieval_schedule_vs_executed_action_sensitivity.png
    figure_3_enabled_retrieval_depth_schematic.png
    figure_4_task_by_retrieval_schedule_heatmap.png
    figure_5_critical_window_complementarity.png
    exploratory_replan_stage_action_deviation.png
  )
  for plot_name in "${expected_plots[@]}"; do
    if [[ ! -s "${PLOT_ROOT}/${plot_name}" ]]; then
      printf 'Existing plot directory is partial; refusing overwrite: %s\n' \
        "${PLOT_ROOT}" >&2
      exit 1
    fi
  done
  printf 'Skipping immutable completed Round-2 plots: %s\n' "${PLOT_ROOT}"
else
  "${PYTHON_BIN}" experiments/asre_diagnosis/round2/plot_results.py \
    --aggregate-dir "${AGGREGATE_ROOT}"
fi

printf 'ASRE Round 2 completed.\nSummary: %s\nPlots: %s\n' \
  "${AGGREGATE_ROOT}/summary.csv" "${AGGREGATE_ROOT}/plots"
