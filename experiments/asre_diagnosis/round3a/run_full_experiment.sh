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
ROUND3A_ROOT="${ASRE_ROUND3A_OUTPUT_ROOT:-${ROUND1_ROOT}/round3a}"
VALID_MANIFEST="${ROUND2_ROOT}/state_bank_valid_manifest.json"
ROUND2_ONLINE_ROOT="${ROUND2_ROOT}/online_full"
ROUND2_OFFLINE_ROOT="${ROUND2_ROOT}/offline"
ROUND2_REFERENCE_METADATA="${ROUND2_ONLINE_ROOT}/keep_15_29/run_metadata.json"
SMOKE_ROOT="${ROUND3A_ROOT}/online_smoke"
FULL_ROOT="${ROUND3A_ROOT}/online_full"
OFFLINE_ROOT="${ROUND3A_ROOT}/offline"
AGGREGATE_ROOT="${ROUND3A_ROOT}/aggregate"

LIBERO_ROOT="${LIBERO_ROOT:-${REPO_ROOT}/../LIBERO}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${LIBERO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${ROUND3A_ROOT}/.matplotlib"
export PYTHON_BIN CHECKPOINT DATASET_STATS LIBERO_ROOT
export ASRE_ROUND1_OUTPUT_ROOT="${ROUND1_ROOT}"
export ASRE_ROUND2_OUTPUT_ROOT="${ROUND2_ROOT}"
export ASRE_ROUND3A_OUTPUT_ROOT="${ROUND3A_ROOT}"

for required_path in \
  "${VALID_MANIFEST}" \
  "${ROUND2_ONLINE_ROOT}/launcher_summary.json" \
  "${ROUND2_REFERENCE_METADATA}" \
  "${ROUND2_OFFLINE_ROOT}/keep_15_29/run_metadata.json"; do
  if [[ ! -r "${required_path}" ]]; then
    printf 'Required frozen Round-2 input is unavailable: %s\n' "${required_path}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" experiments/asre_diagnosis/round3a/preflight_round2.py \
  --round2-online-root "${ROUND2_ONLINE_ROOT}" \
  --round2-offline-root "${ROUND2_OFFLINE_ROOT}" \
  --valid-manifest "${VALID_MANIFEST}"

"${SCRIPT_DIR}/run_offline.sh"

"${PYTHON_BIN}" experiments/asre_diagnosis/round3a/launch_three_gpu.py \
  --mode smoke \
  --python "${PYTHON_BIN}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-stats-path "${DATASET_STATS}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --round2-reference-metadata "${ROUND2_REFERENCE_METADATA}" \
  --output-root "${SMOKE_ROOT}"

"${PYTHON_BIN}" experiments/asre_diagnosis/round3a/launch_three_gpu.py \
  --mode full \
  --python "${PYTHON_BIN}" \
  --checkpoint "${CHECKPOINT}" \
  --dataset-stats-path "${DATASET_STATS}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --round2-reference-metadata "${ROUND2_REFERENCE_METADATA}" \
  --output-root "${FULL_ROOT}" \
  --smoke-summary "${SMOKE_ROOT}/launcher_summary.json"

directory_nonempty() {
  local path="$1"
  [[ -d "${path}" && -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

aggregate_complete() {
  "${PYTHON_BIN}" -c \
    'import json,pathlib,subprocess,sys
root,repo,r2online,r2offline,r3online,r3offline,valid=map(pathlib.Path,sys.argv[1:])
required=(
    "factorial_cells.csv","factorial_cells.json",
    "factorial_simple_effects.csv","factorial_interactions.csv",
    "factorial_task_success.csv","factorial_offline_metrics.csv",
    "factorial_paired_transitions.csv","factorial_replan_stage_metrics.csv",
    "factorial_summary.json","factorial_summary.md","aggregate_metadata.json",
)
expected_paths={
    "round2_online_root":r2online,
    "round2_offline_root":r2offline,
    "round3a_online_root":r3online,
    "round3a_offline_root":r3offline,
    "valid_state_bank_manifest_path":valid,
}
try:
    meta=json.loads((root/"aggregate_metadata.json").read_text(encoding="utf-8"))
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    paths_ok=all(
        pathlib.Path(meta.get(key,"")).resolve()==path.resolve()
        for key,path in expected_paths.items()
    )
    ok=(
        meta.get("status")=="complete"
        and meta.get("analysis_git_commit_hash")==commit
        and paths_ok
        and all((root/name).is_file() and (root/name).stat().st_size>0 for name in required)
    )
except (OSError,ValueError,TypeError,json.JSONDecodeError,subprocess.SubprocessError):
    ok=False
sys.exit(0 if ok else 1)' \
    "${AGGREGATE_ROOT}" "${REPO_ROOT}" \
    "${ROUND2_ONLINE_ROOT}" "${ROUND2_OFFLINE_ROOT}" \
    "${FULL_ROOT}" "${OFFLINE_ROOT}" "${VALID_MANIFEST}"
}

if directory_nonempty "${AGGREGATE_ROOT}"; then
  if ! aggregate_complete; then
    printf 'Existing Round-3A aggregate is partial or incompatible; refusing overwrite: %s\n' \
      "${AGGREGATE_ROOT}" >&2
    exit 1
  fi
  printf 'Skipping immutable completed Round-3A aggregate: %s\n' "${AGGREGATE_ROOT}"
else
  "${PYTHON_BIN}" experiments/asre_diagnosis/round3a/aggregate_factorial.py \
    --round2-online-root "${ROUND2_ONLINE_ROOT}" \
    --round2-offline-root "${ROUND2_OFFLINE_ROOT}" \
    --round3a-online-root "${FULL_ROOT}" \
    --round3a-offline-root "${OFFLINE_ROOT}" \
    --valid-manifest "${VALID_MANIFEST}" \
    --output-dir "${AGGREGATE_ROOT}"
fi

PLOT_ROOT="${AGGREGATE_ROOT}/plots"
EXPECTED_PLOTS=(
  figure_A_complete_factorial_cell_plot.png
  figure_B_contextual_contribution_of_B.png
  figure_C_pairwise_interaction_summary.png
  figure_D_retrieval_schedule_schematic.png
  figure_E_task_level_factorial_heatmap.png
)
if directory_nonempty "${PLOT_ROOT}"; then
  for plot_name in "${EXPECTED_PLOTS[@]}"; do
    if [[ ! -s "${PLOT_ROOT}/${plot_name}" ]]; then
      printf 'Existing Round-3A plot directory is partial; refusing overwrite: %s\n' \
        "${PLOT_ROOT}" >&2
      exit 1
    fi
  done
  printf 'Skipping immutable completed Round-3A plots: %s\n' "${PLOT_ROOT}"
else
  "${PYTHON_BIN}" experiments/asre_diagnosis/round3a/plot_factorial.py \
    --aggregate-dir "${AGGREGATE_ROOT}"
fi

printf 'ASRE Round 3A factorial completion finished; stop rule reached.\nSummary: %s\nPlots: %s\n' \
  "${AGGREGATE_ROOT}/factorial_summary.md" "${PLOT_ROOT}"
