#!/usr/bin/env bash
set -euo pipefail

# Reproduce the complete, frozen move_block_reveal_can experiment.
# Usage:
#   RUN_ROOT=/absolute/output/path bash scripts/run_move_block_reveal_can.sh all
# Stages: gpu_preflight | diagnose_contacts | simulator_reset | simulator_expert |
#         simulator_calibration_expert | freeze | simulator |
#         calibration_place | calibration_block | admission | calibration | pilot |
#         pilot_visible | pilot_missing | pilot_oracle_reveal | aggregate | all

PROJECT_ROOT="/home/zhaizicheng/Reasoning/FastWAM"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROBOTWIN_ROOT="$PROJECT_ROOT/third_party/RoboTwin"
ASSET_DONOR="/local_home/zhaizicheng/fastwam_robotwin/RoboTwin/assets"
PYTHON="/home/zhaizicheng/miniconda3/envs/fastwam-robotwin/bin/python"
CHECKPOINT="/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt"
DATASET_STATS="/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
WAN_BASE="/local_home/zhaizicheng/fastwam_assets/checkpoints/wan_base"
POLICY_SOURCE="$PROJECT_ROOT/experiments/robotwin/fastwam_policy"
POLICY_LINK="$ROBOTWIN_ROOT/policy/fastwam_policy"
SEED_MANIFEST="$PROJECT_ROOT/experiments/robotwin/move_block_reveal_can_seeds.json"
PLACE_CAN_CALIBRATION_SEED_MANIFEST="$PROJECT_ROOT/experiments/robotwin/place_can_basket_calibration_seeds.json"
BLOCK_CALIBRATION_SEED_MANIFEST="$PROJECT_ROOT/experiments/robotwin/move_block_reveal_can_block_calibration_seeds.json"
FORMAL_FREEZE_NAME="MINIBENCH_TASK_FREEZE_V0_3_11"
CALIBRATION_EPISODES=5
SIM_GPU="${MINIBENCH_SIM_GPU:-6}"
POLICY_GPU="${MINIBENCH_POLICY_GPU:-7}"
# SAPIEN is not constrained by CVD, so resolve and pin the selected physical
# simulator GPU by PCI address after the no-overlap guard has identified it.
ROBOTWIN_RENDER_DEVICE="${MINIBENCH_RENDER_DEVICE:-}"
SIM_GPU_UUID=""
POLICY_GPU_UUID=""
GPU_PAIR=""
PROJECT_PREFREEZE_JSON="$PROJECT_ROOT/experiments/robotwin/${FORMAL_FREEZE_NAME}.json"
PROJECT_PREFREEZE_SHA="$PROJECT_ROOT/experiments/robotwin/${FORMAL_FREEZE_NAME}.sha256"

STAGE="${1:-all}"
RUN_ROOT_RAW="${RUN_ROOT:-$PROJECT_ROOT/asre_results/move_block_reveal_can/run_$(date +%Y%m%d_%H%M%S)}"
if [[ "$RUN_ROOT_RAW" = /* ]]; then
  RUN_ROOT="$(readlink -m "$RUN_ROOT_RAW")"
else
  RUN_ROOT="$(readlink -m "$PWD/$RUN_ROOT_RAW")"
fi
mkdir -p "$RUN_ROOT/metadata/gpu_checks" "$RUN_ROOT/logs"

cd "$PROJECT_ROOT"

ensure_python() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: required fastwam-robotwin Python is missing: $PYTHON" >&2
    exit 1
  fi
}

ensure_project_prefreeze() {
  if [[ ! -s "$PROJECT_PREFREEZE_JSON" || ! -s "$PROJECT_PREFREEZE_SHA" ]]; then
    echo "ERROR: project pre-freeze manifest/checksum is missing" >&2
    exit 1
  fi
  sha256sum --check "$PROJECT_PREFREEZE_SHA"
}

ensure_runtime_assets() {
  if [[ ! -e "$ROBOTWIN_ROOT/assets" && ! -L "$ROBOTWIN_ROOT/assets" ]]; then
    ln -s "$ASSET_DONOR" "$ROBOTWIN_ROOT/assets"
  fi
  if [[ "$(readlink -f "$ROBOTWIN_ROOT/assets")" != "$(readlink -f "$ASSET_DONOR")" ]]; then
    echo "ERROR: vendored RoboTwin assets do not resolve to the declared read-only asset donor" >&2
    exit 1
  fi
}

ensure_model_inputs() {
  if [[ ! -f "$CHECKPOINT" || ! -f "$DATASET_STATS" ]]; then
    echo "ERROR: required checkpoint or dataset stats file is missing" >&2
    exit 1
  fi
  if [[ ! -f "$WAN_BASE/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" || \
        ! -f "$WAN_BASE/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" || \
        ! -f "$WAN_BASE/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json" ]]; then
    echo "ERROR: required local Wan model base is incomplete: $WAN_BASE" >&2
    exit 1
  fi
}

ensure_policy_link() {
  if [[ -L "$POLICY_LINK" ]]; then
    if [[ "$(readlink -f "$POLICY_LINK")" != "$(readlink -f "$POLICY_SOURCE")" ]]; then
      echo "ERROR: policy symlink points outside the sole FastWAM policy source" >&2
      exit 1
    fi
  elif [[ -e "$POLICY_LINK" ]]; then
    echo "ERROR: policy runtime path exists and is not the expected symlink: $POLICY_LINK" >&2
    exit 1
  else
    ln -s "$POLICY_SOURCE" "$POLICY_LINK"
  fi
}

gpu_guard() {
  local label="$1"
  shift
  local stamp evidence_base snapshot map_file compute_file pmon_file
  local index uuid busy graphics_busy full_busy
  stamp="$(date +%Y%m%d_%H%M%S_%N)"
  evidence_base="$RUN_ROOT/metadata/gpu_checks/${stamp}_${label}"
  snapshot="${evidence_base}_nvidia_smi.txt"
  map_file="${evidence_base}_gpu_map.csv"
  compute_file="${evidence_base}_compute_apps.csv"
  pmon_file="${evidence_base}_pmon.txt"
  nvidia-smi | tee "$snapshot"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits > "$map_file"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits > "$compute_file"
  nvidia-smi pmon -c 1 > "$pmon_file"
  for index in "$@"; do
    uuid="$(awk -F, -v wanted="$index" '$1 + 0 == wanted {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' "$map_file")"
    if [[ -z "$uuid" ]]; then
      echo "ERROR: GPU index $index was not found" >&2
      return 1
    fi
    busy="$(awk -F, -v wanted="$uuid" '$1 ~ wanted {print}' "$compute_file")"
    if [[ -n "$busy" ]]; then
      echo "ERROR: refusing to launch because GPU $index has a compute process: $busy" >&2
      return 1
    fi
    # The tabular process section of plain `nvidia-smi` retains inactive
    # graphics contexts that `pmon` can intermittently omit between frames.
    full_busy="$(awk -v wanted="$index" \
      '$1 == "|" && $2 == wanted && $5 ~ /^[0-9]+$/ && $7 !~ /Xorg/ {print}' \
      "$snapshot")"
    if [[ -n "$full_busy" ]]; then
      echo "ERROR: refusing to launch because GPU $index has a process/context: $full_busy" >&2
      return 1
    fi
    graphics_busy="$(awk -v wanted="$index" \
      '$1 !~ /^#/ && $1 == wanted && $2 != "-" && $10 !~ /^Xorg/ {print}' "$pmon_file")"
    if [[ -n "$graphics_busy" ]]; then
      echo "ERROR: refusing to launch because GPU $index has a non-Xorg process: $graphics_busy" >&2
      return 1
    fi
  done
  echo "GPU_GUARD_OK label=$label indices=$*" | tee -a "$snapshot"
}

resolve_gpu_mapping() {
  local observed resolved_render index
  for index in "$SIM_GPU" "$POLICY_GPU"; do
    if [[ ! "$index" =~ ^[4-7]$ ]]; then
      echo "ERROR: MiniBench GPU index must be one of the user-approved cards 4-7: $index" >&2
      exit 1
    fi
  done
  if [[ "$SIM_GPU" == "$POLICY_GPU" ]]; then
    echo "ERROR: simulator and policy GPUs must be distinct" >&2
    exit 1
  fi
  observed="$(nvidia-smi -i "$SIM_GPU" --query-gpu=pci.bus_id --format=csv,noheader,nounits)"
  if [[ "$observed" == 00000000:* ]]; then
    observed="0000:${observed#00000000:}"
  fi
  observed="${observed,,}"
  resolved_render="pci:$observed"
  if [[ -n "$ROBOTWIN_RENDER_DEVICE" && "$resolved_render" != "$ROBOTWIN_RENDER_DEVICE" ]]; then
    echo "ERROR: GPU $SIM_GPU maps to $resolved_render, expected $ROBOTWIN_RENDER_DEVICE" >&2
    exit 1
  fi
  ROBOTWIN_RENDER_DEVICE="$resolved_render"
  SIM_GPU_UUID="$(nvidia-smi -i "$SIM_GPU" --query-gpu=uuid --format=csv,noheader,nounits)"
  POLICY_GPU_UUID="$(nvidia-smi -i "$POLICY_GPU" --query-gpu=uuid --format=csv,noheader,nounits)"
  GPU_PAIR="$SIM_GPU_UUID,$POLICY_GPU_UUID"
}

gpu_isolated_exec() {
  local gpu_indices="$1"
  shift
  local index pci_full pci minor nvidia_node card_node render_node common_node
  local expected_nvidia="" expected_dri=""
  local sandbox=(
    bwrap
    --bind / /
    --unshare-pid
    --unshare-ipc
    --die-with-parent
    --dev /dev
    --proc /proc
    --tmpfs /dev/shm
  )

  for common_node in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset; do
    [[ -e "$common_node" ]] && sandbox+=(--dev-bind "$common_node" "$common_node")
  done
  sandbox+=(--dir /dev/dri)

  IFS=',' read -r -a requested_indices <<< "$gpu_indices"
  for index in "${requested_indices[@]}"; do
    pci_full="$(nvidia-smi -i "$index" --query-gpu=pci.bus_id --format=csv,noheader,nounits)"
    pci="$pci_full"
    if [[ "$pci" == 00000000:* ]]; then
      pci="0000:${pci#00000000:}"
    fi
    pci="${pci,,}"
    minor="$(awk -F: '/Device Minor/ {gsub(/[ \t]/, "", $2); print $2}' \
      "/proc/driver/nvidia/gpus/$pci/information")"
    nvidia_node="/dev/nvidia$minor"
    card_node="$(readlink -f "/dev/dri/by-path/pci-$pci-card")"
    render_node="$(readlink -f "/dev/dri/by-path/pci-$pci-render")"
    expected_nvidia+="${nvidia_node##*/}"$'\n'
    expected_dri+="${card_node##*/}"$'\n'"${render_node##*/}"$'\n'
    for common_node in "$nvidia_node" "$card_node" "$render_node"; do
      if [[ ! -e "$common_node" ]]; then
        echo "ERROR: GPU $index device node is missing: $common_node" >&2
        exit 1
      fi
      sandbox+=(--dev-bind "$common_node" "$common_node")
    done
  done
  sandbox+=(
    --setenv MINIBENCH_EXPECTED_NVIDIA_NODES "$expected_nvidia"
    --setenv MINIBENCH_EXPECTED_DRI_NODES "$expected_dri"
  )
  "${sandbox[@]}" -- /bin/bash -c '
    set -euo pipefail
    expected_nvidia="$(printf "%s" "$MINIBENCH_EXPECTED_NVIDIA_NODES" | sed "/^$/d" | sort)"
    actual_nvidia="$(compgen -G "/dev/nvidia[0-9]*" | sed "s#^/dev/##" | sort)"
    expected_dri="$(printf "%s" "$MINIBENCH_EXPECTED_DRI_NODES" | sed "/^$/d" | sort)"
    actual_dri="$({ compgen -G "/dev/dri/card*"; compgen -G "/dev/dri/renderD*"; } | sed "s#^/dev/dri/##" | sort)"
    if [[ "$actual_nvidia" != "$expected_nvidia" || "$actual_dri" != "$expected_dri" ]]; then
      echo "ERROR: isolated GPU device-node audit failed" >&2
      echo "expected NVIDIA: $expected_nvidia" >&2
      echo "actual NVIDIA: $actual_nvidia" >&2
      echo "expected DRI: $expected_dri" >&2
      echo "actual DRI: $actual_dri" >&2
      exit 1
    fi
    echo "GPU_DEVICE_NAMESPACE_OK nvidia=$(echo "$actual_nvidia" | paste -sd, -) dri=$(echo "$actual_dri" | paste -sd, -)"
    exec "$@"
  ' minibench-gpu-isolation "$@"
}

capture_diff() {
  ensure_model_inputs
  local patch_file="$RUN_ROOT/metadata/git_diff_at_freeze.patch"
  local checksum_file="$RUN_ROOT/metadata/${FORMAL_FREEZE_NAME}.sha256"
  git status --short > "$RUN_ROOT/metadata/git_status_at_freeze.txt"
  git diff --binary HEAD -- . > "$patch_file"
  local path rc
  for path in \
    experiments/robotwin/MINIBENCH_TASK_FREEZE_V0_3_11.json \
    experiments/robotwin/MINIBENCH_TASK_FREEZE_V0_3_11.sha256 \
    experiments/robotwin/move_block_reveal_can_seeds.json \
    experiments/robotwin/place_can_basket_calibration_seeds.json \
    experiments/robotwin/move_block_reveal_can_block_calibration_seeds.json \
    experiments/robotwin/move_block_reveal_can_insertion_geometry_audit.json \
    experiments/robotwin/move_block_reveal_can_transfer_geometry_audit.json \
    experiments/robotwin/move_block_reveal_can_native_parent_expert_audit.json \
    experiments/robotwin/move_block_reveal_can_success_predicate_audit.json \
    experiments/robotwin/move_block_reveal_can_expert_solvability_audit.json \
    experiments/robotwin/test_move_block_reveal_can.py \
    experiments/robotwin/summarize_move_block_reveal_can.py \
    scripts/run_move_block_reveal_can.sh \
    third_party/RoboTwin/description/task_instruction/move_block_reveal_can.json \
    third_party/RoboTwin/description/task_instruction/move_block_reveal_can_block_calibration.json \
    third_party/RoboTwin/envs/move_block_reveal_can.py \
    third_party/RoboTwin/envs/move_block_reveal_can_block_calibration.py \
    third_party/RoboTwin/task_config/_camera_config.yml \
    third_party/RoboTwin/task_config/_embodiment_config.yml \
    third_party/RoboTwin/task_config/_eval_step_limit.yml \
    third_party/RoboTwin/task_config/demo_clean.yml \
    third_party/RoboTwin/task_config/demo_randomized.yml; do
    if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
      continue
    fi
    if [[ -e "$path" ]]; then
      rc=0
      git diff --binary --no-index /dev/null "$path" >> "$patch_file" || rc=$?
      if [[ "$rc" -ne 0 && "$rc" -ne 1 ]]; then
        return "$rc"
      fi
    fi
  done
  cp "$SEED_MANIFEST" "$RUN_ROOT/metadata/paired_seeds.json"
  cp "$PLACE_CAN_CALIBRATION_SEED_MANIFEST" \
    "$RUN_ROOT/metadata/place_can_basket_calibration_seeds.json"
  cp "$BLOCK_CALIBRATION_SEED_MANIFEST" \
    "$RUN_ROOT/metadata/move_block_reveal_can_block_calibration_seeds.json"
  cp "$PROJECT_PREFREEZE_JSON" "$RUN_ROOT/metadata/project_prefreeze_manifest.json"
  cp "$PROJECT_PREFREEZE_SHA" "$RUN_ROOT/metadata/project_prefreeze_manifest.sha256"
  cp "$SCRIPT_PATH" "$RUN_ROOT/metadata/reproduction_command.sh"
  readlink -f "$ROBOTWIN_ROOT/assets" > "$RUN_ROOT/metadata/asset_donor_path.txt"
  printf '%s -> %s\n' "$POLICY_LINK" "$(readlink -f "$POLICY_LINK")" \
    > "$RUN_ROOT/metadata/policy_symlink.txt"
  sha256sum "$CHECKPOINT" "$DATASET_STATS" > "$RUN_ROOT/metadata/model_inputs.sha256"
  if [[ ! -f "$checksum_file" ]]; then
    echo "ERROR: formal freeze checksum is missing before provenance capture: $checksum_file" >&2
    exit 1
  fi
  sha256sum \
    "$RUN_ROOT/metadata/paired_seeds.json" \
    "$RUN_ROOT/metadata/place_can_basket_calibration_seeds.json" \
    "$RUN_ROOT/metadata/move_block_reveal_can_block_calibration_seeds.json" \
    "$RUN_ROOT/metadata/project_prefreeze_manifest.json" \
    "$RUN_ROOT/metadata/project_prefreeze_manifest.sha256" \
    "$RUN_ROOT/metadata/reproduction_command.sh" \
    "$RUN_ROOT/metadata/git_diff_at_freeze.patch" \
    "$RUN_ROOT/metadata/git_status_at_freeze.txt" \
    "$RUN_ROOT/metadata/model_inputs.sha256" \
    "$RUN_ROOT/metadata/asset_donor_path.txt" \
    "$RUN_ROOT/metadata/policy_symlink.txt" >> "$checksum_file"
}

verify_freeze() {
  local checksum_file="$RUN_ROOT/metadata/${FORMAL_FREEZE_NAME}.sha256"
  local amendment_dir="$RUN_ROOT/metadata/post_freeze_text_encoder_split"
  local amendment_json="$amendment_dir/POST_FREEZE_ENGINEERING_AMENDMENT_0001.json"
  if [[ ! -f "$checksum_file" ]]; then
    echo "ERROR: freeze checksum is missing: $checksum_file" >&2
    exit 1
  fi
  local required
  for required in \
    "$RUN_ROOT/metadata/${FORMAL_FREEZE_NAME}.json" \
    "$RUN_ROOT/metadata/simulator_provenance.json" \
    "$RUN_ROOT/simulator/reset_audit.json" \
    "$RUN_ROOT/simulator/expert_audit.json" \
    "$RUN_ROOT/simulator/calibration_expert_audit.json" \
    "$RUN_ROOT/simulator/screenshots.sha256" \
    "$RUN_ROOT/metadata/model_inputs.sha256" \
    "$RUN_ROOT/metadata/paired_seeds.json" \
    "$RUN_ROOT/metadata/place_can_basket_calibration_seeds.json" \
    "$RUN_ROOT/metadata/move_block_reveal_can_block_calibration_seeds.json" \
    "$RUN_ROOT/metadata/project_prefreeze_manifest.json" \
    "$RUN_ROOT/metadata/project_prefreeze_manifest.sha256" \
    "$RUN_ROOT/metadata/reproduction_command.sh"; do
    if [[ ! -s "$required" ]]; then
      echo "ERROR: required freeze artifact is missing or empty: $required" >&2
      exit 1
    fi
  done
  for required in \
    "$RUN_ROOT/metadata/git_diff_at_freeze.patch" \
    "$RUN_ROOT/metadata/git_status_at_freeze.txt"; do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: required freeze artifact is missing: $required" >&2
      exit 1
    fi
  done
  if [[ -s "$amendment_json" ]]; then
    "$PYTHON" -u experiments/robotwin/verify_move_block_reveal_can_amendment.py \
      --run-root "$RUN_ROOT" \
      --freeze-name "$FORMAL_FREEZE_NAME" \
      --amendment "$amendment_json"
  else
    sha256sum --check "$checksum_file"
  fi
  sha256sum --check "$RUN_ROOT/simulator/screenshots.sha256"
  sha256sum --check "$RUN_ROOT/metadata/model_inputs.sha256"
}

run_gpu_preflight() {
  resolve_gpu_mapping
  gpu_guard gpu_preflight "$SIM_GPU" "$POLICY_GPU"
  gpu_isolated_exec "$SIM_GPU,$POLICY_GPU" /bin/true
}

prepare_simulator_stage() {
  ensure_python
  ensure_project_prefreeze
  ensure_runtime_assets
  ensure_model_inputs
  ensure_policy_link
  resolve_gpu_mapping
}

refuse_audit_overwrite_after_freeze() {
  local stage="$1"
  local existing_freeze
  existing_freeze="$(find "$RUN_ROOT/metadata" -maxdepth 1 \
    \( -name 'MINIBENCH_TASK_FREEZE_V*.json' -o -name 'MINIBENCH_TASK_FREEZE_V*.sha256' \) \
    -print -quit)"
  if [[ -n "$existing_freeze" ]]; then
    echo "ERROR: refusing to rerun $stage after a formal task freeze: $existing_freeze" >&2
    exit 1
  fi
}

run_simulator_reset() {
  refuse_audit_overwrite_after_freeze simulator_reset
  prepare_simulator_stage
  gpu_guard simulator_reset "$SIM_GPU"
  gpu_isolated_exec "$SIM_GPU" env CUDA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    NVIDIA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    MINIBENCH_SIM_GPU_INDEX="$SIM_GPU" \
    MINIBENCH_POLICY_GPU_INDEX="$POLICY_GPU" \
    MINIBENCH_SIM_GPU_UUID="$SIM_GPU_UUID" \
    MINIBENCH_POLICY_GPU_UUID="$POLICY_GPU_UUID" \
    ROBOTWIN_RENDER_DEVICE="$ROBOTWIN_RENDER_DEVICE" \
    PATH="$(dirname "$PYTHON"):$PATH" \
    "$PYTHON" -u \
    experiments/robotwin/test_move_block_reveal_can.py \
    --mode reset \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --seed-manifest "$SEED_MANIFEST" \
    --output-dir "$RUN_ROOT" \
    2>&1 | tee "$RUN_ROOT/logs/simulator_reset.log"
}

run_simulator_expert() {
  refuse_audit_overwrite_after_freeze simulator_expert
  prepare_simulator_stage
  gpu_guard simulator_expert "$SIM_GPU"
  gpu_isolated_exec "$SIM_GPU" env CUDA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    NVIDIA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ROBOTWIN_RENDER_DEVICE="$ROBOTWIN_RENDER_DEVICE" \
    PATH="$(dirname "$PYTHON"):$PATH" \
    "$PYTHON" -u \
    experiments/robotwin/test_move_block_reveal_can.py \
    --mode expert \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --seed-manifest "$SEED_MANIFEST" \
    --output-dir "$RUN_ROOT" \
    2>&1 | tee "$RUN_ROOT/logs/scripted_expert.log"
}

run_simulator_calibration_expert() {
  refuse_audit_overwrite_after_freeze simulator_calibration_expert
  prepare_simulator_stage
  gpu_guard simulator_calibration_expert "$SIM_GPU"
  gpu_isolated_exec "$SIM_GPU" env CUDA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    NVIDIA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ROBOTWIN_RENDER_DEVICE="$ROBOTWIN_RENDER_DEVICE" \
    PATH="$(dirname "$PYTHON"):$PATH" \
    "$PYTHON" -u \
    experiments/robotwin/test_move_block_reveal_can.py \
    --mode calibration_expert \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --seed-manifest "$SEED_MANIFEST" \
    --output-dir "$RUN_ROOT" \
    2>&1 | tee "$RUN_ROOT/logs/calibration_scripted_expert.log"
}

run_contact_diagnostic() {
  prepare_simulator_stage
  gpu_guard diagnose_contacts "$SIM_GPU"
  gpu_isolated_exec "$SIM_GPU" env CUDA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    NVIDIA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ROBOTWIN_RENDER_DEVICE="$ROBOTWIN_RENDER_DEVICE" \
    PATH="$(dirname "$PYTHON"):$PATH" \
    "$PYTHON" -u \
    experiments/robotwin/diagnose_move_block_contacts.py \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --output "$RUN_ROOT/metadata/contact_diagnostic.json" \
    2>&1 | tee "$RUN_ROOT/logs/contact_diagnostic.log"
}

run_task_freeze() {
  prepare_simulator_stage
  local existing_freeze
  existing_freeze="$(find "$RUN_ROOT/metadata" -maxdepth 1 \
    \( -name 'MINIBENCH_TASK_FREEZE_V*.json' -o -name 'MINIBENCH_TASK_FREEZE_V*.sha256' \) \
    -print -quit)"
  if [[ -n "$existing_freeze" ]]; then
    echo "ERROR: refusing to overwrite an existing task freeze: $existing_freeze" >&2
    exit 1
  fi
  if [[ -d "$RUN_ROOT/fastwam" && -n "$(find "$RUN_ROOT/fastwam" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: refusing to freeze after FastWAM outputs already exist" >&2
    exit 1
  fi
  gpu_guard simulator_freeze "$SIM_GPU"
  gpu_isolated_exec "$SIM_GPU" env CUDA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    NVIDIA_VISIBLE_DEVICES="$SIM_GPU_UUID" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    MINIBENCH_SIM_GPU_INDEX="$SIM_GPU" \
    MINIBENCH_POLICY_GPU_INDEX="$POLICY_GPU" \
    MINIBENCH_SIM_GPU_UUID="$SIM_GPU_UUID" \
    MINIBENCH_POLICY_GPU_UUID="$POLICY_GPU_UUID" \
    ROBOTWIN_RENDER_DEVICE="$ROBOTWIN_RENDER_DEVICE" \
    "$PYTHON" -u experiments/robotwin/test_move_block_reveal_can.py \
    --mode freeze \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --seed-manifest "$SEED_MANIFEST" \
    --output-dir "$RUN_ROOT" \
    2>&1 | tee "$RUN_ROOT/logs/freeze.log"
  capture_diff
  verify_freeze
}

run_simulator() {
  run_simulator_reset
  run_simulator_expert
  run_simulator_calibration_expert
  run_task_freeze
}

run_fastwam() {
  local task="$1"
  local output="$2"
  local episodes="$3"
  local condition="${4:-}"
  local manifest="${5:-}"
  local label="${task}${condition:+_$condition}"
  local condition_tag="${condition:-native}"
  local episode_dir="$output/$task/$condition_tag"
  ensure_python
  ensure_runtime_assets
  ensure_model_inputs
  ensure_policy_link
  resolve_gpu_mapping
  local args=(
    "ckpt=$CHECKPOINT"
    "gpu_id='$GPU_PAIR'"
    "EVALUATION.device=cuda:1"
    "EVALUATION.text_encoder_device=cuda:0"
    "EVALUATION.simulator_cuda_device=cuda:0"
    "EVALUATION.render_device=$ROBOTWIN_RENDER_DEVICE"
    "EVALUATION.robotwin_root=$ROBOTWIN_ROOT"
    "EVALUATION.task_name=$task"
    "EVALUATION.task_config=demo_clean"
    "EVALUATION.eval_num_episodes=$episodes"
    "EVALUATION.dataset_stats_path=$DATASET_STATS"
    "EVALUATION.output_dir=$output"
    "EVALUATION.skip_get_obs_within_replan=false"
  )
  if [[ -n "$condition" ]]; then
    args+=("EVALUATION.minibench_condition=$condition")
  fi
  if [[ -n "$manifest" ]]; then
    args+=("EVALUATION.seed_manifest_path=$manifest")
  fi

  verify_freeze
  if [[ -d "$episode_dir" && -n "$(find "$episode_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: refusing to append/overwrite a non-empty formal result: $episode_dir" >&2
    exit 1
  fi
  gpu_guard "$label" "$SIM_GPU" "$POLICY_GPU"
  gpu_isolated_exec "$SIM_GPU,$POLICY_GPU" env \
    CUDA_VISIBLE_DEVICES="$GPU_PAIR" \
    NVIDIA_VISIBLE_DEVICES="$GPU_PAIR" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    DIFFSYNTH_MODEL_BASE_PATH="$WAN_BASE" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    FASTWAM_TEXT_ENCODER_DEVICE=cuda:0 \
    FASTWAM_SIMULATOR_CUDA_DEVICE=cuda:0 \
    PATH="$(dirname "$PYTHON"):$PATH" \
    "$PYTHON" -u experiments/robotwin/eval_robotwin_single.py "${args[@]}" \
    2>&1 | tee "$RUN_ROOT/logs/fastwam_${label}.log"
}

run_calibration_place() {
  run_fastwam place_can_basket \
    "$RUN_ROOT/fastwam/calibration/place_can_basket" \
    "$CALIBRATION_EPISODES" \
    "" \
    "$PLACE_CAN_CALIBRATION_SEED_MANIFEST"
}

run_calibration_block() {
  run_fastwam move_block_reveal_can_block_calibration \
    "$RUN_ROOT/fastwam/calibration/move_block_reveal_can_block_calibration" \
    "$CALIBRATION_EPISODES" \
    "" \
    "$BLOCK_CALIBRATION_SEED_MANIFEST"
}

run_admission() {
  verify_freeze
  "$PYTHON" -u experiments/robotwin/summarize_move_block_reveal_can.py \
    --run-root "$RUN_ROOT" \
    --mode admission
  if ! admission_is_go; then
    echo "Calibration decision is REJECT; frozen three-condition evaluation is not launched." >&2
  fi
}

run_calibration() {
  run_calibration_place
  run_calibration_block
  run_admission
}

require_admission_go() {
  verify_freeze
  "$PYTHON" -u experiments/robotwin/summarize_move_block_reveal_can.py \
    --run-root "$RUN_ROOT" \
    --mode admission
  if ! admission_is_go; then
    echo "ERROR: pilot requires a strictly validated GO admission decision" >&2
    exit 1
  fi
}

admission_is_go() {
  local admission="$RUN_ROOT/metadata/admission.json"
  [[ -f "$admission" ]] && grep -q '"decision": "GO"' "$admission"
}

run_pilot_condition() {
  local condition="$1"
  require_admission_go
  run_fastwam move_block_reveal_can \
    "$RUN_ROOT/fastwam/pilot/$condition" \
    3 \
    "$condition" \
    "$SEED_MANIFEST"
}

run_pilot() {
  local condition
  for condition in visible missing oracle_reveal; do
    run_pilot_condition "$condition"
  done
}

run_aggregate() {
  ensure_python
  ensure_model_inputs
  verify_freeze
  "$PYTHON" -u experiments/robotwin/summarize_move_block_reveal_can.py \
    --run-root "$RUN_ROOT"
}

case "$STAGE" in
  gpu_preflight)
    run_gpu_preflight
    ;;
  diagnose_contacts)
    run_contact_diagnostic
    ;;
  simulator_reset)
    run_simulator_reset
    ;;
  simulator_expert)
    run_simulator_expert
    ;;
  simulator_calibration_expert)
    run_simulator_calibration_expert
    ;;
  freeze)
    run_task_freeze
    ;;
  simulator)
    run_simulator
    ;;
  calibration_place)
    run_calibration_place
    ;;
  calibration_block)
    run_calibration_block
    ;;
  admission)
    run_admission
    if ! admission_is_go; then
      run_aggregate
    fi
    ;;
  calibration)
    run_calibration
    if ! admission_is_go; then
      run_aggregate
    fi
    ;;
  pilot)
    run_pilot
    ;;
  pilot_visible)
    run_pilot_condition visible
    ;;
  pilot_missing)
    run_pilot_condition missing
    ;;
  pilot_oracle_reveal)
    run_pilot_condition oracle_reveal
    ;;
  aggregate)
    run_aggregate
    ;;
  all)
    run_simulator
    run_calibration
    if admission_is_go; then
      run_pilot
    else
      echo "Admission is REJECT; writing a calibration-only aggregate without pilot data."
    fi
    run_aggregate
    ;;
  *)
    echo "Usage: $0 [gpu_preflight|diagnose_contacts|simulator_reset|simulator_expert|simulator_calibration_expert|freeze|simulator|calibration_place|calibration_block|admission|calibration|pilot|pilot_visible|pilot_missing|pilot_oracle_reveal|aggregate|all]" >&2
    exit 2
    ;;
esac

echo "RUN_ROOT=$RUN_ROOT"
