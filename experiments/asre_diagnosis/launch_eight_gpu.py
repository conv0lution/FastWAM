"""Launch diagnosis conditions in four independent two-GPU slots (no DDP)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import atomic_write_json, build_conditions


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--smoke-summary", type=Path)
    parser.add_argument("--correctness-report", type=Path)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--task-config", default="libero_uncond_2cam224_1e-4")
    parser.add_argument("--dataset-stats-path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--save-rollout-video", action="store_true")
    return parser.parse_args()


def _resolve_num_layers(task_config: str, explicit_num_layers: int | None) -> int:
    if explicit_num_layers is not None:
        return int(explicit_num_layers)
    from hydra import compose, initialize_config_dir

    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="sim_libero.yaml", overrides=[f"task={task_config}"])
    return int(cfg.model.action_dit_config.num_layers)


def _validate_smoke_results(summary_path: Path | None, correctness_path: Path | None) -> None:
    if summary_path is None or correctness_path is None:
        raise ValueError(
            "--mode full requires both --smoke-summary and --correctness-report."
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("mode") != "smoke" or not summary.get("all_succeeded", False):
        raise ValueError(
            f"Smoke summary does not record a successful online smoke run: {summary_path}"
        )
    with correctness_path.open("r", encoding="utf-8") as handle:
        correctness = json.load(handle)
    required_checks = (
        "test_a_empty_diagnosis_matches_original",
        "test_b_intervention_not_identical",
        "test_c_drop_all_valid_action",
    )
    if not all(bool(correctness.get(key, False)) for key in required_checks):
        raise ValueError(
            f"Correctness report does not pass Test A/B/C: {correctness_path}"
        )


def main() -> None:
    args = _parse_args()
    if args.mode == "full":
        _validate_smoke_results(args.smoke_summary, args.correctness_report)

    num_layers = _resolve_num_layers(args.task_config, args.num_layers)
    conditions = build_conditions(num_layers)
    if len(conditions) != 8:
        raise AssertionError(f"Expected 8 conditions, got {len(conditions)}.")
    selected_indices = [0, 7] if args.mode == "smoke" else list(range(8))
    task_ids = [0] if args.mode == "smoke" else list(range(10))
    num_trials = 2 if args.mode == "smoke" else 10

    output_root = args.output_root.resolve()
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    launches: list[dict[str, Any]] = []
    eval_script = Path(__file__).resolve().parents[1] / "libero" / "eval_libero_single.py"

    # Wan 2.2 and the T5 text encoder do not fit reliably beside EGL on one
    # device. Each slot therefore owns an even-numbered model GPU and the next
    # GPU for T5 + EGL. Full mode runs two waves of four conditions.
    wave_size = 4
    waves = [
        selected_indices[offset : offset + wave_size]
        for offset in range(0, len(selected_indices), wave_size)
    ]
    for wave_index, wave_condition_indices in enumerate(waves):
        processes: list[tuple[subprocess.Popen, Any, dict[str, Any]]] = []
        for slot_index, condition_index in enumerate(wave_condition_indices):
            condition = conditions[condition_index]
            model_gpu_id = slot_index * 2
            auxiliary_gpu_id = model_gpu_id + 1
            visible_devices = f"{model_gpu_id},{auxiliary_gpu_id}"
            condition_output = output_root / condition.name
            condition_output.mkdir(parents=True, exist_ok=True)
            log_path = logs_dir / f"wave{wave_index}_gpu{model_gpu_id}_{condition.name}.log"
            status_path = condition_output / "launcher_status.json"
            command = [
                args.python,
                str(eval_script),
                f"task={args.task_config}",
                f"ckpt={args.checkpoint}",
                "gpu_id=0",
                "EVALUATION.device=cuda:0",
                "EVALUATION.text_encoder_device=cuda:1",
                "EVALUATION.task_suite_name=libero_spatial",
                f"EVALUATION.task_ids={json.dumps(task_ids, separators=(',', ':'))}",
                f"EVALUATION.num_trials={num_trials}",
                f"EVALUATION.output_dir={condition_output}",
                "EVALUATION.visualize_future_video=false",
                "ASRE_DIAGNOSIS.enabled=true",
                "ASRE_DIAGNOSIS.mode=drop_video_kv",
                f"ASRE_DIAGNOSIS.condition_index={condition_index}",
                f"ASRE_DIAGNOSIS.condition_name={condition.name}",
                "ASRE_DIAGNOSIS.disabled_video_layers=[]",
                f"ASRE_DIAGNOSIS.save_rollout_video={str(args.save_rollout_video).lower()}",
            ]
            if args.dataset_stats_path:
                command.append(f"EVALUATION.dataset_stats_path={args.dataset_stats_path}")
            if args.seed is not None:
                command.append(f"seed={args.seed}")

            environment = os.environ.copy()
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["CUDA_VISIBLE_DEVICES"] = visible_devices
            environment["MUJOCO_GL"] = "egl"
            environment["PYOPENGL_PLATFORM"] = "egl"
            # robosuite interprets this as a physical EGL device index.
            environment["MUJOCO_EGL_DEVICE_ID"] = str(auxiliary_gpu_id)
            launch_record = {
                "condition": condition.name,
                "condition_index": condition_index,
                "disabled_video_layers": list(condition.disabled_video_layers),
                "wave": wave_index,
                "slot": slot_index,
                "physical_gpu": model_gpu_id,
                "text_encoder_gpu": auxiliary_gpu_id,
                "render_gpu": auxiliary_gpu_id,
                "cuda_visible_devices": visible_devices,
                "model_device": "cuda:0",
                "text_encoder_device": "cuda:1",
                "mujoco_egl_device_id": str(auxiliary_gpu_id),
                "output_dir": str(condition_output),
                "log_path": str(log_path),
                "command": shlex.join(command),
                "start_timestamp": _timestamp(),
                "end_timestamp": None,
                "exit_status": None,
            }
            atomic_write_json(status_path, launch_record)
            log_handle = log_path.open("a", encoding="utf-8")
            log_handle.write(
                f"[{_timestamp()}] wave={wave_index} "
                f"CUDA_VISIBLE_DEVICES={visible_devices} "
                f"model=cuda:0 text_encoder=cuda:1 "
                f"MUJOCO_EGL_DEVICE_ID={auxiliary_gpu_id}\n"
            )
            log_handle.write(shlex.join(command) + "\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            launch_record["pid"] = process.pid
            atomic_write_json(status_path, launch_record)
            launches.append(launch_record)
            processes.append((process, log_handle, launch_record))

        wave_succeeded = True
        for process, log_handle, launch_record in processes:
            exit_status = process.wait()
            log_handle.close()
            launch_record["exit_status"] = int(exit_status)
            launch_record["end_timestamp"] = _timestamp()
            status_path = Path(launch_record["output_dir"]) / "launcher_status.json"
            atomic_write_json(status_path, launch_record)
            wave_succeeded = wave_succeeded and exit_status == 0
            print(
                f"{launch_record['condition']}: exit={exit_status} "
                f"log={launch_record['log_path']}"
            )
        if not wave_succeeded:
            print(f"Wave {wave_index} failed; later waves will not be launched.")
            break

    summary = {
        "mode": args.mode,
        "all_succeeded": all(record["exit_status"] == 0 for record in launches),
        "launches": launches,
        "end_timestamp": _timestamp(),
    }
    summary_path = output_root / "launcher_summary.json"
    atomic_write_json(summary_path, summary)
    print(f"Launcher summary: {summary_path}")
    if not summary["all_succeeded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
