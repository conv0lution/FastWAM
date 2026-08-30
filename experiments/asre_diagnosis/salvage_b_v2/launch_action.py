"""Launch isolated-GPU native-joint LIBERO action conditions."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.asre_diagnosis.common import (
    build_salvage_b_v2_conditions,
    git_commit,
    sha256_file,
)

from .definitions import PROTOCOL


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_CONFIG = "libero_uncond_2cam224_1e-4"
TASK_SUITE = "libero_spatial"


def _child_environment(physical_gpu: int) -> dict[str, str]:
    """Bind CUDA and MuJoCo EGL to the same physical GPU.

    robosuite validates ``MUJOCO_EGL_DEVICE_ID`` against the physical IDs in
    ``CUDA_VISIBLE_DEVICES`` at import time.  Always overwrite a possibly
    stale value inherited from the parent shell.
    """

    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(physical_gpu),
        }
    )
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        environment.pop(key, None)
    return environment


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _complete(root: Path, condition: str, *, tasks: tuple[int, ...], trials: int) -> bool:
    metadata = root / condition / "run_metadata.json"
    if not metadata.is_file():
        return False
    value = _read(metadata)
    if not (
        value.get("status") == "completed"
        and value.get("condition_protocol") == PROTOCOL
        and value.get("diagnosis_condition") == condition
        and value.get("git_commit_hash") == git_commit(PROJECT_ROOT)
        and value.get("task_ids") == list(tasks)
        and value.get("number_of_trials") == trials
        and value.get("number_of_inference_steps") == 10
        and value.get("replan_steps") == 10
    ):
        return False
    files = list((root / condition / TASK_SUITE).glob("gpu*_task*_results.json"))
    if len(files) != len(tasks):
        return False
    observed_tasks = set()
    for path in files:
        result = _read(path)
        task = int(result.get("task_id", -1))
        successes = set(map(int, result.get("success_episodes", [])))
        failures = set(map(int, result.get("failure_episodes", [])))
        if (
            task not in tasks
            or task in observed_tasks
            or successes & failures
            or successes | failures != set(range(trials))
        ):
            return False
        observed_tasks.add(task)
    trace_dir = root / condition / TASK_SUITE / "action_traces"
    traces = list(trace_dir.glob("task*_trial*.jsonl"))
    if observed_tasks != set(tasks) or len(traces) != len(tasks) * trials:
        return False
    for path in traces:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records or any(
            record.get("native_causal_input_audit", {}).get("passed") is not True
            for record in records
        ):
            return False
    return True


def launch(args: argparse.Namespace) -> None:
    preflight = _read(args.preflight.resolve())
    if preflight.get("protocol") != PROTOCOL or preflight.get("status") != "compatible":
        raise ValueError("Native v2 preflight is incomplete.")
    conditions = build_salvage_b_v2_conditions(30)
    by_name = {condition.name: (index, condition) for index, condition in enumerate(conditions)}
    names = tuple(args.conditions)
    if any(name not in by_name for name in names):
        raise ValueError(f"Unknown v2 action condition(s): {names}")
    if len(args.gpu_ids) < len(names) or len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("Action launch requires one distinct physical GPU per condition.")
    tasks = (0,) if args.mode == "smoke" else tuple(range(10))
    trials = 2 if args.mode == "smoke" else 10
    state = preflight["state"]
    donors = preflight["donors"]
    basis = preflight["basis"]
    output = args.output_root.resolve() / f"action_{args.mode}"
    logs = args.output_root.resolve() / "logs"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    children = []
    for slot, name in enumerate(names):
        index, condition = by_name[name]
        if _complete(output, name, tasks=tasks, trials=trials):
            continue
        condition_root = output / name
        condition_root.mkdir(parents=True, exist_ok=True)
        scalar = lambda value: "null" if value is None else str(value)
        compact = lambda value: json.dumps(list(value), separators=(",", ":"))
        command = [
            str(args.python.resolve()),
            str(PROJECT_ROOT / "experiments/libero/eval_libero_single.py"),
            f"task={TASK_CONFIG}",
            f"ckpt={state['checkpoint_path']}",
            "model.load_text_encoder=false",
            "gpu_id=0",
            "seed=42",
            "EVALUATION.device=cuda:0",
            "EVALUATION.text_encoder_device=null",
            f"EVALUATION.prompt_context_cache_path={state['prompt_context_cache_path']}",
            f"EVALUATION.task_suite_name={TASK_SUITE}",
            f"EVALUATION.task_ids={compact(tasks)}",
            f"EVALUATION.num_trials={trials}",
            "EVALUATION.action_horizon=32",
            "EVALUATION.num_inference_steps=10",
            "EVALUATION.replan_steps=10",
            f"EVALUATION.output_dir={condition_root}",
            "EVALUATION.visualize_future_video=false",
            f"EVALUATION.dataset_stats_path={state['dataset_stats_path']}",
            "ASRE_DIAGNOSIS.enabled=true",
            "ASRE_DIAGNOSIS.mode=replace_video_kv",
            f"ASRE_DIAGNOSIS.protocol={PROTOCOL}",
            f"ASRE_DIAGNOSIS.condition_index={index}",
            f"ASRE_DIAGNOSIS.condition_name={name}",
            f"ASRE_DIAGNOSIS.enabled_video_retrieval_layers={compact(range(30))}",
            "ASRE_DIAGNOSIS.disabled_video_layers=[]",
            f"ASRE_DIAGNOSIS.replacement_video_layers={compact(condition.replacement_video_layers)}",
            f"ASRE_DIAGNOSIS.subspace_basis_kind={scalar(condition.basis_kind)}",
            f"ASRE_DIAGNOSIS.subspace_rank={scalar(condition.subspace_rank)}",
            f"ASRE_DIAGNOSIS.subspace_basis_manifest_path={basis['path']}",
            f"ASRE_DIAGNOSIS.subspace_basis_manifest_sha256={basis['sha256']}",
            f"ASRE_DIAGNOSIS.checkpoint_sha256={state['checkpoint_sha256']}",
            f"ASRE_DIAGNOSIS.dataset_stats_sha256={state['dataset_stats_sha256']}",
            f"ASRE_DIAGNOSIS.state_bank_manifest_path={state['source_manifest_path']}",
            f"ASRE_DIAGNOSIS.state_bank_manifest_sha256={state['source_manifest_sha256']}",
            f"ASRE_DIAGNOSIS.valid_state_bank_manifest_path={state['valid_manifest_path']}",
            f"ASRE_DIAGNOSIS.valid_state_bank_manifest_sha256={state['valid_manifest_sha256']}",
            f"ASRE_DIAGNOSIS.prompt_context_cache_sha256={state['prompt_context_cache_sha256']}",
            f"ASRE_DIAGNOSIS.donor_mapping_path={donors['mapping_path']}",
            f"ASRE_DIAGNOSIS.donor_mapping_sha256={donors['mapping_sha256']}",
            f"ASRE_DIAGNOSIS.donor_observation_manifest_path={donors['manifest_path']}",
            f"ASRE_DIAGNOSIS.donor_observation_manifest_sha256={donors['manifest_sha256']}",
            f"ASRE_DIAGNOSIS.donor_observation_root={donors['root']}",
            "ASRE_DIAGNOSIS.save_rollout_video=false",
            "ASRE_DIAGNOSIS.save_action_trace=true",
        ]
        log = logs / f"action_{args.mode}.{name}.log"
        handle = log.open("w", encoding="utf-8")
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        env = _child_environment(args.gpu_ids[slot])
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append((name, process, handle, log))
        if args.launch_stagger_seconds and slot + 1 < len(names):
            time.sleep(args.launch_stagger_seconds)
    failure = None
    while children:
        for child in list(children):
            name, process, handle, log = child
            code = process.poll()
            if code is None:
                continue
            children.remove(child)
            handle.close()
            if code:
                failure = f"Native action condition {name} exited {code}; inspect {log}."
                for _, other, other_handle, _ in children:
                    if other.poll() is None:
                        os.killpg(other.pid, signal.SIGTERM)
                    other_handle.close()
                children.clear()
                break
        if children:
            time.sleep(2)
    if failure:
        raise RuntimeError(failure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.0)
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
