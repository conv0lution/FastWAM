"""Launch four isolated, paired, non-DDP Salvage-B world-metric shards."""

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
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_B_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.g0.launch_four_gpu import (  # noqa: E402
    _child_environment,
    _launcher_lock,
)
from experiments.asre_diagnosis.round4a.launch_wave import (  # noqa: E402
    _gpu_inventory,
    stable_gpu_inventory,
)
from experiments.asre_diagnosis.salvage_b.definitions import (  # noqa: E402
    WORLD_DRAWS_PER_SAMPLE,
)
from experiments.asre_diagnosis.salvage_b.world_manifest import (  # noqa: E402
    NATIVE_WORLD_METRIC,
    VIDEO_INFERENCE_SHIFT,
    VIDEO_INFERENCE_STEPS,
)
from experiments.asre_diagnosis.salvage_b.world_worker import (  # noqa: E402
    PHASES,
    ROWS_PER_WORKER,
    SAMPLES_PER_WORKER,
    WORKERS,
    load_frozen_contract,
    phase_conditions,
    shard_paths,
    validate_endpoint_gate_payload,
    validate_completed_shard,
    worker_records,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _resolve_python(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise FileNotFoundError(f"Salvage-B Python executable is unavailable: {resolved}")
    return resolved


def _validate_gpu_ids(values: Sequence[int]) -> tuple[int, int, int, int]:
    gpu_ids = tuple(int(value) for value in values)
    if len(gpu_ids) != WORKERS or len(set(gpu_ids)) != WORKERS or any(
        value < 0 for value in gpu_ids
    ):
        raise ValueError("Salvage-B requires exactly four distinct nonnegative GPU IDs.")
    return gpu_ids  # type: ignore[return-value]


def _artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    names = {
        "preflight": args.preflight,
        "world": args.world_manifest,
        "stochastic": args.stochastic_manifest,
        "draws": args.draw_tensors,
        "targets": args.target_manifest,
        "machinery": args.machinery,
    }
    result = {key: Path(value).expanduser().resolve() for key, value in names.items()}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen Salvage-B launcher inputs: {missing}")
    return result


def _validate_endpoint_gate(
    *,
    phase: str,
    endpoint_gate: Path | None,
    hashes: Mapping[str, str],
    commit: str,
) -> tuple[Path | None, str | None]:
    if phase == "endpoint":
        if endpoint_gate is not None:
            raise ValueError("Endpoint phase must not receive --endpoint-gate.")
        return None, None
    if endpoint_gate is None:
        raise ValueError("Projected phase requires the passed --endpoint-gate artifact.")
    path = endpoint_gate.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Projected endpoint gate is unavailable: {path}")
    payload = _read(path)
    validate_endpoint_gate_payload(
        payload,
        world_manifest_sha256=hashes["world"],
        stochastic_manifest_sha256=hashes["stochastic"],
        target_manifest_sha256=hashes["targets"],
        machinery_sha256=hashes["machinery"],
        git_commit_hash=commit,
    )
    return path, sha256_file(path)


def build_launcher_identity(
    *,
    args: argparse.Namespace,
    gpu_ids: Sequence[int],
    selected_inventory: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = _artifact_paths(args)
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    commit = git_commit(PROJECT_ROOT)
    gate_path, gate_sha = _validate_endpoint_gate(
        phase=args.phase,
        endpoint_gate=args.endpoint_gate,
        hashes=hashes,
        commit=commit,
    )
    if gate_path is not None:
        paths["endpoint_gate"] = gate_path
    world = _read(paths["world"])
    records = world.get("records")
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("Frozen world manifest lacks exactly 100 records.")
    partitions = [worker_records(records, index) for index in range(WORKERS)]
    identity = {
        "artifact_type": "asre_salvage_b_world_launcher_config",
        "schema_version": 2,
        "protocol": SALVAGE_B_PROTOCOL,
        "phase": args.phase,
        "git_commit_hash": commit,
        "gpu_ids": list(gpu_ids),
        "gpu_inventory": stable_gpu_inventory(selected_inventory),
        "python_path": str(_resolve_python(args.python)),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "runtime_work_dir": str(args.runtime_work_dir.expanduser().resolve()),
        "preflight_report_path": str(paths["preflight"]),
        "preflight_report_sha256": hashes["preflight"],
        "world_manifest_path": str(paths["world"]),
        "world_manifest_sha256": hashes["world"],
        "stochastic_manifest_path": str(paths["stochastic"]),
        "stochastic_manifest_sha256": hashes["stochastic"],
        "draw_tensors_path": str(paths["draws"]),
        "draw_tensors_sha256": hashes["draws"],
        "target_manifest_path": str(paths["targets"]),
        "target_manifest_sha256": hashes["targets"],
        "machinery_path": str(paths["machinery"]),
        "machinery_sha256": hashes["machinery"],
        "endpoint_gate_path": None if gate_path is None else str(gate_path),
        "endpoint_gate_sha256": gate_sha,
        "conditions": list(phase_conditions(args.phase)),
        "draws_per_sample": WORLD_DRAWS_PER_SAMPLE,
        "worker_count": WORKERS,
        "samples_per_worker": SAMPLES_PER_WORKER,
        "rows_per_worker": ROWS_PER_WORKER,
        "worker_sample_ids_sha256": [
            sha256_json([str(record["sample_id"]) for record in partition])
            for partition in partitions
        ],
        "partition_rule": "world_manifest_records[worker_index::4]",
        "native_metric": NATIVE_WORLD_METRIC,
        "metric_direction": "lower_is_better",
        "inference_steps": VIDEO_INFERENCE_STEPS,
        "inference_shift": VIDEO_INFERENCE_SHIFT,
        "initial_state": "pure_gaussian_future_latent_noise",
        "target_usage": "scoring_only_after_inference",
        "online_episodes": 0,
        "environment_rollouts": 0,
        "action_rerun": False,
        "no_ddp": True,
        "max_concurrent_gpus": 4,
    }
    identity["identity_sha256"] = sha256_json(identity)
    return identity, paths


def worker_command(
    *,
    args: argparse.Namespace,
    worker_index: int,
    launcher_config: Path,
    paths: Mapping[str, Path],
    runtime_namespace: str,
) -> list[str]:
    command = [
        str(_resolve_python(args.python)),
        "-m",
        "experiments.asre_diagnosis.salvage_b.world_worker",
        "--phase",
        args.phase,
        "--worker-index",
        str(worker_index),
        "--preflight",
        str(paths["preflight"]),
        "--world-manifest",
        str(paths["world"]),
        "--stochastic-manifest",
        str(paths["stochastic"]),
        "--draw-tensors",
        str(paths["draws"]),
        "--target-manifest",
        str(paths["targets"]),
        "--machinery",
        str(paths["machinery"]),
        "--launcher-config",
        str(launcher_config),
        "--output-dir",
        str(args.output_dir.expanduser().resolve()),
        "--runtime-work-dir",
        str(
            args.runtime_work_dir.expanduser().resolve()
            / f"config_{runtime_namespace}"
        ),
    ]
    if args.phase == "projected":
        command.extend(["--endpoint-gate", str(paths["endpoint_gate"])])
    return command


def _next_log(log_dir: Path, phase: str, worker_index: int) -> Path:
    attempt = 1
    while True:
        path = log_dir / f"{phase}.worker_{worker_index:02d}.attempt_{attempt:02d}.log"
        if not path.exists():
            return path
        attempt += 1


def _terminate_children(
    children: Sequence[tuple[int, subprocess.Popen[Any], Any, Path]],
    *,
    grace_seconds: float = 30.0,
) -> None:
    """Terminate all live worker process groups and close every log handle."""

    for _worker_index, process, _handle, _log_path in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except OSError:
                pass
    deadline = time.monotonic() + max(0.0, grace_seconds)
    for _worker_index, process, _handle, _log_path in children:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
    for _worker_index, process, handle, _log_path in children:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if not handle.closed:
            handle.close()


def launch(args: argparse.Namespace) -> Path:
    if args.phase not in PHASES:
        raise ValueError(f"Unsupported world phase: {args.phase}.")
    gpu_ids = _validate_gpu_ids(args.gpu_ids)
    inventory = _gpu_inventory()
    by_index = {int(record["index"]): record for record in inventory}
    missing = sorted(set(gpu_ids) - set(by_index))
    if missing:
        raise ValueError(f"Requested Salvage-B GPUs are unavailable: {missing}.")
    selected_inventory = [by_index[index] for index in gpu_ids]
    output_dir = args.output_dir.expanduser().resolve()
    phase_root = output_dir / args.phase
    log_dir = args.log_dir.expanduser().resolve() / args.phase
    identity, paths = build_launcher_identity(
        args=args,
        gpu_ids=gpu_ids,
        selected_inventory=selected_inventory,
    )
    phase_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = phase_root / "launcher_config.json"
    if config_path.exists():
        if _read(config_path) != identity:
            raise RuntimeError(f"Refusing incompatible Salvage-B resume in {phase_root}.")
    else:
        # A non-empty phase directory without its immutable identity is unsafe.
        unexpected = [path for path in phase_root.iterdir() if path.name != config_path.name]
        if unexpected:
            raise RuntimeError(
                f"Refusing unregistered Salvage-B phase directory: {phase_root}."
            )
        atomic_write_json(config_path, identity)

    # Re-run the worker's full manifest/provenance contract before any process
    # is created.  Worker processes repeat it independently after launch.
    contract_args = argparse.Namespace(
        phase=args.phase,
        preflight=paths["preflight"],
        world_manifest=paths["world"],
        stochastic_manifest=paths["stochastic"],
        draw_tensors=paths["draws"],
        target_manifest=paths["targets"],
        machinery=paths["machinery"],
        launcher_config=config_path,
        endpoint_gate=paths.get("endpoint_gate"),
    )
    contract = load_frozen_contract(contract_args, load_draw_tensors=False)
    records_by_worker = [
        worker_records(contract["records"], worker_index)
        for worker_index in range(WORKERS)
    ]
    provenance = {
        "world_manifest_sha256": contract["shas"]["world"],
        "stochastic_manifest_sha256": contract["shas"]["stochastic"],
        "target_manifest_sha256": contract["shas"]["targets"],
        "machinery_sha256": contract["shas"]["machinery"],
        "git_commit": contract["commit"],
    }
    conditions = contract["conditions"]
    config_sha = sha256_file(config_path)
    children: list[tuple[int, subprocess.Popen[Any], Any, Path]] = []
    start = now_iso()
    failure: str | None = None
    with _launcher_lock(phase_root) as lock_fd:
        try:
            for worker_index, physical_gpu in enumerate(gpu_ids):
                sample_ids = [
                    str(record["sample_id"]) for record in records_by_worker[worker_index]
                ]
                if validate_completed_shard(
                    output_dir=output_dir,
                    phase=args.phase,
                    worker_index=worker_index,
                    expected_sample_ids=sample_ids,
                    conditions=conditions,
                    provenance=provenance,
                    launcher_config_sha256=config_sha,
                ):
                    continue
                command = worker_command(
                    args=args,
                    worker_index=worker_index,
                    launcher_config=config_path,
                    paths=paths,
                    runtime_namespace=config_sha[:16],
                )
                log_path = _next_log(log_dir, args.phase, worker_index)
                handle = log_path.open("x", encoding="utf-8")
                try:
                    handle.write(shlex.join(command) + "\n")
                    handle.flush()
                    environment = _child_environment(physical_gpu)
                    environment["ASRE_SALVAGE_B_PHYSICAL_GPU"] = str(physical_gpu)
                    environment["PYTHONHASHSEED"] = "0"
                    environment["TOKENIZERS_PARALLELISM"] = "false"
                    inductor_cache = Path(
                        f"/tmp/fastwam-salvage-b-{args.phase}-"
                        f"{config_sha[:16]}-gpu{physical_gpu}-"
                        f"worker{worker_index}-torchinductor"
                    )
                    inductor_cache.mkdir(parents=True, exist_ok=True)
                    environment["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(lock_fd,),
                    )
                except BaseException:
                    handle.close()
                    raise
                children.append((worker_index, process, handle, log_path))
                if args.launch_stagger_seconds > 0 and worker_index < WORKERS - 1:
                    time.sleep(args.launch_stagger_seconds)

            pending = list(children)
            while pending:
                for child in list(pending):
                    worker_index, process, handle, log_path = child
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    pending.remove(child)
                    handle.close()
                    if returncode != 0:
                        failure = (
                            f"Salvage-B {args.phase} worker {worker_index} exited "
                            f"{returncode}; inspect {log_path}."
                        )
                        _terminate_children(pending)
                        pending.clear()
                        break
                if pending:
                    time.sleep(2)
        except BaseException:
            _terminate_children(children)
            raise
        finally:
            for _worker_index, _process, handle, _log_path in children:
                if not handle.closed:
                    handle.close()

    completed = True
    shards = []
    for worker_index in range(WORKERS):
        sample_ids = [
            str(record["sample_id"]) for record in records_by_worker[worker_index]
        ]
        valid = validate_completed_shard(
            output_dir=output_dir,
            phase=args.phase,
            worker_index=worker_index,
            expected_sample_ids=sample_ids,
            conditions=conditions,
            provenance=provenance,
            launcher_config_sha256=config_sha,
        )
        completed = completed and valid
        if valid:
            metadata_path, rows_path = shard_paths(output_dir, args.phase, worker_index)
            shards.append(
                {
                    "worker_index": worker_index,
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": sha256_file(metadata_path),
                    "rows_path": str(rows_path),
                    "rows_sha256": sha256_file(rows_path),
                }
            )
    summary_path = phase_root / "launcher_summary.json"
    atomic_write_json(
        summary_path,
        {
            "artifact_type": "asre_salvage_b_world_launcher_summary",
            "schema_version": 2,
            "protocol": SALVAGE_B_PROTOCOL,
            "status": "completed" if completed and failure is None else "failed",
            "phase": args.phase,
            "all_succeeded": completed and failure is None,
            "failure": failure,
            "conditions": list(conditions),
            "worker_count": WORKERS,
            "samples_per_worker": SAMPLES_PER_WORKER,
            "rows_per_worker": ROWS_PER_WORKER,
            "launcher_config_path": str(config_path),
            "launcher_config_sha256": config_sha,
            "shards": shards,
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "online_episodes": 0,
            "environment_rollouts": 0,
            "action_rerun": False,
            "heldout_svd_refit": False,
            "no_ddp": True,
        },
    )
    if not completed or failure is not None:
        raise RuntimeError(failure or f"Salvage-B {args.phase} shards are incomplete.")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--world-manifest", type=Path, required=True)
    parser.add_argument("--stochastic-manifest", type=Path, required=True)
    parser.add_argument("--draw-tensors", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--machinery", type=Path, required=True)
    parser.add_argument("--endpoint-gate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--runtime-work-dir", type=Path, required=True)
    parser.add_argument("--launch-stagger-seconds", type=float, default=70.0)
    args = parser.parse_args()
    if args.launch_stagger_seconds < 0:
        parser.error("--launch-stagger-seconds must be nonnegative")
    path = launch(args)
    print(f"Salvage-B {args.phase} world phase complete: {path}", flush=True)


if __name__ == "__main__":
    main()
