"""Launch one four-GPU Round-4B held-out offline wave."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4B_PROTOCOL,
    atomic_write_json,
    build_round4b_conditions,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.g0.launch_four_gpu import (  # noqa: E402
    _child_environment,
    _launcher_lock,
)
from experiments.asre_diagnosis.round4a.launch_wave import _gpu_inventory  # noqa: E402


WAVES = {1: (0, 1, 2, 3), 2: (4, 5, 6, 7)}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _complete(root: Path, condition: str, expected: int, commit: str) -> bool:
    metadata = root / condition / "run_metadata.json"
    if not metadata.is_file():
        return False
    value = _read(metadata)
    return (
        value.get("status") == "complete"
        and value.get("protocol") == ROUND4B_PROTOCOL
        and value.get("diagnosis_condition") == condition
        and value.get("completed_samples") == expected
        and value.get("git_commit_hash") == commit
    )


def launch(args: argparse.Namespace) -> Path:
    conditions = build_round4b_conditions(30)
    indices = WAVES[args.wave]
    gpu_ids = tuple(args.gpu_ids)
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("Offline wave requires four distinct GPUs.")
    inventory = _gpu_inventory()
    available = {int(item["index"]) for item in inventory}
    if set(gpu_ids) - available:
        raise ValueError("Requested GPU is unavailable.")
    split = _read(args.split.resolve())
    expected = int(split["holdout_sample_count"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root.parent / "logs" / f"offline_wave{args.wave}"
    logs.mkdir(parents=True, exist_ok=True)
    commit = git_commit(PROJECT_ROOT)
    children = []
    failure = None
    start = now_iso()
    with _launcher_lock(output_root.parent / f"offline_wave{args.wave}") as lock_fd:
        for slot, index in enumerate(indices):
            condition = conditions[index]
            if _complete(output_root, condition.name, expected, commit):
                continue
            attempt = 1
            log = logs / f"{condition.name}.attempt{attempt:02d}.log"
            while log.exists():
                attempt += 1
                log = logs / f"{condition.name}.attempt{attempt:02d}.log"
            command = [
                str(args.python.resolve()),
                "-m",
                "experiments.asre_diagnosis.round4b.replay_state_bank",
                "--condition-index",
                str(index),
                "--condition-name",
                condition.name,
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--dataset-stats",
                str(args.dataset_stats.resolve()),
                "--split",
                str(args.split.resolve()),
                "--basis-manifest",
                str(args.basis_manifest.resolve()),
                "--donor-mapping",
                str(args.donor_mapping.resolve()),
                "--donor-manifest",
                str(args.donor_manifest.resolve()),
                "--donor-root",
                str(args.donor_root.resolve()),
                "--output-root",
                str(output_root),
            ]
            handle = log.open("x", encoding="utf-8")
            handle.write(shlex.join(command) + "\n")
            handle.flush()
            environment = _child_environment(gpu_ids[slot])
            environment["ASRE_ROUND4B_PHYSICAL_GPU"] = str(gpu_ids[slot])
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock_fd,),
            )
            children.append((condition.name, process, handle, log))
            if args.launch_stagger_seconds and slot < 3:
                time.sleep(args.launch_stagger_seconds)
        pending = list(children)
        while pending:
            for child in list(pending):
                name, process, handle, log = child
                code = process.poll()
                if code is None:
                    continue
                pending.remove(child)
                handle.close()
                if code:
                    failure = f"Offline {name} exited {code}; inspect {log}."
                    for _, other, other_handle, _ in pending:
                        if other.poll() is None:
                            os.killpg(other.pid, signal.SIGTERM)
                        other_handle.close()
                    pending.clear()
                    break
            if pending:
                time.sleep(2)
    states = {
        conditions[index].name: _complete(
            output_root, conditions[index].name, expected, commit
        )
        for index in indices
    }
    summary = output_root / f"offline_wave{args.wave}_launcher_summary.json"
    atomic_write_json(
        summary,
        {
            "artifact_type": "asre_round4b_offline_wave_summary",
            "schema_version": 1,
            "protocol": ROUND4B_PROTOCOL,
            "wave": args.wave,
            "condition_indices": list(indices),
            "gpu_ids": list(gpu_ids),
            "gpu_inventory": [
                item for item in inventory if int(item["index"]) in gpu_ids
            ],
            "conditions": [conditions[index].name for index in indices],
            "all_succeeded": all(states.values()) and failure is None,
            "condition_complete": states,
            "failure": failure,
            "git_commit_hash": commit,
            "split_sha256": sha256_file(args.split.resolve()),
            "basis_manifest_sha256": sha256_file(args.basis_manifest.resolve()),
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "no_ddp": True,
        },
    )
    if failure or not all(states.values()):
        raise RuntimeError(failure or "Round-4B offline wave incomplete.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, choices=(1, 2), required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--launch-stagger-seconds", type=float, default=30.0)
    path = launch(parser.parse_args())
    print(f"Round-4B offline wave complete: {path}")


if __name__ == "__main__":
    main()
