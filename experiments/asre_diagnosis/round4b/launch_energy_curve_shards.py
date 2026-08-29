"""Launch four isolated Round-4B cumulative-energy workers with safe resume."""

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
from experiments.asre_diagnosis.round4b.basis import LATE_LAYERS  # noqa: E402
from experiments.asre_diagnosis.round4b.energy_curve_definitions import (  # noqa: E402
    COLLECT_STAGE,
    RECOVER_STAGE,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _report_name(stage: str, index: int) -> str:
    return f"{stage}_shard{index}.json"


def _valid_report(
    path: Path,
    *,
    stage: str,
    index: int,
    commit: str,
    basis_sha: str,
    split_sha: str | None,
    diagnostics_sha: str | None,
) -> bool:
    if not path.is_file():
        return False
    report = _read(path)
    common = (
        report.get("status") == "complete",
        report.get("protocol") == ROUND4B_PROTOCOL,
        report.get("stage") == stage,
        report.get("worker_index") == index,
        report.get("layers") == list(LATE_LAYERS[index::4]),
        report.get("git_commit_hash") == commit,
        report.get("basis_manifest_sha256") == basis_sha,
    )
    if not all(common):
        return False
    if stage == COLLECT_STAGE:
        return report.get("split_sha256") == split_sha
    return report.get("diagnostics_sha256") == diagnostics_sha


def launch(args: argparse.Namespace) -> Path:
    gpu_ids = tuple(args.gpu_ids)
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or any(gpu < 0 for gpu in gpu_ids):
        raise ValueError("Cumulative-energy collection requires four distinct GPU IDs.")
    inventory = _gpu_inventory()
    available = {int(record["index"]) for record in inventory}
    if set(gpu_ids) - available:
        raise ValueError(f"Unavailable GPUs: {sorted(set(gpu_ids) - available)}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit(PROJECT_ROOT)
    basis_sha = sha256_file(args.basis_manifest.resolve())
    split_sha = None if args.split is None else sha256_file(args.split.resolve())
    diagnostics_sha = (
        None if args.diagnostics is None else sha256_file(args.diagnostics.resolve())
    )
    identity = {
        "artifact_type": "asre_round4b_energy_curve_launcher_config",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "stage": args.stage,
        "git_commit_hash": commit,
        "gpu_ids": list(gpu_ids),
        "gpu_inventory": stable_gpu_inventory(
            [record for record in inventory if int(record["index"]) in gpu_ids]
        ),
        "basis_manifest_path": str(args.basis_manifest.resolve()),
        "basis_manifest_sha256": basis_sha,
        "split_sha256": split_sha,
        "diagnostics_sha256": diagnostics_sha,
        "checkpoint_sha256": (
            None if args.checkpoint is None else sha256_file(args.checkpoint.resolve())
        ),
        "donor_mapping_sha256": (
            None if args.donor_mapping is None else sha256_file(args.donor_mapping.resolve())
        ),
        "donor_manifest_sha256": (
            None if args.donor_manifest is None else sha256_file(args.donor_manifest.resolve())
        ),
        "heldout_gram_dir": (
            None if args.heldout_gram_dir is None else str(args.heldout_gram_dir.resolve())
        ),
        "fit_dir": None if args.fit_dir is None else str(args.fit_dir.resolve()),
        "no_ddp": True,
    }
    identity["identity_sha256"] = sha256_json(identity)
    config_path = output_dir / f"{args.stage}_launcher_config.json"
    if config_path.exists() and _read(config_path) != identity:
        raise RuntimeError(f"Refusing incompatible resume in {output_dir}.")
    if not config_path.exists():
        atomic_write_json(config_path, identity)

    children: list[tuple[int, subprocess.Popen[Any], Any, Path]] = []
    start = now_iso()
    failure: str | None = None
    with _launcher_lock(output_dir / args.stage) as lock_fd:
        try:
            for index, physical_gpu in enumerate(gpu_ids):
                report_path = output_dir / _report_name(args.stage, index)
                if _valid_report(
                    report_path,
                    stage=args.stage,
                    index=index,
                    commit=commit,
                    basis_sha=basis_sha,
                    split_sha=split_sha,
                    diagnostics_sha=diagnostics_sha,
                ):
                    continue
                attempt = 1
                log_path = log_dir / f"{args.stage}.shard{index}.attempt{attempt:02d}.log"
                while log_path.exists():
                    attempt += 1
                    log_path = log_dir / f"{args.stage}.shard{index}.attempt{attempt:02d}.log"
                command = [
                    str(args.python.resolve()),
                    "-m",
                    "experiments.asre_diagnosis.round4b.energy_curve_worker",
                    "--stage",
                    args.stage,
                    "--worker-index",
                    str(index),
                    "--basis-manifest",
                    str(args.basis_manifest.resolve()),
                    "--output-dir",
                    str(output_dir),
                ]
                if args.stage == COLLECT_STAGE:
                    command.extend(
                        [
                            "--checkpoint",
                            str(args.checkpoint.resolve()),
                            "--split",
                            str(args.split.resolve()),
                            "--donor-mapping",
                            str(args.donor_mapping.resolve()),
                            "--donor-manifest",
                            str(args.donor_manifest.resolve()),
                            "--donor-root",
                            str(args.donor_root.resolve()),
                        ]
                    )
                else:
                    command.extend(
                        [
                            "--fit-dir",
                            str(args.fit_dir.resolve()),
                            "--heldout-gram-dir",
                            str(args.heldout_gram_dir.resolve()),
                            "--diagnostics",
                            str(args.diagnostics.resolve()),
                        ]
                    )
                handle = log_path.open("x", encoding="utf-8")
                handle.write(shlex.join(command) + "\n")
                handle.flush()
                environment = _child_environment(physical_gpu)
                environment["ASRE_ROUND4B_ENERGY_PHYSICAL_GPU"] = str(physical_gpu)
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(lock_fd,),
                )
                children.append((index, process, handle, log_path))
                if args.launch_stagger_seconds and index < 3:
                    time.sleep(args.launch_stagger_seconds)
            pending = list(children)
            while pending:
                for child in list(pending):
                    index, process, handle, log_path = child
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    pending.remove(child)
                    handle.close()
                    if returncode != 0:
                        failure = (
                            f"Energy stage {args.stage} shard {index} exited "
                            f"{returncode}; inspect {log_path}"
                        )
                        for _, other, other_handle, _ in pending:
                            if other.poll() is None:
                                os.killpg(other.pid, signal.SIGTERM)
                                other.wait(timeout=30)
                            other_handle.close()
                        pending.clear()
                        break
                if pending:
                    time.sleep(2)
        except KeyboardInterrupt:
            for _, process, handle, _ in children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                handle.close()
            raise
    all_complete = all(
        _valid_report(
            output_dir / _report_name(args.stage, index),
            stage=args.stage,
            index=index,
            commit=commit,
            basis_sha=basis_sha,
            split_sha=split_sha,
            diagnostics_sha=diagnostics_sha,
        )
        for index in range(4)
    )
    summary_path = output_dir / f"{args.stage}_launcher_summary.json"
    atomic_write_json(
        summary_path,
        {
            "artifact_type": "asre_round4b_energy_curve_launcher_summary",
            "schema_version": 1,
            "protocol": ROUND4B_PROTOCOL,
            "stage": args.stage,
            "all_succeeded": all_complete and failure is None,
            "failure": failure,
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "identity": identity,
            "online_episodes": 0,
            "environment_rollouts": 0,
            "heldout_svd_refit": False,
            "no_ddp": True,
        },
    )
    if not all_complete or failure:
        raise RuntimeError(failure or f"Energy stage {args.stage} is incomplete.")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(COLLECT_STAGE, RECOVER_STAGE), required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--basis-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--donor-mapping", type=Path)
    parser.add_argument("--donor-manifest", type=Path)
    parser.add_argument("--donor-root", type=Path)
    parser.add_argument("--fit-dir", type=Path)
    parser.add_argument("--heldout-gram-dir", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--launch-stagger-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.stage == COLLECT_STAGE and any(
        value is None
        for value in (
            args.checkpoint,
            args.split,
            args.donor_mapping,
            args.donor_manifest,
            args.donor_root,
        )
    ):
        parser.error("collect_heldout_grams requires checkpoint/split/donor inputs")
    if args.stage == RECOVER_STAGE and any(
        value is None for value in (args.fit_dir, args.heldout_gram_dir, args.diagnostics)
    ):
        parser.error("recover_coordinate_energy requires fit/heldout/diagnostics inputs")
    path = launch(args)
    print(f"Round-4B cumulative-energy stage complete: {path}")


if __name__ == "__main__":
    main()
