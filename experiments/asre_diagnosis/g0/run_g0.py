"""End-to-end, fail-closed, safely resumable driver for ASRE G0."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import G0_PROTOCOL, git_commit, now_iso, sha256_file
from experiments.asre_diagnosis.g0.definitions import (
    SUITE_ORDER,
    assert_output_scope,
    validate_gpu_mapping,
)
from experiments.asre_diagnosis.g0.preflight import validate_git_state


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _base_environment(output_root: Path, rendering_gpu: int) -> dict[str, str]:
    environment = os.environ.copy()
    libero_root = Path(
        environment.get("LIBERO_ROOT", str(project_root.parent / "LIBERO"))
    ).resolve()
    python_entries = [str(project_root / "src"), str(project_root), str(libero_root)]
    if environment.get("PYTHONPATH"):
        python_entries.append(environment["PYTHONPATH"])
    matplotlib_root = output_root / ".matplotlib"
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    numba_root = output_root / ".numba"
    numba_root.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(rendering_gpu),
            "PYTHONPATH": os.pathsep.join(python_entries),
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(rendering_gpu),
            "MPLCONFIGDIR": str(matplotlib_root),
            "NUMBA_CACHE_DIR": str(numba_root),
        }
    )
    return environment


def _prompt_cache_environment(
    base_environment: Mapping[str, str], gpu_ids: Sequence[int]
) -> dict[str, str]:
    """Expose two dedicated cards: model on logical 0, T5 on logical 1."""

    if len(gpu_ids) < 2 or gpu_ids[0] == gpu_ids[1]:
        raise ValueError("G0 prompt preparation requires two distinct physical GPUs.")
    environment = dict(base_environment)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": f"{int(gpu_ids[0])},{int(gpu_ids[1])}",
            "MUJOCO_EGL_DEVICE_ID": str(int(gpu_ids[0])),
        }
    )
    return environment


def _run_stage(
    command: Sequence[str], *, log_path: Path, environment: Mapping[str, str]
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stage_name = log_path.stem
    print(f"[G0] Starting {stage_name}; log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now_iso()}] {shlex.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            list(command),
            cwd=project_root,
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"G0 stage exited {process.returncode}; inspect {log_path}."
        )
    print(f"[G0] Completed {stage_name}", flush=True)


def _validate_existing_preflight(
    path: Path, checkpoint: Path, dataset_stats: Path, *, allow_dirty: bool
) -> None:
    payload = _read_json(path, "G0 preflight report")
    expected = {
        "artifact_type": "asre_g0_preflight_report",
        "status": "compatible",
        "output_root": str(path.parent.resolve()),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing preflight is incompatible: {path}.")
    if payload.get("git", {}).get("head") != git_commit(project_root):
        raise ValueError("Existing G0 preflight belongs to another Git commit.")
    if bool(payload.get("git", {}).get("allow_dirty")) != bool(allow_dirty):
        raise ValueError("Existing G0 preflight dirty-worktree mode differs.")
    if payload.get("checkpoint", {}).get("path") != str(checkpoint.resolve()) or payload.get(
        "dataset_statistics", {}
    ).get("path") != str(dataset_stats.resolve()):
        raise ValueError("Existing G0 preflight uses different checkpoint/statistics paths.")


def _validate_existing_machinery(path: Path, preflight: Path) -> None:
    payload = _read_json(path, "G0 machinery report")
    expected = {
        "artifact_type": "asre_g0_machinery_report",
        "protocol": G0_PROTOCOL,
        "status": "passed",
        "passed": True,
        "git_commit_hash": git_commit(project_root),
        "preflight_report_path": str(preflight.resolve()),
        "preflight_report_sha256": sha256_file(preflight),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing G0 machinery report is incompatible: {path}.")


def run_driver(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    assert_output_scope(output_root, project_root)
    output_root.mkdir(parents=True, exist_ok=True)
    python = args.python.expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python executable is unavailable: {python}.")
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_stats = args.dataset_stats.expanduser().resolve()
    valid_manifest = args.valid_manifest.expanduser().resolve()
    state_bank_dir = args.state_bank_dir.expanduser().resolve()
    gpu_ids = validate_gpu_mapping(args.gpu_ids)
    environment = _base_environment(output_root, gpu_ids[0])
    logs = output_root / "logs"
    preflight = output_root / "preflight_report.json"
    machinery = output_root / "machinery_report.json"

    if preflight.exists():
        validate_git_state(allow_dirty=args.allow_dirty)
        _validate_existing_preflight(
            preflight, checkpoint, dataset_stats, allow_dirty=args.allow_dirty
        )
    else:
        command = [
            str(python),
            "-m",
            "experiments.asre_diagnosis.g0.preflight",
            "--checkpoint",
            str(checkpoint),
            "--dataset-stats",
            str(dataset_stats),
            "--output-root",
            str(output_root),
            "--output",
            str(preflight),
        ]
        if args.allow_dirty:
            command.append("--allow-dirty")
        _run_stage(command, log_path=logs / "preflight.log", environment=environment)
    if args.preflight_only:
        print(f"Development preflight completed: {preflight}")
        return
    if args.allow_dirty:
        raise RuntimeError(
            "--allow-dirty is development-only and cannot proceed to formal G0 tests/runs."
        )

    _run_stage(
        [
            str(python),
            "-m",
            "pytest",
            "experiments/asre_diagnosis/tests",
            "-q",
        ],
        log_path=logs / "unit_tests.log",
        environment=environment,
    )

    if machinery.exists():
        _validate_existing_machinery(machinery, preflight)
    else:
        machinery_environment = dict(environment)
        machinery_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_ids[0]),
                "MUJOCO_EGL_DEVICE_ID": str(gpu_ids[0]),
            }
        )
        _run_stage(
            [
                str(python),
                "-m",
                "experiments.asre_diagnosis.g0.machinery_tests",
                "--checkpoint",
                str(checkpoint),
                "--dataset-stats",
                str(dataset_stats),
                "--valid-manifest",
                str(valid_manifest),
                "--state-bank-dir",
                str(state_bank_dir),
                "--preflight-report",
                str(preflight),
                "--output",
                str(machinery),
            ],
            log_path=logs / "machinery_tests.log",
            environment=machinery_environment,
        )

    prompt_cache_root = output_root / "prompt_contexts"
    _run_stage(
        [
            str(python),
            "-m",
            "experiments.asre_diagnosis.g0.prepare_prompt_contexts",
            "--checkpoint",
            str(checkpoint),
            "--preflight-report",
            str(preflight),
            "--output-root",
            str(prompt_cache_root),
        ],
        log_path=logs / "prepare_prompt_contexts.log",
        environment=_prompt_cache_environment(environment, gpu_ids),
    )

    for suite in SUITE_ORDER:
        suite_root = output_root / suite
        donor_root = suite_root / "donors"
        _run_stage(
            [
                str(python),
                "-m",
                "experiments.asre_diagnosis.g0.prepare_donors",
                "--suite",
                suite,
                "--output-root",
                str(donor_root),
                "--dataset-stats",
                str(dataset_stats),
            ],
            log_path=logs / f"{suite}.prepare_donors.log",
            environment=environment,
        )
        donor_mapping = donor_root / "donor_mapping.json"
        donor_manifest = donor_root / "donor_observation_manifest.json"
        common = [
            "--suite",
            suite,
            "--checkpoint",
            str(checkpoint),
            "--dataset-stats",
            str(dataset_stats),
            "--prompt-context-cache",
            str(prompt_cache_root / f"{suite}.pt"),
            "--donor-mapping",
            str(donor_mapping),
            "--donor-manifest",
            str(donor_manifest),
            "--donor-root",
            str(donor_root),
            "--preflight-report",
            str(preflight),
            "--machinery-report",
            str(machinery),
            "--python",
            str(python),
            "--gpu-ids",
            *(str(gpu_id) for gpu_id in gpu_ids),
            "--launch-stagger-seconds",
            str(args.launch_stagger_seconds),
        ]
        smoke_root = suite_root / "smoke"
        _run_stage(
            [
                str(python),
                "-m",
                "experiments.asre_diagnosis.g0.launch_four_gpu",
                "--mode",
                "smoke",
                "--output-root",
                str(smoke_root),
                *common,
            ],
            log_path=logs / f"{suite}.smoke.log",
            environment=environment,
        )
        full_root = suite_root / "full"
        _run_stage(
            [
                str(python),
                "-m",
                "experiments.asre_diagnosis.g0.launch_four_gpu",
                "--mode",
                "full",
                "--output-root",
                str(full_root),
                "--smoke-summary",
                str(smoke_root / "launcher_summary.json"),
                *common,
            ],
            log_path=logs / f"{suite}.full.log",
            environment=environment,
        )
        full_summary = _read_json(full_root / "launcher_summary.json", "full summary")
        if full_summary.get("all_succeeded") is not True:
            raise RuntimeError(f"G0 suite did not validate: {suite}.")
        # The loop advances only after this suite's four full conditions validate.

    aggregate_dir = output_root / "aggregate"
    _run_stage(
        [
            str(python),
            "-m",
            "experiments.asre_diagnosis.g0.aggregate_results",
            "--g0-root",
            str(output_root),
            "--output-dir",
            str(aggregate_dir),
            "--bootstrap-samples",
            str(args.bootstrap_samples),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
        ],
        log_path=logs / "aggregate.log",
        environment=environment,
    )
    _run_stage(
        [
            str(python),
            "-m",
            "experiments.asre_diagnosis.g0.plot_results",
            "--summary",
            str(aggregate_dir / "g0_summary.json"),
            "--output-dir",
            str(aggregate_dir / "plots"),
        ],
        log_path=logs / "plots.log",
        environment=environment,
    )
    print(f"G0 completed. Final report: {aggregate_dir / 'g0_summary.md'}")
    print("Stage 2 Round 4A was NOT launched.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--state-bank-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--launch-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not math.isfinite(args.launch_stagger_seconds) or args.launch_stagger_seconds < 0:
        raise ValueError("--launch-stagger-seconds must be finite and nonnegative.")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive.")
    run_driver(args)


if __name__ == "__main__":
    main()
