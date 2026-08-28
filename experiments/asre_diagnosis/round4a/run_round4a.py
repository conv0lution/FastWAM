"""End-to-end fail-closed driver for ASRE Stage-2 Round-4A."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4A_PROTOCOL,
    atomic_write_json,
    now_iso,
)


DEFAULT_G0_ROOT = (
    PROJECT_ROOT
    / "asre_results/g0_cross_suite/retry_20260828_promptcache"
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _require_output_scope(output_root: Path) -> None:
    output_root = output_root.resolve()
    allowed_root = (PROJECT_ROOT / "asre_results/round4a").resolve()
    if output_root != allowed_root and not output_root.is_relative_to(allowed_root):
        raise ValueError(
            "ROUND4A_OUTPUT_ROOT must be asre_results/round4a or one of its "
            f"descendants, got {output_root}."
        )
    protected = [
        (PROJECT_ROOT / "asre_results/round2").resolve(),
        (PROJECT_ROOT / "asre_results/round3b").resolve(),
        (PROJECT_ROOT / "asre_results/g0_cross_suite").resolve(),
        (PROJECT_ROOT / "asre_results/state_bank").resolve(),
    ]
    if any(
        output_root == path
        or output_root.is_relative_to(path)
        or path.is_relative_to(output_root)
        for path in protected
    ):
        raise ValueError(f"Round-4A output overlaps a frozen parent tree: {output_root}.")


def _run_stage(
    *,
    name: str,
    command: Sequence[str],
    log_dir: Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    if log_path.exists():
        attempt = 2
        while (log_dir / f"{name}.attempt{attempt:02d}.log").exists():
            attempt += 1
        log_path = log_dir / f"{name}.attempt{attempt:02d}.log"
    print(f"[Round4A] Starting {name}; log: {log_path}", flush=True)
    merged_environment = os.environ.copy()
    if environment:
        merged_environment.update(environment)
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            env=merged_environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Round-4A stage {name} exited {completed.returncode}; inspect {log_path}."
        )
    print(f"[Round4A] Completed {name}", flush=True)


def _gpu_ids(value: str) -> tuple[int, int, int, int]:
    tokens = value.replace(",", " ").split()
    ids = tuple(int(token) for token in tokens)
    if len(ids) != 4 or len(set(ids)) != 4 or any(gpu < 0 for gpu in ids):
        raise ValueError(
            f"ROUND4A_GPU_IDS must contain four distinct nonnegative IDs, got {ids}."
        )
    return ids  # type: ignore[return-value]


def run(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    _require_output_scope(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    python = str(Path(args.python).expanduser().resolve())
    gpu_ids = _gpu_ids(args.gpu_ids)
    g0_root = args.g0_root.expanduser().resolve()

    valid_manifest = PROJECT_ROOT / "asre_results/round2/state_bank_valid_manifest.json"
    online_donor_root = PROJECT_ROOT / "asre_results/round3b/donors/online"
    online_mapping = online_donor_root / "donor_mapping.json"
    online_manifest = online_donor_root / "donor_observation_manifest.json"
    offline_mapping = PROJECT_ROOT / "asre_results/round3b/donors/offline_donor_mapping.json"
    round3b_summary = PROJECT_ROOT / "asre_results/round3b/aggregate/round3b_summary.json"
    g0_summary = g0_root / "aggregate/g0_summary.json"
    preflight = output_root / "preflight_report.json"
    machinery = output_root / "machinery_report.json"
    masks = output_root / "masks/mask_manifest.json"
    valid = _read_json(valid_manifest)
    checkpoint = Path(str(valid["checkpoint_path"])).resolve()
    dataset_stats = Path(str(valid["dataset_stats_path"])).resolve()
    state_bank_dir = Path(str(valid["source_manifest_path"])).resolve().parent

    start = now_iso()
    driver_status_path = output_root / "driver_status.json"
    driver_status = {
        "artifact_type": "asre_round4a_driver_status",
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "status": "running",
        "start_timestamp": start,
        "output_root": str(output_root),
        "gpu_ids": list(gpu_ids),
        "python": python,
        "later_stage_launched": False,
    }
    atomic_write_json(driver_status_path, driver_status)

    _run_stage(
        name="preflight",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.preflight",
            "--valid-manifest",
            str(valid_manifest),
            "--online-donor-mapping",
            str(online_mapping),
            "--online-donor-manifest",
            str(online_manifest),
            "--online-donor-root",
            str(online_donor_root),
            "--offline-donor-mapping",
            str(offline_mapping),
            "--round3b-summary",
            str(round3b_summary),
            "--g0-summary",
            str(g0_summary),
            "--output-report",
            str(preflight),
        ],
    )
    _run_stage(
        name="unit_tests",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "pytest",
            "-q",
            "experiments/asre_diagnosis/tests",
        ],
    )
    _run_stage(
        name="machinery_tests",
        log_dir=log_dir,
        environment={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.machinery_tests",
            "--checkpoint",
            str(checkpoint),
            "--dataset-stats",
            str(dataset_stats),
            "--valid-manifest",
            str(valid_manifest),
            "--state-bank-dir",
            str(state_bank_dir),
            "--offline-donor-mapping",
            str(offline_mapping),
            "--round3b-current-actions",
            str(PROJECT_ROOT / "asre_results/round3b/offline/late_current_correct/actions.npz"),
            "--round3b-wrong-actions",
            str(PROJECT_ROOT / "asre_results/round3b/offline/late_wrong_scene/actions.npz"),
            "--round3b-wrong-cache-stats",
            str(PROJECT_ROOT / "asre_results/round3b/offline/late_wrong_scene/video_cache_stats.jsonl"),
            "--preflight-report",
            str(preflight),
            "--mask-manifest",
            str(masks),
            "--output",
            str(machinery),
        ],
    )

    launcher_common = [
        "--preflight-report",
        str(preflight),
        "--machinery-report",
        str(machinery),
        "--mask-manifest",
        str(masks),
        "--python",
        python,
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
    ]
    smoke1_root = output_root / "online_smoke/wave1"
    smoke2_root = output_root / "online_smoke/wave2"
    full1_root = output_root / "online_full/wave1"
    full2_root = output_root / "online_full/wave2"
    _run_stage(
        name="online_smoke_wave1",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.launch_wave",
            "--mode",
            "smoke",
            "--wave",
            "1",
            "--output-root",
            str(smoke1_root),
            "--gpu-ids",
            *(str(gpu) for gpu in gpu_ids),
            *launcher_common,
        ],
    )
    _run_stage(
        name="online_smoke_wave2",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.launch_wave",
            "--mode",
            "smoke",
            "--wave",
            "2",
            "--output-root",
            str(smoke2_root),
            "--gpu-ids",
            str(gpu_ids[0]),
            str(gpu_ids[2]),
            *launcher_common,
        ],
    )
    _run_stage(
        name="online_full_wave1",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.launch_wave",
            "--mode",
            "full",
            "--wave",
            "1",
            "--output-root",
            str(full1_root),
            "--smoke-summary",
            str(smoke1_root / "launcher_summary.json"),
            "--gpu-ids",
            *(str(gpu) for gpu in gpu_ids),
            *launcher_common,
        ],
    )
    _run_stage(
        name="online_full_wave2",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.launch_wave",
            "--mode",
            "full",
            "--wave",
            "2",
            "--output-root",
            str(full2_root),
            "--smoke-summary",
            str(smoke2_root / "launcher_summary.json"),
            "--wave1-full-summary",
            str(full1_root / "launcher_summary.json"),
            "--gpu-ids",
            *(str(gpu) for gpu in gpu_ids),
            *launcher_common,
        ],
    )

    offline_root = output_root / "offline"
    offline_common = [
        "--preflight-report",
        str(preflight),
        "--machinery-report",
        str(machinery),
        "--mask-manifest",
        str(masks),
        "--output-root",
        str(offline_root),
        "--python",
        python,
        "--gpu-ids",
        *(str(gpu) for gpu in gpu_ids),
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
    ]
    _run_stage(
        name="offline_wave1",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.launch_offline_wave",
            "--wave",
            "1",
            *offline_common,
        ],
    )
    _run_stage(
        name="offline_wave2",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.launch_offline_wave",
            "--wave",
            "2",
            "--wave1-summary",
            str(offline_root / "offline_wave1_launcher_summary.json"),
            *offline_common,
        ],
    )

    aggregate_dir = output_root / "aggregate"
    _run_stage(
        name="aggregate",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.aggregate_results",
            "--preflight-report",
            str(preflight),
            "--machinery-report",
            str(machinery),
            "--mask-manifest",
            str(masks),
            "--smoke-wave1-summary",
            str(smoke1_root / "launcher_summary.json"),
            "--smoke-wave2-summary",
            str(smoke2_root / "launcher_summary.json"),
            "--online-wave1-root",
            str(full1_root),
            "--online-wave2-root",
            str(full2_root),
            "--offline-root",
            str(offline_root),
            "--output-dir",
            str(aggregate_dir),
        ],
    )
    _run_stage(
        name="plots",
        log_dir=log_dir,
        command=[
            python,
            "-m",
            "experiments.asre_diagnosis.round4a.plot_results",
            "--aggregate-dir",
            str(aggregate_dir),
        ],
    )
    driver_status.update(
        {
            "status": "complete",
            "end_timestamp": now_iso(),
            "final_report": str(aggregate_dir / "round4a_summary.md"),
            "later_stage_launched": False,
        }
    )
    atomic_write_json(driver_status_path, driver_status)
    print(f"Round 4A completed. Final report: {driver_status['final_report']}")
    print("No later Stage-2 experiment was launched.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "ROUND4A_OUTPUT_ROOT", PROJECT_ROOT / "asre_results/round4a"
            )
        ),
    )
    parser.add_argument(
        "--g0-root",
        type=Path,
        default=Path(os.environ.get("ROUND4A_G0_ROOT", DEFAULT_G0_ROOT)),
    )
    parser.add_argument(
        "--python", default=os.environ.get("ROUND4A_PYTHON", sys.executable)
    )
    parser.add_argument(
        "--gpu-ids", default=os.environ.get("ROUND4A_GPU_IDS", "0 1 2 3")
    )
    parser.add_argument(
        "--launch-stagger-seconds",
        type=float,
        default=float(os.environ.get("ROUND4A_LAUNCH_STAGGER_SECONDS", "30")),
    )
    return parser.parse_args()


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
