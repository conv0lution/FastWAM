"""End-to-end fail-closed driver for ASRE Stage-2 Round-4C."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND4C_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)


DEFAULT_ROUND4B = PROJECT_ROOT / "asre_results/round4b_subspace"


def _run(
    name: str,
    command: Sequence[str],
    logs: Path,
    *,
    environment: Mapping[str, str] | None = None,
    stable_output: Path | None = None,
) -> None:
    if stable_output is not None and stable_output.exists():
        if stable_output.suffix == ".json":
            payload = json.loads(stable_output.read_text(encoding="utf-8"))
            recorded_commit = payload.get("git_commit_hash")
            if recorded_commit is None and isinstance(payload.get("git"), dict):
                recorded_commit = payload["git"].get("current_head")
            if recorded_commit is not None and recorded_commit != git_commit(PROJECT_ROOT):
                raise RuntimeError(
                    f"Refusing to reuse {name} from source commit {recorded_commit}; "
                    f"current HEAD is {git_commit(PROJECT_ROOT)}. Use a new output root."
                )
        print(f"[Round4C] Reusing {name}: {stable_output}", flush=True)
        return
    logs.mkdir(parents=True, exist_ok=True)
    attempt = 1
    path = logs / f"{name}.log"
    while path.exists():
        attempt += 1
        path = logs / f"{name}.attempt{attempt:02d}.log"
    print(f"[Round4C] Starting {name}; log: {path}", flush=True)
    merged = os.environ.copy()
    if environment:
        merged.update(environment)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            env=merged,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            f"Round-4C stage {name} exited {result.returncode}; inspect {path}."
        )
    print(f"[Round4C] Completed {name}", flush=True)


def run(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    allowed = (PROJECT_ROOT / "asre_results/round4c_energy_sufficiency").resolve()
    if output != allowed and not output.is_relative_to(allowed):
        raise ValueError(f"Output must be within {allowed}, got {output}.")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    python = str(args.python.resolve())
    gpu_ids = tuple(args.gpu_ids)
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("ROUND4C_GPU_IDS must name four distinct GPUs.")
    round4b = args.round4b_root.resolve()
    basis = round4b / "calibration/basis_manifest.json"
    split = round4b / "calibration/calibration_split_manifest.json"
    diagnostics = round4b / "calibration/subspace_diagnostics.json"
    energy_manifest = round4b / "energy_curve_analysis/energy_curve_analysis_manifest.json"
    energy_candidates = round4b / "energy_curve_analysis/candidate_next_online_ranks.json"
    round4b_summary = round4b / "aggregate/round4b_summary.json"
    preflight = output / "preflight_report.json"
    machinery = output / "machinery_report.json"
    status_path = output / "driver_status.json"
    start = now_iso()
    atomic_write_json(
        status_path,
        {
            "artifact_type": "asre_round4c_driver_status",
            "schema_version": 1,
            "protocol": ROUND4C_PROTOCOL,
            "status": "running",
            "start_timestamp": start,
            "output_root": str(output),
            "gpu_ids": list(gpu_ids),
            "wave_gpu_mapping": {
                "wave1": {
                    str(gpu_ids[0]): "current_all",
                    str(gpu_ids[1]): "wrong_all",
                    str(gpu_ids[2]): "svd_r36",
                    str(gpu_ids[3]): "svd_r97",
                },
                "wave2": {str(gpu_ids[0]): "svd_r170"},
            },
            "python": python,
            "later_stage_launched": False,
        },
    )
    _run(
        "preflight",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4c.preflight",
            "--round4b-root",
            str(round4b),
            "--output-root",
            str(output),
            "--output",
            str(preflight),
        ],
        logs,
        stable_output=preflight,
    )
    _run(
        "unit_tests",
        [python, "-m", "pytest", "-q", "experiments/asre_diagnosis/tests"],
        logs,
    )
    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    state = preflight_payload["state_bank"]
    donors = preflight_payload["donors"]
    _run(
        "machinery_tests",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4c.machinery_tests",
            "--checkpoint",
            state["checkpoint_path"],
            "--preflight",
            str(preflight),
            "--split",
            str(split),
            "--basis-manifest",
            str(basis),
            "--donor-mapping",
            donors["mapping_path"],
            "--donor-manifest",
            donors["observation_manifest_path"],
            "--donor-root",
            donors["observation_root"],
            "--output",
            str(machinery),
        ],
        logs,
        environment={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
        stable_output=machinery,
    )
    machinery_payload = json.loads(machinery.read_text(encoding="utf-8"))
    expected_links = {
        "preflight_report_sha256": sha256_file(preflight),
        "split_manifest_sha256": sha256_file(split),
        "basis_manifest_sha256": sha256_file(basis),
    }
    if machinery_payload.get("passed") is not True or any(
        machinery_payload.get(key) != value for key, value in expected_links.items()
    ):
        raise RuntimeError("Refusing stale or failed Round-4C machinery report.")
    online_common = [
        "--python",
        python,
        "--preflight",
        str(preflight),
        "--machinery",
        str(machinery),
        "--split",
        str(split),
        "--basis-manifest",
        str(basis),
        "--diagnostics",
        str(diagnostics),
        "--energy-manifest",
        str(energy_manifest),
        "--energy-candidates",
        str(energy_candidates),
        "--round4b-summary",
        str(round4b_summary),
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
    ]
    smoke1 = output / "online_smoke/wave1"
    smoke2 = output / "online_smoke/wave2"
    full1 = output / "online_full/wave1"
    full2 = output / "online_full/wave2"
    _run(
        "online_smoke_wave1",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4c.launch_wave",
            "--mode",
            "smoke",
            "--wave",
            "1",
            "--gpu-ids",
            *(str(value) for value in gpu_ids),
            "--output-root",
            str(smoke1),
            *online_common,
        ],
        logs,
    )
    _run(
        "online_full_wave1",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4c.launch_wave",
            "--mode",
            "full",
            "--wave",
            "1",
            "--gpu-ids",
            *(str(value) for value in gpu_ids),
            "--output-root",
            str(full1),
            "--smoke-summary",
            str(smoke1 / "launcher_summary.json"),
            *online_common,
        ],
        logs,
    )
    _run(
        "online_smoke_wave2",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4c.launch_wave",
            "--mode",
            "smoke",
            "--wave",
            "2",
            "--gpu-ids",
            str(gpu_ids[0]),
            "--output-root",
            str(smoke2),
            "--wave1-smoke-summary",
            str(smoke1 / "launcher_summary.json"),
            "--wave1-full-summary",
            str(full1 / "launcher_summary.json"),
            *online_common,
        ],
        logs,
    )
    _run(
        "online_full_wave2",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4c.launch_wave",
            "--mode",
            "full",
            "--wave",
            "2",
            "--gpu-ids",
            str(gpu_ids[0]),
            "--output-root",
            str(full2),
            "--smoke-summary",
            str(smoke2 / "launcher_summary.json"),
            "--wave1-full-summary",
            str(full1 / "launcher_summary.json"),
            *online_common,
        ],
        logs,
    )
    aggregate = output / "aggregate"
    _run(
        "aggregate",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4c.aggregate_results",
            "--online-wave1",
            str(full1),
            "--online-wave2",
            str(full2),
            "--preflight",
            str(preflight),
            "--machinery",
            str(machinery),
            "--energy-manifest",
            str(energy_manifest),
            "--energy-candidates",
            str(energy_candidates),
            "--round4b-summary",
            str(round4b_summary),
            "--output-dir",
            str(aggregate),
        ],
        logs,
    )
    _run(
        "plots",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.round4c.plot_results",
            "--aggregate-dir",
            str(aggregate),
        ],
        logs,
    )
    report = aggregate / "round4c_summary.md"
    atomic_write_json(
        status_path,
        {
            "artifact_type": "asre_round4c_driver_status",
            "schema_version": 1,
            "protocol": ROUND4C_PROTOCOL,
            "status": "complete",
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "output_root": str(output),
            "gpu_ids": list(gpu_ids),
            "python": python,
            "final_report": str(report),
            "later_stage_launched": False,
        },
    )
    print(f"Round 4C completed. Final report: {report}")
    print("No later-stage experiment was launched.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round4b-root", type=Path, default=DEFAULT_ROUND4B)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--launch-stagger-seconds", type=float, default=30.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
