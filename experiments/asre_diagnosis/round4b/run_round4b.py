"""End-to-end fail-closed driver for ASRE Stage-2 Round-4B."""

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
    ROUND4B_PROTOCOL,
    atomic_write_json,
    now_iso,
    git_commit,
    sha256_file,
)


DEFAULT_ROUND4A = PROJECT_ROOT / "asre_results/round4a/retry_20260829_configfix"


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
        print(f"[Round4B] Reusing {name}: {stable_output}", flush=True)
        return
    logs.mkdir(parents=True, exist_ok=True)
    attempt = 1
    path = logs / f"{name}.log"
    while path.exists():
        attempt += 1
        path = logs / f"{name}.attempt{attempt:02d}.log"
    print(f"[Round4B] Starting {name}; log: {path}", flush=True)
    merged = os.environ.copy()
    if environment:
        merged.update(environment)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            list(command), cwd=PROJECT_ROOT, env=merged, stdout=handle, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"Round-4B stage {name} exited {result.returncode}; inspect {path}.")
    print(f"[Round4B] Completed {name}", flush=True)


def run(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    allowed = (PROJECT_ROOT / "asre_results/round4b_subspace").resolve()
    if output != allowed and not output.is_relative_to(allowed):
        raise ValueError(f"Output must be within {allowed}, got {output}.")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    python = str(args.python.resolve())
    gpu_ids = tuple(args.gpu_ids)
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("ROUND4B_GPU_IDS must name four distinct GPUs.")
    valid = PROJECT_ROOT / "asre_results/round2/state_bank_valid_manifest.json"
    donor_root = PROJECT_ROOT / "asre_results/round3b/donors/online"
    donor_mapping = donor_root / "donor_mapping.json"
    donor_manifest = donor_root / "donor_observation_manifest.json"
    round4a_summary = args.round4a_root.resolve() / "aggregate/round4a_summary.json"
    valid_payload = json.loads(valid.read_text(encoding="utf-8"))
    checkpoint = Path(valid_payload["checkpoint_path"]).resolve()
    stats = Path(valid_payload["dataset_stats_path"]).resolve()
    preflight = output / "preflight_report.json"
    split = output / "calibration/calibration_split_manifest.json"
    fit_dir = output / "calibration"
    basis = output / "calibration/basis_manifest.json"
    diagnostics = output / "calibration/subspace_diagnostics.json"
    machinery = output / "machinery_report.json"
    common_fit = [
        "--python", python,
        "--gpu-ids", *(str(value) for value in gpu_ids),
        "--checkpoint", str(checkpoint),
        "--split", str(split),
        "--donor-mapping", str(donor_mapping),
        "--donor-manifest", str(donor_manifest),
        "--donor-root", str(donor_root),
        "--output-dir", str(fit_dir),
        "--launch-stagger-seconds", str(args.launch_stagger_seconds),
    ]
    status_path = output / "driver_status.json"
    start = now_iso()
    atomic_write_json(
        status_path,
        {
            "artifact_type": "asre_round4b_driver_status",
            "schema_version": 1,
            "protocol": ROUND4B_PROTOCOL,
            "status": "running",
            "start_timestamp": start,
            "output_root": str(output),
            "gpu_ids": list(gpu_ids),
            "python": python,
            "later_stage_launched": False,
        },
    )
    _run(
        "preflight",
        [
            python, "-m", "experiments.asre_diagnosis.round4b.preflight",
            "--valid-manifest", str(valid),
            "--online-donor-mapping", str(donor_mapping),
            "--online-donor-manifest", str(donor_manifest),
            "--online-donor-root", str(donor_root),
            "--round4a-summary", str(round4a_summary),
            "--output", str(preflight),
        ],
        logs,
        stable_output=preflight,
    )
    _run(
        "unit_tests",
        [python, "-m", "pytest", "-q", "experiments/asre_diagnosis/tests"],
        logs,
    )
    _run(
        "prepare_split",
        [
            python, "-m", "experiments.asre_diagnosis.round4b.split",
            "--valid-manifest", str(valid), "--output", str(split),
        ],
        logs,
        stable_output=split,
    )
    _run(
        "fit_shards",
        [python, "-m", "experiments.asre_diagnosis.round4b.launch_fit", "--phase", "fit", *common_fit],
        logs,
    )
    _run(
        "finalize_bases",
        [
            python, "-m", "experiments.asre_diagnosis.round4b.finalize_bases",
            "--fit-dir", str(fit_dir), "--split", str(split), "--output", str(basis),
        ],
        logs,
        environment={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
        stable_output=basis,
    )
    _run(
        "holdout_diagnostics_shards",
        [
            python, "-m", "experiments.asre_diagnosis.round4b.launch_fit",
            "--phase", "holdout", *common_fit,
            "--basis-manifest", str(basis),
        ],
        logs,
    )
    _run(
        "merge_diagnostics",
        [
            python, "-m", "experiments.asre_diagnosis.round4b.merge_diagnostics",
            "--basis-manifest", str(basis), "--holdout-dir", str(fit_dir),
            "--output", str(diagnostics),
        ],
        logs,
        stable_output=diagnostics,
    )
    diagnostics_payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    if diagnostics_payload.get("basis_manifest_sha256") != sha256_file(basis):
        raise RuntimeError("Refusing stale Round-4B diagnostics from another basis manifest.")
    _run(
        "machinery_tests",
        [
            python, "-m", "experiments.asre_diagnosis.round4b.machinery_tests",
            "--checkpoint", str(checkpoint), "--preflight", str(preflight),
            "--split", str(split), "--basis-manifest", str(basis),
            "--diagnostics", str(diagnostics), "--donor-mapping", str(donor_mapping),
            "--donor-manifest", str(donor_manifest), "--donor-root", str(donor_root),
            "--output", str(machinery),
        ],
        logs,
        environment={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
        stable_output=machinery,
    )
    machinery_payload = json.loads(machinery.read_text(encoding="utf-8"))
    expected_machinery_links = {
        "preflight_report_sha256": sha256_file(preflight),
        "split_manifest_sha256": sha256_file(split),
        "basis_manifest_sha256": sha256_file(basis),
        "diagnostics_sha256": sha256_file(diagnostics),
    }
    if any(
        machinery_payload.get(key) != value
        for key, value in expected_machinery_links.items()
    ):
        raise RuntimeError("Refusing stale Round-4B machinery report provenance.")
    online_common = [
        "--python", python,
        "--preflight", str(preflight), "--machinery", str(machinery),
        "--split", str(split), "--basis-manifest", str(basis),
        "--diagnostics", str(diagnostics), "--round4a-summary", str(round4a_summary),
        "--launch-stagger-seconds", str(args.launch_stagger_seconds),
    ]
    smoke1 = output / "online_smoke/wave1"
    smoke2 = output / "online_smoke/wave2"
    full1 = output / "online_full/wave1"
    full2 = output / "online_full/wave2"
    _run(
        "online_smoke_wave1",
        [python, "-m", "experiments.asre_diagnosis.round4b.launch_wave", "--mode", "smoke", "--wave", "1", "--gpu-ids", *(str(x) for x in gpu_ids), "--output-root", str(smoke1), *online_common],
        logs,
    )
    _run(
        "online_smoke_wave2",
        [python, "-m", "experiments.asre_diagnosis.round4b.launch_wave", "--mode", "smoke", "--wave", "2", "--gpu-ids", *(str(x) for x in gpu_ids[:2]), "--output-root", str(smoke2), *online_common],
        logs,
    )
    _run(
        "online_full_wave1",
        [python, "-m", "experiments.asre_diagnosis.round4b.launch_wave", "--mode", "full", "--wave", "1", "--gpu-ids", *(str(x) for x in gpu_ids), "--output-root", str(full1), "--smoke-summary", str(smoke1 / "launcher_summary.json"), *online_common],
        logs,
    )
    _run(
        "online_full_wave2",
        [python, "-m", "experiments.asre_diagnosis.round4b.launch_wave", "--mode", "full", "--wave", "2", "--gpu-ids", *(str(x) for x in gpu_ids), "--output-root", str(full2), "--smoke-summary", str(smoke2 / "launcher_summary.json"), "--wave1-full-summary", str(full1 / "launcher_summary.json"), *online_common],
        logs,
    )
    offline = output / "offline"
    offline_common = [
        "--python", python, "--gpu-ids", *(str(x) for x in gpu_ids),
        "--checkpoint", str(checkpoint), "--dataset-stats", str(stats),
        "--split", str(split), "--basis-manifest", str(basis),
        "--donor-mapping", str(donor_mapping), "--donor-manifest", str(donor_manifest),
        "--donor-root", str(donor_root), "--output-root", str(offline),
        "--launch-stagger-seconds", str(args.launch_stagger_seconds),
    ]
    for wave in (1, 2):
        _run(
            f"offline_wave{wave}",
            [python, "-m", "experiments.asre_diagnosis.round4b.launch_offline_wave", "--wave", str(wave), *offline_common],
            logs,
        )
    aggregate = output / "aggregate"
    _run(
        "aggregate",
        [
            python, "-m", "experiments.asre_diagnosis.round4b.aggregate_results",
            "--online-wave1", str(full1), "--online-wave2", str(full2),
            "--offline-root", str(offline), "--dataset-stats", str(stats),
            "--preflight", str(preflight), "--split", str(split),
            "--basis-manifest", str(basis), "--diagnostics", str(diagnostics),
            "--machinery", str(machinery), "--round4a-summary", str(round4a_summary),
            "--output-dir", str(aggregate),
        ],
        logs,
    )
    _run(
        "plots",
        [python, "-m", "experiments.asre_diagnosis.round4b.plot_results", "--aggregate-dir", str(aggregate)],
        logs,
    )
    report = aggregate / "round4b_summary.md"
    atomic_write_json(
        status_path,
        {
            "artifact_type": "asre_round4b_driver_status",
            "schema_version": 1,
            "protocol": ROUND4B_PROTOCOL,
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
    print(f"Round 4B completed. Final report: {report}")
    print("No later-stage experiment was launched.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round4a-root", type=Path, default=DEFAULT_ROUND4A)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--launch-stagger-seconds", type=float, default=30.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
