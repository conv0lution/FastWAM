"""End-to-end fail-closed driver for the final compact-ASRE Salvage A test."""

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
    SALVAGE_A_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.salvage_a.definitions import (  # noqa: E402
    CONDITIONS,
    WAVES,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "asre_results/salvage_a_action_sensitive"
DEFAULT_ROUND4C_SUMMARY = (
    PROJECT_ROOT
    / "asre_results/round4c_energy_sufficiency/retry_20260829_donorcheckfix"
    / "aggregate/round4c_summary.json"
)
DEFAULT_VALID_MANIFEST = PROJECT_ROOT / "asre_results/round2/state_bank_valid_manifest.json"
DEFAULT_DONOR_ROOT = PROJECT_ROOT / "asre_results/round3b/donors/online"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _recorded_commit(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("git_commit_hash")
    if value is None and isinstance(payload.get("git"), Mapping):
        value = payload["git"].get("current_head")
    return None if value is None else str(value)


def _assert_clean_worktree(output: Path) -> None:
    """Keep resume checks meaningful even when the preflight JSON is reused."""

    relative = output.relative_to(PROJECT_ROOT)
    result = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=normal",
            "--",
            ".",
            f":(exclude){relative}/**",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if result:
        raise RuntimeError(
            "Salvage-A formal execution requires committed reviewed source code.\n"
            + result
        )


def _run(
    name: str,
    command: Sequence[str],
    logs: Path,
    *,
    environment: Mapping[str, str] | None = None,
    stable_output: Path | None = None,
) -> None:
    if stable_output is not None and stable_output.is_file():
        payload = _read(stable_output)
        recorded = _recorded_commit(payload)
        current = git_commit(PROJECT_ROOT)
        if recorded is not None and recorded != current:
            raise RuntimeError(
                f"Refusing to reuse {name} from source commit {recorded}; "
                f"current HEAD is {current}. Use a new output root."
            )
        print(f"[SalvageA] Reusing {name}: {stable_output}", flush=True)
        return
    logs.mkdir(parents=True, exist_ok=True)
    attempt = 1
    path = logs / f"{name}.log"
    while path.exists():
        attempt += 1
        path = logs / f"{name}.attempt{attempt:02d}.log"
    print(f"[SalvageA] Starting {name}; log: {path}", flush=True)
    merged = os.environ.copy()
    libero_root = Path(
        merged.get("LIBERO_ROOT", str(PROJECT_ROOT.parent / "LIBERO"))
    ).resolve()
    python_entries = [
        str(PROJECT_ROOT / "src"),
        str(PROJECT_ROOT),
        str(libero_root),
    ]
    python_entries.extend(
        entry
        for entry in merged.get("PYTHONPATH", "").split(os.pathsep)
        if entry and entry not in python_entries
    )
    merged["PYTHONPATH"] = os.pathsep.join(python_entries)
    merged.setdefault(
        "DIFFSYNTH_MODEL_BASE_PATH",
        "/local_home/zhaizicheng/fastwam_assets/checkpoints/wan_base",
    )
    merged.setdefault("PYTHONUNBUFFERED", "1")
    merged.setdefault("HYDRA_FULL_ERROR", "1")
    merged.setdefault("NUMBA_DISABLE_JIT", "1")
    merged.setdefault("NUMBA_CACHE_DIR", "/tmp/fastwam-salvage-a-numba-cache")
    merged.setdefault("MPLCONFIGDIR", "/tmp/fastwam-salvage-a-matplotlib-cache")
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
            f"Salvage-A stage {name} exited {result.returncode}; inspect {path}."
        )
    print(f"[SalvageA] Completed {name}", flush=True)


def _require_links(
    payload: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    mismatch = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Refusing stale or incompatible {label}: {mismatch}")


def run(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    if output != allowed and not output.is_relative_to(allowed):
        raise ValueError(f"Output must be within {allowed}, got {output}.")
    output.mkdir(parents=True, exist_ok=True)
    _assert_clean_worktree(output)

    gpu_ids = tuple(int(value) for value in args.gpu_ids)
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or any(value < 0 for value in gpu_ids):
        raise ValueError("SALVAGE_A_GPU_IDS must name four distinct nonnegative GPUs.")
    python = str(args.python.resolve())
    logs = output / "logs"
    round4c = args.round4c_summary.resolve()
    valid = args.valid_manifest.resolve()
    source_donor_root = args.source_donor_root.resolve()
    donor_manifest = source_donor_root / "donor_observation_manifest.json"

    preflight = output / "preflight_report.json"
    split = output / "split/calibration_split_manifest.json"
    donor_mapping = output / "donors/donor_mapping.json"
    calibration = output / "calibration"
    state_selection = calibration / "state_selection_manifest.json"
    differentiable = calibration / "differentiable_path_report.json"
    fit_dir = calibration / "fit_shards"
    heldout_dir = calibration / "heldout_diagnostics_shards"
    basis = calibration / "basis_manifest.json"
    diagnostics = calibration / "subspace_diagnostics.json"
    machinery = output / "machinery_report.json"
    status = output / "driver_status.json"
    aggregate_dir = output / "aggregate"
    start = now_iso()
    atomic_write_json(
        status,
        {
            "artifact_type": "asre_salvage_a_driver_status",
            "schema_version": 1,
            "protocol": SALVAGE_A_PROTOCOL,
            "status": "running",
            "start_timestamp": start,
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "output_root": str(output),
            "python": python,
            "gpu_ids": list(gpu_ids),
            "wave_gpu_mapping": {
                f"wave{wave}": {
                    str(gpu_ids[slot]): CONDITIONS[index]
                    for slot, index in enumerate(WAVES[wave])
                }
                for wave in (1, 2)
            },
            "primary_analysis_scope": "heldout_50",
            "calibration_and_all100_descriptive_only": True,
            "later_stage_launched": False,
        },
    )

    _run(
        "preflight",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.preflight",
            "--round4c-summary",
            str(round4c),
            "--valid-manifest",
            str(valid),
            "--donor-manifest",
            str(donor_manifest),
            "--donor-root",
            str(source_donor_root),
            "--output-root",
            str(output),
            "--output",
            str(preflight),
        ],
        logs,
        stable_output=preflight,
    )
    preflight_payload = _read(preflight)
    _require_links(
        preflight_payload,
        {"protocol": SALVAGE_A_PROTOCOL, "status": "compatible"},
        label="preflight report",
    )
    if preflight_payload.get("git", {}).get("current_head") != git_commit(PROJECT_ROOT):
        raise RuntimeError("Refusing a preflight report from another source commit.")
    state = preflight_payload["state_bank"]

    _run(
        "unit_tests",
        [python, "-m", "pytest", "-q", "experiments/asre_diagnosis/tests"],
        logs,
    )
    _run(
        "prepare_split",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.split",
            "--valid-manifest",
            str(valid),
            "--output",
            str(split),
        ],
        logs,
    )
    _run(
        "prepare_split_local_donors",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.donor",
            "--split",
            str(split),
            "--observation-manifest",
            str(donor_manifest),
            "--observation-root",
            str(source_donor_root),
            "--output",
            str(donor_mapping),
        ],
        logs,
    )
    _run(
        "prepare_state_selection",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.state_selection",
            "--split",
            str(split),
            "--output",
            str(state_selection),
        ],
        logs,
    )
    _run(
        "differentiable_path_check",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.differentiable_path_check",
            "--preflight",
            str(preflight),
            "--split",
            str(split),
            "--state-selection",
            str(state_selection),
            "--donor-mapping",
            str(donor_mapping),
            "--donor-manifest",
            str(donor_manifest),
            "--donor-root",
            str(source_donor_root),
            "--checkpoint",
            state["checkpoint_path"],
            "--output",
            str(differentiable),
        ],
        logs,
        environment={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
        stable_output=differentiable,
    )
    differentiable_payload = _read(differentiable)
    _require_links(
        differentiable_payload,
        {
            "protocol": SALVAGE_A_PROTOCOL,
            "passed": True,
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "preflight_sha256": sha256_file(preflight),
            "split_sha256": sha256_file(split),
            "state_selection_sha256": sha256_file(state_selection),
            "donor_mapping_sha256": sha256_file(donor_mapping),
            "donor_manifest_sha256": sha256_file(donor_manifest),
            "checkpoint_sha256": state["checkpoint_sha256"],
        },
        label="differentiable-path report",
    )

    fit_common = [
        "--python",
        python,
        "--gpu-ids",
        *(str(value) for value in gpu_ids),
        "--checkpoint",
        state["checkpoint_path"],
        "--split",
        str(split),
        "--state-selection",
        str(state_selection),
        "--donor-mapping",
        str(donor_mapping),
        "--donor-manifest",
        str(donor_manifest),
        "--donor-root",
        str(source_donor_root),
        "--differentiable-path-report",
        str(differentiable),
        "--launch-stagger-seconds",
        str(args.launch_stagger_seconds),
    ]
    _run(
        "fit_basis_shards",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.launch_fit",
            "--phase",
            "fit",
            "--output-dir",
            str(fit_dir),
            *fit_common,
        ],
        logs,
    )
    _run(
        "heldout_diagnostics_shards",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.launch_fit",
            "--phase",
            "heldout",
            "--output-dir",
            str(heldout_dir),
            *fit_common,
        ],
        logs,
    )
    _run(
        "finalize_bases_and_diagnostics",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.finalize_bases",
            "--fit-dir",
            str(fit_dir),
            "--heldout-dir",
            str(heldout_dir),
            "--split",
            str(split),
            "--state-selection",
            str(state_selection),
            "--device",
            "cuda:0",
            "--output",
            str(basis),
            "--diagnostics-output",
            str(diagnostics),
        ],
        logs,
        environment={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
        stable_output=basis,
    )
    basis_payload = _read(basis)
    diagnostics_payload = _read(diagnostics)
    _require_links(
        basis_payload,
        {
            "protocol": SALVAGE_A_PROTOCOL,
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "split_sha256": sha256_file(split),
            "state_selection_sha256": sha256_file(state_selection),
            "differentiable_path_gate_passed": True,
            "differentiable_path_report_sha256": sha256_file(differentiable),
            "donor_mapping_sha256": sha256_file(donor_mapping),
            "donor_observation_manifest_sha256": sha256_file(donor_manifest),
            "checkpoint_sha256": state["checkpoint_sha256"],
            "subspace_diagnostics_sha256": sha256_file(diagnostics),
        },
        label="basis manifest",
    )
    _require_links(
        diagnostics_payload,
        {
            "protocol": SALVAGE_A_PROTOCOL,
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "heldout_used_for_fitting": False,
            "split_sha256": sha256_file(split),
            "state_selection_sha256": sha256_file(state_selection),
            "differentiable_path_report_sha256": sha256_file(differentiable),
        },
        label="subspace diagnostics",
    )

    _run(
        "machinery_tests",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.machinery_tests",
            "--preflight",
            str(preflight),
            "--differentiable-path",
            str(differentiable),
            "--split",
            str(split),
            "--state-selection",
            str(state_selection),
            "--donor-mapping",
            str(donor_mapping),
            "--donor-manifest",
            str(donor_manifest),
            "--donor-root",
            str(source_donor_root),
            "--checkpoint",
            state["checkpoint_path"],
            "--basis-manifest",
            str(basis),
            "--diagnostics",
            str(diagnostics),
            "--output",
            str(machinery),
        ],
        logs,
        environment={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
        stable_output=machinery,
    )
    machinery_payload = _read(machinery)
    _require_links(
        machinery_payload,
        {
            "protocol": SALVAGE_A_PROTOCOL,
            "passed": True,
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "preflight_report_sha256": sha256_file(preflight),
            "differentiable_path_report_sha256": sha256_file(differentiable),
            "split_manifest_sha256": sha256_file(split),
            "state_selection_manifest_sha256": sha256_file(state_selection),
            "donor_mapping_sha256": sha256_file(donor_mapping),
            "donor_observation_manifest_sha256": sha256_file(donor_manifest),
            "basis_manifest_sha256": sha256_file(basis),
            "diagnostics_sha256": sha256_file(diagnostics),
            "checkpoint_sha256": state["checkpoint_sha256"],
        },
        label="machinery report",
    )

    online_common = [
        "--python",
        python,
        "--gpu-ids",
        *(str(value) for value in gpu_ids),
        "--preflight",
        str(preflight),
        "--machinery",
        str(machinery),
        "--split",
        str(split),
        "--state-selection",
        str(state_selection),
        "--differentiable-path-report",
        str(differentiable),
        "--basis-manifest",
        str(basis),
        "--diagnostics",
        str(diagnostics),
        "--donor-mapping",
        str(donor_mapping),
        "--donor-manifest",
        str(donor_manifest),
        "--donor-root",
        str(source_donor_root),
        "--round4c-summary",
        str(round4c),
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
            "experiments.asre_diagnosis.salvage_a.launch_wave",
            "--mode",
            "smoke",
            "--wave",
            "1",
            "--output-root",
            str(smoke1),
            *online_common,
        ],
        logs,
    )
    _run(
        "online_smoke_wave2",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.launch_wave",
            "--mode",
            "smoke",
            "--wave",
            "2",
            "--output-root",
            str(smoke2),
            "--wave1-smoke-summary",
            str(smoke1 / "launcher_summary.json"),
            *online_common,
        ],
        logs,
    )
    _run(
        "online_full_wave1",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.launch_wave",
            "--mode",
            "full",
            "--wave",
            "1",
            "--output-root",
            str(full1),
            "--smoke-summary",
            str(smoke1 / "launcher_summary.json"),
            *online_common,
        ],
        logs,
    )
    _run(
        "online_full_wave2",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.launch_wave",
            "--mode",
            "full",
            "--wave",
            "2",
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

    _run(
        "aggregate",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.aggregate_results",
            "--online-wave1",
            str(full1),
            "--online-wave2",
            str(full2),
            "--preflight",
            str(preflight),
            "--machinery",
            str(machinery),
            "--split",
            str(split),
            "--state-selection",
            str(state_selection),
            "--differentiable-path-report",
            str(differentiable),
            "--basis-manifest",
            str(basis),
            "--diagnostics",
            str(diagnostics),
            "--donor-mapping",
            str(donor_mapping),
            "--donor-manifest",
            str(donor_manifest),
            "--donor-root",
            str(source_donor_root),
            "--round4c-summary",
            str(round4c),
            "--output-dir",
            str(aggregate_dir),
        ],
        logs,
    )
    _run(
        "plots",
        [
            python,
            "-m",
            "experiments.asre_diagnosis.salvage_a.plot_results",
            "--aggregate-dir",
            str(aggregate_dir),
        ],
        logs,
    )

    report = aggregate_dir / "salvage_a_summary.md"
    atomic_write_json(
        status,
        {
            "artifact_type": "asre_salvage_a_driver_status",
            "schema_version": 1,
            "protocol": SALVAGE_A_PROTOCOL,
            "status": "complete",
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "git_commit_hash": git_commit(PROJECT_ROOT),
            "output_root": str(output),
            "python": python,
            "gpu_ids": list(gpu_ids),
            "final_report": str(report),
            "primary_analysis_scope": "heldout_50",
            "calibration_and_all100_descriptive_only": True,
            "later_stage_launched": False,
            "stop_rule_applied": True,
        },
    )
    print(f"Salvage A completed. Final report: {report}")
    print("Stop rule applied; no later-stage experiment was launched.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round4c-summary", type=Path, default=DEFAULT_ROUND4C_SUMMARY)
    parser.add_argument("--valid-manifest", type=Path, default=DEFAULT_VALID_MANIFEST)
    parser.add_argument("--source-donor-root", type=Path, default=DEFAULT_DONOR_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--launch-stagger-seconds", type=float, default=70.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
