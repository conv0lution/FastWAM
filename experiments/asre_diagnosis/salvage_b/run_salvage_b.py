"""Fail-closed end-to-end driver for the final ASRE Salvage-B gate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
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
)
from experiments.asre_diagnosis.salvage_b.definitions import (  # noqa: E402
    SPECIAL_FAILURE_CLASSIFICATIONS,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "asre_results/salvage_b_world_action_dissociation"
)
DEFAULT_ROUND4B_ROOT = PROJECT_ROOT / "asre_results/round4b_subspace"
DEFAULT_ROUND4C_ROOT = (
    PROJECT_ROOT
    / "asre_results/round4c_energy_sufficiency/retry_20260829_donorcheckfix"
)
DEFAULT_SALVAGE_A_SUMMARY = (
    PROJECT_ROOT
    / "asre_results/salvage_a_action_sensitive/aggregate/salvage_a_summary.json"
)
DEFAULT_VALID_MANIFEST = PROJECT_ROOT / "asre_results/round2/state_bank_valid_manifest.json"
DEFAULT_DONOR_ROOT = PROJECT_ROOT / "asre_results/round3b/donors/online"
DEFAULT_DATASET_ROOT = Path(
    "/local_home/zhaizicheng/fastwam_assets/datasets/LIBERO-fastwam/"
    "lerobot_v30/libero_spatial_no_noops_lerobot"
)
FINAL_COMPLETION_FILENAME = "salvage_b_completion.json"
WORLD_BUNDLE_COMPLETION_FILENAME = "world_bundle_complete.json"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _recorded_commit(payload: Mapping[str, Any]) -> str | None:
    for key in ("git_commit_hash", "git_commit"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    for parent, key in (
        ("git", "head"),
        ("git", "current_head"),
        ("identity", "git_commit_hash"),
        ("provenance", "git_commit_hash"),
    ):
        value = payload.get(parent)
        if isinstance(value, Mapping):
            commit = value.get(key)
            if isinstance(commit, str) and commit:
                return commit
    return None


def _assert_clean_worktree(output: Path) -> None:
    relative = output.relative_to(PROJECT_ROOT)
    dirty = subprocess.check_output(
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
    if dirty:
        raise RuntimeError(
            "Salvage-B formal execution requires committed reviewed source code.\n"
            + dirty
        )


def _assert_frozen_source(output: Path, expected_commit: str) -> None:
    _assert_clean_worktree(output)
    observed = git_commit(PROJECT_ROOT)
    if observed != expected_commit:
        raise RuntimeError(
            "Salvage-B source commit changed during execution: "
            f"{observed} != {expected_commit}. Use a fresh output child directory."
        )


def _resolve_python(value: Path) -> str:
    raw = str(value.expanduser())
    candidate = shutil.which(raw) if os.sep not in raw else None
    resolved = Path(candidate if candidate is not None else raw).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Python executable is unavailable: {value}")
    return str(resolved)


def _validate_gpu_ids(values: Sequence[int]) -> tuple[int, ...]:
    gpu_ids = tuple(int(value) for value in values)
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4 or any(value < 0 for value in gpu_ids):
        raise ValueError("SALVAGE_B_GPU_IDS must name four distinct nonnegative GPUs.")
    return gpu_ids


def _environment(output: Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    merged = os.environ.copy()
    libero_root = Path(
        merged.get("LIBERO_ROOT", str(PROJECT_ROOT.parent / "LIBERO"))
    ).resolve()
    entries = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), str(libero_root)]
    entries.extend(
        value
        for value in merged.get("PYTHONPATH", "").split(os.pathsep)
        if value and value not in entries
    )
    runtime = output / "runtime"
    for directory in (
        runtime,
        runtime / "hf_home",
        runtime / "hf_home/datasets",
        runtime / "numba_cache",
        runtime / "matplotlib_cache",
        runtime / "torchinductor/machinery",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    merged.update(
        {
            "PYTHONPATH": os.pathsep.join(entries),
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NUMBA_DISABLE_JIT": "1",
            "NUMBA_CACHE_DIR": str(runtime / "numba_cache"),
            "MPLCONFIGDIR": str(runtime / "matplotlib_cache"),
            "HF_HOME": str(runtime / "hf_home"),
            "HF_DATASETS_CACHE": str(runtime / "hf_home/datasets"),
            "TORCHINDUCTOR_CACHE_DIR": str(runtime / "torchinductor/machinery"),
        }
    )
    merged.setdefault(
        "DIFFSYNTH_MODEL_BASE_PATH",
        "/local_home/zhaizicheng/fastwam_assets/checkpoints/wan_base",
    )
    # Salvage B is explicitly four independent single-GPU workers, never DDP.
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        merged.pop(key, None)
    if extra:
        merged.update({str(key): str(value) for key, value in extra.items()})
    return merged


def _next_log(logs: Path, name: str) -> Path:
    logs.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f".attempt{attempt:02d}"
        path = logs / f"{name}{suffix}.log"
        if not path.exists():
            return path
        attempt += 1


def _run_stage(
    name: str,
    command: Sequence[str],
    logs: Path,
    *,
    environment: Mapping[str, str],
    allow_failure: bool = False,
) -> int:
    log = _next_log(logs, name)
    print(f"[SalvageB] Starting {name}; log: {log}", flush=True)
    with log.open("x", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        result = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        message = f"Salvage-B stage {name} exited {result.returncode}; inspect {log}."
        if not allow_failure:
            raise RuntimeError(message)
        print(f"[SalvageB] {message}", flush=True)
    else:
        print(f"[SalvageB] Completed {name}", flush=True)
    return int(result.returncode)


def _require_json(
    path: Path,
    *,
    label: str,
    artifact_type: str,
    statuses: Sequence[str] | None = None,
    current_commit: str | None = None,
) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("artifact_type") != artifact_type:
        raise RuntimeError(f"Refusing incompatible {label}: wrong artifact type.")
    if statuses is not None and payload.get("status") not in set(statuses):
        raise RuntimeError(
            f"Refusing incompatible {label}: status={payload.get('status')!r}."
        )
    if current_commit is not None and _recorded_commit(payload) != current_commit:
        raise RuntimeError(
            f"Refusing {label} from source commit {_recorded_commit(payload)}; "
            f"current HEAD is {current_commit}. Use a fresh output child directory."
        )
    return payload


def _run_or_reuse_json(
    name: str,
    command: Sequence[str],
    logs: Path,
    output: Path,
    *,
    environment: Mapping[str, str],
    artifact_type: str,
    statuses: Sequence[str],
    current_commit: str,
    companion: Path | None = None,
    allow_failure: bool = False,
) -> tuple[dict[str, Any], int]:
    if output.is_file():
        if companion is not None and not companion.is_file():
            raise RuntimeError(f"Incomplete {name} resume: missing {companion}.")
        payload = _require_json(
            output,
            label=name,
            artifact_type=artifact_type,
            statuses=statuses,
            current_commit=current_commit,
        )
        print(f"[SalvageB] Reusing {name}: {output}", flush=True)
        return payload, 0
    if companion is not None and companion.exists():
        # The companion is published before the JSON completion sentinel. An
        # orphan companion therefore records an interrupted, safely
        # replaceable attempt rather than an incompatible completed stage.
        print(
            f"[SalvageB] Recovering incomplete {name}; completion sentinel "
            f"is absent: {output}",
            flush=True,
        )
    code = _run_stage(
        name,
        command,
        logs,
        environment=environment,
        allow_failure=allow_failure,
    )
    if not output.is_file():
        raise RuntimeError(f"Salvage-B stage {name} did not emit {output}.")
    payload = _require_json(
        output,
        label=name,
        artifact_type=artifact_type,
        statuses=statuses,
        current_commit=current_commit,
    )
    return payload, code


def _validate_frozen_bundle(
    *,
    world_manifest: Path,
    stochastic_manifest: Path,
    draw_tensors: Path,
    bundle_completion: Path,
    preflight: Path,
    current_commit: str,
) -> None:
    completion = _require_json(
        bundle_completion,
        label="world bundle completion sentinel",
        artifact_type="asre_salvage_b_world_bundle_completion",
        statuses=("complete",),
        current_commit=current_commit,
    )
    if completion.get("schema_version") != 2:
        raise RuntimeError(
            "Refusing incompatible world bundle completion sentinel: "
            f"schema_version={completion.get('schema_version')!r}."
        )
    expected_artifacts = {
        path.name: path.resolve()
        for path in (world_manifest, stochastic_manifest, draw_tensors)
    }
    raw_artifacts = completion.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != set(
        expected_artifacts
    ):
        raise RuntimeError(
            "World bundle completion sentinel does not bind exactly the registered "
            "three-artifact bundle."
        )
    marker_failures: dict[str, Any] = {}
    for name, expected_path in expected_artifacts.items():
        record = raw_artifacts.get(name)
        if not isinstance(record, Mapping):
            marker_failures[name] = "malformed artifact record"
            continue
        recorded_path = Path(str(record.get("path", ""))).resolve()
        if recorded_path != expected_path:
            marker_failures[f"{name}.path"] = (str(recorded_path), str(expected_path))
            continue
        if not expected_path.is_file():
            marker_failures[f"{name}.exists"] = False
            continue
        observed_sha = sha256_file(expected_path)
        if record.get("sha256") != observed_sha:
            marker_failures[f"{name}.sha256"] = (record.get("sha256"), observed_sha)
    expected_preflight_sha = sha256_file(preflight)
    if completion.get("preflight_report_sha256") != expected_preflight_sha:
        marker_failures["preflight_report_sha256"] = (
            completion.get("preflight_report_sha256"),
            expected_preflight_sha,
        )
    if marker_failures:
        raise RuntimeError(
            "Refusing incomplete or drifted frozen world bundle marker: "
            f"{marker_failures}"
        )

    world = _require_json(
        world_manifest,
        label="world evaluation manifest",
        artifact_type="asre_salvage_b_world_evaluation_manifest",
        statuses=("frozen_before_metrics",),
        current_commit=current_commit,
    )
    stochastic = _require_json(
        stochastic_manifest,
        label="stochastic manifest",
        artifact_type="asre_salvage_b_stochastic_manifest",
        statuses=("frozen_before_metrics",),
        current_commit=current_commit,
    )
    mismatch = {
        "world.preflight_report_sha256": (
            world.get("preflight_report_sha256"),
            sha256_file(preflight),
        ),
        "world.draw_tensor_sha256": (
            world.get("draw_tensor_sha256"),
            sha256_file(draw_tensors),
        ),
        "stochastic.preflight_report_sha256": (
            stochastic.get("preflight_report_sha256"),
            sha256_file(preflight),
        ),
        "stochastic.world_manifest_sha256": (
            stochastic.get("world_manifest_sha256"),
            sha256_file(world_manifest),
        ),
        "stochastic.draw_tensor_sha256": (
            stochastic.get("draw_tensor_sha256"),
            sha256_file(draw_tensors),
        ),
    }
    failed = {key: value for key, value in mismatch.items() if value[0] != value[1]}
    if failed:
        raise RuntimeError(f"Refusing incompatible frozen world bundle: {failed}")


def _validate_target_manifest(
    target_manifest: Path,
    *,
    preflight: Path,
    world_manifest: Path,
    current_commit: str,
) -> dict[str, Any]:
    targets = _require_json(
        target_manifest,
        label="processed target manifest",
        artifact_type="asre_salvage_b_processed_world_target_manifest",
        statuses=("frozen_before_gpu_metrics",),
        current_commit=current_commit,
    )
    expected = {
        "schema_version": 2,
        "protocol": SALVAGE_B_PROTOCOL,
        "preflight_report_sha256": sha256_file(preflight),
        "world_manifest_sha256": sha256_file(world_manifest),
        "sample_count": 100,
    }
    mismatch = {
        key: (targets.get(key), value)
        for key, value in expected.items()
        if targets.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Refusing incompatible processed targets: {mismatch}")
    world = _read(world_manifest)
    world_records = world.get("records")
    records = targets.get("records")
    if not (
        isinstance(world_records, list)
        and len(world_records) == 100
        and isinstance(records, list)
        and len(records) == 100
    ):
        raise RuntimeError("Processed-target manifest lacks exactly 100 frozen records.")
    identity_fields = ("sample_id", "task_id", "episode_id", "trial", "dataset_index")
    for position, (source, target) in enumerate(zip(world_records, records, strict=True)):
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise RuntimeError(f"Malformed processed-target record at position {position}.")
        identity_mismatch = {
            key: (target.get(key), source.get(key))
            for key in identity_fields
            if target.get(key) != source.get(key)
        }
        current_hash = target.get("current_image_sha256")
        if identity_mismatch or not (
            isinstance(current_hash, str)
            and len(current_hash) == 64
            and all(character in "0123456789abcdef" for character in current_hash.lower())
        ):
            raise RuntimeError(
                "Processed-target identity/current-image hash drifted at position "
                f"{position}: {identity_mismatch}."
            )
    verification = targets.get("source_artifact_verification")
    expected_verification = {
        "before_decode": True,
        "after_decode": True,
        "metadata_file_count": len(world.get("metadata_files", [])),
        "target_source_file_count": len(world.get("target_source_files", [])),
        "checks": "resolved regular file, byte size, and SHA-256",
    }
    if not isinstance(verification, Mapping) or any(
        verification.get(key) != value for key, value in expected_verification.items()
    ):
        raise RuntimeError(
            "Processed targets lack complete before/after source-artifact verification."
        )
    return targets


def _validate_preflight_inputs(
    payload: Mapping[str, Any],
    *,
    output: Path,
    architecture: Path,
    round4b: Path,
    round4c: Path,
    salvage_a: Path,
    valid: Path,
    donor_root: Path,
    donor_mapping: Path,
    donor_manifest: Path,
    dataset: Path,
) -> None:
    expected = {
        "output_root": (payload.get("output_root"), str(output)),
        "phase_a.architecture_audit_path": (
            payload.get("phase_a", {}).get("architecture_audit_path"),
            str(architecture),
        ),
        "phase_a.architecture_audit_sha256": (
            payload.get("phase_a", {}).get("architecture_audit_sha256"),
            sha256_file(architecture),
        ),
        "basis.path": (
            payload.get("basis", {}).get("path"),
            str((round4b / "calibration/basis_manifest.json").resolve()),
        ),
        "basis.split_path": (
            payload.get("basis", {}).get("split_path"),
            str((round4b / "calibration/calibration_split_manifest.json").resolve()),
        ),
        "frozen_action.round4c_root": (
            payload.get("frozen_action", {}).get("round4c_root"),
            str(round4c),
        ),
        "salvage_a.summary_path": (
            payload.get("salvage_a", {}).get("summary_path"),
            str(salvage_a),
        ),
        "state.valid_manifest_path": (
            payload.get("state", {}).get("valid_manifest_path"),
            str(valid),
        ),
        "donors.mapping_path": (
            payload.get("donors", {}).get("mapping_path"),
            str(donor_mapping),
        ),
        "donors.manifest_path": (
            payload.get("donors", {}).get("manifest_path"),
            str(donor_manifest),
        ),
        "donors.root": (
            payload.get("donors", {}).get("root"),
            str(donor_root),
        ),
        "world_data.dataset_root": (
            payload.get("world_data", {}).get("dataset_root"),
            str(dataset),
        ),
    }
    mismatch = {key: values for key, values in expected.items() if values[0] != values[1]}
    if mismatch:
        raise RuntimeError(
            "Refusing preflight reuse with changed input paths: "
            f"{mismatch}. Use a fresh output child directory."
        )
    hashed_inputs = {
        "state.valid_manifest_sha256": (
            payload.get("state", {}).get("valid_manifest_sha256"),
            sha256_file(valid),
        ),
        "salvage_a.summary_sha256": (
            payload.get("salvage_a", {}).get("summary_sha256"),
            sha256_file(salvage_a),
        ),
        "donors.mapping_sha256": (
            payload.get("donors", {}).get("mapping_sha256"),
            sha256_file(donor_mapping),
        ),
        "donors.manifest_sha256": (
            payload.get("donors", {}).get("manifest_sha256"),
            sha256_file(donor_manifest),
        ),
    }
    drifted = {
        key: values for key, values in hashed_inputs.items() if values[0] != values[1]
    }
    if drifted:
        raise RuntimeError(f"Refusing preflight reuse after frozen input drift: {drifted}")


def _validate_phase_summary(
    payload: Mapping[str, Any],
    *,
    phase: str,
    world_manifest: Path,
    stochastic_manifest: Path,
    target_manifest: Path,
    machinery: Path,
    preflight: Path,
) -> None:
    expected = {
        "schema_version": 2,
        "phase": phase,
        "sample_count": 100,
        "draws_per_sample": 4,
        "worker_count": 4,
        "world_manifest_sha256": sha256_file(world_manifest),
        "stochastic_manifest_sha256": sha256_file(stochastic_manifest),
        "target_manifest_sha256": sha256_file(target_manifest),
        "machinery_sha256": sha256_file(machinery),
        "preflight_report_sha256": sha256_file(preflight),
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Refusing stale {phase} aggregate: {mismatch}")
    sample_identity = payload.get("sample_identity_sha256")
    if not (
        isinstance(sample_identity, str)
        and len(sample_identity) == 64
        and all(character in "0123456789abcdef" for character in sample_identity.lower())
    ):
        raise RuntimeError(f"Refusing {phase} aggregate without sample identity binding.")
    shards = payload.get("shards")
    if not isinstance(shards, list) or len(shards) != 4:
        raise RuntimeError(f"Refusing incomplete {phase} aggregate shard inventory.")
    for expected_worker, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or shard.get("worker_index") != expected_worker:
            raise RuntimeError(f"Invalid {phase} aggregate shard ordering.")
        for prefix in ("metadata", "rows"):
            path = Path(str(shard.get(f"{prefix}_path", ""))).resolve()
            expected_sha = shard.get(f"{prefix}_sha256")
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise RuntimeError(f"Frozen {phase} worker {expected_worker} {prefix} drifted.")


def _validate_final_summary_links(
    payload: Mapping[str, Any],
    *,
    architecture: Path,
    preflight: Path,
    machinery: Path,
    world_manifest: Path,
    stochastic_manifest: Path,
    target_manifest: Path,
    endpoint_summary: Path,
    projected_summary: Path,
) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("Final Salvage-B report lacks provenance.")
    expected = {
        "architecture_audit_sha256": sha256_file(architecture),
        "preflight_sha256": sha256_file(preflight),
        "machinery_sha256": sha256_file(machinery),
        "world_manifest_sha256": sha256_file(world_manifest),
        "stochastic_manifest_sha256": sha256_file(stochastic_manifest),
        "target_manifest_sha256": sha256_file(target_manifest),
        "endpoint_summary_sha256": sha256_file(endpoint_summary),
        "projected_summary_sha256": sha256_file(projected_summary),
    }
    mismatch = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"Refusing stale final Salvage-B aggregate: {mismatch}")
    figure_manifest = payload.get("figures")
    figures = (
        figure_manifest.get("figures")
        if isinstance(figure_manifest, Mapping)
        else None
    )
    if not isinstance(figures, list) or len(figures) != 4:
        raise RuntimeError("Final Salvage-B report lacks the four registered figures.")
    for figure in figures:
        if not isinstance(figure, Mapping):
            raise RuntimeError("Final Salvage-B figure inventory is malformed.")
        path = Path(str(figure.get("path", ""))).resolve()
        if not path.is_file() or sha256_file(path) != figure.get("sha256"):
            raise RuntimeError(f"Final Salvage-B figure drifted: {path}")


def _validate_final_bundle(
    *,
    summary_path: Path,
    completion_path: Path,
    current_commit: str,
    expected_status: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the last-published completion marker and every file it binds."""

    if expected_status not in {"complete", "stopped"}:
        raise ValueError(f"Invalid expected final status: {expected_status!r}.")
    if not summary_path.is_file() or not completion_path.is_file():
        raise RuntimeError(
            "Salvage-B final bundle is partial: both summary and completion sentinel "
            "must exist before it can be reused."
        )
    summary = _require_json(
        summary_path,
        label="final aggregate",
        artifact_type="asre_salvage_b_final_aggregate",
        statuses=(expected_status,),
        current_commit=current_commit,
    )
    if summary.get("protocol") != SALVAGE_B_PROTOCOL:
        raise RuntimeError("Final aggregate has the wrong Salvage-B protocol.")
    completion = _require_json(
        completion_path,
        label="final completion sentinel",
        artifact_type="asre_salvage_b_final_completion",
        statuses=(expected_status,),
        current_commit=current_commit,
    )
    classification = summary.get("classification")
    classification_name = (
        classification.get("classification")
        if isinstance(classification, Mapping)
        else None
    )
    expected = {
        "schema_version": 1,
        "protocol": SALVAGE_B_PROTOCOL,
        "publication_complete": True,
        "classification": classification_name,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "later_stage_launched": False,
        "salvage_c_exists": False,
    }
    mismatch = {
        key: (completion.get(key), value)
        for key, value in expected.items()
        if completion.get(key) != value
    }
    if mismatch or not isinstance(classification_name, str):
        raise RuntimeError(f"Final completion sentinel disagrees with summary: {mismatch}")

    aggregate_dir = summary_path.resolve().parent
    expected_reports = {
        "salvage_b_summary_md": aggregate_dir / "salvage_b_summary.md",
        "result_summary_for_gpt": aggregate_dir / "result_summary_for_gpt.md",
    }
    raw_reports = completion.get("reports")
    if not isinstance(raw_reports, list) or len(raw_reports) != len(expected_reports):
        raise RuntimeError("Final completion sentinel lacks both Markdown reports.")
    reports: dict[str, Mapping[str, Any]] = {}
    for raw in raw_reports:
        if not isinstance(raw, Mapping):
            raise RuntimeError("Malformed final Markdown artifact record.")
        name = str(raw.get("name", ""))
        if name in reports:
            raise RuntimeError(f"Duplicate final Markdown artifact: {name!r}.")
        reports[name] = raw
    if set(reports) != set(expected_reports):
        raise RuntimeError("Final Markdown artifact names are incomplete or unexpected.")
    for name, expected_path in expected_reports.items():
        record = reports[name]
        observed_path = Path(str(record.get("path", ""))).resolve()
        if (
            observed_path != expected_path
            or not observed_path.is_file()
            or sha256_file(observed_path) != record.get("sha256")
        ):
            raise RuntimeError(f"Final Markdown artifact drifted: {name}.")

    if expected_status == "complete":
        figure_record = completion.get("figure_manifest")
        expected_figure_manifest = aggregate_dir / "plots/figure_manifest.json"
        if not isinstance(figure_record, Mapping):
            raise RuntimeError("Completed final bundle lacks its figure manifest record.")
        observed_manifest = Path(str(figure_record.get("path", ""))).resolve()
        if (
            observed_manifest != expected_figure_manifest
            or not observed_manifest.is_file()
            or sha256_file(observed_manifest) != figure_record.get("sha256")
        ):
            raise RuntimeError("Final figure manifest is absent or drifted.")
        persisted_manifest = _read(observed_manifest)
        if persisted_manifest != summary.get("figures"):
            raise RuntimeError("Final summary and figure manifest disagree.")
        figures = persisted_manifest.get("figures")
        if not isinstance(figures, list) or len(figures) != 4:
            raise RuntimeError("Completed final bundle lacks exactly four figures.")
        if completion.get("figures") != figures:
            raise RuntimeError("Completion sentinel and figure manifest disagree.")
        expected_names = {"figure_A", "figure_B", "figure_C", "figure_D"}
        observed_names: set[str] = set()
        for figure in figures:
            if not isinstance(figure, Mapping):
                raise RuntimeError("Malformed completed figure record.")
            name = str(figure.get("name", ""))
            path = Path(str(figure.get("path", ""))).resolve()
            if name in observed_names or not path.is_file() or sha256_file(path) != figure.get(
                "sha256"
            ):
                raise RuntimeError(f"Completed final figure drifted: {name!r}.")
            observed_names.add(name)
        if observed_names != expected_names:
            raise RuntimeError("Completed final figure names are incomplete or unexpected.")
    elif (
        completion.get("figure_manifest") is not None
        or completion.get("figures") != []
        or summary.get("figures") != []
    ):
        raise RuntimeError("Technical-stop bundle must not contain result figures.")
    return summary, completion


def _finalize_special(
    *,
    classification: str,
    reason: str,
    python: str,
    logs: Path,
    environment: Mapping[str, str],
    aggregate_dir: Path,
    optional_artifacts: Mapping[str, Path],
    current_commit: str,
) -> dict[str, Any]:
    if classification not in SPECIAL_FAILURE_CLASSIFICATIONS:
        raise ValueError(f"Unknown Salvage-B technical stop: {classification}")
    summary_path = aggregate_dir / "salvage_b_summary.json"
    completion_path = aggregate_dir / FINAL_COMPLETION_FILENAME
    if completion_path.is_file():
        summary, _completion = _validate_final_bundle(
            summary_path=summary_path,
            completion_path=completion_path,
            current_commit=current_commit,
            expected_status="stopped",
        )
        if summary.get("classification", {}).get("classification") != classification:
            raise RuntimeError("Existing final report has a different classification.")
        provenance = summary.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError("Existing special final report lacks provenance.")
        for label, path in optional_artifacts.items():
            if not path.is_file():
                continue
            normalized = label.replace("-", "_")
            if provenance.get(f"{normalized}_sha256") != sha256_file(path):
                raise RuntimeError(
                    f"Existing special final report has stale {normalized} provenance."
                )
            artifact = _read(path)
            recorded = _recorded_commit(artifact)
            if recorded is not None and recorded != current_commit:
                raise RuntimeError(
                    f"Existing {normalized} artifact came from source commit {recorded}."
                )
        print(f"[SalvageB] Reusing special final report: {summary_path}", flush=True)
        return summary
    if summary_path.exists():
        print(
            "[SalvageB] Recovering incomplete special final bundle; no completion "
            f"sentinel was published: {aggregate_dir}",
            flush=True,
        )
    command = [
        python,
        "-m",
        "experiments.asre_diagnosis.salvage_b.aggregate_final",
        "--output-dir",
        str(aggregate_dir),
        "--special-classification",
        classification,
        "--special-reason",
        reason,
        "--git-commit-hash",
        current_commit,
    ]
    for argument, path in optional_artifacts.items():
        if path.is_file():
            command.extend([f"--{argument}", str(path)])
    _run_stage(
        "aggregate_special",
        command,
        logs,
        environment=environment,
    )
    summary, _completion = _validate_final_bundle(
        summary_path=summary_path,
        completion_path=completion_path,
        current_commit=current_commit,
        expected_status="stopped",
    )
    provenance = summary.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("Generated special final report lacks provenance.")
    for label, path in optional_artifacts.items():
        if path.is_file():
            normalized = label.replace("-", "_")
            if provenance.get(f"{normalized}_sha256") != sha256_file(path):
                raise RuntimeError(
                    f"Generated special final report has stale {normalized} provenance."
                )
    return summary


def _write_terminal_status(
    status_path: Path,
    *,
    start: str,
    output: Path,
    python: str,
    gpu_ids: Sequence[int],
    status: str,
    classification: str,
    summary_path: Path,
    current_commit: str,
) -> None:
    if status not in {"complete", "stopped"}:
        raise ValueError(f"Invalid terminal driver status: {status!r}.")
    completion_path = summary_path.with_name(FINAL_COMPLETION_FILENAME)
    expected_summary_status = "complete" if status == "complete" else "stopped"
    summary, _completion = _validate_final_bundle(
        summary_path=summary_path,
        completion_path=completion_path,
        current_commit=current_commit,
        expected_status=expected_summary_status,
    )
    summary_classification = summary.get("classification", {}).get("classification")
    if classification != summary_classification:
        raise RuntimeError(
            "Driver terminal classification disagrees with the published final summary."
        )
    report_path = summary_path.with_name("result_summary_for_gpt.md")
    atomic_write_json(
        status_path,
        {
            "artifact_type": "asre_salvage_b_driver_status",
            "schema_version": 1,
            "protocol": SALVAGE_B_PROTOCOL,
            "status": status,
            "start_timestamp": start,
            "end_timestamp": now_iso(),
            "git_commit_hash": current_commit,
            "output_root": str(output),
            "python": python,
            "gpu_ids": list(gpu_ids),
            "classification": classification,
            "final_report": str(report_path),
            "final_report_sha256": sha256_file(report_path),
            "final_summary_path": str(summary_path),
            "final_summary_sha256": sha256_file(summary_path),
            "final_completion_path": str(completion_path),
            "final_completion_sha256": sha256_file(completion_path),
            "action_rerun": False,
            "online_episodes": 0,
            "environment_rollouts": 0,
            "later_stage_launched": False,
            "salvage_c_exists": False,
        },
    )


def _completed_resume(
    status_path: Path, *, current_commit: str, output: Path
) -> dict[str, Any] | None:
    if not status_path.is_file():
        return None
    status = _read(status_path)
    if status.get("artifact_type") != "asre_salvage_b_driver_status":
        raise RuntimeError(f"Refusing incompatible driver status: {status_path}")
    if status.get("protocol") != SALVAGE_B_PROTOCOL:
        raise RuntimeError("Existing Salvage-B driver status has the wrong protocol.")
    if status.get("git_commit_hash") != current_commit:
        raise RuntimeError(
            "Existing Salvage-B output was created from another source commit; "
            "use a fresh child output directory."
        )
    if status.get("status") not in {"complete", "stopped"}:
        return None
    summary_path = Path(str(status.get("final_summary_path", ""))).resolve()
    allowed_summary = (output / "aggregate/salvage_b_summary.json").resolve()
    if summary_path != allowed_summary or not summary_path.is_file():
        raise RuntimeError("Completed Salvage-B status references a missing final summary.")
    if sha256_file(summary_path) != status.get("final_summary_sha256"):
        raise RuntimeError("Completed Salvage-B final summary drifted after execution.")
    completion_path = Path(str(status.get("final_completion_path", ""))).resolve()
    expected_completion = (
        output / f"aggregate/{FINAL_COMPLETION_FILENAME}"
    ).resolve()
    if (
        completion_path != expected_completion
        or not completion_path.is_file()
        or sha256_file(completion_path) != status.get("final_completion_sha256")
    ):
        raise RuntimeError("Completed Salvage-B completion sentinel drifted after execution.")
    report_path = Path(str(status.get("final_report", ""))).resolve()
    expected_report = (output / "aggregate/result_summary_for_gpt.md").resolve()
    if (
        report_path != expected_report
        or not report_path.is_file()
        or sha256_file(report_path) != status.get("final_report_sha256")
    ):
        raise RuntimeError("Completed Salvage-B GPT report drifted after execution.")
    summary_status = "complete" if status.get("status") == "complete" else "stopped"
    summary, _completion = _validate_final_bundle(
        summary_path=summary_path,
        completion_path=completion_path,
        current_commit=current_commit,
        expected_status=summary_status,
    )
    if summary.get("classification", {}).get("classification") != status.get(
        "classification"
    ):
        raise RuntimeError("Completed driver status/final classification disagree.")
    if any(
        status.get(key) != expected
        for key, expected in (
            ("action_rerun", False),
            ("online_episodes", 0),
            ("environment_rollouts", 0),
            ("later_stage_launched", False),
            ("salvage_c_exists", False),
        )
    ):
        raise RuntimeError("Completed driver status violates the Salvage-B stop contract.")
    return status


def run(args: argparse.Namespace) -> None:
    output = args.output_root.expanduser().resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    if output != allowed and not output.is_relative_to(allowed):
        raise ValueError(f"Output must be within {allowed}, got {output}.")
    output.mkdir(parents=True, exist_ok=True)
    _assert_clean_worktree(output)
    current_commit = git_commit(PROJECT_ROOT)
    gpu_ids = _validate_gpu_ids(args.gpu_ids)
    python = _resolve_python(args.python)
    logs = output / "logs"
    status_path = output / "driver_status.json"
    previous = _completed_resume(status_path, current_commit=current_commit, output=output)
    if previous is not None:
        print(
            "Salvage B already reached its terminal classification: "
            f"{previous['classification']}",
            flush=True,
        )
        print(f"Final report: {previous['final_report']}", flush=True)
        print("No later ASRE experiment was launched.", flush=True)
        return

    start = now_iso()
    environment = _environment(output)
    round4b = args.round4b_root.expanduser().resolve()
    round4c = args.round4c_root.expanduser().resolve()
    salvage_a = args.salvage_a_summary.expanduser().resolve()
    valid = args.valid_manifest.expanduser().resolve()
    donor_root = args.donor_root.expanduser().resolve()
    donor_mapping = donor_root / "donor_mapping.json"
    donor_manifest = donor_root / "donor_observation_manifest.json"
    dataset = args.dataset_root.expanduser().resolve()

    phase_a_dir = output / "phase_a"
    architecture_json = phase_a_dir / "architecture_audit.json"
    architecture_md = phase_a_dir / "architecture_audit.md"
    preflight = output / "preflight_report.json"
    manifests = output / "manifests"
    world_manifest = manifests / "world_evaluation_manifest.json"
    stochastic_manifest = manifests / "stochastic_manifest.json"
    draw_tensors = manifests / "fixed_draw_tensors.pt"
    world_bundle_completion = manifests / WORLD_BUNDLE_COMPLETION_FILENAME
    target_manifest = manifests / "processed_world_targets.json"
    machinery = output / "machinery_report.json"
    machinery_md = output / "machinery_report.md"
    world_eval = output / "world_eval"
    aggregate = output / "aggregate"
    endpoint_dir = aggregate / "endpoint"
    projected_dir = aggregate / "projected"
    endpoint_summary = endpoint_dir / "endpoint_world_summary.json"
    endpoint_gate = endpoint_dir / "world_endpoint_gate.json"
    projected_summary = projected_dir / "projected_world_summary.json"
    final_summary = aggregate / "salvage_b_summary.json"
    final_completion = aggregate / FINAL_COMPLETION_FILENAME

    atomic_write_json(
        status_path,
        {
            "artifact_type": "asre_salvage_b_driver_status",
            "schema_version": 1,
            "protocol": SALVAGE_B_PROTOCOL,
            "status": "running",
            "start_timestamp": start,
            "git_commit_hash": current_commit,
            "output_root": str(output),
            "python": python,
            "gpu_ids": list(gpu_ids),
            "world_worker_gpu_mapping": {
                str(index): int(gpu_id) for index, gpu_id in enumerate(gpu_ids)
            },
            "action_rerun": False,
            "online_episodes": 0,
            "environment_rollouts": 0,
            "later_stage_launched": False,
            "salvage_c_exists": False,
        },
    )

    try:
        architecture, audit_code = _run_or_reuse_json(
            "architecture_audit",
            [
                python,
                "-m",
                "experiments.asre_diagnosis.salvage_b.architecture_audit",
                "--output-json",
                str(architecture_json),
                "--output-md",
                str(architecture_md),
            ],
            logs,
            architecture_json,
            environment=environment,
            artifact_type="asre_salvage_b_phase_a_architecture_audit",
            statuses=(
                "preferred_path_a_pending_runtime_machinery",
                "shared_interface_not_available",
            ),
            current_commit=current_commit,
            companion=architecture_md,
            allow_failure=True,
        )
        if architecture.get("static_audit_passed") is not True:
            reason = (
                "Phase-A static audit could not establish an eligible shared causal "
                "interface for both action and native future-video prediction."
            )
            special = _finalize_special(
                classification="SHARED-INTERFACE-NOT-AVAILABLE",
                reason=reason,
                python=python,
                logs=logs,
                environment=environment,
                aggregate_dir=aggregate,
                optional_artifacts={"architecture-audit": architecture_json},
                current_commit=current_commit,
            )
            classification = special["classification"]["classification"]
            _write_terminal_status(
                status_path,
                start=start,
                output=output,
                python=python,
                gpu_ids=gpu_ids,
                status="stopped",
                classification=classification,
                summary_path=final_summary,
                current_commit=current_commit,
            )
            print(f"Salvage B stopped: {classification}", flush=True)
            print(f"Final report: {aggregate / 'result_summary_for_gpt.md'}", flush=True)
            return
        if audit_code != 0:
            raise RuntimeError("Architecture audit exited nonzero despite reporting a pass.")

        _assert_frozen_source(output, current_commit)
        preflight_payload, _ = _run_or_reuse_json(
            "preflight",
            [
                python,
                "-m",
                "experiments.asre_diagnosis.salvage_b.preflight",
                "--output-root",
                str(output),
                "--architecture-audit",
                str(architecture_json),
                "--round4b-root",
                str(round4b),
                "--round4c-root",
                str(round4c),
                "--salvage-a-summary",
                str(salvage_a),
                "--valid-manifest",
                str(valid),
                "--donor-mapping",
                str(donor_mapping),
                "--donor-manifest",
                str(donor_manifest),
                "--donor-root",
                str(donor_root),
                "--dataset-root",
                str(dataset),
                "--output",
                str(preflight),
            ],
            logs,
            preflight,
            environment=environment,
            artifact_type="asre_salvage_b_preflight_report",
            statuses=("compatible",),
            current_commit=current_commit,
        )
        _validate_preflight_inputs(
            preflight_payload,
            output=output,
            architecture=architecture_json.resolve(),
            round4b=round4b,
            round4c=round4c,
            salvage_a=salvage_a,
            valid=valid,
            donor_root=donor_root,
            donor_mapping=donor_mapping,
            donor_manifest=donor_manifest,
            dataset=dataset,
        )
        _assert_frozen_source(output, current_commit)

        _run_stage(
            "unit_tests",
            [python, "-m", "pytest", "-q", "experiments/asre_diagnosis/tests"],
            logs,
            environment=environment,
        )
        _assert_frozen_source(output, current_commit)

        manifest_paths = (world_manifest, stochastic_manifest, draw_tensors)
        if not world_bundle_completion.is_file():
            existing_manifest_paths = [path for path in manifest_paths if path.exists()]
            if existing_manifest_paths:
                print(
                    "[SalvageB] Recovering incomplete frozen world bundle; no "
                    f"completion sentinel was published: {manifests}",
                    flush=True,
                )
            _run_stage(
                "freeze_world_manifest",
                [
                    python,
                    "-m",
                    "experiments.asre_diagnosis.salvage_b.world_manifest",
                    "--dataset-root",
                    str(dataset),
                    "--preflight",
                    str(preflight),
                    "--donor-mapping",
                    str(donor_mapping),
                    "--donor-manifest",
                    str(donor_manifest),
                    "--world-manifest",
                    str(world_manifest),
                    "--stochastic-manifest",
                    str(stochastic_manifest),
                    "--draw-tensors",
                    str(draw_tensors),
                ],
                logs,
                environment=environment,
            )
        else:
            print(f"[SalvageB] Reusing frozen world bundle: {manifests}", flush=True)
        _validate_frozen_bundle(
            world_manifest=world_manifest,
            stochastic_manifest=stochastic_manifest,
            draw_tensors=draw_tensors,
            bundle_completion=world_bundle_completion,
            preflight=preflight,
            current_commit=current_commit,
        )
        _assert_frozen_source(output, current_commit)

        if not target_manifest.is_file():
            _run_stage(
                "freeze_world_targets",
                [
                    python,
                    "-m",
                    "experiments.asre_diagnosis.salvage_b.freeze_world_targets",
                    "--preflight",
                    str(preflight),
                    "--world-manifest",
                    str(world_manifest),
                    "--runtime-work-dir",
                    str(output / "runtime/target_freeze"),
                    "--output",
                    str(target_manifest),
                ],
                logs,
                environment=environment,
            )
        else:
            print(f"[SalvageB] Reusing frozen processed targets: {target_manifest}", flush=True)
        _validate_target_manifest(
            target_manifest,
            preflight=preflight,
            world_manifest=world_manifest,
            current_commit=current_commit,
        )
        _assert_frozen_source(output, current_commit)

        machinery_payload, machinery_code = _run_or_reuse_json(
            "machinery_tests",
            [
                python,
                "-m",
                "experiments.asre_diagnosis.salvage_b.machinery_tests",
                "--preflight",
                str(preflight),
                "--world-manifest",
                str(world_manifest),
                "--stochastic-manifest",
                str(stochastic_manifest),
                "--draw-tensors",
                str(draw_tensors),
                "--processed-targets",
                str(target_manifest),
                "--runtime-work-dir",
                str(output / "runtime/machinery"),
                "--output",
                str(machinery),
                "--output-md",
                str(machinery_md),
            ],
            logs,
            machinery,
            environment=_environment(
                output, {"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])}
            ),
            artifact_type="asre_salvage_b_shared_interface_machinery_report",
            statuses=("passed", "failed"),
            current_commit=current_commit,
            companion=machinery_md,
            allow_failure=True,
        )
        machinery_links = {
            "preflight_report_sha256": sha256_file(preflight),
            "world_manifest_sha256": sha256_file(world_manifest),
            "stochastic_manifest_sha256": sha256_file(stochastic_manifest),
            "draw_tensors_sha256": sha256_file(draw_tensors),
            "processed_targets_sha256": sha256_file(target_manifest),
        }
        stale_machinery = {
            key: (machinery_payload.get(key), value)
            for key, value in machinery_links.items()
            if machinery_payload.get(key) != value
        }
        if stale_machinery:
            raise RuntimeError(f"Refusing stale machinery report: {stale_machinery}")
        if machinery_payload.get("passed") is not True:
            classification = machinery_payload.get("recommended_special_classification")
            if classification not in {
                "SHARED-INTERFACE-NOT-AVAILABLE",
                "WORLD-METRIC-NOT-VALIDATABLE",
            }:
                raise RuntimeError(
                    "Salvage-B machinery failed for an operational reason; inspect "
                    f"{machinery}. Use a fresh output child directory after fixing it."
                )
            failure = machinery_payload.get("failure", {})
            reason = (
                f"Runtime machinery gate `{failure.get('active_gate', 'unknown')}` "
                f"failed: {failure.get('message', 'see machinery report')}"
            )
            special = _finalize_special(
                classification=str(classification),
                reason=reason,
                python=python,
                logs=logs,
                environment=environment,
                aggregate_dir=aggregate,
                optional_artifacts={
                    "architecture-audit": architecture_json,
                    "preflight": preflight,
                    "machinery": machinery,
                },
                current_commit=current_commit,
            )
            classification = special["classification"]["classification"]
            _write_terminal_status(
                status_path,
                start=start,
                output=output,
                python=python,
                gpu_ids=gpu_ids,
                status="stopped",
                classification=classification,
                summary_path=final_summary,
                current_commit=current_commit,
            )
            print(f"Salvage B stopped: {classification}", flush=True)
            print(f"Final report: {aggregate / 'result_summary_for_gpt.md'}", flush=True)
            return
        if machinery_code != 0:
            raise RuntimeError("Machinery command exited nonzero despite reporting a pass.")
        _assert_frozen_source(output, current_commit)

        launcher_common = [
            "--python",
            python,
            "--gpu-ids",
            *(str(value) for value in gpu_ids),
            "--preflight",
            str(preflight),
            "--world-manifest",
            str(world_manifest),
            "--stochastic-manifest",
            str(stochastic_manifest),
            "--draw-tensors",
            str(draw_tensors),
            "--target-manifest",
            str(target_manifest),
            "--machinery",
            str(machinery),
            "--output-dir",
            str(world_eval),
            "--log-dir",
            str(output / "logs/workers"),
            "--runtime-work-dir",
            str(output / "runtime/workers"),
            "--launch-stagger-seconds",
            str(args.launch_stagger_seconds),
        ]
        _run_stage(
            "world_endpoint",
            [
                python,
                "-m",
                "experiments.asre_diagnosis.salvage_b.launch_world",
                "--phase",
                "endpoint",
                *launcher_common,
            ],
            logs,
            environment=environment,
        )
        _assert_frozen_source(output, current_commit)

        if not endpoint_summary.is_file():
            _run_stage(
                "aggregate_endpoint",
                [
                    python,
                    "-m",
                    "experiments.asre_diagnosis.salvage_b.aggregate_world",
                    "--phase",
                    "endpoint",
                    "--phase-root",
                    str(world_eval / "endpoint"),
                    "--world-manifest",
                    str(world_manifest),
                    "--stochastic-manifest",
                    str(stochastic_manifest),
                    "--target-manifest",
                    str(target_manifest),
                    "--machinery",
                    str(machinery),
                    "--preflight",
                    str(preflight),
                    "--output-dir",
                    str(endpoint_dir),
                ],
                logs,
                environment=environment,
            )
        else:
            print(f"[SalvageB] Reusing endpoint aggregate: {endpoint_summary}", flush=True)
        endpoint_payload = _require_json(
            endpoint_summary,
            label="endpoint aggregate",
            artifact_type="asre_salvage_b_world_phase_aggregate",
            statuses=("passed", "failed"),
            current_commit=current_commit,
        )
        _validate_phase_summary(
            endpoint_payload,
            phase="endpoint",
            world_manifest=world_manifest,
            stochastic_manifest=stochastic_manifest,
            target_manifest=target_manifest,
            machinery=machinery,
            preflight=preflight,
        )
        gate_payload = _require_json(
            endpoint_gate,
            label="world endpoint gate",
            artifact_type="asre_salvage_b_world_endpoint_gate",
            statuses=("passed", "failed"),
        )
        if endpoint_payload.get("endpoint_gate") != gate_payload:
            raise RuntimeError("Endpoint aggregate and standalone endpoint gate disagree.")
        if gate_payload.get("passed") is not True:
            mean = gate_payload.get("mean_wrong_minus_current")
            low = gate_payload.get("paired_bootstrap_ci_low")
            reason = (
                "The registered Current/Wrong native-world endpoint was not informative: "
                f"mean loss(wrong-current)={mean}, paired-bootstrap CI lower={low}."
            )
            special = _finalize_special(
                classification="WORLD-ENDPOINT-UNINFORMATIVE",
                reason=reason,
                python=python,
                logs=logs,
                environment=environment,
                aggregate_dir=aggregate,
                optional_artifacts={
                    "architecture-audit": architecture_json,
                    "preflight": preflight,
                    "machinery": machinery,
                    "endpoint-summary": endpoint_summary,
                },
                current_commit=current_commit,
            )
            classification = special["classification"]["classification"]
            _write_terminal_status(
                status_path,
                start=start,
                output=output,
                python=python,
                gpu_ids=gpu_ids,
                status="stopped",
                classification=classification,
                summary_path=final_summary,
                current_commit=current_commit,
            )
            print(f"Salvage B stopped: {classification}", flush=True)
            print(f"Final report: {aggregate / 'result_summary_for_gpt.md'}", flush=True)
            return

        _run_stage(
            "world_projected",
            [
                python,
                "-m",
                "experiments.asre_diagnosis.salvage_b.launch_world",
                "--phase",
                "projected",
                "--endpoint-gate",
                str(endpoint_gate),
                *launcher_common,
            ],
            logs,
            environment=environment,
        )
        _assert_frozen_source(output, current_commit)
        if not projected_summary.is_file():
            _run_stage(
                "aggregate_projected",
                [
                    python,
                    "-m",
                    "experiments.asre_diagnosis.salvage_b.aggregate_world",
                    "--phase",
                    "projected",
                    "--phase-root",
                    str(world_eval / "projected"),
                    "--world-manifest",
                    str(world_manifest),
                    "--stochastic-manifest",
                    str(stochastic_manifest),
                    "--target-manifest",
                    str(target_manifest),
                    "--machinery",
                    str(machinery),
                    "--preflight",
                    str(preflight),
                    "--output-dir",
                    str(projected_dir),
                ],
                logs,
                environment=environment,
            )
        else:
            print(f"[SalvageB] Reusing projected aggregate: {projected_summary}", flush=True)
        projected_payload = _require_json(
            projected_summary,
            label="projected aggregate",
            artifact_type="asre_salvage_b_world_phase_aggregate",
            statuses=("passed",),
            current_commit=current_commit,
        )
        _validate_phase_summary(
            projected_payload,
            phase="projected",
            world_manifest=world_manifest,
            stochastic_manifest=stochastic_manifest,
            target_manifest=target_manifest,
            machinery=machinery,
            preflight=preflight,
        )
        if endpoint_payload["sample_identity_sha256"] != projected_payload[
            "sample_identity_sha256"
        ]:
            raise RuntimeError(
                "Endpoint/projected aggregates use different frozen sample identities."
            )
        if projected_payload.get("passed") is not True:
            raise RuntimeError("Projected world aggregate did not pass its integrity checks.")
        _assert_frozen_source(output, current_commit)

        if not final_completion.is_file():
            if final_summary.exists():
                print(
                    "[SalvageB] Recovering incomplete final aggregate; no completion "
                    f"sentinel was published: {aggregate}",
                    flush=True,
                )
            _run_stage(
                "aggregate_final",
                [
                    python,
                    "-m",
                    "experiments.asre_diagnosis.salvage_b.aggregate_final",
                    "--endpoint-summary",
                    str(endpoint_summary),
                    "--projected-summary",
                    str(projected_summary),
                    "--preflight",
                    str(preflight),
                    "--architecture-audit",
                    str(architecture_json),
                    "--machinery",
                    str(machinery),
                    "--world-manifest",
                    str(world_manifest),
                    "--stochastic-manifest",
                    str(stochastic_manifest),
                    "--target-manifest",
                    str(target_manifest),
                    "--output-dir",
                    str(aggregate),
                ],
                logs,
                environment=environment,
            )
        else:
            print(
                f"[SalvageB] Reusing published final aggregate: {final_summary}",
                flush=True,
            )
        final_payload, _completion_payload = _validate_final_bundle(
            summary_path=final_summary,
            completion_path=final_completion,
            current_commit=current_commit,
            expected_status="complete",
        )
        _validate_final_summary_links(
            final_payload,
            architecture=architecture_json,
            preflight=preflight,
            machinery=machinery,
            world_manifest=world_manifest,
            stochastic_manifest=stochastic_manifest,
            target_manifest=target_manifest,
            endpoint_summary=endpoint_summary,
            projected_summary=projected_summary,
        )
        _assert_frozen_source(output, current_commit)
        classification = str(final_payload["classification"]["classification"])
        _write_terminal_status(
            status_path,
            start=start,
            output=output,
            python=python,
            gpu_ids=gpu_ids,
            status="complete",
            classification=classification,
            summary_path=final_summary,
            current_commit=current_commit,
        )
        print(f"Salvage B completed: {classification}", flush=True)
        print(f"Final report: {aggregate / 'result_summary_for_gpt.md'}", flush=True)
        print("Stop unconditionally. No Salvage C was launched.", flush=True)
    except Exception as exc:
        atomic_write_json(
            status_path,
            {
                "artifact_type": "asre_salvage_b_driver_status",
                "schema_version": 1,
                "protocol": SALVAGE_B_PROTOCOL,
                "status": "failed",
                "start_timestamp": start,
                "end_timestamp": now_iso(),
                "git_commit_hash": current_commit,
                "output_root": str(output),
                "python": python,
                "gpu_ids": list(gpu_ids),
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "action_rerun": False,
                "online_episodes": 0,
                "environment_rollouts": 0,
                "later_stage_launched": False,
                "salvage_c_exists": False,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--round4b-root", type=Path, default=DEFAULT_ROUND4B_ROOT)
    parser.add_argument("--round4c-root", type=Path, default=DEFAULT_ROUND4C_ROOT)
    parser.add_argument(
        "--salvage-a-summary", type=Path, default=DEFAULT_SALVAGE_A_SUMMARY
    )
    parser.add_argument("--valid-manifest", type=Path, default=DEFAULT_VALID_MANIFEST)
    parser.add_argument("--donor-root", type=Path, default=DEFAULT_DONOR_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-ids", nargs=4, type=int, required=True)
    parser.add_argument("--launch-stagger-seconds", type=float, default=70.0)
    args = parser.parse_args()
    if args.launch_stagger_seconds < 0:
        parser.error("--launch-stagger-seconds must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
