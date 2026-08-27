"""Fail-fast integrity checks for the frozen Round-3A parent and donor artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.asre_diagnosis.common import (  # noqa: E402
    ROUND3A_PROTOCOL,
    atomic_write_json,
    load_manifest,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (  # noqa: E402
    EXPECTED_SOURCE_SAMPLES,
    _validate_sample_partition,
    validate_existing_artifacts,
)
from experiments.asre_diagnosis.round3b.donor import (  # noqa: E402
    DONOR_MAPPING_RULE,
    OnlineDonorBundle,
    tensor_is_finite,
    tensor_sha256,
)


ROUND3A_TAG = "ASRE-round3a-factorial"
ROUND3A_FROZEN_COMMIT = "d36383c16d974ba1e5a750c088327a7a88baa8fb"
ROUND3A_RUN_COMMIT = "92e842c79f5209b31c2653944cc0c1719e95eb9e"
ROUND3A_CONDITIONS = ("keep_none_late", "keep_20_24", "keep_15_24")
FULL_TASK_IDS = tuple(range(10))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen Round-3A provenance and Round-3B donors."
    )
    parser.add_argument("--round3a-root", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path)
    parser.add_argument("--donor-observation-manifest", type=Path)
    parser.add_argument("--donor-observation-root", type=Path)
    parser.add_argument("--parent-only", action="store_true")
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=project_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "output", "")
        raise RuntimeError(f"Git command failed: git {' '.join(args)}\n{output}") from exc


def _require_git_freeze() -> dict[str, str]:
    tag_commit = _git("rev-list", "-n", "1", ROUND3A_TAG)
    if tag_commit != ROUND3A_FROZEN_COMMIT:
        raise ValueError(
            f"{ROUND3A_TAG} moved: expected {ROUND3A_FROZEN_COMMIT}, got {tag_commit}."
        )
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", ROUND3A_TAG, "HEAD"],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Current HEAD does not descend from frozen {ROUND3A_TAG}.") from exc

    prior_status = _git(
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        "asre_results",
        ":(exclude)asre_results/round3b/**",
    )
    if prior_status:
        raise RuntimeError(
            "Tracked Round-1/2/3A artifacts are modified; refusing Round-3B preflight:\n"
            f"{prior_status}"
        )
    return {
        "round3a_tag": ROUND3A_TAG,
        "round3a_frozen_commit": tag_commit,
        "round3a_run_commit": ROUND3A_RUN_COMMIT,
        "current_head": _git("rev-parse", "HEAD"),
        "current_branch": _git("branch", "--show-current"),
    }


def _mismatches(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        key: {"existing": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }


def _resolve_declared_file(value: Any, *, base: Path, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Valid manifest field {key!r} must be a path string.")
    path = Path(os.path.expanduser(os.path.expandvars(value)))
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact declared by {key!r} is unavailable: {path}")
    return path


def _validate_round3a(
    round3a_root: Path, valid_manifest_path: Path
) -> dict[str, Any]:
    round3a_root = round3a_root.resolve()
    valid_manifest_path = valid_manifest_path.resolve()
    valid_manifest = _read_json(valid_manifest_path, label="Round-2 valid manifest")
    source_manifest_path = _resolve_declared_file(
        valid_manifest.get("source_manifest_path"),
        base=valid_manifest_path.parent,
        key="source_manifest_path",
    )
    checkpoint_path = _resolve_declared_file(
        valid_manifest.get("checkpoint_path"),
        base=valid_manifest_path.parent,
        key="checkpoint_path",
    )
    dataset_stats_path = _resolve_declared_file(
        valid_manifest.get("dataset_stats_path"),
        base=valid_manifest_path.parent,
        key="dataset_stats_path",
    )
    prompt_cache_path = _resolve_declared_file(
        valid_manifest.get("prompt_context_cache_path"),
        base=valid_manifest_path.parent,
        key="prompt_context_cache_path",
    )
    source_records = load_manifest(source_manifest_path)
    valid_ids = _validate_sample_partition(valid_manifest, source_records)
    if len(source_records) != EXPECTED_SOURCE_SAMPLES or len(valid_ids) != 499:
        raise ValueError(
            "Round 3B requires the frozen 500-to-499 state-bank population; "
            f"got {len(source_records)} source and {len(valid_ids)} valid samples."
        )
    validate_existing_artifacts(
        valid_manifest_path=valid_manifest_path,
        prompt_cache_path=prompt_cache_path,
        source_manifest_path=source_manifest_path,
        source_records=source_records,
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        verify_checkpoint_hash=True,
    )

    launcher = _read_json(
        round3a_root / "online_full" / "launcher_summary.json",
        label="Round-3A full launcher summary",
    )
    launcher_expected = {
        "schema_version": 1,
        "protocol": ROUND3A_PROTOCOL,
        "mode": "full",
        "all_succeeded": True,
        "interrupted": False,
        "expected_conditions": list(ROUND3A_CONDITIONS),
        "task_ids": list(FULL_TASK_IDS),
        "num_trials": 10,
        "seed": 42,
        "git_commit_hash": ROUND3A_RUN_COMMIT,
    }
    launcher_mismatches = _mismatches(launcher, launcher_expected)
    states = launcher.get("condition_states")
    if not isinstance(states, dict) or any(
        not isinstance(states.get(name), dict)
        or states[name].get("state") != "complete"
        or states[name].get("completed_task_ids") != list(FULL_TASK_IDS)
        for name in ROUND3A_CONDITIONS
    ):
        launcher_mismatches["condition_states"] = {
            "existing": states,
            "expected": "all three conditions complete for tasks 0-9",
        }
    if launcher_mismatches:
        raise ValueError(
            "Frozen Round-3A launcher summary is incompatible: "
            f"{json.dumps(launcher_mismatches, sort_keys=True)}"
        )

    digest_fields = {
        "checkpoint_sha256": valid_manifest["checkpoint_sha256"],
        "dataset_stats_sha256": valid_manifest["dataset_stats_sha256"],
        "state_bank_manifest_sha256": valid_manifest["source_manifest_sha256"],
        "valid_state_bank_manifest_sha256": sha256_file(valid_manifest_path),
        "prompt_context_cache_sha256": valid_manifest["prompt_context_cache_sha256"],
    }
    for condition in ROUND3A_CONDITIONS:
        metadata = _read_json(
            round3a_root / "online_full" / condition / "run_metadata.json",
            label=f"Round-3A {condition} metadata",
        )
        expected = {
            "status": "completed",
            "condition_protocol": ROUND3A_PROTOCOL,
            "diagnosis_condition": condition,
            "git_commit_hash": ROUND3A_RUN_COMMIT,
            "task_ids": list(FULL_TASK_IDS),
            "number_of_trials": 10,
            **digest_fields,
        }
        mismatch = _mismatches(metadata, expected)
        if mismatch:
            raise ValueError(
                f"Frozen Round-3A {condition} metadata is incompatible: "
                f"{json.dumps(mismatch, sort_keys=True)}"
            )

    aggregate = _read_json(
        round3a_root / "aggregate" / "aggregate_metadata.json",
        label="Round-3A aggregate metadata",
    )
    aggregate_expected = {
        "artifact_type": "asre_round3a_factorial_aggregate_metadata",
        "schema_version": 1,
        "status": "complete",
        "analysis_git_commit_hash": ROUND3A_RUN_COMMIT,
        "valid_state_bank_manifest_sha256": sha256_file(valid_manifest_path),
    }
    mismatch = _mismatches(aggregate, aggregate_expected)
    if mismatch:
        raise ValueError(
            "Frozen Round-3A aggregate is incompatible: "
            f"{json.dumps(mismatch, sort_keys=True)}"
        )
    required_aggregate = tuple(str(name) for name in aggregate.get("output_files", ()))
    if not required_aggregate or any(
        not (round3a_root / "aggregate" / name).is_file() for name in required_aggregate
    ):
        raise FileNotFoundError("Frozen Round-3A aggregate output set is incomplete.")

    return {
        "round3a_launcher_summary_path": str(
            (round3a_root / "online_full" / "launcher_summary.json").resolve()
        ),
        "round3a_launcher_summary_sha256": sha256_file(
            round3a_root / "online_full" / "launcher_summary.json"
        ),
        "round3a_aggregate_metadata_path": str(
            (round3a_root / "aggregate" / "aggregate_metadata.json").resolve()
        ),
        "round3a_aggregate_metadata_sha256": sha256_file(
            round3a_root / "aggregate" / "aggregate_metadata.json"
        ),
        "checkpoint_path": str(checkpoint_path),
        "dataset_stats_path": str(dataset_stats_path),
        "source_manifest_path": str(source_manifest_path),
        "valid_manifest_path": str(valid_manifest_path),
        "prompt_context_cache_path": str(prompt_cache_path),
        "valid_sample_count": len(valid_ids),
        **digest_fields,
    }


def _validate_donors(
    mapping_path: Path,
    observation_manifest_path: Path,
    observation_root: Path,
) -> dict[str, Any]:
    mapping_path = mapping_path.resolve()
    observation_manifest_path = observation_manifest_path.resolve()
    observation_root = observation_root.resolve()
    bundle = OnlineDonorBundle.load(
        mapping_path=mapping_path,
        observation_manifest_path=observation_manifest_path,
        observation_root=observation_root,
    )
    mapping = bundle.mapping_payload
    manifest = bundle.observation_payload
    mappings = bundle.mappings
    observations = bundle.observations
    expected_top_level = {
        "task_suite": "libero_spatial",
        "seed": 42,
        "num_tasks": 10,
        "num_trials": 10,
    }
    for label, payload in (("mapping", mapping), ("manifest", manifest)):
        mismatch = _mismatches(payload, expected_top_level)
        if mismatch:
            raise ValueError(
                f"Donor {label} protocol mismatch: {json.dumps(mismatch, sort_keys=True)}"
            )
    manifest_expected = {
        "num_steps_wait": 30,
        "first_policy_query_environment_step": 30,
        "task_config": "libero_uncond_2cam224_1e-4",
        "model_ready_image_shape": [1, 3, 224, 448],
        "model_ready_image_dtype": "torch.bfloat16",
    }
    mismatch = _mismatches(manifest, manifest_expected)
    if mismatch:
        raise ValueError(
            f"Donor observation protocol mismatch: {json.dumps(mismatch, sort_keys=True)}"
        )
    if mapping.get("mapping_rule") != DONOR_MAPPING_RULE:
        raise ValueError(f"Unexpected donor mapping rule: {mapping.get('mapping_rule')!r}.")
    if len(mappings) != 100 or len(observations) != 100:
        raise ValueError(
            "Round-3B online donors require exactly 100 mapping and observation "
            f"records; got {len(mappings)} and {len(observations)}."
        )

    observed_pairs: set[tuple[int, int]] = set(mappings)
    for (task_id, recipient), record in mappings.items():
        donor = int(record["donor_trial"])
        if task_id not in FULL_TASK_IDS or recipient not in range(10):
            raise ValueError(f"Invalid donor key: task={task_id}, recipient={recipient}.")
        if donor != (recipient + 1) % 10 or donor == recipient:
            raise ValueError(
                f"Donor mapping is not the frozen cyclic derangement for "
                f"task={task_id}, recipient={recipient}: donor={donor}."
            )
        for key in (
            "recipient_initial_state_sha256",
            "donor_initial_state_sha256",
            "task_text_sha256",
        ):
            value = record.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Donor mapping record has invalid {key}: {record}")
        if record["recipient_initial_state_sha256"] == record["donor_initial_state_sha256"]:
            raise ValueError(f"Recipient and donor state hashes are identical: {record}")
    expected_pairs = {(task_id, trial) for task_id in FULL_TASK_IDS for trial in range(10)}
    if observed_pairs != expected_pairs:
        raise ValueError("Donor mapping does not cover each task/trial exactly once.")

    observation_keys: set[tuple[int, int]] = set(observations)
    for key, record in observations.items():
        relative = record.get("artifact_relative_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"Donor observation has invalid relative path: {record}")
        path = Path(relative)
        path = (observation_root / path).resolve()
        if not path.is_file() or not path.is_relative_to(observation_root):
            raise FileNotFoundError(f"Invalid donor observation path: {path}")
        expected_digest = record.get("artifact_sha256")
        if not isinstance(expected_digest, str) or sha256_file(path) != expected_digest:
            raise ValueError(f"Donor observation digest mismatch: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        image = artifact.get("input_image") if isinstance(artifact, Mapping) else None
        if not torch.is_tensor(image) or not tensor_is_finite(image):
            raise ValueError(f"Malformed or non-finite donor observation: {path}")
        if tensor_sha256(image) != record.get("processed_image_sha256"):
            raise ValueError(f"Donor image digest mismatch: {path}")
        if list(image.shape) != record.get("processed_image_shape"):
            raise ValueError(f"Donor image shape mismatch: {path}")
        if (int(artifact.get("task_id", -1)), int(artifact.get("source_trial", -1))) != key:
            raise ValueError(f"Donor artifact identity mismatch: {path}")
    if observation_keys != expected_pairs:
        raise ValueError("Donor observation manifest does not cover task/trial grid.")

    manifest_sha256 = sha256_file(observation_manifest_path)
    declared_manifest_digest = mapping.get("donor_observation_manifest_sha256")
    if declared_manifest_digest is not None and declared_manifest_digest != manifest_sha256:
        raise ValueError("Donor mapping was not frozen against this observation manifest.")
    return {
        "donor_mapping_path": str(mapping_path),
        "donor_mapping_sha256": sha256_file(mapping_path),
        "donor_observation_manifest_path": str(observation_manifest_path),
        "donor_observation_manifest_sha256": manifest_sha256,
        "donor_observation_root": str(observation_root),
        "online_mapping_count": len(mappings),
        "online_observation_count": len(observations),
        "mapping_rule": DONOR_MAPPING_RULE,
    }


def run_preflight(
    *,
    round3a_root: Path,
    valid_manifest_path: Path,
    donor_mapping_path: Path | None = None,
    donor_observation_manifest_path: Path | None = None,
    donor_observation_root: Path | None = None,
    parent_only: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "artifact_type": "asre_round3b_preflight_report",
        "schema_version": 1,
        "status": "running",
        "created_at": now_iso(),
        "git": _require_git_freeze(),
        "round3a": _validate_round3a(round3a_root, valid_manifest_path),
    }
    donor_args = (
        donor_mapping_path,
        donor_observation_manifest_path,
        donor_observation_root,
    )
    if parent_only:
        if any(value is not None for value in donor_args):
            raise ValueError("--parent-only cannot be combined with donor artifact paths.")
        report["donors"] = None
    else:
        if any(value is None for value in donor_args):
            raise ValueError(
                "Full preflight requires donor mapping, observation manifest, and root."
            )
        assert all(value is not None for value in donor_args)
        report["donors"] = _validate_donors(
            donor_mapping_path, donor_observation_manifest_path, donor_observation_root
        )
    report["status"] = "compatible"
    report["completed_at"] = now_iso()
    return report


def main() -> None:
    args = _parse_args()
    report = run_preflight(
        round3a_root=args.round3a_root,
        valid_manifest_path=args.valid_manifest,
        donor_mapping_path=args.donor_mapping,
        donor_observation_manifest_path=args.donor_observation_manifest,
        donor_observation_root=args.donor_observation_root,
        parent_only=args.parent_only,
    )
    if args.output_report is not None:
        output_report = args.output_report.resolve()
        if output_report.exists():
            existing = _read_json(output_report, label="existing Round-3B preflight report")
            stable_keys = (
                "artifact_type",
                "schema_version",
                "status",
                "git",
                "round3a",
                "donors",
            )
            mismatch = {
                key: {"existing": existing.get(key), "requested": report.get(key)}
                for key in stable_keys
                if existing.get(key) != report.get(key)
            }
            if mismatch:
                raise FileExistsError(
                    "Refusing to overwrite an incompatible frozen preflight report: "
                    f"{json.dumps(mismatch, sort_keys=True)}"
                )
            report = existing
        else:
            atomic_write_json(output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Round-3B preflight passed; frozen prior artifacts were not modified.")


if __name__ == "__main__":
    main()
