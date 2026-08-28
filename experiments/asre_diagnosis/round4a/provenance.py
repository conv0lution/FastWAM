from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.asre_diagnosis.common import (
    ROUND4A_PROTOCOL,
    git_commit,
    sha256_file,
)
from experiments.asre_diagnosis.round4a.masks import load_mask_manifest
from experiments.asre_diagnosis.round4a.preflight import (
    G0_COMMIT,
    G0_TAG,
    PROJECT_ROOT,
    ROUND3B_COMMIT,
    ROUND3B_TAG,
)


@dataclass(frozen=True)
class Round4AProvenance:
    git_commit_hash: str
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_stats_path: Path
    dataset_stats_sha256: str
    source_manifest_path: Path
    source_manifest_sha256: str
    valid_manifest_path: Path
    valid_manifest_sha256: str
    prompt_context_cache_path: Path
    prompt_context_cache_sha256: str
    online_donor_mapping_path: Path
    online_donor_mapping_sha256: str
    online_donor_manifest_path: Path
    online_donor_manifest_sha256: str
    online_donor_root: Path
    offline_donor_mapping_path: Path
    offline_donor_mapping_sha256: str
    preflight_report_path: Path
    preflight_report_sha256: str
    machinery_report_path: Path
    machinery_report_sha256: str
    mask_manifest_path: Path
    mask_manifest_sha256: str
    token_mask_manifest_path: Path
    token_mask_manifest_sha256: str
    head_mask_manifest_path: Path
    head_mask_manifest_sha256: str
    round3b_parent_tag: str
    round3b_parent_commit: str
    round3b_summary_path: Path
    round3b_summary_sha256: str
    g0_parent_tag: str
    g0_parent_commit: str
    g0_summary_path: Path
    g0_summary_sha256: str
    valid_sample_count: int

    @property
    def state_bank_dir(self) -> Path:
        return self.source_manifest_path.parent

    def identity_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object.")
    return payload


def _required_path(value: Any, *, label: str, directory: bool = False) -> Path:
    path = Path(str(value)).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise FileNotFoundError(f"Round-4A {label} is unavailable: {path}")
    return path


def _expect(payload: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    mismatch = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"{label} mismatch: {json.dumps(mismatch, sort_keys=True)}")


def load_round4a_provenance(
    *,
    preflight_report_path: Path,
    machinery_report_path: Path,
    mask_manifest_path: Path,
) -> Round4AProvenance:
    preflight_report_path = _required_path(
        preflight_report_path, label="preflight report"
    )
    machinery_report_path = _required_path(
        machinery_report_path, label="machinery report"
    )
    mask_manifest_path = _required_path(mask_manifest_path, label="mask manifest")
    preflight = _read_json(preflight_report_path, label="Round-4A preflight report")
    machinery = _read_json(machinery_report_path, label="Round-4A machinery report")
    current_head = git_commit(PROJECT_ROOT)
    _expect(
        preflight,
        {
            "artifact_type": "asre_round4a_preflight_report",
            "schema_version": 1,
            "protocol": ROUND4A_PROTOCOL,
            "status": "compatible",
        },
        "Round-4A preflight",
    )
    _expect(
        preflight.get("git", {}),
        {
            "current_head": current_head,
            "round3b_parent_tag": ROUND3B_TAG,
            "round3b_parent_commit": ROUND3B_COMMIT,
            "g0_parent_tag": G0_TAG,
            "g0_parent_commit": G0_COMMIT,
        },
        "Round-4A Git provenance",
    )
    _expect(
        machinery,
        {
            "artifact_type": "asre_round4a_machinery_report",
            "schema_version": 1,
            "protocol": ROUND4A_PROTOCOL,
            "status": "passed",
            "passed": True,
            "git_commit_hash": current_head,
            "preflight_report_path": str(preflight_report_path),
            "preflight_report_sha256": sha256_file(preflight_report_path),
            "mask_manifest_path": str(mask_manifest_path),
            "mask_manifest_sha256": sha256_file(mask_manifest_path),
            "all_action_outputs_shape_32x7_and_finite": True,
        },
        "Round-4A machinery",
    )
    mask_sha256 = sha256_file(mask_manifest_path)
    combined_mask = load_mask_manifest(
        path=mask_manifest_path, expected_sha256=mask_sha256
    )
    token_mask_manifest = _required_path(
        machinery.get("token_mask_manifest_path"), label="token mask manifest"
    )
    head_mask_manifest = _required_path(
        machinery.get("head_mask_manifest_path"), label="head mask manifest"
    )
    for axis, path in (
        ("token", token_mask_manifest),
        ("head", head_mask_manifest),
    ):
        expected_axis_digest = str(machinery.get(f"{axis}_mask_manifest_sha256"))
        if sha256_file(path) != expected_axis_digest:
            raise ValueError(f"Round-4A {axis} mask manifest SHA256 drifted.")
        axis_payload = _read_json(path, label=f"Round-4A {axis} mask manifest")
        _expect(
            axis_payload,
            {
                "schema_version": 1,
                "protocol": ROUND4A_PROTOCOL,
                "axis": axis,
                "parent_mask_manifest_path": str(mask_manifest_path),
                "parent_mask_manifest_sha256": mask_sha256,
                "selection_algorithm": combined_mask["selection_algorithm"],
                "rounding_rule": combined_mask["rounding_rule"],
                "mask_fraction_target": combined_mask["mask_fraction_target"],
                "runtime_layout": combined_mask["runtime_layout"],
                "conditions": {
                    name: condition
                    for name, condition in combined_mask["conditions"].items()
                    if condition["axis"] == axis
                },
            },
            f"Round-4A {axis} mask manifest",
        )

    state = preflight["state_bank"]
    online = preflight["online_donors"]
    offline = preflight["offline_donors"]
    stage1 = preflight["stage1"]
    checkpoint = _required_path(state["checkpoint_path"], label="checkpoint")
    dataset_stats = _required_path(
        state["dataset_stats_path"], label="dataset stats"
    )
    source_manifest = _required_path(
        state["source_manifest_path"], label="source state manifest"
    )
    valid_manifest = _required_path(
        state["valid_state_bank_manifest_path"], label="QC-valid state manifest"
    )
    prompt_cache = _required_path(
        state["prompt_context_cache_path"], label="prompt context cache"
    )
    online_mapping = _required_path(
        online["donor_mapping_path"], label="online donor mapping"
    )
    online_manifest = _required_path(
        online["donor_observation_manifest_path"],
        label="online donor observation manifest",
    )
    online_root = _required_path(
        online["donor_observation_root"],
        label="online donor root",
        directory=True,
    )
    offline_mapping = _required_path(
        offline["offline_donor_mapping_path"], label="offline donor mapping"
    )
    round3b_summary = _required_path(
        stage1["round3b_summary_path"], label="Round-3B summary"
    )
    g0_summary = _required_path(stage1["g0_summary_path"], label="G0 summary")
    digest_pairs = (
        (checkpoint, state["checkpoint_sha256"], "checkpoint"),
        (dataset_stats, state["dataset_stats_sha256"], "dataset stats"),
        (source_manifest, state["source_manifest_sha256"], "source manifest"),
        (valid_manifest, state["valid_state_bank_manifest_sha256"], "valid manifest"),
        (prompt_cache, state["prompt_context_cache_sha256"], "prompt cache"),
        (online_mapping, online["donor_mapping_sha256"], "online donor mapping"),
        (
            online_manifest,
            online["donor_observation_manifest_sha256"],
            "online donor manifest",
        ),
        (
            offline_mapping,
            offline["offline_donor_mapping_sha256"],
            "offline donor mapping",
        ),
        (round3b_summary, stage1["round3b_summary_sha256"], "Round-3B summary"),
        (g0_summary, stage1["g0_summary_sha256"], "G0 summary"),
    )
    for path, expected_digest, label in digest_pairs:
        observed = sha256_file(path)
        if observed != expected_digest:
            raise ValueError(
                f"Round-4A {label} SHA256 drift: {observed} != {expected_digest}."
            )
    return Round4AProvenance(
        git_commit_hash=current_head,
        checkpoint_path=checkpoint,
        checkpoint_sha256=str(state["checkpoint_sha256"]),
        dataset_stats_path=dataset_stats,
        dataset_stats_sha256=str(state["dataset_stats_sha256"]),
        source_manifest_path=source_manifest,
        source_manifest_sha256=str(state["source_manifest_sha256"]),
        valid_manifest_path=valid_manifest,
        valid_manifest_sha256=str(state["valid_state_bank_manifest_sha256"]),
        prompt_context_cache_path=prompt_cache,
        prompt_context_cache_sha256=str(state["prompt_context_cache_sha256"]),
        online_donor_mapping_path=online_mapping,
        online_donor_mapping_sha256=str(online["donor_mapping_sha256"]),
        online_donor_manifest_path=online_manifest,
        online_donor_manifest_sha256=str(
            online["donor_observation_manifest_sha256"]
        ),
        online_donor_root=online_root,
        offline_donor_mapping_path=offline_mapping,
        offline_donor_mapping_sha256=str(offline["offline_donor_mapping_sha256"]),
        preflight_report_path=preflight_report_path,
        preflight_report_sha256=sha256_file(preflight_report_path),
        machinery_report_path=machinery_report_path,
        machinery_report_sha256=sha256_file(machinery_report_path),
        mask_manifest_path=mask_manifest_path,
        mask_manifest_sha256=mask_sha256,
        token_mask_manifest_path=token_mask_manifest,
        token_mask_manifest_sha256=sha256_file(token_mask_manifest),
        head_mask_manifest_path=head_mask_manifest,
        head_mask_manifest_sha256=sha256_file(head_mask_manifest),
        round3b_parent_tag=ROUND3B_TAG,
        round3b_parent_commit=ROUND3B_COMMIT,
        round3b_summary_path=round3b_summary,
        round3b_summary_sha256=str(stage1["round3b_summary_sha256"]),
        g0_parent_tag=G0_TAG,
        g0_parent_commit=G0_COMMIT,
        g0_summary_path=g0_summary,
        g0_summary_sha256=str(stage1["g0_summary_sha256"]),
        valid_sample_count=int(state["valid_sample_count"]),
    )
