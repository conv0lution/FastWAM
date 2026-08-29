from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.asre_diagnosis.salvage_a import donor as donor_module
from experiments.asre_diagnosis.salvage_a.donor import (
    build_split_local_donor_mapping,
    validate_split_local_donor_mapping,
)
from experiments.asre_diagnosis.salvage_a.split import build_split_manifest
from experiments.asre_diagnosis.salvage_a.state_selection import (
    build_state_selection_manifest,
    validate_state_selection_payload,
)


def _source_records() -> tuple[list[dict], dict]:
    records: list[dict] = []
    for task_id in range(10):
        for episode_id in range(10):
            for replan_id in range(5):
                records.append(
                    {
                        "sample_id": (
                            f"task{task_id:02d}_episode{episode_id:02d}_"
                            f"replan{replan_id:03d}"
                        ),
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "replan_id": replan_id,
                    }
                )
    excluded = records[0]["sample_id"]
    valid = {
        "source_manifest_path": "synthetic",
        "valid_sample_ids": [
            record["sample_id"]
            for record in records
            if record["sample_id"] != excluded
        ],
        "excluded_samples": [
            {"sample_id": excluded, "reasons": ["synthetic_replay_qc"]}
        ],
    }
    return records, valid


def _split(tmp_path: Path) -> tuple[dict, Path, list[dict]]:
    records, valid = _source_records()
    valid_path = tmp_path / "valid_manifest.json"
    source_path = tmp_path / "state_manifest.jsonl"
    valid_path.write_text("{}\n", encoding="utf-8")
    source_path.write_text("synthetic\n", encoding="utf-8")
    payload = build_split_manifest(
        valid_manifest=valid,
        source_records=records,
        valid_manifest_path=valid_path,
        source_manifest_path=source_path,
    )
    split_path = tmp_path / "calibration_split_manifest.json"
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, split_path, records


def test_salvage_a_split_is_task_stratified_and_cluster_disjoint(
    tmp_path: Path,
) -> None:
    split, _path, records = _split(tmp_path)
    assert len(split["per_task"]) == 10
    calibration_keys: set[tuple[int, int]] = set()
    heldout_keys: set[tuple[int, int]] = set()
    for task in split["per_task"]:
        task_id = int(task["task_id"])
        calibration = set(task["calibration_episode_ids"])
        heldout = set(task["heldout_episode_ids"])
        assert len(calibration) == len(heldout) == 5
        assert calibration.isdisjoint(heldout)
        assert calibration | heldout == set(range(10))
        calibration_keys |= {(task_id, episode) for episode in calibration}
        heldout_keys |= {(task_id, episode) for episode in heldout}
    assert len(calibration_keys) == len(heldout_keys) == 50
    assert calibration_keys.isdisjoint(heldout_keys)
    by_id = {record["sample_id"]: record for record in records}
    assert {
        (by_id[identifier]["task_id"], by_id[identifier]["episode_id"])
        for identifier in split["calibration_sample_ids"]
    } == calibration_keys
    assert {
        (by_id[identifier]["task_id"], by_id[identifier]["episode_id"])
        for identifier in split["heldout_sample_ids"]
    } == heldout_keys


def test_salvage_a_exact_states_and_stability_halves_are_frozen(
    tmp_path: Path,
) -> None:
    split, split_path, records = _split(tmp_path)
    selection = build_state_selection_manifest(
        split_manifest=split,
        split_manifest_path=split_path,
        source_records=records,
    )
    validate_state_selection_payload(selection, split_manifest=split)
    assert len(selection["calibration_sample_ids"]) == 100
    assert len(selection["heldout_sample_ids"]) == 100
    assert len(selection["stability_half_a_episode_keys"]) == 25
    assert len(selection["stability_half_b_episode_keys"]) == 25
    assert len(selection["stability_half_a_sample_ids"]) == 50
    assert len(selection["stability_half_b_sample_ids"]) == 50
    assert set(selection["stability_half_a_sample_ids"]).isdisjoint(
        selection["stability_half_b_sample_ids"]
    )
    for record in (
        selection["selected_by_episode"]
        + selection["heldout_selected_by_episode"]
    ):
        replans = record["valid_replan_ids"]
        assert record["selected_replan_ids"] == [min(replans), max(replans)]
    assert selection["split_sha256"] == selection["split_manifest_sha256"]

    corrupted = dict(selection)
    corrupted["stability_half_b_sample_ids"] = list(
        selection["stability_half_a_sample_ids"]
    )
    with pytest.raises(ValueError, match="stability halves"):
        validate_state_selection_payload(corrupted, split_manifest=split)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _observation_manifest(tmp_path: Path) -> dict:
    records = []
    for task_id in range(10):
        text_sha = _digest(f"task text {task_id}")
        for trial in range(10):
            records.append(
                {
                    "task_id": task_id,
                    "source_task_id": task_id,
                    "source_trial": trial,
                    "task_text_sha256": text_sha,
                    "initial_state_sha256": _digest(f"state {task_id} {trial}"),
                    "processed_image_sha256": _digest(f"image {task_id} {trial}"),
                    "processed_image_shape": [1],
                    "processed_image_dtype": "torch.float32",
                    "processed_image_finite": True,
                    "processed_image_min": float(trial) / 10.0,
                    "processed_image_max": float(trial) / 10.0,
                    "artifact_relative_path": f"observations/{task_id}_{trial}.pt",
                    "artifact_absolute_path": str(
                        tmp_path / "observations" / f"{task_id}_{trial}.pt"
                    ),
                    "artifact_sha256": _digest(f"artifact {task_id} {trial}"),
                }
            )
    return {
        "schema_version": 1,
        "task_suite": "libero_spatial",
        "seed": 42,
        "num_tasks": 10,
        "num_trials": 10,
        "task_ids": list(range(10)),
        "records": records,
    }


def test_salvage_a_donor_derangement_never_crosses_frozen_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split, split_path, _records = _split(tmp_path)
    observations = _observation_manifest(tmp_path)
    observation_path = tmp_path / "donor_observation_manifest.json"
    observation_path.write_text(json.dumps(observations), encoding="utf-8")

    def fake_image(record, *, observation_root):
        del observation_root
        return torch.tensor([float(record["source_trial"])], dtype=torch.float32)

    monkeypatch.setattr(donor_module, "_load_observation_image", fake_image)
    mapping = build_split_local_donor_mapping(
        split_manifest=split,
        split_manifest_path=split_path,
        observation_manifest=observations,
        observation_manifest_path=observation_path,
        observation_root=tmp_path,
    )
    indexed = validate_split_local_donor_mapping(
        mapping,
        observation_manifest=observations,
        split_manifest=split,
    )
    assert len(indexed) == 100
    assert mapping["split_sha256"] == mapping["split_manifest_sha256"]
    for (task_id, recipient), record in indexed.items():
        assert int(record["donor_trial"]) != recipient
        assert int(record["task_id"]) == task_id
        assert record["recipient_partition"] == record["donor_partition"]

    corrupted = json.loads(json.dumps(mapping))
    first = corrupted["records"][0]
    first["donor_trial"] = first["recipient_trial"]
    with pytest.raises(ValueError, match="derangement"):
        validate_split_local_donor_mapping(
            corrupted,
            observation_manifest=observations,
            split_manifest=split,
        )
