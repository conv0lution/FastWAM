from __future__ import annotations

import inspect

import pandas as pd
import pytest
import torch

from experiments.asre_diagnosis.salvage_b.architecture_audit import build_report
from experiments.asre_diagnosis.salvage_b import world_manifest as world_manifest_module
from experiments.asre_diagnosis.common import sha256_file
from experiments.asre_diagnosis.salvage_b.world_manifest import (
    ACTION_NOISE_SHAPE,
    DRAWS_PER_SAMPLE,
    PROMPT_PREFIX,
    VIDEO_NOISE_SHAPE,
    _draws,
    _load_official_task_identities,
    _select_records,
)
from experiments.asre_diagnosis.salvage_b.world_runtime import load_processed_sample
from fastwam.models.wan22.mot import MoT


def _record(sample_id: str = "sample-0") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_id": 0,
        "episode_id": 101,
        "trial": 0,
        "dataset_index": 0,
        "task_description": "move the object",
    }


def _sample() -> dict[str, object]:
    return {
        "video": torch.zeros(3, 9, 224, 448),
        "action": torch.zeros(32, 7),
        "proprio": torch.zeros(32, 8),
        "image_is_pad": torch.zeros(9, dtype=torch.bool),
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "prompt": (
            "A video recorded from a robot's point of view executing the following "
            "instruction: move the object"
        ),
    }


class _Dataset:
    def __init__(self, sample: dict[str, object]):
        self.sample = sample

    def _get(self, index: int) -> dict[str, object]:
        assert index == 0
        result = dict(self.sample)
        result["_source_dataset_index"] = index
        return result


class _SubstitutingDataset(_Dataset):
    def _get(self, index: int) -> dict[str, object]:
        result = super()._get(index)
        result["_source_dataset_index"] = index + 1
        return result


def _prompt_cache() -> dict[str, object]:
    prompt = str(_sample()["prompt"])
    return {
        "prompts": {
            prompt: {
                "task_id": 0,
                "task_description": "move the object",
                "context": torch.zeros(1, 128, 4096),
                "context_mask": torch.ones(1, 128, dtype=torch.bool),
            }
        }
    }


def test_phase_a_static_contract_and_no_raw_prefix_api() -> None:
    report = build_report()
    assert report["static_audit_passed"] is True
    assert report["path"] == "preferred_path_a"
    assert report["shared_interface"]["layers"] == list(range(15, 30))
    assert report["shared_interface"]["shape_per_tensor"] == [1, 98, 3072]
    assert report["native_metric"]["name"] == (
        "pure_noise_native_future_latent_reconstruction_mse"
    )
    assert report["native_metric"]["inference_steps"] == 10
    assert report["native_metric"]["inference_shift"] == 5.0
    assert report["native_metric"]["target_enters_predictor"] is False
    assert report["native_metric"]["carrier_prefix"] == (
        "constant zero placeholder; discarded before MoT"
    )
    assert report["matched_condition_semantics"][
        "current_observation_enters_world_only_through_selected_kv"
    ] is True
    assert "true-current external cache" in report["required_runtime_gates"][0]
    assert "stock joint future slice" in report["required_runtime_gates"][0]
    assert report["native_metric"]["draws_per_sample"] == 4
    parameters = inspect.signature(
        MoT.forward_future_video_with_video_cache_tensor
    ).parameters
    assert "video_cache_k" in parameters and "video_cache_v" in parameters
    assert not any("prefix_token" in name for name in parameters)


def test_four_draw_artifact_is_deterministic_and_uses_native_shifts() -> None:
    first, first_manifest = _draws([_record()])
    second, second_manifest = _draws([_record()])
    assert first_manifest == second_manifest
    assert len(first_manifest) == DRAWS_PER_SAMPLE
    draws = first["samples"]["sample-0"]
    assert len(draws) == DRAWS_PER_SAMPLE
    for left, right, manifest in zip(
        draws, second["samples"]["sample-0"], first_manifest, strict=True
    ):
        assert tuple(left["video_noise"].shape) == VIDEO_NOISE_SHAPE
        assert tuple(left["action_noise"].shape) == ACTION_NOISE_SHAPE
        assert torch.equal(left["video_noise"], right["video_noise"])
        assert torch.equal(left["action_noise"], right["action_noise"])
        assert "video_timestep" not in left
        assert "video_timestep" not in manifest
        assert float(left["action_timestep"]) == manifest["action_timestep"]
    assert first["native_world_metric"] == (
        "pure_noise_native_future_latent_reconstruction_mse"
    )
    assert first["schedulers"]["video"]["inference_steps"] == 10
    assert first["schedulers"]["video"]["inference_shift"] == 5.0
    assert first["schedulers"]["video"]["initial_state"] == (
        "pure Gaussian future-latent noise"
    )
    assert first["schedulers"]["action"]["shift"] == 1.0


def test_processed_world_target_pairing_is_fail_closed() -> None:
    record = _record()
    frozen = load_processed_sample(
        dataset=_Dataset(_sample()),
        record=record,
        prompt_cache=_prompt_cache(),
    )
    target_record = {
        "processed_tensor_sha256": frozen["processed_tensor_sha256"]
    }
    paired = load_processed_sample(
        dataset=_Dataset(_sample()),
        record=record,
        prompt_cache=_prompt_cache(),
        target_record=target_record,
    )
    assert paired["processed_tensor_sha256"] == target_record[
        "processed_tensor_sha256"
    ]

    changed = _sample()
    changed["action"] = torch.ones(32, 7)
    with pytest.raises(ValueError, match="Processed target drifted"):
        load_processed_sample(
            dataset=_Dataset(changed),
            record=record,
            prompt_cache=_prompt_cache(),
            target_record=target_record,
        )

    with pytest.raises(ValueError, match="substituted an unregistered sample"):
        load_processed_sample(
            dataset=_SubstitutingDataset(_sample()),
            record=record,
            prompt_cache=_prompt_cache(),
        )


def test_world_manifest_resolves_lerobot_task_indices_through_frozen_prompt_cache(
    tmp_path, monkeypatch
) -> None:
    descriptions = {task_id: f"official task {task_id}" for task_id in range(10)}
    # This is the permutation present in the official LeRobot export: its
    # internal task_index is not the online LIBERO suite task_id.
    dataset_task_for_official = {
        0: 0,
        1: 5,
        2: 1,
        3: 6,
        4: 2,
        5: 7,
        6: 3,
        7: 8,
        8: 4,
        9: 9,
    }
    prompts = {
        f"{PROMPT_PREFIX}{description}": {
            "task_id": task_id,
            "task_description": description,
            "context": torch.zeros(1, 128, 4096),
            "context_mask": torch.ones(1, 128, dtype=torch.bool),
        }
        for task_id, description in descriptions.items()
    }
    prompt_cache = tmp_path / "prompt_context_cache.pt"
    torch.save({"prompts": prompts}, prompt_cache)
    preflight = {
        "state": {
            "prompt_context_cache_path": str(prompt_cache),
            "prompt_context_cache_sha256": sha256_file(prompt_cache),
        }
    }
    by_description, identities = _load_official_task_identities(preflight)
    assert by_description[descriptions[1]] == 1
    assert identities[1] == {
        "task_id": 1,
        "task_description": descriptions[1],
    }

    rows = []
    for official_task_id in range(10):
        for trial in range(10):
            episode_id = official_task_id * 10 + trial
            start = episode_id * 33
            row = {
                "episode_index": episode_id,
                "length": 33,
                "dataset_from_index": start,
                "dataset_to_index": start + 33,
                "tasks": [descriptions[official_task_id]],
                "stats/task_index/min": dataset_task_for_official[official_task_id],
                "data/chunk_index": 0,
                "data/file_index": episode_id,
            }
            for camera in (
                "observation.images.image",
                "observation.images.wrist_image",
            ):
                row[f"videos/{camera}/chunk_index"] = 0
                row[f"videos/{camera}/file_index"] = episode_id
                row[f"videos/{camera}/from_timestamp"] = float(episode_id)
                row[f"videos/{camera}/to_timestamp"] = float(episode_id + 1)
            rows.append(row)
    monkeypatch.setattr(
        world_manifest_module, "_episodes", lambda _dataset_root: pd.DataFrame(rows)
    )
    donor_mapping = {
        "mapping_rule": "next trial within official task",
        "records": [
            {
                "task_id": task_id,
                "recipient_trial": trial,
                "donor_trial": (trial + 1) % 10,
            }
            for task_id in range(10)
            for trial in range(10)
        ],
    }
    donor_manifest = {
        "records": [
            {
                "task_id": task_id,
                "source_trial": trial,
                "task_description": descriptions[task_id],
                "processed_image_sha256": "a" * 64,
                "artifact_relative_path": f"task{task_id:02d}_{trial:02d}.pt",
                "artifact_sha256": "b" * 64,
            }
            for task_id in range(10)
            for trial in range(10)
        ]
    }
    records = _select_records(
        dataset_root=tmp_path,
        donor_mapping=donor_mapping,
        donor_manifest=donor_manifest,
        official_task_id_by_description=by_description,
    )
    assert len(records) == 100
    task_one = [record for record in records if record["task_id"] == 1]
    assert len(task_one) == 10
    assert {record["task_description"] for record in task_one} == {
        descriptions[1]
    }
    assert {record["dataset_task_id"] for record in task_one} == {5}
    assert {record["donor_task_id"] for record in task_one} == {1}

    mismatched_donor = dict(donor_manifest)
    mismatched_donor["records"] = [dict(record) for record in donor_manifest["records"]]
    mismatched_donor["records"][11]["task_description"] = "wrong semantic task"
    with pytest.raises(ValueError, match="donor semantic identity"):
        _select_records(
            dataset_root=tmp_path,
            donor_mapping=donor_mapping,
            donor_manifest=mismatched_donor,
            official_task_id_by_description=by_description,
        )
