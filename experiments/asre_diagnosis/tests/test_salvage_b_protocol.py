from __future__ import annotations

import inspect

import pytest
import torch

from experiments.asre_diagnosis.salvage_b.architecture_audit import build_report
from experiments.asre_diagnosis.salvage_b.world_manifest import (
    ACTION_NOISE_SHAPE,
    DRAWS_PER_SAMPLE,
    VIDEO_NOISE_SHAPE,
    _draws,
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
