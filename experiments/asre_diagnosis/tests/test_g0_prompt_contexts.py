from pathlib import Path

import pytest
import torch

from experiments.asre_diagnosis.g0.prepare_prompt_contexts import (
    EXPECTED_CONTEXT_SHAPE,
    EXPECTED_MASK_SHAPE,
    _build_cache_payload,
    validate_prompt_context_cache,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


def _make_payload(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    tasks = {task_id: f"synthetic task {task_id}" for task_id in range(10)}
    context = torch.zeros(EXPECTED_CONTEXT_SHAPE, dtype=torch.bfloat16)
    mask = torch.ones(EXPECTED_MASK_SHAPE, dtype=torch.bool)
    tensors = {
        DEFAULT_PROMPT.format(task=description): (context, mask)
        for description in tasks.values()
    }
    payload = _build_cache_payload(
        suite="libero_object",
        checkpoint=checkpoint,
        checkpoint_sha256="a" * 64,
        task_descriptions=tasks,
        prompt_tensors=tensors,
        commit="b" * 40,
    )
    return checkpoint, tasks, payload


def test_g0_prompt_cache_validates_tensor_and_semantic_identity(tmp_path: Path) -> None:
    checkpoint, tasks, payload = _make_payload(tmp_path)
    cache = tmp_path / "libero_object.pt"
    torch.save(payload, cache)

    observed = validate_prompt_context_cache(
        cache,
        suite="libero_object",
        checkpoint=checkpoint,
        checkpoint_sha256="a" * 64,
        expected_tasks=tasks,
        expected_git_commit="b" * 40,
    )

    assert observed["prompt_count"] == 10
    assert len(observed["prompt_context_manifest_sha256"]) == 64


def test_g0_prompt_cache_rejects_tensor_tampering(tmp_path: Path) -> None:
    checkpoint, tasks, payload = _make_payload(tmp_path)
    first_record = next(iter(payload["prompts"].values()))
    first_record["context"][0, 0, 0] = 1
    cache = tmp_path / "libero_object.pt"
    torch.save(payload, cache)

    with pytest.raises(ValueError, match="tensor digest/shape drift"):
        validate_prompt_context_cache(
            cache,
            suite="libero_object",
            checkpoint=checkpoint,
            checkpoint_sha256="a" * 64,
            expected_tasks=tasks,
            expected_git_commit="b" * 40,
        )
