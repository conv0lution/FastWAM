from __future__ import annotations

import copy

import pytest
import torch

from experiments.asre_diagnosis.round3b.offline_donor import (
    build_offline_donor_pairs,
    tensor_sha256,
)


def _record(task: int, episode: int, replan: int) -> dict:
    return {
        "sample_id": f"task{task:02d}_episode{episode:02d}_replan{replan:03d}",
        "task_suite": "libero_spatial",
        "task_id": task,
        "episode_id": episode,
        "replan_id": replan,
    }


def test_offline_mapping_is_same_group_cyclic_derangement() -> None:
    records = [
        _record(task, episode, replan)
        for task in range(2)
        for replan in range(2)
        for episode in range(3)
    ]
    mapping = build_offline_donor_pairs(records)
    by_id = {record["sample_id"]: record for record in records}
    assert set(mapping) == set(by_id)
    for recipient_id, donor_id in mapping.items():
        recipient = by_id[recipient_id]
        donor = by_id[donor_id]
        assert recipient_id != donor_id
        assert recipient["task_id"] == donor["task_id"]
        assert recipient["replan_id"] == donor["replan_id"]
        assert recipient["episode_id"] != donor["episode_id"]


def test_offline_mapping_rejects_duplicate_or_singleton_groups() -> None:
    record = _record(0, 0, 0)
    with pytest.raises(ValueError, match="at least two"):
        build_offline_donor_pairs([record])
    with pytest.raises(ValueError, match="Duplicate"):
        build_offline_donor_pairs([record, copy.deepcopy(record)])


def test_tensor_hash_includes_exact_values_and_metadata() -> None:
    value = torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 2, 2)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    changed = value.clone()
    changed.view(-1)[0] += 1
    assert tensor_sha256(value) != tensor_sha256(changed)
    assert tensor_sha256(value) != tensor_sha256(value.float())
