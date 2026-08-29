from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.asre_diagnosis.common import (
    ROUND4B_PROTOCOL,
    build_round4b_conditions,
    resolve_condition,
)
from experiments.asre_diagnosis.round4b.classification import classify_subspace
from experiments.asre_diagnosis.round4b.split import build_split_manifest
from fastwam.models.wan22.video_cache_replacement import (
    project_replacement_video_cache,
)


def _cfg(index: int) -> dict:
    condition = build_round4b_conditions(30)[index]
    return {
        "enabled": True,
        "mode": "replace_video_kv",
        "protocol": ROUND4B_PROTOCOL,
        "condition_index": index,
        "condition_name": condition.name,
        "enabled_video_retrieval_layers": list(range(15, 30)),
        "disabled_video_layers": list(range(15)),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "subspace_basis_kind": condition.basis_kind,
        "subspace_rank": condition.subspace_rank,
    }


def test_round4b_condition_matrix_and_ranks_are_frozen() -> None:
    conditions = build_round4b_conditions(30)
    assert [item.name for item in conditions] == [
        "current_all",
        "wrong_all",
        "svd_r256",
        "random_r256",
        "svd_r768",
        "random_r768",
        "svd_r1536",
        "random_r1536",
    ]
    assert [item.subspace_rank for item in conditions] == [
        None,
        None,
        256,
        256,
        768,
        768,
        1536,
        1536,
    ]
    for index, condition in enumerate(conditions):
        assert resolve_condition(_cfg(index), 30) == condition


def test_round4b_condition_rejects_rank_or_basis_drift() -> None:
    cfg = _cfg(4)
    cfg["subspace_rank"] = 256
    with pytest.raises(ValueError, match="subspace_rank"):
        resolve_condition(cfg, 30)
    cfg = _cfg(4)
    cfg["subspace_basis_kind"] = "random"
    with pytest.raises(ValueError, match="subspace_basis_kind"):
        resolve_condition(cfg, 30)


def _cache():
    current_k = [torch.randn(1, 4, 6)]
    current_v = [torch.randn(1, 4, 6)]
    donor_k = [torch.randn(1, 4, 6)]
    donor_v = [torch.randn(1, 4, 6)]
    return current_k, current_v, donor_k, donor_v


def test_feature_projection_has_exact_rank0_and_rankD_endpoints() -> None:
    current_k, current_v, donor_k, donor_v = _cache()
    for rank, basis, expected_k, expected_v in (
        (0, torch.empty(6, 0), donor_k, donor_v),
        (6, torch.eye(6), current_k, current_v),
    ):
        out_k, out_v, audit = project_replacement_video_cache(
            current_cache_k=current_k,
            current_cache_v=current_v,
            replacement_cache_k=donor_k,
            replacement_cache_v=donor_v,
            replacement_video_layers=(0,),
            action_visible_token_indices=(0, 1, 2, 3),
            feature_bases_by_layer={0: {"k": basis, "v": basis}},
            projection_rank=rank,
            num_layers=1,
        )
        torch.testing.assert_close(out_k[0], expected_k[0])
        torch.testing.assert_close(out_v[0], expected_v[0])
        assert audit["tokens_modified"] is False
        assert audit["heads_modified"] is False
        assert audit["shape_preserved"] is True


def test_feature_projection_uses_independent_k_and_v_bases_and_visible_rows_only() -> None:
    current_k, current_v, donor_k, donor_v = _cache()
    k_basis = torch.eye(6)[:, :2]
    v_basis = torch.eye(6)[:, 2:4]
    out_k, out_v, _ = project_replacement_video_cache(
        current_cache_k=current_k,
        current_cache_v=current_v,
        replacement_cache_k=donor_k,
        replacement_cache_v=donor_v,
        replacement_video_layers=(0,),
        action_visible_token_indices=(0, 2),
        feature_bases_by_layer={0: {"k": k_basis, "v": v_basis}},
        projection_rank=2,
        num_layers=1,
    )
    torch.testing.assert_close(out_k[0][:, 1], current_k[0][:, 1])
    torch.testing.assert_close(out_v[0][:, 3], current_v[0][:, 3])
    assert not torch.equal(out_k[0][:, 0], out_v[0][:, 0])


def test_task_stratified_split_has_no_episode_leakage(tmp_path: Path) -> None:
    records = []
    ids = []
    for task in range(10):
        for episode in range(10):
            for replan in range(5):
                identifier = f"t{task}e{episode}r{replan}"
                records.append(
                    {
                        "sample_id": identifier,
                        "task_id": task,
                        "episode_id": episode,
                        "replan_id": replan,
                    }
                )
                ids.append(identifier)
    excluded = ids.pop(0)
    valid = {
        "valid_sample_ids": ids,
        "excluded_samples": [{"sample_id": excluded, "reasons": ["synthetic_qc"]}],
    }
    valid_path = tmp_path / "valid.json"
    source_path = tmp_path / "manifest.jsonl"
    valid_path.write_text("{}")
    source_path.write_text("\n")
    payload = build_split_manifest(
        valid_manifest=valid,
        source_records=records,
        valid_manifest_path=valid_path,
        source_manifest_path=source_path,
    )
    assert payload["fit_episode_clusters"] == 80
    assert payload["holdout_episode_clusters"] == 20
    assert payload["fit_sample_count"] + payload["holdout_sample_count"] == 499
    fit_episodes = {
        (record["task_id"], record["episode_id"])
        for record in records
        if record["sample_id"] in set(payload["fit_sample_ids"])
    }
    holdout_episodes = {
        (record["task_id"], record["episode_id"])
        for record in records
        if record["sample_id"] in set(payload["holdout_sample_ids"])
    }
    assert fit_episodes.isdisjoint(holdout_episodes)


def _task_rates(current: float, **rates: float):
    values = {"current_all": {task: current for task in range(10)}}
    for name, rate in rates.items():
        values[name] = {task: rate for task in range(10)}
    values["wrong_all"] = {task: 0.0 for task in range(10)}
    return values


def test_classification_strong_and_generic_are_deterministic() -> None:
    strong_rates = {
        "current_all": 0.95,
        "wrong_all": 0.0,
        "svd_r256": 0.60,
        "random_r256": 0.30,
        "svd_r768": 0.92,
        "random_r768": 0.60,
        "svd_r1536": 0.94,
        "random_r1536": 0.70,
    }
    assert classify_subspace(
        success_rates=strong_rates,
        task_success=_task_rates(0.95, **{k: v for k, v in strong_rates.items() if k not in {"current_all", "wrong_all"}}),
    )["classification"] == "STRONG"
    generic_rates = dict(strong_rates)
    generic_rates.update(
        {"svd_r768": 0.93, "random_r768": 0.92, "svd_r1536": 0.94, "random_r1536": 0.94}
    )
    assert classify_subspace(
        success_rates=generic_rates,
        task_success=_task_rates(0.95, **{k: v for k, v in generic_rates.items() if k not in {"current_all", "wrong_all"}}),
    )["classification"] == "GENERIC"
