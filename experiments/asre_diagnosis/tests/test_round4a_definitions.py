from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.asre_diagnosis.common import (
    ROUND4A_PROTOCOL,
    build_round4a_conditions,
    resolve_condition,
    sha256_file,
)
from experiments.asre_diagnosis.round4a.masks import (
    LATE_LAYERS,
    build_mask_manifest,
    load_mask_spec,
    runtime_layout_from_cache_stats,
    validate_mask_manifest,
    write_or_verify_frozen_manifest,
)


def _cache_stats() -> dict:
    layers = []
    for layer in range(30):
        summarized = layer in LATE_LAYERS
        layers.append(
            {
                "layer": layer,
                "summarized": summarized,
                "current": (
                    {
                        "k": {"shape": [1, 6, 12]},
                        "v": {"shape": [1, 6, 12]},
                    }
                    if summarized
                    else None
                ),
            }
        )
    return {
        "current_video_seq_len": 6,
        "action_visible_video_token_indices": [0, 1, 2, 3, 4, 5],
        "num_heads": 3,
        "head_dim": 4,
        "current_video_tokens_per_frame": 6,
        "current_video_grid_size": [1, 1, 6],
        "current_input_image_shape": [1, 3, 224, 448],
        "action_attention_mask_shape": [32, 38],
        "layers": layers,
    }


def _cfg(name: str, index: int) -> dict:
    condition = build_round4a_conditions(30)[index]
    return {
        "enabled": True,
        "mode": "replace_video_kv",
        "protocol": ROUND4A_PROTOCOL,
        "condition_name": name,
        "condition_index": index,
        "enabled_video_retrieval_layers": list(range(15, 30)),
        "disabled_video_layers": list(range(15)),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "hybrid_axis": condition.hybrid_axis,
        "hybrid_mask_seed": condition.mask_seed,
    }


def test_round4a_condition_matrix_is_frozen() -> None:
    conditions = build_round4a_conditions(30)
    assert [condition.name for condition in conditions] == [
        "current_all",
        "wrong_all",
        "head50_seed1",
        "head50_seed2",
        "head50_seed3",
        "token50_seed1",
        "token50_seed2",
        "token50_seed3",
    ]
    assert all(condition.disabled_video_layers == tuple(range(15)) for condition in conditions)
    assert conditions[0].replacement_video_layers == ()
    assert all(
        condition.replacement_video_layers == tuple(range(15, 30))
        for condition in conditions[1:]
    )
    for index, condition in enumerate(conditions):
        assert resolve_condition(_cfg(condition.name, index), 30) == condition


def test_round4a_condition_rejects_axis_or_seed_drift() -> None:
    cfg = _cfg("head50_seed1", 2)
    cfg["hybrid_axis"] = "token"
    with pytest.raises(ValueError, match="hybrid_axis"):
        resolve_condition(cfg, 30)
    cfg = _cfg("head50_seed1", 2)
    cfg["hybrid_mask_seed"] = 3
    with pytest.raises(ValueError, match="hybrid_mask_seed"):
        resolve_condition(cfg, 30)


def test_runtime_layout_and_masks_are_deterministic_and_exact_half(tmp_path: Path) -> None:
    layout = runtime_layout_from_cache_stats(_cache_stats())
    payload = build_mask_manifest(layout)
    validate_mask_manifest(payload)
    assert layout["view_stratified"] is True
    assert layout["action_visible_token_indices_by_view"] == {
        "view0": [0, 1, 2],
        "view1": [3, 4, 5],
    }
    # Six visible tokens and three heads use the registered ceil(n/2) rule.
    for seed in (1, 2, 3):
        token = payload["conditions"][f"token50_seed{seed}"]
        assert token["retained_current_token_count"] == 3
        assert set(token["retained_current_token_indices"]) <= set(range(6))
        assert token["view_stratified"] is True
        assert sorted(token["retained_current_token_quota_by_view"].values()) == [
            1,
            2,
        ]
        head = payload["conditions"][f"head50_seed{seed}"]
        assert set(head["retained_current_heads_by_layer"]) == {
            str(layer) for layer in LATE_LAYERS
        }
        assert all(len(heads) == 2 for heads in head["retained_current_heads_by_layer"].values())

    path = tmp_path / "mask_manifest.json"
    digest = write_or_verify_frozen_manifest(path, layout)
    assert digest == sha256_file(path)
    assert digest == write_or_verify_frozen_manifest(path, layout)
    spec = load_mask_spec(
        path=path, expected_sha256=digest, condition_name="head50_seed2"
    )
    assert spec.axis == "head"
    assert spec.seed == 2
    assert set(spec.retained_current_heads_by_layer or {}) == set(LATE_LAYERS)


def test_frozen_mask_manifest_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "mask_manifest.json"
    path.write_text(json.dumps({"not": "the mask"}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_or_verify_frozen_manifest(path, runtime_layout_from_cache_stats(_cache_stats()))
