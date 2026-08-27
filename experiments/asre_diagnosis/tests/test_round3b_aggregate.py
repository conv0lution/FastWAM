from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.asre_diagnosis.common import (
    ROUND3B_PROTOCOL,
    build_round3b_conditions,
    sha256_file,
)
from experiments.asre_diagnosis.round3b.aggregate_results import (
    CONDITION_ORDER,
    load_online_results,
    summarize_kv_stats,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _cache_tensor_stats(value: float) -> dict:
    return {
        "shape": [1, 2, 3],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "numel": 6,
        "finite": True,
        "mean": value,
        "std": abs(value) + 1.0,
        "rms": abs(value) + 2.0,
    }


def test_kv_summary_checks_and_aggregates_replacement_layers() -> None:
    records = []
    for sample_index, sample_id in enumerate(("a", "b")):
        layers = []
        for layer in range(30):
            summarized = layer >= 15
            layers.append(
                {
                    "layer": layer,
                    "selected_source": "replacement" if summarized else "disabled",
                    "current": (
                        {
                            "k": _cache_tensor_stats(1.0 + sample_index),
                            "v": _cache_tensor_stats(2.0 + sample_index),
                        }
                        if summarized
                        else None
                    ),
                    "replacement": (
                        {
                            "k": _cache_tensor_stats(3.0 + sample_index),
                            "v": _cache_tensor_stats(4.0 + sample_index),
                        }
                        if summarized
                        else None
                    ),
                    "difference": None,
                }
            )
        records.append({"sample_id": sample_id, "layers": layers})

    rows = summarize_kv_stats(records, expected_sample_ids=["a", "b"])
    assert len(rows) == 30
    first_k = rows[0]
    assert first_k["layer"] == 15
    assert first_k["tensor"] == "K"
    assert first_k["current_mean"] == pytest.approx(1.5)
    assert first_k["donor_mean"] == pytest.approx(3.5)
    assert first_k["donor_current_mean_ratio"] == pytest.approx(3.5 / 1.5)


def _metadata(condition: str, donor_mapping_sha: str, donor_manifest_sha: str) -> dict:
    configs = {value.name: value.to_dict() for value in build_round3b_conditions(30)}
    config = configs[condition]
    payload = {
        "status": "completed",
        "diagnosis_condition": condition,
        "condition_protocol": ROUND3B_PROTOCOL,
        "num_model_layers": 30,
        "number_of_trials": 10,
        "seed": 42,
        "action_horizon": 32,
        "number_of_inference_steps": 10,
        "replan_steps": 10,
        "disabled_video_layers": config["disabled_video_layers"],
        "replacement_video_layers": config["replacement_video_layers"],
        "git_commit_hash": "abc",
        "checkpoint_sha256": "checkpoint",
        "dataset_stats_sha256": "stats",
        "state_bank_manifest_sha256": "source",
        "valid_state_bank_manifest_sha256": "valid",
        "prompt_context_cache_sha256": "prompt",
        "task_suite": "libero_spatial",
        "task_ids": list(range(10)),
        "compile_action_infer": True,
        "binarize_gripper": True,
        "sigma_shift": None,
        "rand_device": "cpu",
    }
    payload["donor_mapping_sha256"] = donor_mapping_sha
    payload["donor_observation_manifest_sha256"] = donor_manifest_sha
    return payload


def test_online_loader_requires_exact_paired_100_episode_protocol(tmp_path: Path) -> None:
    mapping = tmp_path / "donor_mapping.json"
    manifest = tmp_path / "donor_manifest.json"
    _write_json(mapping, {"mapping": "fixed"})
    _write_json(manifest, {"observations": "fixed"})
    mapping_sha = sha256_file(mapping)
    manifest_sha = sha256_file(manifest)
    online = tmp_path / "online"
    _write_json(
        online / "launcher_summary.json",
        {
            "schema_version": 1,
            "protocol": ROUND3B_PROTOCOL,
            "mode": "full",
            "all_succeeded": True,
            "interrupted": False,
            "task_ids": list(range(10)),
            "num_trials": 10,
            "seed": 42,
            "git_commit_hash": "abc",
            "expected_conditions": list(CONDITION_ORDER),
            "condition_states": {
                condition: {
                    "state": "complete",
                    "completed_task_ids": list(range(10)),
                }
                for condition in CONDITION_ORDER
            },
        },
    )
    for condition in CONDITION_ORDER:
        condition_dir = online / condition
        _write_json(
            condition_dir / "run_metadata.json",
            _metadata(condition, mapping_sha, manifest_sha),
        )
        for task_id in range(10):
            successes = list(range((task_id + len(condition)) % 11))
            successes = [value for value in successes if value < 10]
            failures = [value for value in range(10) if value not in successes]
            _write_json(
                condition_dir
                / "libero_spatial"
                / f"gpu0_task{task_id}_results.json",
                {
                    "task_suite": "libero_spatial",
                    "task_id": task_id,
                    "task_description": f"task {task_id}",
                    "successes": len(successes),
                    "total_episodes": 10,
                    "success_episodes": successes,
                    "failure_episodes": failures,
                    "diagnosis_condition": condition,
                    "condition_protocol": ROUND3B_PROTOCOL,
                },
            )

    outcomes, metadata, descriptions = load_online_results(
        online_root=online,
        donor_mapping_path=mapping,
        donor_manifest_path=manifest,
    )
    assert set(outcomes) == set(CONDITION_ORDER)
    assert all(len(values) == 100 for values in outcomes.values())
    assert set(metadata) == set(CONDITION_ORDER)
    assert descriptions[9] == "task 9"

    broken = online / "late_wrong_scene/libero_spatial/gpu0_task0_results.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["failure_episodes"].pop()
    _write_json(broken, payload)
    with pytest.raises(ValueError, match="partition"):
        load_online_results(
            online_root=online,
            donor_mapping_path=mapping,
            donor_manifest_path=manifest,
        )
