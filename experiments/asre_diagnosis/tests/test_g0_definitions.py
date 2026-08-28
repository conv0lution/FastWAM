from pathlib import Path

import pytest

from experiments.asre_diagnosis.common import (
    G0_PROTOCOL,
    build_g0_conditions,
    resolve_condition,
)
from experiments.asre_diagnosis.g0.definitions import (
    CONDITION_ORDER,
    GPU_BY_CONDITION,
    SUITE_ORDER,
    assert_output_scope,
    condition_registration_payload,
    validate_gpu_mapping,
    validate_resume_metadata,
)
from experiments.asre_diagnosis.round3b.donor import donor_trial_for


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_suite_condition_registration_is_exact() -> None:
    assert SUITE_ORDER == ("libero_object", "libero_goal", "libero_10")
    assert CONDITION_ORDER == (
        "full_current",
        "late_current_15_29",
        "early_current_00_19",
        "late_wrong_scene_15_29",
    )
    payload = condition_registration_payload()
    assert payload["protocol"] == G0_PROTOCOL
    assert [row["condition_slot"] for row in payload["conditions"]] == [0, 1, 2, 3]
    assert GPU_BY_CONDITION == dict(zip(CONDITION_ORDER, range(4)))


def test_retrieval_layer_schedules() -> None:
    full, late, early, wrong = build_g0_conditions(30)
    assert full.enabled_video_retrieval_layers(30) == tuple(range(30))
    assert late.enabled_video_retrieval_layers(30) == tuple(range(15, 30))
    assert early.enabled_video_retrieval_layers(30) == tuple(range(20))
    assert wrong.enabled_video_retrieval_layers(30) == tuple(range(15, 30))
    assert wrong.replacement_video_layers == tuple(range(15, 30))
    with pytest.raises(ValueError, match="exactly 30"):
        build_g0_conditions(29)


def test_g0_condition_resolution_is_fail_closed() -> None:
    selected = resolve_condition(
        {
            "enabled": True,
            "mode": "replace_video_kv",
            "protocol": G0_PROTOCOL,
            "condition_name": "early_current_00_19",
            "enabled_video_retrieval_layers": list(range(20)),
            "disabled_video_layers": list(range(20, 30)),
            "replacement_video_layers": [],
        },
        30,
    )
    assert selected.name == "early_current_00_19"
    with pytest.raises(ValueError, match="requires"):
        resolve_condition(
            {
                "enabled": True,
                "mode": "drop_video_kv",
                "protocol": G0_PROTOCOL,
                "condition_name": "early_current_00_19",
                "enabled_video_retrieval_layers": list(range(20)),
                "disabled_video_layers": list(range(20, 30)),
            },
            30,
        )


def test_four_gpu_mapping_is_strict() -> None:
    assert validate_gpu_mapping([0, 1, 2, 3]) == (0, 1, 2, 3)
    assert validate_gpu_mapping([4, 5, 6, 7]) == (4, 5, 6, 7)
    with pytest.raises(ValueError, match="exactly four"):
        validate_gpu_mapping([0, 1, 2])
    with pytest.raises(ValueError, match="distinct"):
        validate_gpu_mapping([0, 1, 1, 3])


def test_donor_mapping_is_same_task_next_trial_derangement() -> None:
    for task_id in range(10):
        donors = [donor_trial_for(trial, 10) for trial in range(10)]
        assert donors == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        assert all(donor != recipient for recipient, donor in enumerate(donors))
        assert len(set(donors)) == 10
        assert task_id == task_id  # mapping never changes the task identity


def test_resume_metadata_validation() -> None:
    expected = {"suite": "libero_goal", "condition": "full_current", "trials": 10}
    validate_resume_metadata(dict(expected), expected)
    with pytest.raises(ValueError, match="Incompatible G0 resume metadata"):
        validate_resume_metadata({**expected, "trials": 2}, expected)


def test_stage1_artifact_overwrite_protection() -> None:
    assert_output_scope(PROJECT_ROOT / "asre_results/g0_cross_suite/libero_goal", PROJECT_ROOT)
    for protected in ("round2", "round3a", "round3b", "state_bank"):
        with pytest.raises(ValueError):
            assert_output_scope(PROJECT_ROOT / "asre_results" / protected, PROJECT_ROOT)
