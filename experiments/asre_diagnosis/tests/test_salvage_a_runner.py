from __future__ import annotations

import pytest

from experiments.asre_diagnosis.common import (
    SALVAGE_A_PROTOCOL,
    build_salvage_a_conditions,
    resolve_condition,
)
from experiments.asre_diagnosis.salvage_a.classification import classify_salvage
from experiments.asre_diagnosis.salvage_a.definitions import (
    CONDITIONS,
    RANKS,
    WAVES,
    validate_frozen_condition_matrix,
)
from experiments.asre_diagnosis.salvage_a.launch_wave import (
    _donor_pairs,
    _split_trials,
    _validate_assignments,
)


def _cfg(index: int) -> dict:
    condition = build_salvage_a_conditions(30)[index]
    return {
        "enabled": True,
        "mode": "replace_video_kv",
        "protocol": SALVAGE_A_PROTOCOL,
        "condition_index": index,
        "condition_name": condition.name,
        "enabled_video_retrieval_layers": list(range(15, 30)),
        "disabled_video_layers": list(range(15)),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "subspace_basis_kind": condition.basis_kind,
        "subspace_rank": condition.subspace_rank,
    }


def _rates(**updates: float) -> dict[str, float]:
    values = {
        "current_all": 0.95,
        "wrong_all": 0.0,
        "svd_r36": 0.0,
        "actionaware_r36": 0.0,
        "random_r36": 0.0,
        "svd_r97": 0.79,
        "actionaware_r97": 0.79,
        "random_r97": 0.0,
    }
    values.update(updates)
    return values


def _split() -> dict:
    return {
        "per_task": [
            {
                "task_id": task,
                "calibration_episode_ids": [0, 2, 4, 6, 8],
                "heldout_episode_ids": [1, 3, 5, 7, 9],
            }
            for task in range(10)
        ]
    }


def _mapping() -> dict:
    records = []
    for task in range(10):
        for group in ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9]):
            for index, recipient in enumerate(group):
                records.append(
                    {
                        "task_id": task,
                        "recipient_trial": recipient,
                        "donor_trial": group[(index + 1) % len(group)],
                    }
                )
    return {"records": records}


def test_salvage_a_condition_matrix_and_waves_are_exactly_frozen() -> None:
    conditions = validate_frozen_condition_matrix()
    assert tuple(item.name for item in conditions) == CONDITIONS
    assert WAVES == {1: (0, 1, 2, 3), 2: (4, 5, 6, 7)}
    assert tuple(item.subspace_rank for item in conditions) == (
        None,
        None,
        36,
        36,
        36,
        97,
        97,
        97,
    )
    assert RANKS == (36, 97)
    for index, condition in enumerate(conditions):
        assert resolve_condition(_cfg(index), 30) == condition


def test_salvage_a_rejects_basis_rank_or_schedule_drift() -> None:
    cfg = _cfg(3)
    cfg["subspace_basis_kind"] = "svd"
    with pytest.raises(ValueError, match="subspace_basis_kind"):
        resolve_condition(cfg, 30)
    cfg = _cfg(6)
    cfg["subspace_rank"] = 96
    with pytest.raises(ValueError, match="subspace_rank"):
        resolve_condition(cfg, 30)
    cfg = _cfg(4)
    cfg["replacement_video_layers"] = list(range(16, 30))
    with pytest.raises(ValueError, match="replacement layers"):
        resolve_condition(cfg, 30)


def test_salvage_a_registered_decision_hierarchy() -> None:
    strong_a = classify_salvage(
        success_rates=_rates(actionaware_r36=0.90), analysis_scope="heldout"
    )
    assert strong_a["classification"] == "STRONG"
    assert strong_a["strong_rule_a_r36"] is True
    strong_b = classify_salvage(
        success_rates=_rates(actionaware_r97=0.94, svd_r97=0.79),
        analysis_scope="heldout",
    )
    assert strong_b["classification"] == "STRONG"
    assert strong_b["strong_rule_b_r97"] is True
    moderate = classify_salvage(
        success_rates=_rates(actionaware_r97=0.84, svd_r97=0.70),
        analysis_scope="heldout",
    )
    assert moderate["classification"] == "MODERATE"
    weak = classify_salvage(success_rates=_rates(), analysis_scope="heldout")
    assert weak["classification"] == "WEAK"
    assert weak["compact_asre_status"] == "closed"
    with pytest.raises(ValueError, match="held-out"):
        classify_salvage(success_rates=_rates(), analysis_scope="all")


def test_split_local_donor_assignment_validation() -> None:
    split_by_trial = _split_trials(_split())
    pairs = _donor_pairs(_mapping())
    current, wrong = build_salvage_a_conditions(30)[:2]
    _validate_assignments(
        [],
        condition=current,
        task_id=0,
        trials=2,
        split_by_trial=split_by_trial,
        donor_pairs=pairs,
    )
    assignments = [
        {
            "recipient_task_id": 0,
            "recipient_trial": trial,
            "donor_task_id": 0,
            "donor_trial": pairs[(0, trial)][1],
            "recipient_first_query_image_verified": True,
        }
        for trial in (0, 1)
    ]
    _validate_assignments(
        assignments,
        condition=wrong,
        task_id=0,
        trials=2,
        split_by_trial=split_by_trial,
        donor_pairs=pairs,
    )
    assignments[0]["donor_trial"] = 1
    with pytest.raises(ValueError, match="drifted"):
        _validate_assignments(
            assignments,
            condition=wrong,
            task_id=0,
            trials=2,
            split_by_trial=split_by_trial,
            donor_pairs=pairs,
        )
