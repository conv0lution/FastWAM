from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.asre_diagnosis.common import (
    ROUND4C_PROTOCOL,
    build_round4c_conditions,
    git_commit,
    resolve_condition,
)
from experiments.asre_diagnosis.round4c.aggregate_results import (
    _write_csv,
    build_curve_and_saturation,
    load_online,
)
from experiments.asre_diagnosis.round4c.classification import (
    classify_action_sufficiency,
)
from experiments.asre_diagnosis.round4c.definitions import (
    CONDITIONS,
    RANKS,
    REGISTERED_HELDOUT_ENERGY,
    WAVES,
    validate_energy_candidates,
)
from experiments.asre_diagnosis.round4c.launch_wave import _complete
from experiments.asre_diagnosis.round4c.plot_results import plot


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _cfg(index: int) -> dict:
    condition = build_round4c_conditions(30)[index]
    return {
        "enabled": True,
        "mode": "replace_video_kv",
        "protocol": ROUND4C_PROTOCOL,
        "condition_index": index,
        "condition_name": condition.name,
        "enabled_video_retrieval_layers": list(range(15, 30)),
        "disabled_video_layers": list(range(15)),
        "replacement_video_layers": list(condition.replacement_video_layers),
        "subspace_basis_kind": condition.basis_kind,
        "subspace_rank": condition.subspace_rank,
    }


def _candidates() -> dict:
    return {
        "candidate_r50": {
            "rank": 36,
            "rank_fraction": 36 / 3072,
            "heldout_global_energy": 0.5047502042862683,
        },
        "candidate_r70": {
            "rank": 97,
            "rank_fraction": 97 / 3072,
            "heldout_global_energy": 0.7002513655553791,
        },
        "candidate_r80": {
            "rank": 170,
            "rank_fraction": 170 / 3072,
            "heldout_global_energy": 0.8004933480103876,
        },
    }


def test_round4c_condition_matrix_and_two_wave_mapping_are_frozen() -> None:
    conditions = build_round4c_conditions(30)
    assert tuple(item.name for item in conditions) == CONDITIONS
    assert tuple(item.subspace_rank for item in conditions) == (None, None, *RANKS)
    assert WAVES == {1: (0, 1, 2, 3), 2: (4,)}
    for index, condition in enumerate(conditions):
        assert resolve_condition(_cfg(index), 30) == condition
        assert condition.disabled_video_layers == tuple(range(15))
        if condition.name == "current_all":
            assert condition.replacement_video_layers == ()
        else:
            assert condition.replacement_video_layers == tuple(range(15, 30))


def test_round4c_rejects_rank_basis_and_schedule_drift() -> None:
    cfg = _cfg(2)
    cfg["subspace_rank"] = 37
    with pytest.raises(ValueError, match="subspace_rank"):
        resolve_condition(cfg, 30)
    cfg = _cfg(3)
    cfg["subspace_basis_kind"] = "random"
    with pytest.raises(ValueError, match="subspace_basis_kind"):
        resolve_condition(cfg, 30)
    cfg = _cfg(4)
    cfg["replacement_video_layers"] = list(range(16, 30))
    with pytest.raises(ValueError, match="replacement layers"):
        resolve_condition(cfg, 30)


def test_registered_energy_labels_are_exactly_provenanced() -> None:
    observed = validate_energy_candidates(_candidates())
    assert tuple(observed) == RANKS
    assert all(abs(observed[rank] - REGISTERED_HELDOUT_ENERGY[rank]) < 5e-4 for rank in RANKS)
    drifted = _candidates()
    drifted["candidate_r70"]["heldout_global_energy"] = 0.71
    with pytest.raises(ValueError, match="energy drifted"):
        validate_energy_candidates(drifted)
    adaptive = _candidates()
    adaptive["candidate_extra"] = {"rank": 120}
    with pytest.raises(ValueError, match="exactly"):
        validate_energy_candidates(adaptive)


def _task_success(current: float, r36: float, r97: float, r170: float):
    return {
        "current_all": {task: current for task in range(10)},
        "wrong_all": {task: 0.0 for task in range(10)},
        "svd_r36": {task: r36 for task in range(10)},
        "svd_r97": {task: r97 for task in range(10)},
        "svd_r170": {task: r170 for task in range(10)},
    }


def test_round4c_classification_thresholds_are_frozen() -> None:
    strong_a = {
        "current_all": 0.95,
        "wrong_all": 0.0,
        "svd_r36": 0.85,
        "svd_r97": 0.70,
        "svd_r170": 0.80,
    }
    assert classify_action_sufficiency(
        success_rates=strong_a,
        task_success=_task_success(0.95, 0.85, 0.70, 0.80),
        round4b_r256_success=0.95,
    )["classification"] == "STRONG"
    strong_b = dict(strong_a, svd_r36=0.40, svd_r97=0.91)
    assert classify_action_sufficiency(
        success_rates=strong_b,
        task_success=_task_success(0.95, 0.40, 0.91, 0.90),
        round4b_r256_success=0.95,
    )["classification"] == "STRONG"
    moderate = dict(strong_a, svd_r36=0.20, svd_r97=0.80, svd_r170=0.88)
    assert classify_action_sufficiency(
        success_rates=moderate,
        task_success=_task_success(0.95, 0.20, 0.80, 0.88),
        round4b_r256_success=0.95,
    )["classification"] == "MODERATE"
    weak = dict(strong_a, svd_r36=0.02, svd_r97=0.40, svd_r170=0.70)
    decision = classify_action_sufficiency(
        success_rates=weak,
        task_success=_task_success(0.95, 0.02, 0.40, 0.70),
        round4b_r256_success=0.95,
    )
    assert decision["classification"] == "WEAK"
    assert decision["explicit_weak_manifold_tracking_rule"] is True


def test_repeated_catastrophic_task_collapse_blocks_strong_rule() -> None:
    rates = {
        "current_all": 0.95,
        "wrong_all": 0.0,
        "svd_r36": 0.86,
        "svd_r97": 0.70,
        "svd_r170": 0.70,
    }
    tasks = _task_success(0.95, 0.86, 0.70, 0.70)
    tasks["svd_r36"][0] = 0.0
    tasks["svd_r97"][0] = 0.0
    decision = classify_action_sufficiency(
        success_rates=rates, task_success=tasks, round4b_r256_success=0.95
    )
    assert decision["repeated_catastrophic_task_collapse"] is True
    assert decision["classification"] == "WEAK"


def _write_result(path: Path, *, task_id: int, successes: list[int], trials: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = sorted(set(range(trials)) - set(successes))
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "task_description": f"task {task_id}",
                "success_episodes": successes,
                "failure_episodes": failures,
            }
        ),
        encoding="utf-8",
    )


def _write_condition_metadata(root: Path, condition: str) -> None:
    rank = None if not condition.startswith("svd_r") else int(condition.removeprefix("svd_r"))
    replacement = [] if condition == "current_all" else list(range(15, 30))
    path = root / condition / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    shared_identity = {
        key: "x"
        for key in (
            "checkpoint_sha256",
            "dataset_stats_sha256",
            "state_bank_manifest_sha256",
            "valid_state_bank_manifest_sha256",
            "prompt_context_cache_sha256",
            "donor_mapping_sha256",
            "donor_observation_manifest_sha256",
            "preflight_report_sha256",
            "machinery_report_sha256",
            "calibration_split_manifest_sha256",
            "subspace_basis_manifest_sha256",
            "subspace_diagnostics_sha256",
            "energy_analysis_manifest_sha256",
            "energy_candidate_ranks_sha256",
            "round4b_summary_sha256",
            "round4b_source_commit",
            "cumulative_energy_analysis_commit",
        )
    }
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "condition_protocol": ROUND4C_PROTOCOL,
                "diagnosis_condition": condition,
                "git_commit_hash": git_commit(PROJECT_ROOT),
                "task_suite": "libero_spatial",
                "task_ids": list(range(10)),
                "number_of_trials": 10,
                "seed": 42,
                "action_horizon": 32,
                "number_of_inference_steps": 10,
                "replan_steps": 10,
                "subspace_basis_kind": "svd" if rank is not None else None,
                "subspace_rank": rank,
                **shared_identity,
                "condition_config": {
                    "subspace_basis_kind": "svd" if rank is not None else None,
                    "subspace_rank": rank,
                    "disabled_video_layers": list(range(15)),
                    "replacement_video_layers": replacement,
                },
            }
        ),
        encoding="utf-8",
    )


def test_safe_resume_and_paired_alignment(tmp_path: Path) -> None:
    wave1 = tmp_path / "wave1"
    wave2 = tmp_path / "wave2"
    for wave, root in ((1, wave1), (2, wave2)):
        root.mkdir(parents=True)
        (root / "launcher_summary.json").write_text(
            json.dumps(
                {
                    "protocol": ROUND4C_PROTOCOL,
                    "mode": "full",
                    "wave": wave,
                    "all_succeeded": True,
                }
            ),
            encoding="utf-8",
        )
    for index, condition in enumerate(CONDITIONS):
        root = wave1 if index in WAVES[1] else wave2
        _write_condition_metadata(root, condition)
        for task_id in range(10):
            successes = [trial for trial in range(10) if (trial + task_id + index) % 3]
            _write_result(
                root / condition / "libero_spatial" / f"gpu0_task{task_id}_results.json",
                task_id=task_id,
                successes=successes,
                trials=10,
            )
    assert _complete(wave1, index=0, task_ids=tuple(range(10)), trials=10)
    outcomes, rows = load_online(wave1, wave2)
    assert len(rows) == 50
    assert all(len(outcomes[condition]) == 100 for condition in CONDITIONS)
    metadata = wave1 / "current_all/run_metadata.json"
    payload = json.loads(metadata.read_text())
    payload["seed"] = 43
    metadata.write_text(json.dumps(payload))
    assert not _complete(wave1, index=0, task_ids=tuple(range(10)), trials=10)


def test_curve_keeps_round4b_reference_separate_and_uses_frozen_energy() -> None:
    online = []
    rates = {"current_all": 0.95, "wrong_all": 0.0, "svd_r36": 0.8, "svd_r97": 0.9, "svd_r170": 0.94}
    for condition in CONDITIONS:
        online.append(
            {
                "condition": condition,
                "successes": round(100 * rates[condition]),
                "episodes": 100,
                "success_rate": rates[condition],
                "paired_ci_low": rates[condition],
                "paired_ci_high": rates[condition],
            }
        )
    round4b = {
        "current_all": {"success_rate": 0.95, "paired_ci_low": 0.9, "paired_ci_high": 0.99},
        "wrong_all": {"success_rate": 0.0, "paired_ci_low": 0.0, "paired_ci_high": 0.0},
        "svd_r256": {
            "successes": 95,
            "episodes": 100,
            "success_rate": 0.95,
            "paired_ci_low": 0.9,
            "paired_ci_high": 0.99,
        },
    }
    curve, saturation, reproducibility = build_curve_and_saturation(
        online, round4b, _candidates()
    )
    assert [row["rank"] for row in curve] == [0, 36, 97, 170, 256, 3072]
    r256 = next(row for row in curve if row["rank"] == 256)
    assert r256["newly_run_round4c"] is False
    assert r256["source_round"] == "Round 4B frozen reference"
    assert len(saturation) == 4
    assert len(reproducibility) == 2


def test_round4c_registered_figures_render_from_aggregate_tables(tmp_path: Path) -> None:
    online = []
    rates = {
        "current_all": 0.95,
        "wrong_all": 0.0,
        "svd_r36": 0.80,
        "svd_r97": 0.90,
        "svd_r170": 0.94,
    }
    for condition in CONDITIONS:
        online.append(
            {
                "condition": condition,
                "successes": round(100 * rates[condition]),
                "episodes": 100,
                "success_rate": rates[condition],
                "paired_ci_low": max(0.0, rates[condition] - 0.05),
                "paired_ci_high": min(1.0, rates[condition] + 0.05),
            }
        )
    round4b = {
        "current_all": {"success_rate": 0.95, "paired_ci_low": 0.90, "paired_ci_high": 0.99},
        "wrong_all": {"success_rate": 0.0, "paired_ci_low": 0.0, "paired_ci_high": 0.0},
        "svd_r256": {
            "successes": 95,
            "episodes": 100,
            "success_rate": 0.95,
            "paired_ci_low": 0.90,
            "paired_ci_high": 0.99,
        },
    }
    curve, saturation, _ = build_curve_and_saturation(online, round4b, _candidates())
    tasks = [
        {
            "condition": condition,
            "task_id": task,
            "success_rate": rates[condition],
        }
        for condition in CONDITIONS
        for task in range(10)
    ]
    _write_csv(tmp_path / "energy_success_curve.csv", curve)
    _write_csv(tmp_path / "action_saturation.csv", saturation)
    _write_csv(tmp_path / "task_success.csv", tasks)
    paths = plot(tmp_path)
    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
