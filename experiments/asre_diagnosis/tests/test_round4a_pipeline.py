from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.asre_diagnosis.common import ROUND4A_PROTOCOL, build_round4a_conditions
from experiments.asre_diagnosis.round4a.aggregate_results import (
    CONDITIONS,
    validate_paired_outcome_keys,
)
from experiments.asre_diagnosis.round4a.classification import (
    classify_axis,
    recommend_axis,
)
from experiments.asre_diagnosis.round4a.launch_offline_wave import (
    inspect_offline_condition,
    offline_condition_command,
)
from experiments.asre_diagnosis.round4a.launch_wave import (
    Runtime,
    WAVE_CONDITIONS,
    _root_identity,
    condition_command,
    resolve_runtime,
)
from experiments.asre_diagnosis.round4a.provenance import Round4AProvenance
from experiments.asre_diagnosis.round4a.run_round4a import _require_output_scope
from experiments.asre_diagnosis.round3a.launch_three_gpu import DirectorySafetyError


def _provenance(tmp_path: Path) -> Round4AProvenance:
    return Round4AProvenance(
        git_commit_hash="a" * 40,
        checkpoint_path=tmp_path / "libero_uncond_2cam224.pt",
        checkpoint_sha256="1" * 64,
        dataset_stats_path=tmp_path / "stats.json",
        dataset_stats_sha256="2" * 64,
        source_manifest_path=tmp_path / "state_bank/manifest.jsonl",
        source_manifest_sha256="3" * 64,
        valid_manifest_path=tmp_path / "valid.json",
        valid_manifest_sha256="4" * 64,
        prompt_context_cache_path=tmp_path / "prompt.pt",
        prompt_context_cache_sha256="5" * 64,
        online_donor_mapping_path=tmp_path / "donor_mapping.json",
        online_donor_mapping_sha256="6" * 64,
        online_donor_manifest_path=tmp_path / "donor_manifest.json",
        online_donor_manifest_sha256="7" * 64,
        online_donor_root=tmp_path / "donors",
        offline_donor_mapping_path=tmp_path / "offline_donors.json",
        offline_donor_mapping_sha256="8" * 64,
        preflight_report_path=tmp_path / "preflight.json",
        preflight_report_sha256="9" * 64,
        machinery_report_path=tmp_path / "machinery.json",
        machinery_report_sha256="b" * 64,
        mask_manifest_path=tmp_path / "mask.json",
        mask_manifest_sha256="c" * 64,
        token_mask_manifest_path=tmp_path / "token_mask.json",
        token_mask_manifest_sha256="d" * 64,
        head_mask_manifest_path=tmp_path / "head_mask.json",
        head_mask_manifest_sha256="e" * 64,
        round3b_parent_tag="ASRE-round3b-kv-replacement",
        round3b_parent_commit="f" * 40,
        round3b_summary_path=tmp_path / "round3b.json",
        round3b_summary_sha256="0" * 64,
        g0_parent_tag="ASRE-g0-cross-suite",
        g0_parent_commit="1" * 40,
        g0_summary_path=tmp_path / "g0.json",
        g0_summary_sha256="a" * 64,
        valid_sample_count=499,
    )


def _task_rates(value: float) -> dict[int, float]:
    return {task: value for task in range(10)}


def test_four_gpu_two_wave_schedule_is_exact() -> None:
    assert WAVE_CONDITIONS == {1: (0, 1, 2, 5), 2: (3, 4, 6, 7)}
    assert resolve_runtime("full", 1).condition_indices == (0, 1, 2, 5)
    assert resolve_runtime("full", 2).condition_indices == (3, 4, 6, 7)
    assert resolve_runtime("smoke", 1).condition_indices == (0, 1, 2, 5)
    assert resolve_runtime("smoke", 2).condition_indices == (3, 6)


def test_online_resume_identity_ignores_dynamic_free_memory(tmp_path: Path) -> None:
    runtime = Runtime("full", 1, tuple(range(10)), 10, WAVE_CONDITIONS[1])
    base = [
        {
            "index": index,
            "name": "GPU",
            "uuid": f"uuid-{index}",
            "pci_bus_id": f"bus-{index}",
            "driver_version": "1",
            "memory_total_mib": 100,
            "memory_free_mib_at_launch": 90,
        }
        for index in range(4)
    ]
    changed = [dict(record, memory_free_mib_at_launch=10) for record in base]
    kwargs = {
        "runtime": runtime,
        "provenance": _provenance(tmp_path),
        "python_path": Path("/usr/bin/python3"),
        "gpu_ids": (0, 1, 2, 3),
        "stagger_seconds": 0.0,
    }
    first = _root_identity(gpu_inventory=base, **kwargs)
    second = _root_identity(gpu_inventory=changed, **kwargs)
    assert first == second
    assert all(
        "memory_free_mib_at_launch" not in record
        for record in first["gpu_inventory"]
    )


def test_online_and_offline_commands_bind_masks_provenance_and_no_ddp(tmp_path: Path) -> None:
    provenance = _provenance(tmp_path)
    condition = build_round4a_conditions(30)[2]
    runtime = Runtime("full", 1, tuple(range(10)), 10, WAVE_CONDITIONS[1])
    online = condition_command(
        python_path=Path("/usr/bin/python3"),
        condition_index=2,
        condition=condition,
        condition_output=tmp_path / "online",
        runtime=runtime,
        provenance=provenance,
    )
    offline = offline_condition_command(
        python_path=Path("/usr/bin/python3"),
        condition_index=2,
        condition=condition,
        output_root=tmp_path / "offline",
        provenance=provenance,
    )
    for command in (online, offline):
        joined = " ".join(command)
        assert f"ASRE_DIAGNOSIS.protocol={ROUND4A_PROTOCOL}" in joined
        assert "ASRE_DIAGNOSIS.hybrid_axis=head" in joined
        assert "ASRE_DIAGNOSIS.hybrid_mask_seed=1" in joined
        assert "ASRE_DIAGNOSIS.token_mask_manifest_sha256=" + "d" * 64 in joined
        assert "ASRE_DIAGNOSIS.head_mask_manifest_sha256=" + "e" * 64 in joined
        assert "torchrun" not in joined
        assert "WORLD_SIZE" not in joined


def test_paired_trial_alignment_rejects_missing_condition_or_key() -> None:
    keys = [("libero_spatial", task, trial) for task in range(10) for trial in range(10)]
    outcomes = {condition: {key: 1 for key in keys} for condition in CONDITIONS}
    assert validate_paired_outcome_keys(outcomes) == keys
    broken = {condition: dict(values) for condition, values in outcomes.items()}
    broken["token50_seed3"].pop(keys[-1])
    with pytest.raises(ValueError, match="not exactly paired"):
        validate_paired_outcome_keys(broken)
    broken = {condition: dict(values) for condition, values in outcomes.items()}
    broken.pop("head50_seed1")
    with pytest.raises(ValueError, match="exactly"):
        validate_paired_outcome_keys(broken)


def test_axis_classification_thresholds_and_recommendation() -> None:
    current_tasks = _task_rates(0.9)
    strong = classify_axis(
        axis="token",
        mask_success_rates=(0.85, 0.80, 0.85),
        current_success_rate=0.90,
        wrong_success_rate=0.0,
        mask_task_success=[_task_rates(value) for value in (0.85, 0.80, 0.85)],
        current_task_success=current_tasks,
    )
    assert strong["classification"] == "STRONG"
    intermediate = classify_axis(
        axis="head",
        mask_success_rates=(0.55, 0.70, 0.65),
        current_success_rate=0.90,
        wrong_success_rate=0.0,
        mask_task_success=[_task_rates(value) for value in (0.55, 0.70, 0.65)],
        current_task_success=current_tasks,
    )
    assert intermediate["classification"] == "INTERMEDIATE"
    weak = classify_axis(
        axis="head",
        mask_success_rates=(0.02, 0.04, 0.10),
        current_success_rate=0.90,
        wrong_success_rate=0.0,
        mask_task_success=[_task_rates(value) for value in (0.02, 0.04, 0.10)],
        current_task_success=current_tasks,
    )
    assert weak["classification"] == "WEAK"
    recommendation = recommend_axis(strong, weak)
    assert recommendation["recommended_primary_axis"] == "token"
    assert recommendation["next_step_category"] == "cross-suite 50% replication"
    assert recommendation["auto_launch"] is False


def test_repeated_catastrophic_task_collapse_blocks_strong() -> None:
    current = _task_rates(1.0)
    masks = [_task_rates(0.95) for _ in range(3)]
    masks[0][0] = 0.0
    masks[1][0] = 0.0
    result = classify_axis(
        axis="token",
        mask_success_rates=(0.90, 0.90, 0.95),
        current_success_rate=0.95,
        wrong_success_rate=0.0,
        mask_task_success=masks,
        current_task_success=current,
    )
    assert result["repeated_catastrophic_task_collapse"] is True
    assert result["classification"] != "STRONG"


def test_one_broadly_bad_mask_is_not_mislabeled_as_repeated_across_masks() -> None:
    current = _task_rates(1.0)
    masks = [_task_rates(0.95) for _ in range(3)]
    masks[0][0] = 0.0
    masks[0][1] = 0.0
    result = classify_axis(
        axis="token",
        mask_success_rates=(0.90, 0.95, 0.95),
        current_success_rate=0.95,
        wrong_success_rate=0.0,
        mask_task_success=masks,
        current_task_success=current,
    )
    assert result["repeated_catastrophic_task_collapse"] is False


def test_offline_resume_metadata_is_validated_before_resume(tmp_path: Path) -> None:
    provenance = _provenance(tmp_path)
    condition = build_round4a_conditions(30)[0]
    directory = tmp_path / "offline" / condition.name
    directory.mkdir(parents=True)
    metadata = {
        "artifact_type": "asre_round4a_offline_state_bank_replay",
        "condition_protocol": ROUND4A_PROTOCOL,
        "diagnosis_condition": condition.name,
        "git_commit_hash": provenance.git_commit_hash,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "valid_state_bank_manifest_sha256": provenance.valid_manifest_sha256,
        "offline_donor_mapping_sha256": provenance.offline_donor_mapping_sha256,
        "hybrid_axis": None,
        "hybrid_mask_seed": None,
        "hybrid_mask_manifest_sha256": provenance.mask_manifest_sha256,
        "token_mask_manifest_sha256": provenance.token_mask_manifest_sha256,
        "head_mask_manifest_sha256": provenance.head_mask_manifest_sha256,
        "num_valid_samples": 499,
        "executed_prefix_length": 10,
        "status": "running",
        "completed_samples": 0,
    }
    (directory / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert inspect_offline_condition(
        tmp_path / "offline", condition=condition, provenance=provenance
    )[:2] == ("partial", 0)
    metadata["git_commit_hash"] = "bad"
    (directory / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(DirectorySafetyError, match="Incompatible"):
        inspect_offline_condition(
            tmp_path / "offline", condition=condition, provenance=provenance
        )


def test_round4a_output_scope_protects_stage1_and_g0() -> None:
    _require_output_scope(Path("asre_results/round4a/test_retry"))
    with pytest.raises(ValueError, match="must be"):
        _require_output_scope(Path("asre_results/round3b/never"))
    with pytest.raises(ValueError, match="must be"):
        _require_output_scope(Path("asre_results/g0_cross_suite/never"))


def test_round4a_runtime_provenance_fields_are_declared_in_struct_config() -> None:
    project_root = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(project_root / "configs/sim_libero.yaml")
    OmegaConf.set_struct(cfg, True)
    cfg.ASRE_DIAGNOSIS.g0_gate_classification = "g0-strong"
    assert cfg.ASRE_DIAGNOSIS.g0_gate_classification == "g0-strong"
