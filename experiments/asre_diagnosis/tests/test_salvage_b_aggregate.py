from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.asre_diagnosis.common import sha256_file, sha256_json
from experiments.asre_diagnosis.salvage_b.aggregate_final import (
    FINAL_COMPLETION_FILENAME,
    _recommendation,
    aggregate_final,
    write_special_report,
)
from experiments.asre_diagnosis.salvage_b.aggregate_world import aggregate_phase
from experiments.asre_diagnosis.salvage_b.plots import generate_plots


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _endpoint_stop_artifacts(tmp_path: Path, *, commit: str) -> dict[str, Path]:
    architecture = tmp_path / "stop_architecture.json"
    _json(
        architecture,
        {
            "artifact_type": "asre_salvage_b_phase_a_architecture_audit",
            "git_commit_hash": commit,
            "status": "preferred_path_a_pending_runtime_machinery",
            "static_audit_passed": True,
            "path": "preferred_path_a",
        },
    )
    preflight = tmp_path / "stop_preflight.json"
    _json(
        preflight,
        {
            "artifact_type": "asre_salvage_b_preflight_report",
            "protocol": "salvage_b_world_action_functional_dissociation",
            "git_commit_hash": commit,
            "status": "compatible",
            "phase_a": {
                "architecture_audit_path": str(architecture.resolve()),
                "architecture_audit_sha256": sha256_file(architecture),
            },
        },
    )
    machinery = tmp_path / "stop_machinery.json"
    _json(
        machinery,
        {
            "artifact_type": "asre_salvage_b_shared_interface_machinery_report",
            "protocol": "salvage_b_world_action_functional_dissociation",
            "git_commit_hash": commit,
            "status": "passed",
            "passed": True,
            "phase_b_authorized": True,
            "recommended_special_classification": None,
            "preflight_report_sha256": sha256_file(preflight),
            "inputs": {
                "preflight": {
                    "path": str(preflight.resolve()),
                    "sha256": sha256_file(preflight),
                }
            },
        },
    )
    endpoint = tmp_path / "stop_endpoint.json"
    endpoint_gate = {
        "status": "failed",
        "passed": False,
        "classification": "WORLD-ENDPOINT-UNINFORMATIVE",
        "machinery_sha256": sha256_file(machinery),
        "git_commit_hash": commit,
    }
    _json(
        endpoint,
        {
            "artifact_type": "asre_salvage_b_world_phase_aggregate",
            "protocol": "salvage_b_world_action_functional_dissociation",
            "git_commit_hash": commit,
            "phase": "endpoint",
            "conditions": ["current_all", "wrong_all"],
            "status": "failed",
            "passed": False,
            "classification": "WORLD-ENDPOINT-UNINFORMATIVE",
            "preflight_report_sha256": sha256_file(preflight),
            "machinery_sha256": sha256_file(machinery),
            "endpoint_gate": endpoint_gate,
        },
    )
    return {
        "architecture_audit_path": architecture,
        "preflight_path": preflight,
        "machinery_path": machinery,
        "endpoint_summary_path": endpoint,
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    commit = "a" * 40
    preflight = tmp_path / "preflight.json"
    _json(
        preflight,
        {
            "artifact_type": "asre_salvage_b_preflight_report",
            "status": "compatible",
            "git_commit_hash": commit,
        },
    )
    preflight_sha = sha256_file(preflight)
    world = tmp_path / "world.json"
    world_records = [
        {
            "sample_id": f"sample_{index:03d}",
            "task_id": index // 10,
            "episode_id": 1000 + index,
            "trial": index % 10,
            "donor_processed_image_sha256": "d" * 64,
        }
        for index in range(100)
    ]
    _json(
        world,
        {
            "artifact_type": "asre_salvage_b_world_evaluation_manifest",
            "schema_version": 2,
            "status": "frozen_before_metrics",
            "sample_count": 100,
            "git_commit_hash": commit,
            "preflight_report_sha256": preflight_sha,
            "native_world_metric": {
                "name": "pure_noise_native_future_latent_reconstruction_mse",
                "direction": "lower_is_better",
                "inference_steps": 10,
                "inference_shift": 5.0,
                "target_usage": "scoring_only_after_inference",
            },
            "records": world_records,
        },
    )
    world_sha = sha256_file(world)
    stochastic = tmp_path / "stochastic.json"
    stochastic_records = []
    for record in world_records:
        for draw_id in range(4):
            stochastic_records.append(
                {
                    "sample_id": record["sample_id"],
                    "draw_id": draw_id,
                    "video_noise_sha256": f"{draw_id + 1:064x}",
                }
            )
    _json(
        stochastic,
        {
            "artifact_type": "asre_salvage_b_stochastic_manifest",
            "schema_version": 2,
            "status": "frozen_before_metrics",
            "sample_count": 100,
            "draws_per_sample": 4,
            "native_world_metric": "pure_noise_native_future_latent_reconstruction_mse",
            "video_inference_steps": 10,
            "video_inference_shift": 5.0,
            "git_commit_hash": commit,
            "preflight_report_sha256": preflight_sha,
            "world_manifest_sha256": world_sha,
            "records": stochastic_records,
        },
    )
    target = tmp_path / "targets.json"
    _json(
        target,
        {
            "artifact_type": "asre_salvage_b_processed_world_target_manifest",
            "schema_version": 2,
            "status": "frozen_before_gpu_metrics",
            "sample_count": 100,
            "git_commit_hash": commit,
            "preflight_report_sha256": preflight_sha,
            "world_manifest_sha256": world_sha,
            "source_artifact_verification": {
                "before_decode": True,
                "after_decode": True,
                "metadata_file_count": 0,
                "target_source_file_count": 0,
                "checks": "resolved regular file, byte size, and SHA-256",
            },
            "records": [
                {
                    **record,
                    "processed_tensor_sha256": {"video": "f" * 64},
                    "current_image_sha256": "c" * 64,
                }
                for record in world_records
            ],
        },
    )
    machinery = tmp_path / "machinery.json"
    _json(machinery, {"passed": True})
    hashes = {
        "world_manifest_sha256": sha256_file(world),
        "stochastic_manifest_sha256": sha256_file(stochastic),
        "target_manifest_sha256": sha256_file(target),
        "machinery_sha256": sha256_file(machinery),
        "git_commit": commit,
    }
    phase = tmp_path / "endpoint"
    launcher_config = {
        "schema_version": 2,
        "phase": "endpoint",
        "conditions": ["current_all", "wrong_all"],
        "native_metric": "pure_noise_native_future_latent_reconstruction_mse",
        "inference_steps": 10,
        "inference_shift": 5.0,
        "world_manifest_sha256": hashes["world_manifest_sha256"],
        "stochastic_manifest_sha256": hashes["stochastic_manifest_sha256"],
        "target_manifest_sha256": hashes["target_manifest_sha256"],
        "machinery_sha256": hashes["machinery_sha256"],
        "git_commit_hash": commit,
        "endpoint_gate_sha256": None,
    }
    launcher_config["identity_sha256"] = sha256_json(launcher_config)
    launcher_config_path = phase / "launcher_config.json"
    _json(launcher_config_path, launcher_config)
    launcher_config_sha256 = sha256_file(launcher_config_path)
    for worker_index in range(4):
        directory = phase / f"worker_{worker_index:02d}"
        directory.mkdir(parents=True)
        assigned = [
            record for index, record in enumerate(world_records) if index % 4 == worker_index
        ]
        rows = []
        for record in assigned:
            for condition in ("current_all", "wrong_all"):
                for draw_id in range(4):
                    latent = 1.0 if condition == "current_all" else 2.0
                    rows.append(
                        {
                            "schema_version": 2,
                            "protocol": "salvage_b_world_action_functional_dissociation",
                            "phase": "endpoint",
                            "worker_index": worker_index,
                            "sample_id": record["sample_id"],
                            "task_id": record["task_id"],
                            "episode_id": record["episode_id"],
                            "trial_index": record["trial"],
                            "draw_id": draw_id,
                            "condition": condition,
                            "rank": 3072 if condition == "current_all" else 0,
                            "native_world_loss": latent,
                            "future_latent_mse": latent,
                            "inference_steps": 10,
                            "inference_shift": 5.0,
                            "target_sha256": "f" * 64,
                            "target_latent_sha256": "b" * 64,
                            "current_image_sha256": "c" * 64,
                            "current_frame_latent_sha256": "e" * 64,
                            "donor_image_sha256": "d" * 64,
                            "video_noise_sha256": f"{draw_id + 1:064x}",
                            "prediction_shape": [1, 48, 2, 14, 28],
                            "target_shape": [1, 48, 2, 14, 28],
                            "future_token_shape": [1, 196, 3072],
                            **hashes,
                        }
                    )
        metadata = {
            "schema_version": 2,
            "status": "completed",
            "phase": "endpoint",
            "worker_index": worker_index,
            "sample_count": 25,
            "record_count": 200,
            "conditions": ["current_all", "wrong_all"],
            "draws_per_sample": 4,
            "launcher_config_sha256": launcher_config_sha256,
            "endpoint_gate_sha256": None,
            "sample_ids_sha256": sha256_json(
                [record["sample_id"] for record in assigned]
            ),
            "native_metric": "pure_noise_native_future_latent_reconstruction_mse",
            "inference_steps": 10,
            "inference_shift": 5.0,
            "initial_state": "pure_gaussian_future_latent_noise",
            "target_usage": "scoring_only_after_inference",
            "no_ddp": True,
            "provenance": hashes,
        }
        rows_path = directory / "rows.jsonl"
        rows_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        metadata["rows_sha256"] = sha256_file(rows_path)
        _json(directory / "metadata.json", metadata)
    return {
        "phase": phase,
        "world": world,
        "stochastic": stochastic,
        "target": target,
        "machinery": machinery,
        "preflight": preflight,
    }


def test_endpoint_aggregate_strict_pairing_and_gate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "aggregate"
    result = aggregate_phase(
        phase="endpoint",
        phase_root=fixture["phase"],
        world_manifest_path=fixture["world"],
        stochastic_manifest_path=fixture["stochastic"],
        target_manifest_path=fixture["target"],
        machinery_path=fixture["machinery"],
        preflight_path=fixture["preflight"],
        output_dir=output,
        bootstrap_samples=100,
        bootstrap_seed=5,
    )
    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["endpoint_gate"]["mean_wrong_minus_current"] == 1.0
    assert result["endpoint_gate"]["paired_bootstrap_ci_low"] == 1.0
    assert result["endpoint_gate"]["gate"]["median_descriptive_only"] is True
    assert len(result["sample_results"]) == 200
    assert len(result["sample_identity"]) == 100
    assert result["sample_identity_sha256"] == result["endpoint_gate"][
        "sample_identity_sha256"
    ]
    assert (output / "world_sample_identity.csv").is_file()
    assert (output / "endpoint_world_summary.json").is_file()
    assert (output / "world_endpoint_gate.json").is_file()


def test_endpoint_aggregate_rejects_duplicate_pair(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows_path = fixture["phase"] / "worker_00/rows.jsonl"
    lines = rows_path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[0]
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata_path = fixture["phase"] / "worker_00/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["rows_sha256"] = sha256_file(rows_path)
    _json(metadata_path, metadata)
    with pytest.raises(ValueError, match="paired"):
        aggregate_phase(
            phase="endpoint",
            phase_root=fixture["phase"],
            world_manifest_path=fixture["world"],
            stochastic_manifest_path=fixture["stochastic"],
            target_manifest_path=fixture["target"],
            machinery_path=fixture["machinery"],
            preflight_path=fixture["preflight"],
            output_dir=tmp_path / "aggregate",
            bootstrap_samples=20,
        )


def test_endpoint_aggregate_rejects_target_hash_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rows_path = fixture["phase"] / "worker_00/rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["target_sha256"] = "0" * 64
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata_path = fixture["phase"] / "worker_00/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["rows_sha256"] = sha256_file(rows_path)
    _json(metadata_path, metadata)
    with pytest.raises(ValueError, match="processed video target hash drifted"):
        aggregate_phase(
            phase="endpoint",
            phase_root=fixture["phase"],
            world_manifest_path=fixture["world"],
            stochastic_manifest_path=fixture["stochastic"],
            target_manifest_path=fixture["target"],
            machinery_path=fixture["machinery"],
            preflight_path=fixture["preflight"],
            output_dir=tmp_path / "aggregate",
            bootstrap_samples=20,
        )


def test_special_failure_report_never_claims_phase_b(tmp_path: Path) -> None:
    commit = "a" * 40
    summary = write_special_report(
        classification="WORLD-ENDPOINT-UNINFORMATIVE",
        reason="The paired endpoint interval included zero.",
        output_dir=tmp_path,
        git_commit_hash=commit,
        **_endpoint_stop_artifacts(tmp_path, commit=commit),
    )
    assert summary["status"] == "stopped"
    assert summary["phase_b_projected_conditions_run"] is False
    assert summary["later_stage_launched"] is False
    text = (tmp_path / "result_summary_for_gpt.md").read_text(encoding="utf-8")
    assert "projected r97/r170 did not launch" in text
    assert "No Salvage C" in text


@pytest.mark.parametrize("classification", ("MODERATE", "WEAK"))
def test_non_strong_recommendation_uses_registered_close_language(
    classification: str,
) -> None:
    _, recommendation = _recommendation(classification)
    assert recommendation == "Paper 1 should be frozen/closed"


def test_registered_plot_bundle_contains_exactly_four_figures(tmp_path: Path) -> None:
    conditions = ("current_all", "wrong_all", "svd_r97", "svd_r170")
    recoveries = []
    world = []
    action = []
    for index, condition in enumerate(conditions):
        ar = (1.0, 0.0, 0.8, 1.0)[index]
        wr = (1.0, 0.0, 0.5, 0.7)[index]
        recoveries.append(
            {
                "condition": condition,
                "action_recovery": ar,
                "action_recovery_paired_ci_low": ar,
                "action_recovery_paired_ci_high": ar,
                "world_recovery": wr,
                "world_recovery_paired_ci_low": wr,
                "world_recovery_paired_ci_high": wr,
            }
        )
        loss = (1.0, 3.0, 2.0, 1.6)[index]
        world.append(
            {
                "condition": condition,
                "mean_native_world_loss": loss,
                "mean_loss_paired_ci_low": loss,
                "mean_loss_paired_ci_high": loss,
            }
        )
        success = (0.95, 0.0, 0.79, 0.96)[index]
        action.append(
            {
                "condition": condition,
                "success_rate": success,
                "paired_ci_low": success,
                "paired_ci_high": success,
            }
        )
    summary = {
        "protocol": "salvage_b_world_action_functional_dissociation",
        "status": "complete",
        "classification": {"classification": "STRONG"},
        "functional_recovery": recoveries,
        "world_condition_summary": world,
        "action_condition_summary": action,
    }
    manifest = generate_plots(summary, output_dir=tmp_path)
    assert len(manifest["figures"]) == 4
    assert manifest["primary_figure"] == "figure_A"
    assert all(Path(row["path"]).is_file() for row in manifest["figures"])


def test_normal_final_aggregate_reuses_action_and_emits_report(tmp_path: Path) -> None:
    commit = "a" * 40
    world_manifest = tmp_path / "world.json"
    stochastic_manifest = tmp_path / "stochastic.json"
    target_manifest = tmp_path / "target.json"
    machinery = tmp_path / "machinery.json"
    architecture = tmp_path / "architecture.json"
    draw_tensors = tmp_path / "fixed_draw_tensors.pt"
    draw_tensors.write_bytes(b"fixed draws")
    frozen_files: dict[str, Path] = {}
    for name in (
        "checkpoint",
        "valid_manifest",
        "source_manifest",
        "dataset_stats",
        "prompt_context_cache",
        "basis",
        "split",
        "donor_mapping",
        "donor_manifest",
        "salvage_a_summary",
    ):
        path = tmp_path / f"frozen/{name}.json"
        _json(path, {"name": name})
        frozen_files[name] = path
    dataset_root = tmp_path / "official_dataset"
    dataset_root.mkdir()
    dataset_info = dataset_root / "info.json"
    dataset_tasks = dataset_root / "tasks.json"
    _json(dataset_info, {"dataset": "official"})
    _json(dataset_tasks, {"tasks": 10})
    donor_root = tmp_path / "donors"
    donor_root.mkdir()
    action_root = tmp_path / "actions"
    action_root.mkdir()
    architecture.with_suffix(".md").write_text("architecture\n", encoding="utf-8")
    machinery.with_suffix(".md").write_text("machinery\n", encoding="utf-8")
    (tmp_path / "world_bundle_complete.json").write_text("{}\n", encoding="utf-8")
    _json(tmp_path / "driver_status.json", {"status": "running"})
    for path, payload in (
        (
            world_manifest,
            {
                "kind": "world",
                "draw_tensor_path": str(draw_tensors),
                "draw_tensor_sha256": sha256_file(draw_tensors),
                "basis_fit_namespace": "online_libero_spatial_state_bank",
                "world_evaluation_namespace": "official_lerobot_v30_trajectory",
                "source_identifier_overlap_with_basis_fit": [],
                "overlap_check_scope": (
                    "source namespaces and exact artifacts; not a cross-dataset "
                    "semantic episode-identity proof"
                ),
                "basis_fit_overlap_evidence": "separate frozen source artifacts",
            },
        ),
        (stochastic_manifest, {"kind": "stochastic"}),
        (target_manifest, {"kind": "target"}),
        (machinery, {"passed": True}),
        (
            architecture,
            {
                "static_audit_passed": True,
                "path": "preferred_path_a",
                "shared_interface": {"name": "late_first_frame_video_kv_prefix"},
                "tensor_flow": ["shared cache -> action", "shared cache -> future video"],
                "eligibility_criteria": {"same_tensor_intervention_supported": True},
                "source_locations": {
                    "world_cache_consumer": {
                        "path": "mot.py",
                        "class": "MoT",
                        "function": "forward_future_video_with_video_cache_tensor",
                        "line": 1,
                    }
                },
                "native_metric": {
                    "name": "pure_noise_native_future_latent_reconstruction_mse",
                    "lower_is_better": True,
                },
                "native_training_objective": {
                    "source": "FastWAM.training_loss",
                    "construction": "native flow matching",
                    "not_used_as_primary_here_because": "teacher-forced target bypass",
                },
            },
        ),
    ):
        _json(path, payload)

    successes = {
        "current_all": [10] * 9 + [5],
        "wrong_all": [0] * 10,
        "svd_r97": [10] * 7 + [9, 0, 0],
        "svd_r170": [10] * 9 + [6],
    }
    action_artifacts = []
    action_rows = []
    for condition, task_counts in successes.items():
        task_results = []
        metadata_path = action_root / condition / "run_metadata.json"
        _json(metadata_path, {"condition": condition, "status": "completed"})
        for task_id, count in enumerate(task_counts):
            path = tmp_path / f"actions/{condition}/task{task_id}.json"
            _json(
                path,
                {
                    "task_id": task_id,
                    "success_episodes": list(range(count)),
                    "failure_episodes": list(range(count, 10)),
                },
            )
            task_results.append({"path": str(path), "sha256": sha256_file(path)})
        total = sum(task_counts)
        action_artifacts.append(
            {
                "condition": condition,
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "task_results": task_results,
            }
        )
        action_rows.append(
            {
                "condition": condition,
                "successes": total,
                "episodes": 100,
                "success_rate": total / 100,
                "paired_ci_low": max(0.0, total / 100 - 0.05),
                "paired_ci_high": min(1.0, total / 100 + 0.05),
                "task_hierarchical_ci_low": max(0.0, total / 100 - 0.1),
                "task_hierarchical_ci_high": min(1.0, total / 100 + 0.1),
            }
        )
    action_summary = tmp_path / "round4c_summary.json"
    _json(action_summary, {"online_condition_summary": action_rows})
    preflight = tmp_path / "preflight.json"
    _json(
        preflight,
        {
            "status": "compatible",
            "git_commit_hash": commit,
            "git": {"branch": "test", "round4c_execution_commit": "b" * 40},
            "state": {
                "checkpoint_path": str(frozen_files["checkpoint"]),
                "checkpoint_sha256": sha256_file(frozen_files["checkpoint"]),
                "valid_manifest_path": str(frozen_files["valid_manifest"]),
                "valid_manifest_sha256": sha256_file(frozen_files["valid_manifest"]),
                "source_manifest_path": str(frozen_files["source_manifest"]),
                "source_manifest_sha256": sha256_file(frozen_files["source_manifest"]),
                "dataset_stats_path": str(frozen_files["dataset_stats"]),
                "dataset_stats_sha256": sha256_file(frozen_files["dataset_stats"]),
                "prompt_context_cache_path": str(frozen_files["prompt_context_cache"]),
                "prompt_context_cache_sha256": sha256_file(
                    frozen_files["prompt_context_cache"]
                ),
            },
            "basis": {
                "path": str(frozen_files["basis"]),
                "sha256": sha256_file(frozen_files["basis"]),
                "split_path": str(frozen_files["split"]),
                "split_sha256": sha256_file(frozen_files["split"]),
            },
            "donors": {
                "mapping_path": str(frozen_files["donor_mapping"]),
                "mapping_sha256": sha256_file(frozen_files["donor_mapping"]),
                "manifest_path": str(frozen_files["donor_manifest"]),
                "manifest_sha256": sha256_file(frozen_files["donor_manifest"]),
                "root": str(donor_root),
            },
            "world_data": {
                "official_split_note": "held out from basis fitting",
                "dataset_root": str(dataset_root),
                "info_path": str(dataset_info),
                "info_sha256": sha256_file(dataset_info),
                "tasks_path": str(dataset_tasks),
                "tasks_sha256": sha256_file(dataset_tasks),
            },
            "frozen_action": {
                "round4c_root": str(action_root),
                "summary_path": str(action_summary),
                "summary_sha256": sha256_file(action_summary),
                "results": {"artifacts": action_artifacts},
            },
            "salvage_a": {
                "summary_path": str(frozen_files["salvage_a_summary"]),
                "summary_sha256": sha256_file(frozen_files["salvage_a_summary"]),
            },
        },
    )
    common_hashes = {
        "world_manifest_sha256": sha256_file(world_manifest),
        "stochastic_manifest_sha256": sha256_file(stochastic_manifest),
        "target_manifest_sha256": sha256_file(target_manifest),
        "machinery_sha256": sha256_file(machinery),
        "preflight_report_sha256": sha256_file(preflight),
        "git_commit_hash": commit,
    }

    sample_ids = [f"sample_{index:03d}" for index in range(100)]
    sample_identity = [
        {
            "sample_id": sample_id,
            "task_id": index // 10,
            "episode_id": 1000 + index,
            "trial_index": index % 10,
            "target_sha256": "6" * 64,
            "target_latent_sha256": "7" * 64,
            "current_image_sha256": "8" * 64,
            "current_frame_latent_sha256": "9" * 64,
            "donor_image_sha256": "a" * 64,
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    sample_identity_sha256 = sha256_json(sample_identity)
    condition_loss = {
        "current_all": 1.0,
        "wrong_all": 3.0,
        "svd_r97": 2.0,
        "svd_r170": 1.5,
    }

    def phase_summary(phase: str, conditions: tuple[str, str]) -> Path:
        root = tmp_path / f"{phase}_aggregate"
        root.mkdir()
        launcher_config = root / "launcher_config.json"
        _json(launcher_config, {"phase": phase})
        shards = []
        for worker in range(4):
            metadata_path = root / f"worker_{worker:02d}_metadata.json"
            rows_path = root / f"worker_{worker:02d}_rows.jsonl"
            _json(metadata_path, {"worker_index": worker})
            rows_path.write_text("{}\n", encoding="utf-8")
            shards.append(
                {
                    "worker_index": worker,
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": sha256_file(metadata_path),
                    "rows_path": str(rows_path),
                    "rows_sha256": sha256_file(rows_path),
                }
            )
        sample_results = [
            {
                "sample_id": sample_id,
                "task_id": index // 10,
                "episode_id": 1000 + index,
                "trial_index": index % 10,
                "condition": condition,
                "mean_native_world_loss": condition_loss[condition],
                "median_native_world_loss": condition_loss[condition],
                "mean_future_latent_mse": condition_loss[condition],
            }
            for index, sample_id in enumerate(sample_ids)
            for condition in conditions
        ]
        files = {}
        for stem in (
            "draw_rows",
            "sample_rows",
            "sample_identity_rows",
            "condition_summary",
        ):
            path = root / f"{stem}.csv"
            path.write_text("placeholder\n", encoding="utf-8")
            files[f"{stem}_path"] = str(path)
            files[f"{stem}_sha256"] = sha256_file(path)
        endpoint_gate = None
        if phase == "endpoint":
            endpoint_gate = {
                "status": "passed",
                "passed": True,
                "classification": None,
                "mean_wrong_minus_current": 2.0,
                "median_wrong_minus_current": 2.0,
                "paired_bootstrap_ci_low": 2.0,
                "paired_bootstrap_ci_high": 2.0,
                "gate": {"median_descriptive_only": True},
            }
        payload = {
            "artifact_type": "asre_salvage_b_world_phase_aggregate",
            "schema_version": 2,
            "protocol": "salvage_b_world_action_functional_dissociation",
            "phase": phase,
            "status": "passed",
            "passed": True,
            "classification": None,
            "conditions": list(conditions),
            "sample_count": 100,
            "draws_per_sample": 4,
            "pairing_complete": True,
            "worker_count": 4,
            "sample_results": sample_results,
            "sample_identity": sample_identity,
            "sample_identity_sha256": sample_identity_sha256,
            "endpoint_gate": endpoint_gate,
            "launcher_config_path": str(launcher_config),
            "launcher_config_sha256": sha256_file(launcher_config),
            "shards": shards,
            **common_hashes,
            **files,
        }
        path = root / f"{phase}_world_summary.json"
        _json(path, payload)
        if phase == "endpoint":
            _json(root / "world_endpoint_gate.json", endpoint_gate)
        return path

    endpoint_summary = phase_summary("endpoint", ("current_all", "wrong_all"))
    projected_summary = phase_summary("projected", ("svd_r97", "svd_r170"))
    output = tmp_path / "final"
    result = aggregate_final(
        endpoint_summary_path=endpoint_summary,
        projected_summary_path=projected_summary,
        preflight_path=preflight,
        architecture_audit_path=architecture,
        machinery_path=machinery,
        world_manifest_path=world_manifest,
        stochastic_manifest_path=stochastic_manifest,
        target_manifest_path=target_manifest,
        output_dir=output,
        bootstrap_samples=50,
        bootstrap_seed=7,
    )
    assert result["classification"]["classification"] == "STRONG"
    assert result["later_stage_launched"] is False
    assert result["provenance"]["action_rerun"] is False
    assert (
        result["world_dataset"]["source_separation"]
        ["semantic_episode_or_seed_crosswalk_established"]
        is False
    )
    assert all(
        "delta_vs_wrong_task_hierarchical_ci_low" in row
        and "delta_vs_wrong_task_hierarchical_ci_high" in row
        for row in result["world_condition_summary"]
    )
    assert len(result["figures"]["figures"]) == 4
    assert (output / "salvage_b_summary.json").is_file()
    assert (output / "result_summary_for_gpt.md").is_file()
    markdown = (output / "result_summary_for_gpt.md").read_text(encoding="utf-8")
    assert "| Role | Path | Class.function | Line |" in markdown
    assert "MoT.forward_future_video_with_video_cache_tensor" in markdown
    assert "Z_wrong + (Z_current-Z_wrong) B_r B_r^T" in markdown
    assert "frozen Round-4B SVD" in markdown
    assert "no refit" in markdown
    assert "Δ vs Wrong" in markdown
    assert "task-hier independent" in markdown
    assert "does **not** establish a semantic episode/seed crosswalk" in markdown
    for required_path in (
        architecture,
        architecture.with_suffix(".md"),
        machinery,
        machinery.with_suffix(".md"),
        world_manifest,
        draw_tensors,
        endpoint_summary,
        projected_summary,
        action_summary,
        metadata_path,
        dataset_root,
        output / "plots/figure_manifest.json",
        output / "salvage_b_summary.json",
        output / "salvage_b_summary.md",
        output / "result_summary_for_gpt.md",
        output / FINAL_COMPLETION_FILENAME,
        tmp_path / "driver_status.json",
    ):
        assert str(required_path.resolve()) in markdown
    completion_path = output / FINAL_COMPLETION_FILENAME
    assert completion_path.is_file()
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["publication_complete"] is True
    assert completion["classification"] == "STRONG"
    assert completion["summary_sha256"] == sha256_file(
        output / "salvage_b_summary.json"
    )
    assert len(completion["figures"]) == 4
