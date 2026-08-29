from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.asre_diagnosis.salvage_b import aggregate_final as aggregate_module
from experiments.asre_diagnosis.salvage_b.aggregate_final import (
    FINAL_COMPLETION_FILENAME,
    write_special_report,
)
from experiments.asre_diagnosis.salvage_b.run_salvage_b import (
    _completed_resume,
    _run_or_reuse_json,
    _validate_final_bundle,
    _validate_frozen_bundle,
    _validate_preflight_inputs,
    _validate_target_manifest,
    _write_terminal_status,
)
from experiments.asre_diagnosis.common import sha256_file


COMMIT = "a" * 40


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _special_chain(
    tmp_path: Path,
    classification: str,
    *,
    shared_static_failure: bool = True,
) -> dict[str, Path]:
    root = tmp_path / "special_provenance"
    architecture = root / "architecture.json"
    static_failure = (
        classification == "SHARED-INTERFACE-NOT-AVAILABLE"
        and shared_static_failure
    )
    _write(
        architecture,
        {
            "artifact_type": "asre_salvage_b_phase_a_architecture_audit",
            "git_commit_hash": COMMIT,
            "status": (
                "shared_interface_not_available"
                if static_failure
                else "preferred_path_a_pending_runtime_machinery"
            ),
            "static_audit_passed": not static_failure,
            "path": None if static_failure else "preferred_path_a",
        },
    )
    result = {"architecture_audit_path": architecture}
    if static_failure:
        return result

    preflight = root / "preflight.json"
    action_root = root / "round4c"
    action_root.mkdir()
    action_summary = action_root / "round4c_summary.json"
    _write(action_summary, {"status": "complete"})
    frozen_successes = {
        "current_all": 95,
        "wrong_all": 0,
        "svd_r97": 79,
        "svd_r170": 96,
    }
    _write(
        preflight,
        {
            "artifact_type": "asre_salvage_b_preflight_report",
            "protocol": "salvage_b_world_action_functional_dissociation",
            "git_commit_hash": COMMIT,
            "status": "compatible",
            "git": {"branch": "test"},
            "phase_a": {
                "architecture_audit_path": str(architecture.resolve()),
                "architecture_audit_sha256": sha256_file(architecture),
            },
            "frozen_action": {
                "round4c_root": str(action_root),
                "summary_path": str(action_summary),
                "summary_sha256": sha256_file(action_summary),
                "results": {
                    "successes": frozen_successes,
                    "episodes_per_condition": 100,
                    "artifacts": [],
                },
            },
        },
    )
    machinery = root / "machinery.json"
    machinery_passed = classification == "WORLD-ENDPOINT-UNINFORMATIVE"
    _write(
        machinery,
        {
            "artifact_type": "asre_salvage_b_shared_interface_machinery_report",
            "protocol": "salvage_b_world_action_functional_dissociation",
            "git_commit_hash": COMMIT,
            "status": "passed" if machinery_passed else "failed",
            "passed": machinery_passed,
            "phase_b_authorized": machinery_passed,
            "recommended_special_classification": (
                None if machinery_passed else classification
            ),
            "preflight_report_sha256": sha256_file(preflight),
            "inputs": {
                "preflight": {
                    "path": str(preflight.resolve()),
                    "sha256": sha256_file(preflight),
                }
            },
        },
    )
    result.update({"preflight_path": preflight, "machinery_path": machinery})
    if classification != "WORLD-ENDPOINT-UNINFORMATIVE":
        return result

    endpoint = root / "endpoint.json"
    endpoint_gate = {
        "status": "failed",
        "passed": False,
        "classification": classification,
        "machinery_sha256": sha256_file(machinery),
        "git_commit_hash": COMMIT,
    }
    gate_path = root / "world_endpoint_gate.json"
    _write(gate_path, endpoint_gate)
    launcher = root / "endpoint_launcher_config.json"
    _write(launcher, {"phase": "endpoint"})
    shards = []
    for worker in range(4):
        metadata = root / f"worker_{worker:02d}_metadata.json"
        rows = root / f"worker_{worker:02d}_rows.jsonl"
        _write(metadata, {"worker_index": worker})
        rows.write_text("{}\n", encoding="utf-8")
        shards.append(
            {
                "worker_index": worker,
                "metadata_path": str(metadata),
                "metadata_sha256": sha256_file(metadata),
                "rows_path": str(rows),
                "rows_sha256": sha256_file(rows),
            }
        )
    _write(
        endpoint,
        {
            "artifact_type": "asre_salvage_b_world_phase_aggregate",
            "protocol": "salvage_b_world_action_functional_dissociation",
            "git_commit_hash": COMMIT,
            "phase": "endpoint",
            "conditions": ["current_all", "wrong_all"],
            "status": "failed",
            "passed": False,
            "classification": classification,
            "preflight_report_sha256": sha256_file(preflight),
            "machinery_sha256": sha256_file(machinery),
            "endpoint_gate": endpoint_gate,
            "launcher_config_path": str(launcher),
            "launcher_config_sha256": sha256_file(launcher),
            "shards": shards,
        },
    )
    result["endpoint_summary_path"] = endpoint
    return result


def test_stage_resume_replaces_orphan_companion_without_json_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "stage.json"
    companion = tmp_path / "stage.md"
    companion.write_text("partial\n", encoding="utf-8")

    def fake_run_stage(*_args, **_kwargs) -> int:
        companion.write_text("complete\n", encoding="utf-8")
        _write(
            output,
            {
                "artifact_type": "test_stage",
                "status": "passed",
                "git_commit_hash": COMMIT,
            },
        )
        return 0

    monkeypatch.setattr(
        "experiments.asre_diagnosis.salvage_b.run_salvage_b._run_stage",
        fake_run_stage,
    )
    payload, code = _run_or_reuse_json(
        "test_stage",
        [sys.executable, "-c", "pass"],
        tmp_path / "logs",
        output,
        environment={},
        artifact_type="test_stage",
        statuses=("passed",),
        current_commit=COMMIT,
        companion=companion,
    )
    assert code == 0
    assert payload["status"] == "passed"
    assert companion.read_text(encoding="utf-8") == "complete\n"


def test_special_publication_is_recoverable_until_completion_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aggregate = tmp_path / "aggregate"
    real_write = aggregate_module._write_text
    calls = 0

    def interrupted_write(path: Path, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption before final publication")
        real_write(path, text)

    monkeypatch.setattr(aggregate_module, "_write_text", interrupted_write)
    provenance = _special_chain(tmp_path, "WORLD-ENDPOINT-UNINFORMATIVE")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_special_report(
            classification="WORLD-ENDPOINT-UNINFORMATIVE",
            reason="endpoint interval includes zero",
            output_dir=aggregate,
            git_commit_hash=COMMIT,
            **provenance,
        )
    assert (aggregate / "salvage_b_summary.json").is_file()
    assert not (aggregate / FINAL_COMPLETION_FILENAME).exists()

    monkeypatch.setattr(aggregate_module, "_write_text", real_write)
    summary = write_special_report(
        classification="WORLD-ENDPOINT-UNINFORMATIVE",
        reason="endpoint interval includes zero",
        output_dir=aggregate,
        git_commit_hash=COMMIT,
        **provenance,
    )
    assert summary["status"] == "stopped"
    validated, completion = _validate_final_bundle(
        summary_path=aggregate / "salvage_b_summary.json",
        completion_path=aggregate / FINAL_COMPLETION_FILENAME,
        current_commit=COMMIT,
        expected_status="stopped",
    )
    assert validated["provenance"]["git_commit_hash"] == COMMIT
    assert completion["publication_complete"] is True


def test_terminal_resume_cross_checks_completion_and_classification(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate"
    write_special_report(
        classification="WORLD-METRIC-NOT-VALIDATABLE",
        reason="native objective identity failed",
        output_dir=aggregate,
        git_commit_hash=COMMIT,
        **_special_chain(tmp_path, "WORLD-METRIC-NOT-VALIDATABLE"),
    )
    status_path = tmp_path / "driver_status.json"
    _write_terminal_status(
        status_path,
        start="2026-08-29T00:00:00+00:00",
        output=tmp_path,
        python=sys.executable,
        gpu_ids=(0, 1, 2, 3),
        status="stopped",
        classification="WORLD-METRIC-NOT-VALIDATABLE",
        summary_path=aggregate / "salvage_b_summary.json",
        current_commit=COMMIT,
    )
    resumed = _completed_resume(
        status_path,
        current_commit=COMMIT,
        output=tmp_path,
    )
    assert resumed is not None
    assert resumed["final_completion_sha256"]

    status = _read(status_path)
    status["classification"] = "WEAK"
    _write(status_path, status)
    with pytest.raises(RuntimeError, match="classification disagree"):
        _completed_resume(status_path, current_commit=COMMIT, output=tmp_path)


def test_published_bundle_rejects_missing_final_report(tmp_path: Path) -> None:
    aggregate = tmp_path / "aggregate"
    write_special_report(
        classification="SHARED-INTERFACE-NOT-AVAILABLE",
        reason="no eligible shared interface",
        output_dir=aggregate,
        git_commit_hash=COMMIT,
        **_special_chain(tmp_path, "SHARED-INTERFACE-NOT-AVAILABLE"),
    )
    (aggregate / "result_summary_for_gpt.md").unlink()
    with pytest.raises(RuntimeError, match="Markdown artifact drifted"):
        _validate_final_bundle(
            summary_path=aggregate / "salvage_b_summary.json",
            completion_path=aggregate / FINAL_COMPLETION_FILENAME,
            current_commit=COMMIT,
            expected_status="stopped",
        )


def test_special_report_rejects_missing_provenance_before_publication(
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "aggregate"
    with pytest.raises(ValueError, match="requires a Phase-A architecture audit"):
        write_special_report(
            classification="WORLD-METRIC-NOT-VALIDATABLE",
            reason="missing evidence must fail closed",
            output_dir=aggregate,
            git_commit_hash=COMMIT,
        )
    assert not (aggregate / "salvage_b_summary.json").exists()
    assert not (aggregate / FINAL_COMPLETION_FILENAME).exists()


def test_special_report_accepts_runtime_shared_interface_failure_chain(
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "aggregate"
    summary = write_special_report(
        classification="SHARED-INTERFACE-NOT-AVAILABLE",
        reason="runtime shared-cache identity gate failed",
        output_dir=aggregate,
        git_commit_hash=COMMIT,
        **_special_chain(
            tmp_path,
            "SHARED-INTERFACE-NOT-AVAILABLE",
            shared_static_failure=False,
        ),
    )
    assert summary["provenance"]["validated_chain"] == [
        "architecture_audit",
        "preflight",
        "machinery",
    ]
    assert (aggregate / FINAL_COMPLETION_FILENAME).is_file()


def test_special_report_rejects_endpoint_semantic_mismatch_before_publication(
    tmp_path: Path,
) -> None:
    provenance = _special_chain(tmp_path, "WORLD-ENDPOINT-UNINFORMATIVE")
    endpoint = provenance["endpoint_summary_path"]
    payload = _read(endpoint)
    payload["endpoint_gate"]["classification"] = "WORLD-METRIC-NOT-VALIDATABLE"
    _write(endpoint, payload)
    aggregate = tmp_path / "aggregate"
    with pytest.raises(ValueError, match="closed, failed"):
        write_special_report(
            classification="WORLD-ENDPOINT-UNINFORMATIVE",
            reason="mismatched endpoint label",
            output_dir=aggregate,
            git_commit_hash=COMMIT,
            **provenance,
        )
    assert not (aggregate / "salvage_b_summary.json").exists()
    assert not (aggregate / FINAL_COMPLETION_FILENAME).exists()


def test_special_report_rejects_broken_architecture_preflight_hash_chain(
    tmp_path: Path,
) -> None:
    provenance = _special_chain(tmp_path, "WORLD-METRIC-NOT-VALIDATABLE")
    preflight = provenance["preflight_path"]
    payload = _read(preflight)
    payload["phase_a"]["architecture_audit_sha256"] = "0" * 64
    _write(preflight, payload)
    aggregate = tmp_path / "aggregate"
    with pytest.raises(ValueError, match="hash-bound child"):
        write_special_report(
            classification="WORLD-METRIC-NOT-VALIDATABLE",
            reason="broken provenance hash",
            output_dir=aggregate,
            git_commit_hash=COMMIT,
            **provenance,
        )
    assert not (aggregate / "salvage_b_summary.json").exists()
    assert not (aggregate / FINAL_COMPLETION_FILENAME).exists()


@pytest.mark.parametrize(
    "classification",
    (
        "SHARED-INTERFACE-NOT-AVAILABLE",
        "WORLD-METRIC-NOT-VALIDATABLE",
        "WORLD-ENDPOINT-UNINFORMATIVE",
    ),
)
def test_each_special_report_closes_all_twenty_registered_items(
    tmp_path: Path,
    classification: str,
) -> None:
    provenance = _special_chain(tmp_path, classification)
    aggregate = tmp_path / "aggregate"
    summary = write_special_report(
        classification=classification,
        reason="registered technical stop",
        output_dir=aggregate,
        git_commit_hash=COMMIT,
        **provenance,
    )
    markdown = (aggregate / "result_summary_for_gpt.md").read_text(encoding="utf-8")
    for number, title in enumerate(aggregate_module.FINAL_REPORT_ITEM_TITLES, start=1):
        assert f"## {number}. {title}" in markdown
    assert "N/A —" in markdown
    assert "Paper 1 should be frozen/closed" in markdown
    assert "NO later ASRE experiment was launched" in markdown
    for path in provenance.values():
        assert str(path.resolve()) in markdown
    assert str((aggregate / "salvage_b_summary.json").resolve()) in markdown
    assert str((aggregate / FINAL_COMPLETION_FILENAME).resolve()) in markdown
    assert summary["phase_b_projected_conditions_run"] is False
    if classification == "WORLD-ENDPOINT-UNINFORMATIVE":
        assert "endpoint worker 0 metadata" in markdown
        assert str(
            (tmp_path / "special_provenance/world_endpoint_gate.json").resolve()
        ) in markdown
    if classification != "SHARED-INTERFACE-NOT-AVAILABLE":
        assert "Current | 95 / 100" in markdown
        assert "SVD-170 | 96 / 100" in markdown
        assert "no action episode was rerun" in markdown


def test_world_bundle_completion_marker_binds_exact_three_artifacts(
    tmp_path: Path,
) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    preflight = tmp_path / "preflight_report.json"
    world = manifests / "world_evaluation_manifest.json"
    stochastic = manifests / "stochastic_manifest.json"
    draws = manifests / "fixed_draw_tensors.pt"
    marker = manifests / "world_bundle_complete.json"
    _write(preflight, {"status": "compatible"})
    draws.write_bytes(b"frozen draws")
    _write(
        world,
        {
            "artifact_type": "asre_salvage_b_world_evaluation_manifest",
            "status": "frozen_before_metrics",
            "git_commit_hash": COMMIT,
            "preflight_report_sha256": sha256_file(preflight),
            "draw_tensor_sha256": sha256_file(draws),
        },
    )
    _write(
        stochastic,
        {
            "artifact_type": "asre_salvage_b_stochastic_manifest",
            "status": "frozen_before_metrics",
            "git_commit_hash": COMMIT,
            "preflight_report_sha256": sha256_file(preflight),
            "world_manifest_sha256": sha256_file(world),
            "draw_tensor_sha256": sha256_file(draws),
        },
    )
    artifacts = (world, stochastic, draws)
    _write(
        marker,
        {
            "artifact_type": "asre_salvage_b_world_bundle_completion",
            "schema_version": 2,
            "status": "complete",
            "git_commit_hash": COMMIT,
            "preflight_report_sha256": sha256_file(preflight),
            "artifacts": {
                path.name: {"path": str(path), "sha256": sha256_file(path)}
                for path in artifacts
            },
        },
    )

    _validate_frozen_bundle(
        world_manifest=world,
        stochastic_manifest=stochastic,
        draw_tensors=draws,
        bundle_completion=marker,
        preflight=preflight,
        current_commit=COMMIT,
    )

    draws.write_bytes(b"tampered draws")
    with pytest.raises(RuntimeError, match="drifted frozen world bundle marker"):
        _validate_frozen_bundle(
            world_manifest=world,
            stochastic_manifest=stochastic,
            draw_tensors=draws,
            bundle_completion=marker,
            preflight=preflight,
            current_commit=COMMIT,
        )


def test_preflight_resume_is_bound_to_output_root(tmp_path: Path) -> None:
    architecture = tmp_path / "architecture.json"
    architecture.write_text("{}\n", encoding="utf-8")
    round4b = tmp_path / "round4b"
    round4c = tmp_path / "round4c"
    salvage_a = tmp_path / "salvage_a.json"
    valid = tmp_path / "valid.json"
    donor_root = tmp_path / "donors"
    donor_mapping = donor_root / "mapping.json"
    donor_manifest = donor_root / "manifest.json"
    dataset = tmp_path / "dataset"
    expected_output = tmp_path / "expected_output"
    payload = {
        "output_root": str(tmp_path / "different_output"),
        "phase_a": {
            "architecture_audit_path": str(architecture),
            "architecture_audit_sha256": sha256_file(architecture),
        },
        "basis": {
            "path": str(round4b / "calibration/basis_manifest.json"),
            "split_path": str(
                round4b / "calibration/calibration_split_manifest.json"
            ),
        },
        "frozen_action": {"round4c_root": str(round4c)},
        "salvage_a": {"summary_path": str(salvage_a)},
        "state": {"valid_manifest_path": str(valid)},
        "donors": {
            "mapping_path": str(donor_mapping),
            "manifest_path": str(donor_manifest),
            "root": str(donor_root),
        },
        "world_data": {"dataset_root": str(dataset)},
    }
    with pytest.raises(RuntimeError, match="output_root"):
        _validate_preflight_inputs(
            payload,
            output=expected_output,
            architecture=architecture,
            round4b=round4b,
            round4c=round4c,
            salvage_a=salvage_a,
            valid=valid,
            donor_root=donor_root,
            donor_mapping=donor_mapping,
            donor_manifest=donor_manifest,
            dataset=dataset,
        )


def test_processed_targets_require_schema2_current_hash_and_source_verification(
    tmp_path: Path,
) -> None:
    preflight = tmp_path / "preflight.json"
    world = tmp_path / "world.json"
    target = tmp_path / "targets.json"
    _write(preflight, {"status": "compatible"})
    world_records = [
        {
            "sample_id": f"sample_{index:03d}",
            "task_id": index // 10,
            "episode_id": 1000 + index,
            "trial": index % 10,
            "dataset_index": 2000 + index,
        }
        for index in range(100)
    ]
    _write(
        world,
        {
            "records": world_records,
            "metadata_files": [],
            "target_source_files": [],
        },
    )
    target_payload = {
        "artifact_type": "asre_salvage_b_processed_world_target_manifest",
        "schema_version": 2,
        "protocol": "salvage_b_world_action_functional_dissociation",
        "status": "frozen_before_gpu_metrics",
        "git_commit_hash": COMMIT,
        "preflight_report_sha256": sha256_file(preflight),
        "world_manifest_sha256": sha256_file(world),
        "sample_count": 100,
        "records": [
            {**record, "current_image_sha256": f"{index + 1:064x}"}
            for index, record in enumerate(world_records)
        ],
        "source_artifact_verification": {
            "before_decode": True,
            "after_decode": True,
            "metadata_file_count": 0,
            "target_source_file_count": 0,
            "checks": "resolved regular file, byte size, and SHA-256",
        },
    }
    _write(target, target_payload)
    _validate_target_manifest(
        target,
        preflight=preflight,
        world_manifest=world,
        current_commit=COMMIT,
    )

    target_payload["records"][0]["current_image_sha256"] = "not-a-hash"
    _write(target, target_payload)
    with pytest.raises(RuntimeError, match="current-image hash drifted"):
        _validate_target_manifest(
            target,
            preflight=preflight,
            world_manifest=world,
            current_commit=COMMIT,
        )
