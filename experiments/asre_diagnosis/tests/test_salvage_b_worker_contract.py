from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from experiments.asre_diagnosis.common import (
    SALVAGE_B_PROTOCOL,
    atomic_write_json,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.salvage_b import world_runtime, world_worker
from experiments.asre_diagnosis.salvage_b.world_runtime import (
    validate_architecture_source_inventory,
)
from experiments.asre_diagnosis.salvage_b.world_worker import (
    ENDPOINT_CONDITIONS,
    PROJECTED_CONDITIONS,
    ROWS_PER_WORKER,
    _atomic_write_jsonl,
    _guard_frozen_source,
    phase_conditions,
    run_worker,
    validate_completed_shard,
    validate_endpoint_gate_payload,
    validate_rows,
    worker_records,
)


PROVENANCE = {
    "world_manifest_sha256": "1" * 64,
    "stochastic_manifest_sha256": "2" * 64,
    "target_manifest_sha256": "3" * 64,
    "machinery_sha256": "4" * 64,
    "git_commit": "5" * 40,
}


def _architecture_contract(
    *, repository: Path, output_root: Path, commit: str
) -> tuple[dict, Path, Path]:
    source = repository / "src/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    audit = output_root / "phase_a/architecture_audit.json"
    atomic_write_json(
        audit,
        {
            "artifact_type": "asre_salvage_b_phase_a_architecture_audit",
            "schema_version": 1,
            "git_commit_hash": commit,
            "source_inventory": [
                {"path": "src/example.py", "sha256": sha256_file(source)}
            ],
        },
    )
    preflight = {
        "output_root": str(output_root),
        "phase_a": {
            "architecture_audit_path": str(audit),
            "architecture_audit_sha256": sha256_file(audit),
        },
    }
    return preflight, audit, source


def _records():
    return [
        {
            "sample_id": f"sample_{index:03d}",
            "task_id": index // 10,
            "trial": index % 10,
            "episode_id": index,
            "dataset_index": 1_000 + index,
            "contains_padding": False,
        }
        for index in range(100)
    ]


def _rows(worker_index: int = 0, phase: str = "endpoint"):
    records = worker_records(_records(), worker_index)
    conditions = phase_conditions(phase)
    rows = []
    for record in records:
        for draw_id in range(4):
            for condition_index, condition in enumerate(conditions):
                rows.append(
                    {
                        "schema_version": 2,
                        "protocol": SALVAGE_B_PROTOCOL,
                        "phase": phase,
                        "worker_index": worker_index,
                        "sample_id": record["sample_id"],
                        "task_id": record["task_id"],
                        "episode_id": record["episode_id"],
                        "trial_index": record["trial"],
                        "draw_id": draw_id,
                        "condition": condition,
                        "rank": {
                            "current_all": 3072,
                            "wrong_all": 0,
                            "svd_r97": 97,
                            "svd_r170": 170,
                        }[condition],
                        "native_world_loss": 1.0 + condition_index,
                        "future_latent_mse": 1.0 + condition_index,
                        "inference_steps": 10,
                        "inference_shift": 5.0,
                        "target_sha256": "a" * 64,
                        "target_latent_sha256": "c" * 64,
                        "current_image_sha256": "d" * 64,
                        "current_frame_latent_sha256": "e" * 64,
                        "donor_image_sha256": "f" * 64,
                        "video_noise_sha256": f"{draw_id + 6:x}" * 64,
                        "prediction_shape": [1, 48, 2, 14, 28],
                        "target_shape": [1, 48, 2, 14, 28],
                        "future_token_shape": [1, 196, 3072],
                        **PROVENANCE,
                    }
                )
    return records, rows


def test_phases_have_only_the_registered_two_conditions() -> None:
    assert phase_conditions("endpoint") == ENDPOINT_CONDITIONS
    assert phase_conditions("projected") == PROJECTED_CONDITIONS
    with pytest.raises(ValueError, match="Unsupported"):
        phase_conditions("extra-rank")


def test_four_stride_shards_are_disjoint_and_exact() -> None:
    partitions = [worker_records(_records(), index) for index in range(4)]
    assert [len(partition) for partition in partitions] == [25, 25, 25, 25]
    sample_sets = [{record["sample_id"] for record in partition} for partition in partitions]
    assert len(set.union(*sample_sets)) == 100
    assert sum(len(left & right) for left in sample_sets for right in sample_sets if left is not right) == 0


def test_row_contract_requires_exact_pairing_and_200_rows() -> None:
    records, rows = _rows()
    sample_ids = [record["sample_id"] for record in records]
    validate_rows(
        rows,
        phase="endpoint",
        worker_index=0,
        expected_sample_ids=sample_ids,
        conditions=ENDPOINT_CONDITIONS,
        provenance=PROVENANCE,
    )
    assert len(rows) == ROWS_PER_WORKER

    broken = [dict(row) for row in rows]
    broken[1]["video_noise_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="pairing drifted"):
        validate_rows(
            broken,
            phase="endpoint",
            worker_index=0,
            expected_sample_ids=sample_ids,
            conditions=ENDPOINT_CONDITIONS,
            provenance=PROVENANCE,
        )


def test_completed_shard_resume_binds_metadata_rows_and_provenance(tmp_path: Path) -> None:
    records, rows = _rows()
    sample_ids = [record["sample_id"] for record in records]
    shard = tmp_path / "endpoint/worker_00"
    rows_path = shard / "rows.jsonl"
    metadata_path = shard / "metadata.json"
    _atomic_write_jsonl(rows_path, rows)
    launcher_sha = "b" * 64
    atomic_write_json(
        metadata_path,
        {
            "schema_version": 2,
            "status": "completed",
            "phase": "endpoint",
            "worker_index": 0,
            "record_count": 200,
            "sample_count": 25,
            "conditions": list(ENDPOINT_CONDITIONS),
            "draws_per_sample": 4,
            **PROVENANCE,
            "launcher_config_sha256": launcher_sha,
            "rows_sha256": sha256_file(rows_path),
            "sample_ids_sha256": sha256_json(sample_ids),
            "native_metric": "pure_noise_native_future_latent_reconstruction_mse",
            "inference_steps": 10,
            "inference_shift": 5.0,
            "no_ddp": True,
        },
    )
    assert validate_completed_shard(
        output_dir=tmp_path,
        phase="endpoint",
        worker_index=0,
        expected_sample_ids=sample_ids,
        conditions=ENDPOINT_CONDITIONS,
        provenance=PROVENANCE,
        launcher_config_sha256=launcher_sha,
    )

    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incompatible"):
        validate_completed_shard(
            output_dir=tmp_path,
            phase="endpoint",
            worker_index=0,
            expected_sample_ids=sample_ids,
            conditions=ENDPOINT_CONDITIONS,
            provenance=PROVENANCE,
            launcher_config_sha256=launcher_sha,
        )


def test_resume_refuses_an_empty_published_shard_directory(tmp_path: Path) -> None:
    records = worker_records(_records(), 0)
    (tmp_path / "endpoint/worker_00").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_completed_shard(
            output_dir=tmp_path,
            phase="endpoint",
            worker_index=0,
            expected_sample_ids=[record["sample_id"] for record in records],
            conditions=ENDPOINT_CONDITIONS,
            provenance=PROVENANCE,
            launcher_config_sha256="b" * 64,
        )


def test_launcher_has_four_isolated_processes_and_no_ddp_command() -> None:
    path = Path("experiments/asre_diagnosis/salvage_b/launch_world.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Popen"
    ]
    assert len(calls) == 1
    assert "torchrun" not in source
    assert "torch.distributed.run" not in source
    assert 'environment["TORCHINDUCTOR_CACHE_DIR"]' in source
    assert 'environment["PYTHONHASHSEED"] = "0"' in source
    assert 'environment["TOKENIZERS_PARALLELISM"] = "false"' in source
    assert "except BaseException:" in source
    assert "_terminate_children(children)" in source
    assert "config_sha[:16]" in source
    assert "gpu{physical_gpu}" in source


def test_projected_gate_binds_four_shards_and_aggregate_hashes(tmp_path: Path) -> None:
    shards = []
    for index in range(4):
        metadata_path = tmp_path / f"worker_{index:02d}.metadata.json"
        rows_path = tmp_path / f"worker_{index:02d}.rows.jsonl"
        metadata_path.write_text(f"metadata-{index}\n", encoding="utf-8")
        rows_path.write_text(f"rows-{index}\n", encoding="utf-8")
        shards.append(
            {
                "worker_index": index,
                "record_count": 200,
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "rows_path": str(rows_path),
                "rows_sha256": sha256_file(rows_path),
            }
        )
    draw_rows = tmp_path / "draw_rows.csv"
    sample_rows = tmp_path / "sample_rows.csv"
    identity_rows = tmp_path / "sample_identity.csv"
    for path in (draw_rows, sample_rows, identity_rows):
        path.write_text(f"{path.name}\n", encoding="utf-8")
    identities = [
        {
            "sample_id": f"sample_{index:03d}",
            "task_id": index // 10,
            "episode_id": index,
            "trial_index": index % 10,
            "target_sha256": "a" * 64,
            "target_latent_sha256": "b" * 64,
            "current_image_sha256": "c" * 64,
            "current_frame_latent_sha256": "d" * 64,
            "donor_image_sha256": "e" * 64,
        }
        for index in range(100)
    ]
    payload = {
        "artifact_type": "asre_salvage_b_world_endpoint_gate",
        "schema_version": 2,
        "status": "passed",
        "passed": True,
        "classification": None,
        "sample_count": 100,
        "loss_direction": "lower_is_better",
        "native_world_metric": "pure_noise_native_future_latent_reconstruction_mse",
        "inference_steps": 10,
        "inference_shift": 5.0,
        "world_manifest_sha256": PROVENANCE["world_manifest_sha256"],
        "stochastic_manifest_sha256": PROVENANCE["stochastic_manifest_sha256"],
        "target_manifest_sha256": PROVENANCE["target_manifest_sha256"],
        "machinery_sha256": PROVENANCE["machinery_sha256"],
        "git_commit_hash": PROVENANCE["git_commit"],
        "endpoint_shards": shards,
        "draw_rows_path": str(draw_rows),
        "draw_rows_sha256": sha256_file(draw_rows),
        "sample_rows_path": str(sample_rows),
        "sample_rows_sha256": sha256_file(sample_rows),
        "sample_identity": identities,
        "sample_identity_sha256": sha256_json(identities),
        "sample_identity_rows_path": str(identity_rows),
        "sample_identity_rows_sha256": sha256_file(identity_rows),
    }
    kwargs = {
        "world_manifest_sha256": PROVENANCE["world_manifest_sha256"],
        "stochastic_manifest_sha256": PROVENANCE["stochastic_manifest_sha256"],
        "target_manifest_sha256": PROVENANCE["target_manifest_sha256"],
        "machinery_sha256": PROVENANCE["machinery_sha256"],
        "git_commit_hash": PROVENANCE["git_commit"],
    }
    validate_endpoint_gate_payload(payload, **kwargs)

    broken = dict(payload)
    broken["endpoint_shards"] = payload["endpoint_shards"][:3]
    with pytest.raises(ValueError, match="four frozen shard"):
        validate_endpoint_gate_payload(broken, **kwargs)

    rows_path = Path(shards[0]["rows_path"])
    rows_path.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="referenced rows_path changed"):
        validate_endpoint_gate_payload(payload, **kwargs)


def test_architecture_inventory_rehashes_repo_relative_sources_and_detects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    commit = "a" * 40
    preflight, _audit, source = _architecture_contract(
        repository=repository,
        output_root=repository / "asre_results/salvage_b",
        commit=commit,
    )
    monkeypatch.setattr(world_runtime, "PROJECT_ROOT", repository)

    report = validate_architecture_source_inventory(
        preflight,
        expected_commit=commit,
    )
    assert report["git_commit_hash"] == commit

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audited source drifted"):
        validate_architecture_source_inventory(preflight, expected_commit=commit)


@pytest.mark.parametrize(
    ("inventory", "message"),
    [
        ([{"path": "../outside.py", "sha256": "1" * 64}], "repo-relative"),
        (
            [
                {"path": "src/example.py", "sha256": "1" * 64},
                {"path": "src/example.py", "sha256": "1" * 64},
            ],
            "Duplicate",
        ),
    ],
)
def test_architecture_inventory_rejects_escape_and_duplicate_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory: list[dict[str, str]],
    message: str,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    commit = "a" * 40
    preflight, audit, source = _architecture_contract(
        repository=repository,
        output_root=repository / "asre_results/salvage_b",
        commit=commit,
    )
    if message == "Duplicate":
        inventory = [
            {**record, "sha256": sha256_file(source)} for record in inventory
        ]
    payload = world_runtime.read_json(audit)
    payload["source_inventory"] = inventory
    atomic_write_json(audit, payload)
    preflight["phase_a"]["architecture_audit_sha256"] = sha256_file(audit)
    monkeypatch.setattr(world_runtime, "PROJECT_ROOT", repository)
    with pytest.raises(ValueError, match=message):
        validate_architecture_source_inventory(preflight, expected_commit=commit)


def test_architecture_inventory_detects_audit_mutation_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    commit = "a" * 40
    preflight, audit, source = _architecture_contract(
        repository=repository,
        output_root=repository / "asre_results/salvage_b",
        commit=commit,
    )
    monkeypatch.setattr(world_runtime, "PROJECT_ROOT", repository)
    real_sha256_file = sha256_file

    def mutate_audit_after_source_hash(path: Path) -> str:
        digest = real_sha256_file(path)
        if Path(path).resolve() == source.resolve():
            audit.write_text("{}\n", encoding="utf-8")
        return digest

    monkeypatch.setattr(world_runtime, "sha256_file", mutate_audit_after_source_hash)
    with pytest.raises(ValueError, match="changed during validation"):
        validate_architecture_source_inventory(preflight, expected_commit=commit)


def test_worker_source_guard_checks_clean_git_before_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    contract = {
        "commit": "a" * 40,
        "payloads": {"preflight": {"output_root": str(tmp_path)}},
    }

    def clean(expected_commit: str, *, output_root: Path) -> None:
        assert expected_commit == contract["commit"]
        assert output_root == tmp_path.resolve()
        events.append("clean")

    def inventory(preflight: dict, *, expected_commit: str) -> None:
        assert preflight is contract["payloads"]["preflight"]
        assert expected_commit == contract["commit"]
        events.append("inventory")

    monkeypatch.setattr(world_worker, "require_clean_source", clean)
    monkeypatch.setattr(
        world_worker,
        "validate_architecture_source_inventory",
        inventory,
    )
    _guard_frozen_source(contract)
    assert events == ["clean", "inventory"]

    def dirty(*_args, **_kwargs) -> None:
        raise RuntimeError("dirty source")

    events.clear()
    monkeypatch.setattr(world_worker, "require_clean_source", dirty)
    with pytest.raises(RuntimeError, match="dirty source"):
        _guard_frozen_source(contract)
    assert events == []


def test_completed_shard_fast_return_still_runs_source_guard_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    contract = {
        "commit": PROVENANCE["git_commit"],
        "payloads": {
            "launcher": {"gpu_ids": [4, 5, 6, 7]},
            "preflight": {"output_root": str(tmp_path)},
        },
        "conditions": ENDPOINT_CONDITIONS,
        "records": _records(),
        "shas": {
            "world": PROVENANCE["world_manifest_sha256"],
            "stochastic": PROVENANCE["stochastic_manifest_sha256"],
            "targets": PROVENANCE["target_manifest_sha256"],
            "machinery": PROVENANCE["machinery_sha256"],
            "launcher": "b" * 64,
        },
    }

    def load_contract(*_args, **_kwargs):
        events.append("load")
        return contract

    def guard(observed) -> None:
        assert observed is contract
        events.append("guard")

    def completed(**_kwargs) -> bool:
        events.append("resume")
        return True

    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(world_worker, "_isolated_gpu", lambda _index: 4)
    monkeypatch.setattr(world_worker, "load_frozen_contract", load_contract)
    monkeypatch.setattr(world_worker, "_guard_frozen_source", guard)
    monkeypatch.setattr(world_worker, "validate_completed_shard", completed)
    monkeypatch.setattr(
        world_worker,
        "shard_paths",
        lambda *_args: (metadata, tmp_path / "rows.jsonl"),
    )
    result = run_worker(
        argparse.Namespace(phase="endpoint", worker_index=0, output_dir=tmp_path)
    )
    assert result == {}
    assert events == ["load", "guard", "resume"]


def test_worker_places_two_source_guards_around_long_pass_before_publish() -> None:
    source = Path(
        "experiments/asre_diagnosis/salvage_b/world_worker.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_worker"
    )

    def call_lines(name: str) -> list[int]:
        return sorted(
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == name)
                or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
            )
        )

    load = call_lines("load_frozen_contract")
    guards = call_lines("_guard_frozen_source")
    completed = call_lines("validate_completed_shard")
    temporary = call_lines("mkdtemp")
    publish = call_lines("replace")
    assert len(load) == 1
    assert len(guards) == 2
    assert len(completed) == 2
    assert len(temporary) == 1
    assert len(publish) == 1
    assert load[0] < guards[0] < completed[0] < guards[1] < temporary[0]
    assert temporary[0] < publish[0] < completed[1]
