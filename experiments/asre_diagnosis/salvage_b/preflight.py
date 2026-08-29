"""Fail-closed provenance, data, and frozen-action gate for Salvage B."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_B_PROTOCOL,
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round3b.donor import OnlineDonorBundle  # noqa: E402
from experiments.asre_diagnosis.round4b.basis import validate_basis_manifest  # noqa: E402


EXPECTED = {
    "checkpoint": "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579",
    "basis": "1a3fcd8ed44e012abbfdf8da119b82e76baf01755a3a8472eb470845c7122d02",
    "split": "38a6cb79e3006fd8b1bacc24691df6a79751d323fdff8cc3b5912601f217024d",
    "donor_mapping": "b072adef3aebda9e5449f6e9bc90baade8e84e4703c9fa500642157b59643a6c",
    "donor_manifest": "3a568b0a47e7cacf26914c881c859362d84743d4b5194017c5d126a0d85ba5bd",
    "round4c_summary": "b6108d134197c0825c4f8f2bcb11f87bab6645d3c714033cae5515d16b40f4d1",
    "valid_manifest": "a05ebb6b50d64eb022103269a0eef595e749fe8c16e49e996ba08669f44378cd",
    "state_bank_manifest": "ed582c1420a705b08105acc5481f93ef899d6677eefa48a78a3f6ed0b1f50ab8",
    "salvage_a_summary": "50f55146ae62be28b1f601273a0d16f767053056cd46bc0ca12d277049b6b619",
}
ROUND4C_EXECUTION_COMMIT = "0bd05934d68ffe31c5cd107bc49508b19e8a3ab1"
CONDITIONS = ("current_all", "wrong_all", "svd_r97", "svd_r170")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _require_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path.resolve())
    if observed != expected:
        raise ValueError(f"Frozen {label} drifted: {observed} != {expected}.")
    return observed


def _function_node(source: str, class_name: str | None, function_name: str) -> ast.AST:
    tree = ast.parse(source)
    nodes: list[ast.AST] = list(tree.body)
    if class_name is not None:
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise ValueError(f"Cannot locate class {class_name} for source identity.")
        nodes = list(classes[0].body)
    matches = [
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot locate function {function_name} for source identity.")
    return matches[0]


def _ast_hash(source: str, class_name: str | None, function_name: str) -> str:
    node = _function_node(source, class_name, function_name)
    canonical = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_action_implementation_identity() -> list[dict[str, Any]]:
    checks = (
        (
            "src/fastwam/models/wan22/video_cache_replacement.py",
            None,
            "validate_matching_video_cache",
        ),
        (
            "src/fastwam/models/wan22/video_cache_replacement.py",
            None,
            "select_replacement_video_cache",
        ),
        (
            "src/fastwam/models/wan22/video_cache_replacement.py",
            None,
            "project_replacement_video_cache",
        ),
        (
            "src/fastwam/models/wan22/fastwam.py",
            "FastWAM",
            "_encode_input_image_latents_tensor",
        ),
        (
            "src/fastwam/models/wan22/fastwam.py",
            "FastWAM",
            "_build_mot_attention_mask",
        ),
        (
            "src/fastwam/models/wan22/fastwam.py",
            "FastWAM",
            "_denoise_action_with_video_cache",
        ),
        (
            "src/fastwam/models/wan22/mot.py",
            "MoT",
            "_apply_expert_post_block_tensor",
        ),
        (
            "src/fastwam/models/wan22/mot.py",
            "MoT",
            "_build_expert_attention_io",
        ),
        (
            "src/fastwam/models/wan22/mot.py",
            "MoT",
            "prefill_video_cache_tensor",
        ),
        (
            "src/fastwam/models/wan22/mot.py",
            "MoT",
            "forward_action_with_video_cache_tensor",
        ),
        (
            "src/fastwam/models/wan22/wan_video_dit.py",
            "WanVideoDiT",
            "prepare",
        ),
        (
            "src/fastwam/models/wan22/action_dit.py",
            "ActionDiT",
            "prepare",
        ),
        (
            "src/fastwam/models/wan22/action_dit.py",
            "ActionDiT",
            "post",
        ),
    )
    rows = []
    for relative, class_name, function_name in checks:
        current_source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        frozen_source = _git("show", f"{ROUND4C_EXECUTION_COMMIT}:{relative}")
        current_hash = _ast_hash(current_source, class_name, function_name)
        frozen_hash = _ast_hash(frozen_source, class_name, function_name)
        if current_hash != frozen_hash:
            raise ValueError(
                f"Round-4C action implementation drifted at {class_name}.{function_name}."
            )
        rows.append(
            {
                "path": relative,
                "class": class_name,
                "function": function_name,
                "current_ast_sha256": current_hash,
                "round4c_execution_ast_sha256": frozen_hash,
                "identical": True,
                "strict_shared_intervention_dependency": True,
            }
        )
    # The post-Round-4C Salvage-A commit extended the public `infer_action`
    # wrapper with an optional gradient path.  Frozen Round-4C action artifacts
    # are not recomputed by that wrapper here.  Its tensor core, cache prefill,
    # cache selection/projection, image encoding, masking, action consumer, and
    # action prepare/post functions are all checked strictly above.  Record the
    # wrapper drift explicitly instead of falsely requiring byte/AST identity
    # from code that is outside the reused artifact execution.
    relative = "src/fastwam/models/wan22/fastwam.py"
    current_source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
    frozen_source = _git("show", f"{ROUND4C_EXECUTION_COMMIT}:{relative}")
    current_hash = _ast_hash(current_source, "FastWAM", "infer_action")
    frozen_hash = _ast_hash(frozen_source, "FastWAM", "infer_action")
    rows.append(
        {
            "path": relative,
            "class": "FastWAM",
            "function": "infer_action",
            "current_ast_sha256": current_hash,
            "round4c_execution_ast_sha256": frozen_hash,
            "identical": current_hash == frozen_hash,
            "strict_shared_intervention_dependency": False,
            "reuse_basis": (
                "frozen action files were executed at the registered Round-4C commit; "
                "all tensor-core dependencies used by its standard projection path are "
                "strictly identical and runtime machinery remains mandatory"
            ),
        }
    )
    return rows


def _validate_clean(output_root: Path) -> dict[str, Any]:
    relative = output_root.resolve().relative_to(PROJECT_ROOT.resolve())
    dirty = _git(
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".",
        f":(exclude){relative}/**",
    )
    if dirty:
        raise RuntimeError(
            "Salvage-B formal execution requires committed reviewed source code.\n" + dirty
        )
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", ROUND4C_EXECUTION_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
    )
    return {
        "branch": _git("branch", "--show-current"),
        "head": git_commit(PROJECT_ROOT),
        "worktree_clean_excluding_output": True,
        "round4c_execution_commit": ROUND4C_EXECUTION_COMMIT,
        "round4c_is_ancestor": True,
    }


def _condition_dir(round4c_root: Path, condition: str) -> Path:
    wave = "wave2" if condition == "svd_r170" else "wave1"
    return round4c_root / f"online_full/{wave}/{condition}"


def _validate_action_results(round4c_root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    if not (
        summary.get("protocol") == "round4c_energy_controlled_action_sufficiency"
        and summary.get("status") == "complete"
        and summary.get("later_stage_launched") is False
        and summary.get("machinery", {}).get("passed") is True
    ):
        raise ValueError("Frozen Round-4C summary protocol/machinery/stop status drifted.")
    summary_rows = {row["condition"]: row for row in summary["online_condition_summary"]}
    if not set(CONDITIONS).issubset(summary_rows):
        raise ValueError("Round-4C summary lacks a Salvage-B action condition.")
    shared_identity: dict[str, Any] | None = None
    identities: list[dict[str, Any]] = []
    outcome_counts: dict[str, int] = {}
    identity_keys = (
        "checkpoint_sha256",
        "subspace_basis_manifest_sha256",
        "calibration_split_manifest_sha256",
        "donor_mapping_sha256",
        "donor_observation_manifest_sha256",
        "state_bank_manifest_sha256",
        "valid_state_bank_manifest_sha256",
        "prompt_context_cache_sha256",
    )
    for condition in CONDITIONS:
        root = _condition_dir(round4c_root, condition)
        metadata_path = root / "run_metadata.json"
        metadata = _read(metadata_path)
        if (
            metadata.get("status") != "completed"
            or metadata.get("git_commit_hash") != ROUND4C_EXECUTION_COMMIT
            or metadata.get("diagnosis_condition") != condition
            or metadata.get("condition_protocol")
            != "round4c_energy_controlled_action_sufficiency"
            or metadata.get("task_suite") != "libero_spatial"
            or metadata.get("task_ids") != list(range(10))
            or metadata.get("number_of_trials") != 10
            or metadata.get("seed") != 42
            or metadata.get("number_of_inference_steps") != 10
            or metadata.get("action_horizon") != 32
            or metadata.get("replan_steps") != 10
            or metadata.get("enabled_video_retrieval_layers") != list(range(15, 30))
        ):
            raise ValueError(f"Incompatible frozen Round-4C action metadata: {condition}.")
        identity = {key: metadata.get(key) for key in identity_keys}
        if shared_identity is None:
            shared_identity = identity
        elif identity != shared_identity:
            raise ValueError("Round-4C action provenance differs across conditions.")
        config = metadata.get("condition_config", {})
        expected_rank = int(condition.removeprefix("svd_r")) if condition.startswith("svd_r") else None
        expected_replacement = [] if condition == "current_all" else list(range(15, 30))
        if (
            config.get("disabled_video_layers") != list(range(15))
            or config.get("replacement_video_layers") != expected_replacement
            or config.get("subspace_rank") != expected_rank
            or config.get("subspace_basis_kind") != ("svd" if expected_rank else None)
            or config.get("protocol") != "round4c_energy_controlled_action_sufficiency"
            or config.get("mode") != "replace_video_kv"
            or config.get("hybrid_axis") is not None
            or config.get("hybrid_mask_seed") is not None
            or config.get("subspace_basis_manifest_sha256") != EXPECTED["basis"]
        ):
            raise ValueError(f"Round-4C condition semantics drifted: {condition}.")
        result_files = sorted((root / "libero_spatial").glob("gpu*_task*_results.json"))
        if len(result_files) != 10:
            raise ValueError(f"Frozen action condition {condition} lacks 10 task files.")
        successes = 0
        result_hashes = []
        observed_tasks: set[int] = set()
        for path in result_files:
            result = _read(path)
            task_id = int(result.get("task_id", -1))
            success = set(map(int, result["success_episodes"]))
            failure = set(map(int, result["failure_episodes"]))
            if (
                task_id not in range(10)
                or task_id in observed_tasks
                or result.get("condition_protocol")
                != "round4c_energy_controlled_action_sufficiency"
                or result.get("diagnosis_condition") != condition
                or result.get("disabled_video_layers") != list(range(15))
                or result.get("replacement_video_layers") != expected_replacement
                or result.get("subspace_rank") != expected_rank
                or result.get("subspace_basis_kind")
                != ("svd" if expected_rank else None)
                or result.get("subspace_basis_manifest_sha256") != EXPECTED["basis"]
                or success & failure
                or success | failure != set(range(10))
            ):
                raise ValueError(f"Malformed Round-4C paired action result: {path}")
            observed_tasks.add(task_id)
            successes += len(success)
            result_hashes.append({"path": str(path), "sha256": sha256_file(path)})
        if observed_tasks != set(range(10)):
            raise ValueError(f"Frozen action tasks are incomplete for {condition}.")
        summary_successes = int(summary_rows[condition]["successes"])
        if successes != summary_successes:
            raise ValueError(f"Round-4C summary/action files disagree for {condition}.")
        outcome_counts[condition] = successes
        identities.append(
            {
                "condition": condition,
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "task_results": result_hashes,
            }
        )
    assert shared_identity is not None
    if (
        shared_identity["checkpoint_sha256"] != EXPECTED["checkpoint"]
        or shared_identity["subspace_basis_manifest_sha256"] != EXPECTED["basis"]
        or shared_identity["calibration_split_manifest_sha256"] != EXPECTED["split"]
        or shared_identity["donor_mapping_sha256"] != EXPECTED["donor_mapping"]
        or shared_identity["donor_observation_manifest_sha256"] != EXPECTED["donor_manifest"]
    ):
        raise ValueError("Round-4C frozen action identity is not the registered one.")
    return {
        "conditions": list(CONDITIONS),
        "successes": outcome_counts,
        "episodes_per_condition": 100,
        "shared_identity": shared_identity,
        "artifacts": identities,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    git = _validate_clean(output_root)
    audit_path = args.architecture_audit.resolve()
    audit = _read(audit_path)
    if not (
        audit.get("static_audit_passed") is True
        and audit.get("status") == "preferred_path_a_pending_runtime_machinery"
        and audit.get("git_commit_hash") == git_commit(PROJECT_ROOT)
    ):
        raise ValueError("Phase-A static architecture audit did not establish Preferred Path A.")
    for source in audit.get("source_inventory", []):
        source_path = PROJECT_ROOT / str(source["path"])
        if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"Phase-A audited source drifted: {source_path}")
    basis_path = args.round4b_root.resolve() / "calibration/basis_manifest.json"
    split_path = args.round4b_root.resolve() / "calibration/calibration_split_manifest.json"
    _require_hash(basis_path, EXPECTED["basis"], "Round-4B basis")
    _require_hash(split_path, EXPECTED["split"], "Round-4B split")
    basis = validate_basis_manifest(
        basis_path, expected_sha256=EXPECTED["basis"], verify_files=True
    )
    round4c_summary_path = args.round4c_root.resolve() / "aggregate/round4c_summary.json"
    _require_hash(round4c_summary_path, EXPECTED["round4c_summary"], "Round-4C summary")
    round4c_summary = _read(round4c_summary_path)
    if round4c_summary.get("status") != "complete":
        raise ValueError("Round-4C action summary is incomplete.")
    action = _validate_action_results(args.round4c_root.resolve(), round4c_summary)
    action_impl = _validate_action_implementation_identity()
    _require_hash(args.valid_manifest.resolve(), EXPECTED["valid_manifest"], "valid manifest")
    valid = _read(args.valid_manifest.resolve())
    source_manifest = Path(str(valid["source_manifest_path"])).resolve()
    if valid.get("source_manifest_sha256") != EXPECTED["state_bank_manifest"]:
        raise ValueError("Frozen valid/source state-bank identity drifted.")
    _require_hash(
        source_manifest, EXPECTED["state_bank_manifest"], "state-bank source manifest"
    )
    for stem in ("checkpoint", "dataset_stats", "prompt_context_cache"):
        path = Path(str(valid[f"{stem}_path"])).resolve()
        _require_hash(path, str(valid[f"{stem}_sha256"]), f"state-bank {stem}")
    if valid["checkpoint_sha256"] != EXPECTED["checkpoint"]:
        raise ValueError("Checkpoint identity differs from frozen Round-4C.")
    _require_hash(args.donor_mapping.resolve(), EXPECTED["donor_mapping"], "donor mapping")
    _require_hash(args.donor_manifest.resolve(), EXPECTED["donor_manifest"], "donor manifest")
    bundle = OnlineDonorBundle.load(
        mapping_path=args.donor_mapping.resolve(),
        observation_manifest_path=args.donor_manifest.resolve(),
        observation_root=args.donor_root.resolve(),
    )
    if len(bundle.mappings) != 100:
        raise ValueError("Frozen donor bundle must cover 100 task/trial pairs.")
    for observation in bundle.observations.values():
        artifact = (bundle.observation_root / observation["artifact_relative_path"]).resolve()
        _require_hash(artifact, observation["artifact_sha256"], "donor observation artifact")
    dataset_root = args.dataset_root.resolve()
    info_path = dataset_root / "meta/info.json"
    tasks_path = dataset_root / "meta/tasks.parquet"
    if not info_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(f"Official LIBERO-Spatial v3 data is unavailable: {dataset_root}")
    info = _read(info_path)
    if (
        info.get("codebase_version") != "v3.0"
        or int(info.get("total_episodes", -1)) != 434
        or int(info.get("total_frames", -1)) != 53229
        or int(info.get("fps", -1)) != 20
        or int(info.get("total_tasks", -1)) != 10
        or info.get("splits") != {"train": "0:434"}
    ):
        raise ValueError(f"Unexpected official LIBERO-Spatial dataset metadata: {info_path}")
    features = info.get("features", {})
    expected_features = {
        "observation.images.image": ("video", [512, 512, 3]),
        "observation.images.wrist_image": ("video", [512, 512, 3]),
        "observation.state": ("float32", [8]),
        "action": ("float32", [7]),
    }
    if any(
        key not in features
        or features[key].get("dtype") != dtype
        or features[key].get("shape") != shape
        for key, (dtype, shape) in expected_features.items()
    ):
        raise ValueError("Official world-data feature schema drifted.")
    _require_hash(
        args.salvage_a_summary.resolve(),
        EXPECTED["salvage_a_summary"],
        "Salvage-A summary",
    )
    salvage_a = _read(args.salvage_a_summary.resolve())
    if not (
        salvage_a.get("status") == "complete"
        and salvage_a.get("classification", {}).get("classification") == "WEAK"
        and salvage_a.get("classification", {}).get("compact_asre_status") == "closed"
    ):
        raise ValueError("Salvage A must be complete/WEAK/closed before the final gate.")
    return {
        "artifact_type": "asre_salvage_b_preflight_report",
        "schema_version": 1,
        "protocol": SALVAGE_B_PROTOCOL,
        "status": "compatible",
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "output_root": str(output_root),
        "git": git,
        "phase_a": {
            "path": "preferred_path_a",
            "architecture_audit_path": str(audit_path),
            "architecture_audit_sha256": sha256_file(audit_path),
            "runtime_machinery_required_before_phase_b": True,
        },
        "state": {
            "valid_manifest_path": str(args.valid_manifest.resolve()),
            "valid_manifest_sha256": EXPECTED["valid_manifest"],
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": EXPECTED["state_bank_manifest"],
            "checkpoint_path": valid["checkpoint_path"],
            "checkpoint_sha256": valid["checkpoint_sha256"],
            "dataset_stats_path": valid["dataset_stats_path"],
            "dataset_stats_sha256": valid["dataset_stats_sha256"],
            "prompt_context_cache_path": valid["prompt_context_cache_path"],
            "prompt_context_cache_sha256": valid["prompt_context_cache_sha256"],
        },
        "basis": {
            "path": str(basis_path),
            "sha256": EXPECTED["basis"],
            "split_path": str(split_path),
            "split_sha256": EXPECTED["split"],
            "runtime_layout": basis["runtime_layout"],
            "ranks": [97, 170],
            "refit": False,
        },
        "donors": {
            "mapping_path": str(args.donor_mapping.resolve()),
            "mapping_sha256": EXPECTED["donor_mapping"],
            "manifest_path": str(args.donor_manifest.resolve()),
            "manifest_sha256": EXPECTED["donor_manifest"],
            "root": str(args.donor_root.resolve()),
            "same_round4c_pool_and_derangement": True,
        },
        "frozen_action": {
            "round4c_root": str(args.round4c_root.resolve()),
            "summary_path": str(round4c_summary_path),
            "summary_sha256": EXPECTED["round4c_summary"],
            "results": action,
            "implementation_identity": action_impl,
            "rerun_required": False,
        },
        "world_data": {
            "dataset_root": str(dataset_root),
            "info_path": str(info_path),
            "info_sha256": sha256_file(info_path),
            "tasks_path": str(tasks_path),
            "tasks_sha256": sha256_file(tasks_path),
            "official_split_note": (
                "official local dataset has no separate test split; evaluation is strictly "
                "held out from basis fitting and frozen before outcomes"
            ),
            "native_future_targets_available": True,
        },
        "salvage_a": {
            "summary_path": str(args.salvage_a_summary.resolve()),
            "summary_sha256": sha256_file(args.salvage_a_summary.resolve()),
            "classification": "WEAK",
            "compact_asre_status": "closed",
        },
        "scope": {
            "conditions": list(CONDITIONS),
            "primary_condition": "svd_r170",
            "action_rerun": False,
            "world_only_gpu_evaluation": True,
            "later_asre_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--architecture-audit", type=Path, required=True)
    parser.add_argument("--round4b-root", type=Path, required=True)
    parser.add_argument("--round4c-root", type=Path, required=True)
    parser.add_argument("--salvage-a-summary", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--donor-mapping", type=Path, required=True)
    parser.add_argument("--donor-manifest", type=Path, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    atomic_write_json(args.output.resolve(), report)
    print(f"Salvage-B preflight compatible: {args.output.resolve()}")


if __name__ == "__main__":
    main()
