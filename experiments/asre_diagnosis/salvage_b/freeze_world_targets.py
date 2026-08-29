"""Decode and hash the exact processed world targets before any GPU metric."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from collections import Counter
from pathlib import Path
from typing import Any


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
from experiments.asre_diagnosis.salvage_b.world_runtime import (  # noqa: E402
    load_processed_sample,
    load_prompt_cache,
    load_world_dataset,
    read_json,
    validate_declared_source_artifacts,
)
from experiments.asre_diagnosis.round3b.donor import tensor_sha256  # noqa: E402


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen processed world targets: {output}"
        )
    preflight_path = args.preflight.resolve()
    world_path = args.world_manifest.resolve()
    preflight = read_json(preflight_path)
    world = read_json(world_path)
    current_commit = git_commit(PROJECT_ROOT)
    preflight_sha256 = sha256_file(preflight_path)
    world_sha256 = sha256_file(world_path)
    if (
        preflight.get("status") != "compatible"
        or preflight.get("git_commit_hash") != current_commit
        or world.get("status") != "frozen_before_metrics"
        or int(world.get("schema_version", -1)) != 2
        or world.get("git_commit_hash") != current_commit
        or world.get("preflight_report_sha256") != preflight_sha256
        or world.get("sample_count") != 100
    ):
        raise ValueError("Incompatible Salvage-B preflight/world manifest.")
    validate_declared_source_artifacts(world, include_targets=True)
    dataset = load_world_dataset(
        preflight=preflight,
        runtime_work_dir=args.runtime_work_dir.resolve(),
    )
    prompt_cache = load_prompt_cache(preflight)
    records = []
    task_counts: Counter[int] = Counter()
    for position, record in enumerate(world["records"], start=1):
        processed = load_processed_sample(
            dataset=dataset,
            record=record,
            prompt_cache=prompt_cache,
        )
        task_id = int(record["task_id"])
        task_counts[task_id] += 1
        records.append(
            {
                "sample_id": str(record["sample_id"]),
                "task_id": task_id,
                "episode_id": int(record["episode_id"]),
                "trial": int(record["trial"]),
                "dataset_index": int(record["dataset_index"]),
                "processed_tensor_sha256": processed["processed_tensor_sha256"],
                "current_image_sha256": tensor_sha256(
                    processed["video"][:, :, 0].detach().cpu()
                ),
                "video_shape": list(processed["video"].shape[1:]),
                "action_shape": list(processed["action"].shape[1:]),
                "proprio_shape": list(processed["proprio"].shape[1:]),
                "contains_padding": False,
            }
        )
        if position % 20 == 0:
            print(f"[SalvageB target freeze] {position}/100", flush=True)
    if task_counts != Counter({task_id: 10 for task_id in range(10)}):
        raise ValueError(f"World target task stratification drifted: {task_counts}")
    # Decoding all clips can be long.  Re-hash every declared metadata/data/video
    # source after the pass as well, and reject manifest mutation during decoding.
    validate_declared_source_artifacts(world, include_targets=True)
    if (
        sha256_file(preflight_path) != preflight_sha256
        or sha256_file(world_path) != world_sha256
    ):
        raise ValueError("Frozen target inputs changed while processed targets were decoded.")
    return {
        "artifact_type": "asre_salvage_b_processed_world_target_manifest",
        "schema_version": 2,
        "protocol": SALVAGE_B_PROTOCOL,
        "status": "frozen_before_gpu_metrics",
        "created_at": now_iso(),
        "git_commit_hash": current_commit,
        "preflight_report_path": str(preflight_path),
        "preflight_report_sha256": preflight_sha256,
        "world_manifest_path": str(world_path),
        "world_manifest_sha256": world_sha256,
        "sample_count": len(records),
        "task_counts": {str(key): value for key, value in sorted(task_counts.items())},
        "records": records,
        "hash_scope": (
            "deterministically decoded and normalized 9-frame two-camera video, "
            "32-step action/proprio/padding tensors, and exact prompt"
        ),
        "source_artifact_verification": {
            "before_decode": True,
            "after_decode": True,
            "metadata_file_count": len(world["metadata_files"]),
            "target_source_file_count": len(world["target_source_files"]),
            "checks": "resolved regular file, byte size, and SHA-256",
        },
        "decoder_and_runtime_versions": {
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "av": _version("av"),
            "pandas": _version("pandas"),
        },
        "model_outcomes_inspected": False,
        "gpu_metric_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--world-manifest", type=Path, required=True)
    parser.add_argument("--runtime-work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    atomic_write_json(args.output.resolve(), report)
    print(f"Frozen processed world targets: {args.output.resolve()}")


if __name__ == "__main__":
    main()
