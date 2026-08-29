"""Freeze the task-stratified 50/50 episode split for ASRE Salvage A."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_A_PROTOCOL,
    atomic_write_json,
    load_manifest,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (  # noqa: E402
    _validate_sample_partition,
)


SPLIT_SEED = 4210
CALIBRATION_EPISODES_PER_TASK = 5
HELDOUT_EPISODES_PER_TASK = 5


def _episode_score(seed: int, task_id: int, episode_id: int) -> str:
    return hashlib.sha256(
        f"salvage-a-split\0{seed}\0{task_id}\0{episode_id}".encode("ascii")
    ).hexdigest()


def _episode_sets(payload: Mapping[str, Any]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    calibration: set[tuple[int, int]] = set()
    heldout: set[tuple[int, int]] = set()
    per_task = payload.get("per_task")
    if not isinstance(per_task, list) or len(per_task) != 10:
        raise ValueError("Salvage-A split must contain ten per-task records.")
    for record in per_task:
        task_id = int(record["task_id"])
        calibration_ids = tuple(int(value) for value in record["calibration_episode_ids"])
        heldout_ids = tuple(int(value) for value in record["heldout_episode_ids"])
        if (
            len(calibration_ids) != CALIBRATION_EPISODES_PER_TASK
            or len(set(calibration_ids)) != CALIBRATION_EPISODES_PER_TASK
            or len(heldout_ids) != HELDOUT_EPISODES_PER_TASK
            or len(set(heldout_ids)) != HELDOUT_EPISODES_PER_TASK
            or set(calibration_ids) & set(heldout_ids)
            or set(calibration_ids) | set(heldout_ids) != set(range(10))
        ):
            raise ValueError(f"Malformed 5/5 split for task {task_id}.")
        calibration.update((task_id, episode) for episode in calibration_ids)
        heldout.update((task_id, episode) for episode in heldout_ids)
    if {task for task, _ in calibration | heldout} != set(range(10)):
        raise ValueError("Salvage-A split does not cover task IDs 0..9.")
    if len(calibration) != 50 or len(heldout) != 50 or calibration & heldout:
        raise ValueError("Salvage-A split must be a disjoint 50/50 episode partition.")
    return calibration, heldout


def build_split_manifest(
    *,
    valid_manifest: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    valid_manifest_path: Path,
    source_manifest_path: Path,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Build an outcome-independent split while keeping every episode cluster intact."""

    valid_ids = _validate_sample_partition(valid_manifest, source_records)
    by_id = {str(record["sample_id"]): record for record in source_records}
    episodes: dict[int, set[int]] = defaultdict(set)
    for identifier in valid_ids:
        record = by_id[identifier]
        episodes[int(record["task_id"])].add(int(record["episode_id"]))
    if set(episodes) != set(range(10)) or any(
        episode_ids != set(range(10)) for episode_ids in episodes.values()
    ):
        raise ValueError("Salvage A requires ten tasks with episode IDs 0..9.")

    per_task: list[dict[str, Any]] = []
    calibration_keys: set[tuple[int, int]] = set()
    heldout_keys: set[tuple[int, int]] = set()
    for task_id in range(10):
        ordered = sorted(
            episodes[task_id],
            key=lambda episode: _episode_score(seed, task_id, episode),
        )
        calibration = tuple(sorted(ordered[:CALIBRATION_EPISODES_PER_TASK]))
        heldout = tuple(sorted(ordered[CALIBRATION_EPISODES_PER_TASK:]))
        calibration_keys.update((task_id, episode) for episode in calibration)
        heldout_keys.update((task_id, episode) for episode in heldout)
        per_task.append(
            {
                "task_id": task_id,
                "calibration_episode_ids": list(calibration),
                "heldout_episode_ids": list(heldout),
                "episode_order_by_seeded_hash": ordered,
            }
        )

    calibration_ids = [
        identifier
        for identifier in valid_ids
        if (
            int(by_id[identifier]["task_id"]),
            int(by_id[identifier]["episode_id"]),
        )
        in calibration_keys
    ]
    calibration_set = set(calibration_ids)
    heldout_ids = [identifier for identifier in valid_ids if identifier not in calibration_set]
    if (
        calibration_set & set(heldout_ids)
        or calibration_set | set(heldout_ids) != set(valid_ids)
    ):
        raise AssertionError("Salvage-A sample lists do not partition all valid states.")

    payload: dict[str, Any] = {
        "artifact_type": "asre_salvage_a_calibration_split_manifest",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "created_at": now_iso(),
        "selection_rule": (
            "within each task, sort episode IDs by "
            "SHA256(salvage-a-split,seed,task,episode); first 5 calibration, "
            "final 5 held-out"
        ),
        "outcome_independent": True,
        "episode_cluster_integrity": True,
        "seed": int(seed),
        "valid_manifest_path": str(valid_manifest_path.resolve()),
        "valid_manifest_sha256": sha256_file(valid_manifest_path.resolve()),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path.resolve()),
        "num_valid_samples": len(valid_ids),
        "num_tasks": 10,
        "num_episode_clusters": 100,
        "calibration_episode_clusters": 50,
        "heldout_episode_clusters": 50,
        "calibration_sample_count": len(calibration_ids),
        "heldout_sample_count": len(heldout_ids),
        "per_task": per_task,
        "calibration_sample_ids": calibration_ids,
        "heldout_sample_ids": heldout_ids,
    }
    payload["partition_sha256"] = sha256_json(
        {
            "calibration_sample_ids": calibration_ids,
            "heldout_sample_ids": heldout_ids,
        }
    )
    _episode_sets(payload)
    return payload


def validate_split_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "artifact_type": "asre_salvage_a_calibration_split_manifest",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "outcome_independent": True,
        "episode_cluster_integrity": True,
        "num_tasks": 10,
        "num_episode_clusters": 100,
        "calibration_episode_clusters": 50,
        "heldout_episode_clusters": 50,
    }
    mismatch = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Salvage-A split manifest mismatch: {mismatch}")
    _episode_sets(payload)
    calibration_ids = list(payload.get("calibration_sample_ids", ()))
    heldout_ids = list(payload.get("heldout_sample_ids", ()))
    if (
        len(calibration_ids) != int(payload.get("calibration_sample_count", -1))
        or len(heldout_ids) != int(payload.get("heldout_sample_count", -1))
        or set(calibration_ids) & set(heldout_ids)
        or len(calibration_ids) + len(heldout_ids) != int(payload.get("num_valid_samples", -1))
    ):
        raise ValueError("Salvage-A split sample lists are malformed.")
    expected_partition = sha256_json(
        {
            "calibration_sample_ids": calibration_ids,
            "heldout_sample_ids": heldout_ids,
        }
    )
    if payload.get("partition_sha256") != expected_partition:
        raise ValueError("Salvage-A split partition digest mismatch.")
    return dict(payload)


def validate_split_manifest(
    path: Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Salvage-A split manifest SHA256 mismatch.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return validate_split_payload(payload)


def write_split(*, valid_manifest_path: Path, output: Path) -> dict[str, Any]:
    valid_manifest_path = valid_manifest_path.resolve()
    valid = json.loads(valid_manifest_path.read_text(encoding="utf-8"))
    source_path = Path(str(valid["source_manifest_path"])).resolve()
    payload = build_split_manifest(
        valid_manifest=valid,
        source_records=load_manifest(source_path),
        valid_manifest_path=valid_manifest_path,
        source_manifest_path=source_path,
    )
    output = output.resolve()
    if output.exists():
        existing = validate_split_manifest(output)
        comparable = dict(payload)
        comparable["created_at"] = existing.get("created_at")
        if existing != comparable:
            raise FileExistsError(f"Refusing to overwrite incompatible split: {output}")
        return existing
    atomic_write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_split(valid_manifest_path=args.valid_manifest, output=args.output)
    print(
        "Frozen Salvage-A split: "
        f"calibration={payload['calibration_sample_count']} states/50 episodes, "
        f"heldout={payload['heldout_sample_count']} states/50 episodes"
    )


if __name__ == "__main__":
    main()
