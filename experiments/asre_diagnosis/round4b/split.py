"""Freeze the outcome-independent task-stratified Round-4B episode split."""

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
    ROUND4B_PROTOCOL,
    atomic_write_json,
    load_manifest,
    now_iso,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round2.validate_state_bank import (  # noqa: E402
    _validate_sample_partition,
)


SPLIT_SEED = 4204
FIT_EPISODES_PER_TASK = 8
HOLDOUT_EPISODES_PER_TASK = 2


def _episode_score(seed: int, task_id: int, episode_id: int) -> str:
    return hashlib.sha256(
        f"round4b-split\0{seed}\0{task_id}\0{episode_id}".encode("ascii")
    ).hexdigest()


def build_split_manifest(
    *,
    valid_manifest: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    valid_manifest_path: Path,
    source_manifest_path: Path,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    valid_ids = _validate_sample_partition(valid_manifest, source_records)
    by_id = {str(record["sample_id"]): record for record in source_records}
    episodes: dict[int, set[int]] = defaultdict(set)
    for identifier in valid_ids:
        record = by_id[identifier]
        episodes[int(record["task_id"])].add(int(record["episode_id"]))
    if set(episodes) != set(range(10)) or any(
        values != set(range(10)) for values in episodes.values()
    ):
        raise ValueError("Round-4B requires ten tasks with episode IDs 0..9.")

    per_task: list[dict[str, Any]] = []
    fit_keys: set[tuple[int, int]] = set()
    holdout_keys: set[tuple[int, int]] = set()
    for task_id in range(10):
        ordered = sorted(
            episodes[task_id], key=lambda episode: _episode_score(seed, task_id, episode)
        )
        fit = tuple(sorted(ordered[:FIT_EPISODES_PER_TASK]))
        holdout = tuple(sorted(ordered[FIT_EPISODES_PER_TASK:]))
        if len(fit) != 8 or len(holdout) != 2 or set(fit) & set(holdout):
            raise AssertionError("Invalid task-stratified Round-4B split.")
        fit_keys.update((task_id, episode) for episode in fit)
        holdout_keys.update((task_id, episode) for episode in holdout)
        per_task.append(
            {
                "task_id": task_id,
                "fit_episode_ids": list(fit),
                "holdout_episode_ids": list(holdout),
                "episode_order_by_seeded_hash": ordered,
            }
        )

    fit_ids = [
        identifier
        for identifier in valid_ids
        if (int(by_id[identifier]["task_id"]), int(by_id[identifier]["episode_id"]))
        in fit_keys
    ]
    holdout_ids = [identifier for identifier in valid_ids if identifier not in set(fit_ids)]
    if set(fit_ids) & set(holdout_ids) or set(fit_ids) | set(holdout_ids) != set(valid_ids):
        raise AssertionError("Round-4B sample split does not partition the 499 valid states.")
    payload: dict[str, Any] = {
        "artifact_type": "asre_round4b_calibration_split_manifest",
        "schema_version": 1,
        "protocol": ROUND4B_PROTOCOL,
        "created_at": now_iso(),
        "selection_rule": (
            "within each task, sort episode IDs by SHA256(round4b-split,seed,task,episode); "
            "first 8 fit, final 2 holdout"
        ),
        "outcome_independent": True,
        "seed": int(seed),
        "valid_manifest_path": str(valid_manifest_path.resolve()),
        "valid_manifest_sha256": sha256_file(valid_manifest_path.resolve()),
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest_path.resolve()),
        "num_valid_samples": len(valid_ids),
        "num_tasks": 10,
        "num_episode_clusters": 100,
        "fit_episode_clusters": 80,
        "holdout_episode_clusters": 20,
        "fit_sample_count": len(fit_ids),
        "holdout_sample_count": len(holdout_ids),
        "per_task": per_task,
        "fit_sample_ids": fit_ids,
        "holdout_sample_ids": holdout_ids,
    }
    payload["partition_sha256"] = sha256_json(
        {"fit_sample_ids": fit_ids, "holdout_sample_ids": holdout_ids}
    )
    return payload


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
        existing = json.loads(output.read_text(encoding="utf-8"))
        # Creation time is provenance, not split identity.
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
        f"Frozen Round-4B split: fit={payload['fit_sample_count']} states/80 episodes, "
        f"holdout={payload['holdout_sample_count']} states/20 episodes"
    )


if __name__ == "__main__":
    main()
