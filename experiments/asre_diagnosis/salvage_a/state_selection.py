"""Freeze exact calibration states and diagnostic halves for ASRE Salvage A."""

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
from experiments.asre_diagnosis.salvage_a.split import (  # noqa: E402
    validate_split_manifest,
    validate_split_payload,
)


STABILITY_SEED = 4211
STATES_PER_CALIBRATION_EPISODE = 2


def _stability_score(seed: int, task_id: int, episode_id: int) -> str:
    return hashlib.sha256(
        f"salvage-a-stability-half\0{seed}\0{task_id}\0{episode_id}".encode(
            "ascii"
        )
    ).hexdigest()


def _record_key(record: Mapping[str, Any]) -> tuple[int, str]:
    return int(record["replan_id"]), str(record["sample_id"])


def build_state_selection_manifest(
    *,
    split_manifest: Mapping[str, Any],
    split_manifest_path: Path,
    source_records: Sequence[Mapping[str, Any]],
    seed: int = STABILITY_SEED,
) -> dict[str, Any]:
    split = validate_split_payload(split_manifest)
    valid_calibration_ids = set(str(value) for value in split["calibration_sample_ids"])
    valid_heldout_ids = set(str(value) for value in split["heldout_sample_ids"])
    by_episode: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    heldout_by_episode: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in source_records:
        identifier = str(record["sample_id"])
        if identifier in valid_calibration_ids:
            by_episode[(int(record["task_id"]), int(record["episode_id"]))].append(
                record
            )
        elif identifier in valid_heldout_ids:
            heldout_by_episode[
                (int(record["task_id"]), int(record["episode_id"]))
            ].append(record)
    expected_episodes = {
        (int(task["task_id"]), int(episode))
        for task in split["per_task"]
        for episode in task["calibration_episode_ids"]
    }
    if set(by_episode) != expected_episodes or len(expected_episodes) != 50:
        raise ValueError("Calibration state records do not cover exactly 50 frozen episodes.")
    expected_heldout_episodes = {
        (int(task["task_id"]), int(episode))
        for task in split["per_task"]
        for episode in task["heldout_episode_ids"]
    }
    if (
        set(heldout_by_episode) != expected_heldout_episodes
        or len(expected_heldout_episodes) != 50
    ):
        raise ValueError("Held-out state records do not cover exactly 50 frozen episodes.")

    selected_records: list[dict[str, Any]] = []
    selected_by_episode: list[dict[str, Any]] = []
    selected_ids_by_episode: dict[tuple[int, int], list[str]] = {}
    for task_id, episode_id in sorted(expected_episodes):
        records = sorted(by_episode[(task_id, episode_id)], key=_record_key)
        if len(records) < STATES_PER_CALIBRATION_EPISODE:
            raise ValueError(
                f"Episode {(task_id, episode_id)} has fewer than two valid replans."
            )
        chosen = (records[0], records[-1])
        chosen_ids = [str(record["sample_id"]) for record in chosen]
        if len(set(chosen_ids)) != STATES_PER_CALIBRATION_EPISODE:
            raise ValueError(f"Earliest/latest states collapse for {(task_id, episode_id)}.")
        selected_ids_by_episode[(task_id, episode_id)] = chosen_ids
        selected_by_episode.append(
            {
                "task_id": task_id,
                "episode_id": episode_id,
                "valid_replan_ids": [int(record["replan_id"]) for record in records],
                "selected_sample_ids": chosen_ids,
                "selected_replan_ids": [int(record["replan_id"]) for record in chosen],
            }
        )
        selected_records.extend(
            {
                "sample_id": str(record["sample_id"]),
                "task_id": task_id,
                "episode_id": episode_id,
                "replan_id": int(record["replan_id"]),
            }
            for record in chosen
        )

    heldout_selected_ids: list[str] = []
    heldout_selected_by_episode: list[dict[str, Any]] = []
    for task_id, episode_id in sorted(expected_heldout_episodes):
        records = sorted(heldout_by_episode[(task_id, episode_id)], key=_record_key)
        if len(records) < STATES_PER_CALIBRATION_EPISODE:
            raise ValueError(
                f"Held-out episode {(task_id, episode_id)} has fewer than two valid replans."
            )
        chosen = (records[0], records[-1])
        chosen_ids = [str(record["sample_id"]) for record in chosen]
        if len(set(chosen_ids)) != 2:
            raise ValueError(
                f"Held-out earliest/latest states collapse for {(task_id, episode_id)}."
            )
        heldout_selected_ids.extend(chosen_ids)
        heldout_selected_by_episode.append(
            {
                "task_id": task_id,
                "episode_id": episode_id,
                "valid_replan_ids": [int(record["replan_id"]) for record in records],
                "selected_sample_ids": chosen_ids,
                "selected_replan_ids": [int(record["replan_id"]) for record in chosen],
            }
        )

    half_a_episodes: set[tuple[int, int]] = set()
    half_b_episodes: set[tuple[int, int]] = set()
    stability_per_task: list[dict[str, Any]] = []
    for task in sorted(split["per_task"], key=lambda item: int(item["task_id"])):
        task_id = int(task["task_id"])
        ordered = sorted(
            (int(value) for value in task["calibration_episode_ids"]),
            key=lambda episode: _stability_score(seed, task_id, episode),
        )
        half_a_count = 3 if task_id % 2 == 0 else 2
        half_a = tuple(sorted(ordered[:half_a_count]))
        half_b = tuple(sorted(ordered[half_a_count:]))
        half_a_episodes.update((task_id, episode) for episode in half_a)
        half_b_episodes.update((task_id, episode) for episode in half_b)
        stability_per_task.append(
            {
                "task_id": task_id,
                "episode_order_by_seeded_hash": ordered,
                "half_a_episode_ids": list(half_a),
                "half_b_episode_ids": list(half_b),
            }
        )
    if (
        len(half_a_episodes) != 25
        or len(half_b_episodes) != 25
        or half_a_episodes & half_b_episodes
        or half_a_episodes | half_b_episodes != expected_episodes
    ):
        raise AssertionError("Stability halves are not a disjoint 25/25 partition.")

    def ids_for(episodes: set[tuple[int, int]]) -> list[str]:
        return [
            identifier
            for key in sorted(episodes)
            for identifier in selected_ids_by_episode[key]
        ]

    selected_ids = [record["sample_id"] for record in selected_records]
    half_a_ids = ids_for(half_a_episodes)
    half_b_ids = ids_for(half_b_episodes)
    half_a_keys = [f"task{task:02d}_episode{episode:02d}" for task, episode in sorted(half_a_episodes)]
    half_b_keys = [f"task{task:02d}_episode{episode:02d}" for task, episode in sorted(half_b_episodes)]
    payload: dict[str, Any] = {
        "artifact_type": "asre_salvage_a_state_selection_manifest",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "created_at": now_iso(),
        "split_manifest_path": str(split_manifest_path.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest_path.resolve()),
        # ``split_sha256`` is the compact identity consumed by the shard
        # launcher.  Keep the explicit manifest spelling above for humans and
        # require the two aliases to remain identical during validation.
        "split_sha256": sha256_file(split_manifest_path.resolve()),
        "source_manifest_path": str(Path(str(split["source_manifest_path"])).resolve()),
        "source_manifest_sha256": str(split["source_manifest_sha256"]),
        "selection_rule": (
            "earliest and latest valid saved replan per frozen episode; calibration "
            "states fit bases and held-out states are diagnostics only"
        ),
        "selection_reason": (
            "pre-registered runtime reduction for exact full 10-step differentiable "
            "action inference; no outcome or action-deviation criterion is used"
        ),
        "outcome_independent": True,
        "same_exact_states_for_svd_and_actionaware": True,
        "states_per_calibration_episode": STATES_PER_CALIBRATION_EPISODE,
        "calibration_episode_clusters": 50,
        "selected_state_count": len(selected_ids),
        "heldout_episode_clusters": 50,
        "heldout_state_count": len(heldout_selected_ids),
        "selected_sample_ids": selected_ids,
        "calibration_sample_ids": selected_ids,
        "heldout_sample_ids": heldout_selected_ids,
        "selected_records": selected_records,
        "selected_by_episode": selected_by_episode,
        "heldout_selected_by_episode": heldout_selected_by_episode,
        "stability_seed": int(seed),
        "stability_rule": (
            "within each task order five calibration episodes by seeded hash; "
            "even task IDs assign 3/2 to A/B and odd task IDs assign 2/3"
        ),
        "stability_per_task": stability_per_task,
        "stability_half_a_episode_clusters": 25,
        "stability_half_b_episode_clusters": 25,
        "stability_half_a_episode_keys": half_a_keys,
        "stability_half_b_episode_keys": half_b_keys,
        "stability_half_a_sample_ids": half_a_ids,
        "stability_half_b_sample_ids": half_b_ids,
    }
    payload["selection_sha256"] = sha256_json(selected_ids)
    payload["heldout_selection_sha256"] = sha256_json(heldout_selected_ids)
    payload["stability_partition_sha256"] = sha256_json(
        {"half_a_sample_ids": half_a_ids, "half_b_sample_ids": half_b_ids}
    )
    validate_state_selection_payload(payload, split_manifest=split)
    return payload


def validate_state_selection_payload(
    payload: Mapping[str, Any], *, split_manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    expected = {
        "artifact_type": "asre_salvage_a_state_selection_manifest",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "outcome_independent": True,
        "same_exact_states_for_svd_and_actionaware": True,
        "states_per_calibration_episode": 2,
        "calibration_episode_clusters": 50,
        "selected_state_count": 100,
        "heldout_episode_clusters": 50,
        "heldout_state_count": 100,
        "stability_half_a_episode_clusters": 25,
        "stability_half_b_episode_clusters": 25,
    }
    mismatch = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Salvage-A state selection mismatch: {mismatch}")
    if payload.get("split_sha256") != payload.get("split_manifest_sha256"):
        raise ValueError("Salvage-A state selection split SHA aliases disagree.")
    selected = [str(value) for value in payload.get("selected_sample_ids", ())]
    calibration = [str(value) for value in payload.get("calibration_sample_ids", ())]
    heldout = [str(value) for value in payload.get("heldout_sample_ids", ())]
    half_a = [str(value) for value in payload.get("stability_half_a_sample_ids", ())]
    half_b = [str(value) for value in payload.get("stability_half_b_sample_ids", ())]
    if (
        len(selected) != 100
        or calibration != selected
        or len(heldout) != 100
        or len(set(heldout)) != 100
        or set(calibration) & set(heldout)
        or len(set(selected)) != 100
        or len(half_a) != 50
        or len(half_b) != 50
        or set(half_a) & set(half_b)
        or set(half_a) | set(half_b) != set(selected)
    ):
        raise ValueError("Salvage-A exact state IDs or stability halves are malformed.")
    if payload.get("selection_sha256") != sha256_json(selected):
        raise ValueError("Salvage-A selected-state digest mismatch.")
    if payload.get("heldout_selection_sha256") != sha256_json(heldout):
        raise ValueError("Salvage-A held-out state digest mismatch.")
    if payload.get("stability_partition_sha256") != sha256_json(
        {"half_a_sample_ids": half_a, "half_b_sample_ids": half_b}
    ):
        raise ValueError("Salvage-A stability partition digest mismatch.")
    records = payload.get("selected_by_episode")
    if not isinstance(records, list) or len(records) != 50:
        raise ValueError("Salvage-A state selection lacks 50 episode records.")
    keys = {(int(row["task_id"]), int(row["episode_id"])) for row in records}
    if (
        len(keys) != 50
        or any(len(row["selected_sample_ids"]) != 2 for row in records)
        or any(
            list(row["selected_replan_ids"])
            != [min(row["valid_replan_ids"]), max(row["valid_replan_ids"])]
            for row in records
        )
    ):
        raise ValueError("Salvage-A per-episode state selection is malformed.")
    heldout_records = payload.get("heldout_selected_by_episode")
    if not isinstance(heldout_records, list) or len(heldout_records) != 50:
        raise ValueError("Salvage-A state selection lacks 50 held-out episode records.")
    heldout_keys = {
        (int(row["task_id"]), int(row["episode_id"])) for row in heldout_records
    }
    if (
        len(heldout_keys) != 50
        or any(len(row["selected_sample_ids"]) != 2 for row in heldout_records)
        or any(
            list(row["selected_replan_ids"])
            != [min(row["valid_replan_ids"]), max(row["valid_replan_ids"])]
            for row in heldout_records
        )
    ):
        raise ValueError("Salvage-A held-out state selection is malformed.")
    half_a_keys = payload.get("stability_half_a_episode_keys")
    half_b_keys = payload.get("stability_half_b_episode_keys")
    if (
        not isinstance(half_a_keys, list)
        or not isinstance(half_b_keys, list)
        or len(half_a_keys) != 25
        or len(half_b_keys) != 25
        or set(half_a_keys) & set(half_b_keys)
    ):
        raise ValueError("Salvage-A stability episode keys are malformed.")
    if split_manifest is not None:
        split = validate_split_payload(split_manifest)
        calibration_ids = set(str(value) for value in split["calibration_sample_ids"])
        heldout_ids = set(str(value) for value in split["heldout_sample_ids"])
        if not set(selected) <= calibration_ids:
            raise ValueError("Selected basis states escape the calibration partition.")
        if not set(heldout) <= heldout_ids:
            raise ValueError("Selected diagnostic states escape the held-out partition.")
        expected_keys = {
            (int(task["task_id"]), int(episode))
            for task in split["per_task"]
            for episode in task["calibration_episode_ids"]
        }
        if keys != expected_keys:
            raise ValueError("Selected basis episodes differ from the calibration split.")
        expected_heldout_keys = {
            (int(task["task_id"]), int(episode))
            for task in split["per_task"]
            for episode in task["heldout_episode_ids"]
        }
        if heldout_keys != expected_heldout_keys:
            raise ValueError("Selected diagnostic episodes differ from the held-out split.")
        expected_key_strings = {
            f"task{task:02d}_episode{episode:02d}" for task, episode in expected_keys
        }
        if set(half_a_keys) | set(half_b_keys) != expected_key_strings:
            raise ValueError("Stability episode keys do not partition calibration episodes.")
    return dict(payload)


def validate_state_selection_manifest(
    path: Path,
    *,
    expected_sha256: str | None = None,
    split_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Salvage-A state selection SHA256 mismatch.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return validate_state_selection_payload(payload, split_manifest=split_manifest)


def write_state_selection(*, split_path: Path, output: Path) -> dict[str, Any]:
    split_path = split_path.resolve()
    split = validate_split_manifest(split_path)
    source_path = Path(str(split["source_manifest_path"])).resolve()
    payload = build_state_selection_manifest(
        split_manifest=split,
        split_manifest_path=split_path,
        source_records=load_manifest(source_path),
    )
    output = output.resolve()
    if output.exists():
        existing = validate_state_selection_manifest(output, split_manifest=split)
        comparable = dict(payload)
        comparable["created_at"] = existing.get("created_at")
        if existing != comparable:
            raise FileExistsError(
                f"Refusing to overwrite incompatible state selection: {output}"
            )
        return existing
    atomic_write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_state_selection(split_path=args.split, output=args.output)
    print(
        "Frozen Salvage-A basis states: "
        f"{payload['selected_state_count']} states, stability halves 25/25 episodes"
    )


if __name__ == "__main__":
    main()
