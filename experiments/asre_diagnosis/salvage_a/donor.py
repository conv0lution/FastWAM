"""Split-local deterministic donor mapping for ASRE Salvage A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import (  # noqa: E402
    SALVAGE_A_PROTOCOL,
    sha256_file,
)
from experiments.asre_diagnosis.round3b.donor import (  # noqa: E402
    LoadedDonor,
    canonical_sha256,
    image_difference,
    task_text_sha256,
    tensor_is_finite,
    tensor_sha256,
    validate_observation_manifest,
    write_frozen_json,
)
from experiments.asre_diagnosis.salvage_a.split import (  # noqa: E402
    validate_split_manifest,
    validate_split_payload,
)


DONOR_SEED = 4212
DONOR_MAPPING_RULE = (
    "within each task and frozen split, order five trials by "
    "SHA256(salvage-a-donor,seed,partition,task,trial) and map each to the next"
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _donor_score(
    seed: int, partition: str, task_id: int, episode_id: int
) -> str:
    return hashlib.sha256(
        (
            f"salvage-a-donor\0{seed}\0{partition}\0{task_id}\0{episode_id}"
        ).encode("ascii")
    ).hexdigest()


def _partition_lookup(
    split: Mapping[str, Any],
) -> tuple[dict[tuple[int, int], str], dict[tuple[int, str], tuple[int, ...]]]:
    split = validate_split_payload(split)
    lookup: dict[tuple[int, int], str] = {}
    groups: dict[tuple[int, str], tuple[int, ...]] = {}
    for record in split["per_task"]:
        task_id = int(record["task_id"])
        for partition, key in (
            ("calibration", "calibration_episode_ids"),
            ("heldout", "heldout_episode_ids"),
        ):
            episodes = tuple(int(value) for value in record[key])
            groups[(task_id, partition)] = episodes
            for episode in episodes:
                identity = (task_id, episode)
                if identity in lookup:
                    raise ValueError(f"Duplicate split episode key: {identity}")
                lookup[identity] = partition
    if len(lookup) != 100:
        raise ValueError("Split-local donors require exactly 100 recipient trials.")
    return lookup, groups


def _observation_artifact_path(
    record: Mapping[str, Any], *, observation_root: Path
) -> Path:
    relative = Path(str(record["artifact_relative_path"]))
    if relative.is_absolute():
        raise ValueError(f"Donor artifact_relative_path must be relative: {relative}")
    path = (observation_root / relative).resolve()
    if not path.is_relative_to(observation_root):
        raise ValueError(f"Donor artifact escapes observation root: {path}")
    return path


def _load_observation_image(
    record: Mapping[str, Any], *, observation_root: Path
) -> torch.Tensor:
    path = _observation_artifact_path(record, observation_root=observation_root)
    if sha256_file(path) != record["artifact_sha256"]:
        raise ValueError(f"Donor observation artifact SHA256 mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    image = payload.get("input_image") if isinstance(payload, Mapping) else None
    if not torch.is_tensor(image) or not tensor_is_finite(image):
        raise ValueError(f"Malformed donor observation artifact: {path}")
    image = image.detach().cpu().contiguous()
    if tensor_sha256(image) != record["processed_image_sha256"]:
        raise ValueError(f"Donor image identity mismatch: {path}")
    if list(image.shape) != list(record["processed_image_shape"]):
        raise ValueError(f"Donor image shape mismatch: {path}")
    if str(image.dtype) != str(record["processed_image_dtype"]):
        raise ValueError(f"Donor image dtype mismatch: {path}")
    return image


def build_split_local_donor_mapping(
    *,
    split_manifest: Mapping[str, Any],
    split_manifest_path: Path,
    observation_manifest: Mapping[str, Any],
    observation_manifest_path: Path,
    observation_root: Path,
    seed: int = DONOR_SEED,
) -> dict[str, Any]:
    split = validate_split_payload(split_manifest)
    observations = validate_observation_manifest(observation_manifest)
    partition_by_key, groups = _partition_lookup(split)
    if set(observations) != set(partition_by_key):
        raise ValueError("Frozen donor observations do not cover the split task/trial keys.")
    observation_root = observation_root.resolve()
    images = {
        key: _load_observation_image(record, observation_root=observation_root)
        for key, record in observations.items()
    }

    records: list[dict[str, Any]] = []
    order_records: list[dict[str, Any]] = []
    for task_id in range(10):
        for partition in ("calibration", "heldout"):
            ordered = sorted(
                groups[(task_id, partition)],
                key=lambda episode: _donor_score(
                    seed, partition, task_id, episode
                ),
            )
            if len(ordered) != 5:
                raise ValueError(
                    f"Split-local donor group {(task_id, partition)} does not have five trials."
                )
            order_records.append(
                {
                    "task_id": task_id,
                    "partition": partition,
                    "episode_order_by_seeded_hash": ordered,
                }
            )
            donor_for = {
                recipient: ordered[(index + 1) % len(ordered)]
                for index, recipient in enumerate(ordered)
            }
            for recipient_trial in sorted(ordered):
                donor_trial = donor_for[recipient_trial]
                recipient = observations[(task_id, recipient_trial)]
                donor = observations[(task_id, donor_trial)]
                differences = image_difference(
                    images[(task_id, recipient_trial)],
                    images[(task_id, donor_trial)],
                )
                if recipient["processed_image_sha256"] == donor["processed_image_sha256"]:
                    raise ValueError(
                        f"Identical split-local donor image for {(task_id, recipient_trial)}."
                    )
                records.append(
                    {
                        "task_id": task_id,
                        "recipient_trial": recipient_trial,
                        "donor_task_id": task_id,
                        "donor_trial": donor_trial,
                        "recipient_partition": partition,
                        "donor_partition": partition,
                        "recipient_initial_state_sha256": recipient[
                            "initial_state_sha256"
                        ],
                        "donor_initial_state_sha256": donor["initial_state_sha256"],
                        "task_text_sha256": recipient["task_text_sha256"],
                        "recipient_image_sha256": recipient["processed_image_sha256"],
                        "donor_image_sha256": donor["processed_image_sha256"],
                        "image_hash_identical": False,
                        "processed_image_shape": list(recipient["processed_image_shape"]),
                        "processed_image_dtype": recipient["processed_image_dtype"],
                        **differences,
                    }
                )
    records.sort(key=lambda row: (int(row["task_id"]), int(row["recipient_trial"])))
    payload = {
        "artifact_type": "asre_salvage_a_split_local_donor_mapping",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "task_suite": observation_manifest["task_suite"],
        "seed": observation_manifest["seed"],
        "mapping_seed": int(seed),
        "num_tasks": 10,
        "num_trials": 10,
        "task_ids": list(range(10)),
        "mapping_rule": DONOR_MAPPING_RULE,
        "split_manifest_path": str(split_manifest_path.resolve()),
        "split_manifest_sha256": sha256_file(split_manifest_path.resolve()),
        "split_sha256": sha256_file(split_manifest_path.resolve()),
        "donor_observation_manifest_path": str(
            observation_manifest_path.resolve()
        ),
        "donor_observation_manifest_sha256": sha256_file(
            observation_manifest_path.resolve()
        ),
        "split_local": True,
        "same_task": True,
        "derangement_verified": True,
        "orders": order_records,
        "records": records,
    }
    validate_split_local_donor_mapping(
        payload,
        observation_manifest=observation_manifest,
        split_manifest=split,
    )
    return payload


def validate_split_local_donor_mapping(
    payload: Mapping[str, Any],
    *,
    observation_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    expected = {
        "artifact_type": "asre_salvage_a_split_local_donor_mapping",
        "schema_version": 1,
        "protocol": SALVAGE_A_PROTOCOL,
        "mapping_rule": DONOR_MAPPING_RULE,
        "num_tasks": 10,
        "num_trials": 10,
        "split_local": True,
        "same_task": True,
        "derangement_verified": True,
    }
    mismatch = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"Salvage-A donor mapping mismatch: {mismatch}")
    if payload.get("split_sha256") != payload.get("split_manifest_sha256"):
        raise ValueError("Salvage-A donor mapping split SHA aliases disagree.")
    observations = validate_observation_manifest(observation_manifest)
    partition_by_key, groups = _partition_lookup(split_manifest)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("Salvage-A donor mapping must contain 100 records.")
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        key = (int(record["task_id"]), int(record["recipient_trial"]))
        if key in indexed:
            raise ValueError(f"Duplicate Salvage-A donor recipient: {key}")
        indexed[key] = record
    if set(indexed) != set(observations) or set(indexed) != set(partition_by_key):
        raise ValueError("Salvage-A donor recipients do not cover all frozen trials.")

    orders = payload.get("orders")
    if not isinstance(orders, list) or len(orders) != 20:
        raise ValueError("Salvage-A donor mapping lacks 20 task/partition orders.")
    order_lookup = {
        (int(row["task_id"]), str(row["partition"])): tuple(
            int(value) for value in row["episode_order_by_seeded_hash"]
        )
        for row in orders
    }
    for group_key, members in groups.items():
        ordered = order_lookup.get(group_key)
        if ordered is None or set(ordered) != set(members) or len(ordered) != 5:
            raise ValueError(f"Malformed donor order for {group_key}.")
        expected_donors = {
            recipient: ordered[(index + 1) % 5]
            for index, recipient in enumerate(ordered)
        }
        task_id, partition = group_key
        for recipient_trial in members:
            record = indexed[(task_id, recipient_trial)]
            donor_trial = int(record["donor_trial"])
            donor_key = (task_id, donor_trial)
            if (
                donor_trial != expected_donors[recipient_trial]
                or donor_trial == recipient_trial
                or int(record.get("donor_task_id", -1)) != task_id
                or partition_by_key[donor_key] != partition
                or record.get("recipient_partition") != partition
                or record.get("donor_partition") != partition
            ):
                raise ValueError(
                    f"Donor is not the frozen split-local derangement for "
                    f"{(task_id, recipient_trial)}."
                )
            recipient = observations[(task_id, recipient_trial)]
            donor = observations[donor_key]
            required = {
                "recipient_initial_state_sha256": recipient["initial_state_sha256"],
                "donor_initial_state_sha256": donor["initial_state_sha256"],
                "task_text_sha256": recipient["task_text_sha256"],
                "recipient_image_sha256": recipient["processed_image_sha256"],
                "donor_image_sha256": donor["processed_image_sha256"],
            }
            if any(record.get(key) != value for key, value in required.items()):
                raise ValueError(
                    f"Donor provenance mismatch for {(task_id, recipient_trial)}."
                )
            if donor["task_text_sha256"] != recipient["task_text_sha256"]:
                raise ValueError("Split-local donor task text differs from recipient.")
            if record.get("image_hash_identical") is not False:
                raise ValueError("Split-local donor image must be non-identical.")
            for metric in ("processed_pixel_mae", "processed_pixel_mse"):
                value = float(record.get(metric, float("nan")))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid donor {metric} for {(task_id, recipient_trial)}.")
    return indexed


@dataclass(frozen=True)
class SalvageADonorBundle:
    mapping_path: Path
    mapping_sha256: str
    observation_manifest_path: Path
    observation_manifest_sha256: str
    observation_root: Path
    mapping_payload: dict[str, Any]
    observation_payload: dict[str, Any]
    split_payload: dict[str, Any]
    mappings: dict[tuple[int, int], dict[str, Any]]
    observations: dict[tuple[int, int], dict[str, Any]]

    @classmethod
    def load(
        cls,
        mapping_path: Path,
        observation_manifest_path: Path,
        observation_root: Path | None = None,
        expected_mapping_sha256: str | None = None,
        expected_observation_manifest_sha256: str | None = None,
    ) -> "SalvageADonorBundle":
        mapping_path = mapping_path.expanduser().resolve()
        observation_manifest_path = observation_manifest_path.expanduser().resolve()
        root = (
            observation_manifest_path.parent
            if observation_root is None
            else observation_root.expanduser().resolve()
        )
        mapping_sha = sha256_file(mapping_path)
        observation_sha = sha256_file(observation_manifest_path)
        if expected_mapping_sha256 is not None and mapping_sha != expected_mapping_sha256:
            raise ValueError("Salvage-A donor mapping SHA256 mismatch.")
        if (
            expected_observation_manifest_sha256 is not None
            and observation_sha != expected_observation_manifest_sha256
        ):
            raise ValueError("Salvage-A donor observation manifest SHA256 mismatch.")
        mapping = _read_json(mapping_path, "Salvage-A donor mapping")
        observations_payload = _read_json(
            observation_manifest_path, "donor observation manifest"
        )
        if mapping.get("donor_observation_manifest_sha256") != observation_sha:
            raise ValueError("Salvage-A mapping references a different observation manifest.")
        split_path = Path(str(mapping["split_manifest_path"])).resolve()
        split = validate_split_manifest(
            split_path, expected_sha256=str(mapping["split_manifest_sha256"])
        )
        mappings = validate_split_local_donor_mapping(
            mapping,
            observation_manifest=observations_payload,
            split_manifest=split,
        )
        return cls(
            mapping_path=mapping_path,
            mapping_sha256=mapping_sha,
            observation_manifest_path=observation_manifest_path,
            observation_manifest_sha256=observation_sha,
            observation_root=root,
            mapping_payload=mapping,
            observation_payload=observations_payload,
            split_payload=split,
            mappings=mappings,
            observations=validate_observation_manifest(observations_payload),
        )

    def load_for_recipient(
        self,
        *,
        task_id: int,
        recipient_trial: int,
        recipient_initial_state: Any,
        task_description: str,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> LoadedDonor:
        key = (int(task_id), int(recipient_trial))
        mapping = self.mappings.get(key)
        if mapping is None:
            raise KeyError(f"No split-local donor mapping for recipient {key}.")
        state_hash = canonical_sha256(recipient_initial_state)
        if state_hash != mapping["recipient_initial_state_sha256"]:
            raise ValueError(f"Recipient initial-state hash mismatch for {key}.")
        text_hash = task_text_sha256(task_description)
        if text_hash != mapping["task_text_sha256"]:
            raise ValueError(f"Recipient task text hash mismatch for {key}.")
        donor_key = (key[0], int(mapping["donor_trial"]))
        record = self.observations[donor_key]
        image = _load_observation_image(record, observation_root=self.observation_root)
        provenance = {
            "recipient_task_id": key[0],
            "recipient_trial": key[1],
            "recipient_partition": mapping["recipient_partition"],
            "donor_task_id": donor_key[0],
            "donor_trial": donor_key[1],
            "donor_partition": mapping["donor_partition"],
            "recipient_initial_state_sha256": state_hash,
            "donor_initial_state_sha256": mapping["donor_initial_state_sha256"],
            "task_text_sha256": text_hash,
            "recipient_image_sha256": mapping["recipient_image_sha256"],
            "donor_image_sha256": mapping["donor_image_sha256"],
            "processed_pixel_mae": float(mapping["processed_pixel_mae"]),
            "processed_pixel_mse": float(mapping["processed_pixel_mse"]),
            "donor_artifact_path": str(
                _observation_artifact_path(
                    record, observation_root=self.observation_root
                )
            ),
            "donor_artifact_sha256": record["artifact_sha256"],
        }
        return LoadedDonor(
            image=image.to(device=device, dtype=dtype), provenance=provenance
        )


def load_donor_image(
    bundle: SalvageADonorBundle, *, task_id: int, episode_id: int
) -> torch.Tensor:
    """Load the fixed donor image for an offline saved-state recipient."""

    key = (int(task_id), int(episode_id))
    mapping = bundle.mappings.get(key)
    if mapping is None:
        raise KeyError(f"No split-local donor mapping for saved episode {key}.")
    donor_key = (key[0], int(mapping["donor_trial"]))
    return _load_observation_image(
        bundle.observations[donor_key], observation_root=bundle.observation_root
    )


def write_mapping(
    *,
    split_path: Path,
    observation_manifest_path: Path,
    observation_root: Path,
    output: Path,
) -> dict[str, Any]:
    split_path = split_path.resolve()
    observation_manifest_path = observation_manifest_path.resolve()
    split = validate_split_manifest(split_path)
    observations = _read_json(observation_manifest_path, "donor observation manifest")
    payload = build_split_local_donor_mapping(
        split_manifest=split,
        split_manifest_path=split_path,
        observation_manifest=observations,
        observation_manifest_path=observation_manifest_path,
        observation_root=observation_root.resolve(),
    )
    write_frozen_json(output.resolve(), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--observation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_mapping(
        split_path=args.split,
        observation_manifest_path=args.observation_manifest,
        observation_root=args.observation_root,
        output=args.output,
    )
    print(
        f"Frozen Salvage-A split-local donor mapping: {len(payload['records'])} recipients"
    )


if __name__ == "__main__":
    main()
