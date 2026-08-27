"""Deterministic donor artifacts and validation for ASRE Round 3B.

The online control stores only the model-ready visual tensor captured at the
first policy query.  At each recipient replan the model recomputes donor K/V
from this fixed image while retaining the recipient's current proprioceptive
and text context.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from experiments.asre_diagnosis.common import atomic_write_json, sha256_file


DONOR_SCHEMA_VERSION = 1
DONOR_MAPPING_RULE = "donor_trial=(recipient_trial+1)%num_trials"
DONOR_OBSERVATION_MANIFEST_NAME = "donor_observation_manifest.json"
DONOR_MAPPING_NAME = "donor_mapping.json"


def _update_canonical_hash(digest: "hashlib._Hash", value: Any) -> None:
    """Hash nested tensor-like simulator state without pickle metadata."""

    if torch.is_tensor(value):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(b"torch\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        if tensor.dtype == torch.bfloat16:
            raw = tensor.view(torch.uint16).numpy().tobytes(order="C")
        else:
            raw = tensor.numpy().tobytes(order="C")
        digest.update(raw)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"numpy\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_canonical_hash(digest, str(key))
            _update_canonical_hash(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _update_canonical_hash(digest, item)
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        )
        return
    raise TypeError(f"Unsupported value in canonical artifact hash: {type(value)!r}")


def canonical_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_canonical_hash(digest, value)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    if not torch.is_tensor(tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}.")
    return canonical_sha256(tensor)


def task_text_sha256(task_description: str) -> str:
    return hashlib.sha256(str(task_description).encode("utf-8")).hexdigest()


def tensor_is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach().to(dtype=torch.float32, device="cpu")).all())


def image_difference(
    recipient_image: torch.Tensor,
    donor_image: torch.Tensor,
) -> dict[str, float]:
    if tuple(recipient_image.shape) != tuple(donor_image.shape):
        raise ValueError(
            "Recipient/donor image shapes differ: "
            f"{tuple(recipient_image.shape)} != {tuple(donor_image.shape)}."
        )
    recipient = recipient_image.detach().to(device="cpu", dtype=torch.float32)
    donor = donor_image.detach().to(device="cpu", dtype=torch.float32)
    if not tensor_is_finite(recipient) or not tensor_is_finite(donor):
        raise ValueError("Recipient/donor image tensors must be finite.")
    difference = recipient - donor
    return {
        "processed_pixel_mae": float(difference.abs().mean().item()),
        "processed_pixel_mse": float(difference.square().mean().item()),
    }


def donor_trial_for(recipient_trial: int, num_trials: int) -> int:
    if num_trials < 2:
        raise ValueError(f"A donor derangement requires at least two trials, got {num_trials}.")
    if recipient_trial < 0 or recipient_trial >= num_trials:
        raise ValueError(
            f"recipient_trial must be in [0, {num_trials - 1}], got {recipient_trial}."
        )
    return (recipient_trial + 1) % num_trials


def donor_artifact_relative_path(task_id: int, trial: int) -> Path:
    return Path("observations") / f"task{int(task_id):02d}_trial{int(trial):02d}.pt"


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is unavailable: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _records_by_key(
    records: Iterable[Mapping[str, Any]],
    *,
    trial_key: str,
    label: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} records must be JSON objects.")
        key = (int(record["task_id"]), int(record[trial_key]))
        if key in indexed:
            raise ValueError(f"Duplicate {label} record for task/trial {key}.")
        indexed[key] = dict(record)
    return indexed


def validate_observation_manifest(payload: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    if int(payload.get("schema_version", -1)) != DONOR_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported donor observation manifest schema: "
            f"{payload.get('schema_version')!r}."
        )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Donor observation manifest records must be a non-empty list.")
    indexed = _records_by_key(
        records,
        trial_key="source_trial",
        label="donor observation manifest",
    )
    expected_count = int(payload.get("num_tasks", 0)) * int(payload.get("num_trials", 0))
    if expected_count <= 0 or len(indexed) != expected_count:
        raise ValueError(
            "Donor observation manifest count mismatch: "
            f"records={len(indexed)}, expected={expected_count}."
        )
    num_trials = int(payload["num_trials"])
    declared_task_ids = payload.get("task_ids")
    if declared_task_ids is None:
        task_ids = sorted({task_id for task_id, _trial in indexed})
    else:
        if not isinstance(declared_task_ids, list):
            raise ValueError("Donor observation manifest task_ids must be a list.")
        task_ids = [int(task_id) for task_id in declared_task_ids]
    if len(task_ids) != int(payload["num_tasks"]) or len(set(task_ids)) != len(task_ids):
        raise ValueError(f"Invalid donor observation task_ids: {task_ids}.")
    expected_keys = {
        (task_id, trial) for task_id in task_ids for trial in range(num_trials)
    }
    if set(indexed) != expected_keys:
        raise ValueError("Donor observation manifest does not cover every declared task/trial.")
    for key, record in indexed.items():
        if record.get("source_task_id") is not None and int(record["source_task_id"]) != key[0]:
            raise ValueError(f"Donor observation task identity mismatch for {key}.")
        shape = record.get("processed_image_shape")
        if not isinstance(shape, list) or not shape or any(int(dim) <= 0 for dim in shape):
            raise ValueError(f"Invalid processed image shape for donor {key}: {shape!r}.")
        if record.get("processed_image_finite") is not True:
            raise ValueError(f"Donor observation is not marked finite for {key}.")
        image_min = float(record.get("processed_image_min", float("nan")))
        image_max = float(record.get("processed_image_max", float("nan")))
        if (
            not math.isfinite(image_min)
            or not math.isfinite(image_max)
            or image_min < -1.001
            or image_max > 1.001
            or image_min > image_max
        ):
            raise ValueError(
                f"Donor observation has invalid normalized image range for {key}: "
                f"[{image_min}, {image_max}]."
            )
        for digest_key in (
            "task_text_sha256",
            "initial_state_sha256",
            "processed_image_sha256",
            "artifact_sha256",
        ):
            value = str(record.get(digest_key, ""))
            if len(value) != 64:
                raise ValueError(f"Invalid {digest_key} for donor {key}: {value!r}.")
        if not record.get("artifact_relative_path"):
            raise ValueError(f"Missing artifact_relative_path for donor {key}.")
        if not record.get("artifact_absolute_path"):
            raise ValueError(f"Missing artifact_absolute_path for donor {key}.")
    return indexed


def validate_donor_mapping(
    payload: Mapping[str, Any],
    observation_manifest: Mapping[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    if int(payload.get("schema_version", -1)) != DONOR_SCHEMA_VERSION:
        raise ValueError(f"Unsupported donor mapping schema: {payload.get('schema_version')!r}.")
    if str(payload.get("mapping_rule")) != DONOR_MAPPING_RULE:
        raise ValueError(f"Unexpected donor mapping rule: {payload.get('mapping_rule')!r}.")
    num_trials = int(payload.get("num_trials", 0))
    num_tasks = int(payload.get("num_tasks", 0))
    records = payload.get("records")
    if num_trials < 2 or num_tasks <= 0 or not isinstance(records, list):
        raise ValueError("Donor mapping has invalid task/trial counts or records.")
    indexed = _records_by_key(records, trial_key="recipient_trial", label="donor mapping")
    if len(indexed) != num_tasks * num_trials:
        raise ValueError(
            f"Donor mapping count mismatch: records={len(indexed)}, "
            f"expected={num_tasks * num_trials}."
        )
    observations = validate_observation_manifest(observation_manifest)
    if set(indexed) != set(observations):
        raise ValueError("Donor mapping recipient keys differ from observation manifest keys.")
    for (task_id, recipient_trial), record in indexed.items():
        donor_trial = int(record["donor_trial"])
        expected_donor = donor_trial_for(recipient_trial, num_trials)
        if donor_trial != expected_donor or donor_trial == recipient_trial:
            raise ValueError(
                f"Donor mapping is not the declared derangement for task {task_id}, "
                f"recipient {recipient_trial}: donor={donor_trial}, expected={expected_donor}."
            )
        recipient = observations[(task_id, recipient_trial)]
        donor = observations.get((task_id, donor_trial))
        if donor is None:
            raise ValueError(
                f"Missing same-task donor observation for task {task_id}, trial {donor_trial}."
            )
        required_pairs = {
            "recipient_initial_state_sha256": recipient["initial_state_sha256"],
            "donor_initial_state_sha256": donor["initial_state_sha256"],
            "recipient_image_sha256": recipient["processed_image_sha256"],
            "donor_image_sha256": donor["processed_image_sha256"],
            "task_text_sha256": recipient["task_text_sha256"],
        }
        for key, expected in required_pairs.items():
            if record.get(key) != expected:
                raise ValueError(
                    f"Donor mapping {key} mismatch for task {task_id}, "
                    f"recipient {recipient_trial}."
                )
        if donor["task_text_sha256"] != recipient["task_text_sha256"]:
            raise ValueError(
                f"Recipient/donor task text differs for task {task_id}, "
                f"recipient {recipient_trial}."
            )
        if (
            donor["processed_image_shape"] != recipient["processed_image_shape"]
            or donor["processed_image_dtype"] != recipient["processed_image_dtype"]
        ):
            raise ValueError(
                f"Recipient/donor model-ready image structure differs for task {task_id}, "
                f"recipient {recipient_trial}."
            )
        if record.get("image_hash_identical") is not False:
            raise ValueError(
                f"Donor image must be non-identical for task {task_id}, "
                f"recipient {recipient_trial}."
            )
        if recipient["processed_image_sha256"] == donor["processed_image_sha256"]:
            raise ValueError(
                f"Recipient/donor image hashes are identical for task {task_id}, "
                f"recipient {recipient_trial}."
            )
        for key in ("processed_pixel_mae", "processed_pixel_mse"):
            value = float(record.get(key, float("nan")))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Donor mapping {key} must be positive and finite for "
                    f"task {task_id}, recipient {recipient_trial}."
                )
    return indexed


def build_donor_mapping_payload(
    observation_manifest: Mapping[str, Any],
    *,
    observation_manifest_path: Path,
    observation_manifest_sha256: str,
    images_by_key: Mapping[tuple[int, int], torch.Tensor],
) -> dict[str, Any]:
    observations = validate_observation_manifest(observation_manifest)
    num_tasks = int(observation_manifest["num_tasks"])
    num_trials = int(observation_manifest["num_trials"])
    if set(images_by_key) != set(observations):
        raise ValueError("In-memory donor image keys differ from observation manifest keys.")
    records: list[dict[str, Any]] = []
    for task_id, recipient_trial in sorted(observations):
        donor_trial = donor_trial_for(recipient_trial, num_trials)
        recipient = observations[(task_id, recipient_trial)]
        donor = observations[(task_id, donor_trial)]
        differences = image_difference(
            images_by_key[(task_id, recipient_trial)],
            images_by_key[(task_id, donor_trial)],
        )
        image_hash_identical = (
            recipient["processed_image_sha256"] == donor["processed_image_sha256"]
        )
        if image_hash_identical:
            raise ValueError(
                f"Accidentally identical donor image for task {task_id}, "
                f"recipient {recipient_trial}, donor {donor_trial}."
            )
        records.append(
            {
                "task_id": task_id,
                "recipient_trial": recipient_trial,
                "donor_trial": donor_trial,
                "recipient_initial_state_sha256": recipient["initial_state_sha256"],
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
    payload = {
        "schema_version": DONOR_SCHEMA_VERSION,
        "task_suite": observation_manifest["task_suite"],
        "seed": observation_manifest["seed"],
        "num_tasks": num_tasks,
        "num_trials": num_trials,
        "task_ids": list(observation_manifest.get("task_ids", sorted({key[0] for key in observations}))),
        "mapping_rule": DONOR_MAPPING_RULE,
        "donor_observation_manifest_path": str(observation_manifest_path.resolve()),
        "donor_observation_manifest_sha256": observation_manifest_sha256,
        "records": records,
    }
    validate_donor_mapping(payload, observation_manifest)
    return payload


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a torch artifact; never overwrite an existing file."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen donor artifact: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@dataclass(frozen=True)
class LoadedDonor:
    image: torch.Tensor
    provenance: dict[str, Any]


@dataclass(frozen=True)
class OnlineDonorBundle:
    mapping_path: Path
    mapping_sha256: str
    observation_manifest_path: Path
    observation_manifest_sha256: str
    observation_root: Path
    mapping_payload: dict[str, Any]
    observation_payload: dict[str, Any]
    mappings: dict[tuple[int, int], dict[str, Any]]
    observations: dict[tuple[int, int], dict[str, Any]]

    @classmethod
    def load(
        cls,
        *,
        mapping_path: Path,
        observation_manifest_path: Path,
        observation_root: Path | None = None,
        expected_mapping_sha256: str | None = None,
        expected_observation_manifest_sha256: str | None = None,
    ) -> "OnlineDonorBundle":
        mapping_path = mapping_path.expanduser().resolve()
        observation_manifest_path = observation_manifest_path.expanduser().resolve()
        root = (
            observation_manifest_path.parent
            if observation_root is None
            else observation_root.expanduser().resolve()
        )
        mapping_digest = sha256_file(mapping_path)
        manifest_digest = sha256_file(observation_manifest_path)
        if expected_mapping_sha256 is not None and mapping_digest != expected_mapping_sha256:
            raise ValueError(
                "Donor mapping SHA256 mismatch: "
                f"observed={mapping_digest}, expected={expected_mapping_sha256}."
            )
        if (
            expected_observation_manifest_sha256 is not None
            and manifest_digest != expected_observation_manifest_sha256
        ):
            raise ValueError(
                "Donor observation manifest SHA256 mismatch: "
                f"observed={manifest_digest}, "
                f"expected={expected_observation_manifest_sha256}."
            )
        mapping = _read_json_object(mapping_path, "donor mapping")
        observations = _read_json_object(
            observation_manifest_path, "donor observation manifest"
        )
        recorded_manifest_digest = mapping.get("donor_observation_manifest_sha256")
        if recorded_manifest_digest != manifest_digest:
            raise ValueError(
                "Donor mapping was frozen against a different observation manifest: "
                f"recorded={recorded_manifest_digest}, observed={manifest_digest}."
            )
        observation_index = validate_observation_manifest(observations)
        mapping_index = validate_donor_mapping(mapping, observations)
        return cls(
            mapping_path=mapping_path,
            mapping_sha256=mapping_digest,
            observation_manifest_path=observation_manifest_path,
            observation_manifest_sha256=manifest_digest,
            observation_root=root,
            mapping_payload=mapping,
            observation_payload=observations,
            mappings=mapping_index,
            observations=observation_index,
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
            raise KeyError(f"No frozen donor mapping for recipient {key}.")
        observed_recipient_hash = canonical_sha256(recipient_initial_state)
        if mapping["recipient_initial_state_sha256"] != observed_recipient_hash:
            raise ValueError(
                f"Recipient initial-state hash mismatch for {key}: "
                f"observed={observed_recipient_hash}, "
                f"frozen={mapping['recipient_initial_state_sha256']}."
            )
        observed_text_hash = task_text_sha256(task_description)
        if mapping["task_text_sha256"] != observed_text_hash:
            raise ValueError(
                f"Task text hash mismatch for donor recipient {key}: "
                f"observed={observed_text_hash}, frozen={mapping['task_text_sha256']}."
            )
        donor_key = (key[0], int(mapping["donor_trial"]))
        observation = self.observations[donor_key]
        relative_path = Path(str(observation["artifact_relative_path"]))
        if relative_path.is_absolute():
            raise ValueError(
                f"Donor artifact_relative_path must be relative: {relative_path}."
            )
        artifact_path = (self.observation_root / relative_path).resolve()
        if not artifact_path.is_relative_to(self.observation_root):
            raise ValueError(f"Donor artifact escapes configured root: {artifact_path}.")
        if sha256_file(artifact_path) != observation["artifact_sha256"]:
            raise ValueError(f"Donor artifact SHA256 mismatch: {artifact_path}.")
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not torch.is_tensor(payload.get("input_image")):
            raise ValueError(f"Malformed donor observation artifact: {artifact_path}.")
        image = payload["input_image"].detach().to(device="cpu").contiguous()
        if tensor_sha256(image) != observation["processed_image_sha256"]:
            raise ValueError(f"Donor image SHA256 mismatch: {artifact_path}.")
        if list(image.shape) != list(observation["processed_image_shape"]):
            raise ValueError(f"Donor image shape mismatch: {artifact_path}.")
        if str(image.dtype) != str(observation["processed_image_dtype"]):
            raise ValueError(f"Donor image dtype mismatch: {artifact_path}.")
        if not tensor_is_finite(image):
            raise ValueError(f"Donor image contains non-finite values: {artifact_path}.")
        payload_identity = (
            str(payload.get("task_suite")),
            int(payload.get("task_id", -1)),
            int(payload.get("source_trial", -1)),
        )
        expected_identity = (
            str(self.observation_payload["task_suite"]),
            donor_key[0],
            donor_key[1],
        )
        if payload_identity != expected_identity:
            raise ValueError(
                f"Donor artifact identity mismatch: {payload_identity} != {expected_identity}."
            )
        provenance = {
            "recipient_task_id": key[0],
            "recipient_trial": key[1],
            "donor_task_id": donor_key[0],
            "donor_trial": donor_key[1],
            "recipient_initial_state_sha256": observed_recipient_hash,
            "donor_initial_state_sha256": mapping["donor_initial_state_sha256"],
            "task_text_sha256": observed_text_hash,
            "recipient_image_sha256": mapping["recipient_image_sha256"],
            "donor_image_sha256": mapping["donor_image_sha256"],
            "processed_pixel_mae": float(mapping["processed_pixel_mae"]),
            "processed_pixel_mse": float(mapping["processed_pixel_mse"]),
            "donor_artifact_path": str(artifact_path),
            "donor_artifact_sha256": observation["artifact_sha256"],
        }
        return LoadedDonor(image=image.to(device=device, dtype=dtype), provenance=provenance)


def write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create immutable-by-policy JSON, accepting only byte-equivalent reruns."""

    path = path.resolve()
    if path.exists():
        existing = _read_json_object(path, path.name)
        if existing != dict(payload):
            raise FileExistsError(f"Refusing to overwrite incompatible frozen file: {path}")
        return
    atomic_write_json(path, payload)
