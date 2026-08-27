"""Deterministic same-task/same-replan donor mapping for the 499-state bank."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.asre_diagnosis.common import load_manifest, sha256_file, sha256_json


OFFLINE_DONOR_SCHEMA_VERSION = 1


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor identity, metadata, and exact contiguous CPU bytes."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a tensor, got {type(tensor).__name__}.")
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_image(sample: Mapping[str, Any], *, sample_id: str) -> torch.Tensor:
    infer_kwargs = sample.get("infer_action_kwargs")
    if not isinstance(infer_kwargs, Mapping):
        raise TypeError(f"Sample {sample_id} has no infer_action_kwargs mapping.")
    image = infer_kwargs.get("input_image")
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Sample {sample_id} has no input_image tensor.")
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(
            f"Sample {sample_id} input_image must have shape [1,C,H,W], got "
            f"{list(image.shape)}."
        )
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError(f"Sample {sample_id} input_image contains NaN or Inf.")
    return image.detach().cpu()


def build_offline_donor_pairs(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Map every valid state to the next episode at the same task/replan."""

    if not records:
        raise ValueError("Cannot build an offline donor mapping from zero records.")
    by_group: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id:
            raise ValueError("Every offline donor record must have a sample_id.")
        if sample_id in seen:
            raise ValueError(f"Duplicate state-bank sample_id {sample_id!r}.")
        seen.add(sample_id)
        group = (
            str(record.get("task_suite", "")),
            int(record.get("task_id", -1)),
            int(record.get("replan_id", -1)),
        )
        by_group[group].append(record)

    pairs: dict[str, str] = {}
    for group, members in sorted(by_group.items()):
        ordered = sorted(members, key=lambda item: (int(item["episode_id"]), str(item["sample_id"])))
        episode_ids = [int(item["episode_id"]) for item in ordered]
        if len(ordered) < 2 or len(set(episode_ids)) != len(ordered):
            raise ValueError(
                "Each task/replan donor group needs at least two distinct episodes; "
                f"group={group}, episodes={episode_ids}."
            )
        for index, recipient in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)]
            recipient_id = str(recipient["sample_id"])
            donor_id = str(donor["sample_id"])
            if donor_id == recipient_id or int(donor["episode_id"]) == int(
                recipient["episode_id"]
            ):
                raise AssertionError(f"Offline donor derangement failed for {recipient_id}.")
            pairs[recipient_id] = donor_id
    if set(pairs) != seen:
        raise AssertionError("Offline donor mapping does not cover every valid state.")
    return pairs


def build_offline_donor_manifest(
    *,
    state_bank_dir: Path,
    source_manifest_path: Path,
    valid_manifest_path: Path,
) -> dict[str, Any]:
    """Load, audit, and describe the pre-outcome offline donor mapping."""

    state_bank_dir = state_bank_dir.resolve()
    source_manifest_path = source_manifest_path.resolve()
    valid_manifest_path = valid_manifest_path.resolve()
    with valid_manifest_path.open("r", encoding="utf-8") as handle:
        valid_manifest = json.load(handle)
    source_records = load_manifest(source_manifest_path)
    by_id = {str(record["sample_id"]): record for record in source_records}
    valid_ids = [str(value) for value in valid_manifest.get("valid_sample_ids", [])]
    if len(valid_ids) != 499 or len(set(valid_ids)) != 499:
        raise ValueError(
            "Round-3B requires exactly 499 unique QC-valid sample IDs; "
            f"got {len(valid_ids)} entries and {len(set(valid_ids))} unique IDs."
        )
    missing = [sample_id for sample_id in valid_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Valid manifest references missing source samples: {missing}.")
    if sha256_file(source_manifest_path) != str(valid_manifest.get("source_manifest_sha256")):
        raise ValueError("Source state-bank manifest SHA256 disagrees with valid manifest.")

    selected = [by_id[sample_id] for sample_id in valid_ids]
    pairs = build_offline_donor_pairs(selected)
    cache: dict[str, tuple[Mapping[str, Any], torch.Tensor, str]] = {}
    for record in selected:
        sample_id = str(record["sample_id"])
        sample_path = (state_bank_dir / str(record["sample_path"])).resolve()
        try:
            sample_path.relative_to(state_bank_dir)
        except ValueError as exc:
            raise ValueError(f"Sample path escapes state bank: {sample_path}.") from exc
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        if str(sample.get("sample_id", "")) != sample_id:
            raise ValueError(f"Loaded sample identity mismatch for {sample_id}.")
        image = _finite_image(sample, sample_id=sample_id)
        cache[sample_id] = (sample, image, tensor_sha256(image))

    entries: list[dict[str, Any]] = []
    for recipient_record in selected:
        recipient_id = str(recipient_record["sample_id"])
        donor_id = pairs[recipient_id]
        donor_record = by_id[donor_id]
        recipient_sample, recipient_image, recipient_hash = cache[recipient_id]
        donor_sample, donor_image, donor_hash = cache[donor_id]
        invariant_fields = ("task_suite", "task_id", "task_description", "replan_id")
        mismatched = [
            field
            for field in invariant_fields
            if recipient_record.get(field) != donor_record.get(field)
        ]
        if mismatched:
            raise ValueError(
                f"Offline pair {recipient_id}->{donor_id} violates {mismatched}."
            )
        if int(recipient_record["episode_id"]) == int(donor_record["episode_id"]):
            raise ValueError(f"Offline donor reuses recipient episode for {recipient_id}.")
        if tuple(recipient_image.shape) != tuple(donor_image.shape):
            raise ValueError(
                f"Offline donor image shape mismatch for {recipient_id}->{donor_id}."
            )
        difference = recipient_image.float() - donor_image.float()
        mae = float(difference.abs().mean().item())
        mse = float(difference.square().mean().item())
        if not math_is_finite_positive(mae) or not math_is_finite_positive(mse):
            raise ValueError(
                f"Offline donor image is identical or nonfinite for {recipient_id}->{donor_id}: "
                f"mae={mae}, mse={mse}."
            )
        if recipient_hash == donor_hash:
            raise ValueError(f"Offline donor image hash equals recipient for {recipient_id}.")
        recipient_context = recipient_sample["infer_action_kwargs"].get("context")
        donor_context = donor_sample["infer_action_kwargs"].get("context")
        if not isinstance(recipient_context, torch.Tensor) or not isinstance(
            donor_context, torch.Tensor
        ):
            raise TypeError("Offline samples must contain cached text context tensors.")
        if not torch.equal(recipient_context, donor_context):
            raise ValueError(
                f"Same-task offline pair has different cached text context: "
                f"{recipient_id}->{donor_id}."
            )
        entries.append(
            {
                "recipient_sample_id": recipient_id,
                "donor_sample_id": donor_id,
                "task_suite": str(recipient_record["task_suite"]),
                "task_id": int(recipient_record["task_id"]),
                "task_description": str(recipient_record["task_description"]),
                "task_text_sha256": sha256_json(str(recipient_record["task_description"])),
                "recipient_episode_id": int(recipient_record["episode_id"]),
                "donor_episode_id": int(donor_record["episode_id"]),
                "replan_id": int(recipient_record["replan_id"]),
                "recipient_image_sha256": recipient_hash,
                "donor_image_sha256": donor_hash,
                "image_shape": list(recipient_image.shape),
                "image_dtype": str(recipient_image.dtype),
                "pixel_mae": mae,
                "pixel_mse": mse,
            }
        )

    return {
        "artifact_type": "asre_round3b_offline_donor_mapping",
        "schema_version": OFFLINE_DONOR_SCHEMA_VERSION,
        "mapping_rule": (
            "within each (task_suite, task_id, replan_id), sort QC-valid states by "
            "episode_id and map each recipient to the next entry cyclically"
        ),
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "valid_manifest_path": str(valid_manifest_path),
        "valid_manifest_sha256": sha256_file(valid_manifest_path),
        "num_pairs": len(entries),
        "same_task": True,
        "same_replan": True,
        "different_episode": True,
        "derangement_verified": True,
        "outcome_independent": True,
        "entries": entries,
    }


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def load_offline_donor_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("artifact_type") != "asre_round3b_offline_donor_mapping":
        raise ValueError(f"Not a Round-3B offline donor mapping: {path}.")
    if int(payload.get("schema_version", -1)) != OFFLINE_DONOR_SCHEMA_VERSION:
        raise ValueError(f"Unsupported offline donor mapping schema in {path}.")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 499:
        raise ValueError(f"Offline donor mapping must contain 499 entries: {path}.")
    recipient_ids = [str(entry.get("recipient_sample_id", "")) for entry in entries]
    if len(set(recipient_ids)) != 499 or not all(recipient_ids):
        raise ValueError(f"Offline donor mapping recipients are not unique: {path}.")
    for entry in entries:
        if entry["recipient_sample_id"] == entry["donor_sample_id"]:
            raise ValueError("Offline donor mapping contains self replacement.")
        if int(entry["recipient_episode_id"]) == int(entry["donor_episode_id"]):
            raise ValueError("Offline donor mapping contains a same-episode donor.")
        if not math_is_finite_positive(float(entry["pixel_mae"])):
            raise ValueError("Offline donor mapping contains a zero/nonfinite pixel MAE.")
    return payload
