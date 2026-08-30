"""Evaluate one v2 condition on the frozen 100-sample native world set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from experiments.asre_diagnosis.common import (
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)
from experiments.asre_diagnosis.round3b.donor import tensor_sha256
from experiments.asre_diagnosis.round4b.basis import load_runtime_basis
from experiments.asre_diagnosis.round4b.fit_worker import _load_donor_image
from experiments.asre_diagnosis.salvage_b.world_runtime import (
    _validate_and_align_donor_image,
    load_donor_bundle,
    load_frozen_model,
    load_processed_sample,
    load_prompt_cache,
    load_world_dataset,
)

from .definitions import CONDITIONS, PROTOCOL, RANK_BY_CONDITION
from .native_runtime import run_native_condition


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _load_contract(preflight_path: Path) -> dict[str, Any]:
    preflight = _read(preflight_path.resolve())
    if preflight.get("protocol") != PROTOCOL or preflight.get("status") != "compatible":
        raise ValueError("Salvage-B-v2 preflight is incompatible.")
    reused = preflight["reused_frozen_inputs"]
    for stem in ("world_manifest", "target_manifest", "stochastic_manifest", "draw_tensors"):
        path = Path(str(reused[f"{stem}_path"])).resolve()
        if not path.is_file() or sha256_file(path) != reused[f"{stem}_sha256"]:
            raise ValueError(f"Frozen v2 input drifted: {path}")
    world = _read(Path(reused["world_manifest_path"]))
    targets = _read(Path(reused["target_manifest_path"]))
    draws = torch.load(
        Path(reused["draw_tensors_path"]), map_location="cpu", weights_only=False
    )
    return {
        "preflight": preflight,
        "records": world["records"],
        "targets": {str(row["sample_id"]): row for row in targets["records"]},
        "draws": draws,
    }


def _validate_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("A v2 world worker requires exactly one visible CUDA GPU.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, ""}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must bind the v2 worker explicitly.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.condition not in CONDITIONS:
        raise ValueError(f"Unknown v2 condition: {args.condition}")
    _validate_gpu()
    contract = _load_contract(args.preflight)
    output_dir = args.output_root.resolve() / "world" / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = output_dir / "completion.json"
    if completion.is_file():
        value = _read(completion)
        if (
            value.get("status") == "complete"
            and value.get("condition") == args.condition
            and value.get("git_commit_hash") == git_commit(PROJECT_ROOT)
            and value.get("preflight_git_commit")
            == _read(args.preflight.resolve()).get("git_commit_hash")
        ):
            return value

    preflight = contract["preflight"]
    dataset = load_world_dataset(
        preflight=preflight,
        runtime_work_dir=args.output_root.resolve() / "runtime" / args.condition,
    )
    prompt_cache = load_prompt_cache(preflight)
    donor_bundle = load_donor_bundle(preflight)
    model, _cfg = load_frozen_model(preflight)
    if (model.mot.num_layers, model.mot.num_heads, model.mot.attn_head_dim) != (30, 24, 128):
        raise ValueError("Frozen native joint architecture drifted.")
    rank = int(RANK_BY_CONDITION[args.condition])
    bases = None
    if rank in (97, 170):
        parameter = next(model.parameters())
        spec = load_runtime_basis(
            manifest_path=Path(preflight["basis"]["path"]),
            expected_sha256=preflight["basis"]["sha256"],
            basis_kind="svd",
            rank=rank,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        bases = spec.bases_by_layer

    rows: list[dict[str, Any]] = []
    records = contract["records"]
    if args.max_samples is not None:
        records = records[: int(args.max_samples)]
    with torch.inference_mode():
        for record in records:
            sample_id = str(record["sample_id"])
            processed = load_processed_sample(
                dataset=dataset,
                record=record,
                prompt_cache=prompt_cache,
                target_record=contract["targets"][sample_id],
            )
            model_inputs = model.build_inputs(dict(processed), tiled=False)
            target = model_inputs["input_latents"].detach()
            current_image = processed["video"][:, :, 0]
            raw_donor = _load_donor_image(
                donor_bundle,
                task_id=int(record["task_id"]),
                episode_id=int(record["trial"]),
            )
            donor_image, frozen_donor_hash, current_hash = _validate_and_align_donor_image(
                donor_image=raw_donor,
                current_image=current_image,
                expected_frozen_sha256=str(record["donor_processed_image_sha256"]),
            )
            if current_hash != contract["targets"][sample_id]["current_image_sha256"]:
                raise ValueError(f"Frozen current world image drifted for {sample_id}.")
            for draw in contract["draws"]["samples"][sample_id]:
                infer_kwargs = {
                    "prompt": None,
                    "num_video_frames": 9,
                    "action_horizon": 32,
                    "action": None,
                    "proprio": processed["proprio"][:, 0],
                    "context": processed["context"],
                    "context_mask": processed["context_mask"],
                    "negative_prompt": None,
                    "text_cfg_scale": 1.0,
                    "num_inference_steps": 10,
                    "sigma_shift": 5.0,
                    "seed": None,
                    "rand_device": "cpu",
                    "tiled": False,
                    "initial_video_noise": draw["video_noise"],
                    "initial_action_noise": draw["action_noise"],
                }
                result = run_native_condition(
                    model=model,
                    condition=args.condition,
                    current_input_image=current_image,
                    donor_input_image=None if args.condition == "current" else donor_image,
                    infer_kwargs=infer_kwargs,
                    bases_by_layer=bases,
                )
                predicted = result.prediction["video_latents"][:, :, 1:].float()
                target_future = target[:, :, 1:].to(predicted.device).float()
                loss = float(torch.mean((predicted - target_future) ** 2).item())
                rows.append(
                    {
                        "protocol": PROTOCOL,
                        "condition": args.condition,
                        "sample_id": sample_id,
                        "task_id": int(record["task_id"]),
                        "trial": int(record["trial"]),
                        "episode_id": int(record["episode_id"]),
                        "draw_id": int(draw["draw_id"]),
                        "loss": loss,
                        "current_image_sha256": current_hash,
                        "donor_image_sha256": frozen_donor_hash,
                        "target_latent_sha256": tensor_sha256(target.detach().cpu()),
                        "native_joint_graph": True,
                        "factorized_helper_used": False,
                    }
                )
            atomic_write_json(output_dir / f"{sample_id}.json", {"rows": rows[-4:]})
    rows_path = output_dir / "rows.json"
    atomic_write_json(rows_path, {"rows": rows})
    payload = {
        "artifact_type": "asre_salvage_b_v2_world_condition",
        "protocol": PROTOCOL,
        "status": "complete",
        "condition": args.condition,
        "sample_count": len(records),
        "draw_count": len(rows),
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "preflight_git_commit": preflight["git_commit_hash"],
        "rows_path": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
    }
    atomic_write_json(completion, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--max-samples", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
