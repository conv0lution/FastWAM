"""Real-checkpoint shared-interface and native-world-metric gates for Salvage B.

This stage is deliberately diagnostic only.  It evaluates one frozen world
sample and one pre-registered stochastic draw.  It does not inspect a
Current/Wrong endpoint result and it never launches an online action rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


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
from experiments.asre_diagnosis.round3b.donor import tensor_sha256  # noqa: E402
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
)
from experiments.asre_diagnosis.salvage_b.definitions import (  # noqa: E402
    CONDITIONS,
    WORLD_DRAWS_PER_SAMPLE,
)
from experiments.asre_diagnosis.salvage_b.world_runtime import (  # noqa: E402
    ACTION_TOKENS,
    EARLY_DISABLED,
    NATIVE_WORLD_INFERENCE_SHIFT,
    NATIVE_WORLD_INFERENCE_STEPS,
    PREFIX_TOKENS,
    condition_cache,
    load_donor_bundle,
    load_frozen_model,
    load_processed_sample,
    load_prompt_cache,
    load_rank_specs,
    load_world_dataset,
    native_world_loss,
    prepare_world_sample,
    projection_endpoint_caches,
    read_json,
)
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


# These are numerical-equivalence tolerances, not outcome thresholds.  The
# stock joint path attends over a larger masked sequence than the factorized
# path, so BF16 FlashAttention is not expected to be bitwise identical.
VAE_EQUIVALENCE_ATOL = 2.0e-2
VAE_EQUIVALENCE_RTOL = 2.0e-2
FACTORIZATION_ATOL = 2.0e-2
FACTORIZATION_RTOL = 2.0e-2
# The original fixed 2e-3 bound was below one BF16 representable precision
# unit (eps=7.8125e-3), even though the stock joint and factorized paths invoke
# SDPA with different masked query/key extents.  Derive the actual bound from
# the execution dtype before any endpoint/world outcome is inspected.  The
# fixed floor remains active for FP16/FP32, while BF16 receives a conservative
# two-epsilon numerical-partition budget.
FACTORIZATION_RELATIVE_RMSE_FLOOR = 2.0e-3
FACTORIZATION_DTYPE_EPS_MULTIPLIER = 2.0
MANUAL_METRIC_ATOL = 1.0e-7
CACHE_PATH_CHANGE_MIN = 1.0e-6
EXPECTED_NATIVE_WORLD_METRIC = (
    "pure_noise_native_future_latent_reconstruction_mse"
)


class MachineryGateError(RuntimeError):
    """A scientific machinery gate failed and implies a special stop."""

    def __init__(self, message: str, *, classification: str):
        super().__init__(message)
        self.classification = classification


def _tensor_manifest_sha256(tensor: torch.Tensor) -> str:
    """Match the canonical tensor encoding frozen by ``world_manifest.py``."""

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _as_real_float(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().to(device="cpu")
    if value.is_complex():
        value = torch.view_as_real(value)
    return value.to(dtype=torch.float32)


def _comparison(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    a = _as_real_float(left)
    b = _as_real_float(right)
    if tuple(a.shape) != tuple(b.shape):
        return {
            "passed": False,
            "left_shape": list(a.shape),
            "right_shape": list(b.shape),
            "shape_equal": False,
            "atol": atol,
            "rtol": rtol,
        }
    difference = (a - b).abs()
    rmse = float(torch.sqrt(torch.mean((a - b).square())).item())
    reference_rms = float(torch.sqrt(torch.mean(a.square())).item())
    relative_rmse = rmse / max(reference_rms, 1.0e-12)
    allclose = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
    return {
        "passed": bool(allclose and finite),
        "left_shape": list(a.shape),
        "right_shape": list(b.shape),
        "shape_equal": True,
        "finite": finite,
        "exact_equal": bool(torch.equal(a, b)),
        "torch_allclose": allclose,
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": relative_rmse,
        "atol": atol,
        "rtol": rtol,
    }


def _factorization_relative_rmse_budget(dtype: torch.dtype) -> dict[str, Any]:
    try:
        epsilon = float(torch.finfo(dtype).eps)
    except TypeError as error:
        raise ValueError(f"Factorization comparison requires a floating dtype: {dtype}") from error
    maximum = max(
        FACTORIZATION_RELATIVE_RMSE_FLOOR,
        FACTORIZATION_DTYPE_EPS_MULTIPLIER * epsilon,
    )
    return {
        "dtype": str(dtype),
        "machine_epsilon": epsilon,
        "epsilon_multiplier": FACTORIZATION_DTYPE_EPS_MULTIPLIER,
        "fixed_floor": FACTORIZATION_RELATIVE_RMSE_FLOOR,
        "relative_rmse_max": maximum,
        "policy": (
            "max(fixed floor, 2 * execution-dtype epsilon); stock joint and "
            "factorized SDPA use mathematically equivalent masks with different "
            "masked sequence extents"
        ),
        "outcome_independent": True,
    }


def _factorization_comparison_passes(
    comparison: Mapping[str, Any], *, relative_rmse_max: float
) -> bool:
    return bool(
        comparison.get("shape_equal") is True
        and comparison.get("finite") is True
        and math.isfinite(float(comparison.get("relative_rmse", math.inf)))
        and float(comparison["relative_rmse"]) <= float(relative_rmse_max)
    )


def _scalar_comparison(
    left: float,
    right: float,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    difference = abs(float(left) - float(right))
    tolerance = float(atol) + float(rtol) * abs(float(right))
    finite = math.isfinite(float(left)) and math.isfinite(float(right))
    return {
        "left": float(left),
        "right": float(right),
        "absolute_difference": difference,
        "relative_difference": difference / max(abs(float(right)), 1.0e-12),
        "tolerance": tolerance,
        "finite": finite,
        "passed": bool(finite and difference <= tolerance),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _cache_comparison(
    left_k: Sequence[torch.Tensor],
    left_v: Sequence[torch.Tensor],
    right_k: Sequence[torch.Tensor],
    right_v: Sequence[torch.Tensor],
) -> dict[str, Any]:
    if not (len(left_k) == len(left_v) == len(right_k) == len(right_v) == 30):
        return {
            "passed": False,
            "reason": "cache layer count mismatch",
            "lengths": [len(left_k), len(left_v), len(right_k), len(right_v)],
        }
    max_difference = 0.0
    all_exact = True
    shape_dtype_device_equal = True
    finite = True
    rows: list[dict[str, Any]] = []
    for layer in range(30):
        row: dict[str, Any] = {"layer": layer}
        for kind, left, right in (
            ("k", left_k[layer], right_k[layer]),
            ("v", left_v[layer], right_v[layer]),
        ):
            structural = bool(
                left.shape == right.shape
                and left.dtype == right.dtype
                and left.device == right.device
            )
            shape_dtype_device_equal &= structural
            values_finite = bool(
                torch.isfinite(left).all().item() and torch.isfinite(right).all().item()
            )
            finite &= values_finite
            if structural:
                difference = float(
                    (left.detach().float() - right.detach().float()).abs().max().item()
                )
                exact = bool(torch.equal(left, right))
            else:
                difference = math.inf
                exact = False
            max_difference = max(max_difference, difference)
            all_exact &= exact
            row[kind] = {
                "shape": list(left.shape),
                "structural_identity": structural,
                "finite": values_finite,
                "exact_equal": exact,
                "max_abs_diff": difference,
            }
        rows.append(row)
    return {
        "passed": bool(shape_dtype_device_equal and finite and all_exact),
        "shape_dtype_device_equal": shape_dtype_device_equal,
        "finite": finite,
        "all_exact": all_exact,
        "max_abs_diff": max_difference,
        "rows": rows,
    }


def _require(
    condition: bool,
    message: str,
    *,
    classification: str,
) -> None:
    if not condition:
        raise MachineryGateError(message, classification=classification)


def _hardware() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "distributed_initialized": bool(
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ),
        "world_size_environment": os.environ.get("WORLD_SIZE"),
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        properties = torch.cuda.get_device_properties(0)
        payload["device_0"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        }
    return payload


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
        "size": resolved.stat().st_size if resolved.is_file() else None,
    }


def _validate_frozen_inputs(args: argparse.Namespace) -> dict[str, Any]:
    preflight_path = args.preflight.resolve()
    world_path = args.world_manifest.resolve()
    stochastic_path = args.stochastic_manifest.resolve()
    draw_path = args.draw_tensors.resolve()
    targets_path = args.processed_targets.resolve()
    preflight = read_json(preflight_path)
    world = read_json(world_path)
    stochastic = read_json(stochastic_path)
    targets = read_json(targets_path)
    current_commit = git_commit(PROJECT_ROOT)
    if not (
        preflight.get("artifact_type") == "asre_salvage_b_preflight_report"
        and preflight.get("protocol") == SALVAGE_B_PROTOCOL
        and preflight.get("status") == "compatible"
        and preflight.get("git_commit_hash") == current_commit
        and world.get("artifact_type") == "asre_salvage_b_world_evaluation_manifest"
        and world.get("status") == "frozen_before_metrics"
        and world.get("git_commit_hash") == current_commit
        and stochastic.get("artifact_type") == "asre_salvage_b_stochastic_manifest"
        and stochastic.get("status") == "frozen_before_metrics"
        and stochastic.get("git_commit_hash") == current_commit
        and targets.get("artifact_type")
        == "asre_salvage_b_processed_world_target_manifest"
        and targets.get("status") == "frozen_before_gpu_metrics"
        and targets.get("git_commit_hash") == current_commit
    ):
        raise ValueError("Salvage-B machinery inputs are not frozen at the current commit.")
    preflight_sha = sha256_file(preflight_path)
    world_sha = sha256_file(world_path)
    draw_sha = sha256_file(draw_path)
    if not (
        world.get("preflight_report_sha256") == preflight_sha
        and stochastic.get("preflight_report_sha256") == preflight_sha
        and stochastic.get("world_manifest_sha256") == world_sha
        and targets.get("preflight_report_sha256") == preflight_sha
        and targets.get("world_manifest_sha256") == world_sha
        and world.get("draw_tensor_sha256") == draw_sha
        and stochastic.get("draw_tensor_sha256") == draw_sha
        and Path(str(world.get("draw_tensor_path", ""))).resolve() == draw_path
        and Path(str(stochastic.get("draw_tensor_path", ""))).resolve() == draw_path
        and Path(str(stochastic.get("world_manifest_path", ""))).resolve()
        == world_path
        and Path(str(targets.get("world_manifest_path", ""))).resolve() == world_path
    ):
        raise ValueError("Frozen world/target/draw artifact binding is incompatible.")
    if not (
        world.get("sample_count") == 100
        and stochastic.get("sample_count") == 100
        and stochastic.get("draws_per_sample") == WORLD_DRAWS_PER_SAMPLE
        and targets.get("sample_count") == 100
        and stochastic.get("pairing_rule")
        == (
            "same real target, pure-noise initialization, native inference "
            "schedule, text/proprio, and scheduler state across all conditions"
        )
        and world.get("outcome_based_selection") is False
        and world.get("future_targets_available") is True
        and targets.get("model_outcomes_inspected") is False
        and targets.get("gpu_metric_executed") is False
        and preflight.get("scope", {}).get("conditions")
        == list(CONDITIONS)
        and preflight.get("scope", {}).get("action_rerun") is False
        and preflight.get("frozen_action", {}).get("rerun_required") is False
        and preflight.get("basis", {}).get("ranks") == [97, 170]
        and preflight.get("basis", {}).get("refit") is False
    ):
        raise ValueError("Frozen world manifests do not contain exactly 100 x 4 draws.")
    native_metric = world.get("native_world_metric", {})
    if not (
        isinstance(native_metric, Mapping)
        and native_metric.get("name") == EXPECTED_NATIVE_WORLD_METRIC
        and native_metric.get("direction") == "lower_is_better"
        and native_metric.get("initialization")
        == "pure Gaussian noise in the two future latent frames"
        and native_metric.get("conditioning")
        == (
            "frozen text/proprio plus the selected external first-frame K/V cache; "
            "the carrier prefix is a discarded constant-zero placeholder"
        )
        and native_metric.get("inference_steps")
        == NATIVE_WORLD_INFERENCE_STEPS
        and native_metric.get("inference_shift")
        == NATIVE_WORLD_INFERENCE_SHIFT
        and native_metric.get("target_usage") == "scoring_only_after_inference"
        and native_metric.get("reduction")
        == "mean squared error over batch/channel/time/height/width"
        and stochastic.get("native_world_metric")
        == EXPECTED_NATIVE_WORLD_METRIC
        and stochastic.get("video_inference_steps")
        == NATIVE_WORLD_INFERENCE_STEPS
        and stochastic.get("video_inference_shift")
        == NATIVE_WORLD_INFERENCE_SHIFT
    ):
        raise ValueError("Frozen native pure-noise world metric definition drifted.")
    world_records = world.get("records")
    target_records = targets.get("records")
    if not isinstance(world_records, list) or not isinstance(target_records, list):
        raise TypeError("Frozen world/target records must be JSON arrays.")
    world_by_id = {str(row["sample_id"]): row for row in world_records}
    target_by_id = {str(row["sample_id"]): row for row in target_records}
    if (
        len(world_by_id) != 100
        or len(target_by_id) != 100
        or set(world_by_id) != set(target_by_id)
    ):
        raise ValueError("Frozen world/processed-target sample identities differ.")

    draw_payload = torch.load(draw_path, map_location="cpu", weights_only=False)
    if not isinstance(draw_payload, Mapping):
        raise TypeError("Frozen stochastic tensor artifact must be a mapping.")
    samples = draw_payload.get("samples")
    if (
        draw_payload.get("artifact_type")
        != "asre_salvage_b_fixed_stochastic_tensors"
        or draw_payload.get("native_world_metric")
        != EXPECTED_NATIVE_WORLD_METRIC
        or draw_payload.get("draws_per_sample") != WORLD_DRAWS_PER_SAMPLE
        or draw_payload.get("video_noise_shape") != [48, 2, 14, 28]
        or draw_payload.get("action_noise_shape") != [32, 7]
        or not isinstance(samples, Mapping)
        or set(map(str, samples)) != set(world_by_id)
    ):
        raise ValueError("Frozen stochastic tensor structure drifted.")
    schedulers = draw_payload.get("schedulers", {})
    if not (
        schedulers.get("video", {}).get("inference_steps")
        == NATIVE_WORLD_INFERENCE_STEPS
        and schedulers.get("video", {}).get("inference_shift")
        == NATIVE_WORLD_INFERENCE_SHIFT
        and schedulers.get("video", {}).get("num_train_timesteps") == 1000
        and schedulers.get("video", {}).get("initial_state")
        == "pure Gaussian future-latent noise"
        and schedulers.get("action", {}).get("shift") == 1.0
        and schedulers.get("action", {}).get("num_train_timesteps") == 1000
    ):
        raise ValueError("Frozen stochastic scheduler definitions drifted.")
    stochastic_rows = stochastic.get("records")
    if not isinstance(stochastic_rows, list) or len(stochastic_rows) != 400:
        raise ValueError("Stochastic manifest must contain exactly 400 draw rows.")
    stochastic_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in stochastic_rows:
        key = (str(row["sample_id"]), int(row["draw_id"]))
        if key in stochastic_by_key:
            raise ValueError(f"Duplicate frozen stochastic draw: {key}")
        stochastic_by_key[key] = row
    expected_keys = {
        (sample_id, draw_id)
        for sample_id in world_by_id
        for draw_id in range(WORLD_DRAWS_PER_SAMPLE)
    }
    if set(stochastic_by_key) != expected_keys:
        raise ValueError("Frozen stochastic manifest lacks a sample/draw pair.")

    # Validate the complete 100 x 4 tensor artifact, not just the machinery row.
    for sample_id in sorted(world_by_id):
        draws = samples[sample_id]
        if not isinstance(draws, list) or len(draws) != WORLD_DRAWS_PER_SAMPLE:
            raise ValueError(f"Frozen sample {sample_id} does not have exactly four draws.")
        observed_draw_ids: set[int] = set()
        for draw in draws:
            draw_id = int(draw.get("draw_id", -1))
            if draw_id in observed_draw_ids or draw_id not in range(WORLD_DRAWS_PER_SAMPLE):
                raise ValueError(f"Invalid draw IDs for {sample_id}.")
            observed_draw_ids.add(draw_id)
            row = stochastic_by_key[(sample_id, draw_id)]
            video_noise = draw.get("video_noise")
            action_noise = draw.get("action_noise")
            if (
                not torch.is_tensor(video_noise)
                or tuple(video_noise.shape) != (48, 2, 14, 28)
                or video_noise.dtype != torch.float32
                or not bool(torch.isfinite(video_noise).all().item())
                or _tensor_manifest_sha256(video_noise) != row["video_noise_sha256"]
                or not torch.is_tensor(action_noise)
                or tuple(action_noise.shape) != (32, 7)
                or action_noise.dtype != torch.float32
                or not bool(torch.isfinite(action_noise).all().item())
                or _tensor_manifest_sha256(action_noise) != row["action_noise_sha256"]
            ):
                raise ValueError(f"Frozen noise tensor drifted for {(sample_id, draw_id)}.")
            action_timestep = float(torch.as_tensor(draw["action_timestep"]).item())
            if not math.isclose(
                action_timestep, float(row["action_timestep"]), abs_tol=1e-6
            ):
                raise ValueError(f"Frozen timestep drifted for {(sample_id, draw_id)}.")

    sample_id = str(world_records[0]["sample_id"])
    first_draw = sorted(samples[sample_id], key=lambda row: int(row["draw_id"]))[0]
    frozen_action = preflight.get("frozen_action", {})
    implementation_identity = frozen_action.get("implementation_identity")
    if not (
        isinstance(implementation_identity, list)
        and implementation_identity
        and any(
            isinstance(row, Mapping)
            and row.get("strict_shared_intervention_dependency") is True
            for row in implementation_identity
        )
        and all(
            isinstance(row, Mapping)
            and (
                row.get("strict_shared_intervention_dependency") is False
                or row.get("identical") is True
            )
            for row in implementation_identity
        )
        and isinstance(frozen_action.get("results"), Mapping)
        and frozen_action.get("rerun_required") is False
    ):
        raise ValueError("Frozen Round-4C action reuse identity is incomplete.")
    return {
        "preflight": preflight,
        "world": world,
        "stochastic": stochastic,
        "targets": targets,
        "world_by_id": world_by_id,
        "target_by_id": target_by_id,
        "sample_id": sample_id,
        "record": world_by_id[sample_id],
        "target_record": target_by_id[sample_id],
        "draw": first_draw,
        "action_reuse_validation": {
            "passed": True,
            "rerun_required": False,
            "implementation_identity_rows": len(implementation_identity),
            "strict_dependencies_identical": True,
            "same_checkpoint_basis_donor_and_intervention_validated": True,
        },
        "draw_validation": {
            "passed": True,
            "samples": 100,
            "draws_per_sample": WORLD_DRAWS_PER_SAMPLE,
            "draw_rows": 400,
            "video_initial_state": "pure Gaussian future-latent noise",
            "video_inference_steps": NATIVE_WORLD_INFERENCE_STEPS,
            "video_inference_shift": NATIVE_WORLD_INFERENCE_SHIFT,
            "action_scheduler_shift": 1.0,
            "all_noise_tensors_finite_and_hash_identical": True,
            "all_action_timesteps_identical_to_manifest": True,
        },
    }


def _prepare_draw_state(
    *,
    model,
    prepared,
    draw: Mapping[str, Any],
) -> dict[str, Any]:
    target_latents = prepared.input_latents
    initial_future_noise = draw["video_noise"].to(
        device=target_latents.device, dtype=target_latents.dtype
    ).unsqueeze(0)
    expected_future_shape = tuple(target_latents[:, :, 1:].shape)
    if tuple(initial_future_noise.shape) != expected_future_shape:
        raise ValueError(
            "Frozen pure-noise future layout drifted: "
            f"{initial_future_noise.shape} != {expected_future_shape}."
        )
    inference_timesteps, inference_deltas = (
        model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=NATIVE_WORLD_INFERENCE_STEPS,
            device=target_latents.device,
            dtype=target_latents.dtype,
            shift_override=NATIVE_WORLD_INFERENCE_SHIFT,
        )
    )
    if not (
        inference_timesteps.numel() == NATIVE_WORLD_INFERENCE_STEPS
        and inference_deltas.numel() == NATIVE_WORLD_INFERENCE_STEPS
    ):
        raise ValueError("Native video inference schedule length drifted.")
    video_timestep = inference_timesteps[0].reshape(1)
    prefix_placeholder = torch.zeros(
        (
            initial_future_noise.shape[0],
            initial_future_noise.shape[1],
            1,
            initial_future_noise.shape[3],
            initial_future_noise.shape[4],
        ),
        device=initial_future_noise.device,
        dtype=initial_future_noise.dtype,
    )
    carrier = torch.cat([prefix_placeholder, initial_future_noise], dim=2)
    video = model.video_expert.prepare(
        x=carrier,
        timestep=video_timestep,
        context=prepared.inputs["context"],
        context_mask=prepared.inputs["context_mask"],
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    (
        video_tokens,
        video_t,
        video_t_mod,
        video_context,
        video_context_mask,
        video_freqs,
        f,
        h,
        w,
        tokens_per_frame,
    ) = video
    if (int(f), int(h), int(w), int(tokens_per_frame), int(video_tokens.shape[1])) != (
        3,
        7,
        14,
        PREFIX_TOKENS,
        294,
    ):
        raise ValueError("Native full-video draw preparation layout drifted.")

    action = prepared.inputs["action"]
    action_noise = draw["action_noise"].to(
        device=action.device, dtype=action.dtype
    ).unsqueeze(0)
    action_timestep = torch.as_tensor(
        draw["action_timestep"], device=action.device, dtype=action.dtype
    ).reshape(1)
    noisy_action = model.train_action_scheduler.add_noise(
        action, action_noise, action_timestep
    )
    action_prepared = model.action_expert.prepare(
        action_tokens=noisy_action,
        timestep=action_timestep,
        context=prepared.inputs["context"],
        context_mask=prepared.inputs["context_mask"],
    )
    (
        action_tokens,
        _action_t,
        action_t_mod,
        action_context,
        action_context_mask,
        action_freqs,
    ) = action_prepared
    if tuple(action_tokens.shape) != (1, ACTION_TOKENS, 1024):
        raise ValueError(f"Native action-token layout drifted: {action_tokens.shape}")

    full_joint_mask = model._build_mot_attention_mask(
        video_seq_len=int(video_tokens.shape[1]),
        action_seq_len=int(action_tokens.shape[1]),
        video_tokens_per_frame=PREFIX_TOKENS,
        device=video_tokens.device,
    )
    prefix_joint_mask = model._build_mot_attention_mask(
        video_seq_len=PREFIX_TOKENS,
        action_seq_len=int(action_tokens.shape[1]),
        video_tokens_per_frame=PREFIX_TOKENS,
        device=video_tokens.device,
    )
    video_mask = model.video_expert.build_video_to_video_mask(
        video_seq_len=int(video_tokens.shape[1]),
        video_tokens_per_frame=PREFIX_TOKENS,
        device=video_tokens.device,
    )
    if bool(full_joint_mask[:294, 294:].any().item()):
        raise ValueError("Native video queries unexpectedly attend to action keys.")
    return {
        "carrier": carrier,
        "prefix_placeholder": prefix_placeholder,
        "initial_future_noise": initial_future_noise,
        "inference_timesteps": inference_timesteps,
        "inference_deltas": inference_deltas,
        "video_timestep": video_timestep,
        "video_tokens": video_tokens,
        "video_t": video_t,
        "video_t_mod": video_t_mod,
        "video_context": video_context,
        "video_context_mask": video_context_mask,
        "video_freqs": video_freqs,
        "f": int(f),
        "h": int(h),
        "w": int(w),
        "video_mask": video_mask,
        "noisy_action": noisy_action,
        "action_timestep": action_timestep,
        "action_tokens": action_tokens,
        "action_t_mod": action_t_mod,
        "action_context": action_context,
        "action_context_mask": action_context_mask,
        "action_freqs": action_freqs,
        "full_joint_mask": full_joint_mask,
        "action_attention_mask": prefix_joint_mask[PREFIX_TOKENS:, :],
    }


def _factorized_future_tokens(
    *,
    model,
    state: Mapping[str, Any],
    cache_k: list[torch.Tensor],
    cache_v: list[torch.Tensor],
    disabled_layers: tuple[int, ...],
) -> torch.Tensor:
    return model.mot.forward_future_video_with_video_cache_tensor(
        future_video_tokens=state["video_tokens"][:, PREFIX_TOKENS:],
        future_video_freqs=state["video_freqs"][PREFIX_TOKENS:],
        future_video_t_mod=state["video_t_mod"][:, PREFIX_TOKENS:],
        future_video_context=state["video_context"],
        future_video_context_mask=state["video_context_mask"][:, PREFIX_TOKENS:],
        video_cache_k=cache_k,
        video_cache_v=cache_v,
        future_video_attention_mask=state["video_mask"][PREFIX_TOKENS:, :],
        disabled_video_prefix_layers=disabled_layers,
    )


def _post_future(model, state: Mapping[str, Any], tokens: torch.Tensor) -> torch.Tensor:
    return model.video_expert.post(
        tokens,
        state["video_t"][:, PREFIX_TOKENS:],
        state["f"] - 1,
        state["h"],
        state["w"],
    )


def _action_prediction(
    *,
    model,
    state: Mapping[str, Any],
    cache_k: list[torch.Tensor],
    cache_v: list[torch.Tensor],
) -> torch.Tensor:
    return model._denoise_action_with_video_cache(
        latents_action=state["noisy_action"],
        timestep_action=state["action_timestep"],
        context=state["prepared_context"],
        context_mask=state["prepared_context_mask"],
        video_cache_k=cache_k,
        video_cache_v=cache_v,
        action_attention_mask=state["action_attention_mask"],
        disabled_video_layers=EARLY_DISABLED,
    )


def _manual_reconstruction_metric(
    *, prepared, prediction: torch.Tensor
) -> dict[str, float]:
    """Reconstruct the registered terminal metric without the runtime helper."""

    target = prepared.input_latents[:, :, 1:]
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(
            f"Prediction/target layout mismatch: {prediction.shape} != {target.shape}."
        )
    mse = torch.nn.functional.mse_loss(
        prediction.float(), target.float(), reduction="mean"
    )
    return {
        "native_world_loss": float(mse.item()),
        "future_latent_mse": float(mse.item()),
    }


def _install_consumer_capture(model, captures: dict[str, Any]):
    """Install transparent instance wrappers; return a restoration callback."""

    mot = model.mot
    names = (
        "forward_action_with_video_cache_tensor",
        "forward_future_video_with_video_cache_tensor",
    )
    originals = {name: getattr(mot, name) for name in names}
    prior_instance_values = {
        name: (name in mot.__dict__, mot.__dict__.get(name)) for name in names
    }

    def action_wrapper(*args, **kwargs):
        cache_k = kwargs["video_cache_k"]
        cache_v = kwargs["video_cache_v"]
        identity = {
            "cache_k_list_id": id(cache_k),
            "cache_v_list_id": id(cache_v),
            "cache_k_tensor_ids": [id(tensor) for tensor in cache_k],
            "cache_v_tensor_ids": [id(tensor) for tensor in cache_v],
        }
        captures.setdefault("action_calls", []).append(identity)
        captures["action"] = identity
        return originals["forward_action_with_video_cache_tensor"](*args, **kwargs)

    def world_wrapper(*args, **kwargs):
        cache_k = kwargs["video_cache_k"]
        cache_v = kwargs["video_cache_v"]
        identity = {
            "cache_k_list_id": id(cache_k),
            "cache_v_list_id": id(cache_v),
            "cache_k_tensor_ids": [id(tensor) for tensor in cache_k],
            "cache_v_tensor_ids": [id(tensor) for tensor in cache_v],
        }
        captures.setdefault("world_calls", []).append(identity)
        captures["world"] = identity
        return originals["forward_future_video_with_video_cache_tensor"](*args, **kwargs)

    object.__setattr__(mot, names[0], action_wrapper)
    object.__setattr__(mot, names[1], world_wrapper)

    def restore() -> None:
        for name in names:
            existed, value = prior_instance_values[name]
            if existed:
                object.__setattr__(mot, name, value)
            else:
                object.__delattr__(mot, name)

    return restore


def _install_world_rollout_capture(model, captures: dict[str, Any]):
    """Capture native inference inputs without changing any numerical value."""

    prepare_owner = model.video_expert
    scheduler = model.infer_video_scheduler
    owners = {
        "prepare": (prepare_owner, "prepare"),
        "schedule": (scheduler, "build_inference_schedule"),
        "step": (scheduler, "step"),
    }
    originals = {
        name: getattr(owner, attribute)
        for name, (owner, attribute) in owners.items()
    }
    prior_instance_values = {
        name: (attribute in owner.__dict__, owner.__dict__.get(attribute))
        for name, (owner, attribute) in owners.items()
    }

    def prepare_wrapper(*args, **kwargs):
        value = kwargs.get("x", args[0] if args else None)
        if not torch.is_tensor(value):
            raise TypeError("Captured video prepare input is not a tensor.")
        captures.setdefault("prepare_inputs", []).append(value.detach().clone())
        return originals["prepare"](*args, **kwargs)

    def schedule_wrapper(*args, **kwargs):
        steps = kwargs.get(
            "num_inference_steps", args[0] if len(args) > 0 else None
        )
        shift = kwargs.get("shift_override", args[3] if len(args) > 3 else None)
        result = originals["schedule"](*args, **kwargs)
        captures.setdefault("schedule_calls", []).append(
            {
                "num_inference_steps": int(steps),
                "shift_override": None if shift is None else float(shift),
                "timesteps": result[0].detach().clone(),
                "deltas": result[1].detach().clone(),
            }
        )
        return result

    def step_wrapper(*args, **kwargs):
        model_output = kwargs.get(
            "model_output", args[0] if len(args) > 0 else None
        )
        delta = kwargs.get("delta", args[1] if len(args) > 1 else None)
        sample = kwargs.get("sample", args[2] if len(args) > 2 else None)
        if not all(torch.is_tensor(value) for value in (model_output, delta, sample)):
            raise TypeError("Captured native scheduler step inputs are not tensors.")
        result = originals["step"](*args, **kwargs)
        captures.setdefault("scheduler_steps", []).append(
            {
                "sample": sample.detach().clone(),
                "model_output": model_output.detach().clone(),
                "delta": delta.detach().clone(),
                "result": result.detach().clone(),
            }
        )
        return result

    wrappers = {
        "prepare": prepare_wrapper,
        "schedule": schedule_wrapper,
        "step": step_wrapper,
    }
    for name, (owner, attribute) in owners.items():
        object.__setattr__(owner, attribute, wrappers[name])

    def restore() -> None:
        for name, (owner, attribute) in owners.items():
            existed, value = prior_instance_values[name]
            if existed:
                object.__setattr__(owner, attribute, value)
            else:
                object.__delattr__(owner, attribute)

    return restore


def _markdown(report: Mapping[str, Any]) -> str:
    passed = report.get("passed") is True
    lines = [
        "# Salvage B Shared-Interface Machinery Gate",
        "",
        f"Status: **{'PASSED' if passed else 'FAILED'}**",
        "",
        f"Phase B authorized: `{bool(report.get('phase_b_authorized'))}`.",
        "",
    ]
    if report.get("recommended_special_classification"):
        lines.extend(
            [
                "Recommended special stop: "
                f"`{report['recommended_special_classification']}`.",
                "",
            ]
        )
    failure = report.get("failure")
    if isinstance(failure, Mapping):
        lines.extend(
            [
                "## Failure",
                "",
                f"- Gate: `{failure.get('active_gate')}`",
                f"- Type: `{failure.get('type')}`",
                f"- Message: {failure.get('message')}",
                "",
            ]
        )
    lines.extend(["## Checks", ""])
    checks = report.get("checks", {})
    if isinstance(checks, Mapping):
        for name, value in checks.items():
            check_passed = value.get("passed") if isinstance(value, Mapping) else None
            lines.append(f"- `{name}`: `{check_passed}`")
    lines.extend(
        [
            "",
            "This diagnostic used one frozen sample/draw, ran no environment rollout, "
            "and inspected no Current/Wrong world endpoint outcome.",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text.rstrip() + "\n")
    os.replace(temporary, path)


def _publish_report_bundle(
    *, output: Path, output_md: Path, report: Mapping[str, Any]
) -> None:
    """Publish JSON last so its existence is the completed-stage sentinel.

    An orphan Markdown companion means a previous attempt stopped between the
    two atomic writes.  It is intentionally replaceable on resume; a JSON
    report is never overwritten.
    """

    output = output.resolve()
    output_md = output_md.resolve()
    if output.exists():
        raise FileExistsError(
            "Refusing to overwrite a completed Salvage-B machinery report: "
            f"{output}"
        )
    _atomic_text(output_md, _markdown(report))
    atomic_write_json(output, dict(report))


def run(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "artifact_type": "asre_salvage_b_shared_interface_machinery_report",
        "schema_version": 1,
        "protocol": SALVAGE_B_PROTOCOL,
        "status": "running",
        "passed": False,
        "phase_b_authorized": False,
        "created_at": now_iso(),
        "git_commit_hash": None,
        "hardware": {},
        "checks": {},
        "inputs": {
            "preflight": {"path": str(args.preflight.resolve())},
            "world_manifest": {"path": str(args.world_manifest.resolve())},
            "stochastic_manifest": {"path": str(args.stochastic_manifest.resolve())},
            "draw_tensors": {"path": str(args.draw_tensors.resolve())},
            "processed_targets": {"path": str(args.processed_targets.resolve())},
        },
        "scope": {
            "machinery_samples": 1,
            "machinery_draws": 1,
            "online_action_rollouts": 0,
            "world_endpoint_outcomes_inspected": False,
            "projected_world_outcomes_inspected": False,
            "later_asre_launched": False,
            "ddp_used": False,
        },
    }
    active_gate = "report_initialization"
    try:
        current_commit = git_commit(PROJECT_ROOT)
        report["git_commit_hash"] = current_commit
        report["hardware"] = _hardware()
        report["inputs"] = {
            "preflight": _artifact(args.preflight),
            "world_manifest": _artifact(args.world_manifest),
            "stochastic_manifest": _artifact(args.stochastic_manifest),
            "draw_tensors": _artifact(args.draw_tensors),
            "processed_targets": _artifact(args.processed_targets),
        }
        report.update(
            {
                "preflight_report_sha256": report["inputs"]["preflight"]["sha256"],
                "world_manifest_sha256": report["inputs"]["world_manifest"]["sha256"],
                "stochastic_manifest_sha256": report["inputs"]["stochastic_manifest"][
                    "sha256"
                ],
                "draw_tensors_sha256": report["inputs"]["draw_tensors"]["sha256"],
                "processed_targets_sha256": report["inputs"]["processed_targets"][
                    "sha256"
                ],
            }
        )

        active_gate = "runtime_preconditions"
        if not torch.cuda.is_available():
            raise RuntimeError("Salvage-B real-checkpoint machinery requires CUDA.")
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if distributed or world_size != 1:
            raise RuntimeError("Salvage-B machinery must run on one process without DDP.")

        active_gate = "frozen_input_identity"
        frozen = _validate_frozen_inputs(args)
        report["sample_id"] = frozen["sample_id"]
        report["draw_id"] = int(frozen["draw"]["draw_id"])
        report["checks"]["fixed_stochasticity_manifest"] = frozen["draw_validation"]
        report["checks"]["frozen_round4c_action_reuse_identity"] = frozen[
            "action_reuse_validation"
        ]

        active_gate = "load_frozen_world_runtime"
        set_global_seed(42, get_worker_init_fn=False)
        dataset = load_world_dataset(
            preflight=frozen["preflight"],
            runtime_work_dir=args.runtime_work_dir.resolve(),
        )
        prompt_cache = load_prompt_cache(frozen["preflight"])
        donor_bundle = load_donor_bundle(frozen["preflight"])
        processed = load_processed_sample(
            dataset=dataset,
            record=frozen["record"],
            prompt_cache=prompt_cache,
            target_record=frozen["target_record"],
        )
        model, _cfg = load_frozen_model(frozen["preflight"])
        if (
            float(model.infer_video_scheduler.shift)
            != NATIVE_WORLD_INFERENCE_SHIFT
            or int(model.infer_video_scheduler.num_train_timesteps) != 1000
            or float(model.train_action_scheduler.shift) != 1.0
            or int(model.train_action_scheduler.num_train_timesteps) != 1000
        ):
            raise ValueError("Loaded model scheduler configuration drifted from the frozen draws.")
        prepared = prepare_world_sample(
            model=model,
            processed=processed,
            record=frozen["record"],
            donor_bundle=donor_bundle,
        )
        report["runtime_identity"] = {
            "checkpoint_sha256": frozen["preflight"]["state"]["checkpoint_sha256"],
            "basis_manifest_sha256": frozen["preflight"]["basis"]["sha256"],
            "split_manifest_sha256": frozen["preflight"]["basis"]["split_sha256"],
            "donor_mapping_sha256": donor_bundle.mapping_sha256,
            "donor_observation_manifest_sha256": donor_bundle.observation_manifest_sha256,
            "current_image_sha256": prepared.current_image_sha256,
            "donor_image_sha256": prepared.donor_image_sha256,
            "target_latent_sha256": prepared.target_latent_sha256,
            "round4c_action_summary_sha256": frozen["preflight"]["frozen_action"][
                "summary_sha256"
            ],
            "round4c_action_rerun_required": frozen["preflight"]["frozen_action"][
                "rerun_required"
            ],
        }
        model_parameters_without_gradients = all(
            parameter.grad is None for parameter in model.parameters()
        )
        report["checks"]["frozen_model_and_scheduler_identity"] = {
            "passed": model_parameters_without_gradients,
            "model_eval": not model.training,
            "all_parameter_gradients_none": model_parameters_without_gradients,
            "video_action_conditioned": bool(model.video_expert.action_conditioned),
            "video_inference_steps": NATIVE_WORLD_INFERENCE_STEPS,
            "video_inference_shift": float(model.infer_video_scheduler.shift),
            "action_train_shift": float(model.train_action_scheduler.shift),
        }
        _require(
            model_parameters_without_gradients and not model.training,
            "Frozen model unexpectedly has gradients or is in training mode.",
            classification="WORLD-METRIC-NOT-VALIDATABLE",
        )

        with torch.no_grad():
            active_gate = "independent_current_frame_encode_identity"
            current_image = processed["video"][:, :, 0].to(
                device=model.device, dtype=model.torch_dtype
            )
            standalone_first = model._encode_input_image_latents_tensor(
                current_image, tiled=False
            )
            prepared_current = prepared.current_frame_latent
            vae_comparison = _comparison(
                standalone_first,
                prepared_current,
                atol=VAE_EQUIVALENCE_ATOL,
                rtol=VAE_EQUIVALENCE_RTOL,
            )
            vae_comparison.update(
                {
                    "standalone_sha256": tensor_sha256(
                        standalone_first.detach().float().cpu()
                    ),
                    "prepared_current_sha256": tensor_sha256(
                        prepared_current.detach().float().cpu()
                    ),
                    "full_clip_target_used": False,
                }
            )
            report["checks"][active_gate] = vae_comparison
            _require(
                vae_comparison["passed"],
                "Repeated independent current-image VAE encoding is not stable.",
                classification="SHARED-INTERFACE-NOT-AVAILABLE",
            )

            draw_state = _prepare_draw_state(
                model=model, prepared=prepared, draw=frozen["draw"]
            )
            # _action_prediction uses the original shared text/proprio context,
            # while action_context below is already projected to the action expert.
            draw_state["prepared_context"] = prepared.inputs["context"]
            draw_state["prepared_context_mask"] = prepared.inputs["context_mask"]

            active_gate = "stock_joint_vs_factorized_first_pure_noise_step"
            stock_carrier = torch.cat(
                [prepared.current_frame_latent, draw_state["initial_future_noise"]],
                dim=2,
            )
            stock_prepared = model.video_expert.prepare(
                x=stock_carrier,
                timestep=draw_state["video_timestep"],
                context=prepared.inputs["context"],
                context_mask=prepared.inputs["context_mask"],
                action=None,
                fuse_vae_embedding_in_latents=True,
            )
            (
                stock_input_tokens,
                stock_t,
                stock_t_mod,
                stock_context,
                stock_context_mask,
                stock_freqs,
                stock_f,
                stock_h,
                stock_w,
                stock_tokens_per_frame,
            ) = stock_prepared
            if (
                int(stock_f),
                int(stock_h),
                int(stock_w),
                int(stock_tokens_per_frame),
                int(stock_input_tokens.shape[1]),
            ) != (3, 7, 14, PREFIX_TOKENS, 294):
                raise ValueError("Stock current-prefix carrier layout drifted.")
            factorized_future_input_identity = {
                "tokens": _comparison(
                    stock_input_tokens[:, PREFIX_TOKENS:],
                    draw_state["video_tokens"][:, PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
                "t": _comparison(
                    stock_t[:, PREFIX_TOKENS:],
                    draw_state["video_t"][:, PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
                "t_mod": _comparison(
                    stock_t_mod[:, PREFIX_TOKENS:],
                    draw_state["video_t_mod"][:, PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
                "context": _comparison(
                    stock_context,
                    draw_state["video_context"],
                    atol=0.0,
                    rtol=0.0,
                ),
                "context_mask": _comparison(
                    stock_context_mask[:, PREFIX_TOKENS:],
                    draw_state["video_context_mask"][:, PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
                "freqs": _comparison(
                    stock_freqs[PREFIX_TOKENS:],
                    draw_state["video_freqs"][PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
            }
            future_inputs_identical = all(
                value.get("exact_equal") is True
                for value in factorized_future_input_identity.values()
            )
            stock_video_tokens, _stock_action_tokens = model.mot.forward_joint_core(
                video_tokens=stock_input_tokens,
                action_tokens=draw_state["action_tokens"],
                video_freqs=stock_freqs,
                action_freqs=draw_state["action_freqs"],
                video_t_mod=stock_t_mod,
                action_t_mod=draw_state["action_t_mod"],
                video_context=stock_context,
                video_context_mask=stock_context_mask,
                action_context=draw_state["action_context"],
                action_context_mask=draw_state["action_context_mask"],
                attention_mask=draw_state["full_joint_mask"],
            )
            factorized_tokens = _factorized_future_tokens(
                model=model,
                state=draw_state,
                cache_k=prepared.current_cache_k,
                cache_v=prepared.current_cache_v,
                disabled_layers=(),
            )
            stock_future_tokens = stock_video_tokens[:, PREFIX_TOKENS:]
            stock_state = dict(draw_state)
            stock_state.update(
                {
                    "video_t": stock_t,
                    "f": int(stock_f),
                    "h": int(stock_h),
                    "w": int(stock_w),
                }
            )
            stock_prediction = _post_future(model, stock_state, stock_future_tokens)
            factorized_prediction = _post_future(model, draw_state, factorized_tokens)
            token_equivalence = _comparison(
                factorized_tokens,
                stock_future_tokens,
                atol=FACTORIZATION_ATOL,
                rtol=FACTORIZATION_RTOL,
            )
            prediction_equivalence = _comparison(
                factorized_prediction,
                stock_prediction,
                atol=FACTORIZATION_ATOL,
                rtol=FACTORIZATION_RTOL,
            )
            factorization_budget = _factorization_relative_rmse_budget(
                factorized_tokens.dtype
            )
            relative_rmse_max = float(factorization_budget["relative_rmse_max"])
            token_factorization_passed = _factorization_comparison_passes(
                token_equivalence,
                relative_rmse_max=relative_rmse_max,
            )
            prediction_factorization_passed = _factorization_comparison_passes(
                prediction_equivalence,
                relative_rmse_max=relative_rmse_max,
            )
            token_equivalence["factorization_gate_passed"] = (
                token_factorization_passed
            )
            prediction_equivalence["factorization_gate_passed"] = (
                prediction_factorization_passed
            )
            factorization_passed = bool(
                future_inputs_identical
                and token_factorization_passed
                and prediction_factorization_passed
            )
            report["checks"][active_gate] = {
                "passed": factorization_passed,
                "stock_future_token_shape": list(stock_future_tokens.shape),
                "factorized_future_token_shape": list(factorized_tokens.shape),
                "token_equivalence": token_equivalence,
                "prediction_equivalence": prediction_equivalence,
                "stock_vs_zero_carrier_future_input_identity": (
                    factorized_future_input_identity
                ),
                "numerical_equivalence_policy": factorization_budget,
                "early_prefix_layers_disabled_for_equivalence_check": [],
                "stock_oracle_prefix": "independently encoded real current frame",
                "factorized_carrier_prefix": (
                    "constant-zero placeholder; discarded before MoT"
                ),
                "factorized_external_cache": "prepared current-observation K/V",
                "patch_t": 1,
                "inference_step": 1,
                "scoring_target_used": False,
            }
            _require(
                factorization_passed,
                "Cached-prefix factorization does not reproduce the stock joint future path.",
                classification="SHARED-INTERFACE-NOT-AVAILABLE",
            )

            active_gate = "same_cache_object_reaches_action_and_world"
            captures: dict[str, Any] = {}
            restore_consumers = _install_consumer_capture(model, captures)
            restore_rollout = _install_world_rollout_capture(model, captures)
            try:
                shared_action_prediction = _action_prediction(
                    model=model,
                    state=draw_state,
                    cache_k=prepared.current_cache_k,
                    cache_v=prepared.current_cache_v,
                )
                shared_world = native_world_loss(
                    model=model,
                    prepared=prepared,
                    video_cache_k=prepared.current_cache_k,
                    video_cache_v=prepared.current_cache_v,
                    video_noise=frozen["draw"]["video_noise"],
                    disabled_video_prefix_layers=EARLY_DISABLED,
                )
            finally:
                restore_rollout()
                restore_consumers()
            expected_ids = {
                "cache_k_list_id": id(prepared.current_cache_k),
                "cache_v_list_id": id(prepared.current_cache_v),
                "cache_k_tensor_ids": [id(tensor) for tensor in prepared.current_cache_k],
                "cache_v_tensor_ids": [id(tensor) for tensor in prepared.current_cache_v],
            }
            object_identity_passed = bool(
                captures.get("action_calls") == [expected_ids]
                and captures.get("world_calls")
                == [expected_ids] * NATIVE_WORLD_INFERENCE_STEPS
            )
            report["checks"][active_gate] = {
                "passed": object_identity_passed,
                "expected": expected_ids,
                "captured_action_consumer": captures.get("action"),
                "captured_world_consumer": captures.get("world"),
                "action_consumer_calls": len(captures.get("action_calls", [])),
                "world_consumer_calls": len(captures.get("world_calls", [])),
                "same_list_objects": bool(
                    captures.get("action", {}).get("cache_k_list_id")
                    == captures.get("world", {}).get("cache_k_list_id")
                    and captures.get("action", {}).get("cache_v_list_id")
                    == captures.get("world", {}).get("cache_v_list_id")
                ),
                "same_all_30_k_v_tensor_objects": bool(
                    captures.get("action", {}).get("cache_k_tensor_ids")
                    == captures.get("world", {}).get("cache_k_tensor_ids")
                    and captures.get("action", {}).get("cache_v_tensor_ids")
                    == captures.get("world", {}).get("cache_v_tensor_ids")
                ),
            }
            _require(
                object_identity_passed,
                "Action and world consumers did not receive the exact same cache tensor objects.",
                classification="SHARED-INTERFACE-NOT-AVAILABLE",
            )

            active_gate = "native_pure_noise_10_step_rollout"
            prepare_inputs = captures.get("prepare_inputs", [])
            scheduler_steps = captures.get("scheduler_steps", [])
            schedule_calls = captures.get("schedule_calls", [])
            first_carrier_prefix = (
                _comparison(
                    prepare_inputs[0][:, :, :1],
                    torch.zeros_like(prepare_inputs[0][:, :, :1]),
                    atol=0.0,
                    rtol=0.0,
                )
                if prepare_inputs
                else {"passed": False, "reason": "no prepare calls captured"}
            )
            first_carrier_future = (
                _comparison(
                    prepare_inputs[0][:, :, 1:],
                    draw_state["initial_future_noise"],
                    atol=0.0,
                    rtol=0.0,
                )
                if prepare_inputs
                else {"passed": False, "reason": "no prepare calls captured"}
            )
            first_scheduler_sample = (
                _comparison(
                    scheduler_steps[0]["sample"],
                    draw_state["initial_future_noise"],
                    atol=0.0,
                    rtol=0.0,
                )
                if scheduler_steps
                else {"passed": False, "reason": "no scheduler steps captured"}
            )
            all_prefix_placeholders_zero = bool(
                len(prepare_inputs) == NATIVE_WORLD_INFERENCE_STEPS
                and all(
                    torch.count_nonzero(value[:, :, :1]).item() == 0
                    for value in prepare_inputs
                )
            )
            carrier_matches_step_state = bool(
                len(prepare_inputs)
                == len(scheduler_steps)
                == NATIVE_WORLD_INFERENCE_STEPS
                and all(
                    torch.equal(
                        prepare_inputs[index][:, :, 1:],
                        scheduler_steps[index]["sample"],
                    )
                    for index in range(NATIVE_WORLD_INFERENCE_STEPS)
                )
                and all(
                    torch.equal(
                        scheduler_steps[index - 1]["result"],
                        scheduler_steps[index]["sample"],
                    )
                    for index in range(1, NATIVE_WORLD_INFERENCE_STEPS)
                )
            )
            schedule_passed = bool(
                len(schedule_calls) == 1
                and schedule_calls[0]["num_inference_steps"]
                == NATIVE_WORLD_INFERENCE_STEPS
                and schedule_calls[0]["shift_override"]
                == NATIVE_WORLD_INFERENCE_SHIFT
                and torch.equal(
                    schedule_calls[0]["timesteps"],
                    draw_state["inference_timesteps"],
                )
                and torch.equal(
                    schedule_calls[0]["deltas"],
                    draw_state["inference_deltas"],
                )
            )
            terminal_prediction_is_last_step = bool(
                scheduler_steps
                and torch.equal(
                    scheduler_steps[-1]["result"], shared_world["prediction"]
                )
            )
            rollout_passed = bool(
                first_carrier_prefix.get("exact_equal") is True
                and first_carrier_future.get("exact_equal") is True
                and first_scheduler_sample.get("exact_equal") is True
                and all_prefix_placeholders_zero
                and carrier_matches_step_state
                and schedule_passed
                and terminal_prediction_is_last_step
                and shared_world["inference_steps"]
                == NATIVE_WORLD_INFERENCE_STEPS
                and shared_world["inference_shift"]
                == NATIVE_WORLD_INFERENCE_SHIFT
            )
            report["checks"][active_gate] = {
                "passed": rollout_passed,
                "metric_name": EXPECTED_NATIVE_WORLD_METRIC,
                "first_carrier_prefix_is_constant_zero": first_carrier_prefix,
                "first_carrier_future_is_exact_frozen_draw": first_carrier_future,
                "first_scheduler_sample_is_exact_frozen_draw": (
                    first_scheduler_sample
                ),
                "prepare_call_count": len(prepare_inputs),
                "scheduler_step_count": len(scheduler_steps),
                "all_prefix_placeholders_zero": all_prefix_placeholders_zero,
                "carrier_future_matches_scheduler_state_each_step": (
                    carrier_matches_step_state
                ),
                "native_schedule_exact": schedule_passed,
                "inference_steps": shared_world.get("inference_steps"),
                "inference_shift": shared_world.get("inference_shift"),
                "terminal_prediction_is_last_scheduler_result": (
                    terminal_prediction_is_last_step
                ),
                "real_future_target_used_during_rollout": False,
            }
            _require(
                rollout_passed,
                "World inference did not start from the exact frozen pure-noise draw "
                "and execute the exact frozen 10-step native schedule.",
                classification="WORLD-METRIC-NOT-VALIDATABLE",
            )

            active_gate = "cache_intervention_changes_both_consumers"
            altered_k = list(prepared.current_cache_k)
            altered_v = list(prepared.current_cache_v)
            for layer in LATE_LAYERS:
                altered_k[layer] = torch.zeros_like(prepared.current_cache_k[layer])
                altered_v[layer] = torch.zeros_like(prepared.current_cache_v[layer])
            altered_action_prediction = _action_prediction(
                model=model,
                state=draw_state,
                cache_k=altered_k,
                cache_v=altered_v,
            )
            altered_world = native_world_loss(
                model=model,
                prepared=prepared,
                video_cache_k=altered_k,
                video_cache_v=altered_v,
                video_noise=frozen["draw"]["video_noise"],
                disabled_video_prefix_layers=EARLY_DISABLED,
            )
            action_change = _comparison(
                shared_action_prediction,
                altered_action_prediction,
                atol=0.0,
                rtol=0.0,
            )
            world_change = _comparison(
                shared_world["prediction"],
                altered_world["prediction"],
                atol=0.0,
                rtol=0.0,
            )
            path_change_passed = bool(
                action_change.get("finite") is True
                and world_change.get("finite") is True
                and action_change["max_abs_diff"] > CACHE_PATH_CHANGE_MIN
                and world_change["max_abs_diff"] > CACHE_PATH_CHANGE_MIN
            )
            report["checks"][active_gate] = {
                "passed": path_change_passed,
                "alteration": "zero K/V at every shared late layer 15-29",
                "minimum_required_max_abs_change": CACHE_PATH_CHANGE_MIN,
                "action_prediction_change": action_change,
                "world_prediction_change": world_change,
            }
            _require(
                path_change_passed,
                "Changing the shared late K/V cache did not affect both consumers.",
                classification="SHARED-INTERFACE-NOT-AVAILABLE",
            )

            active_gate = "no_raw_prefix_residual_bypass"
            altered_carrier = draw_state["carrier"].clone()
            altered_carrier[:, :, 0:1] = altered_carrier[:, :, 0:1] + 3.0
            altered_prepared = model.video_expert.prepare(
                x=altered_carrier,
                timestep=draw_state["video_timestep"],
                context=prepared.inputs["context"],
                context_mask=prepared.inputs["context_mask"],
                action=None,
                fuse_vae_embedding_in_latents=True,
            )
            altered_video_tokens = altered_prepared[0]
            prefix_changed = _comparison(
                draw_state["video_tokens"][:, :PREFIX_TOKENS],
                altered_video_tokens[:, :PREFIX_TOKENS],
                atol=0.0,
                rtol=0.0,
            )
            future_inputs = {
                "tokens": _comparison(
                    draw_state["video_tokens"][:, PREFIX_TOKENS:],
                    altered_video_tokens[:, PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
                "t": _comparison(
                    draw_state["video_t"][:, PREFIX_TOKENS:],
                    altered_prepared[1][:, PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
                "t_mod": _comparison(
                    draw_state["video_t_mod"][:, PREFIX_TOKENS:],
                    altered_prepared[2][:, PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
                "context_mask": _comparison(
                    draw_state["video_context_mask"][:, PREFIX_TOKENS:],
                    altered_prepared[4][:, PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
                "context": _comparison(
                    draw_state["video_context"],
                    altered_prepared[3],
                    atol=0.0,
                    rtol=0.0,
                ),
                "freqs": _comparison(
                    draw_state["video_freqs"][PREFIX_TOKENS:],
                    altered_prepared[5][PREFIX_TOKENS:],
                    atol=0.0,
                    rtol=0.0,
                ),
            }
            altered_state = dict(draw_state)
            altered_state.update(
                {
                    "video_tokens": altered_video_tokens,
                    "video_t": altered_prepared[1],
                    "video_t_mod": altered_prepared[2],
                    "video_context": altered_prepared[3],
                    "video_context_mask": altered_prepared[4],
                    "video_freqs": altered_prepared[5],
                }
            )
            fixed_cache_future = _factorized_future_tokens(
                model=model,
                state=altered_state,
                cache_k=prepared.current_cache_k,
                cache_v=prepared.current_cache_v,
                disabled_layers=EARLY_DISABLED,
            )
            baseline_fixed_cache_future = _factorized_future_tokens(
                model=model,
                state=draw_state,
                cache_k=prepared.current_cache_k,
                cache_v=prepared.current_cache_v,
                disabled_layers=EARLY_DISABLED,
            )
            no_bypass_output = _comparison(
                baseline_fixed_cache_future,
                fixed_cache_future,
                atol=0.0,
                rtol=0.0,
            )
            signature = inspect.signature(
                model.mot.forward_future_video_with_video_cache_tensor
            )
            forbidden_parameters = {
                "prefix_video_tokens",
                "prefix_residual",
                "raw_prefix",
                "input_image",
            }
            signature_has_no_raw_prefix = forbidden_parameters.isdisjoint(signature.parameters)
            no_bypass_passed = bool(
                prefix_changed["max_abs_diff"] > 0.0
                and all(value.get("exact_equal") is True for value in future_inputs.values())
                and no_bypass_output.get("exact_equal") is True
                and signature_has_no_raw_prefix
            )
            report["checks"][active_gate] = {
                "passed": no_bypass_passed,
                "altered_raw_prefix_latent": True,
                "prefix_tokens_changed": prefix_changed,
                "future_consumer_inputs": future_inputs,
                "fixed_cache_future_output": no_bypass_output,
                "world_consumer_parameters": list(signature.parameters),
                "world_consumer_has_no_raw_prefix_argument": signature_has_no_raw_prefix,
                "scope_note": (
                    "The native carrier uses patch_t=1 and a constant-zero discarded "
                    "prefix placeholder. With selected K/V fixed, changing that discarded "
                    "slot cannot affect future-token computation."
                ),
            }
            _require(
                no_bypass_passed,
                "A raw prefix residual bypasses the explicit shared K/V interface.",
                classification="SHARED-INTERFACE-NOT-AVAILABLE",
            )

            active_gate = "scoring_target_not_in_inference"
            altered_target_latents = prepared.input_latents.clone()
            altered_target_latents[:, :, 1:] = (
                altered_target_latents[:, :, 1:] + 3.0
            )
            altered_target_prepared = replace(
                prepared,
                input_latents=altered_target_latents,
                target_latent_sha256=tensor_sha256(
                    altered_target_latents.detach().cpu()
                ),
            )
            altered_target_world = native_world_loss(
                model=model,
                prepared=altered_target_prepared,
                video_cache_k=prepared.current_cache_k,
                video_cache_v=prepared.current_cache_v,
                video_noise=frozen["draw"]["video_noise"],
                disabled_video_prefix_layers=EARLY_DISABLED,
            )
            target_changed = _comparison(
                shared_world["target"],
                altered_target_world["target"],
                atol=0.0,
                rtol=0.0,
            )
            target_invariant_prediction = _comparison(
                shared_world["prediction"],
                altered_target_world["prediction"],
                atol=0.0,
                rtol=0.0,
            )
            target_invariant_tokens = _comparison(
                shared_world["future_tokens"],
                altered_target_world["future_tokens"],
                atol=0.0,
                rtol=0.0,
            )
            manual_shared = _manual_reconstruction_metric(
                prepared=prepared, prediction=shared_world["prediction"]
            )
            manual_altered_target = _manual_reconstruction_metric(
                prepared=altered_target_prepared,
                prediction=altered_target_world["prediction"],
            )
            shared_loss_identity = _scalar_comparison(
                shared_world["native_world_loss"],
                manual_shared["native_world_loss"],
                atol=MANUAL_METRIC_ATOL,
                rtol=0.0,
            )
            altered_loss_identity = _scalar_comparison(
                altered_target_world["native_world_loss"],
                manual_altered_target["native_world_loss"],
                atol=MANUAL_METRIC_ATOL,
                rtol=0.0,
            )
            target_is_scoring_only = bool(
                target_changed.get("max_abs_diff", 0.0) > 0.0
                and target_invariant_prediction.get("exact_equal") is True
                and target_invariant_tokens.get("exact_equal") is True
            )
            metric_reads_target = bool(
                shared_loss_identity["passed"]
                and altered_loss_identity["passed"]
                and shared_world["native_world_loss"]
                != altered_target_world["native_world_loss"]
            )
            report["checks"][active_gate] = {
                "passed": bool(target_is_scoring_only and metric_reads_target),
                "target_perturbation": "add 3.0 to scoring-only future latent",
                "real_future_target_changed": target_changed,
                "generated_prediction_unchanged_exactly": (
                    target_invariant_prediction
                ),
                "final_future_tokens_unchanged_exactly": target_invariant_tokens,
                "original_terminal_mse_identity": shared_loss_identity,
                "altered_terminal_mse_identity": altered_loss_identity,
                "reported_score_changed": metric_reads_target,
                "future_target_enters_inference": False,
            }
            _require(
                target_is_scoring_only,
                "Changing the scoring-only future target changed world inference.",
                classification="SHARED-INTERFACE-NOT-AVAILABLE",
            )
            _require(
                metric_reads_target,
                "Native terminal world score is not the direct future-latent MSE.",
                classification="WORLD-METRIC-NOT-VALIDATABLE",
            )

            active_gate = "projection_endpoints_and_registered_ranks"
            endpoints = projection_endpoint_caches(prepared=prepared, model=model)
            rank0_k, rank0_v, rank0_audit = endpoints["rank0"]
            rankd_k, rankd_v, rankd_audit = endpoints["rankD"]
            rank0_identity = _cache_comparison(
                rank0_k,
                rank0_v,
                prepared.wrong_cache_k,
                prepared.wrong_cache_v,
            )
            rankd_identity = _cache_comparison(
                rankd_k,
                rankd_v,
                prepared.current_cache_k,
                prepared.current_cache_v,
            )
            rank_specs = load_rank_specs(model=model, preflight=frozen["preflight"])
            nested_rows: list[dict[str, Any]] = []
            bases_nested = True
            for layer in LATE_LAYERS:
                for kind in ("k", "v"):
                    basis97 = rank_specs[97].bases_by_layer[layer][kind]
                    basis170 = rank_specs[170].bases_by_layer[layer][kind]
                    exact = bool(torch.equal(basis97, basis170[:, :97]))
                    finite = bool(
                        torch.isfinite(basis97).all().item()
                        and torch.isfinite(basis170).all().item()
                    )
                    shape_ok = bool(
                        tuple(basis97.shape) == (EXPECTED_FEATURE_DIM, 97)
                        and tuple(basis170.shape) == (EXPECTED_FEATURE_DIM, 170)
                    )
                    bases_nested &= exact and finite and shape_ok
                    nested_rows.append(
                        {
                            "layer": layer,
                            "tensor_kind": kind.upper(),
                            "r97_exact_prefix_of_r170": exact,
                            "finite": finite,
                            "shape_r97": list(basis97.shape),
                            "shape_r170": list(basis170.shape),
                        }
                    )
            projected_rows: list[dict[str, Any]] = []
            projected_passed = True
            for condition in ("svd_r97", "svd_r170"):
                cache_k, cache_v, audit = condition_cache(
                    prepared=prepared,
                    condition=condition,
                    rank_specs=rank_specs,
                )
                rank = int(condition.removeprefix("svd_r"))
                structural = all(
                    tuple(tensor.shape) == (1, PREFIX_TOKENS, EXPECTED_FEATURE_DIM)
                    and tensor.dtype == prepared.current_cache_k[0].dtype
                    and tensor.device == prepared.current_cache_k[0].device
                    and bool(torch.isfinite(tensor).all().item())
                    for tensor in (*cache_k, *cache_v)
                )
                audit_ok = bool(
                    audit is not None
                    and audit.get("projection_rank") == rank
                    and audit.get("replacement_video_layers") == list(LATE_LAYERS)
                    and audit.get("action_visible_token_count") == PREFIX_TOKENS
                    and audit.get("tokens_modified") is False
                    and audit.get("heads_modified") is False
                    and audit.get("shape_preserved") is True
                    and audit.get("k_v_bases_independent") is True
                )
                projected_passed &= structural and audit_ok
                projected_rows.append(
                    {
                        "condition": condition,
                        "rank": rank,
                        "passed": bool(structural and audit_ok),
                        "all_30_k_v_shape_dtype_device_finite": structural,
                        "projection_audit": audit,
                    }
                )
            projection_passed = bool(
                rank0_identity["passed"]
                and rankd_identity["passed"]
                and rank0_audit.get("projection_rank") == 0
                and rankd_audit.get("projection_rank") == EXPECTED_FEATURE_DIM
                and bases_nested
                and projected_passed
            )
            report["checks"][active_gate] = {
                "passed": projection_passed,
                "rank0_equals_wrong_cache_exactly": rank0_identity,
                "rankD_equals_current_cache_exactly": rankd_identity,
                "rank0_projection_audit": rank0_audit,
                "rankD_projection_audit": rankd_audit,
                "registered_basis_prefixes": {
                    "passed": bases_nested,
                    "rows": nested_rows,
                },
                "registered_projection_rows": projected_rows,
                "token_count_preserved": True,
                "attention_key_count_preserved": True,
                "head_count_preserved": True,
            }
            _require(
                projection_passed,
                "Projection endpoints, registered rank prefixes, or cache geometry failed.",
                classification="SHARED-INTERFACE-NOT-AVAILABLE",
            )

            active_gate = "native_metric_identity_and_fixed_draw_reproducibility"
            native_result = shared_world
            manual_native = _manual_reconstruction_metric(
                prepared=prepared,
                prediction=native_result["prediction"],
            )
            helper_manual_checks = {
                key: _scalar_comparison(
                    native_result[key],
                    manual_native[key],
                    atol=MANUAL_METRIC_ATOL,
                    rtol=0.0,
                )
                for key in ("native_world_loss", "future_latent_mse")
            }
            native_alias_identity = _scalar_comparison(
                native_result["native_world_loss"],
                native_result["future_latent_mse"],
                atol=0.0,
                rtol=0.0,
            )
            repeated_world = native_world_loss(
                model=model,
                prepared=prepared,
                video_cache_k=prepared.current_cache_k,
                video_cache_v=prepared.current_cache_v,
                video_noise=frozen["draw"]["video_noise"],
                disabled_video_prefix_layers=EARLY_DISABLED,
            )
            repeated_prediction = _comparison(
                shared_world["prediction"],
                repeated_world["prediction"],
                atol=0.0,
                rtol=0.0,
            )
            repeated_loss = _scalar_comparison(
                shared_world["native_world_loss"],
                repeated_world["native_world_loss"],
                atol=0.0,
                rtol=0.0,
            )
            repeated_tokens = _comparison(
                shared_world["future_tokens"],
                repeated_world["future_tokens"],
                atol=0.0,
                rtol=0.0,
            )
            returned_target_identity = _comparison(
                native_result["target"],
                prepared.input_latents[:, :, 1:],
                atol=0.0,
                rtol=0.0,
            )
            native_metric_passed = bool(
                all(check["passed"] for check in helper_manual_checks.values())
                and native_alias_identity["passed"]
                and repeated_prediction.get("exact_equal") is True
                and repeated_tokens.get("exact_equal") is True
                and repeated_loss["passed"]
                and native_result["prediction_shape"] == [1, 48, 2, 14, 28]
                and native_result["target_shape"] == [1, 48, 2, 14, 28]
                and native_result["future_token_shape"] == [1, 196, 3072]
                and native_result["inference_steps"]
                == NATIVE_WORLD_INFERENCE_STEPS
                and native_result["inference_shift"]
                == NATIVE_WORLD_INFERENCE_SHIFT
                and returned_target_identity.get("exact_equal") is True
            )
            report["checks"][active_gate] = {
                "passed": native_metric_passed,
                "metric_name": EXPECTED_NATIVE_WORLD_METRIC,
                "definition": (
                    "unweighted terminal MSE between the 10-step pure-noise native "
                    "future-latent prediction and scoring-only real future latent"
                ),
                "manual_terminal_mse_vs_runtime": helper_manual_checks,
                "native_world_loss_equals_future_latent_mse": (
                    native_alias_identity
                ),
                "fixed_draw_repeated_prediction": repeated_prediction,
                "fixed_draw_repeated_future_tokens": repeated_tokens,
                "fixed_draw_repeated_loss": repeated_loss,
                "returned_scoring_target_identity": returned_target_identity,
                "prediction_shape": native_result["prediction_shape"],
                "target_shape": native_result["target_shape"],
                "future_token_shape": native_result["future_token_shape"],
                "inference_steps": native_result["inference_steps"],
                "inference_shift": native_result["inference_shift"],
                "initial_state": "fixed pure Gaussian future-latent noise",
                "carrier_prefix": "constant-zero discarded placeholder",
                "target_enters_inference": False,
                "draws_per_evaluation_sample": WORLD_DRAWS_PER_SAMPLE,
            }
            _require(
                native_metric_passed,
                "Native world-loss identity or fixed-draw reproducibility failed.",
                classification="WORLD-METRIC-NOT-VALIDATABLE",
            )

        report.update(
            {
                "status": "passed",
                "passed": True,
                "phase_b_authorized": True,
                "recommended_special_classification": None,
                "shared_interface": {
                    "name": "late_first_frame_video_kv_prefix",
                    "layers": list(LATE_LAYERS),
                    "k_v_shape_per_layer": [1, PREFIX_TOKENS, EXPECTED_FEATURE_DIM],
                    "heads": 24,
                    "head_dim": 128,
                    "preferred_path": "A",
                    "same_cache_object_verified_for_both_consumers": True,
                    "raw_prefix_residual_bypass_detected": False,
                    "carrier_prefix": "constant-zero discarded placeholder",
                    "current_observation_pathways": ["selected external K/V cache"],
                    "future_target_enters_inference": False,
                    "native_world_metric": EXPECTED_NATIVE_WORLD_METRIC,
                    "native_world_inference_steps": NATIVE_WORLD_INFERENCE_STEPS,
                    "native_world_inference_shift": NATIVE_WORLD_INFERENCE_SHIFT,
                },
            }
        )
    except Exception as exc:  # Always persist an auditable failed gate.
        recommended = (
            exc.classification if isinstance(exc, MachineryGateError) else None
        )
        report.update(
            {
                "status": "failed",
                "passed": False,
                "phase_b_authorized": False,
                "recommended_special_classification": recommended,
                "failure": {
                    "active_gate": active_gate,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--world-manifest", type=Path, required=True)
    parser.add_argument("--stochastic-manifest", type=Path, required=True)
    parser.add_argument("--draw-tensors", type=Path, required=True)
    parser.add_argument(
        "--processed-targets",
        "--target-manifest",
        dest="processed_targets",
        type=Path,
        required=True,
    )
    parser.add_argument("--runtime-work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output_md = (
        args.output_md.resolve()
        if args.output_md is not None
        else output.with_suffix(".md")
    )
    report = run(args)
    _publish_report_bundle(output=output, output_md=output_md, report=report)
    if not report["passed"]:
        raise RuntimeError(
            "Salvage-B machinery gate failed; Phase B is not authorized. "
            f"Inspect {output}."
        )
    print(f"Salvage-B machinery passed: {output}")


if __name__ == "__main__":
    main()
