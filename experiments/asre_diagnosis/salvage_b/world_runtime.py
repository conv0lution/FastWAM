"""Shared frozen-model runtime helpers for Salvage-B world evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.asre_diagnosis.common import sha256_file  # noqa: E402
from experiments.asre_diagnosis.round3b.donor import (  # noqa: E402
    OnlineDonorBundle,
    tensor_sha256,
)
from experiments.asre_diagnosis.round4b.basis import (  # noqa: E402
    EXPECTED_FEATURE_DIM,
    LATE_LAYERS,
    load_runtime_basis,
)
from fastwam.models.wan22.video_cache_replacement import (  # noqa: E402
    project_replacement_video_cache,
    select_replacement_video_cache,
)
from fastwam.utils import misc  # noqa: E402


TASK_CONFIG = "libero_uncond_2cam224_1e-4"
DATA_CONFIG = "libero_2cam_lerobot_v30"
EARLY_DISABLED = tuple(range(15))
PREFIX_TOKENS = 98
ACTION_TOKENS = 32
EXPECTED_VIDEO_SHAPE = (3, 9, 224, 448)
EXPECTED_ACTION_SHAPE = (32, 7)
NATIVE_WORLD_INFERENCE_STEPS = 10
NATIVE_WORLD_INFERENCE_SHIFT = 5.0


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_clean_source(expected_commit: str, *, output_root: Path | None = None) -> None:
    """Fail closed if source changes during a formal long-running stage."""

    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    status_command = [
        "git",
        "status",
        "--porcelain",
        "--untracked-files=normal",
        "--",
        ".",
    ]
    if output_root is not None:
        resolved_output = output_root.resolve()
        try:
            relative_output = resolved_output.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError(
                f"Salvage-B output root must be inside the repository: {resolved_output}"
            ) from exc
        status_command.append(f":(exclude){relative_output}/**")
    dirty = subprocess.check_output(
        status_command,
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if observed != expected_commit or dirty:
        raise RuntimeError(
            "Salvage-B source changed during formal execution: "
            f"HEAD={observed}, expected={expected_commit}, dirty={dirty!r}."
        )


def validate_architecture_source_inventory(
    preflight: Mapping[str, Any], *, expected_commit: str
) -> dict[str, Any]:
    """Revalidate the exact Phase-A source inventory without trusting paths.

    The architecture audit lives under the ignored experiment output root, so
    a clean Git worktree alone cannot authenticate it.  Bind it through the
    frozen preflight hash, require every inventory entry to be a canonical
    repository-relative regular file, and hash the audit again after walking
    the inventory so an in-flight audit replacement cannot pass unnoticed.
    """

    phase_a = preflight.get("phase_a")
    if not isinstance(phase_a, Mapping):
        raise ValueError("Salvage-B preflight lacks the frozen Phase-A contract.")
    raw_audit_path = phase_a.get("architecture_audit_path")
    declared_audit_sha = phase_a.get("architecture_audit_sha256")
    if not isinstance(raw_audit_path, str) or not raw_audit_path:
        raise ValueError("Salvage-B preflight lacks an architecture-audit path.")
    if not isinstance(declared_audit_sha, str) or len(declared_audit_sha) != 64:
        raise ValueError("Salvage-B preflight has a malformed architecture-audit hash.")
    try:
        int(declared_audit_sha, 16)
    except ValueError as exc:
        raise ValueError(
            "Salvage-B preflight has a malformed architecture-audit hash."
        ) from exc

    audit_path = Path(raw_audit_path).expanduser().resolve()
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing frozen architecture audit: {audit_path}")
    initial_audit_sha = sha256_file(audit_path)
    if initial_audit_sha != declared_audit_sha:
        raise ValueError("Frozen Salvage-B architecture audit drifted.")
    audit = read_json(audit_path)
    if (
        audit.get("artifact_type")
        != "asre_salvage_b_phase_a_architecture_audit"
        or int(audit.get("schema_version", -1)) != 1
        or audit.get("git_commit_hash") != expected_commit
    ):
        raise ValueError("Incompatible frozen Salvage-B architecture audit.")

    inventory = audit.get("source_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Frozen architecture audit lacks a source inventory.")
    repository_root = PROJECT_ROOT.resolve()
    seen: set[Path] = set()
    for record in inventory:
        if not isinstance(record, Mapping):
            raise ValueError("Architecture source inventory contains a malformed record.")
        raw_relative = record.get("path")
        declared_sha = record.get("sha256")
        if not isinstance(raw_relative, str) or not raw_relative:
            raise ValueError("Architecture source inventory contains an empty path.")
        relative = Path(raw_relative)
        if (
            relative.is_absolute()
            or raw_relative != relative.as_posix()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(
                f"Architecture source path is not canonical repo-relative: {raw_relative!r}."
            )
        source_path = (repository_root / relative).resolve()
        try:
            source_path.relative_to(repository_root)
        except ValueError as exc:
            raise ValueError(
                f"Architecture source path escapes the repository: {raw_relative!r}."
            ) from exc
        if source_path in seen:
            raise ValueError(
                f"Duplicate architecture source inventory path: {raw_relative!r}."
            )
        seen.add(source_path)
        if not isinstance(declared_sha, str) or len(declared_sha) != 64:
            raise ValueError(
                f"Malformed architecture source hash for {raw_relative!r}."
            )
        try:
            int(declared_sha, 16)
        except ValueError as exc:
            raise ValueError(
                f"Malformed architecture source hash for {raw_relative!r}."
            ) from exc
        if not source_path.is_file() or sha256_file(source_path) != declared_sha:
            raise ValueError(f"Phase-A audited source drifted: {source_path}")

    final_audit_sha = sha256_file(audit_path)
    if final_audit_sha != initial_audit_sha or final_audit_sha != declared_audit_sha:
        raise ValueError("Frozen Salvage-B architecture audit changed during validation.")
    return audit


def validate_declared_source_artifacts(
    world: Mapping[str, Any], *, include_targets: bool = True
) -> None:
    """Re-hash the metadata and source files frozen by the world manifest."""

    collections = ["metadata_files"]
    if include_targets:
        collections.append("target_source_files")
    for collection in collections:
        records = world.get(collection)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Frozen world manifest lacks {collection}.")
        seen: set[Path] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Malformed frozen source record in {collection}.")
            path = Path(str(record.get("path", ""))).resolve()
            if path in seen:
                raise ValueError(f"Duplicate frozen source path in {collection}: {path}")
            seen.add(path)
            if (
                not path.is_file()
                or path.stat().st_size != int(record.get("size", -1))
                or sha256_file(path) != record.get("sha256")
            ):
                raise ValueError(f"Frozen world source drifted: {path}")


def compose_world_config():
    with initialize_config_dir(
        config_dir=str((PROJECT_ROOT / "configs").resolve()), version_base="1.3"
    ):
        return compose(
            config_name="sim_libero.yaml",
            overrides=[f"task={TASK_CONFIG}", f"data={DATA_CONFIG}"],
        )


def load_world_dataset(
    *, preflight: Mapping[str, Any], runtime_work_dir: Path
):
    """Instantiate the official v3 dataset with frozen training normalization."""

    dataset_root = Path(str(preflight["world_data"]["dataset_root"])).resolve()
    stats_path = Path(str(preflight["state"]["dataset_stats_path"])).resolve()
    if sha256_file(stats_path) != str(preflight["state"]["dataset_stats_sha256"]):
        raise ValueError("Frozen dataset normalization statistics drifted.")
    cfg = compose_world_config()
    train_cfg = cfg.data.train
    with open_dict(train_cfg):
        train_cfg.dataset_dirs = [str(dataset_root)]
        train_cfg.val_set_proportion = 0.0
        train_cfg.is_training_set = False
        train_cfg.skip_padding_as_possible = False
        train_cfg.use_text_embed_cache = False
    runtime_work_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(runtime_work_dir)
    dataset = instantiate(
        train_cfg,
        pretrained_norm_stats=str(stats_path),
    )
    if len(dataset) != 53229:
        raise ValueError(f"Unexpected official LIBERO-Spatial frame count: {len(dataset)}")
    # Expose the lower-level resolved index so a decode retry can never
    # silently substitute a random unregistered trajectory.
    dataset._include_source_dataset_index = True
    return dataset


def load_prompt_cache(preflight: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(preflight["state"]["prompt_context_cache_path"])).resolve()
    if sha256_file(path) != str(preflight["state"]["prompt_context_cache_sha256"]):
        raise ValueError("Frozen prompt-context cache drifted.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("prompts"), dict):
        raise ValueError(f"Malformed prompt-context cache: {path}")
    return payload


def load_donor_bundle(preflight: Mapping[str, Any]) -> OnlineDonorBundle:
    donors = preflight["donors"]
    bundle = OnlineDonorBundle.load(
        mapping_path=Path(str(donors["mapping_path"])).resolve(),
        observation_manifest_path=Path(str(donors["manifest_path"])).resolve(),
        observation_root=Path(str(donors["root"])).resolve(),
    )
    if (
        bundle.mapping_sha256 != donors["mapping_sha256"]
        or bundle.observation_manifest_sha256 != donors["manifest_sha256"]
    ):
        raise ValueError("Frozen donor bundle drifted.")
    return bundle


def load_frozen_model(preflight: Mapping[str, Any]):
    # Keep the LIBERO evaluation stack lazy: CPU manifest/target validation and
    # unit tests do not need the simulator imports pulled in by fit_worker.
    from experiments.asre_diagnosis.round4b.fit_worker import _load_model

    checkpoint = Path(str(preflight["state"]["checkpoint_path"])).resolve()
    if sha256_file(checkpoint) != str(preflight["state"]["checkpoint_sha256"]):
        raise ValueError("Frozen Fast-WAM checkpoint drifted.")
    model, cfg = _load_model(checkpoint)
    if bool(getattr(model.video_expert, "action_conditioned", True)):
        raise ValueError("Native video expert unexpectedly consumes action conditioning.")
    return model, cfg


def _prompt_context(
    *, record: Mapping[str, Any], prompt: str, prompt_cache: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    entry = prompt_cache["prompts"].get(prompt)
    if not isinstance(entry, Mapping):
        raise KeyError(f"Frozen prompt cache lacks world prompt: {prompt}")
    if (
        int(entry.get("task_id", -1)) != int(record["task_id"])
        or str(entry.get("task_description")) != str(record["task_description"])
    ):
        raise ValueError("Prompt cache task identity does not match world manifest.")
    context = entry.get("context")
    mask = entry.get("context_mask")
    if (
        not torch.is_tensor(context)
        or not torch.is_tensor(mask)
        or tuple(context.shape) != (1, 128, 4096)
        or tuple(mask.shape) != (1, 128)
    ):
        raise ValueError("Frozen prompt cache has an unexpected tensor layout.")
    return context.clone(), mask.bool().clone()


def load_processed_sample(
    *,
    dataset,
    record: Mapping[str, Any],
    prompt_cache: Mapping[str, Any],
    target_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    requested_index = int(record["dataset_index"])
    sample = dataset._get(requested_index)
    if int(sample.get("_source_dataset_index", -1)) != requested_index:
        raise ValueError(
            "Dataset decode substituted an unregistered sample for "
            f"{record['sample_id']}: {sample.get('_source_dataset_index')} != "
            f"{requested_index}."
        )
    prompt = str(sample.get("prompt"))
    expected_prompt = (
        "A video recorded from a robot's point of view executing the following "
        f"instruction: {record['task_description']}"
    )
    if prompt != expected_prompt:
        raise ValueError(
            f"World prompt drifted for {record['sample_id']}: {prompt!r} != {expected_prompt!r}"
        )
    required_tensors = {
        "video": EXPECTED_VIDEO_SHAPE,
        "action": EXPECTED_ACTION_SHAPE,
        "proprio": (32, 8),
        "image_is_pad": (9,),
        "action_is_pad": (32,),
    }
    for key, shape in required_tensors.items():
        tensor = sample.get(key)
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape:
            raise ValueError(
                f"Processed {key} layout drifted for {record['sample_id']}: "
                f"{getattr(tensor, 'shape', None)} != {shape}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"Processed {key} is nonfinite for {record['sample_id']}.")
    if bool(sample["image_is_pad"].any()) or bool(sample["action_is_pad"].any()):
        raise ValueError(f"Frozen world clip unexpectedly contains padding: {record['sample_id']}")
    hashes = {
        key: tensor_sha256(sample[key])
        for key in required_tensors
    }
    hashes["prompt_sha256"] = sha256_text(prompt)
    if target_record is not None:
        expected_hashes = target_record.get("processed_tensor_sha256")
        if hashes != expected_hashes:
            raise ValueError(f"Processed target drifted for {record['sample_id']}.")
    context, context_mask = _prompt_context(
        record=record, prompt=prompt, prompt_cache=prompt_cache
    )
    return {
        "video": sample["video"].unsqueeze(0),
        "action": sample["action"].unsqueeze(0),
        "proprio": sample["proprio"].unsqueeze(0),
        "image_is_pad": sample["image_is_pad"].unsqueeze(0),
        "action_is_pad": sample["action_is_pad"].unsqueeze(0),
        "prompt": prompt,
        "context": context,
        "context_mask": context_mask,
        "processed_tensor_sha256": hashes,
    }


@dataclass
class PreparedWorldSample:
    sample_id: str
    task_id: int
    episode_id: int
    trial: int
    inputs: dict[str, torch.Tensor]
    input_latents: torch.Tensor
    # Provenance witness only.  The world metric deliberately does not pass
    # this tensor to its future-video consumer; observation values enter only
    # through current_cache_{k,v} / wrong_cache_{k,v}.
    current_frame_latent: torch.Tensor
    current_cache_k: list[torch.Tensor]
    current_cache_v: list[torch.Tensor]
    wrong_cache_k: list[torch.Tensor]
    wrong_cache_v: list[torch.Tensor]
    prefix_context: torch.Tensor
    prefix_context_mask: torch.Tensor
    prefix_tokens: int
    current_image_sha256: str
    donor_image_sha256: str
    target_latent_sha256: str


def _prefill_prefix(
    *,
    model,
    first_frame_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    timestep = torch.zeros(
        (first_frame_latents.shape[0],),
        device=first_frame_latents.device,
        dtype=first_frame_latents.dtype,
    )
    prepared = model.video_expert.prepare(
        x=first_frame_latents,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    tokens, _t, t_mod, video_context, video_context_mask, freqs, f, h, w, tpf = prepared
    if (int(f), int(h), int(w), int(tpf), int(tokens.shape[1])) != (1, 7, 14, 98, 98):
        raise ValueError("Shared first-frame video-prefix layout drifted.")
    mask = model.video_expert.build_video_to_video_mask(
        video_seq_len=int(tokens.shape[1]),
        video_tokens_per_frame=int(tpf),
        device=tokens.device,
    )
    cache_k, cache_v = model.mot.prefill_video_cache_tensor(
        video_tokens=tokens,
        video_freqs=freqs,
        video_t_mod=t_mod,
        video_context=video_context,
        video_context_mask=video_context_mask,
        video_attention_mask=mask,
    )
    return cache_k, cache_v, {
        "tokens": tokens,
        "t_mod": t_mod,
        "context": video_context,
        "context_mask": video_context_mask,
        "freqs": freqs,
        "mask": mask,
    }


def prepare_world_sample(
    *,
    model,
    processed: Mapping[str, Any],
    record: Mapping[str, Any],
    donor_bundle: OnlineDonorBundle,
) -> PreparedWorldSample:
    from experiments.asre_diagnosis.round4b.fit_worker import _load_donor_image

    model_inputs = model.build_inputs(dict(processed), tiled=False)
    input_latents = model_inputs["input_latents"]
    if tuple(input_latents.shape) != (1, 48, 3, 14, 28):
        raise ValueError(f"Native VAE world-target layout drifted: {input_latents.shape}")
    context = model_inputs["context"]
    context_mask = model_inputs["context_mask"]
    current_image = processed["video"][:, :, 0]
    donor_image = _load_donor_image(
        donor_bundle,
        task_id=int(record["task_id"]),
        episode_id=int(record["trial"]),
    ).to(dtype=current_image.dtype)
    if donor_image.ndim == 3:
        donor_image = donor_image.unsqueeze(0)
    if tuple(donor_image.shape) != tuple(current_image.shape):
        raise ValueError("Frozen current/donor images do not share the exact representation shape.")
    current_hash = tensor_sha256(current_image.cpu())
    donor_hash = tensor_sha256(donor_image.cpu())
    if donor_hash != str(record["donor_processed_image_sha256"]):
        raise ValueError("World manifest donor image hash drifted.")
    if current_hash == donor_hash:
        raise ValueError("Wrong endpoint donor is identical to the current world image.")
    current_latent = model._encode_input_image_latents_tensor(
        current_image.to(device=model.device, dtype=model.torch_dtype), tiled=False
    )
    donor_latent = model._encode_input_image_latents_tensor(
        donor_image.to(device=model.device, dtype=model.torch_dtype), tiled=False
    )
    current_k, current_v, current_prepared = _prefill_prefix(
        model=model,
        first_frame_latents=current_latent,
        context=context,
        context_mask=context_mask,
    )
    donor_k, donor_v, _donor_prepared = _prefill_prefix(
        model=model,
        first_frame_latents=donor_latent,
        context=context,
        context_mask=context_mask,
    )
    wrong_k, wrong_v = select_replacement_video_cache(
        current_cache_k=current_k,
        current_cache_v=current_v,
        replacement_cache_k=donor_k,
        replacement_cache_v=donor_v,
        replacement_video_layers=LATE_LAYERS,
        num_layers=model.mot.num_layers,
    )
    return PreparedWorldSample(
        sample_id=str(record["sample_id"]),
        task_id=int(record["task_id"]),
        episode_id=int(record["episode_id"]),
        trial=int(record["trial"]),
        inputs=model_inputs,
        input_latents=input_latents,
        current_frame_latent=current_latent,
        current_cache_k=current_k,
        current_cache_v=current_v,
        wrong_cache_k=wrong_k,
        wrong_cache_v=wrong_v,
        prefix_context=current_prepared["context"],
        prefix_context_mask=current_prepared["context_mask"],
        prefix_tokens=PREFIX_TOKENS,
        current_image_sha256=current_hash,
        donor_image_sha256=donor_hash,
        target_latent_sha256=tensor_sha256(input_latents.detach().cpu()),
    )


def load_rank_specs(*, model, preflight: Mapping[str, Any]) -> dict[int, Any]:
    basis_path = Path(str(preflight["basis"]["path"])).resolve()
    if sha256_file(basis_path) != str(preflight["basis"]["sha256"]):
        raise ValueError("Frozen Round-4B basis manifest drifted.")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return {
        rank: load_runtime_basis(
            manifest_path=basis_path,
            expected_sha256=str(preflight["basis"]["sha256"]),
            basis_kind="svd",
            rank=rank,
            device=device,
            dtype=dtype,
        )
        for rank in (97, 170)
    }


def condition_cache(
    *, prepared: PreparedWorldSample, condition: str, rank_specs: Mapping[int, Any]
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any] | None]:
    if condition == "current_all":
        return prepared.current_cache_k, prepared.current_cache_v, None
    if condition == "wrong_all":
        return prepared.wrong_cache_k, prepared.wrong_cache_v, None
    if condition not in {"svd_r97", "svd_r170"}:
        raise ValueError(f"Unsupported frozen Salvage-B condition: {condition}")
    rank = int(condition.removeprefix("svd_r"))
    spec = rank_specs[rank]
    cache_k, cache_v, audit = project_replacement_video_cache(
        current_cache_k=prepared.current_cache_k,
        current_cache_v=prepared.current_cache_v,
        replacement_cache_k=prepared.wrong_cache_k,
        replacement_cache_v=prepared.wrong_cache_v,
        replacement_video_layers=LATE_LAYERS,
        action_visible_token_indices=tuple(range(PREFIX_TOKENS)),
        feature_bases_by_layer=spec.bases_by_layer,
        projection_rank=rank,
        num_layers=len(prepared.current_cache_k),
    )
    return cache_k, cache_v, audit


def native_world_loss(
    *,
    model,
    prepared: PreparedWorldSample,
    video_cache_k: list[torch.Tensor],
    video_cache_v: list[torch.Tensor],
    video_noise: torch.Tensor,
    disabled_video_prefix_layers: tuple[int, ...] = EARLY_DISABLED,
) -> dict[str, Any]:
    """Run native pure-noise video inference and score real future latents.

    The causal-VAE target is scoring-only and is never supplied to the
    predictor.  A three-frame carrier preserves native temporal positions,
    but its first-frame tokens are discarded; the frozen cache is the only
    visual prefix consumed by the future-token core.
    """

    target_latents = prepared.input_latents
    target_future = target_latents[:, :, 1:]
    future_latents = video_noise.to(
        device=target_latents.device, dtype=target_latents.dtype
    ).unsqueeze(0)
    if tuple(future_latents.shape) != tuple(target_future.shape):
        raise ValueError(
            f"Frozen future-noise layout drifted: {future_latents.shape} != "
            f"{target_future.shape}"
        )
    timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=NATIVE_WORLD_INFERENCE_STEPS,
        device=target_latents.device,
        dtype=target_latents.dtype,
        shift_override=NATIVE_WORLD_INFERENCE_SHIFT,
    )
    if (
        tuple(timesteps.shape) != (NATIVE_WORLD_INFERENCE_STEPS,)
        or tuple(deltas.shape) != (NATIVE_WORLD_INFERENCE_STEPS,)
        or not bool(torch.isfinite(timesteps).all())
        or not bool(torch.isfinite(deltas).all())
        or not bool((deltas < 0).all())
        or not torch.allclose(
            deltas.float().sum(),
            torch.tensor(-1.0, device=deltas.device),
            atol=2e-2,
            rtol=0.0,
        )
    ):
        raise ValueError("Native video inference schedule drifted.")
    final_tokens: torch.Tensor | None = None
    for step_t, step_delta in zip(timesteps, deltas):
        timestep = step_t.reshape(1).to(
            device=target_latents.device, dtype=target_latents.dtype
        )
        # `prepare` needs the native three-frame temporal layout, but the
        # prefix token it creates is discarded before the MoT call.  Use a
        # constant zero placeholder so the shared K/V cache is literally the
        # only current-observation pathway into future computation.
        prefix_placeholder = torch.zeros(
            (
                future_latents.shape[0],
                future_latents.shape[1],
                1,
                future_latents.shape[3],
                future_latents.shape[4],
            ),
            device=future_latents.device,
            dtype=future_latents.dtype,
        )
        carrier = torch.cat([prefix_placeholder, future_latents], dim=2)
        video = model.video_expert.prepare(
            x=carrier,
            timestep=timestep,
            context=prepared.inputs["context"],
            context_mask=prepared.inputs["context_mask"],
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        tokens, t, t_mod, context, context_mask, freqs, f, h, w, tpf = video
        if (int(f), int(h), int(w), int(tpf), int(tokens.shape[1])) != (
            3,
            7,
            14,
            98,
            294,
        ):
            raise ValueError("Native future-video token layout drifted.")
        prefix = int(tpf)
        video_mask = model.video_expert.build_video_to_video_mask(
            video_seq_len=int(tokens.shape[1]),
            video_tokens_per_frame=prefix,
            device=tokens.device,
        )
        final_tokens = model.mot.forward_future_video_with_video_cache_tensor(
            future_video_tokens=tokens[:, prefix:],
            future_video_freqs=freqs[prefix:],
            future_video_t_mod=t_mod[:, prefix:],
            future_video_context=context,
            future_video_context_mask=context_mask[:, prefix:],
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            future_video_attention_mask=video_mask[prefix:, :],
            disabled_video_prefix_layers=disabled_video_prefix_layers,
        )
        velocity = model.video_expert.post(
            final_tokens,
            t[:, prefix:],
            int(f) - 1,
            int(h),
            int(w),
        )
        future_latents = model.infer_video_scheduler.step(
            velocity, step_delta, future_latents
        )
    if final_tokens is None or tuple(future_latents.shape) != tuple(target_future.shape):
        raise AssertionError("Native future-video inference produced no valid prediction.")
    if not bool(torch.isfinite(future_latents).all()) or not bool(
        torch.isfinite(target_future).all()
    ):
        raise ValueError("Native future-video prediction or scoring target is nonfinite.")
    loss = torch.nn.functional.mse_loss(
        future_latents.float(), target_future.float(), reduction="mean"
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("Native future-video reconstruction loss is nonfinite.")
    return {
        "native_world_loss": float(loss.item()),
        "future_latent_mse": float(loss.item()),
        "inference_steps": NATIVE_WORLD_INFERENCE_STEPS,
        "inference_shift": NATIVE_WORLD_INFERENCE_SHIFT,
        "prediction_shape": list(future_latents.shape),
        "target_shape": list(target_future.shape),
        "future_token_shape": list(final_tokens.shape),
        "prediction": future_latents,
        "future_tokens": final_tokens,
        "target": target_future,
    }


def projection_endpoint_caches(
    *, prepared: PreparedWorldSample, model
) -> dict[str, tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]]:
    device = prepared.current_cache_k[0].device
    dtype = prepared.current_cache_k[0].dtype
    zero = torch.empty((EXPECTED_FEATURE_DIM, 0), device=device, dtype=dtype)
    identity = torch.eye(EXPECTED_FEATURE_DIM, device=device, dtype=dtype)
    return {
        "rank0": project_replacement_video_cache(
            current_cache_k=prepared.current_cache_k,
            current_cache_v=prepared.current_cache_v,
            replacement_cache_k=prepared.wrong_cache_k,
            replacement_cache_v=prepared.wrong_cache_v,
            replacement_video_layers=LATE_LAYERS,
            action_visible_token_indices=tuple(range(PREFIX_TOKENS)),
            feature_bases_by_layer={layer: {"k": zero, "v": zero} for layer in LATE_LAYERS},
            projection_rank=0,
            num_layers=model.mot.num_layers,
        ),
        "rankD": project_replacement_video_cache(
            current_cache_k=prepared.current_cache_k,
            current_cache_v=prepared.current_cache_v,
            replacement_cache_k=prepared.wrong_cache_k,
            replacement_cache_v=prepared.wrong_cache_v,
            replacement_video_layers=LATE_LAYERS,
            action_visible_token_indices=tuple(range(PREFIX_TOKENS)),
            feature_bases_by_layer={
                layer: {"k": identity, "v": identity} for layer in LATE_LAYERS
            },
            projection_rank=EXPECTED_FEATURE_DIM,
            num_layers=model.mot.num_layers,
        ),
    }
