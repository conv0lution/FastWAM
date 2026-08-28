import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from experiments.libero.worker_pool import pop_task, write_worker_status
from experiments.libero.prompt_context_cache import (
    get_cached_prompt_context as _get_cached_prompt_context,
    load_prompt_context_cache as _load_prompt_context_cache,
    prewarm_prompt_contexts_and_release_text_encoder,
)
from experiments.asre_diagnosis.common import (
    G0_PROTOCOL,
    ROUND2_PROTOCOL,
    ROUND3A_PROTOCOL,
    ROUND3B_PROTOCOL,
    atomic_write_json,
    build_run_metadata,
    git_commit,
    get_num_model_layers,
    now_iso,
    resolve_condition,
    sha256_file,
    sha256_json,
)
from experiments.asre_diagnosis.round3b.donor import OnlineDonorBundle, tensor_sha256
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark, get_libero_path
from experiments.libero.action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_required_artifact_path(
    value: Any, *, label: str, protocol_label: str = "ASRE Round 3B"
) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{protocol_label} requires {label}.")
    path = Path(os.path.expanduser(os.path.expandvars(str(value)))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{protocol_label} {label} is unavailable: {path}")
    return path


def _load_donor_bundle(
    cfg: DictConfig,
    *,
    task_ids: list[int],
) -> Optional[OnlineDonorBundle]:
    """Load and verify the frozen donor bundle for a replacement protocol."""

    diagnosis_cfg = cfg.get("ASRE_DIAGNOSIS", {})
    if not bool(diagnosis_cfg.get("enabled", False)):
        return None
    protocol = str(diagnosis_cfg.get("protocol", ""))
    donor_keys = (
        "donor_mapping_path",
        "donor_mapping_sha256",
        "donor_observation_manifest_path",
        "donor_observation_manifest_sha256",
        "donor_observation_root",
    )
    configured = {
        key: diagnosis_cfg.get(key)
        for key in donor_keys
        if diagnosis_cfg.get(key) is not None
        and str(diagnosis_cfg.get(key)).strip() != ""
    }
    if protocol not in {ROUND3B_PROTOCOL, G0_PROTOCOL}:
        if configured:
            raise ValueError(
                "Donor artifacts are only valid for replacement-capable ASRE protocols; "
                f"received protocol={protocol!r}, fields={sorted(configured)}."
            )
        return None
    missing = [key for key in donor_keys if key not in configured]
    if missing:
        raise ValueError(
            "Every replacement-capable condition must record and validate the same frozen donor "
            f"bundle; missing ASRE_DIAGNOSIS fields: {missing}."
        )
    mapping_path = _resolve_required_artifact_path(
        configured["donor_mapping_path"], label="donor_mapping_path"
    )
    manifest_path = _resolve_required_artifact_path(
        configured["donor_observation_manifest_path"],
        label="donor_observation_manifest_path",
    )
    observation_root = Path(
        os.path.expanduser(os.path.expandvars(str(configured["donor_observation_root"])))
    ).resolve()
    if not observation_root.is_dir():
        raise FileNotFoundError(
            f"ASRE donor_observation_root is unavailable: {observation_root}"
        )
    mapping_digest = str(configured["donor_mapping_sha256"])
    manifest_digest = str(configured["donor_observation_manifest_sha256"])
    for label, digest in (
        ("donor_mapping_sha256", mapping_digest),
        ("donor_observation_manifest_sha256", manifest_digest),
    ):
        if len(digest) != 64:
            raise ValueError(f"ASRE donor {label} must be a SHA256 digest, got {digest!r}.")
    bundle = OnlineDonorBundle.load(
        mapping_path=mapping_path,
        observation_manifest_path=manifest_path,
        observation_root=observation_root,
        expected_mapping_sha256=mapping_digest,
        expected_observation_manifest_sha256=manifest_digest,
    )
    expected_suite = str(cfg.EVALUATION.task_suite_name)
    if str(bundle.mapping_payload.get("task_suite")) != expected_suite:
        raise ValueError(
            "Donor bundle task suite mismatch: "
            f"{bundle.mapping_payload.get('task_suite')!r} != {expected_suite!r}."
        )
    expected_seed = None if cfg.get("seed") is None else int(cfg.seed)
    if bundle.mapping_payload.get("seed") != expected_seed:
        raise ValueError(
            "Donor bundle seed mismatch: "
            f"{bundle.mapping_payload.get('seed')!r} != {expected_seed!r}."
        )
    num_trials = int(bundle.mapping_payload.get("num_trials", 0))
    if int(cfg.EVALUATION.num_trials) > num_trials:
        raise ValueError(
            "ASRE evaluation requests more trials than the frozen donor mapping: "
            f"{cfg.EVALUATION.num_trials} > {num_trials}."
        )
    missing_task_ids = [
        task_id
        for task_id in task_ids
        if any((task_id, trial) not in bundle.mappings for trial in range(int(cfg.EVALUATION.num_trials)))
    ]
    if missing_task_ids:
        raise ValueError(f"Frozen donor mapping does not cover task IDs: {missing_task_ids}.")
    return bundle


def _load_round3b_donor_bundle(
    cfg: DictConfig, *, task_ids: list[int]
) -> Optional[OnlineDonorBundle]:
    """Backward-compatible alias for downstream Round-3B callers."""
    return _load_donor_bundle(cfg, task_ids=task_ids)


def _resolve_round3b_run_provenance(cfg: DictConfig) -> dict[str, Any]:
    diagnosis_cfg = cfg.get("ASRE_DIAGNOSIS", {})
    if str(diagnosis_cfg.get("protocol", "")) != ROUND3B_PROTOCOL:
        return {}
    required = (
        "preflight_report_path",
        "preflight_report_sha256",
        "self_replacement_report_path",
        "self_replacement_report_sha256",
        "round3a_parent_tag",
        "round3a_parent_commit",
        "round3a_run_commit",
    )
    missing = [
        key
        for key in required
        if diagnosis_cfg.get(key) is None or str(diagnosis_cfg.get(key)).strip() == ""
    ]
    if missing:
        raise ValueError(f"Round-3B run provenance is incomplete; missing fields: {missing}.")
    preflight_path = _resolve_required_artifact_path(
        diagnosis_cfg.get("preflight_report_path"), label="preflight_report_path"
    )
    expected_preflight_digest = str(diagnosis_cfg.get("preflight_report_sha256"))
    observed_preflight_digest = sha256_file(preflight_path)
    if observed_preflight_digest != expected_preflight_digest:
        raise ValueError(
            "Round-3B preflight report SHA256 mismatch: "
            f"observed={observed_preflight_digest}, "
            f"expected={expected_preflight_digest}."
        )
    self_replacement_path = _resolve_required_artifact_path(
        diagnosis_cfg.get("self_replacement_report_path"),
        label="self_replacement_report_path",
    )
    expected_self_replacement_digest = str(
        diagnosis_cfg.get("self_replacement_report_sha256")
    )
    observed_self_replacement_digest = sha256_file(self_replacement_path)
    if observed_self_replacement_digest != expected_self_replacement_digest:
        raise ValueError(
            "Round-3B self-replacement report SHA256 mismatch: "
            f"observed={observed_self_replacement_digest}, "
            f"expected={expected_self_replacement_digest}."
        )
    for key in ("round3a_parent_commit", "round3a_run_commit"):
        digest = str(diagnosis_cfg.get(key))
        if len(digest) != 40 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Round-3B {key} is not a full lowercase Git SHA: {digest!r}.")
    payload = {
        "preflight_report_path": str(preflight_path),
        "preflight_report_sha256": observed_preflight_digest,
        "self_replacement_report_path": str(self_replacement_path),
        "self_replacement_report_sha256": observed_self_replacement_digest,
        "round3a_parent_tag": str(diagnosis_cfg.get("round3a_parent_tag")),
        "round3a_parent_commit": str(diagnosis_cfg.get("round3a_parent_commit")),
        "round3a_run_commit": str(diagnosis_cfg.get("round3a_run_commit")),
    }
    for key, value in payload.items():
        cfg.ASRE_DIAGNOSIS[key] = value
    return payload


def _resolve_g0_run_provenance(cfg: DictConfig) -> dict[str, Any]:
    """Validate reports and frozen Stage-1 parents attached to a G0 run."""
    diagnosis_cfg = cfg.get("ASRE_DIAGNOSIS", {})
    if str(diagnosis_cfg.get("protocol", "")) != G0_PROTOCOL:
        return {}
    required = (
        "preflight_report_path",
        "preflight_report_sha256",
        "machinery_report_path",
        "machinery_report_sha256",
        "round3a_parent_tag",
        "round3a_parent_commit",
        "round3b_parent_tag",
        "round3b_parent_commit",
    )
    missing = [
        key
        for key in required
        if diagnosis_cfg.get(key) is None or str(diagnosis_cfg.get(key)).strip() == ""
    ]
    if missing:
        raise ValueError(f"G0 run provenance is incomplete; missing fields: {missing}.")

    payload: dict[str, Any] = {}
    for stem in ("preflight_report", "machinery_report"):
        path_key = f"{stem}_path"
        digest_key = f"{stem}_sha256"
        path = _resolve_required_artifact_path(
            diagnosis_cfg.get(path_key), label=path_key, protocol_label="ASRE G0"
        )
        observed = sha256_file(path)
        expected = str(diagnosis_cfg.get(digest_key))
        if observed != expected:
            raise ValueError(
                f"G0 {stem} SHA256 mismatch: observed={observed}, expected={expected}."
            )
        payload[path_key] = str(path)
        payload[digest_key] = observed

    for key in ("round3a_parent_commit", "round3b_parent_commit"):
        digest = str(diagnosis_cfg.get(key))
        if len(digest) != 40 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"G0 {key} is not a full lowercase Git SHA: {digest!r}.")
        payload[key] = digest
    for key in ("round3a_parent_tag", "round3b_parent_tag"):
        payload[key] = str(diagnosis_cfg.get(key))
    for key, value in payload.items():
        cfg.ASRE_DIAGNOSIS[key] = value
    return payload


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _place_text_encoder(model: torch.nn.Module, device: Optional[str]) -> None:
    if device is None:
        return
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is None:
        raise ValueError(
            "EVALUATION.text_encoder_device was set, but the model has no text encoder."
        )

    target = torch.device(str(device))
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Cannot place the text encoder on {target}: CUDA is unavailable."
            )
        device_index = 0 if target.index is None else target.index
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Cannot place the text encoder on {target}: only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )

    text_encoder.to(target).eval()
    if hasattr(model, "text_encoder_device"):
        model.text_encoder_device = target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Placed text encoder on %s", target)
    for device_index in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            logging.info(
                "CUDA memory cuda:%d after model placement: %.2f GiB free / %.2f GiB total; "
                "process allocated %.2f GiB, reserved %.2f GiB",
                device_index,
                free_bytes / 2**30,
                total_bytes / 2**30,
                torch.cuda.memory_allocated(device_index) / 2**30,
                torch.cuda.memory_reserved(device_index) / 2**30,
            )
        except RuntimeError as exc:
            logging.warning("Could not query CUDA memory for cuda:%d: %s", device_index, exc)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _prepare_action_inference(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[dict[str, Any], dict]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    context, context_mask = _get_cached_prompt_context(model, prompt)
    infer_kwargs = {
        "prompt": None,
        "context": context,
        "context_mask": context_mask,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    compile_action_infer = bool(cfg.EVALUATION.get("compile_action_infer", False))
    infer_method = model.infer_joint if visualize_future_video else model.infer_action
    if not visualize_future_video and "action_infer_mode" in inspect.signature(infer_method).parameters:
        infer_kwargs["action_infer_mode"] = str(
            cfg.EVALUATION.get("action_infer_mode", "idm")
        )
    if compile_action_infer and (
        "compile_action_infer" not in inspect.signature(infer_method).parameters
    ):
        raise ValueError(
            f"{type(model).__name__}.{infer_method.__name__} does not support `compile_action_infer`."
        )
    return infer_kwargs, imgs


def _run_prepared_action_inference(
    model: torch.nn.Module,
    cfg: DictConfig,
    infer_kwargs: dict[str, Any],
    *,
    replacement_input_image: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[list[Image.Image]]]:
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    diagnosis_cfg = cfg.get("ASRE_DIAGNOSIS", {})
    diagnosis_enabled = bool(diagnosis_cfg.get("enabled", False))
    if diagnosis_enabled and visualize_future_video:
        raise ValueError(
            "ASRE_DIAGNOSIS targets infer_action and requires "
            "EVALUATION.visualize_future_video=false."
        )

    compile_action_infer = bool(cfg.EVALUATION.get("compile_action_infer", False))
    infer_method = model.infer_joint if visualize_future_video else model.infer_action
    call_kwargs = dict(infer_kwargs)
    if diagnosis_enabled:
        infer_parameters = inspect.signature(infer_method).parameters
        if "disabled_video_layers" not in infer_parameters:
            raise ValueError(
                f"{type(model).__name__}.{infer_method.__name__} does not support "
                "the video-K/V diagnosis intervention."
            )
        call_kwargs["disabled_video_layers"] = tuple(
            int(layer) for layer in diagnosis_cfg.get("disabled_video_layers", ())
        )
        replacement_layers = tuple(
            int(layer) for layer in diagnosis_cfg.get("replacement_video_layers", ())
        )
        if replacement_layers:
            if replacement_input_image is None:
                raise ValueError(
                    "replacement_video_layers are configured but no frozen donor image "
                    "was supplied for this recipient episode."
                )
            missing_parameters = [
                name
                for name in ("replacement_input_image", "replacement_video_layers")
                if name not in infer_parameters
            ]
            if missing_parameters:
                raise ValueError(
                    f"{type(model).__name__}.{infer_method.__name__} does not support "
                    f"Round-3B cache replacement arguments: {missing_parameters}."
                )
            if not bool(
                torch.isfinite(
                    replacement_input_image.detach().to(device="cpu", dtype=torch.float32)
                ).all()
            ):
                raise ValueError("Round-3B replacement input image contains NaN or Inf.")
            call_kwargs["replacement_input_image"] = replacement_input_image
            call_kwargs["replacement_video_layers"] = replacement_layers
        elif replacement_input_image is not None:
            raise ValueError(
                "A replacement image was supplied while replacement_video_layers is empty."
            )
    elif replacement_input_image is not None:
        raise ValueError("A replacement image was supplied while ASRE diagnosis is disabled.")

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(
                **call_kwargs,
                compile_action_infer=compile_action_infer,
            )
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(
                **call_kwargs,
                compile_action_infer=compile_action_infer,
            )
            predicted_future_frames = None
    return pred["action"], predicted_future_frames


def _postprocess_action(
    raw_action: torch.Tensor,
    processor: FastWAMProcessor,
    cfg: DictConfig,
) -> np.ndarray:
    action = _denormalize_action(raw_action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets.
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    replacement_input_image: Optional[torch.Tensor] = None,
    expected_current_image_sha256: Optional[str] = None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]], Optional[dict[str, Any]]]:
    infer_kwargs, imgs = _prepare_action_inference(
        obs=obs,
        task_description=task_description,
        model=model,
        processor=processor,
        cfg=cfg,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    observed_current_image_sha256 = None
    if expected_current_image_sha256 is not None:
        observed_current_image_sha256 = tensor_sha256(infer_kwargs["input_image"])
        if observed_current_image_sha256 != expected_current_image_sha256:
            raise ValueError(
                "Recipient first-query model-ready image differs from the frozen "
                "donor-mapping identity: "
                f"observed={observed_current_image_sha256}, "
                f"expected={expected_current_image_sha256}."
            )
        if replacement_input_image is None:
            raise ValueError(
                "A recipient first-query image identity was requested without a donor image."
            )
        donor_image_sha256 = tensor_sha256(replacement_input_image)
        if donor_image_sha256 == observed_current_image_sha256:
            raise ValueError(
                "Recipient and donor first-query model-ready images are unexpectedly identical."
            )
    raw_action, predicted_future_frames = _run_prepared_action_inference(
        model=model,
        cfg=cfg,
        infer_kwargs=infer_kwargs,
        replacement_input_image=replacement_input_image,
    )
    action = _postprocess_action(raw_action, processor, cfg)
    diagnosis_cfg = cfg.get("ASRE_DIAGNOSIS", {})
    if bool(diagnosis_cfg.get("enabled", False)) and bool(
        diagnosis_cfg.get("save_action_trace", True)
    ):
        trace = {
            "action_inference_seed": infer_kwargs["seed"],
            "replacement_video_layers": [
                int(layer)
                for layer in diagnosis_cfg.get("replacement_video_layers", ())
            ],
            "current_input_image_sha256": observed_current_image_sha256,
            "raw_action": raw_action.detach().to(device="cpu", dtype=torch.float32).numpy(),
            "executed_action": action.copy(),
        }
    else:
        trace = None
    return action, imgs, predicted_future_frames, trace


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    replacement_input_image: Optional[torch.Tensor] = None,
    donor_provenance: Optional[dict[str, Any]] = None,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], list[dict[str, Any]]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    diagnosis_cfg = cfg.get("ASRE_DIAGNOSIS", {})
    diagnosis_enabled = bool(diagnosis_cfg.get("enabled", False))
    record_rollout_video = not diagnosis_enabled or bool(
        diagnosis_cfg.get("save_rollout_video", False)
    )
    save_action_trace = diagnosis_enabled and bool(
        diagnosis_cfg.get("save_action_trace", True)
    )
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    policy_replan_idx = -1
    action_trace: list[dict[str, Any]] = []

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames, trace = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                replacement_input_image=replacement_input_image,
                expected_current_image_sha256=(
                    str(donor_provenance["recipient_image_sha256"])
                    if donor_provenance is not None and policy_replan_idx == -1
                    else None
                ),
            )
            policy_replan_idx += 1
            if save_action_trace:
                assert trace is not None
                trace.update(
                    {
                        "task_suite": str(cfg.EVALUATION.task_suite_name),
                        "task_id": int(cfg.EVALUATION.task_id),
                        "diagnosis_condition": str(
                            diagnosis_cfg.get("condition_name", "baseline")
                        ),
                        "episode_id": int(episode_idx),
                        "replan_id": int(policy_replan_idx),
                        "environment_step": int(t),
                    }
                )
                if donor_provenance is not None:
                    trace.update(
                        {
                            "donor_trial": int(donor_provenance["donor_trial"]),
                            "donor_task_id": int(donor_provenance["donor_task_id"]),
                            "donor_image_sha256": str(
                                donor_provenance["donor_image_sha256"]
                            ),
                            "donor_artifact_sha256": str(
                                donor_provenance["donor_artifact_sha256"]
                            ),
                        }
                    )
                action_trace.append(trace)
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            if record_rollout_video:
                replay_images.append(imgs.copy())
        else:
            if record_rollout_video:
                imgs = get_libero_image(obs)
                replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr, action_trace


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    donor_bundle: Optional[OnlineDonorBundle] = None,
) -> dict:
    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, cfg.get("seed"))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    diagnosis_cfg = cfg.get("ASRE_DIAGNOSIS", {})
    diagnosis_enabled = bool(diagnosis_cfg.get("enabled", False))
    save_rollout = not diagnosis_enabled or bool(diagnosis_cfg.get("save_rollout_video", False))
    save_action_trace = diagnosis_enabled and bool(diagnosis_cfg.get("save_action_trace", True))
    replacement_layers = tuple(
        int(layer) for layer in diagnosis_cfg.get("replacement_video_layers", ())
    )
    if replacement_layers and donor_bundle is None:
        raise ValueError(
            "Round-3B replacement layers are active but the frozen donor bundle "
            "was not loaded."
        )
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
        "replacement_video_layers": list(replacement_layers),
        "donor_assignments": [],
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        replacement_input_image = None
        donor_provenance = None
        if replacement_layers:
            assert donor_bundle is not None
            loaded_donor = donor_bundle.load_for_recipient(
                task_id=int(cfg.EVALUATION.task_id),
                recipient_trial=trial_idx,
                recipient_initial_state=initial_states[trial_idx],
                task_description=task_description,
                device=model_device,
                dtype=model.torch_dtype,
            )
            replacement_input_image = loaded_donor.image
            donor_provenance = loaded_donor.provenance
            results["donor_assignments"].append(dict(donor_provenance))
        success, replay_images, predicted_future_video_clips, episode_mean_psnr, action_trace = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            replacement_input_image=replacement_input_image,
            donor_provenance=donor_provenance,
        )
        if donor_provenance is not None:
            results["donor_assignments"][-1][
                "recipient_first_query_image_verified"
            ] = True
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        if save_rollout:
            save_rollout_video(
                video_dir,
                replay_images,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                success=success,
                task_description=task_description,
            )
        if save_action_trace:
            trace_dir = video_dir.parent / "action_traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"task{cfg.EVALUATION.task_id}_trial{trial_idx}.jsonl"
            with trace_path.open("w", encoding="utf-8") as handle:
                for record in action_trace:
                    handle.write(json.dumps(record, cls=NumpyEncoder) + "\n")
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    close_fn = getattr(env, "close", None)
    if close_fn is not None:
        close_fn()

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


def _required_worker_path(name: str) -> Path:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise ValueError(f"{name} must be set in worker mode.")
    return Path(os.path.expanduser(os.path.expandvars(value))).resolve()


def _result_file(output_root: Path, suite_name: str, task_id: int) -> Path | None:
    matches = sorted((output_root / suite_name).glob(f"gpu*_task{task_id}_results.json"))
    return matches[0] if matches else None


def _run_task_to_file(
    *,
    cfg: DictConfig,
    suite_name: str,
    task_id: int,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    output_root: Path,
    worker_id: str | None,
    donor_bundle: Optional[OnlineDonorBundle],
) -> tuple[Path, dict]:
    task_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    task_cfg.EVALUATION.task_suite_name = suite_name
    task_cfg.EVALUATION.task_id = int(task_id)

    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    initial_states = torch.load(init_states_path, weights_only=False)
    while len(initial_states) < int(task_cfg.EVALUATION.num_trials):
        initial_states.extend(
            initial_states[: int(task_cfg.EVALUATION.num_trials) - len(initial_states)]
        )

    video_dir = output_root / suite_name / "videos"
    diagnosis_cfg = task_cfg.get("ASRE_DIAGNOSIS", {})
    if not bool(diagnosis_cfg.get("enabled", False)) or bool(
        diagnosis_cfg.get("save_rollout_video", False)
    ):
        video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = output_root / suite_name / "predicted_videos"
    if bool(task_cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = {
        "task_suite": suite_name,
        "task_id": task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(task_cfg.EVALUATION.num_trials),
        "gpu_id": int(task_cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }
    if bool(diagnosis_cfg.get("enabled", False)):
        results.update(
            {
                "diagnosis_condition": str(diagnosis_cfg.get("condition_name", "baseline")),
                "condition_protocol": str(diagnosis_cfg.get("protocol", "round1_drop_groups")),
                "enabled_video_retrieval_layers": [
                    int(layer)
                    for layer in diagnosis_cfg.get("enabled_video_retrieval_layers", ())
                ],
                "disabled_video_layers": [
                    int(layer) for layer in diagnosis_cfg.get("disabled_video_layers", ())
                ],
                "replacement_video_layers": [
                    int(layer)
                    for layer in diagnosis_cfg.get("replacement_video_layers", ())
                ],
            }
        )
        if donor_bundle is not None:
            results.update(
                {
                    "donor_mapping_path": str(donor_bundle.mapping_path),
                    "donor_mapping_sha256": donor_bundle.mapping_sha256,
                    "donor_observation_manifest_path": str(
                        donor_bundle.observation_manifest_path
                    ),
                    "donor_observation_manifest_sha256": (
                        donor_bundle.observation_manifest_sha256
                    ),
                    "donor_observation_root": str(donor_bundle.observation_root),
                }
            )
        if str(diagnosis_cfg.get("protocol", "")) == ROUND3B_PROTOCOL:
            results.update(
                {
                    key: diagnosis_cfg.get(key)
                    for key in (
                        "preflight_report_path",
                        "preflight_report_sha256",
                        "self_replacement_report_path",
                        "self_replacement_report_sha256",
                        "round3a_parent_tag",
                        "round3a_parent_commit",
                        "round3a_run_commit",
                    )
                }
            )
        if str(diagnosis_cfg.get("protocol", "")) == G0_PROTOCOL:
            results.update(
                {
                    key: diagnosis_cfg.get(key)
                    for key in (
                        "preflight_report_path",
                        "preflight_report_sha256",
                        "machinery_report_path",
                        "machinery_report_sha256",
                        "round3a_parent_tag",
                        "round3a_parent_commit",
                        "round3b_parent_tag",
                        "round3b_parent_commit",
                    )
                }
            )
    if worker_id is not None:
        results["worker_id"] = worker_id
    results.update(
        run_single_task(
            task=task,
            initial_states=initial_states,
            model=model,
            processor=processor,
            cfg=task_cfg,
            video_dir=video_dir,
            predicted_video_dir=predicted_video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            donor_bundle=donor_bundle,
        )
    )
    results["duration"] = time.time() - start_time

    result_dir = output_root / suite_name
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f"gpu{task_cfg.gpu_id}_task{task_id}_results.json"
    temp_output_file = result_dir / f".{output_file.name}.{os.getpid()}.tmp"
    temp_output_file.write_text(
        json.dumps(results, indent=4, cls=NumpyEncoder),
        encoding="utf-8",
    )
    os.replace(temp_output_file, output_file)
    return output_file, results


def _run_worker_loop(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    output_root: Path,
    donor_bundle: Optional[OnlineDonorBundle],
) -> None:
    pending_file = _required_worker_path("LIBERO_WORKER_PENDING_FILE")
    lock_file = _required_worker_path("LIBERO_WORKER_LOCK_FILE")
    status_dir = _required_worker_path("LIBERO_WORKER_STATUS_DIR")
    failed_file = _required_worker_path("LIBERO_WORKER_FAILED_FILE")
    stop_file = _required_worker_path("LIBERO_WORKER_STOP_FILE")
    worker_id = os.environ.get("LIBERO_WORKER_ID", str(cfg.gpu_id))

    write_worker_status(status_dir, worker_id, "idle", "model loaded")
    completed = 0
    skipped = 0
    while not stop_file.exists():
        task = pop_task(pending_file, lock_file, status_dir, worker_id)
        if task is None:
            time.sleep(0.2)
            continue

        suite_name, task_id = task
        if _result_file(output_root, suite_name, task_id) is not None:
            skipped += 1
            continue

        try:
            output_file, _ = _run_task_to_file(
                cfg=cfg,
                suite_name=suite_name,
                task_id=task_id,
                model=model,
                processor=processor,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                output_root=output_root,
                worker_id=worker_id,
                donor_bundle=donor_bundle,
            )
            completed += 1
            print(f"worker {worker_id} completed {suite_name},{task_id}: {output_file}")
        except Exception as exc:
            with failed_file.open("a", encoding="utf-8") as f:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')},{suite_name},{task_id},"
                    f"gpu={cfg.gpu_id},error={exc!r}\n"
                )
            write_worker_status(
                status_dir,
                worker_id,
                "failed",
                f"{suite_name},{task_id}: {exc!r}",
            )
            raise

    write_worker_status(
        status_dir,
        worker_id,
        "done",
        f"completed={completed} skipped={skipped}",
    )
    print(f"worker {worker_id} done: completed={completed} skipped={skipped}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    run_start_timestamp = now_iso()
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager.py for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    prompt_context_cache_value = cfg.EVALUATION.get("prompt_context_cache_path")
    prompt_context_cache_path = None
    prompt_context_cache_file_sha256 = None
    if prompt_context_cache_value is not None:
        prompt_context_cache_path = Path(
            os.path.expanduser(os.path.expandvars(str(prompt_context_cache_value)))
        ).resolve()
        if not prompt_context_cache_path.is_file():
            raise FileNotFoundError(
                f"Prompt-context cache is unavailable: {prompt_context_cache_path}"
            )
        cfg.model.load_text_encoder = False
        cfg.EVALUATION.text_encoder_device = None
        prompt_context_cache_file_sha256 = sha256_file(prompt_context_cache_path)
        expected_prompt_cache_sha256 = cfg.ASRE_DIAGNOSIS.get(
            "prompt_context_cache_sha256"
        )
        if (
            expected_prompt_cache_sha256 is not None
            and str(expected_prompt_cache_sha256) != prompt_context_cache_file_sha256
        ):
            raise ValueError(
                "Prompt-context cache SHA256 mismatch before model load: "
                f"expected={expected_prompt_cache_sha256}, "
                f"observed={prompt_context_cache_file_sha256}, "
                f"path={prompt_context_cache_path}."
            )
    model = instantiate(
        cfg.model,
        model_dtype=model_dtype,
        device=model_device,
        text_encoder_device=cfg.EVALUATION.get("text_encoder_device"),
    )
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()
    _place_text_encoder(model, cfg.EVALUATION.get("text_encoder_device"))
    loaded_prompt_context_count = None
    prompt_context_cache_metadata: dict[str, Any] = {}
    if prompt_context_cache_path is not None:
        loaded_prompt_context_count = _load_prompt_context_cache(
            model, prompt_context_cache_path
        )
        prompt_context_cache_metadata = dict(
            getattr(model, "_eval_prompt_context_cache_metadata", {})
        )
        # Detect replacement between the pre-load hash check and deserialization.
        post_load_sha256 = sha256_file(prompt_context_cache_path)
        if post_load_sha256 != prompt_context_cache_file_sha256:
            raise ValueError(
                "Prompt-context cache changed while it was being loaded: "
                f"{prompt_context_cache_path}."
            )

    num_model_layers = get_num_model_layers(model)
    diagnosis_cfg = cfg.get("ASRE_DIAGNOSIS", {})
    condition = resolve_condition(diagnosis_cfg, num_model_layers)
    cfg.ASRE_DIAGNOSIS.condition_name = condition.name
    cfg.ASRE_DIAGNOSIS.enabled_video_retrieval_layers = list(
        condition.enabled_video_retrieval_layers(num_model_layers)
    )
    cfg.ASRE_DIAGNOSIS.disabled_video_layers = list(condition.disabled_video_layers)
    cfg.ASRE_DIAGNOSIS.replacement_video_layers = list(
        condition.replacement_video_layers
    )

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    output_root = Path(
        os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir)))
    ).resolve()

    configured_task_ids = cfg.EVALUATION.get("task_ids", None)
    if configured_task_ids is None:
        task_ids = [int(cfg.EVALUATION.task_id)]
    else:
        task_ids = [int(task_id) for task_id in configured_task_ids]
        if not task_ids:
            raise ValueError("EVALUATION.task_ids must not be empty when provided.")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError(f"EVALUATION.task_ids contains duplicates: {task_ids}")

    prewarm_suite_prompts = bool(
        cfg.EVALUATION.get(
            "prewarm_suite_prompts_and_release_text_encoder", False
        )
    )
    prompt_context_manifest_sha256 = None
    prompt_context_count = loaded_prompt_context_count
    if prompt_context_cache_path is not None:
        prompt_context_manifest_sha256 = prompt_context_cache_metadata.get(
            "prompt_context_manifest_sha256"
        )
        if str(diagnosis_cfg.get("protocol", "")) == G0_PROTOCOL:
            expected_g0_cache = {
                "artifact_type": "asre_g0_suite_prompt_context_cache",
                "schema_version": 1,
                "protocol": G0_PROTOCOL,
                "strategy": "suite_cuda_prompt_context_cache",
                "text_conditioning_source": "g0_suite_prompt_context_cache",
                "git_commit_hash": git_commit(project_root),
                "task_suite": str(cfg.EVALUATION.task_suite_name),
                "checkpoint_path": str(Path(str(cfg.ckpt)).expanduser().resolve()),
                "checkpoint_sha256": str(
                    cfg.ASRE_DIAGNOSIS.get("checkpoint_sha256")
                ),
                "prompt_template": DEFAULT_PROMPT,
                "prompt_count": 10,
                "encoding_device_type": "cuda",
                "model_device": "cuda:0",
                "text_encoder_device": "cuda:1",
            }
            g0_cache_mismatches = {
                key: {
                    "observed": prompt_context_cache_metadata.get(key),
                    "expected": value,
                }
                for key, value in expected_g0_cache.items()
                if prompt_context_cache_metadata.get(key) != value
            }
            if g0_cache_mismatches:
                raise ValueError(
                    "External G0 prompt-context cache is incompatible with this run: "
                    f"{g0_cache_mismatches}."
                )
            if (
                not isinstance(prompt_context_manifest_sha256, str)
                or len(prompt_context_manifest_sha256) != 64
            ):
                raise ValueError("External G0 prompt cache has no valid semantic digest.")
    if prewarm_suite_prompts:
        if prompt_context_cache_path is not None:
            raise ValueError(
                "Suite prompt prewarm cannot be combined with an external prompt cache."
            )
        suite_name = str(cfg.EVALUATION.task_suite_name)
        suite_object = benchmark.get_benchmark_dict()[suite_name]()
        suite_prompts = [
            DEFAULT_PROMPT.format(task=str(suite_object.get_task(task_id).language))
            for task_id in range(int(suite_object.n_tasks))
        ]
        prewarm_prompt_contexts_and_release_text_encoder(model, suite_prompts)
        prompt_cache = getattr(model, "_eval_prompt_context_cache")
        prompt_context_manifest_sha256 = sha256_json(
            {
                "strategy": "suite_gpu_prewarm_then_text_encoder_release",
                "task_suite": suite_name,
                "prompt_template": DEFAULT_PROMPT,
                "records": [
                    {
                        "task_id": task_id,
                        "prompt": prompt,
                        "context_sha256": tensor_sha256(prompt_cache[prompt][0]),
                        "context_mask_sha256": tensor_sha256(prompt_cache[prompt][1]),
                    }
                    for task_id, prompt in enumerate(suite_prompts)
                ],
            }
        )
        prompt_context_count = len(suite_prompts)

    donor_bundle = _load_donor_bundle(cfg, task_ids=task_ids)
    run_provenance = {
        **_resolve_round3b_run_provenance(cfg),
        **_resolve_g0_run_provenance(cfg),
    }

    diagnosis_enabled = bool(diagnosis_cfg.get("enabled", False))
    run_metadata = None
    metadata_path = output_root / "run_metadata.json"
    if diagnosis_enabled:
        num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
        num_inference_steps = (
            int(cfg.get("eval_num_inference_steps", 20))
            if num_inference_steps_cfg is None
            else int(num_inference_steps_cfg)
        )
        run_metadata = build_run_metadata(
            repo_root=project_root,
            checkpoint=str(cfg.ckpt),
            dataset_stats_path=str(dataset_stats_path),
            condition=condition,
            num_layers=num_model_layers,
            task_suite=str(cfg.EVALUATION.task_suite_name),
            task_ids=task_ids,
            seed=None if cfg.get("seed") is None else int(cfg.seed),
            num_trials=int(cfg.EVALUATION.num_trials),
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            replan_steps=int(cfg.EVALUATION.get("replan_steps", 5)),
            start_timestamp=run_start_timestamp,
            condition_protocol=str(
                cfg.ASRE_DIAGNOSIS.get("protocol", "round1_drop_groups")
            ),
            checkpoint_sha256=cfg.ASRE_DIAGNOSIS.get("checkpoint_sha256"),
            dataset_stats_sha256=cfg.ASRE_DIAGNOSIS.get("dataset_stats_sha256"),
            state_bank_manifest_path=cfg.ASRE_DIAGNOSIS.get("state_bank_manifest_path"),
            state_bank_manifest_sha256=cfg.ASRE_DIAGNOSIS.get(
                "state_bank_manifest_sha256"
            ),
            valid_state_bank_manifest_path=cfg.ASRE_DIAGNOSIS.get(
                "valid_state_bank_manifest_path"
            ),
            valid_state_bank_manifest_sha256=cfg.ASRE_DIAGNOSIS.get(
                "valid_state_bank_manifest_sha256"
            ),
            prompt_context_cache_path=(
                None if prompt_context_cache_path is None else str(prompt_context_cache_path)
            ),
            prompt_context_cache_sha256=cfg.ASRE_DIAGNOSIS.get(
                "prompt_context_cache_sha256"
            ),
            config_sha256=sha256_json(OmegaConf.to_container(cfg, resolve=True)),
        )
        run_metadata.update(
            {
                "status": "running",
                "condition_config": {
                    "mode": str(cfg.ASRE_DIAGNOSIS.get("mode", "drop_video_kv")),
                    "protocol": str(
                        cfg.ASRE_DIAGNOSIS.get("protocol", "round1_drop_groups")
                    ),
                    "condition_name": condition.name,
                    "enabled_video_retrieval_layers": list(
                        condition.enabled_video_retrieval_layers(num_model_layers)
                    ),
                    "disabled_video_layers": list(condition.disabled_video_layers),
                    "replacement_video_layers": list(
                        condition.replacement_video_layers
                    ),
                },
                "text_conditioning_source": (
                    str(
                        prompt_context_cache_metadata.get(
                            "text_conditioning_source",
                            "round1_state_bank_prompt_context_cache",
                        )
                    )
                    if prompt_context_cache_path is not None
                    else "model_text_encoder"
                ),
                "prompt_context_strategy": (
                    "suite_gpu_prewarm_then_text_encoder_release"
                    if prewarm_suite_prompts
                    else (
                        str(
                            prompt_context_cache_metadata.get(
                                "strategy", "external_prompt_context_cache"
                            )
                        )
                        if prompt_context_cache_path is not None
                        else "model_text_encoder_on_demand"
                    )
                ),
                "prompt_context_count": prompt_context_count,
                "prompt_context_manifest_sha256": prompt_context_manifest_sha256,
                "prompt_template": DEFAULT_PROMPT,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                "environment_seed": None if cfg.get("seed") is None else int(cfg.seed),
                "action_inference_seed": None
                if cfg.get("seed") is None
                else int(cfg.seed),
                "action_noise_seed": None if cfg.get("seed") is None else int(cfg.seed),
                "compile_action_infer": bool(cfg.EVALUATION.get("compile_action_infer", False)),
                "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
                "sigma_shift": (
                    None
                    if cfg.EVALUATION.get("sigma_shift") is None
                    else float(cfg.EVALUATION.get("sigma_shift"))
                ),
                "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
            }
        )
        if donor_bundle is not None:
            run_metadata.update(
                {
                    "donor_mapping_path": str(donor_bundle.mapping_path),
                    "donor_mapping_sha256": donor_bundle.mapping_sha256,
                    "donor_observation_manifest_path": str(
                        donor_bundle.observation_manifest_path
                    ),
                    "donor_observation_manifest_sha256": (
                        donor_bundle.observation_manifest_sha256
                    ),
                    "donor_observation_root": str(donor_bundle.observation_root),
                    "donor_mapping_rule": str(
                        donor_bundle.mapping_payload["mapping_rule"]
                    ),
                }
            )
        run_metadata.update(run_provenance)
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                existing_metadata = json.load(handle)
            resume_keys = [
                "checkpoint_path",
                "dataset_stats_path",
                "diagnosis_condition",
                "disabled_video_layers",
                "num_model_layers",
                "task_suite",
                "task_ids",
                "seed",
                "number_of_trials",
                "action_horizon",
                "number_of_inference_steps",
                "replan_steps",
                "compile_action_infer",
                "binarize_gripper",
                "sigma_shift",
                "rand_device",
            ]
            if str(cfg.ASRE_DIAGNOSIS.get("protocol")) in {
                ROUND2_PROTOCOL,
                ROUND3A_PROTOCOL,
                ROUND3B_PROTOCOL,
                G0_PROTOCOL,
            }:
                resume_keys.extend(
                    [
                        "condition_protocol",
                        "enabled_video_retrieval_layers",
                        "git_commit_hash",
                        "checkpoint_sha256",
                        "dataset_stats_sha256",
                        "state_bank_manifest_sha256",
                        "valid_state_bank_manifest_sha256",
                        "prompt_context_cache_sha256",
                        "config_sha256",
                        "text_conditioning_source",
                        "prompt_template",
                        "torch_version",
                        "cuda_version",
                    ]
                )
            if str(cfg.ASRE_DIAGNOSIS.get("protocol")) in {
                ROUND3B_PROTOCOL,
                G0_PROTOCOL,
            }:
                resume_keys.extend(
                    [
                        "replacement_video_layers",
                        "donor_mapping_path",
                        "donor_mapping_sha256",
                        "donor_observation_manifest_path",
                        "donor_observation_manifest_sha256",
                        "donor_observation_root",
                        "donor_mapping_rule",
                    ]
                )
            if str(cfg.ASRE_DIAGNOSIS.get("protocol")) == ROUND3B_PROTOCOL:
                resume_keys.extend(
                    [
                        "preflight_report_path",
                        "preflight_report_sha256",
                        "self_replacement_report_path",
                        "self_replacement_report_sha256",
                        "round3a_parent_tag",
                        "round3a_parent_commit",
                        "round3a_run_commit",
                    ]
                )
            if str(cfg.ASRE_DIAGNOSIS.get("protocol")) == G0_PROTOCOL:
                resume_keys.extend(
                    [
                        "preflight_report_path",
                        "preflight_report_sha256",
                        "machinery_report_path",
                        "machinery_report_sha256",
                        "round3a_parent_tag",
                        "round3a_parent_commit",
                        "round3b_parent_tag",
                        "round3b_parent_commit",
                    ]
                )
            mismatches = {
                key: {"existing": existing_metadata.get(key), "requested": run_metadata.get(key)}
                for key in resume_keys
                if existing_metadata.get(key) != run_metadata.get(key)
            }
            if mismatches:
                raise ValueError(
                    "Refusing to resume a diagnosis output directory with incompatible settings: "
                    f"{json.dumps(mismatches, sort_keys=True)}"
                )
            run_metadata["start_timestamp"] = existing_metadata.get(
                "start_timestamp", run_metadata["start_timestamp"]
            )
        atomic_write_json(metadata_path, run_metadata)
        print("ASRE diagnosis run metadata:")
        print(json.dumps(run_metadata, indent=2, cls=NumpyEncoder))

    if os.environ.get("LIBERO_WORKER_MODE") == "1":
        _run_worker_loop(
            cfg=cfg,
            model=model,
            processor=processor,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            output_root=output_root,
            donor_bundle=donor_bundle,
        )
        if run_metadata is not None:
            run_metadata["end_timestamp"] = now_iso()
            run_metadata["status"] = "completed"
            atomic_write_json(metadata_path, run_metadata)
        return None

    all_results = []
    for task_id in task_ids:
        existing_result = _result_file(
            output_root,
            str(cfg.EVALUATION.task_suite_name),
            task_id,
        )
        if existing_result is not None:
            logging.info("Skipping completed task %s: %s", task_id, existing_result)
            with existing_result.open("r", encoding="utf-8") as handle:
                all_results.append(json.load(handle))
            continue

        _, results = _run_task_to_file(
            cfg=cfg,
            suite_name=str(cfg.EVALUATION.task_suite_name),
            task_id=task_id,
            model=model,
            processor=processor,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            output_root=output_root,
            worker_id=None,
            donor_bundle=donor_bundle,
        )
        all_results.append(results)
        print(
            f"Task {task_id} completed: "
            f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
        )
        if results.get("future_video_psnr_mean") is not None:
            print(
                f"Task {task_id} future-video PSNR mean: "
                f"{results['future_video_psnr_mean']:.4f}"
            )
        print(f"Time taken: {results['duration']:.2f} seconds")

    if run_metadata is not None:
        run_metadata["end_timestamp"] = now_iso()
        run_metadata["status"] = "completed"
        atomic_write_json(metadata_path, run_metadata)
    return all_results[0] if len(all_results) == 1 else all_results


if __name__ == "__main__":
    eval_single_process()
