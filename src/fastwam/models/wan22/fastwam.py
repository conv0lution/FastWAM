import math
from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT, NativePrefixKVHook
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .video_cache_replacement import (
    action_visible_video_token_indices,
    build_video_cache_stats,
    mix_replacement_video_cache,
    normalize_video_layer_indices,
    project_replacement_video_cache,
    select_replacement_video_cache,
)

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        text_encoder_device: str | None = None,
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        compile_training_denoise: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.text_encoder_device = (
            None if text_encoder_device is None else torch.device(text_encoder_device)
        )
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.compile_training_denoise = bool(compile_training_denoise)
        self.mot.compile_training_layers = self.compile_training_denoise

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        text_encoder_device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = False,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        compile_training_denoise: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            text_encoder_device=text_encoder_device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            text_encoder_device=text_encoder_device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            compile_training_denoise=compile_training_denoise,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        # Module.to recursively moves registered children. Temporarily detach
        # an explicitly sharded T5 so it never lands on the main model GPU.
        text_encoder = self.text_encoder
        self._modules.pop("text_encoder", None)
        try:
            super().to(*args, **kwargs)
        finally:
            self.text_encoder = text_encoder
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            if self.text_encoder_device is None:
                self.text_encoder.to(*args, **kwargs)
            else:
                self.text_encoder.to(self.text_encoder_device)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        try:
            text_encoder_device = next(self.text_encoder.parameters()).device
        except StopIteration:
            text_encoder_device = self.device
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(text_encoder_device)
        mask = mask.to(text_encoder_device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return (
            prompt_emb.to(device=self.device),
            mask.to(device=self.device),
        )

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if tiled:
            raise NotImplementedError("Batched VAE encoding does not support tiled encoding.")
        if not hasattr(self, "_vae_encode_compiled"):
            self._vae_encode_compiled = torch.compile(
                self.vae.model.encode,
                backend="cudagraphs",
                fullgraph=True,
            )
        return self._vae_encode_compiled(
            video_tensor.to(self.device),
            self.vae.scale,
        ).clone()

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        if tiled:
            raise NotImplementedError("Batched VAE image encoding does not support tiled encoding.")
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        return self.vae.model.encode(image.unsqueeze(0), self.vae.scale)

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)
        context = sample.get("context")
        context_mask = sample.get("context_mask")
        if context is None and context_mask is None:
            prompt = sample.get("prompt")
            if prompt is None:
                raise ValueError("FastWAM training requires `context/context_mask` or `prompt`.")
            context, context_mask = self.encode_prompt(prompt)
        elif context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must both exist when either is provided.")

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _joint_denoise_core(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        action_condition: Optional[torch.Tensor] = None,
        native_prefix_kv_hook: Optional[NativePrefixKVHook] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the tensor-only video/action core shared by training and inference."""
        (
            video_tokens,
            t_video,
            t_mod_video,
            context_video,
            context_mask_video,
            freqs_video,
            f_video,
            h_video,
            w_video,
            _tokens_per_frame,
        ) = self.video_expert.prepare(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action_condition,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        (
            action_tokens,
            _t_action,
            t_mod_action,
            context_action,
            context_mask_action,
            freqs_action,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        video_tokens, action_tokens = self.mot.forward_joint_core(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_freqs=freqs_video,
            action_freqs=freqs_action,
            video_t_mod=t_mod_video,
            action_t_mod=t_mod_action,
            video_context=context_video,
            video_context_mask=context_mask_video,
            action_context=context_action,
            action_context_mask=context_mask_action,
            attention_mask=attention_mask,
            video_prefix_seq_len=int(_tokens_per_frame),
            native_prefix_kv_hook=native_prefix_kv_hook,
        )
        return (
            self.video_expert.post(video_tokens, t_video, f_video, h_video, w_video),
            self.action_expert.post(action_tokens),
        )

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = latents.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=noisy_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents.device,
        )
        pred_video, pred_action = self._joint_denoise_core(
            latents_video=latents,
            latents_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            action_condition=action,
        )

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        latent_t, latent_h, latent_w = latents_video.shape[-3:]
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=latents_video.device,
        )
        return self._joint_denoise_core(
            latents_video=latents_video,
            latents_action=latents_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            action_condition=gt_action,
        )

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    def _denoise_action_with_video_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        disabled_video_layers: tuple[int, ...] = (),
    ) -> torch.Tensor:
        (
            action_tokens,
            _t,
            action_t_mod,
            action_context,
            action_context_mask,
            action_freqs,
        ) = self.action_expert.prepare(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache_tensor(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context=action_context,
            action_context_mask=action_context_mask,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask,
            disabled_video_layers=disabled_video_layers,
        )
        return self.action_expert.post(action_tokens)

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        disabled_video_layers: tuple[int, ...] = (),
    ) -> torch.Tensor:
        """Legacy dictionary-cache path retained for the optional IDM variant."""
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            disabled_video_layers=disabled_video_layers,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        compile_action_infer: bool = False,
        native_prefix_kv_hook: Optional[NativePrefixKVHook] = None,
        decode_video: bool = True,
        initial_video_noise: Optional[torch.Tensor] = None,
        initial_action_noise: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                compile_action_infer=compile_action_infer,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        full_video_shape = (1, self.vae.model.z_dim, latent_t, latent_h, latent_w)
        future_video_shape = (
            1,
            self.vae.model.z_dim,
            latent_t - 1,
            latent_h,
            latent_w,
        )
        if initial_video_noise is None:
            latents_video = torch.randn(
                full_video_shape,
                generator=video_generator,
                device=rand_device,
                dtype=torch.float32,
            )
        else:
            supplied_video_noise = initial_video_noise.detach().to(
                device=rand_device, dtype=torch.float32
            )
            if supplied_video_noise.ndim == 4:
                supplied_video_noise = supplied_video_noise.unsqueeze(0)
            if tuple(supplied_video_noise.shape) == future_video_shape:
                latents_video = torch.zeros(full_video_shape, dtype=torch.float32, device=rand_device)
                latents_video[:, :, 1:] = supplied_video_noise
            elif tuple(supplied_video_noise.shape) == full_video_shape:
                latents_video = supplied_video_noise.clone()
            else:
                raise ValueError(
                    "`initial_video_noise` must match the full or future-only latent "
                    f"shape; got {tuple(supplied_video_noise.shape)}, expected "
                    f"{full_video_shape} or {future_video_shape}."
                )
        latents_video = latents_video.to(device=self.device, dtype=self.torch_dtype)
        action_shape = (1, action_horizon, self.action_expert.action_dim)
        if initial_action_noise is None:
            latents_action = torch.randn(
                action_shape,
                generator=action_generator,
                device=rand_device,
                dtype=torch.float32,
            )
        else:
            supplied_action_noise = initial_action_noise.detach().to(
                device=rand_device, dtype=torch.float32
            )
            if supplied_action_noise.ndim == 2:
                supplied_action_noise = supplied_action_noise.unsqueeze(0)
            if tuple(supplied_action_noise.shape) != action_shape:
                raise ValueError(
                    "`initial_action_noise` shape mismatch: "
                    f"{tuple(supplied_action_noise.shape)} != {action_shape}."
                )
            latents_action = supplied_action_noise.clone()
        latents_action = latents_action.to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        patch_t, patch_h, patch_w = (int(size) for size in self.video_expert.patch_size)
        tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)
        joint_attention_mask = self._build_mot_attention_mask(
            video_seq_len=(latent_t // patch_t) * tokens_per_frame,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        if compile_action_infer:
            if native_prefix_kv_hook is not None:
                raise ValueError(
                    "Native prefix-K/V interventions do not support compiled inference."
                )
            if action is not None:
                raise ValueError(
                    "`compile_action_infer=True` does not support action conditioning in `infer_joint`."
                )
            if not hasattr(self, "_joint_denoise_core_compiled_inference"):
                self._joint_denoise_core_compiled_inference = torch.compile(
                    self._joint_denoise_core,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            joint_denoise_core = self._joint_denoise_core_compiled_inference
        else:
            joint_denoise_core = self._joint_denoise_core

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            if compile_action_infer:
                torch.compiler.cudagraph_mark_step_begin()
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = joint_denoise_core(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                attention_mask=joint_attention_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                action_condition=action,
                native_prefix_kv_hook=native_prefix_kv_hook,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled)
            if decode_video
            else None,
            "video_latents": latents_video.detach(),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        compile_action_infer: bool = False,
        disabled_video_layers: Optional[Sequence[int]] = None,
        replacement_input_image: Optional[torch.Tensor] = None,
        replacement_video_layers: Optional[Sequence[int]] = None,
        retained_current_video_token_indices: Optional[Sequence[int]] = None,
        retained_current_video_heads_by_layer: Optional[
            Mapping[int, Sequence[int]]
        ] = None,
        feature_projection_bases_by_layer: Optional[
            Mapping[int, Mapping[str, torch.Tensor]]
        ] = None,
        feature_projection_rank: Optional[int] = None,
        expected_video_cache_layout: Optional[Mapping[str, Any]] = None,
        return_video_cache_stats: bool = False,
        return_video_cache_deltas: bool = False,
        video_cache_delta_layers: Optional[Sequence[int]] = None,
        cache_only: bool = False,
        action_sensitive_interpolation_lambda: Optional[float] = None,
        action_sensitive_layers: Optional[Sequence[int]] = None,
        action_sensitive_gradient_checkpointing: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        disabled_video_layers_tuple = normalize_video_layer_indices(
            disabled_video_layers,
            argument_name="disabled_video_layers",
            num_layers=self.mot.num_layers,
        )
        replacement_video_layers_tuple = normalize_video_layer_indices(
            replacement_video_layers,
            argument_name="replacement_video_layers",
            num_layers=self.mot.num_layers,
        )
        replacement_enabled = bool(replacement_video_layers_tuple)
        token_hybrid_enabled = retained_current_video_token_indices is not None
        head_hybrid_enabled = retained_current_video_heads_by_layer is not None
        feature_projection_enabled = feature_projection_bases_by_layer is not None
        action_sensitive_enabled = (
            action_sensitive_interpolation_lambda is not None
            or action_sensitive_layers is not None
        )
        if (
            action_sensitive_interpolation_lambda is None
        ) != (action_sensitive_layers is None):
            raise ValueError(
                "Action-sensitive inference requires both interpolation lambda and layers."
            )
        action_sensitive_layers_tuple = normalize_video_layer_indices(
            action_sensitive_layers,
            argument_name="action_sensitive_layers",
            num_layers=self.mot.num_layers,
        )
        if sum(
            (
                token_hybrid_enabled,
                head_hybrid_enabled,
                feature_projection_enabled,
                action_sensitive_enabled,
            )
        ) > 1:
            raise ValueError(
                "Token masks, head masks, feature projection, and action-sensitive "
                "injection are mutually exclusive."
            )
        if (
            token_hybrid_enabled or head_hybrid_enabled or feature_projection_enabled
        ) and not replacement_enabled:
            raise ValueError(
                "A cache hybrid/projection requires donor input and non-empty "
                "replacement_video_layers."
            )
        if feature_projection_enabled:
            if feature_projection_rank is None:
                raise ValueError("Feature projection requires `feature_projection_rank`.")
        elif feature_projection_rank is not None:
            raise ValueError(
                "`feature_projection_rank` requires feature_projection_bases_by_layer."
            )
        if expected_video_cache_layout is not None and not (
            token_hybrid_enabled
            or head_hybrid_enabled
            or feature_projection_enabled
            or action_sensitive_enabled
            or return_video_cache_deltas
        ):
            raise ValueError(
                "`expected_video_cache_layout` requires a cache intervention/audit."
            )
        if cache_only and not return_video_cache_deltas:
            raise ValueError("`cache_only` requires `return_video_cache_deltas=true`.")
        if return_video_cache_deltas and not replacement_enabled:
            raise ValueError("Cache-delta export requires current and donor cache pairs.")
        delta_layers_tuple = normalize_video_layer_indices(
            video_cache_delta_layers,
            argument_name="video_cache_delta_layers",
            num_layers=self.mot.num_layers,
        )
        if return_video_cache_deltas and not delta_layers_tuple:
            delta_layers_tuple = replacement_video_layers_tuple
        if delta_layers_tuple and not return_video_cache_deltas:
            raise ValueError(
                "`video_cache_delta_layers` requires return_video_cache_deltas=true."
            )
        if set(delta_layers_tuple) - set(replacement_video_layers_tuple):
            raise ValueError("Cache-delta layers must be replacement-enabled layers.")
        if cache_only and compile_action_infer:
            raise ValueError("Cache-only calibration must disable compiled inference.")
        if action_sensitive_enabled:
            interpolation_lambda = float(action_sensitive_interpolation_lambda)
            if not math.isfinite(interpolation_lambda) or not 0.0 <= interpolation_lambda <= 1.0:
                raise ValueError(
                    "Action-sensitive interpolation lambda must be finite and in [0,1]."
                )
            if not replacement_enabled:
                raise ValueError(
                    "Action-sensitive inference requires current/donor cache construction."
                )
            if action_sensitive_layers_tuple != replacement_video_layers_tuple:
                raise ValueError(
                    "Action-sensitive layers must exactly equal replacement video layers."
                )
            if compile_action_infer:
                raise ValueError("Action-sensitive inference requires uncompiled execution.")
            if cache_only:
                raise ValueError("Action-sensitive inference cannot use cache-only mode.")
            trainable = [
                name
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
            ]
            if trainable:
                raise RuntimeError(
                    "Action-sensitive inference requires every model parameter to be frozen; "
                    f"trainable examples={trainable[:8]}."
                )
        elif action_sensitive_gradient_checkpointing:
            raise ValueError(
                "Action-sensitive gradient checkpointing requires action-sensitive inference."
            )
        if replacement_enabled != (replacement_input_image is not None):
            raise ValueError(
                "`replacement_input_image` and a non-empty `replacement_video_layers` "
                "must be provided together."
            )
        overlap = sorted(
            set(disabled_video_layers_tuple).intersection(replacement_video_layers_tuple)
        )
        if overlap:
            raise ValueError(
                "Video-cache replacement and deletion must be disjoint; overlapping layers: "
                f"{overlap}."
            )
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if replacement_input_image is not None:
            if replacement_input_image.ndim == 3:
                replacement_input_image = replacement_input_image.unsqueeze(0)
            if replacement_input_image.shape != input_image.shape:
                raise ValueError(
                    "`replacement_input_image` must exactly match the current image shape; "
                    f"current={tuple(input_image.shape)}, "
                    f"replacement={tuple(replacement_input_image.shape)}."
                )
            if replacement_input_image.dtype != input_image.dtype:
                raise TypeError(
                    "`replacement_input_image` must exactly match the current image dtype; "
                    f"current={input_image.dtype}, replacement={replacement_input_image.dtype}."
                )
            if replacement_input_image.device != input_image.device:
                raise ValueError(
                    "`replacement_input_image` must initially be on the same device as the "
                    "current image; "
                    f"current={input_image.device}, replacement={replacement_input_image.device}."
                )
            if not bool(torch.isfinite(replacement_input_image).all().item()):
                raise ValueError("`replacement_input_image` must contain only finite values.")
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        replacement_first_frame_latents = None
        if replacement_input_image is not None:
            replacement_input_image = replacement_input_image.to(
                device=self.device, dtype=self.torch_dtype
            )
            replacement_first_frame_latents = self._encode_input_image_latents_tensor(
                input_image=replacement_input_image,
                tiled=tiled,
            )
            if replacement_first_frame_latents.shape != first_frame_latents.shape:
                raise ValueError(
                    "Replacement first-frame latent shape mismatch: "
                    f"current={tuple(first_frame_latents.shape)}, "
                    f"replacement={tuple(replacement_first_frame_latents.shape)}."
                )
            if replacement_first_frame_latents.dtype != first_frame_latents.dtype:
                raise TypeError(
                    "Replacement first-frame latent dtype mismatch: "
                    f"current={first_frame_latents.dtype}, "
                    f"replacement={replacement_first_frame_latents.dtype}."
                )
            if replacement_first_frame_latents.device != first_frame_latents.device:
                raise ValueError(
                    "Replacement first-frame latent device mismatch: "
                    f"current={first_frame_latents.device}, "
                    f"replacement={replacement_first_frame_latents.device}."
                )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f_video,
            _h_video,
            _w_video,
            tokens_per_frame,
        ) = self.video_expert.prepare(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        replacement_video_prepared = None
        if replacement_first_frame_latents is not None:
            replacement_video_prepared = self.video_expert.prepare(
                x=replacement_first_frame_latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            structural_pairs = (
                ("tokens", video_tokens, replacement_video_prepared[0]),
                ("timestep modulation", video_t_mod, replacement_video_prepared[2]),
                ("context", video_context, replacement_video_prepared[3]),
                ("context mask", video_context_mask, replacement_video_prepared[4]),
                ("frequencies", video_freqs, replacement_video_prepared[5]),
            )
            for label, current_tensor, replacement_tensor in structural_pairs:
                if replacement_tensor.shape != current_tensor.shape:
                    raise ValueError(
                        f"Replacement video {label} shape mismatch: "
                        f"current={tuple(current_tensor.shape)}, "
                        f"replacement={tuple(replacement_tensor.shape)}."
                    )
                if replacement_tensor.dtype != current_tensor.dtype:
                    raise TypeError(
                        f"Replacement video {label} dtype mismatch: "
                        f"current={current_tensor.dtype}, "
                        f"replacement={replacement_tensor.dtype}."
                    )
                if replacement_tensor.device != current_tensor.device:
                    raise ValueError(
                        f"Replacement video {label} device mismatch: "
                        f"current={current_tensor.device}, "
                        f"replacement={replacement_tensor.device}."
                    )
            if int(replacement_video_prepared[9]) != int(tokens_per_frame):
                raise ValueError(
                    "Replacement video token layout mismatch: "
                    f"current tokens/frame={tokens_per_frame}, "
                    f"replacement tokens/frame={replacement_video_prepared[9]}."
                )
        video_seq_len = int(video_tokens.shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        video_attention_mask = attention_mask[:video_seq_len, :video_seq_len]
        action_attention_mask = attention_mask[video_seq_len:, :]
        action_visible_indices = action_visible_video_token_indices(
            action_attention_mask,
            video_seq_len=video_seq_len,
        )
        if expected_video_cache_layout is not None:
            expected_layout_values = {
                "video_seq_len": int(expected_video_cache_layout["video_seq_len"]),
                "action_visible_token_indices": tuple(
                    int(index)
                    for index in expected_video_cache_layout[
                        "action_visible_token_indices"
                    ]
                ),
                "tokens_per_frame": int(
                    expected_video_cache_layout["tokens_per_frame"]
                ),
                "num_layers": int(expected_video_cache_layout["num_layers"]),
                "num_heads": int(expected_video_cache_layout["num_heads"]),
                "head_dim": int(expected_video_cache_layout["head_dim"]),
            }
            observed_layout_values = {
                "video_seq_len": video_seq_len,
                "action_visible_token_indices": action_visible_indices,
                "tokens_per_frame": int(tokens_per_frame),
                "num_layers": int(self.mot.num_layers),
                "num_heads": int(self.mot.num_heads),
                "head_dim": int(self.mot.attn_head_dim),
            }
            if "video_grid_size" in expected_video_cache_layout:
                expected_layout_values["video_grid_size"] = tuple(
                    int(value)
                    for value in expected_video_cache_layout["video_grid_size"]
                )
                observed_layout_values["video_grid_size"] = (
                    int(_f_video),
                    int(_h_video),
                    int(_w_video),
                )
            if "input_image_shape" in expected_video_cache_layout:
                expected_layout_values["input_image_shape"] = tuple(
                    int(value)
                    for value in expected_video_cache_layout["input_image_shape"]
                )
                observed_layout_values["input_image_shape"] = tuple(input_image.shape)
            if observed_layout_values != expected_layout_values:
                raise ValueError(
                    "Runtime video-cache layout drifted from the frozen hybrid-mask "
                    f"manifest: observed={observed_layout_values}, "
                    f"expected={expected_layout_values}."
                )
        if compile_action_infer:
            if not hasattr(self, "_prefill_video_cache_compiled"):
                self._prefill_video_cache_compiled = torch.compile(
                    self.mot.prefill_video_cache_tensor,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            if not hasattr(self, "_denoise_action_with_video_cache_compiled"):
                self._denoise_action_with_video_cache_compiled = torch.compile(
                    self._denoise_action_with_video_cache,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
            prefill_video_cache = self._prefill_video_cache_compiled
            denoise_action_with_video_cache = self._denoise_action_with_video_cache_compiled
        else:
            prefill_video_cache = self.mot.prefill_video_cache_tensor
            denoise_action_with_video_cache = self._denoise_action_with_video_cache
        if compile_action_infer:
            torch.compiler.cudagraph_mark_step_begin()
        video_cache_k, video_cache_v = prefill_video_cache(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=video_attention_mask,
        )
        if compile_action_infer:
            # Inductor reduce-overhead may return graph-owned buffers that are overwritten on replay.
            video_cache_k = [cache.clone() for cache in video_cache_k]
            video_cache_v = [cache.clone() for cache in video_cache_v]

        current_video_cache_k = video_cache_k
        current_video_cache_v = video_cache_v
        replacement_video_cache_k = None
        replacement_video_cache_v = None
        replacement_video_seq_len = None
        replacement_tokens_per_frame = None
        hybrid_video_cache_audit = None
        action_sensitive_cache_tensors: dict[str, dict[int, torch.Tensor]] | None = None
        if replacement_video_prepared is not None:
            replacement_video_tokens = replacement_video_prepared[0]
            replacement_video_seq_len = int(replacement_video_tokens.shape[1])
            replacement_tokens_per_frame = int(replacement_video_prepared[9])
            if replacement_video_seq_len != video_seq_len:
                raise ValueError(
                    "Replacement video sequence length mismatch: "
                    f"current={video_seq_len}, replacement={replacement_video_seq_len}."
                )
            if compile_action_infer:
                torch.compiler.cudagraph_mark_step_begin()
            replacement_video_cache_k, replacement_video_cache_v = prefill_video_cache(
                video_tokens=replacement_video_tokens,
                video_freqs=replacement_video_prepared[5],
                video_t_mod=replacement_video_prepared[2],
                video_context=replacement_video_prepared[3],
                video_context_mask=replacement_video_prepared[4],
                video_attention_mask=video_attention_mask,
            )
            if compile_action_infer:
                replacement_video_cache_k = [
                    cache.clone() for cache in replacement_video_cache_k
                ]
                replacement_video_cache_v = [
                    cache.clone() for cache in replacement_video_cache_v
                ]
            if action_sensitive_enabled:
                visible_index = torch.tensor(
                    action_visible_indices,
                    device=current_video_cache_k[action_sensitive_layers_tuple[0]].device,
                    dtype=torch.long,
                )
                action_sensitive_cache_tensors = {"k": {}, "v": {}}
                selected_k = list(current_video_cache_k)
                selected_v = list(current_video_cache_v)
                with torch.enable_grad():
                    for kind, current, wrong, destination in (
                        (
                            "k",
                            current_video_cache_k,
                            replacement_video_cache_k,
                            selected_k,
                        ),
                        (
                            "v",
                            current_video_cache_v,
                            replacement_video_cache_v,
                            selected_v,
                        ),
                    ):
                        for layer in action_sensitive_layers_tuple:
                            current_rows = current[layer].index_select(1, visible_index)
                            wrong_rows = wrong[layer].index_select(1, visible_index)
                            interpolated = (
                                wrong_rows
                                + float(action_sensitive_interpolation_lambda)
                                * (current_rows - wrong_rows)
                            ).detach().clone().requires_grad_(True)
                            destination[layer] = torch.index_copy(
                                current[layer].detach(), 1, visible_index, interpolated
                            )
                            action_sensitive_cache_tensors[kind][layer] = interpolated
                video_cache_k, video_cache_v = selected_k, selected_v
                hybrid_video_cache_audit = {
                    "schema_version": 1,
                    "mode": "action_sensitive_interpolation",
                    "replacement_video_layers": list(action_sensitive_layers_tuple),
                    "interpolation_lambda": float(
                        action_sensitive_interpolation_lambda
                    ),
                    "action_visible_token_indices": list(action_visible_indices),
                    "action_visible_token_count": len(action_visible_indices),
                    "shape_preserved": True,
                    "only_injected_cache_tensors_require_grad": True,
                }
            elif feature_projection_enabled:
                assert feature_projection_bases_by_layer is not None
                assert feature_projection_rank is not None
                video_cache_k, video_cache_v, hybrid_video_cache_audit = (
                    project_replacement_video_cache(
                        current_cache_k=current_video_cache_k,
                        current_cache_v=current_video_cache_v,
                        replacement_cache_k=replacement_video_cache_k,
                        replacement_cache_v=replacement_video_cache_v,
                        replacement_video_layers=replacement_video_layers_tuple,
                        action_visible_token_indices=action_visible_indices,
                        feature_bases_by_layer=feature_projection_bases_by_layer,
                        projection_rank=int(feature_projection_rank),
                        num_layers=self.mot.num_layers,
                    )
                )
            elif token_hybrid_enabled or head_hybrid_enabled:
                video_cache_k, video_cache_v, hybrid_video_cache_audit = (
                    mix_replacement_video_cache(
                        current_cache_k=current_video_cache_k,
                        current_cache_v=current_video_cache_v,
                        replacement_cache_k=replacement_video_cache_k,
                        replacement_cache_v=replacement_video_cache_v,
                        replacement_video_layers=replacement_video_layers_tuple,
                        action_visible_token_indices=action_visible_indices,
                        retained_current_token_indices=(
                            retained_current_video_token_indices
                            if token_hybrid_enabled
                            else None
                        ),
                        retained_current_heads_by_layer=(
                            retained_current_video_heads_by_layer
                            if head_hybrid_enabled
                            else None
                        ),
                        num_heads=int(self.mot.num_heads),
                        head_dim=int(self.mot.attn_head_dim),
                        num_layers=self.mot.num_layers,
                    )
                )
            else:
                video_cache_k, video_cache_v = select_replacement_video_cache(
                    current_cache_k=current_video_cache_k,
                    current_cache_v=current_video_cache_v,
                    replacement_cache_k=replacement_video_cache_k,
                    replacement_cache_v=replacement_video_cache_v,
                    replacement_video_layers=replacement_video_layers_tuple,
                    num_layers=self.mot.num_layers,
                )

        video_cache_stats = None
        if return_video_cache_stats:
            video_cache_stats = build_video_cache_stats(
                current_cache_k=current_video_cache_k,
                current_cache_v=current_video_cache_v,
                replacement_cache_k=replacement_video_cache_k,
                replacement_cache_v=replacement_video_cache_v,
                replacement_video_layers=replacement_video_layers_tuple,
                num_layers=self.mot.num_layers,
                summary_video_layers=(
                    replacement_video_layers_tuple
                    if replacement_video_layers_tuple
                    else None
                ),
            )
            video_cache_stats.update(
                {
                    "disabled_video_layers": list(disabled_video_layers_tuple),
                    "current_video_seq_len": video_seq_len,
                    "replacement_video_seq_len": replacement_video_seq_len,
                    "current_video_tokens_per_frame": int(tokens_per_frame),
                    "current_video_grid_size": [
                        int(_f_video),
                        int(_h_video),
                        int(_w_video),
                    ],
                    "current_input_image_shape": list(input_image.shape),
                    "replacement_video_tokens_per_frame": replacement_tokens_per_frame,
                    "action_attention_mask_shape": list(action_attention_mask.shape),
                    "action_visible_video_token_indices": list(action_visible_indices),
                    "action_visible_video_token_count": len(action_visible_indices),
                    "num_heads": int(self.mot.num_heads),
                    "head_dim": int(self.mot.attn_head_dim),
                    "hybrid_video_cache": hybrid_video_cache_audit,
                }
            )
            if hybrid_video_cache_audit is not None:
                hybrid_source = "hybrid_" + str(hybrid_video_cache_audit["mode"])
                for layer_entry in video_cache_stats["layers"]:
                    if int(layer_entry["layer"]) in replacement_video_layers_tuple:
                        layer_entry["selected_source"] = hybrid_source

        cache_deltas = None
        if return_video_cache_deltas:
            assert replacement_video_cache_k is not None
            assert replacement_video_cache_v is not None
            visible_index = torch.tensor(
                action_visible_indices,
                device=current_video_cache_k[delta_layers_tuple[0]].device,
                dtype=torch.long,
            )
            cache_deltas = {
                kind: {
                    layer: (
                        current[layer].index_select(1, visible_index)
                        - replacement[layer].index_select(1, visible_index)
                    ).detach()
                    for layer in delta_layers_tuple
                }
                for kind, current, replacement in (
                    ("k", current_video_cache_k, replacement_video_cache_k),
                    ("v", current_video_cache_v, replacement_video_cache_v),
                )
            }
        video_cache_layout = None
        if cache_only or action_sensitive_enabled:
            representative = (
                cache_deltas["k"][delta_layers_tuple[0]]
                if cache_deltas is not None
                else action_sensitive_cache_tensors["k"][
                    action_sensitive_layers_tuple[0]
                ]
            )
            video_cache_layout = {
                "num_layers": int(self.mot.num_layers),
                "video_seq_len": video_seq_len,
                "action_visible_token_indices": list(action_visible_indices),
                "action_visible_token_count": len(action_visible_indices),
                "feature_dim": int(representative.shape[-1]),
                "num_heads": int(self.mot.num_heads),
                "head_dim": int(self.mot.attn_head_dim),
                "tokens_per_frame": int(tokens_per_frame),
                "video_grid_size": [int(_f_video), int(_h_video), int(_w_video)],
                "input_image_shape": list(input_image.shape),
                "delta_layers": list(delta_layers_tuple),
                "action_sensitive_layers": list(action_sensitive_layers_tuple),
            }
        if cache_only:
            return {
                "video_cache_deltas": cache_deltas,
                "video_cache_layout": video_cache_layout,
            }

        if replacement_enabled:
            current_video_cache_k = []
            current_video_cache_v = []
            replacement_video_cache_k = None
            replacement_video_cache_v = None
            replacement_video_prepared = None
            replacement_first_frame_latents = None
            replacement_input_image = None

        with torch.set_grad_enabled(action_sensitive_enabled):
            infer_timesteps_action, infer_deltas_action = (
                self.infer_action_scheduler.build_inference_schedule(
                    num_inference_steps=num_inference_steps,
                    device=self.device,
                    dtype=latents_action.dtype,
                    shift_override=sigma_shift,
                )
            )
            for step_t_action, step_delta_action in zip(
                infer_timesteps_action, infer_deltas_action
            ):
                if compile_action_infer:
                    torch.compiler.cudagraph_mark_step_begin()
                timestep_action = step_t_action.unsqueeze(0).to(
                    dtype=latents_action.dtype, device=self.device
                )

                if action_sensitive_enabled and action_sensitive_gradient_checkpointing:
                    flat_cache = tuple(video_cache_k) + tuple(video_cache_v)
                    cache_count = len(video_cache_k)

                    def checkpointed_denoise(
                        action_latents: torch.Tensor,
                        *cache_tensors: torch.Tensor,
                        _timestep: torch.Tensor = timestep_action,
                    ) -> torch.Tensor:
                        return denoise_action_with_video_cache(
                            latents_action=action_latents,
                            timestep_action=_timestep,
                            context=context,
                            context_mask=context_mask,
                            video_cache_k=list(cache_tensors[:cache_count]),
                            video_cache_v=list(cache_tensors[cache_count:]),
                            action_attention_mask=action_attention_mask,
                            disabled_video_layers=disabled_video_layers_tuple,
                        )

                    pred_action_posi = activation_checkpoint(
                        checkpointed_denoise,
                        latents_action,
                        *flat_cache,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    pred_action_posi = denoise_action_with_video_cache(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_cache_k=video_cache_k,
                        video_cache_v=video_cache_v,
                        action_attention_mask=action_attention_mask,
                        disabled_video_layers=disabled_video_layers_tuple,
                    )
                latents_action = self.infer_action_scheduler.step(
                    pred_action_posi, step_delta_action, latents_action
                )
            action_sensitive_output = (
                latents_action[0].to(dtype=torch.float32)
                if action_sensitive_enabled
                else None
            )

        output = {
            "action": (
                action_sensitive_output
                if action_sensitive_enabled
                else latents_action[0].detach().to(device="cpu", dtype=torch.float32)
            ),
        }
        if action_sensitive_enabled:
            assert action_sensitive_cache_tensors is not None
            output["action_sensitive_cache_tensors"] = action_sensitive_cache_tensors
            output["video_cache_layout"] = video_cache_layout
            output["action_sensitive_inference"] = {
                "interpolation_lambda": float(action_sensitive_interpolation_lambda),
                "layers": list(action_sensitive_layers_tuple),
                "num_inference_steps": int(num_inference_steps),
                "gradient_checkpointing": bool(action_sensitive_gradient_checkpointing),
                "raw_action_shape": list(latents_action[0].shape),
                "raw_action_normalized": True,
                "model_parameters_frozen": True,
            }
        if video_cache_stats is not None:
            output["video_cache_stats"] = video_cache_stats
        if cache_deltas is not None:
            output["video_cache_deltas"] = cache_deltas
        return output

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
