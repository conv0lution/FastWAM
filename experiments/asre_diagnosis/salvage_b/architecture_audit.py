"""Static Phase-A audit for the exact shared first-frame video K/V interface."""

from __future__ import annotations

import argparse
import ast
import json
import os
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _function_line(path: Path, class_name: str | None, function_name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes: list[ast.AST] = list(tree.body)
    if class_name is not None:
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise ValueError(f"Expected one class {class_name} in {path}.")
        nodes = list(classes[0].body)
    matches = [
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one function {function_name} in {path}.")
    return int(matches[0].lineno)


def _location(relative: str, class_name: str | None, function_name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative
    return {
        "path": relative,
        "class": class_name,
        "function": function_name,
        "line": _function_line(path, class_name, function_name),
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text.rstrip() + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report() -> dict[str, Any]:
    model_config = OmegaConf.to_container(
        # This file intentionally contains interpolations owned by the task/data
        # composition (for example the action dimension).  The architecture
        # fields audited below are literal values, so resolving the unrelated
        # interpolation in isolation would make a valid static audit fail.
        OmegaConf.load(PROJECT_ROOT / "configs/model/fastwam.yaml"), resolve=False
    )
    video = model_config["video_dit_config"]
    action = model_config["action_dit_config"]
    expected_config = {
        "video_hidden_dim": 3072,
        "video_num_layers": 30,
        "video_num_heads": 24,
        "video_head_dim": 128,
        "patch_size": [1, 2, 2],
        "video_attention_mask_mode": "first_frame_causal",
        "seperated_timestep": True,
        "fuse_vae_embedding_in_latents": True,
        "has_image_input": False,
        "require_clip_embedding": False,
        "video_action_conditioned": False,
        "action_hidden_dim": 1024,
        "action_num_layers": 30,
        "action_num_heads": 24,
        "action_head_dim": 128,
        "video_train_shift": 5.0,
        "video_infer_shift": 5.0,
        "video_num_train_timesteps": 1000,
        "action_train_shift": 1.0,
        "action_num_train_timesteps": 1000,
    }
    observed_config = {
        "video_hidden_dim": int(video["hidden_dim"]),
        "video_num_layers": int(video["num_layers"]),
        "video_num_heads": int(video["num_heads"]),
        "video_head_dim": int(video["attn_head_dim"]),
        "patch_size": list(video["patch_size"]),
        "video_attention_mask_mode": str(video["video_attention_mask_mode"]),
        "seperated_timestep": bool(video["seperated_timestep"]),
        "fuse_vae_embedding_in_latents": bool(video["fuse_vae_embedding_in_latents"]),
        "has_image_input": bool(video["has_image_input"]),
        "require_clip_embedding": bool(video["require_clip_embedding"]),
        "video_action_conditioned": bool(video["action_conditioned"]),
        "action_hidden_dim": int(action["hidden_dim"]),
        "action_num_layers": int(action["num_layers"]),
        "action_num_heads": int(action["num_heads"]),
        "action_head_dim": int(action["attn_head_dim"]),
        "video_train_shift": float(model_config["video_scheduler"]["train_shift"]),
        "video_infer_shift": float(model_config["video_scheduler"]["infer_shift"]),
        "video_num_train_timesteps": int(
            model_config["video_scheduler"]["num_train_timesteps"]
        ),
        "action_train_shift": float(model_config["action_scheduler"]["train_shift"]),
        "action_num_train_timesteps": int(
            model_config["action_scheduler"]["num_train_timesteps"]
        ),
    }
    mismatch = {
        key: {"observed": observed_config[key], "expected": value}
        for key, value in expected_config.items()
        if observed_config[key] != value
    }
    locations = {
        "input_image_latents": _location(
            "src/fastwam/models/wan22/fastwam.py",
            "FastWAM",
            "_encode_input_image_latents_tensor",
        ),
        "attention_mask": _location(
            "src/fastwam/models/wan22/fastwam.py",
            "FastWAM",
            "_build_mot_attention_mask",
        ),
        "native_training_loss": _location(
            "src/fastwam/models/wan22/fastwam.py", "FastWAM", "training_loss"
        ),
        "native_world_evaluation": _location(
            "experiments/asre_diagnosis/salvage_b/world_runtime.py",
            None,
            "native_world_loss",
        ),
        "native_video_inference_schedule": _location(
            "src/fastwam/models/wan22/schedulers/scheduler_continuous.py",
            "WanContinuousFlowMatchScheduler",
            "build_inference_schedule",
        ),
        "action_cache_consumer": _location(
            "src/fastwam/models/wan22/mot.py",
            "MoT",
            "forward_action_with_video_cache_tensor",
        ),
        "cache_prefill": _location(
            "src/fastwam/models/wan22/mot.py", "MoT", "prefill_video_cache_tensor"
        ),
        "world_cache_consumer": _location(
            "src/fastwam/models/wan22/mot.py",
            "MoT",
            "forward_future_video_with_video_cache_tensor",
        ),
        "video_prepare": _location(
            "src/fastwam/models/wan22/wan_video_dit.py",
            "WanVideoDiT",
            "prepare",
        ),
        "video_mask": _location(
            "src/fastwam/models/wan22/wan_video_dit.py",
            "WanVideoDiT",
            "build_video_to_video_mask",
        ),
        "projection": _location(
            "src/fastwam/models/wan22/video_cache_replacement.py",
            None,
            "project_replacement_video_cache",
        ),
    }
    static_passed = not mismatch
    source_paths = sorted(
        {str(value["path"]) for value in locations.values()}
        | {"configs/model/fastwam.yaml"}
    )
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    return {
        "artifact_type": "asre_salvage_b_phase_a_architecture_audit",
        "schema_version": 1,
        "git_commit_hash": current_commit,
        "status": (
            "preferred_path_a_pending_runtime_machinery"
            if static_passed
            else "shared_interface_not_available"
        ),
        "static_audit_passed": static_passed,
        "phase_b_authorized": False,
        "phase_b_authorization_requires_runtime_machinery": True,
        "path": "preferred_path_a" if static_passed else None,
        "shared_interface": {
            "name": "late_first_frame_video_kv_prefix",
            "definition": "Z={(K_l_prefix,V_l_prefix): l=15..29}",
            "source": "same current RGB observation encoded by the frozen video expert",
            "layers": list(range(15, 30)),
            "shape_per_tensor": [1, 98, 3072],
            "tokens": 98,
            "heads": 24,
            "head_dim": 128,
            "projection": "Z_wrong + (Z_current-Z_wrong) @ B_r @ B_r.T",
            "k_v_fitted_separately": True,
        },
        "tensor_flow": [
            "FastWAM.build_inputs: real RGB clip [B,3,9,224,448] -> scoring-only frozen VAE target [B,48,3,14,28]",
            "FastWAM._encode_input_image_latents_tensor: current/donor RGB [B,3,224,448] -> prefix latent [B,48,1,14,28]",
            "WanVideoDiT.prepare: prefix latent -> video prefix tokens [B,98,3072]",
            "MoT.prefill_video_cache_tensor: prefix tokens -> per-layer K/V [B,98,3072]",
            "video_cache_replacement.project_replacement_video_cache: same selected K/V Z",
            "MoT.forward_action_with_video_cache_tensor: selected Z -> action queries",
            "constant zero latent [B,48,1,14,28]: discarded carrier placeholder only; it contains no observation or target value",
            "fixed Gaussian draw: initialize only future latents [B,48,2,14,28] independently of the real future target",
            "native infer-video scheduler: evolve the same future-noise draw for exactly 10 steps",
            "MoT.forward_future_video_with_video_cache_tensor: selected Z -> future-video queries at every inference step",
            "WanVideoDiT.post plus scheduler.step: future tokens -> generated future latent [B,48,2,14,28]",
            "scoring only after step 10: generated future latent versus the real future latent target",
        ],
        "matched_condition_semantics": {
            "layers_0_14": "prefix disabled for both functions",
            "layers_15_29": "the exact same selected cache tensors feed both consumers",
            "conditions": ["current_all", "wrong_all", "svd_r97", "svd_r170"],
            "raw_prefix_residual_enters_world_future": False,
            "carrier_prefix_placeholder": "constant zero and discarded",
            "current_observation_enters_world_only_through_selected_kv": True,
            "real_future_target_enters_world_inference": False,
        },
        "eligibility_criteria": {
            "same_current_observation": True,
            "upstream_of_action": True,
            "upstream_of_native_future_video_prediction": True,
            "same_tensor_intervention_supported": True,
            "untouched_rgb_bypass_removed_by_zero_prefix_and_pure_noise_future_inference": True,
            "token_and_key_counts_unchanged": True,
            "weights_and_architecture_unchanged": True,
            "current_and_wrong_endpoints_technically_defined": True,
        },
        "important_distinction": (
            "The stock joint path is retained only as a one-step factorization oracle: its "
            "true current prefix produces the same external cache used by the factorized "
            "consumer, while patch_t=1 makes the future tokens independent of the discarded "
            "zero carrier slot. Salvage B inference uses the selected external K/V as the "
            "sole current-observation pathway; runtime numerical equivalence is a hard gate."
        ),
        "native_training_objective": {
            "source": "FastWAM.training_loss",
            "scheduler": "WanContinuousFlowMatchScheduler(train_shift=5.0)",
            "construction": (
                "sample a training timestep and Gaussian noise; form noisy full-clip "
                "latents; predict the flow-matching velocity target; exclude the frozen "
                "initial latent step; apply padding-aware latent MSE and the native "
                "scheduler training weight"
            ),
            "not_used_as_primary_here_because": (
                "a one-step teacher-forced score supplies a noised version of the real "
                "future target to the predictor, creating a target-side route around the "
                "shared current-observation K/V intervention"
            ),
        },
        "native_metric": {
            "name": "pure_noise_native_future_latent_reconstruction_mse",
            "lower_is_better": True,
            "inference_initial_state": "fixed pure Gaussian future-latent noise",
            "inference_steps": 10,
            "inference_shift": 5.0,
            "target": "real causal-VAE future latent, scoring only after inference",
            "target_enters_predictor": False,
            "carrier_prefix": "constant zero placeholder; discarded before MoT",
            "current_observation_pathway": "selected external K/V cache only",
            "draws_per_sample": 4,
        },
        "source_locations": locations,
        "source_inventory": [
            {
                "path": relative,
                "sha256": _sha256(PROJECT_ROOT / relative),
            }
            for relative in source_paths
        ],
        "observed_model_config": observed_config,
        "config_mismatch": mismatch,
        "required_runtime_gates": [
            "first pure-noise inference step with the true-current external cache and no disabled layers matches the stock joint future slice whose prefix is the true current latent (patch_t=1)",
            "same cache object reaches action and world consumers",
            "cache intervention changes both consumers",
            "future inference starts from the exact frozen Gaussian draw and executes exactly 10 native scheduler steps",
            "changing the scoring-only real future target cannot change generated future latents when noise/cache/conditioning are fixed",
            "discarded zero carrier prefix cannot change future output when selected external K/V is fixed",
            "reported native world loss exactly equals terminal future-latent reconstruction MSE",
            "rank-0 equals Wrong and rank-D equals Current",
            "r97/r170 keep shape, token count, head count, checkpoint and basis identity",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    status = report["status"]
    lines = [
        "# Salvage B Phase-A Architecture Audit",
        "",
        f"Status: **{status}**",
        "",
        "The eligible shared interface is the exact late-layer first-frame video K/V "
        "prefix: layers 15–29, K and V each `[1,98,3072]` (24 heads × 128).",
        "",
        "Static audit identifies one candidate selected-cache interface after the diagnostic "
        "cached-prefix factorization. Layers 0–14 are configured to disable the prefix for "
        "both consumers. Runtime same-object, numerical-equivalence, intervention-reach, and "
        "pure-noise/no-target-bypass checks remain mandatory before Phase B is authorized.",
        "",
        "Preferred Path A is statically valid, but Phase B remains unauthorized until the "
        "real-checkpoint machinery report passes the numerical equivalence and no-bypass gates.",
        "",
        "## Native world objective",
        "",
        "The checkpoint's training objective is scheduler-weighted, padding-aware "
        "future-latent flow-matching MSE after excluding the frozen initial latent step. "
        "A one-step teacher-forced evaluation would expose a noised real future target "
        "to the predictor and therefore bypass the causal question being tested.",
        "",
        "Primary loss is terminal future-latent reconstruction MSE after the frozen native "
        "10-step video inference scheduler starts from pure Gaussian future noise. The real "
        "future latent is scoring-only and never enters inference. The native three-frame "
        "carrier uses a constant-zero discarded prefix, so selected external K/V is the only "
        "current-observation pathway. Exactly four deterministic noise draws are used per sample.",
        "",
        "## Exact source locations",
        "",
    ]
    for name, location in report["source_locations"].items():
        lines.append(
            f"- `{name}`: `{location['path']}:{location['line']}` "
            f"(`{location['function']}`)."
        )
    lines.extend(["", "## Runtime gates", ""])
    lines.extend(f"- {item}." for item in report["required_runtime_gates"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    _atomic_text(args.output_md.resolve(), _markdown(report))
    # Publish JSON last: the driver treats it as the completed-stage sentinel
    # and may safely replace an orphan Markdown file after interruption.
    _atomic_text(
        args.output_json.resolve(), json.dumps(report, indent=2, sort_keys=True)
    )
    if not report["static_audit_passed"]:
        raise RuntimeError("Salvage-B shared interface is not statically available.")
    print(f"Salvage-B Phase-A static audit: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
