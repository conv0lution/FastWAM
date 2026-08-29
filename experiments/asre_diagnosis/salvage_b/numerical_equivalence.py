"""Independent dtype-scaling audit for Salvage-B cache factorization.

This module is machinery verification only.  It reuses one frozen sample and
draw from the completed Salvage-B run, never reads Phase-B outcomes, and never
launches an environment or action rollout.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
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
from experiments.asre_diagnosis.salvage_b import machinery_tests  # noqa: E402
from experiments.asre_diagnosis.salvage_b.machinery_tests import (  # noqa: E402
    _action_prediction,
    _factorized_future_tokens,
    _install_consumer_capture,
    _prepare_draw_state,
    _validate_frozen_inputs,
)
from experiments.asre_diagnosis.salvage_b.world_runtime import (  # noqa: E402
    PREFIX_TOKENS,
    load_donor_bundle,
    load_frozen_model,
    load_processed_sample,
    load_prompt_cache,
    load_world_dataset,
    prepare_world_sample,
    read_json,
)
from fastwam.models.wan22.wan_video_dit import flash_attention  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


CONFIRMED = "NUMERICAL-EQUIVALENCE-CONFIRMED"
UNCERTAIN = "NUMERICAL-EQUIVALENCE-UNCERTAIN"
POSSIBLE_MISMATCH = "POSSIBLE-LOGICAL-MISMATCH"
REQUIRED_FP32_REDUCTION = 5.0
STRONG_FP32_REDUCTION = 10.0
FP32_PERSISTENT_DISCREPANCY = 5.0e-3
SUDDEN_JUMP_FACTOR = 10.0
SUDDEN_JUMP_MINIMUM = 5.0e-4
LAYER_TENSORS = (
    "input_hidden_state",
    "prefix_hidden_state",
    "q_future",
    "k_future",
    "v_future",
    "prefix_k",
    "prefix_v",
    "attention_output",
    "block_output",
    "future_token_hidden_output",
)


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _atomic_text(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text.rstrip() + "\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to publish an empty layerwise CSV.")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": tensor_sha256(value),
        "finite": bool(torch.isfinite(value).all().item())
        if value.is_floating_point() or value.is_complex()
        else True,
    }


def _tensor_list_record(tensors: Sequence[torch.Tensor]) -> dict[str, Any]:
    return {
        "count": len(tensors),
        "tensors": [_tensor_record(tensor) for tensor in tensors],
    }


def tensor_error_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    """Return shape-strict float64 accumulation metrics."""

    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "finite": False,
        }
    left = reference.detach().to(device="cpu", dtype=torch.float64)
    right = candidate.detach().to(device="cpu", dtype=torch.float64)
    difference = right - left
    rms_error = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_rms = float(torch.sqrt(torch.mean(left.square())).item())
    reference_norm = float(torch.linalg.vector_norm(left).item())
    candidate_norm = float(torch.linalg.vector_norm(right).item())
    relative_rmse = rms_error / max(reference_rms, 1.0e-30)
    return {
        "shape_equal": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "finite": bool(
            torch.isfinite(left).all().item()
            and torch.isfinite(right).all().item()
        ),
        "exact_equal": bool(torch.equal(left, right)),
        "max_abs_error": float(difference.abs().max().item()),
        "rms_absolute_error": rms_error,
        "reference_rms": reference_rms,
        "relative_rmse": relative_rmse,
        "reference_output_norm": reference_norm,
        "candidate_output_norm": candidate_norm,
        "relative_norm_error": abs(candidate_norm - reference_norm)
        / max(reference_norm, 1.0e-30),
    }


def _metrics_row(
    *, dtype_name: str, layer: int, tensor_name: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "dtype": dtype_name,
        "layer": layer,
        "tensor": tensor_name,
        "relative_rmse": metrics.get("relative_rmse"),
        "max_abs_error": metrics.get("max_abs_error"),
        "rms_absolute_error": metrics.get("rms_absolute_error"),
        "reference_output_norm": metrics.get("reference_output_norm"),
        "candidate_output_norm": metrics.get("candidate_output_norm"),
        "relative_norm_error": metrics.get("relative_norm_error"),
        "finite": metrics.get("finite"),
        "shape_equal": metrics.get("shape_equal"),
    }


def _extract_test_count(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"(\d+) passed", text)
    return {
        **_artifact(path),
        "passed_count": int(matches[-1]) if matches else None,
        "reported_passed": bool(matches),
    }


def _git_object_exists(commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _worktree_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _source_namespace(source_root: Path) -> tuple[SimpleNamespace, dict[str, Any]]:
    source_root = source_root.resolve()
    completion_path = source_root / "aggregate/salvage_b_completion.json"
    machinery_path = source_root / "machinery_report.json"
    completion = read_json(completion_path)
    source_machinery = read_json(machinery_path)
    source_commit = str(source_machinery.get("git_commit_hash", ""))
    if not (
        completion.get("status") == "complete"
        and completion.get("publication_complete") is True
        and source_machinery.get("status") == "passed"
        and source_machinery.get("passed") is True
        and source_machinery.get("phase_b_authorized") is True
        and source_machinery.get("protocol") == SALVAGE_B_PROTOCOL
        and len(source_commit) == 40
        and _git_object_exists(source_commit)
    ):
        raise ValueError("Source Salvage-B run is not a complete, passed machinery bundle.")

    args = SimpleNamespace(
        preflight=source_root / "preflight_report.json",
        world_manifest=source_root / "manifests/world_evaluation_manifest.json",
        stochastic_manifest=source_root / "manifests/stochastic_manifest.json",
        draw_tensors=source_root / "manifests/fixed_draw_tensors.pt",
        processed_targets=source_root / "manifests/processed_world_targets.json",
    )
    for key, path in vars(args).items():
        expected = source_machinery["inputs"][key]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"Source machinery input hash drifted: {key}.")
    frozen = _validate_frozen_inputs(args, expected_commit=source_commit)
    provenance = {
        "source_root": str(source_root),
        "source_commit": source_commit,
        "source_machinery": _artifact(machinery_path),
        "source_completion": _artifact(completion_path),
        "source_scientific_outcome_inspected": False,
        "source_inputs": {
            key: _artifact(path) for key, path in vars(args).items()
        },
    }
    return args, {
        "frozen": frozen,
        "provenance": provenance,
        "source_machinery": source_machinery,
    }


def _mask_audit(state: Mapping[str, Any]) -> dict[str, Any]:
    full = state["full_joint_mask"].bool()
    video = state["video_mask"].bool()
    prefix = video[:PREFIX_TOKENS, :PREFIX_TOKENS]
    prefix_future_columns = video[:PREFIX_TOKENS, PREFIX_TOKENS:]
    future = video[PREFIX_TOKENS:, :]
    stock_video = full[: video.shape[0], : video.shape[0]]
    stock_action_columns = full[: video.shape[0], video.shape[0] :]
    logical_checks = {
        "stock_video_subgraph_equals_native_video_mask": bool(
            torch.equal(stock_video, video)
        ),
        "prefix_subgraph_equals_factorized_prefix_mask": bool(
            torch.equal(stock_video[:PREFIX_TOKENS, :PREFIX_TOKENS], prefix)
        ),
        "future_subgraph_equals_factorized_future_mask": bool(
            torch.equal(stock_video[PREFIX_TOKENS:, :], future)
        ),
        "stock_video_has_no_visible_action_key": bool(
            not stock_action_columns.any().item()
        ),
        "prefix_has_no_visible_future_key": bool(
            not prefix_future_columns.any().item()
        ),
    }
    shapes = {
        "stock_joint": list(full.shape),
        "stock_video_subgraph": list(stock_video.shape),
        "factorized_prefix": list(prefix.shape),
        "factorized_future": list(future.shape),
        "fully_masked_stock_action_extent_for_video_queries": list(
            stock_action_columns.shape
        ),
    }
    allowed = {
        "stock_joint_total": int(full.sum().item()),
        "stock_video_subgraph": int(stock_video.sum().item()),
        "factorized_prefix": int(prefix.sum().item()),
        "factorized_future": int(future.sum().item()),
        "future_columns_visible_to_prefix_queries": int(
            prefix_future_columns.sum().item()
        ),
        "stock_action_columns_visible_to_video_queries": int(
            stock_action_columns.sum().item()
        ),
    }
    return {
        "passed": all(logical_checks.values()),
        "logical_checks": logical_checks,
        "shapes": shapes,
        "allowed_entries": allowed,
        "logical_relation": (
            "Dropping the 32 fully masked action-key columns and the 98 prefix "
            "query rows changes SDPA extent only. The prefix and future query "
            "rows retain exactly the same visible video-key relationships."
        ),
        "no_extra_visible_token": bool(
            logical_checks["stock_video_has_no_visible_action_key"]
        ),
    }


def _build_prefix_state(model, prepared) -> dict[str, Any]:
    timestep = torch.zeros(
        (1,), device=prepared.current_frame_latent.device,
        dtype=prepared.current_frame_latent.dtype,
    )
    values = model.video_expert.prepare(
        x=prepared.current_frame_latent,
        timestep=timestep,
        context=prepared.inputs["context"],
        context_mask=prepared.inputs["context_mask"],
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    tokens, t, t_mod, context, context_mask, freqs, f, h, w, tpf = values
    if (int(f), int(h), int(w), int(tpf), int(tokens.shape[1])) != (
        1,
        7,
        14,
        PREFIX_TOKENS,
        PREFIX_TOKENS,
    ):
        raise ValueError("Frozen prefix preparation layout drifted.")
    mask = model.video_expert.build_video_to_video_mask(
        video_seq_len=PREFIX_TOKENS,
        video_tokens_per_frame=PREFIX_TOKENS,
        device=tokens.device,
    )
    return {
        "tokens": tokens,
        "t": t,
        "t_mod": t_mod,
        "context": context,
        "context_mask": context_mask,
        "freqs": freqs,
        "mask": mask,
    }


def _cast(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if value.is_floating_point():
        return value.detach().to(dtype=dtype)
    return value.detach()


def _float64_attention_reference(
    *,
    q_video: torch.Tensor,
    k_video: torch.Tensor,
    v_video: torch.Tensor,
    k_action: torch.Tensor,
    v_action: torch.Tensor,
    full_mask: torch.Tensor,
    video_mask: torch.Tensor,
    num_heads: int,
) -> dict[str, Any]:
    """Compare one isolated attention relation with explicit float64 math."""

    def heads(tensor: torch.Tensor) -> torch.Tensor:
        b, s, width = tensor.shape
        return (
            tensor.detach()
            .cpu()
            .double()
            .reshape(b, s, num_heads, width // num_heads)
            .permute(0, 2, 1, 3)
        )

    def explicit(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask.cpu().bool().view(1, 1, *mask.shape), -math.inf)
        result = torch.matmul(torch.softmax(scores, dim=-1), v)
        return result.permute(0, 2, 1, 3).reshape(result.shape[0], result.shape[2], -1)

    qh = heads(q_video)
    kv = heads(k_video)
    vv = heads(v_video)
    ka = heads(k_action)
    va = heads(v_action)
    stock = explicit(
        qh,
        torch.cat([kv, ka], dim=2),
        torch.cat([vv, va], dim=2),
        full_mask[: q_video.shape[1], : k_video.shape[1] + k_action.shape[1]],
    )
    factorized = explicit(qh[:, :, PREFIX_TOKENS:], kv, vv, video_mask[PREFIX_TOKENS:, :])
    return {
        "layer": 0,
        "reference_dtype": "torch.float64",
        "scope": "explicit attention only; identical stock video Q/K/V",
        "comparison": tensor_error_metrics(stock[:, PREFIX_TOKENS:], factorized),
    }


def _run_dtype(
    *,
    model,
    stock_state: Mapping[str, Any],
    prefix_state: Mapping[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Run stock and factorized video cores with one layer promoted at a time."""

    dtype_name = str(dtype)
    mot = model.mot
    video_expert = mot.mixtures["video"]
    action_expert = mot.mixtures["action"]
    original_video_dtype = next(video_expert.parameters()).dtype
    original_action_dtype = next(action_expert.parameters()).dtype

    stock_video_x = _cast(stock_state["stock_input_tokens"], dtype)
    stock_action_x = _cast(stock_state["action_tokens"], dtype)
    prefix_x = _cast(prefix_state["tokens"], dtype)
    factor_x = _cast(stock_state["factorized_future_tokens"], dtype)
    video_freqs = _cast(stock_state["stock_freqs"], dtype)
    action_freqs = _cast(stock_state["action_freqs"], dtype)
    prefix_freqs = _cast(prefix_state["freqs"], dtype)
    video_t_mod = _cast(stock_state["stock_t_mod"], dtype)
    action_t_mod = _cast(stock_state["action_t_mod"], dtype)
    prefix_t_mod = _cast(prefix_state["t_mod"], dtype)
    video_context = _cast(stock_state["stock_context"], dtype)
    action_context = _cast(stock_state["action_context"], dtype)
    prefix_context = _cast(prefix_state["context"], dtype)
    video_context_mask = stock_state["stock_context_mask"].bool()
    action_context_mask = stock_state["action_context_mask"].bool()
    prefix_context_mask = prefix_state["context_mask"].bool()
    full_mask = stock_state["full_joint_mask"].bool()
    video_mask = stock_state["video_mask"].bool()
    prefix_mask = prefix_state["mask"].bool()
    future_mask = video_mask[PREFIX_TOKENS:, :]

    rows: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    isolated_reference: dict[str, Any] | None = None
    try:
        for layer_idx in range(mot.num_layers):
            video_block = video_expert.blocks[layer_idx]
            action_block = action_expert.blocks[layer_idx]
            video_block.to(dtype=dtype)
            action_block.to(dtype=dtype)

            input_metrics = tensor_error_metrics(
                stock_video_x[:, PREFIX_TOKENS:], factor_x
            )
            prefix_input_metrics = tensor_error_metrics(
                stock_video_x[:, :PREFIX_TOKENS], prefix_x
            )

            stock_video_io = mot._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=stock_video_x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            stock_action_io = mot._build_expert_attention_io(
                expert=action_expert,
                block=action_block,
                x=stock_action_x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            prefix_io = mot._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=prefix_x,
                freqs=prefix_freqs,
                t_mod=prefix_t_mod,
            )
            factor_io = mot._build_expert_attention_io(
                expert=video_expert,
                block=video_block,
                x=factor_x,
                freqs=video_freqs[PREFIX_TOKENS:],
                t_mod=video_t_mod[:, PREFIX_TOKENS:],
            )

            (
                sqv,
                skv,
                svv,
                s_residual_v,
                s_gate_msa_v,
                s_shift_mlp_v,
                s_scale_mlp_v,
                s_gate_mlp_v,
                _,
            ) = stock_video_io
            (
                sqa,
                ska,
                sva,
                s_residual_a,
                s_gate_msa_a,
                s_shift_mlp_a,
                s_scale_mlp_a,
                s_gate_mlp_a,
                _,
            ) = stock_action_io
            (
                pq,
                pk,
                pv,
                p_residual,
                p_gate_msa,
                p_shift_mlp,
                p_scale_mlp,
                p_gate_mlp,
                _,
            ) = prefix_io
            (
                fq,
                fk,
                fv,
                f_residual,
                f_gate_msa,
                f_shift_mlp,
                f_scale_mlp,
                f_gate_mlp,
                _,
            ) = factor_io

            stock_mixed = flash_attention(
                q=torch.cat([sqv, sqa], dim=1),
                k=torch.cat([skv, ska], dim=1),
                v=torch.cat([svv, sva], dim=1),
                num_heads=mot.num_heads,
                ctx_mask=full_mask,
            )
            prefix_mixed = flash_attention(
                q=pq,
                k=pk,
                v=pv,
                num_heads=mot.num_heads,
                ctx_mask=prefix_mask,
            )
            factor_mixed = flash_attention(
                q=fq,
                k=torch.cat([pk, fk], dim=1),
                v=torch.cat([pv, fv], dim=1),
                num_heads=mot.num_heads,
                ctx_mask=future_mask,
            )

            stock_video_next = mot._apply_expert_post_block_tensor(
                block=video_block,
                residual_x=s_residual_v,
                mixed_attn_out=stock_mixed[:, : stock_video_x.shape[1]],
                gate_msa=s_gate_msa_v,
                shift_mlp=s_shift_mlp_v,
                scale_mlp=s_scale_mlp_v,
                gate_mlp=s_gate_mlp_v,
                context=video_context,
                context_mask=video_context_mask,
            )
            stock_action_next = mot._apply_expert_post_block_tensor(
                block=action_block,
                residual_x=s_residual_a,
                mixed_attn_out=stock_mixed[:, stock_video_x.shape[1] :],
                gate_msa=s_gate_msa_a,
                shift_mlp=s_shift_mlp_a,
                scale_mlp=s_scale_mlp_a,
                gate_mlp=s_gate_mlp_a,
                context=action_context,
                context_mask=action_context_mask,
            )
            prefix_next = mot._apply_expert_post_block_tensor(
                block=video_block,
                residual_x=p_residual,
                mixed_attn_out=prefix_mixed,
                gate_msa=p_gate_msa,
                shift_mlp=p_shift_mlp,
                scale_mlp=p_scale_mlp,
                gate_mlp=p_gate_mlp,
                context=prefix_context,
                context_mask=prefix_context_mask,
            )
            factor_next = mot._apply_expert_post_block_tensor(
                block=video_block,
                residual_x=f_residual,
                mixed_attn_out=factor_mixed,
                gate_msa=f_gate_msa,
                shift_mlp=f_shift_mlp,
                scale_mlp=f_scale_mlp,
                gate_mlp=f_gate_mlp,
                context=video_context,
                context_mask=video_context_mask[:, PREFIX_TOKENS:],
            )

            comparisons = {
                "input_hidden_state": input_metrics,
                "prefix_hidden_state": prefix_input_metrics,
                "q_future": tensor_error_metrics(
                    sqv[:, PREFIX_TOKENS:], fq
                ),
                "k_future": tensor_error_metrics(
                    skv[:, PREFIX_TOKENS:], fk
                ),
                "v_future": tensor_error_metrics(
                    svv[:, PREFIX_TOKENS:], fv
                ),
                "prefix_k": tensor_error_metrics(
                    skv[:, :PREFIX_TOKENS], pk
                ),
                "prefix_v": tensor_error_metrics(
                    svv[:, :PREFIX_TOKENS], pv
                ),
                "attention_output": tensor_error_metrics(
                    stock_mixed[:, PREFIX_TOKENS : stock_video_x.shape[1]],
                    factor_mixed,
                ),
                "block_output": tensor_error_metrics(
                    stock_video_next[:, PREFIX_TOKENS:], factor_next
                ),
                "future_token_hidden_output": tensor_error_metrics(
                    stock_video_next[:, PREFIX_TOKENS:], factor_next
                ),
            }
            if layer_idx == 0 and dtype == torch.float32:
                isolated_reference = _float64_attention_reference(
                    q_video=sqv,
                    k_video=skv,
                    v_video=svv,
                    k_action=ska,
                    v_action=sva,
                    full_mask=full_mask,
                    video_mask=video_mask,
                    num_heads=mot.num_heads,
                )
            for name in LAYER_TENSORS:
                rows.append(
                    _metrics_row(
                        dtype_name=dtype_name,
                        layer=layer_idx,
                        tensor_name=name,
                        metrics=comparisons[name],
                    )
                )
            layers.append({"layer": layer_idx, "comparisons": comparisons})
            stock_video_x = stock_video_next
            stock_action_x = stock_action_next
            prefix_x = prefix_next
            factor_x = factor_next

            video_block.to(dtype=original_video_dtype)
            action_block.to(dtype=original_action_dtype)

        head = video_expert.head
        original_head_dtype = next(head.parameters()).dtype
        head.to(dtype=dtype)
        try:
            stock_prediction = video_expert.post(
                stock_video_x[:, PREFIX_TOKENS:],
                _cast(stock_state["stock_t"][:, PREFIX_TOKENS:], dtype),
                int(stock_state["f"]) - 1,
                int(stock_state["h"]),
                int(stock_state["w"]),
            )
            factor_prediction = video_expert.post(
                factor_x,
                _cast(stock_state["factorized_t"][:, PREFIX_TOKENS:], dtype),
                int(stock_state["f"]) - 1,
                int(stock_state["h"]),
                int(stock_state["w"]),
            )
        finally:
            head.to(dtype=original_head_dtype)
    finally:
        for block in video_expert.blocks:
            block.to(dtype=original_video_dtype)
        for block in action_expert.blocks:
            block.to(dtype=original_action_dtype)

    return {
        "dtype": dtype_name,
        "machine_epsilon": float(torch.finfo(dtype).eps),
        "future_token_output": tensor_error_metrics(
            stock_video_x[:, PREFIX_TOKENS:], factor_x
        ),
        "future_prediction_output": tensor_error_metrics(
            stock_prediction, factor_prediction
        ),
        "layers": layers,
        "rows": rows,
        "isolated_float64_attention_reference": isolated_reference,
        "model_parameter_dtypes_restored": bool(
            next(video_expert.parameters()).dtype == original_video_dtype
            and next(action_expert.parameters()).dtype == original_action_dtype
        ),
        "execution_method": (
            "The identical frozen prepared tensors were cast to the requested "
            "dtype. One video block and its matching action block were promoted "
            "at a time, executed through the stock and factorized graphs, then "
            "restored. This avoids materializing a second full fp32 checkpoint."
        ),
    }


def _sudden_jumps(rows: Sequence[Mapping[str, Any]], dtype_name: str) -> list[dict[str, Any]]:
    jumps: list[dict[str, Any]] = []
    for tensor_name in LAYER_TENSORS:
        selected = sorted(
            (
                row
                for row in rows
                if row["dtype"] == dtype_name and row["tensor"] == tensor_name
            ),
            key=lambda row: int(row["layer"]),
        )
        for previous, current in zip(selected, selected[1:]):
            before = float(previous["relative_rmse"])
            after = float(current["relative_rmse"])
            ratio = after / max(before, 1.0e-30)
            if ratio >= SUDDEN_JUMP_FACTOR and after >= SUDDEN_JUMP_MINIMUM:
                jumps.append(
                    {
                        "tensor": tensor_name,
                        "previous_layer": int(previous["layer"]),
                        "layer": int(current["layer"]),
                        "previous_relative_rmse": before,
                        "relative_rmse": after,
                        "jump_factor": ratio,
                    }
                )
    return jumps


def classify_numerical_equivalence(
    *,
    bf16_prediction_relative_rmse: float,
    fp32_prediction_relative_rmse: float,
    structural_passed: bool,
    fp32_sudden_jumps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reduction = bf16_prediction_relative_rmse / max(
        fp32_prediction_relative_rmse, 1.0e-30
    )
    reasons: list[str] = []
    if not structural_passed:
        reasons.append("Pre-operation input/mask/cache/shared-interface audit failed.")
        decision = POSSIBLE_MISMATCH
    elif fp32_prediction_relative_rmse >= FP32_PERSISTENT_DISCREPANCY:
        reasons.append("FP32 prediction discrepancy remains at or above 0.5%.")
        decision = POSSIBLE_MISMATCH
    elif fp32_sudden_jumps:
        reasons.append("FP32 layerwise audit contains a meaningful >=10x jump.")
        decision = POSSIBLE_MISMATCH
    elif reduction >= REQUIRED_FP32_REDUCTION:
        strength = "at least 10x" if reduction >= STRONG_FP32_REDUCTION else "at least 5x"
        reasons.append(f"FP32 prediction relative RMSE is {strength} smaller than BF16.")
        decision = CONFIRMED
    elif reduction >= 2.0:
        reasons.append(
            "FP32 helps, but the reduction is below the registered 5x "
            "confirmation criterion."
        )
        decision = UNCERTAIN
    else:
        reasons.append("FP32 does not materially reduce the prediction discrepancy.")
        decision = POSSIBLE_MISMATCH
    return {
        "classification": decision,
        "bf16_to_fp32_prediction_error_reduction_factor": reduction,
        "reasons": reasons,
    }


def _first_meaningful_divergence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for row in sorted(rows, key=lambda x: (int(x["layer"]), LAYER_TENSORS.index(str(x["tensor"])))):
        if row["dtype"] != "torch.float32":
            continue
        if float(row["relative_rmse"]) >= SUDDEN_JUMP_MINIMUM:
            return {
                "layer": int(row["layer"]),
                "tensor": str(row["tensor"]),
                "fp32_relative_rmse": float(row["relative_rmse"]),
                "threshold_used_for_reporting": SUDDEN_JUMP_MINIMUM,
            }
    return None


def _markdown(summary: Mapping[str, Any]) -> str:
    by_dtype = summary["dtype_results"]
    bf16 = by_dtype["torch.bfloat16"]
    fp32 = by_dtype["torch.float32"]
    lines = [
        "# Fast-WAM ASRE Salvage B — Final Numerical-Equivalence Confirmation",
        "",
        "This is machinery verification only. No Phase-B condition, online episode,",
        "world-evaluation sample, rank, metric, or scientific threshold was changed or rerun.",
        "",
        "## Final machinery classification",
        "",
        f"**{summary['decision']['classification']}**",
        "",
        *[f"- {reason}" for reason in summary["decision"]["reasons"]],
        "",
        "## Dtype scaling",
        "",
        "| Dtype | Future-token rel-RMSE | Prediction rel-RMSE | "
        "Prediction max abs | Prediction RMS abs | Relative norm error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("torch.bfloat16", "torch.float32", "torch.float16"):
        if name not in by_dtype:
            continue
        result = by_dtype[name]
        token = result["future_token_output"]
        pred = result["future_prediction_output"]
        lines.append(
            f"| {name} | {token['relative_rmse']:.8g} | {pred['relative_rmse']:.8g} | "
            f"{pred['max_abs_error']:.8g} | {pred['rms_absolute_error']:.8g} | "
            f"{pred['relative_norm_error']:.8g} |"
        )
    reduction = summary["decision"][
        "bf16_to_fp32_prediction_error_reduction_factor"
    ]
    lines.extend(
        [
            "",
            f"BF16-to-FP32 final-prediction error reduction: **{reduction:.2f}×**.",
            "",
            "## Layer-by-layer divergence",
            "",
            "The full table (input/prefix hidden state, Q/K/V, attention output and block output)",
            "is in `layerwise_equivalence.csv`. The compact block-output view is:",
            "",
            "| Layer | BF16 rel-RMSE | FP32 rel-RMSE |",
            "|---:|---:|---:|",
        ]
    )
    def layer_metric(result: Mapping[str, Any], layer: int) -> float:
        return float(
            result["layers"][layer]["comparisons"]
            ["future_token_hidden_output"]["relative_rmse"]
        )
    for layer in range(len(bf16["layers"])):
        lines.append(
            f"| {layer} | {layer_metric(bf16, layer):.8g} | {layer_metric(fp32, layer):.8g} |"
        )
    first = summary.get("first_meaningful_divergence")
    lines.extend(
        [
            "",
            "## Structural audits",
            "",
            "- Frozen pre-operation inputs identical: "
            f"`{summary['structural_audit']['preoperation_inputs_identical']}`.",
            f"- Mask/SDPA graph equivalent: `{summary['mask_equivalence']['passed']}`.",
            "- Same cache list and all 30 K/V tensor objects reached both "
            f"consumers: `{summary['cache_shared_interface']['passed']}`.",
            "- No raw-RGB/prefix residual argument or bypass: "
            f"`{summary['structural_audit']['no_raw_prefix_bypass']}`.",
            f"- First meaningful FP32 divergence: `{first}`.",
            "",
            "## Numerical-budget assessment",
            "",
            "The existing `max(0.2%, 2 × execution-dtype epsilon)` gate is "
            f"classified **{summary['budget_assessment']['classification']}**.",
            "",
            "## Frozen provenance",
            "",
            f"- Successful source run commit: `{summary['source']['source_commit']}`.",
            f"- Numerical-audit code commit: `{summary['audit_commit']}`.",
            f"- Frozen sample/draw: `{summary['sample_id']}` / `{summary['draw_id']}`.",
            "- Exact hashes are in `fixed_input_manifest.json`.",
            "",
            "## Exact code changes",
            "",
            *[f"- {change}" for change in summary["code_changes"]],
            "",
            "## Tests",
            "",
            f"- Salvage-B tests: `{summary['tests']['salvage_b']['passed_count']} passed`.",
            f"- Full ASRE tests: `{summary['tests']['full_asre']['passed_count']} passed`.",
            "",
            "## Scientific-result status",
            "",
            summary["scientific_result_status"],
            "",
            "Stop here. No new ASRE experiment was launched.",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    completion = output_root / "final_decision.json"
    if completion.exists():
        raise FileExistsError(
            f"Refusing to overwrite a completed numerical audit: {completion}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    if not _worktree_clean():
        raise ValueError("Numerical-equivalence verification requires a clean worktree.")
    if not torch.cuda.is_available():
        raise RuntimeError("Numerical-equivalence verification requires one CUDA GPU.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Numerical-equivalence verification forbids DDP.")

    _, source_bundle = _source_namespace(args.source_root)
    frozen = source_bundle["frozen"]
    audit_commit = git_commit(PROJECT_ROOT)
    set_global_seed(42, get_worker_init_fn=False)
    dataset = load_world_dataset(
        preflight=frozen["preflight"], runtime_work_dir=args.runtime_work_dir.resolve()
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
    prepared = prepare_world_sample(
        model=model,
        processed=processed,
        record=frozen["record"],
        donor_bundle=donor_bundle,
    )

    with torch.no_grad():
        draw_state = _prepare_draw_state(
            model=model, prepared=prepared, draw=frozen["draw"]
        )
        draw_state["prepared_context"] = prepared.inputs["context"]
        draw_state["prepared_context_mask"] = prepared.inputs["context_mask"]
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
            stock_tokens,
            stock_t,
            stock_t_mod,
            stock_context,
            stock_context_mask,
            stock_freqs,
            f,
            h,
            w,
            tpf,
        ) = stock_prepared
        if (int(f), int(h), int(w), int(tpf), int(stock_tokens.shape[1])) != (
            3,
            7,
            14,
            PREFIX_TOKENS,
            294,
        ):
            raise ValueError("Stock numerical-audit carrier layout drifted.")
        prefix_state = _build_prefix_state(model, prepared)

        preoperation = {
            "future_tokens": tensor_error_metrics(
                stock_tokens[:, PREFIX_TOKENS:],
                draw_state["video_tokens"][:, PREFIX_TOKENS:],
            ),
            "future_t": tensor_error_metrics(
                stock_t[:, PREFIX_TOKENS:],
                draw_state["video_t"][:, PREFIX_TOKENS:],
            ),
            "future_t_mod": tensor_error_metrics(
                stock_t_mod[:, PREFIX_TOKENS:],
                draw_state["video_t_mod"][:, PREFIX_TOKENS:],
            ),
            "future_context": tensor_error_metrics(
                stock_context, draw_state["video_context"]
            ),
            "future_context_mask": tensor_error_metrics(
                stock_context_mask[:, PREFIX_TOKENS:],
                draw_state["video_context_mask"][:, PREFIX_TOKENS:],
            ),
            "future_freqs": tensor_error_metrics(
                stock_freqs[PREFIX_TOKENS:],
                draw_state["video_freqs"][PREFIX_TOKENS:],
            ),
            "prefix_tokens": tensor_error_metrics(
                stock_tokens[:, :PREFIX_TOKENS], prefix_state["tokens"]
            ),
            "prefix_t_mod": tensor_error_metrics(
                stock_t_mod[:, :PREFIX_TOKENS], prefix_state["t_mod"]
            ),
            "prefix_freqs": tensor_error_metrics(
                stock_freqs[:PREFIX_TOKENS], prefix_state["freqs"]
            ),
            "prefix_context": tensor_error_metrics(
                stock_context, prefix_state["context"]
            ),
            "prefix_context_mask": tensor_error_metrics(
                stock_context_mask[:, :PREFIX_TOKENS],
                prefix_state["context_mask"],
            ),
        }
        preoperation_identical = all(
            comparison.get("exact_equal") is True for comparison in preoperation.values()
        )
        mask_equivalence = _mask_audit(draw_state)
        atomic_write_json(output_root / "mask_equivalence.json", mask_equivalence)

        fixed_input_manifest = {
            "artifact_type": "asre_salvage_b_numerical_equivalence_fixed_inputs",
            "schema_version": 1,
            "created_at": now_iso(),
            "audit_commit": audit_commit,
            "source": source_bundle["provenance"],
            "sample_id": frozen["sample_id"],
            "draw_id": int(frozen["draw"]["draw_id"]),
            "prompt": processed["prompt"],
            "tensor_inputs": {
                "current_observation": _tensor_record(processed["video"][:, :, 0]),
                "prompt_context": _tensor_record(prepared.inputs["context"]),
                "prompt_context_mask": _tensor_record(prepared.inputs["context_mask"]),
                "target_latent_provenance_only": _tensor_record(prepared.input_latents),
                "current_frame_latent": _tensor_record(prepared.current_frame_latent),
                "future_gaussian_noise": _tensor_record(draw_state["initial_future_noise"]),
                "action_gaussian_noise": _tensor_record(frozen["draw"]["action_noise"]),
                "action_timestep": _tensor_record(draw_state["action_timestep"]),
                "prepared_action_tokens": _tensor_record(draw_state["action_tokens"]),
                "video_timestep": _tensor_record(draw_state["video_timestep"]),
                "scheduler_timesteps": _tensor_record(draw_state["inference_timesteps"]),
                "scheduler_deltas": _tensor_record(draw_state["inference_deltas"]),
                "stock_input_tokens": _tensor_record(stock_tokens),
                "factorized_future_input_tokens": _tensor_record(
                    draw_state["video_tokens"][:, PREFIX_TOKENS:]
                ),
                "full_joint_mask": _tensor_record(draw_state["full_joint_mask"]),
                "video_mask": _tensor_record(draw_state["video_mask"]),
                "video_time_modulation": _tensor_record(stock_t_mod),
                "video_frequencies": _tensor_record(stock_freqs),
                "video_context": _tensor_record(stock_context),
                "video_context_mask": _tensor_record(stock_context_mask),
                "action_time_modulation": _tensor_record(draw_state["action_t_mod"]),
                "action_frequencies": _tensor_record(draw_state["action_freqs"]),
                "action_context": _tensor_record(draw_state["action_context"]),
                "action_context_mask": _tensor_record(draw_state["action_context_mask"]),
            },
            "video_cache_k": _tensor_list_record(prepared.current_cache_k),
            "video_cache_v": _tensor_list_record(prepared.current_cache_v),
            "token_order": {
                "stock": "98 current-prefix video + 196 future video + 32 action",
                "factorized_prefix": "98 current-prefix video",
                "factorized_future": "196 future video queries over 98 prefix + 196 future keys",
            },
            "preoperation_comparisons": preoperation,
            "preoperation_inputs_identical": preoperation_identical,
            "scoring_target_used": False,
            "phase_b_outcomes_inspected": False,
        }
        atomic_write_json(output_root / "fixed_input_manifest.json", fixed_input_manifest)

        captures: dict[str, Any] = {}
        restore = _install_consumer_capture(model, captures)
        try:
            _action_prediction(
                model=model,
                state=draw_state,
                cache_k=prepared.current_cache_k,
                cache_v=prepared.current_cache_v,
            )
            _factorized_future_tokens(
                model=model,
                state=draw_state,
                cache_k=prepared.current_cache_k,
                cache_v=prepared.current_cache_v,
                disabled_layers=(),
            )
        finally:
            restore()
        expected_ids = {
            "cache_k_list_id": id(prepared.current_cache_k),
            "cache_v_list_id": id(prepared.current_cache_v),
            "cache_k_tensor_ids": [id(tensor) for tensor in prepared.current_cache_k],
            "cache_v_tensor_ids": [id(tensor) for tensor in prepared.current_cache_v],
        }
        cache_shared = {
            "passed": bool(
                captures.get("action_calls") == [expected_ids]
                and captures.get("world_calls") == [expected_ids]
            ),
            "same_list_objects": bool(
                captures.get("action") == expected_ids
                and captures.get("world") == expected_ids
            ),
            "same_all_30_k_v_tensor_objects": bool(
                captures.get("action", {}).get("cache_k_tensor_ids")
                == captures.get("world", {}).get("cache_k_tensor_ids")
                and captures.get("action", {}).get("cache_v_tensor_ids")
                == captures.get("world", {}).get("cache_v_tensor_ids")
                and len(expected_ids["cache_k_tensor_ids"]) == 30
                and len(expected_ids["cache_v_tensor_ids"]) == 30
            ),
            "action_consumer_calls": len(captures.get("action_calls", [])),
            "world_consumer_calls": len(captures.get("world_calls", [])),
        }

        stock_state = {
            "stock_input_tokens": stock_tokens,
            "factorized_future_tokens": draw_state["video_tokens"][:, PREFIX_TOKENS:],
            "stock_t": stock_t,
            "factorized_t": draw_state["video_t"],
            "stock_t_mod": stock_t_mod,
            "stock_context": stock_context,
            "stock_context_mask": stock_context_mask,
            "stock_freqs": stock_freqs,
            "action_tokens": draw_state["action_tokens"],
            "action_t_mod": draw_state["action_t_mod"],
            "action_context": draw_state["action_context"],
            "action_context_mask": draw_state["action_context_mask"],
            "action_freqs": draw_state["action_freqs"],
            "full_joint_mask": draw_state["full_joint_mask"],
            "video_mask": draw_state["video_mask"],
            "f": int(f),
            "h": int(h),
            "w": int(w),
        }
        dtype_results: dict[str, Any] = {}
        layer_rows: list[dict[str, Any]] = []
        requested = [torch.bfloat16, torch.float32]
        if args.include_fp16:
            requested.append(torch.float16)
        for dtype in requested:
            result = _run_dtype(
                model=model,
                stock_state=stock_state,
                prefix_state=prefix_state,
                dtype=dtype,
            )
            layer_rows.extend(result.pop("rows"))
            dtype_results[str(dtype)] = result

    _atomic_csv(output_root / "layerwise_equivalence.csv", layer_rows)
    fp32_jumps = _sudden_jumps(layer_rows, "torch.float32")
    no_prefix_parameter = not any(
        name in inspect.signature(
            model.mot.forward_future_video_with_video_cache_tensor
        ).parameters
        for name in ("raw_rgb", "input_image", "prefix_tokens", "prefix_residual")
    )
    structural_passed = bool(
        preoperation_identical
        and mask_equivalence["passed"]
        and cache_shared["passed"]
        and no_prefix_parameter
    )
    bf16_relative = float(
        dtype_results["torch.bfloat16"]["future_prediction_output"]["relative_rmse"]
    )
    fp32_relative = float(
        dtype_results["torch.float32"]["future_prediction_output"]["relative_rmse"]
    )
    source_factorization = source_bundle["source_machinery"]["checks"][
        "stock_joint_vs_factorized_first_pure_noise_step"
    ]
    source_anchor = {
        "source_token_relative_rmse": float(
            source_factorization["token_equivalence"]["relative_rmse"]
        ),
        "reconstructed_token_relative_rmse": float(
            dtype_results["torch.bfloat16"]["future_token_output"]["relative_rmse"]
        ),
        "source_prediction_relative_rmse": float(
            source_factorization["prediction_equivalence"]["relative_rmse"]
        ),
        "reconstructed_prediction_relative_rmse": bf16_relative,
        "absolute_tolerance": 1.0e-4,
    }
    source_anchor["passed"] = bool(
        abs(
            source_anchor["source_token_relative_rmse"]
            - source_anchor["reconstructed_token_relative_rmse"]
        )
        <= source_anchor["absolute_tolerance"]
        and abs(
            source_anchor["source_prediction_relative_rmse"]
            - source_anchor["reconstructed_prediction_relative_rmse"]
        )
        <= source_anchor["absolute_tolerance"]
    )
    structural_passed = bool(structural_passed and source_anchor["passed"])
    decision = classify_numerical_equivalence(
        bf16_prediction_relative_rmse=bf16_relative,
        fp32_prediction_relative_rmse=fp32_relative,
        structural_passed=structural_passed,
        fp32_sudden_jumps=fp32_jumps,
    )
    observed_bf16_budget = max(
        machinery_tests.FACTORIZATION_RELATIVE_RMSE_FLOOR,
        machinery_tests.FACTORIZATION_DTYPE_EPS_MULTIPLIER
        * torch.finfo(torch.bfloat16).eps,
    )
    if decision["classification"] == CONFIRMED and bf16_relative <= observed_bf16_budget:
        budget_classification = "SUPPORTED"
    elif decision["classification"] == UNCERTAIN and bf16_relative <= observed_bf16_budget:
        budget_classification = "HEURISTIC-BUT-CONSERVATIVE"
    else:
        budget_classification = "NOT-SUPPORTED"
    first_divergence = _first_meaningful_divergence(layer_rows)
    tests = {
        "salvage_b": _extract_test_count(args.salvage_b_test_log),
        "full_asre": _extract_test_count(args.full_asre_test_log),
    }
    if any(value["passed_count"] is None for value in tests.values()):
        raise ValueError("Test logs do not contain a passing pytest count.")

    scientific_status = (
        "The existing Salvage-B STRONG scientific result can be retained unchanged."
        if decision["classification"] == CONFIRMED
        else (
            "The existing Salvage-B STRONG result must be quarantined pending "
            "machinery resolution."
        )
    )
    summary = {
        "artifact_type": "asre_salvage_b_final_numerical_equivalence_summary",
        "schema_version": 1,
        "created_at": now_iso(),
        "protocol": SALVAGE_B_PROTOCOL,
        "audit_commit": audit_commit,
        "source": source_bundle["provenance"],
        "code_changes": [
            "Added an independent post-publication dtype-scaling verifier.",
            "Added per-layer stock/factorized hidden, Q/K/V, attention, and block instrumentation.",
            "Added logical mask/SDPA, dynamic cache-object, and isolated FP64 attention audits.",
            "Added fail-closed classification, atomic artifacts, launcher, and CPU tests.",
            "Allowed the frozen-input validator to target an explicitly bound "
            "historical source commit; the registered driver default is unchanged.",
        ],
        "sample_id": frozen["sample_id"],
        "draw_id": int(frozen["draw"]["draw_id"]),
        "scope": {
            "machinery_samples": 1,
            "machinery_draws": 1,
            "phase_b_rerun": False,
            "online_action_rollouts": 0,
            "environment_rollouts": 0,
            "scientific_results_changed": False,
            "later_asre_launched": False,
        },
        "dtype_results": dtype_results,
        "fp16_status": "completed" if args.include_fp16 else "optional_not_requested",
        "fp32_sudden_order_of_magnitude_jumps": fp32_jumps,
        "first_meaningful_divergence": first_divergence,
        "mask_equivalence": mask_equivalence,
        "cache_shared_interface": cache_shared,
        "structural_audit": {
            "passed": structural_passed,
            "source_bf16_anchor_reproduced": source_anchor,
            "preoperation_inputs_identical": preoperation_identical,
            "preoperation_comparisons": preoperation,
            "no_raw_prefix_bypass": no_prefix_parameter,
            "future_consumer_parameters": list(
                inspect.signature(
                    model.mot.forward_future_video_with_video_cache_tensor
                ).parameters
            ),
        },
        "isolated_float64_attention_reference": dtype_results["torch.float32"].get(
            "isolated_float64_attention_reference"
        ),
        "budget_assessment": {
            "classification": budget_classification,
            "policy": "max(0.2%, 2 * execution_dtype_epsilon)",
            "bf16_budget": float(observed_bf16_budget),
            "bf16_observed_prediction_relative_rmse": bf16_relative,
            "evidence_based_on_dtype_scaling": True,
        },
        "tests": tests,
        "decision": decision,
        "scientific_result_status": scientific_status,
        "stop_rule": {
            "stopped_after_verification": True,
            "new_asre_experiment_launched": False,
        },
    }
    summary_path = output_root / "dtype_equivalence_summary.json"
    summary_md_path = output_root / "dtype_equivalence_summary.md"
    atomic_write_json(summary_path, summary)
    _atomic_text(summary_md_path, _markdown(summary))
    final = {
        "artifact_type": "asre_salvage_b_numerical_equivalence_final_decision",
        "schema_version": 1,
        "created_at": now_iso(),
        "classification": decision["classification"],
        "dtype_equivalence_summary": _artifact(summary_path),
        "dtype_equivalence_summary_markdown": _artifact(summary_md_path),
        "layerwise_equivalence": _artifact(output_root / "layerwise_equivalence.csv"),
        "mask_equivalence": _artifact(output_root / "mask_equivalence.json"),
        "fixed_input_manifest": _artifact(output_root / "fixed_input_manifest.json"),
        "scientific_result_retained_unchanged": bool(
            decision["classification"] == CONFIRMED
        ),
        "later_asre_launched": False,
    }
    atomic_write_json(completion, final)
    print(f"Salvage-B numerical classification: {decision['classification']}")
    print(f"Final report: {summary_md_path}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-work-dir", type=Path, required=True)
    parser.add_argument("--salvage-b-test-log", type=Path, required=True)
    parser.add_argument("--full-asre-test-log", type=Path, required=True)
    parser.add_argument("--include-fp16", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
