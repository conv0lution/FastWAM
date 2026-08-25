"""Checkpoint-level correctness checks for the video-K/V intervention."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_eval_device,
)
from experiments.asre_diagnosis.common import atomic_write_json, get_num_model_layers, load_manifest
from fastwam.utils.pytorch_utils import set_global_seed


def _infer(
    model: torch.nn.Module,
    infer_kwargs: dict,
    compile_action_infer: bool,
    disabled_video_layers: tuple[int, ...] | None,
) -> torch.Tensor:
    kwargs = dict(infer_kwargs)
    kwargs["compile_action_infer"] = compile_action_infer
    if disabled_video_layers is not None:
        kwargs["disabled_video_layers"] = disabled_video_layers
    return model.infer_action(**kwargs)["action"].detach().float().cpu()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def smoke_test(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    state_bank_value = cfg.ASRE_DIAGNOSIS.get("state_bank_dir")
    if state_bank_value is None:
        raise ValueError("Set ASRE_DIAGNOSIS.state_bank_dir to a collected state bank.")
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    state_bank_dir = Path(os.path.expanduser(os.path.expandvars(str(state_bank_value)))).resolve()
    manifest = load_manifest(state_bank_dir / "manifest.jsonl")
    if not manifest:
        raise ValueError(f"No samples found in {state_bank_dir}.")
    sample = torch.load(state_bank_dir / manifest[0]["sample_path"], weights_only=False)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    # The stored sample already contains context/context_mask; T5 is unused.
    cfg.model.load_text_encoder = False
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    num_layers = get_num_model_layers(model)
    compile_action_infer = bool(cfg.EVALUATION.get("compile_action_infer", False))
    infer_kwargs = sample["infer_action_kwargs"]

    original = _infer(model, infer_kwargs, compile_action_infer, None)
    empty_diagnosis = _infer(model, infer_kwargs, compile_action_infer, ())
    absolute_difference = (original - empty_diagnosis).abs()
    test_a = torch.allclose(original, empty_diagnosis, atol=1e-4, rtol=1e-4)

    middle_layer = num_layers // 2
    middle_disabled = _infer(
        model,
        infer_kwargs,
        compile_action_infer,
        (middle_layer,),
    )
    test_b = not torch.equal(original, middle_disabled)

    all_disabled = _infer(
        model,
        infer_kwargs,
        compile_action_infer,
        tuple(range(num_layers)),
    )
    test_c = all_disabled.shape == original.shape and bool(torch.isfinite(all_disabled).all())

    report = {
        "sample_id": sample["sample_id"],
        "compile_action_infer": compile_action_infer,
        "num_model_layers": num_layers,
        "test_a_empty_diagnosis_matches_original": bool(test_a),
        "test_a_max_absolute_action_difference": float(absolute_difference.max()),
        "test_a_mean_absolute_action_difference": float(absolute_difference.mean()),
        "test_b_middle_layer": middle_layer,
        "test_b_intervention_not_identical": bool(test_b),
        "test_b_max_absolute_action_difference": float((original - middle_disabled).abs().max()),
        "test_c_drop_all_valid_action": bool(test_c),
        "test_c_action_shape": list(all_disabled.shape),
    }
    report_value = cfg.ASRE_DIAGNOSIS.get("smoke_report_path")
    report_path = (
        state_bank_dir / "smoke_test_report.json"
        if report_value is None
        else Path(os.path.expanduser(os.path.expandvars(str(report_value)))).resolve()
    )
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))
    print(f"Smoke-test report: {report_path}")
    if not test_a:
        raise AssertionError("Test A failed: enabled diagnosis with [] changed the action output.")
    if not test_b:
        raise AssertionError("Test B failed: middle-layer intervention produced an identical tensor.")
    if not test_c:
        raise AssertionError("Test C failed: drop-all did not produce a finite, valid action tensor.")


if __name__ == "__main__":
    smoke_test()
