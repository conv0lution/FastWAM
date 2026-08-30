"""Freeze and validate inputs reused by the corrected Salvage-B-v2 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.asre_diagnosis.common import (
    atomic_write_json,
    git_commit,
    now_iso,
    sha256_file,
)

from .definitions import PROTOCOL


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "asre_results/salvage_b_world_action_dissociation/retry_20260830_dtypegatefix"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def create_preflight(*, output_root: Path, source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    source_preflight_path = source_root / "preflight_report.json"
    world_manifest_path = source_root / "manifests/world_evaluation_manifest.json"
    target_manifest_path = source_root / "manifests/processed_world_targets.json"
    stochastic_manifest_path = source_root / "manifests/stochastic_manifest.json"
    draw_tensors_path = source_root / "manifests/fixed_draw_tensors.pt"
    required = (
        source_preflight_path,
        world_manifest_path,
        target_manifest_path,
        stochastic_manifest_path,
        draw_tensors_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen reusable inputs are unavailable: {missing}")
    source = _read(source_preflight_path)
    world = _read(world_manifest_path)
    targets = _read(target_manifest_path)
    stochastic = _read(stochastic_manifest_path)
    if world.get("sample_count") != 100 or targets.get("sample_count") != 100:
        raise ValueError("The frozen official world set must contain exactly 100 samples.")
    if stochastic.get("draws_per_sample") != 4:
        raise ValueError("The frozen world bundle must contain exactly four draws/sample.")
    if sha256_file(draw_tensors_path) != stochastic.get("draw_tensor_sha256"):
        raise ValueError("Frozen stochastic draw tensors drifted.")
    for group, fields in (
        (source["state"], ("checkpoint_path", "dataset_stats_path", "prompt_context_cache_path")),
        (source["donors"], ("mapping_path", "manifest_path")),
        (source["basis"], ("path", "split_path")),
    ):
        for field in fields:
            path = Path(str(group[field])).resolve()
            digest_key = field.replace("_path", "_sha256") if field.endswith("_path") else "sha256"
            expected = group.get(digest_key)
            if not path.is_file() or expected is None or sha256_file(path) != expected:
                raise ValueError(f"Frozen input drifted: {path}")

    payload = {
        "artifact_type": "asre_salvage_b_v2_preflight",
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "compatible",
        "created_at": now_iso(),
        "git_commit_hash": git_commit(PROJECT_ROOT),
        "output_root": str(output_root.resolve()),
        "state": source["state"],
        "donors": source["donors"],
        "basis": source["basis"],
        "world_data": source["world_data"],
        "reused_frozen_inputs": {
            "source_preflight_path": str(source_preflight_path),
            "source_preflight_sha256": sha256_file(source_preflight_path),
            "world_manifest_path": str(world_manifest_path),
            "world_manifest_sha256": sha256_file(world_manifest_path),
            "target_manifest_path": str(target_manifest_path),
            "target_manifest_sha256": sha256_file(target_manifest_path),
            "stochastic_manifest_path": str(stochastic_manifest_path),
            "stochastic_manifest_sha256": sha256_file(stochastic_manifest_path),
            "draw_tensors_path": str(draw_tensors_path),
            "draw_tensors_sha256": sha256_file(draw_tensors_path),
        },
        "quarantine": {
            "old_factorized_results_reused": False,
            "world_samples_targets_draws_only_reused": True,
            "reason": "old scientific outputs used a non-native factorized future-only path",
        },
        "native_contract": {
            "joint_layer": "src/fastwam/models/wan22/mot.py::_forward_joint_layer",
            "prefix_rows": [0, 98],
            "intervened_layers": list(range(15, 30)),
            "disabled_layers": [],
            "forbidden_helper": "forward_future_video_with_video_cache_tensor",
            "donor_capture_semantics": (
                "donor RGB only under current prompt, proprio, noise, and scheduler"
            ),
            "basis_coordinate_gate": {
                "legacy_extractor": (
                    "FastWAM.infer_action -> MoT.prefill_video_cache_tensor"
                ),
                "native_extractor": (
                    "FastWAM.infer_joint -> MoT._forward_joint_layer"
                ),
                "frozen_state_count": 3,
                "layers": list(range(15, 30)),
                "tensor_kinds": ["k", "v"],
                "failure_policy": (
                    "stop before outcomes and fit native-stock basis on frozen "
                    "calibration split"
                ),
            },
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "preflight_report.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args()
    create_preflight(output_root=args.output_root, source_root=args.source_root)


if __name__ == "__main__":
    main()
