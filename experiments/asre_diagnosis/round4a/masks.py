from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.asre_diagnosis.common import ROUND4A_PROTOCOL, sha256_file


LATE_LAYERS = tuple(range(15, 30))
MASK_SEEDS = (1, 2, 3)
ROUNDING_RULE = "ceil(n/2)"
SELECTION_ALGORITHM = "sha256_rank_v1"


@dataclass(frozen=True)
class Round4AMaskSpec:
    condition_name: str
    axis: str
    seed: int
    retained_current_token_indices: tuple[int, ...] | None = None
    retained_current_heads_by_layer: Mapping[int, tuple[int, ...]] | None = None
    runtime_layout: Mapping[str, Any] | None = None

    def inference_kwargs(self) -> dict[str, Any]:
        layout_kwargs = {"expected_video_cache_layout": self.runtime_layout}
        if self.axis == "token":
            return layout_kwargs | {
                "retained_current_video_token_indices":
                    self.retained_current_token_indices
            }
        if self.axis == "head":
            return layout_kwargs | {
                "retained_current_video_heads_by_layer":
                    self.retained_current_heads_by_layer
            }
        raise ValueError(f"Unsupported Round-4A mask axis: {self.axis!r}.")


def _half_count(count: int) -> int:
    if count <= 0:
        raise ValueError(f"Cannot construct a 50% mask over {count} candidates.")
    return (count + 1) // 2


def _select_half(
    candidates: Sequence[int],
    *,
    axis: str,
    seed: int,
    layer: int | None = None,
) -> list[int]:
    unique = sorted(set(int(candidate) for candidate in candidates))
    if len(unique) != len(candidates):
        raise ValueError("Mask candidates must be unique.")
    ranked = sorted(
        unique,
        key=lambda candidate: hashlib.sha256(
            (
                f"asre-round4a|axis={axis}|seed={seed}|layer={layer}|"
                f"index={candidate}"
            ).encode("utf-8")
        ).digest(),
    )
    return sorted(ranked[: _half_count(len(unique))])


def _select_count(
    candidates: Sequence[int],
    *,
    count: int,
    axis: str,
    seed: int,
    layer: int | None = None,
) -> list[int]:
    unique = sorted(set(int(candidate) for candidate in candidates))
    if len(unique) != len(candidates) or count < 0 or count > len(unique):
        raise ValueError("Invalid deterministic mask selection population/count.")
    ranked = sorted(
        unique,
        key=lambda candidate: hashlib.sha256(
            (
                f"asre-round4a|axis={axis}|seed={seed}|layer={layer}|"
                f"index={candidate}"
            ).encode("utf-8")
        ).digest(),
    )
    return sorted(ranked[:count])


def _stratified_token_selection(
    strata: Mapping[str, Sequence[int]], *, seed: int
) -> tuple[list[int], dict[str, int]]:
    target = _half_count(sum(len(values) for values in strata.values()))
    quotas = {name: len(values) // 2 for name, values in strata.items()}
    remaining = target - sum(quotas.values())
    eligible = [name for name, values in strata.items() if len(values) % 2 == 1]
    ranked_strata = sorted(
        eligible,
        key=lambda name: hashlib.sha256(
            f"asre-round4a|axis=token|seed={seed}|quota_stratum={name}".encode(
                "utf-8"
            )
        ).digest(),
    )
    if remaining < 0 or remaining > len(ranked_strata):
        raise ValueError("Cannot allocate the registered stratified 50% token quota.")
    for name in ranked_strata[:remaining]:
        quotas[name] += 1
    selected = []
    for name in sorted(strata):
        selected.extend(
            _select_count(
                strata[name],
                count=quotas[name],
                axis=f"token_view_{name}",
                seed=seed,
            )
        )
    return sorted(selected), quotas


def runtime_layout_from_cache_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and strictly validate the model/runtime layout used to freeze masks."""
    video_seq_len = int(stats["current_video_seq_len"])
    visible = tuple(int(index) for index in stats["action_visible_video_token_indices"])
    if tuple(sorted(set(visible))) != visible:
        raise ValueError("Runtime action-visible token indices must be unique and sorted.")
    if not visible or visible[0] < 0 or visible[-1] >= video_seq_len:
        raise ValueError("Runtime action-visible token indices are empty or out of range.")
    num_heads = int(stats["num_heads"])
    head_dim = int(stats["head_dim"])
    if num_heads <= 0 or head_dim <= 0:
        raise ValueError("Runtime num_heads and head_dim must be positive.")

    layers_by_index = {int(entry["layer"]): entry for entry in stats["layers"]}
    cache_shapes: dict[str, list[int]] = {}
    for layer in LATE_LAYERS:
        entry = layers_by_index.get(layer)
        if entry is None or not bool(entry.get("summarized")):
            raise ValueError(f"Runtime cache stats do not summarize late layer {layer}.")
        k_shape = [int(value) for value in entry["current"]["k"]["shape"]]
        v_shape = [int(value) for value in entry["current"]["v"]["shape"]]
        if k_shape != v_shape or len(k_shape) != 3:
            raise ValueError(f"Runtime K/V geometry mismatch at layer {layer}.")
        if k_shape[1] != video_seq_len or k_shape[2] != num_heads * head_dim:
            raise ValueError(
                f"Runtime cache geometry at layer {layer} disagrees with token/head layout."
            )
        cache_shapes[str(layer)] = k_shape

    grid_size = tuple(int(value) for value in stats["current_video_grid_size"])
    input_shape = tuple(int(value) for value in stats["current_input_image_shape"])
    if len(grid_size) != 3 or math.prod(grid_size) != video_seq_len:
        raise ValueError("Runtime video grid does not match video sequence length.")
    view_stratified = False
    view_reason = (
        "Runtime layout does not establish equal horizontal two-camera boundaries; "
        "uniform selection over action-visible video positions is used."
    )
    view_strata: dict[str, list[int]] = {}
    f_size, h_size, w_size = grid_size
    if (
        len(input_shape) == 4
        and input_shape[0] == 1
        and input_shape[1] == 3
        and input_shape[3] == 2 * input_shape[2]
        and w_size % 2 == 0
    ):
        half_width = w_size // 2
        for view_index, (start_col, stop_col) in enumerate(
            ((0, half_width), (half_width, w_size))
        ):
            indices = []
            for frame in range(f_size):
                for row in range(h_size):
                    for column in range(start_col, stop_col):
                        index = (frame * h_size + row) * w_size + column
                        if index in set(visible):
                            indices.append(index)
            view_strata[f"view{view_index}"] = indices
        if all(view_strata.values()) and set().union(
            *(set(values) for values in view_strata.values())
        ) == set(visible):
            view_stratified = True
            view_reason = (
                "The registered two-camera pipeline horizontally concatenates equal "
                "224x224 views into 224x448, and runtime tokens are flattened in "
                "(frame,row,column) order; the grid-width midpoint is therefore a "
                "reliable camera boundary."
            )
        else:
            view_strata = {}

    return {
        "video_seq_len": video_seq_len,
        "action_visible_token_indices": list(visible),
        "action_visible_token_count": len(visible),
        "non_action_visible_token_count": video_seq_len - len(visible),
        "tokens_per_frame": int(stats["current_video_tokens_per_frame"]),
        "video_grid_size": list(grid_size),
        "input_image_shape": list(input_shape),
        "action_attention_mask_shape": [
            int(value) for value in stats["action_attention_mask_shape"]
        ],
        "num_layers": len(stats["layers"]),
        "late_layers": list(LATE_LAYERS),
        "num_heads": num_heads,
        "head_dim": head_dim,
        "cache_shapes_by_layer": cache_shapes,
        "view_stratified": view_stratified,
        "view_stratification_reason": view_reason,
        "action_visible_token_indices_by_view": view_strata,
    }


def build_mask_manifest(runtime_layout: Mapping[str, Any]) -> dict[str, Any]:
    layout = json.loads(json.dumps(runtime_layout))
    if int(layout.get("num_layers", -1)) != 30:
        raise ValueError("Round-4A masks require exactly 30 model layers.")
    if tuple(int(layer) for layer in layout.get("late_layers", ())) != LATE_LAYERS:
        raise ValueError("Round-4A runtime layout must identify late layers 15--29.")
    visible = tuple(int(index) for index in layout["action_visible_token_indices"])
    num_heads = int(layout["num_heads"])

    conditions: dict[str, Any] = {}
    for seed in MASK_SEEDS:
        retained_heads = {
            str(layer): _select_half(
                range(num_heads), axis="head", seed=seed, layer=layer
            )
            for layer in LATE_LAYERS
        }
        conditions[f"head50_seed{seed}"] = {
            "axis": "head",
            "seed": seed,
            "retained_source": "current",
            "replacement_source": "wrong_same_task_same_replan",
            "retained_current_heads_by_layer": retained_heads,
            "retained_head_count_by_layer": {
                str(layer): len(retained_heads[str(layer)]) for layer in LATE_LAYERS
            },
            "same_mask_for_k_and_v": True,
            "independent_mask_per_late_layer": True,
        }
    for seed in MASK_SEEDS:
        if bool(layout.get("view_stratified")):
            retained_tokens, quotas = _stratified_token_selection(
                {
                    str(name): tuple(int(index) for index in values)
                    for name, values in layout[
                        "action_visible_token_indices_by_view"
                    ].items()
                },
                seed=seed,
            )
        else:
            retained_tokens = _select_half(visible, axis="token", seed=seed)
            quotas = {"uniform": len(retained_tokens)}
        conditions[f"token50_seed{seed}"] = {
            "axis": "token",
            "seed": seed,
            "retained_source": "current",
            "replacement_source": "wrong_same_task_same_replan",
            "retained_current_token_indices": retained_tokens,
            "retained_current_token_count": len(retained_tokens),
            "replacement_token_count": len(visible) - len(retained_tokens),
            "same_mask_for_k_and_v": True,
            "same_mask_for_all_late_layers": True,
            "selection_population": "action_visible_video_positions_only",
            "view_stratified": bool(layout.get("view_stratified")),
            "retained_current_token_quota_by_view": quotas,
            "retained_current_token_indices_by_view": {
                str(name): sorted(set(retained_tokens).intersection(values))
                for name, values in layout.get(
                    "action_visible_token_indices_by_view", {}
                ).items()
            },
        }
    return {
        "schema_version": 1,
        "protocol": ROUND4A_PROTOCOL,
        "experiment_stage": "stage2_round4a",
        "frozen_before_outcomes": True,
        "selection_algorithm": SELECTION_ALGORITHM,
        "rounding_rule": ROUNDING_RULE,
        "mask_fraction_target": 0.5,
        "mask_seeds": list(MASK_SEEDS),
        "runtime_layout": layout,
        "conditions": conditions,
    }


def validate_mask_manifest(payload: Mapping[str, Any]) -> None:
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported Round-4A mask-manifest schema.")
    if str(payload.get("protocol")) != ROUND4A_PROTOCOL:
        raise ValueError("Round-4A mask manifest has the wrong protocol.")
    if payload.get("selection_algorithm") != SELECTION_ALGORITHM:
        raise ValueError("Round-4A mask manifest uses an unknown selection algorithm.")
    if payload.get("rounding_rule") != ROUNDING_RULE:
        raise ValueError("Round-4A mask manifest uses a different rounding rule.")
    rebuilt = build_mask_manifest(payload["runtime_layout"])
    if rebuilt != payload:
        raise ValueError(
            "Round-4A mask manifest is not the deterministic manifest implied by its "
            "runtime layout."
        )


def write_or_verify_frozen_manifest(path: Path, runtime_layout: Mapping[str, Any]) -> str:
    expected = build_mask_manifest(runtime_layout)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            observed = json.load(handle)
        if observed != expected:
            raise FileExistsError(
                f"Refusing to overwrite a different frozen Round-4A mask manifest: {path}"
            )
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    return sha256_file(path)


def write_or_verify_axis_manifests(
    combined_path: Path,
) -> dict[str, dict[str, str]]:
    """Freeze separately hashable token/head views of the combined manifest."""
    combined_path = combined_path.resolve()
    combined_sha256 = sha256_file(combined_path)
    combined = _load_manifest_cached(str(combined_path), combined_sha256)
    outputs: dict[str, dict[str, str]] = {}
    for axis in ("token", "head"):
        payload = {
            "schema_version": 1,
            "protocol": ROUND4A_PROTOCOL,
            "axis": axis,
            "parent_mask_manifest_path": str(combined_path),
            "parent_mask_manifest_sha256": combined_sha256,
            "selection_algorithm": combined["selection_algorithm"],
            "rounding_rule": combined["rounding_rule"],
            "mask_fraction_target": combined["mask_fraction_target"],
            "runtime_layout": combined["runtime_layout"],
            "conditions": {
                name: condition
                for name, condition in combined["conditions"].items()
                if condition["axis"] == axis
            },
        }
        path = combined_path.with_name(f"{axis}_mask_manifest.json")
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                observed = json.load(handle)
            if observed != payload:
                raise FileExistsError(
                    f"Refusing to overwrite a different frozen {axis} manifest: {path}"
                )
        else:
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temporary.replace(path)
        outputs[axis] = {"path": str(path), "sha256": sha256_file(path)}
    return outputs


@lru_cache(maxsize=16)
def _load_manifest_cached(path_string: str, expected_sha256: str) -> dict[str, Any]:
    path = Path(path_string).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Round-4A mask manifest is unavailable: {path}")
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Round-4A mask manifest SHA256 mismatch: "
            f"observed={observed_sha256}, expected={expected_sha256}."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_mask_manifest(payload)
    return payload


def load_mask_spec(
    *,
    path: Path,
    expected_sha256: str,
    condition_name: str,
) -> Round4AMaskSpec:
    if len(expected_sha256) != 64:
        raise ValueError("Round-4A mask manifest requires a 64-character SHA256.")
    payload = _load_manifest_cached(str(path.resolve()), expected_sha256)
    condition = payload["conditions"].get(condition_name)
    if condition is None:
        raise ValueError(
            f"Round-4A mask manifest has no entry for {condition_name!r}."
        )
    axis = str(condition["axis"])
    seed = int(condition["seed"])
    if axis == "token":
        return Round4AMaskSpec(
            condition_name=condition_name,
            axis=axis,
            seed=seed,
            retained_current_token_indices=tuple(
                int(index) for index in condition["retained_current_token_indices"]
            ),
            runtime_layout=payload["runtime_layout"],
        )
    if axis == "head":
        return Round4AMaskSpec(
            condition_name=condition_name,
            axis=axis,
            seed=seed,
            retained_current_heads_by_layer={
                int(layer): tuple(int(head) for head in heads)
                for layer, heads in condition[
                    "retained_current_heads_by_layer"
                ].items()
            },
            runtime_layout=payload["runtime_layout"],
        )
    raise ValueError(f"Round-4A manifest has unsupported axis {axis!r}.")


def load_mask_manifest(*, path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load a frozen manifest even for the two endpoint conditions."""
    if len(expected_sha256) != 64:
        raise ValueError("Round-4A mask manifest requires a 64-character SHA256.")
    return _load_manifest_cached(str(path.resolve()), expected_sha256)
