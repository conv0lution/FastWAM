"""Pre-projection action and world endpoint gates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.asre_diagnosis.common import atomic_write_json
from experiments.asre_diagnosis.salvage_b.statistics import (
    paired_bootstrap_ci,
    world_endpoint_gate,
)

from .definitions import BOOTSTRAP_SAMPLES, PROTOCOL


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _action(root: Path, condition: str) -> np.ndarray:
    rows: dict[tuple[int, int], float] = {}
    paths = sorted((root / "action_full" / condition / "libero_spatial").glob("gpu*_task*_results.json"))
    if len(paths) != 10:
        raise ValueError(f"Action endpoint {condition} is incomplete.")
    for path in paths:
        payload = _read(path)
        task = int(payload["task_id"])
        successes = set(map(int, payload["success_episodes"]))
        failures = set(map(int, payload["failure_episodes"]))
        if successes & failures or successes | failures != set(range(10)):
            raise ValueError(f"Malformed endpoint action result: {path}")
        for trial in range(10):
            rows[(task, trial)] = float(trial in successes)
    return np.asarray([rows[key] for key in sorted(rows)], dtype=np.float64)


def _world(root: Path, condition: str) -> tuple[list[tuple[int, int, str]], np.ndarray]:
    payload = _read(root / "world" / condition / "rows.json")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["rows"]:
        grouped[str(row["sample_id"])].append(row)
    if len(grouped) != 100 or any(len(rows) != 4 for rows in grouped.values()):
        raise ValueError(f"World endpoint {condition} is not 100 x four draws.")
    samples = sorted(grouped)
    keys = [
        (
            int(grouped[sample][0]["task_id"]),
            int(grouped[sample][0]["episode_id"]),
            sample,
        )
        for sample in samples
    ]
    values = np.asarray(
        [np.mean([float(row["loss"]) for row in grouped[sample]]) for sample in samples],
        dtype=np.float64,
    )
    return keys, values


def run(root: Path) -> dict[str, Any]:
    current_action = _action(root, "current")
    wrong_action = _action(root, "wrong")
    action_delta = current_action - wrong_action
    action_ci = paired_bootstrap_ci(
        action_delta, samples=BOOTSTRAP_SAMPLES, seed=424242
    )
    action_valid = bool(current_action.mean() >= 0.5 and action_ci[0] > 0.0)
    current_keys, current_world = _world(root, "current")
    wrong_keys, wrong_world = _world(root, "wrong")
    if current_keys != wrong_keys:
        raise ValueError("World endpoint identities are not paired.")
    world = world_endpoint_gate(current_keys, current_world, wrong_world)
    classification = (
        None
        if action_valid and world["informative"]
        else "ENDPOINT-INVALID"
        if not action_valid
        else "WORLD-ENDPOINT-UNINFORMATIVE"
    )
    payload = {
        "artifact_type": "asre_salvage_b_v2_endpoint_gate",
        "protocol": PROTOCOL,
        "passed": classification is None,
        "classification": classification,
        "action": {
            "current_success": float(current_action.mean()),
            "wrong_success": float(wrong_action.mean()),
            "current_minus_wrong": float(action_delta.mean()),
            "paired_ci": list(action_ci),
            "valid": action_valid,
        },
        "world": world,
    }
    atomic_write_json(root / "endpoint_gate.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args().output_root.resolve())


if __name__ == "__main__":
    main()
