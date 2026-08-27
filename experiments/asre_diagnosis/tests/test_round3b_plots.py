from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.asre_diagnosis.round3b.plot_results import generate_plots


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_generates_exact_five_round3b_figures(tmp_path: Path) -> None:
    conditions = (
        ("late_current_correct", "Correct Current K/V", 0.9),
        ("late_wrong_scene", "Wrong Same-Task Scene K/V", 0.4),
        ("late_no_video", "No Video K/V", 0.0),
    )
    _write_csv(
        tmp_path / "online_condition_summary.csv",
        [
            {
                "condition": condition,
                "display_name": display,
                "successes": int(rate * 100),
                "episodes": 100,
                "success_rate": rate,
                "paired_ci_low": max(0, rate - 0.1),
                "paired_ci_high": min(1, rate + 0.1),
                "task_hierarchical_ci_low": max(0, rate - 0.2),
                "task_hierarchical_ci_high": min(1, rate + 0.2),
            }
            for condition, display, rate in conditions
        ],
    )
    _write_csv(
        tmp_path / "task_success.csv",
        [
            {
                "task_id": task,
                "task_description": f"task {task}",
                "late_current_correct": 0.9,
                "late_wrong_scene": task / 10,
                "late_no_video": 0.0,
                "wrong_minus_correct": task / 10 - 0.9,
                "no_video_minus_correct": -0.9,
                "wrong_minus_no_video": task / 10,
            }
            for task in range(10)
        ],
    )
    _write_csv(
        tmp_path / "offline_condition_metrics.csv",
        [
            {
                "condition": condition,
                "display_name": display,
                "num_samples": 499,
                "executed_prefix_norm_rms": 0.1 + index,
                "full_chunk_norm_rms_0_31": 0.2 + index,
                "translation_norm_rms": 0.3 + index,
                "rotation_norm_rms": 0.4 + index,
                "executed_prefix_cosine_similarity": 0.9 - index * 0.2,
                "executed_prefix_gripper_flip_rate": 0.05 + index * 0.1,
                "full_horizon_gripper_flip_rate": 0.05 + index * 0.1,
            }
            for index, (condition, display, _rate) in enumerate(conditions)
        ],
    )
    _write_csv(
        tmp_path / "kv_scale_sanity.csv",
        [
            {
                "layer": layer,
                "tensor": tensor,
                "shape": "[1,2,3]",
                "dtype": "torch.bfloat16",
                "num_samples": 499,
                "current_mean": 0.0,
                "donor_mean": 0.0,
                "donor_current_mean_ratio": "",
                "current_std": 1.0,
                "donor_std": 1.1,
                "donor_current_std_ratio": 1.1,
                "current_rms": 1.0 + layer / 100,
                "donor_rms": 1.1 + layer / 100,
                "donor_current_rms_ratio": 1.1,
            }
            for layer in range(15, 30)
            for tensor in ("K", "V")
        ],
    )
    (tmp_path / "round3b_summary.json").write_text(
        json.dumps(
            {
                "analysis": {
                    "comparisons": {
                        "wrong_minus_correct": {
                            "reference_failure_to_target_failure": 10,
                            "reference_failure_to_target_success": 0,
                            "reference_success_to_target_failure": 50,
                            "reference_success_to_target_success": 40,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    outputs = generate_plots(tmp_path)
    assert len(outputs) == 5
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
