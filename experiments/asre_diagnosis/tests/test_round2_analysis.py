from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.asre_diagnosis.round2.aggregate_results import (
    ACTION_DIMENSION_NAMES,
    BASELINE,
    DIMENSION_OFFLINE_METRIC,
    SCALAR_OFFLINE_METRICS,
    _build_planned_comparisons,
    _catastrophic_flags,
    _comparison_statistics,
    _condition_definitions,
    _load_authoritative_valid_sample_ids,
    _load_offline_records,
    _validate_online_metadata_compatibility,
    exact_mcnemar_p_value,
    holm_adjust,
    main as aggregate_main,
    task_hierarchical_bootstrap_ci,
)
from experiments.asre_diagnosis.round2.plot_results import (
    _validate_replan_rows,
    main as plot_main,
)


class Round2AnalysisTest(unittest.TestCase):
    def test_online_metadata_accepts_present_null_sigma_shift(self) -> None:
        condition_order, definitions = _condition_definitions()
        common = {
            "git_commit_hash": "1" * 40,
            "checkpoint_path": "/checkpoint.pt",
            "checkpoint_sha256": "a" * 64,
            "dataset_stats_path": "/dataset_stats.json",
            "dataset_stats_sha256": "b" * 64,
            "state_bank_manifest_path": "/manifest.jsonl",
            "state_bank_manifest_sha256": "c" * 64,
            "valid_state_bank_manifest_path": "/valid.json",
            "valid_state_bank_manifest_sha256": "d" * 64,
            "prompt_context_cache_path": "/prompt.pt",
            "prompt_context_cache_sha256": "e" * 64,
            "task_suite": "libero_spatial",
            "task_ids": list(range(10)),
            "seed": 42,
            "number_of_trials": 10,
            "action_horizon": 32,
            "number_of_inference_steps": 10,
            "replan_steps": 10,
            "compile_action_infer": True,
            "binarize_gripper": True,
            "sigma_shift": None,
            "rand_device": "cpu",
            "text_conditioning_source": "round1_state_bank_prompt_context_cache",
            "prompt_template": "task={task}",
            "environment_seed": 42,
            "action_inference_seed": 42,
            "action_noise_seed": 42,
            "gpu_model": "NVIDIA RTX A5000",
            "torch_version": "test",
            "cuda_version": "test",
            "condition_protocol": "round2_keep_schedules",
            "condition_config": {"mode": "drop_video_kv"},
            "config_sha256": "f" * 64,
            "start_timestamp": "2026-01-01T00:00:00+00:00",
            "end_timestamp": "2026-01-01T00:01:00+00:00",
            "status": "completed",
        }
        metadata = {
            condition: {
                **common,
                "enabled_video_retrieval_layers": definitions[condition][
                    "enabled_video_retrieval_layers"
                ],
                "disabled_video_layers": definitions[condition][
                    "disabled_video_layers"
                ],
            }
            for condition in condition_order
        }
        _validate_online_metadata_compatibility(metadata, condition_order)
        del metadata["keep_15_29"]["sigma_shift"]
        with self.assertRaisesRegex(ValueError, "sigma_shift"):
            _validate_online_metadata_compatibility(metadata, condition_order)

    def test_exact_mcnemar_and_holm_adjustment(self) -> None:
        self.assertAlmostEqual(exact_mcnemar_p_value(3, 0), 0.25)
        self.assertEqual(exact_mcnemar_p_value(0, 0), 1.0)
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)

    def test_paired_transition_counts_use_episode_identity(self) -> None:
        keys = [
            ("libero_spatial", task_id, episode_id)
            for task_id in range(2)
            for episode_id in range(2)
        ]
        reference = dict(zip(keys, [1, 1, 0, 0]))
        target = dict(zip(keys, [1, 0, 1, 0]))
        stats = _comparison_statistics(
            reference,
            target,
            bootstrap_samples=100,
            bootstrap_seed=17,
            seed_label="transition-test",
        )
        self.assertEqual(stats["reference_success_to_target_success"], 1)
        self.assertEqual(stats["reference_success_to_target_failure"], 1)
        self.assertEqual(stats["reference_failure_to_target_success"], 1)
        self.assertEqual(stats["reference_failure_to_target_failure"], 1)
        self.assertEqual(stats["paired_delta_success_rate"], 0.0)
        self.assertEqual(stats["mcnemar_exact_p_value"], 1.0)

    def test_task_hierarchical_bootstrap_is_seed_reproducible(self) -> None:
        keys = [
            ("libero_spatial", task_id, episode_id)
            for task_id in range(3)
            for episode_id in range(4)
        ]
        differences = [
            value
            for value in (-1.0, 0.0, 1.0)
            for _episode_id in range(4)
        ]
        first = task_hierarchical_bootstrap_ci(
            keys, differences, samples=250, seed=123
        )
        second = task_hierarchical_bootstrap_ci(
            keys, differences, samples=250, seed=123
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], 0.0)
        self.assertGreaterEqual(first[1], 0.0)

    def test_catastrophic_task_thresholds_include_exact_boundary(self) -> None:
        condition_order, _ = _condition_definitions()
        rows = [
            {
                "condition": "keep_15_29",
                "task_id": 0,
                "baseline_success_rate": 0.80,
                "delta_success_rate": -0.50,
            },
            {
                "condition": "keep_15_29",
                "task_id": 1,
                "baseline_success_rate": 0.70,
                "delta_success_rate": -0.60,
            },
            {
                "condition": "keep_20_29",
                "task_id": 2,
                "baseline_success_rate": 0.90,
                "delta_success_rate": -0.40,
            },
        ]
        flags, task_ids = _catastrophic_flags(
            rows,
            condition_order,
            baseline_rate_threshold=0.80,
            delta_threshold=-0.50,
        )
        self.assertTrue(flags["keep_15_29"])
        self.assertEqual(task_ids["keep_15_29"], [0])
        self.assertFalse(flags["keep_20_29"])

    def test_planned_comparison_matrix_includes_direct_window_tests(self) -> None:
        condition_order, _ = _condition_definitions()
        key = ("libero_spatial", 0, 0)
        outcomes = {condition: {key: 1} for condition in condition_order}
        rows = _build_planned_comparisons(
            outcomes,
            bootstrap_samples=10,
            bootstrap_seed=0,
            baseline_holm={
                condition: 1.0
                for condition in condition_order
                if condition != BASELINE
            },
        )
        comparisons = {row["comparison_id"] for row in rows}
        self.assertEqual(
            comparisons,
            {
                "A",
                "B1",
                "B2",
                "B3",
                "C",
                "D",
                "E1",
                "E2",
                "E3",
                "E4",
                "E5",
            },
        )

    def test_valid_manifest_is_hash_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state_bank_valid_manifest.json"
            path.write_text(json.dumps({"valid_sample_ids": ["a", "b"]}), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata = {
                "baseline_round2": {
                    "valid_state_bank_manifest_path": str(path),
                    "valid_state_bank_manifest_sha256": digest,
                }
            }
            identifiers, observed_path, observed_digest = (
                _load_authoritative_valid_sample_ids(metadata)
            )
            self.assertEqual(identifiers, ["a", "b"])
            self.assertEqual(observed_path, path.resolve())
            self.assertEqual(observed_digest, digest)

    def test_offline_loader_keeps_physical_and_round1_full_rms_distinct(self) -> None:
        condition_order, definitions = _condition_definitions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_manifest = root / "state_bank_valid_manifest.json"
            valid_manifest.write_text("{}\n", encoding="utf-8")
            valid_digest = hashlib.sha256(valid_manifest.read_bytes()).hexdigest()
            shared = {
                "git_commit_hash": "1" * 40,
                "checkpoint_path": "/checkpoint.pt",
                "checkpoint_sha256": "a" * 64,
                "dataset_stats_path": "/dataset_stats.json",
                "dataset_stats_sha256": "b" * 64,
                "state_bank_manifest_path": "/manifest.jsonl",
                "state_bank_manifest_sha256": "c" * 64,
                "valid_state_bank_manifest_path": str(valid_manifest.resolve()),
                "valid_state_bank_manifest_sha256": valid_digest,
                "prompt_context_cache_path": "/prompt.pt",
                "prompt_context_cache_sha256": "d" * 64,
                "action_horizon": 32,
                "number_of_inference_steps": 10,
                "replan_steps": 10,
                "num_model_layers": 30,
                "task_suite": "libero_spatial",
                "task_ids": [0],
                "seed": 42,
                "number_of_trials": 1,
                "compile_action_infer": False,
                "binarize_gripper": True,
                "sigma_shift": 5.0,
                "rand_device": "cpu",
                "torch_version": "test",
                "cuda_version": "test",
            }
            online_metadata = {
                condition: dict(shared) for condition in condition_order
            }
            for condition_index, condition in enumerate(condition_order):
                condition_dir = root / "offline" / condition
                condition_dir.mkdir(parents=True)
                scalar_values = {metric: 0.1 for metric in SCALAR_OFFLINE_METRICS}
                scalar_values["full_chunk_norm_rms_0_31"] = 2.0
                scalar_values["round1_raw_output_full_chunk_rms"] = 3.0
                dimension_values = [float(condition_index + offset) for offset in range(6)]
                record = {
                    "sample_id": "sample-0",
                    "task_suite": "libero_spatial",
                    "task_id": 0,
                    "episode_id": 0,
                    "replan_id": 0,
                    "condition": condition,
                    "enabled_video_retrieval_layers": definitions[condition][
                        "enabled_video_retrieval_layers"
                    ],
                    "disabled_video_layers": definitions[condition][
                        "disabled_video_layers"
                    ],
                    "executed_prefix_length": 10,
                    "action_dimension_names": list(ACTION_DIMENSION_NAMES),
                    DIMENSION_OFFLINE_METRIC: dimension_values,
                    **scalar_values,
                }
                (condition_dir / "per_sample.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
                summary = {
                    "condition": condition,
                    "num_samples": 1,
                    "executed_prefix_length": 10,
                    DIMENSION_OFFLINE_METRIC: json.dumps(dimension_values),
                    **scalar_values,
                }
                with (condition_dir / "summary.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(summary))
                    writer.writeheader()
                    writer.writerow(summary)
                metadata = {
                    **shared,
                    "status": "complete",
                    "condition_protocol": "round2_keep_schedules",
                    "diagnosis_condition": condition,
                    "enabled_video_retrieval_layers": definitions[condition][
                        "enabled_video_retrieval_layers"
                    ],
                    "disabled_video_layers": definitions[condition][
                        "disabled_video_layers"
                    ],
                    "valid_state_bank_manifest_path": str(valid_manifest.resolve()),
                    "valid_state_bank_manifest_sha256": valid_digest,
                    "num_valid_samples": 1,
                    "num_samples": 1,
                    "action_global_std": [1.0] * 7,
                }
                (condition_dir / "run_metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )

            records, _ = _load_offline_records(
                root / "offline",
                condition_order,
                definitions,
                expected_sample_ids=["sample-0"],
                valid_manifest_path=valid_manifest.resolve(),
                valid_manifest_sha256=valid_digest,
                online_metadata=online_metadata,
                expected_action_global_std=[1.0] * 7,
            )
            baseline = records["baseline_round2"][0]
            self.assertEqual(baseline["full_chunk_norm_rms_0_31"], 2.0)
            self.assertEqual(baseline["round1_raw_output_full_chunk_rms"], 3.0)

    def test_replan_plot_requires_all_five_stages_for_all_conditions(self) -> None:
        condition_order, _ = _condition_definitions()
        rows = [
            {
                "condition": condition,
                "replan_id": str(replan_id),
                "executed_prefix_norm_rms": "0.0",
                "episode_cluster_ci_low": "0.0",
                "episode_cluster_ci_high": "0.0",
            }
            for condition in condition_order
            for replan_id in range(5)
        ]
        self.assertEqual(len(_validate_replan_rows(rows, condition_order)), 40)
        with self.assertRaises(ValueError):
            _validate_replan_rows(rows[:-1], condition_order)

    def test_synthetic_eight_condition_aggregation_and_plotting(self) -> None:
        condition_order, definitions = _condition_definitions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            online_root = root / "online"
            offline_root = root / "offline"
            aggregate_root = root / "aggregate"
            valid_ids = [f"sample-{replan_id}" for replan_id in range(5)]
            valid_manifest = root / "state_bank_valid_manifest.json"
            valid_manifest.write_text(
                json.dumps({"valid_sample_ids": valid_ids}) + "\n", encoding="utf-8"
            )
            valid_digest = hashlib.sha256(valid_manifest.read_bytes()).hexdigest()
            dataset_stats = root / "dataset_stats.json"
            dataset_stats.write_text(
                json.dumps(
                    {"action": {"default": {"global_std": [1.0] * 7}}}
                )
                + "\n",
                encoding="utf-8",
            )
            dataset_stats_digest = hashlib.sha256(dataset_stats.read_bytes()).hexdigest()

            common_online = {
                "git_commit_hash": "1" * 40,
                "checkpoint_path": "/checkpoint.pt",
                "checkpoint_sha256": "a" * 64,
                "dataset_stats_path": str(dataset_stats.resolve()),
                "dataset_stats_sha256": dataset_stats_digest,
                "state_bank_manifest_path": "/manifest.jsonl",
                "state_bank_manifest_sha256": "c" * 64,
                "valid_state_bank_manifest_path": str(valid_manifest.resolve()),
                "valid_state_bank_manifest_sha256": valid_digest,
                "prompt_context_cache_path": "/prompt_context_cache.pt",
                "prompt_context_cache_sha256": "d" * 64,
                "task_suite": "libero_spatial",
                "task_ids": [0],
                "seed": 42,
                "number_of_trials": 2,
                "action_horizon": 32,
                "number_of_inference_steps": 10,
                "replan_steps": 10,
                "compile_action_infer": False,
                "binarize_gripper": True,
                "sigma_shift": 5.0,
                "rand_device": "cpu",
                "text_conditioning_source": "round1_state_bank_prompt_context_cache",
                "prompt_template": "task={task}",
                "environment_seed": 42,
                "action_inference_seed": 42,
                "action_noise_seed": 42,
                "gpu_model": "NVIDIA RTX A5000",
                "torch_version": "test",
                "cuda_version": "test",
                "condition_protocol": "round2_keep_schedules",
                "start_timestamp": "2026-01-01T00:00:00+00:00",
                "end_timestamp": "2026-01-01T00:01:00+00:00",
                "status": "completed",
            }
            for condition_index, condition in enumerate(condition_order):
                definition = definitions[condition]
                online_dir = online_root / condition
                task_dir = online_dir / "libero_spatial"
                task_dir.mkdir(parents=True)
                metadata = {
                    **common_online,
                    "diagnosis_condition": condition,
                    "enabled_video_retrieval_layers": definition[
                        "enabled_video_retrieval_layers"
                    ],
                    "disabled_video_layers": definition["disabled_video_layers"],
                    "condition_config": {"name": condition},
                    "config_sha256": f"{condition_index:x}" * 64,
                }
                (online_dir / "run_metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                successes = [0, 1] if condition_index < 2 else [0]
                task_result = {
                    "task_suite": "libero_spatial",
                    "task_id": 0,
                    "task_description": "synthetic spatial task",
                    "total_episodes": 2,
                    "successes": len(successes),
                    "success_episodes": successes,
                    "failure_episodes": [episode for episode in range(2) if episode not in successes],
                    "diagnosis_condition": condition,
                    "disabled_video_layers": definition["disabled_video_layers"],
                }
                (task_dir / "gpu0_task0_results.json").write_text(
                    json.dumps(task_result), encoding="utf-8"
                )

                offline_dir = offline_root / condition
                offline_dir.mkdir(parents=True)
                scalar_values = {
                    metric: float(condition_index) / 10.0
                    for metric in SCALAR_OFFLINE_METRICS
                }
                scalar_values["executed_prefix_cosine_similarity"] = 1.0
                scalar_values["full_chunk_norm_rms_0_31"] = 2.0 + condition_index
                scalar_values["round1_raw_output_full_chunk_rms"] = 3.0 + condition_index
                dimension_values = [float(condition_index) / 10.0] * 6
                records = []
                for replan_id, sample_id in enumerate(valid_ids):
                    records.append(
                        {
                            "sample_id": sample_id,
                            "task_suite": "libero_spatial",
                            "task_id": 0,
                            "episode_id": 0,
                            "replan_id": replan_id,
                            "condition": condition,
                            "enabled_video_retrieval_layers": definition[
                                "enabled_video_retrieval_layers"
                            ],
                            "disabled_video_layers": definition["disabled_video_layers"],
                            "executed_prefix_length": 10,
                            "action_dimension_names": list(ACTION_DIMENSION_NAMES),
                            DIMENSION_OFFLINE_METRIC: dimension_values,
                            **scalar_values,
                        }
                    )
                (offline_dir / "per_sample.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                summary = {
                    "condition": condition,
                    "num_samples": len(records),
                    "executed_prefix_length": 10,
                    DIMENSION_OFFLINE_METRIC: json.dumps(dimension_values),
                    **scalar_values,
                }
                with (offline_dir / "summary.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(summary))
                    writer.writeheader()
                    writer.writerow(summary)
                offline_metadata = {
                    **metadata,
                    "status": "complete",
                    "condition_protocol": "round2_keep_schedules",
                    "diagnosis_condition": condition,
                    "enabled_video_retrieval_layers": definition[
                        "enabled_video_retrieval_layers"
                    ],
                    "disabled_video_layers": definition["disabled_video_layers"],
                    "valid_state_bank_manifest_path": str(valid_manifest.resolve()),
                    "valid_state_bank_manifest_sha256": valid_digest,
                    "num_valid_samples": len(records),
                    "num_samples": len(records),
                    "action_global_std": [1.0] * 7,
                }
                (offline_dir / "run_metadata.json").write_text(
                    json.dumps(offline_metadata), encoding="utf-8"
                )

            aggregate_argv = [
                "aggregate_results.py",
                "--online-root",
                str(online_root),
                "--offline-root",
                str(offline_root),
                "--output-dir",
                str(aggregate_root),
                "--bootstrap-samples",
                "20",
                "--expected-online-episodes",
                "2",
                "--expected-tasks",
                "1",
                "--expected-trials-per-task",
                "2",
            ]
            with mock.patch.object(sys, "argv", aggregate_argv):
                aggregate_main()
            with mock.patch.object(
                sys,
                "argv",
                ["plot_results.py", "--aggregate-dir", str(aggregate_root), "--dpi", "40"],
            ):
                plot_main()

            self.assertTrue((aggregate_root / "summary.csv").is_file())
            self.assertTrue((aggregate_root / "diagnostic_summary.json").is_file())
            plot_files = sorted((aggregate_root / "plots").glob("*.png"))
            self.assertEqual(len(plot_files), 6)


if __name__ == "__main__":
    unittest.main()
