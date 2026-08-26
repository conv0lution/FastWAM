from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.asre_diagnosis.round2.metrics import CONTINUOUS_ACTION_DIMENSIONS
from experiments.asre_diagnosis.round3a import aggregate_factorial as aggregate_module
from experiments.asre_diagnosis.round3a.factorial import CELL_ORDER
from experiments.asre_diagnosis.round3a.plot_factorial import _load_plot_inputs


def _online_fixture():
    keys = [
        ("libero_spatial", task_id, trial_id)
        for task_id in range(10)
        for trial_id in range(10)
    ]
    successes_per_task = (0, 0, 1, 5, 1, 9, 5, 10)
    outcomes = np.zeros((100, 8), dtype=np.int8)
    task_outcomes: dict[str, dict[int, list[int]]] = {}
    metadata: dict[str, dict[str, object]] = {}
    provenance: dict[str, dict[str, object]] = {}
    for cell_index, code in enumerate(CELL_ORDER):
        by_task: dict[int, list[int]] = {}
        for task_id in range(10):
            values = [
                int(trial_id < successes_per_task[cell_index])
                for trial_id in range(10)
            ]
            by_task[task_id] = values
            outcomes[task_id * 10 : (task_id + 1) * 10, cell_index] = values
        task_outcomes[code] = by_task
        metadata[code] = {
            "checkpoint_path": "/checkpoint.pt",
            "checkpoint_sha256": "a" * 64,
            "dataset_stats_path": "/stats.json",
            "dataset_stats_sha256": "b" * 64,
            "prompt_context_cache_sha256": "c" * 64,
        }
        provenance[code] = {"online_fixture": code}
    descriptions = {task_id: f"synthetic task {task_id}" for task_id in range(10)}
    return keys, outcomes, task_outcomes, descriptions, metadata, provenance


def _offline_fixture(valid_ids: list[str]):
    records_by_code: dict[str, list[dict[str, object]]] = {}
    summaries_by_code: dict[str, dict[str, object]] = {}
    provenance_by_code: dict[str, dict[str, object]] = {}
    for cell_index, code in enumerate(CELL_ORDER):
        metric_value = 0.01 * (cell_index + 1)
        records: list[dict[str, object]] = []
        for sample_index, sample_id in enumerate(valid_ids):
            record: dict[str, object] = {
                "sample_id": sample_id,
                "task_suite": "libero_spatial",
                "task_id": sample_index // 50,
                "episode_id": (sample_index // 5) % 10,
                "replan_id": sample_index % 5,
                "executed_prefix_norm_rms_by_dimension": [metric_value] * len(
                    CONTINUOUS_ACTION_DIMENSIONS
                ),
            }
            record.update(
                {
                    metric: metric_value
                    for metric in aggregate_module.SCALAR_OFFLINE_METRICS
                }
            )
            records.append(record)
        summary: dict[str, object] = {
            metric: metric_value
            for metric in aggregate_module.SCALAR_OFFLINE_METRICS
        }
        summary[aggregate_module.DIMENSION_OFFLINE_METRIC] = [metric_value] * len(
            CONTINUOUS_ACTION_DIMENSIONS
        )
        records_by_code[code] = records
        summaries_by_code[code] = summary
        provenance_by_code[code] = {"offline_fixture": code}
    return records_by_code, summaries_by_code, provenance_by_code


class Round3AAggregateIntegrationTest(unittest.TestCase):
    def test_output_cannot_be_created_inside_frozen_round2_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "asre_results"
            round2 = root / "round2"
            round3a = root / "round3a"
            r2_online = round2 / "online_full"
            r2_offline = round2 / "offline"
            r3_online = round3a / "online_full"
            r3_offline = round3a / "offline"
            for path in (r2_online, r2_offline, r3_online, r3_offline):
                path.mkdir(parents=True)
            state_bank = root / "state_bank"
            state_bank.mkdir()
            source_manifest = state_bank / "manifest.jsonl"
            source_manifest.write_text('{}\n', encoding="utf-8")
            valid_manifest = round2 / "state_bank_valid_manifest.json"
            valid_manifest.write_text(
                json.dumps({"source_manifest_path": str(source_manifest)}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "overlaps frozen Round-1/2"):
                aggregate_module.aggregate(
                    round2_online_root=r2_online,
                    round2_offline_root=r2_offline,
                    round3a_online_root=r3_online,
                    round3a_offline_root=r3_offline,
                    valid_manifest_path=valid_manifest,
                    output_dir=round2 / "round3a_aggregate",
                    bootstrap_samples=1,
                    sign_flip_samples=1,
                )

    def test_complete_mock_pipeline_writes_machine_readable_contract(self) -> None:
        valid_ids = [f"sample_{index:03d}" for index in range(499)]
        online = _online_fixture()
        offline = _offline_fixture(valid_ids)
        valid_payload = {
            "checkpoint_sha256": "a" * 64,
            "dataset_stats_sha256": "b" * 64,
            "source_manifest_sha256": "d" * 64,
            "prompt_context_cache_sha256": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            round2_root = root / "asre_results" / "round2"
            round3a_root = root / "asre_results" / "round3a"
            input_dirs = [
                round2_root / "online_full",
                round2_root / "offline",
                round3a_root / "online_full",
                round3a_root / "offline",
            ]
            for path in input_dirs:
                path.mkdir(parents=True)
            output = round3a_root / "aggregate"
            valid_manifest = round2_root / "state_bank_valid_manifest.json"
            state_bank = root / "asre_results" / "state_bank"
            state_bank.mkdir()
            source_manifest = state_bank / "manifest.jsonl"
            source_manifest.write_text('{}\n', encoding="utf-8")
            valid_manifest.write_text(
                json.dumps({"source_manifest_path": str(source_manifest)}),
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    aggregate_module,
                    "_load_valid_manifest",
                    return_value=(valid_payload, valid_ids, "e" * 64),
                ),
                mock.patch.object(
                    aggregate_module, "load_online_factorial", return_value=online
                ),
                mock.patch.object(
                    aggregate_module, "load_offline_factorial", return_value=offline
                ),
            ):
                aggregate_module.aggregate(
                    round2_online_root=input_dirs[0],
                    round2_offline_root=input_dirs[1],
                    round3a_online_root=input_dirs[2],
                    round3a_offline_root=input_dirs[3],
                    valid_manifest_path=valid_manifest,
                    output_dir=output,
                    bootstrap_samples=30,
                    bootstrap_seed=0,
                    sign_flip_samples=30,
                    sign_flip_seed=0,
                )

            expected_files = {
                "factorial_cells.csv",
                "factorial_cells.json",
                "factorial_simple_effects.csv",
                "factorial_interactions.csv",
                "factorial_task_success.csv",
                "factorial_offline_metrics.csv",
                "factorial_replan_stage_metrics.csv",
                "factorial_paired_transitions.csv",
                "factorial_summary.json",
                "factorial_summary.md",
                "aggregate_metadata.json",
            }
            self.assertEqual(
                {path.name for path in output.iterdir() if path.is_file()},
                expected_files,
            )
            metadata_payload = json.loads(
                (output / "aggregate_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(metadata_payload["output_files"]), expected_files)
            summary_payload = json.loads(
                (output / "factorial_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(summary_payload["cells"]), 8)
            self.assertEqual(len(summary_payload["simple_effects"]), 12)
            self.assertEqual(len(summary_payload["factorial_contrasts"]), 7)
            self.assertIsInstance(
                summary_payload["simple_effects"][0]["coefficients"], list
            )
            self.assertIsInstance(
                summary_payload["factorial_contrasts"][0]["coefficient_by_cell"],
                dict,
            )
            with (output / "factorial_paired_transitions.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 12)

            plot_inputs = _load_plot_inputs(output)
            self.assertEqual(len(plot_inputs.cells), 8)
            self.assertEqual(len(plot_inputs.simple_effects), 12)
            self.assertEqual(len(plot_inputs.interactions), 7)
            self.assertEqual(len(plot_inputs.tasks), 10)


if __name__ == "__main__":
    unittest.main()
