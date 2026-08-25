from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.asre_diagnosis.common import sha256_file
from experiments.asre_diagnosis.round2.validate_state_bank import (
    QC_RULE,
    _validate_sample_partition,
    evaluate_replay_qc,
    validate_existing_artifacts,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


class Round2StateBankQCTest(unittest.TestCase):
    def test_exact_replay_passes_pre_registered_qc(self) -> None:
        raw = np.zeros((32, 7), dtype=np.float32)
        executed = np.zeros((32, 7), dtype=np.float32)

        passed, detail = evaluate_replay_qc(
            stored_raw=raw,
            replayed_raw=raw.copy(),
            stored_executed=executed,
            replayed_executed=executed.copy(),
        )

        self.assertTrue(passed)
        self.assertEqual(detail["reasons"], [])
        self.assertEqual(detail["max_raw_absolute_error"], 0.0)
        self.assertTrue(detail["postprocessed_gripper_exact"])

    def test_raw_error_above_fixed_tolerance_is_excluded(self) -> None:
        stored = np.zeros((32, 7), dtype=np.float32)
        replayed = stored.copy()
        replayed[0, 0] = 1.1e-6

        passed, detail = evaluate_replay_qc(
            stored_raw=stored,
            replayed_raw=replayed,
            stored_executed=stored,
            replayed_executed=stored,
        )

        self.assertFalse(passed)
        self.assertIn("raw_max_absolute_error_exceeds_tolerance", detail["reasons"])

    def test_gripper_mismatch_is_excluded_independently(self) -> None:
        raw = np.zeros((32, 7), dtype=np.float32)
        stored_executed = raw.copy()
        replayed_executed = raw.copy()
        replayed_executed[0, -1] = 1.0

        passed, detail = evaluate_replay_qc(
            stored_raw=raw,
            replayed_raw=raw,
            stored_executed=stored_executed,
            replayed_executed=replayed_executed,
        )

        self.assertFalse(passed)
        self.assertIn("postprocessed_gripper_mismatch", detail["reasons"])

    def test_valid_manifest_must_preserve_source_order_and_partition(self) -> None:
        source = [
            {"sample_id": "a"},
            {"sample_id": "b"},
            {"sample_id": "c"},
        ]
        payload = {
            "valid_sample_ids": ["a", "c"],
            "excluded_samples": [{"sample_id": "b", "reasons": ["qc"]}],
        }
        self.assertEqual(_validate_sample_partition(payload, source), ["a", "c"])

        payload["valid_sample_ids"] = ["c", "a"]
        with self.assertRaises(ValueError):
            _validate_sample_partition(payload, source)

    def test_condition_replay_can_trust_qc_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "checkpoint.pt"
            stats = root / "stats.json"
            source_manifest = root / "manifest.jsonl"
            prompt_cache = root / "prompt_context_cache.pt"
            valid_manifest = root / "state_bank_valid_manifest.json"
            checkpoint.write_bytes(b"checkpoint")
            stats.write_text("{}\n", encoding="utf-8")
            description = "test task"
            source_record = {
                "sample_id": "sample-0",
                "task_id": 0,
                "task_description": description,
            }
            source_manifest.write_text(json.dumps(source_record) + "\n", encoding="utf-8")
            torch.save(
                {
                    "schema_version": 1,
                    "prompts": {
                        DEFAULT_PROMPT.format(task=description): {
                            "task_id": 0,
                            "task_description": description,
                            "context": torch.zeros((1, 2, 3)),
                            "context_mask": torch.ones((1, 2), dtype=torch.bool),
                        }
                    },
                },
                prompt_cache,
            )
            payload = {
                "schema_version": 1,
                "checkpoint_path": str(checkpoint),
                # Deliberately differs from the tiny test file's real digest.
                "checkpoint_sha256": "0" * 64,
                "dataset_stats_path": str(stats),
                "dataset_stats_sha256": sha256_file(stats),
                "source_manifest_path": str(source_manifest),
                "source_manifest_sha256": sha256_file(source_manifest),
                "prompt_context_cache_path": str(prompt_cache),
                "prompt_context_cache_sha256": sha256_file(prompt_cache),
                "valid_sample_ids": ["sample-0"],
                "excluded_samples": [],
                "qc_rule": QC_RULE,
            }
            valid_manifest.write_text(json.dumps(payload), encoding="utf-8")

            validated = validate_existing_artifacts(
                valid_manifest_path=valid_manifest,
                prompt_cache_path=prompt_cache,
                source_manifest_path=source_manifest,
                source_records=[source_record],
                checkpoint_path=checkpoint,
                dataset_stats_path=stats,
                verify_checkpoint_hash=False,
            )
            self.assertEqual(validated["checkpoint_sha256"], "0" * 64)
            with self.assertRaises(ValueError):
                validate_existing_artifacts(
                    valid_manifest_path=valid_manifest,
                    prompt_cache_path=prompt_cache,
                    source_manifest_path=source_manifest,
                    source_records=[source_record],
                    checkpoint_path=checkpoint,
                    dataset_stats_path=stats,
                    verify_checkpoint_hash=True,
                )


if __name__ == "__main__":
    unittest.main()
