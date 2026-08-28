from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.asre_diagnosis.common import (
    ROUND3B_PROTOCOL,
    atomic_write_json,
    build_round3b_conditions,
    sha256_file,
)
from experiments.asre_diagnosis.round3b.donor import (
    DONOR_MAPPING_RULE,
    atomic_torch_save,
    build_donor_mapping_payload,
    canonical_sha256,
    task_text_sha256,
    tensor_sha256,
)
from experiments.asre_diagnosis.round3b.launch_three_gpu import (
    ACTION_HORIZON,
    FULL_TASK_IDS,
    FULL_TRIALS,
    INFERENCE_STEPS,
    REPLAN_STEPS,
    SMOKE_TASK_IDS,
    SMOKE_TRIALS,
    Provenance,
    RuntimeSpec,
    _condition_command,
    _require_clean_worktree,
    _resolve_runtime,
    _selected_gpu_records,
    _status_payload,
    _validate_action_traces,
    _validate_gpu_ids,
)
from experiments.asre_diagnosis.round3b.preflight import _validate_donors


def _provenance(path: Path) -> Provenance:
    return Provenance(
        checkpoint_path=path,
        checkpoint_sha256="a" * 64,
        dataset_stats_path=path,
        dataset_stats_sha256="b" * 64,
        source_manifest_path=path,
        source_manifest_sha256="c" * 64,
        valid_manifest_path=path,
        valid_manifest_sha256="d" * 64,
        prompt_context_cache_path=path,
        prompt_context_cache_sha256="e" * 64,
        donor_mapping_path=path,
        donor_mapping_sha256="f" * 64,
        donor_observation_manifest_path=path,
        donor_observation_manifest_sha256="1" * 64,
        donor_observation_root=path.parent,
        preflight_report_path=path,
        preflight_report_sha256="2" * 64,
        self_replacement_report_path=path,
        self_replacement_report_sha256="3" * 64,
        round3a_tag="ASRE-round3a-factorial",
        round3a_frozen_commit="4" * 40,
        round3a_run_commit="5" * 40,
        valid_sample_count=499,
    )


class Round3BLauncherTest(unittest.TestCase):
    def test_explicit_physical_gpu_mapping_is_validated_and_recorded(self) -> None:
        self.assertEqual(_validate_gpu_ids([4, 5, 6]), (4, 5, 6))
        for invalid in ([4, 4, 6], [-1, 5, 6], [4, 5]):
            with self.assertRaises(ValueError):
                _validate_gpu_ids(invalid)

        condition = build_round3b_conditions(30)[1]
        payload = _status_payload(
            1,
            5,
            condition,
            "smoke",
            Path("/output"),
            {"index": 5, "name": "NVIDIA RTX A5000"},
            None,
            {"status": "launching"},
        )
        self.assertEqual(payload["condition_index"], 1)
        self.assertEqual(payload["physical_gpu"], 5)
        self.assertEqual(payload["cuda_visible_devices"], "5")
        self.assertEqual(payload["mujoco_egl_device_id"], "5")

        inventory = [
            {"index": index, "name": "NVIDIA RTX A5000"}
            for index in range(8)
        ]
        selected = _selected_gpu_records(inventory, (4, 5, 6))
        self.assertEqual(tuple(selected), (4, 5, 6))
        with self.assertRaisesRegex(ValueError, "absent"):
            _selected_gpu_records(inventory, (4, 5, 8))

    def test_runtime_and_conditions_are_exactly_frozen(self) -> None:
        smoke = _resolve_runtime("libero_uncond_2cam224_1e-4", "smoke")
        full = _resolve_runtime("libero_uncond_2cam224_1e-4", "full")
        self.assertEqual(smoke.task_ids, SMOKE_TASK_IDS)
        self.assertEqual(smoke.num_trials, SMOKE_TRIALS)
        self.assertEqual(full.task_ids, FULL_TASK_IDS)
        self.assertEqual(full.num_trials, FULL_TRIALS)
        self.assertEqual(
            (full.action_horizon, full.inference_steps, full.replan_steps),
            (ACTION_HORIZON, INFERENCE_STEPS, REPLAN_STEPS),
        )
        conditions = build_round3b_conditions(30)
        self.assertEqual([condition.name for condition in conditions], [
            "late_current_correct", "late_wrong_scene", "late_no_video"
        ])

    def test_commands_preserve_three_gpu_mapping_and_wrong_scene_layers(self) -> None:
        runtime = RuntimeSpec(
            task_config="libero_uncond_2cam224_1e-4",
            task_ids=FULL_TASK_IDS,
            num_trials=FULL_TRIALS,
            action_horizon=ACTION_HORIZON,
            replan_steps=REPLAN_STEPS,
            inference_steps=INFERENCE_STEPS,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact"
            artifact.write_bytes(b"x")
            wrong = build_round3b_conditions(30)[1]
            command = _condition_command(
                python_path=Path("/python"),
                condition_index=1,
                condition=wrong,
                condition_output=Path(directory) / wrong.name,
                runtime=runtime,
                provenance=_provenance(artifact),
            )
        self.assertIn("gpu_id=0", command)
        self.assertIn("ASRE_DIAGNOSIS.mode=replace_video_kv", command)
        self.assertIn(f"ASRE_DIAGNOSIS.protocol={ROUND3B_PROTOCOL}", command)
        self.assertIn(
            "ASRE_DIAGNOSIS.disabled_video_layers=["
            + ",".join(str(i) for i in range(15))
            + "]",
            command,
        )
        self.assertIn(
            "ASRE_DIAGNOSIS.replacement_video_layers=["
            + ",".join(str(i) for i in range(15, 30))
            + "]",
            command,
        )
        self.assertTrue(any("donor_mapping_sha256=" in value for value in command))
        self.assertTrue(any("self_replacement_report_sha256=" in value for value in command))

    def test_wrong_scene_smoke_trace_requires_cyclic_donor_and_finite_action(self) -> None:
        runtime = RuntimeSpec(
            task_config="libero_uncond_2cam224_1e-4",
            task_ids=SMOKE_TASK_IDS,
            num_trials=SMOKE_TRIALS,
            action_horizon=ACTION_HORIZON,
            replan_steps=REPLAN_STEPS,
            inference_steps=INFERENCE_STEPS,
        )
        condition = build_round3b_conditions(30)[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces = root / "libero_spatial" / "action_traces"
            traces.mkdir(parents=True)
            for episode_id in range(2):
                record = {
                    "task_suite": "libero_spatial",
                    "task_id": 0,
                    "episode_id": episode_id,
                    "replan_id": 0,
                    "diagnosis_condition": condition.name,
                    "environment_step": 30,
                    "action_inference_seed": 42,
                    "replacement_video_layers": list(range(15, 30)),
                    "donor_trial": (episode_id + 1) % 10,
                    "raw_action": [[0.0] * 7 for _ in range(32)],
                    "executed_action": [[0.0] * 7 for _ in range(32)],
                }
                (traces / f"task0_trial{episode_id}.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
            _validate_action_traces(
                root, task_id=0, condition=condition, runtime=runtime
            )
            bad = json.loads((traces / "task0_trial1.jsonl").read_text())
            bad["donor_trial"] = 1
            (traces / "task0_trial1.jsonl").write_text(
                json.dumps(bad) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(Exception, "donor_trial"):
                _validate_action_traces(
                    root, task_id=0, condition=condition, runtime=runtime
                )

    def test_dirty_code_worktree_is_rejected(self) -> None:
        with mock.patch(
            "experiments.asre_diagnosis.round3b.launch_three_gpu.subprocess.check_output",
            return_value=" M src/fastwam/models/wan22/fastwam_idm.py\n",
        ):
            with self.assertRaisesRegex(RuntimeError, "clean code worktree"):
                _require_clean_worktree()

    def test_full_donor_bundle_preflight_checks_all_100_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations_dir = root / "observations"
            observations_dir.mkdir()
            images = {}
            records = []
            for task_id in range(10):
                text_hash = task_text_sha256(f"task {task_id}")
                for trial in range(10):
                    key = (task_id, trial)
                    value = -1.0 + 2.0 * float(task_id * 10 + trial) / 99.0
                    image = torch.tensor([value], dtype=torch.bfloat16)
                    images[key] = image
                    artifact = observations_dir / f"task{task_id:02d}_trial{trial:02d}.pt"
                    payload = {
                        "task_id": task_id,
                        "source_trial": trial,
                        "input_image": image,
                    }
                    atomic_torch_save(artifact, payload)
                    records.append(
                        {
                            "task_id": task_id,
                            "source_task_id": task_id,
                            "source_trial": trial,
                            "task_text_sha256": text_hash,
                            "initial_state_sha256": canonical_sha256([task_id, trial]),
                            "processed_image_sha256": tensor_sha256(image),
                            "processed_image_shape": list(image.shape),
                            "processed_image_dtype": str(image.dtype),
                            "processed_image_finite": True,
                            "processed_image_min": float(image.float().min().item()),
                            "processed_image_max": float(image.float().max().item()),
                            "artifact_relative_path": str(artifact.relative_to(root)),
                            "artifact_absolute_path": str(artifact),
                            "artifact_sha256": sha256_file(artifact),
                        }
                    )
            manifest = {
                "schema_version": 1,
                "task_suite": "libero_spatial",
                "seed": 42,
                "num_tasks": 10,
                "num_trials": 10,
                "num_steps_wait": 30,
                "first_policy_query_environment_step": 30,
                "task_config": "libero_uncond_2cam224_1e-4",
                "model_ready_image_shape": [1, 3, 224, 448],
                "model_ready_image_dtype": "torch.bfloat16",
                "records": records,
            }
            manifest_path = root / "donor_observation_manifest.json"
            atomic_write_json(manifest_path, manifest)
            mapping = build_donor_mapping_payload(
                manifest,
                observation_manifest_path=manifest_path,
                observation_manifest_sha256=sha256_file(manifest_path),
                images_by_key=images,
            )
            self.assertEqual(mapping["mapping_rule"], DONOR_MAPPING_RULE)
            mapping_path = root / "donor_mapping.json"
            atomic_write_json(mapping_path, mapping)
            report = _validate_donors(mapping_path, manifest_path, root)
            self.assertEqual(report["online_mapping_count"], 100)
            self.assertEqual(report["online_observation_count"], 100)


if __name__ == "__main__":
    unittest.main()
