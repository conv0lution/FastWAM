from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.asre_diagnosis.common import build_round3a_conditions
from experiments.asre_diagnosis.round3a.launch_three_gpu import (
    ACTION_HORIZON,
    FULL_TASK_IDS,
    FULL_TRIALS,
    INFERENCE_STEPS,
    REPLAN_STEPS,
    SMOKE_CONDITION_INDICES,
    SMOKE_TASK_IDS,
    SMOKE_TRIALS,
    DirectorySafetyError,
    Provenance,
    RuntimeSpec,
    _child_environment,
    _condition_command,
    _condition_indices,
    _exclusive_launcher_lock,
    _gpu_inventory,
    _load_provenance,
    main as launcher_main,
    _recorded_live_pid,
    _require_clean_worktree,
    _resolve_runtime,
    _validate_round3a_output_scope,
    _validate_task_action_traces,
)


def _provenance(artifact: Path) -> Provenance:
    return Provenance(
        checkpoint_path=artifact,
        checkpoint_sha256="a" * 64,
        dataset_stats_path=artifact,
        dataset_stats_sha256="b" * 64,
        source_manifest_path=artifact,
        source_manifest_sha256="c" * 64,
        valid_manifest_path=artifact,
        valid_manifest_sha256="d" * 64,
        prompt_context_cache_path=artifact,
        prompt_context_cache_sha256="e" * 64,
        valid_sample_count=499,
        qc_rule={"fixed": True},
        round2_reference_metadata_path=artifact,
        round2_reference_metadata_sha256="f" * 64,
    )


class Round3ALauncherTest(unittest.TestCase):
    def test_smoke_and_full_use_exactly_the_three_new_cells(self) -> None:
        self.assertEqual(SMOKE_CONDITION_INDICES, (0, 1, 2))
        self.assertEqual(_condition_indices("smoke"), (0, 1, 2))
        self.assertEqual(_condition_indices("full"), (0, 1, 2))
        self.assertEqual(SMOKE_TASK_IDS, (0,))
        self.assertEqual(SMOKE_TRIALS, 2)
        self.assertEqual(FULL_TASK_IDS, tuple(range(10)))
        self.assertEqual(FULL_TRIALS, 10)

    def test_frozen_runtime_resolves_from_the_real_task_config(self) -> None:
        smoke = _resolve_runtime("libero_uncond_2cam224_1e-4", "smoke")
        full = _resolve_runtime("libero_uncond_2cam224_1e-4", "full")
        self.assertEqual(
            (smoke.action_horizon, smoke.inference_steps, smoke.replan_steps),
            (ACTION_HORIZON, INFERENCE_STEPS, REPLAN_STEPS),
        )
        self.assertEqual(smoke.task_ids, SMOKE_TASK_IDS)
        self.assertEqual(smoke.num_trials, SMOKE_TRIALS)
        self.assertEqual(full.task_ids, FULL_TASK_IDS)
        self.assertEqual(full.num_trials, FULL_TRIALS)

    def test_physical_gpu_isolated_while_model_uses_logical_cuda_zero(self) -> None:
        contaminated = {
            "LIBERO_WORKER_MODE": "1",
            "LIBERO_WORKER_PENDING_FILE": "/stale/pending",
            "RANK": "7",
            "WORLD_SIZE": "8",
        }
        with mock.patch.dict(os.environ, contaminated, clear=True):
            environment = _child_environment(2)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(environment["MUJOCO_EGL_DEVICE_ID"], "2")
        self.assertEqual(environment["MUJOCO_GL"], "egl")
        self.assertEqual(environment["PYOPENGL_PLATFORM"], "egl")
        for key in contaminated:
            self.assertNotIn(key, environment)

    def test_child_commands_pass_exact_empty_and_ab_keep_schedules(self) -> None:
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
            artifact.write_text("x", encoding="utf-8")
            conditions = build_round3a_conditions(30)
            none_command = _condition_command(
                python_path=Path("/python"),
                condition_index=0,
                condition=conditions[0],
                condition_output=Path(directory) / "none",
                runtime=runtime,
                provenance=_provenance(artifact),
            )
            ab_command = _condition_command(
                python_path=Path("/python"),
                condition_index=2,
                condition=conditions[2],
                condition_output=Path(directory) / "ab",
                runtime=runtime,
                provenance=_provenance(artifact),
            )
        self.assertIn("model.load_text_encoder=false", none_command)
        self.assertIn("EVALUATION.device=cuda:0", none_command)
        self.assertIn("ASRE_DIAGNOSIS.enabled_video_retrieval_layers=[]", none_command)
        self.assertIn(
            "ASRE_DIAGNOSIS.disabled_video_layers=["
            + ",".join(str(index) for index in range(30))
            + "]",
            none_command,
        )
        self.assertIn(
            "ASRE_DIAGNOSIS.enabled_video_retrieval_layers=["
            + ",".join(str(index) for index in range(15, 25))
            + "]",
            ab_command,
        )
        self.assertIn(
            "ASRE_DIAGNOSIS.disabled_video_layers=["
            + ",".join(str(index) for index in (*range(15), *range(25, 30)))
            + "]",
            ab_command,
        )

    def test_gpu_preflight_selects_only_physical_zero_through_two(self) -> None:
        lines = [
            f"{index}, NVIDIA RTX A5000, GPU-{index}, 0000:0{index}:00.0, 580.65, 24564, 23000"
            for index in range(8)
        ]
        with mock.patch(
            "experiments.asre_diagnosis.round3a.launch_three_gpu.subprocess.check_output",
            return_value="\n".join(lines),
        ):
            inventory = _gpu_inventory()
        self.assertEqual([record["index"] for record in inventory], [0, 1, 2])

    def test_smoke_trace_validation_rejects_nonfinite_actions(self) -> None:
        runtime = RuntimeSpec(
            task_config="libero_uncond_2cam224_1e-4",
            task_ids=SMOKE_TASK_IDS,
            num_trials=SMOKE_TRIALS,
            action_horizon=ACTION_HORIZON,
            replan_steps=REPLAN_STEPS,
            inference_steps=INFERENCE_STEPS,
        )
        condition = build_round3a_conditions(30)[0]
        with tempfile.TemporaryDirectory() as directory:
            trace_dir = (
                Path(directory) / "libero_spatial" / "action_traces"
            )
            trace_dir.mkdir(parents=True)
            base_record = {
                "task_suite": "libero_spatial",
                "task_id": 0,
                "diagnosis_condition": condition.name,
                "replan_id": 0,
                "environment_step": 30,
                "action_inference_seed": 42,
                "raw_action": [[0.0] * 7 for _ in range(ACTION_HORIZON)],
                "executed_action": [[0.0] * 7 for _ in range(ACTION_HORIZON)],
            }
            for episode_id in range(SMOKE_TRIALS):
                record = {**base_record, "episode_id": episode_id}
                (trace_dir / f"task0_trial{episode_id}.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
            _validate_task_action_traces(
                Path(directory),
                task_id=0,
                condition=condition,
                runtime=runtime,
            )

            bad_path = trace_dir / "task0_trial1.jsonl"
            bad_record = {**base_record, "episode_id": 1}
            bad_record["raw_action"] = [[0.0] * 7 for _ in range(ACTION_HORIZON)]
            bad_record["raw_action"][0][0] = float("nan")
            bad_path.write_text(json.dumps(bad_record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DirectorySafetyError, "Non-finite"):
                _validate_task_action_traces(
                    Path(directory),
                    task_id=0,
                    condition=condition,
                    runtime=runtime,
                )

    def test_dirty_worktree_is_rejected_before_launch(self) -> None:
        with mock.patch(
            "experiments.asre_diagnosis.round3a.launch_three_gpu.subprocess.check_output",
            return_value="?? experiments/asre_diagnosis/round3a/new.py\n",
        ):
            with self.assertRaisesRegex(RuntimeError, "clean code worktree"):
                _require_clean_worktree()

    def test_main_checks_cleanliness_before_creating_launcher_lock(self) -> None:
        with mock.patch(
            "experiments.asre_diagnosis.round3a.launch_three_gpu._parse_args",
            return_value=SimpleNamespace(),
        ), mock.patch(
            "experiments.asre_diagnosis.round3a.launch_three_gpu._require_clean_worktree",
            side_effect=RuntimeError("dirty"),
        ), mock.patch(
            "experiments.asre_diagnosis.round3a.launch_three_gpu._exclusive_launcher_lock"
        ) as lock:
            with self.assertRaisesRegex(RuntimeError, "dirty"):
                launcher_main()
        lock.assert_not_called()

    def test_output_scope_rejects_round1_and_round2_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            round1 = Path(directory) / "asre_results"
            round2 = round1 / "round2"
            state_bank = round1 / "state_bank"
            round2.mkdir(parents=True)
            state_bank.mkdir()
            source_manifest = state_bank / "manifest.jsonl"
            source_manifest.write_text('{}\n', encoding="utf-8")
            valid_manifest = round2 / "state_bank_valid_manifest.json"
            valid_manifest.write_text(
                json.dumps({"source_manifest_path": str(source_manifest)}),
                encoding="utf-8",
            )
            _validate_round3a_output_scope(
                round1 / "round3a" / "online_full",
                valid_manifest_path=valid_manifest,
            )
            for unsafe in (round1, round2 / "new_output", state_bank / "new_output"):
                with self.subTest(unsafe=unsafe):
                    with self.assertRaisesRegex(
                        DirectorySafetyError, "overlaps frozen Round-1/2"
                    ):
                        _validate_round3a_output_scope(
                            unsafe, valid_manifest_path=valid_manifest
                        )

    def test_weak_or_malformed_qc_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            stats = root / "stats.json"
            source = root / "manifest.jsonl"
            cache = root / "prompt_cache.pt"
            reference = root / "round2_metadata.json"
            checkpoint.write_bytes(b"checkpoint")
            stats.write_text("{}", encoding="utf-8")
            source.write_text('{"sample_id":"only_one"}\n', encoding="utf-8")
            cache.write_bytes(b"not-a-torch-payload")
            reference.write_text("{}", encoding="utf-8")

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            valid = root / "valid.json"
            valid.write_text(
                json.dumps(
                    {
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": digest(checkpoint),
                        "dataset_stats_path": str(stats),
                        "dataset_stats_sha256": digest(stats),
                        "source_manifest_path": str(source),
                        "source_manifest_sha256": digest(source),
                        "prompt_context_cache_path": str(cache),
                        "prompt_context_cache_sha256": digest(cache),
                        "valid_sample_ids": ["only_one"],
                        "excluded_samples": [],
                        "qc_rule": {"arbitrary": True},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                _load_provenance(valid, str(checkpoint), str(stats), reference)

    def test_live_recorded_child_blocks_unsafe_resume(self) -> None:
        payload = {
            "status": "running",
            "attempts": [{"status": "running", "pid": os.getpid()}],
        }
        self.assertEqual(_recorded_live_pid(payload), os.getpid())
        payload["status"] = "failed"
        self.assertIsNone(_recorded_live_pid(payload))

    def test_output_root_lock_rejects_concurrent_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "online_full"
            with _exclusive_launcher_lock(output_root):
                with self.assertRaisesRegex(RuntimeError, "Another launcher"):
                    with _exclusive_launcher_lock(output_root):
                        self.fail("A concurrent launcher unexpectedly acquired the lock.")

    def test_child_inherited_lock_survives_parent_descriptor_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "online_full"
            child = None
            with _exclusive_launcher_lock(output_root) as lock_fd:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import sys; print('ready', flush=True); sys.stdin.read(1)",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    pass_fds=(lock_fd,),
                )
                assert child.stdout is not None
                self.assertEqual(child.stdout.readline().strip(), "ready")
            try:
                with self.assertRaisesRegex(RuntimeError, "Another launcher"):
                    with _exclusive_launcher_lock(output_root):
                        self.fail("The inherited child lock was released by the parent.")
            finally:
                assert child is not None and child.stdin is not None
                child.stdin.write("x")
                child.stdin.flush()
                child.stdin.close()
                child.wait(timeout=10)
                assert child.stdout is not None and child.stderr is not None
                child.stdout.close()
                child.stderr.close()
            with _exclusive_launcher_lock(output_root):
                pass


if __name__ == "__main__":
    unittest.main()
