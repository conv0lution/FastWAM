from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.asre_diagnosis.common import build_round2_conditions
from experiments.asre_diagnosis.round2.launch_eight_gpu import (
    ACTION_HORIZON,
    FULL_TASK_IDS,
    FULL_TRIALS,
    INFERENCE_STEPS,
    REPLAN_STEPS,
    SMOKE_CONDITION_INDICES,
    SMOKE_TASK_IDS,
    SMOKE_TRIALS,
    Provenance,
    RuntimeSpec,
    _child_environment,
    _condition_command,
    _condition_indices,
    _exclusive_launcher_lock,
    _recorded_live_pid,
)


class Round2LauncherTest(unittest.TestCase):
    def test_smoke_and_full_condition_indices_are_fixed(self) -> None:
        self.assertEqual(_condition_indices("smoke"), SMOKE_CONDITION_INDICES)
        self.assertEqual(_condition_indices("full"), tuple(range(8)))

    def test_physical_gpu_isolated_but_egl_keeps_physical_index(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            environment = _child_environment(4)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "4")
        self.assertEqual(environment["MUJOCO_EGL_DEVICE_ID"], "4")
        self.assertEqual(environment["MUJOCO_GL"], "egl")
        self.assertEqual(environment["PYOPENGL_PLATFORM"], "egl")

    def test_child_command_disables_text_encoder_and_passes_exact_schedule(self) -> None:
        condition = build_round2_conditions(30)[7]
        runtime = RuntimeSpec(
            task_config="libero_uncond_2cam224_1e-4",
            task_ids=FULL_TASK_IDS,
            num_trials=FULL_TRIALS,
            action_horizon=ACTION_HORIZON,
            replan_steps=REPLAN_STEPS,
            inference_steps=INFERENCE_STEPS,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.write_text("x", encoding="utf-8")
            provenance = Provenance(
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
            )
            command = _condition_command(
                python_path=Path("/python"),
                condition_index=7,
                condition=condition,
                condition_output=root / "condition",
                runtime=runtime,
                provenance=provenance,
            )
        self.assertIn("model.load_text_encoder=false", command)
        self.assertIn("EVALUATION.device=cuda:0", command)
        self.assertIn(
            "ASRE_DIAGNOSIS.enabled_video_retrieval_layers="
            "[15,16,17,18,19,25,26,27,28,29]",
            command,
        )
        self.assertIn(
            "ASRE_DIAGNOSIS.disabled_video_layers="
            "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,20,21,22,23,24]",
            command,
        )

    def test_smoke_runtime_constants_match_protocol(self) -> None:
        self.assertEqual(SMOKE_TASK_IDS, (0,))
        self.assertEqual(SMOKE_TRIALS, 2)

    def test_live_recorded_child_blocks_unsafe_resume(self) -> None:
        payload = {
            "status": "running",
            "attempts": [{"status": "running", "pid": os.getpid()}],
        }
        self.assertEqual(_recorded_live_pid(payload), os.getpid())
        payload["status"] = "failed"
        self.assertIsNone(_recorded_live_pid(payload))

    def test_dead_recorded_child_is_resumable(self) -> None:
        payload = {
            "status": "running",
            "attempts": [{"status": "running", "pid": 2**31 - 1}],
        }
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
