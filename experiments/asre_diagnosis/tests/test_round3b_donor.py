from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from experiments.asre_diagnosis.common import sha256_file
from experiments.asre_diagnosis.round3b.donor import (
    DONOR_MAPPING_NAME,
    DONOR_OBSERVATION_MANIFEST_NAME,
    DONOR_SCHEMA_VERSION,
    OnlineDonorBundle,
    atomic_torch_save,
    build_donor_mapping_payload,
    canonical_sha256,
    donor_artifact_relative_path,
    task_text_sha256,
    tensor_sha256,
    validate_donor_mapping,
    write_frozen_json,
)
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
try:
    from experiments.libero.eval_libero_single import _run_prepared_action_inference
except ModuleNotFoundError:
    _run_prepared_action_inference = None


class _ReplacementAwareModel:
    def __init__(self) -> None:
        self.received = None

    def infer_action(
        self,
        *,
        disabled_video_layers=(),
        replacement_input_image=None,
        replacement_video_layers=(),
        compile_action_infer=False,
        **kwargs,
    ):
        self.received = {
            "disabled_video_layers": tuple(disabled_video_layers),
            "replacement_input_image": replacement_input_image,
            "replacement_video_layers": tuple(replacement_video_layers),
            "compile_action_infer": compile_action_infer,
            "kwargs": kwargs,
        }
        return {"action": torch.zeros(1, 32, 7)}


def _make_bundle(root: Path) -> tuple[OnlineDonorBundle, list[torch.Tensor], list[torch.Tensor]]:
    description = "move the object"
    states = [torch.tensor([0.0, 1.0]), torch.tensor([2.0, 3.0])]
    images = [
        torch.full((1, 3, 2, 4), -0.5, dtype=torch.bfloat16),
        torch.full((1, 3, 2, 4), 0.75, dtype=torch.bfloat16),
    ]
    records = []
    for trial in range(2):
        relative = donor_artifact_relative_path(0, trial)
        artifact = root / relative
        payload = {
            "schema_version": DONOR_SCHEMA_VERSION,
            "task_suite": "libero_spatial",
            "task_id": 0,
            "source_trial": trial,
            "task_description": description,
            "task_text_sha256": task_text_sha256(description),
            "initial_state_sha256": canonical_sha256(states[trial]),
            "processed_image_sha256": tensor_sha256(images[trial]),
            "input_image": images[trial],
        }
        atomic_torch_save(artifact, payload)
        records.append(
            {
                "task_id": 0,
                "source_task_id": 0,
                "source_trial": trial,
                "task_description": description,
                "task_text_sha256": task_text_sha256(description),
                "initial_state_sha256": canonical_sha256(states[trial]),
                "processed_image_sha256": tensor_sha256(images[trial]),
                "processed_image_shape": list(images[trial].shape),
                "processed_image_dtype": str(images[trial].dtype),
                "processed_image_finite": True,
                "processed_image_min": float(images[trial].float().min()),
                "processed_image_max": float(images[trial].float().max()),
                "artifact_relative_path": str(relative),
                "artifact_absolute_path": str(artifact.resolve()),
                "artifact_sha256": sha256_file(artifact),
            }
        )
    manifest = {
        "schema_version": DONOR_SCHEMA_VERSION,
        "task_suite": "libero_spatial",
        "seed": 42,
        "num_tasks": 1,
        "num_trials": 2,
        "task_ids": [0],
        "records": records,
    }
    manifest_path = root / DONOR_OBSERVATION_MANIFEST_NAME
    write_frozen_json(manifest_path, manifest)
    mapping = build_donor_mapping_payload(
        manifest,
        observation_manifest_path=manifest_path,
        observation_manifest_sha256=sha256_file(manifest_path),
        images_by_key={(0, trial): image for trial, image in enumerate(images)},
    )
    mapping_path = root / DONOR_MAPPING_NAME
    write_frozen_json(mapping_path, mapping)
    bundle = OnlineDonorBundle.load(
        mapping_path=mapping_path,
        observation_manifest_path=manifest_path,
        observation_root=root,
    )
    return bundle, states, images


class Round3BDonorTest(unittest.TestCase):
    def test_canonical_hash_is_layout_independent_but_content_sensitive(self) -> None:
        contiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        equivalent = contiguous.t().contiguous().t()
        changed = contiguous.clone()
        changed[0, 0] = 99
        self.assertEqual(canonical_sha256(contiguous), canonical_sha256(equivalent))
        self.assertNotEqual(canonical_sha256(contiguous), canonical_sha256(changed))

    def test_mapping_is_same_task_next_trial_derangement_and_loads_donor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, states, images = _make_bundle(Path(directory))
            self.assertEqual(bundle.mappings[(0, 0)]["donor_trial"], 1)
            self.assertEqual(bundle.mappings[(0, 1)]["donor_trial"], 0)
            loaded = bundle.load_for_recipient(
                task_id=0,
                recipient_trial=0,
                recipient_initial_state=states[0],
                task_description="move the object",
                device="cpu",
                dtype=torch.bfloat16,
            )
            self.assertTrue(torch.equal(loaded.image, images[1]))
            self.assertEqual(loaded.provenance["donor_trial"], 1)
            self.assertGreater(loaded.provenance["processed_pixel_mae"], 0.0)

    def test_loader_rejects_recipient_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, _states, _images = _make_bundle(Path(directory))
            with self.assertRaisesRegex(ValueError, "initial-state hash mismatch"):
                bundle.load_for_recipient(
                    task_id=0,
                    recipient_trial=0,
                    recipient_initial_state=torch.tensor([9.0, 9.0]),
                    task_description="move the object",
                    device="cpu",
                    dtype=torch.bfloat16,
                )

    def test_mapping_validation_rejects_self_donor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, _states, _images = _make_bundle(Path(directory))
            malformed = dict(bundle.mapping_payload)
            malformed["records"] = [dict(record) for record in malformed["records"]]
            malformed["records"][0]["donor_trial"] = 0
            with self.assertRaisesRegex(ValueError, "not the declared derangement"):
                validate_donor_mapping(malformed, bundle.observation_payload)

    @unittest.skipIf(
        _run_prepared_action_inference is None,
        "Optional LIBERO checkout is not on PYTHONPATH.",
    )
    def test_evaluator_passes_replacement_only_when_layers_are_configured(self) -> None:
        model = _ReplacementAwareModel()
        image = torch.zeros(1, 3, 2, 4, dtype=torch.bfloat16)
        cfg = OmegaConf.create(
            {
                "EVALUATION": {
                    "visualize_future_video": False,
                    "compile_action_infer": True,
                },
                "ASRE_DIAGNOSIS": {
                    "enabled": True,
                    "disabled_video_layers": list(range(15)),
                    "replacement_video_layers": list(range(15, 30)),
                },
            }
        )
        action, future = _run_prepared_action_inference(
            model,
            cfg,
            {"input_image": torch.ones_like(image)},
            replacement_input_image=image,
        )
        self.assertEqual(tuple(action.shape), (1, 32, 7))
        self.assertIsNone(future)
        self.assertIs(model.received["replacement_input_image"], image)
        self.assertEqual(model.received["replacement_video_layers"], tuple(range(15, 30)))
        self.assertEqual(model.received["disabled_video_layers"], tuple(range(15)))

    @unittest.skipIf(
        _run_prepared_action_inference is None,
        "Optional LIBERO checkout is not on PYTHONPATH.",
    )
    def test_evaluator_rejects_missing_donor_image(self) -> None:
        model = _ReplacementAwareModel()
        cfg = OmegaConf.create(
            {
                "EVALUATION": {
                    "visualize_future_video": False,
                    "compile_action_infer": False,
                },
                "ASRE_DIAGNOSIS": {
                    "enabled": True,
                    "disabled_video_layers": list(range(15)),
                    "replacement_video_layers": list(range(15, 30)),
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "no frozen donor image"):
            _run_prepared_action_inference(model, cfg, {})


if __name__ == "__main__":
    unittest.main()
