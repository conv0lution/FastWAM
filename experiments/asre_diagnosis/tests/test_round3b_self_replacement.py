from __future__ import annotations

import unittest

from experiments.asre_diagnosis.round3b.self_replacement_test import (
    REPLACEMENT_LAYERS,
    _validate_audit,
)


def _tensor_stats():
    return {
        "shape": [1, 8, 16],
        "dtype": "torch.bfloat16",
        "device": "cuda:0",
        "numel": 128,
        "finite": True,
        "mean": 0.0,
        "std": 1.0,
        "rms": 1.0,
    }


class Round3BSelfReplacementTest(unittest.TestCase):
    def _audit(self):
        layers = []
        for layer in range(30):
            selected = layer in REPLACEMENT_LAYERS
            layers.append(
                {
                    "layer": layer,
                    "selected_source": "replacement" if selected else "current",
                    "summarized": selected,
                    "current": (
                        {"k": _tensor_stats(), "v": _tensor_stats()} if selected else None
                    ),
                    "replacement": (
                        {"k": _tensor_stats(), "v": _tensor_stats()} if selected else None
                    ),
                    "difference": (
                        {
                            "k": {"max_abs": 0.0, "mean_abs": 0.0, "rms": 0.0, "exact_equal": True},
                            "v": {"max_abs": 0.0, "mean_abs": 0.0, "rms": 0.0, "exact_equal": True},
                        }
                        if selected
                        else None
                    ),
                }
            )
        return {
            "schema_version": 1,
            "replacement_video_layers": list(REPLACEMENT_LAYERS),
            "summarized_video_layers": list(REPLACEMENT_LAYERS),
            "disabled_video_layers": list(range(15)),
            "has_replacement": True,
            "all_replacement_caches_exact_equal": True,
            "max_abs_current_replacement": 0.0,
            "current_video_seq_len": 4096,
            "replacement_video_seq_len": 4096,
            "current_video_tokens_per_frame": 1024,
            "replacement_video_tokens_per_frame": 1024,
            "action_attention_mask_shape": [1, 1, 512, 4608],
            "layers": layers,
        }

    def test_exact_same_input_audit_passes_all_late_layers(self) -> None:
        summary = _validate_audit(self._audit())
        self.assertEqual(summary["num_layers_audited"], 30)
        self.assertEqual(summary["max_layer_cache_absolute_difference"], 0.0)
        self.assertTrue(summary["all_replacement_caches_exact_equal"])

    def test_wrong_selected_layer_or_large_difference_fails(self) -> None:
        audit = self._audit()
        audit["layers"][15]["selected_source"] = "current"
        with self.assertRaisesRegex(ValueError, "selected"):
            _validate_audit(audit)
        audit = self._audit()
        audit["layers"][20]["difference"]["k"]["max_abs"] = 1e-3
        with self.assertRaisesRegex(ValueError, "differs"):
            _validate_audit(audit)


if __name__ == "__main__":
    unittest.main()
