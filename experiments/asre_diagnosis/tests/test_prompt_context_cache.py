from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.libero.prompt_context_cache import (
    get_cached_prompt_context,
    load_prompt_context_cache,
)


class _ModelWithoutTextEncoder:
    text_encoder = None


class PromptContextCacheTest(unittest.TestCase):
    def test_fixed_context_cache_supports_text_encoder_free_evaluation(self) -> None:
        prompt = "fixed task prompt"
        context = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        mask = torch.ones((1, 3), dtype=torch.bool)
        payload = {
            "schema_version": 1,
            "prompts": {
                prompt: {
                    "task_id": 0,
                    "task_description": "task",
                    "context": context,
                    "context_mask": mask,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt_cache.pt"
            torch.save(payload, path)
            model = _ModelWithoutTextEncoder()
            self.assertEqual(load_prompt_context_cache(model, path), 1)
            loaded_context, loaded_mask = get_cached_prompt_context(model, prompt)
        torch.testing.assert_close(loaded_context, context)
        torch.testing.assert_close(loaded_mask, mask)

    def test_missing_prompt_fails_closed_without_text_encoder(self) -> None:
        model = _ModelWithoutTextEncoder()
        model._eval_prompt_context_cache = {}
        with self.assertRaises(KeyError):
            get_cached_prompt_context(model, "unknown")


if __name__ == "__main__":
    unittest.main()
