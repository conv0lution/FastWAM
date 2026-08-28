from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.libero.prompt_context_cache import (
    get_cached_prompt_context,
    load_prompt_context_cache,
    prewarm_prompt_contexts_and_release_text_encoder,
)


class _ModelWithoutTextEncoder:
    text_encoder = None


class _PrewarmModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = torch.nn.Linear(2, 2)
        self.text_encoder_device = torch.device("cuda:0")
        self.tokenizer = object()
        self.encoded_prompts: list[str] = []

    def encode_prompt(self, prompt: str):
        self.encoded_prompts.append(prompt)
        value = float(len(self.encoded_prompts))
        return torch.full((1, 2, 3), value), torch.ones((1, 2), dtype=torch.bool)


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

    def test_prewarm_retains_contexts_and_releases_text_encoder(self) -> None:
        model = _PrewarmModel()

        count = prewarm_prompt_contexts_and_release_text_encoder(
            model, ["prompt one", "prompt two", "prompt one"]
        )

        self.assertEqual(count, 2)
        self.assertEqual(model.encoded_prompts, ["prompt one", "prompt two"])
        self.assertIsNone(model.text_encoder)
        self.assertIsNone(model.text_encoder_device)
        self.assertIsNone(model.tokenizer)
        context, mask = get_cached_prompt_context(model, "prompt two")
        self.assertEqual(context.device.type, "cpu")
        self.assertEqual(mask.device.type, "cpu")
        with self.assertRaises(KeyError):
            get_cached_prompt_context(model, "unknown prompt")


if __name__ == "__main__":
    unittest.main()
