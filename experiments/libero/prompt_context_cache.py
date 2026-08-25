"""Small, environment-independent helpers for fixed evaluation prompt contexts."""

from __future__ import annotations

import logging
from pathlib import Path

import torch


@torch.no_grad()
def get_cached_prompt_context(
    model: torch.nn.Module,
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a cached context, encoding once only when an encoder is available."""

    cache = getattr(model, "_eval_prompt_context_cache", None)
    if cache is None:
        cache = {}
        setattr(model, "_eval_prompt_context_cache", cache)
    if prompt not in cache:
        if getattr(model, "text_encoder", None) is None:
            raise KeyError(
                "Prompt is absent from EVALUATION.prompt_context_cache_path and the "
                f"text encoder is disabled: {prompt!r}."
            )
        logging.info("Encoding and caching evaluation prompt: %s", prompt)
        context, context_mask = model.encode_prompt(prompt)
        cache[prompt] = (context.detach(), context_mask.detach())
    return cache[prompt]


def load_prompt_context_cache(model: torch.nn.Module, cache_path: Path) -> int:
    """Load the immutable ASRE prompt cache on CPU and attach it to ``model``."""

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported prompt-context cache schema: {cache_path}")
    prompt_records = payload.get("prompts")
    if not isinstance(prompt_records, dict) or not prompt_records:
        raise ValueError(f"Prompt-context cache has no prompt records: {cache_path}")

    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for prompt, record in prompt_records.items():
        if not isinstance(record, dict):
            raise TypeError(f"Invalid prompt cache record for {prompt!r}: {type(record)}")
        context = record.get("context")
        context_mask = record.get("context_mask")
        if not isinstance(context, torch.Tensor) or not isinstance(context_mask, torch.Tensor):
            raise TypeError(f"Prompt cache tensors are missing for {prompt!r} in {cache_path}.")
        cache[str(prompt)] = (context.detach().cpu(), context_mask.detach().cpu())
    setattr(model, "_eval_prompt_context_cache", cache)
    logging.info("Loaded %d fixed prompt contexts from %s", len(cache), cache_path)
    return len(cache)
