"""Small, environment-independent helpers for fixed evaluation prompt contexts."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Iterable

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


@torch.no_grad()
def prewarm_prompt_contexts_and_release_text_encoder(
    model: torch.nn.Module,
    prompts: Iterable[str],
) -> int:
    """Encode fixed prompts, retain CPU contexts, then release the text encoder."""

    ordered_prompts = tuple(dict.fromkeys(str(prompt) for prompt in prompts))
    if not ordered_prompts or any(not prompt for prompt in ordered_prompts):
        raise ValueError("Prompt prewarm requires at least one non-empty prompt.")
    if getattr(model, "text_encoder", None) is None:
        raise ValueError("Prompt prewarm requires a loaded text encoder.")

    for prompt in ordered_prompts:
        context, context_mask = get_cached_prompt_context(model, prompt)
        cache = getattr(model, "_eval_prompt_context_cache")
        cache[prompt] = (
            context.detach().to(device="cpu"),
            context_mask.detach().to(device="cpu"),
        )
        del context, context_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    text_encoder = getattr(model, "text_encoder")
    tokenizer = getattr(model, "tokenizer", None)
    model.text_encoder = None
    if hasattr(model, "tokenizer"):
        model.tokenizer = None
    if hasattr(model, "text_encoder_device"):
        model.text_encoder_device = None
    del text_encoder, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(
        "Prewarmed %d prompt contexts on the model text encoder and released it.",
        len(ordered_prompts),
    )
    return len(ordered_prompts)


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
    setattr(
        model,
        "_eval_prompt_context_cache_metadata",
        {key: value for key, value in payload.items() if key != "prompts"},
    )
    logging.info("Loaded %d fixed prompt contexts from %s", len(cache), cache_path)
    return len(cache)
