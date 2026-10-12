from __future__ import annotations

import json
from typing import Any

from vllm_tuner.helper.logging import get_logger


logger = get_logger()

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False


def _extract_repo_id(model_id: str) -> str:
    """Extract HuggingFace repo ID from model identifier.

    Handles GGUF-style colon syntax: ``Qwen/Model-GGUF:Q4_0`` → ``Qwen/Model-GGUF``.
    """
    return model_id.split(":")[0]


def _base_repo_candidates(repo_id: str) -> list[str]:
    """Generate candidate repo IDs by stripping GGUF/GGML suffixes."""
    candidates = [repo_id]
    for suffix in ("-GGUF", "-GGML", "-gguf", "-ggml", "_GGUF", "_GGML"):
        stripped = repo_id.replace(suffix, "")
        if stripped != repo_id and stripped not in candidates:
            candidates.append(stripped)
    return candidates


def _download_config(repo_id: str, token: str | None = None) -> dict[str, Any] | None:
    """Download and parse config.json from a HuggingFace repo."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename="config.json", token=token, force_download=False)
        with open(path) as f:
            return json.load(f)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception as e:
        logger.debug("Failed to download config.json from {}: {}", repo_id, e)
        return None


def get_model_config(model_id: str) -> dict[str, Any]:
    """Download and return the HuggingFace model config for a given model ID.

    Tries the repo directly first; for GGUF models, falls back to base repo
    (e.g. ``Qwen/Qwen3-0.6B-GGUF`` → ``Qwen/Qwen3-0.6B``).

    Returns:
        Parsed ``config.json`` dict.

    Raises:
        RuntimeError: If ``huggingface-hub`` is not installed.
        ValueError: If config cannot be found for any candidate repo.

    """
    if not _HF_HUB_AVAILABLE:
        raise RuntimeError(
            "huggingface-hub is required for model config download. Install with: pip install 'llm-vllm-tuner[hf]'"
        )

    from vllm_tuner.settings import settings

    token = settings.HF_TOKEN.get_secret_value() if settings.HF_TOKEN else None
    repo_id = _extract_repo_id(model_id)
    candidates = _base_repo_candidates(repo_id)

    for candidate in candidates:
        config = _download_config(candidate, token=token)
        if config is not None:
            logger.info("Loaded model config from '{}'", candidate)
            return config

    raise ValueError(f"Could not find config.json for model '{model_id}'. Tried: {', '.join(candidates)}.")
