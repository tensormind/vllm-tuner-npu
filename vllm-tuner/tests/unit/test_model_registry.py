from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vllm_tuner.utils.model_registry import (
    _base_repo_candidates,
    _extract_repo_id,
)


class TestExtractRepoId:
    def test_plain_model(self):
        assert _extract_repo_id("meta-llama/Llama-3-8B") == "meta-llama/Llama-3-8B"

    def test_gguf_colon_syntax(self):
        assert _extract_repo_id("Qwen/Qwen3-0.6B-GGUF:Q4_0") == "Qwen/Qwen3-0.6B-GGUF"

    def test_multiple_colons(self):
        assert _extract_repo_id("org/model:tag:sub") == "org/model"


class TestBaseRepoCandidates:
    def test_plain_model_returns_self(self):
        candidates = _base_repo_candidates("meta-llama/Llama-3-8B")
        assert candidates == ["meta-llama/Llama-3-8B"]

    def test_gguf_generates_stripped_candidate(self):
        candidates = _base_repo_candidates("Qwen/Qwen3-0.6B-GGUF")
        assert "Qwen/Qwen3-0.6B-GGUF" in candidates
        assert "Qwen/Qwen3-0.6B" in candidates

    def test_ggml_generates_stripped_candidate(self):
        candidates = _base_repo_candidates("org/model-GGML")
        assert "org/model" in candidates


class TestGetModelConfig:
    @patch("vllm_tuner.utils.model_registry._HF_HUB_AVAILABLE", False)
    def test_raises_without_hf_hub(self):
        from vllm_tuner.utils.model_registry import get_model_config

        with pytest.raises(RuntimeError, match="huggingface-hub is required"):
            get_model_config("meta-llama/Llama-3-8B")

    @patch("vllm_tuner.utils.model_registry._HF_HUB_AVAILABLE", True)
    @patch("vllm_tuner.utils.model_registry._download_config")
    @patch("vllm_tuner.settings.settings")
    def test_returns_config_for_direct_repo(self, mock_settings, mock_download):
        mock_settings.HF_TOKEN = None
        mock_download.return_value = {"hidden_size": 4096}

        from vllm_tuner.utils.model_registry import get_model_config

        result = get_model_config("meta-llama/Llama-3-8B")
        assert result == {"hidden_size": 4096}
        mock_download.assert_called_once_with("meta-llama/Llama-3-8B", token=None)

    @patch("vllm_tuner.utils.model_registry._HF_HUB_AVAILABLE", True)
    @patch("vllm_tuner.utils.model_registry._download_config")
    @patch("vllm_tuner.settings.settings")
    def test_falls_back_to_base_repo_for_gguf(self, mock_settings, mock_download):
        mock_settings.HF_TOKEN = None
        # First call (GGUF repo) returns None, second (base repo) returns config
        mock_download.side_effect = [None, {"hidden_size": 2048}]

        from vllm_tuner.utils.model_registry import get_model_config

        result = get_model_config("Qwen/Qwen3-0.6B-GGUF:Q4_0")
        assert result == {"hidden_size": 2048}
        assert mock_download.call_count == 2

    @patch("vllm_tuner.utils.model_registry._HF_HUB_AVAILABLE", True)
    @patch("vllm_tuner.utils.model_registry._download_config")
    @patch("vllm_tuner.settings.settings")
    def test_raises_when_config_not_found(self, mock_settings, mock_download):
        mock_settings.HF_TOKEN = None
        mock_download.return_value = None

        from vllm_tuner.utils.model_registry import get_model_config

        with pytest.raises(ValueError, match="Could not find config.json"):
            get_model_config("nonexistent/model")

    @patch("vllm_tuner.utils.model_registry._HF_HUB_AVAILABLE", True)
    @patch("vllm_tuner.utils.model_registry._download_config")
    @patch("vllm_tuner.settings.settings")
    def test_uses_hf_token(self, mock_settings, mock_download):
        mock_token = MagicMock()
        mock_token.get_secret_value.return_value = "hf_test_token"
        mock_settings.HF_TOKEN = mock_token
        mock_download.return_value = {"hidden_size": 4096}

        from vllm_tuner.utils.model_registry import get_model_config

        get_model_config("org/model")
        mock_download.assert_called_once_with("org/model", token="hf_test_token")
