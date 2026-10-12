from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from vllm_tuner.core.models import BenchmarkConfig, BenchmarkResult


class BenchmarkProvider(ABC):
    """Abstract benchmark execution provider."""

    last_error: str = ""
    log_callback: Callable[[str], Any] | None = None

    @abstractmethod
    def run(self, server_url: str, config: BenchmarkConfig) -> BenchmarkResult:
        """Execute benchmark against a vLLM server and return results."""

    @abstractmethod
    def supports_synthetic_workloads(self) -> bool:
        """Whether this provider can generate synthetic prompts."""

    @abstractmethod
    def supports_real_datasets(self) -> bool:
        """Whether this provider can use real HF datasets or JSONL files."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
