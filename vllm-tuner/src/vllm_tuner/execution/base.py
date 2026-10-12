from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from vllm_tuner.core.models import TrialConfig, TrialResult


@dataclass
class JobHandle:
    """Handle representing a submitted trial job."""

    job_id: str
    trial_number: int
    backend: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionBackend(ABC):
    """Abstract base class for trial execution backends."""

    @abstractmethod
    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit a trial for execution. Returns a handle for tracking."""

    @abstractmethod
    def poll_trials(self, handles: list[JobHandle]) -> tuple[list[TrialResult], list[JobHandle]]:
        """Check submitted trials. Returns (completed_results, still_running_handles)."""

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources held by the backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""

    @property
    @abstractmethod
    def supports_parallel(self) -> bool:
        """Whether this backend supports parallel trial execution."""
