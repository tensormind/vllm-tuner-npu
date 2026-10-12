from __future__ import annotations

from abc import ABC, abstractmethod

from vllm_tuner.core.models import TrialResult


class StorageBackend(ABC):
    """Abstract storage backend for persisting study and trial data."""

    @abstractmethod
    def get_storage_url(self) -> str:
        """Return the Optuna-compatible storage URL."""

    @abstractmethod
    def save_trial(self, study_name: str, result: TrialResult) -> None:
        """Persist a trial result (beyond what Optuna stores)."""

    @abstractmethod
    def load_trials(self, study_name: str) -> list[TrialResult]:
        """Load all trial results for a study."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable storage backend name."""
