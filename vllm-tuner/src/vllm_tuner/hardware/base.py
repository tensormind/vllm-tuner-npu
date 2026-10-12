from __future__ import annotations

from abc import ABC, abstractmethod

from vllm_tuner.core.models import AcceleratorInfo, AggregateHardwareStats, HardwareStats


class HardwareMonitor(ABC):
    """Abstract hardware monitoring — works for GPU, TPU, or no monitoring."""

    @abstractmethod
    def start_collection(self, interval_seconds: float = 1.0) -> None:
        """Start background metrics collection at the given interval."""

    @abstractmethod
    def stop_collection(self) -> None:
        """Stop background metrics collection."""

    @abstractmethod
    def get_current_stats(self) -> HardwareStats:
        """Get the most recent hardware stats snapshot."""

    @abstractmethod
    def get_aggregate_stats(self) -> AggregateHardwareStats:
        """Get aggregated stats over the entire collection period."""

    @abstractmethod
    def get_accelerator_info(self) -> AcceleratorInfo:
        """Get static information about the accelerator hardware."""
