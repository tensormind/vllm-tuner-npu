from __future__ import annotations

from vllm_tuner.core.models import AcceleratorInfo, AggregateHardwareStats, HardwareStats
from vllm_tuner.hardware.base import HardwareMonitor


class NullMonitor(HardwareMonitor):
    """No-op monitor for environments without hardware access (CI, tests)."""

    def start_collection(self, interval_seconds: float = 1.0) -> None:
        pass

    def stop_collection(self) -> None:
        pass

    def get_current_stats(self) -> HardwareStats:
        return HardwareStats()

    def get_aggregate_stats(self) -> AggregateHardwareStats:
        return AggregateHardwareStats()

    def get_accelerator_info(self) -> AcceleratorInfo:
        return AcceleratorInfo(name="none", count=0, memory_per_device_mb=0.0, total_memory_mb=0.0)
