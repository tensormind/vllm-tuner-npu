from __future__ import annotations

from vllm_tuner.core.models import AcceleratorInfo, AcceleratorType, AggregateHardwareStats, HardwareStats
from vllm_tuner.hardware.base import HardwareMonitor
from vllm_tuner.helper.logging import get_logger


logger = get_logger()


class TPUMonitor(HardwareMonitor):
    """Google TPU monitoring via GKE Metrics API.

    Currently a mock — returns dummy stats.
    Real implementation will query Cloud Monitoring API or libtpu runtime stats.
    """

    def __init__(self, tpu_type: str = "v6e-4"):
        self._tpu_type = tpu_type
        self._collecting = False
        logger.info("TPUMonitor initialized for type: {}", tpu_type)

    def start_collection(self, interval_seconds: float = 1.0) -> None:
        logger.info("TPUMonitor: starting collection (interval={}s)", interval_seconds)
        self._collecting = True
        # TODO: start background thread querying GKE Metrics API

    def stop_collection(self) -> None:
        logger.info("TPUMonitor: stopping collection")
        self._collecting = False

    def get_current_stats(self) -> HardwareStats:
        raise NotImplementedError("TPUMonitor: GKE Metrics API integration not yet implemented")

    def get_aggregate_stats(self) -> AggregateHardwareStats:
        raise NotImplementedError("TPUMonitor: GKE Metrics API integration not yet implemented")

    def get_accelerator_info(self) -> AcceleratorInfo:
        return AcceleratorInfo(
            accelerator_type=AcceleratorType.TPU,
            name=f"Google TPU {self._tpu_type}",
            count=int(self._tpu_type.split("-")[-1]) if "-" in self._tpu_type else 1,
            memory_per_device_mb=16384.0,
            total_memory_mb=32768.0,
        )
