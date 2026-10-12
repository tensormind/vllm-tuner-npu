from __future__ import annotations

import threading

from vllm_tuner.core.models import AcceleratorInfo, AcceleratorType, AggregateHardwareStats, HardwareStats
from vllm_tuner.hardware.base import HardwareMonitor
from vllm_tuner.helper.logging import get_logger


logger = get_logger()

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


class NVMLMonitor(HardwareMonitor):
    """NVIDIA GPU monitoring via pynvml.

    Falls back to zero-value stats if pynvml (nvidia-ml-py) is not installed.
    Install with: pip install 'llm-vllm-tuner[gpu]'
    """

    def __init__(self, device_ids: list[int] | None = None):
        self._collecting = False
        self._samples: list[HardwareStats] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._initialized = False

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._initialized = True
                if device_ids is not None:
                    self._device_ids = device_ids
                else:
                    count = pynvml.nvmlDeviceGetCount()
                    self._device_ids = list(range(count))
                logger.info("NVMLMonitor initialized for devices: {}", self._device_ids)
            except pynvml.NVMLError as e:
                self._device_ids = device_ids or [0]
                logger.warning("Failed to initialize NVML: {} — GPU monitoring disabled", e)
        else:
            self._device_ids = device_ids or [0]
            logger.warning("pynvml not installed — GPU monitoring disabled. Install: pip install 'llm-vllm-tuner[gpu]'")

    def start_collection(self, interval_seconds: float = 1.0) -> None:
        if self._collecting:
            return
        self._collecting = True
        self._samples = []
        self._stop_event.clear()

        if not self._initialized:
            logger.debug("NVMLMonitor: collection started (no-op, pynvml unavailable)")
            return

        self._thread = threading.Thread(
            target=self._collection_loop,
            args=(interval_seconds,),
            daemon=True,
        )
        self._thread.start()
        logger.info("NVMLMonitor: background collection started (interval={}s)", interval_seconds)

    def stop_collection(self) -> None:
        if not self._collecting:
            return
        self._collecting = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("NVMLMonitor: collection stopped ({} samples)", len(self._samples))

    def _collection_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            try:
                stats = self._query_devices()
                self._samples.append(stats)
            except Exception:
                logger.debug("NVMLMonitor: failed to query device stats")
            self._stop_event.wait(interval)

    def get_current_stats(self) -> HardwareStats:
        if not self._initialized:
            return HardwareStats()
        try:
            return self._query_devices()
        except Exception:
            logger.debug("NVMLMonitor: failed to get current stats")
            return HardwareStats()

    def _query_devices(self) -> HardwareStats:
        """Query NVML for stats of the first device (aggregate across devices)."""
        total_used = 0.0
        total_mem = 0.0
        total_util = 0.0
        max_temp = 0.0
        total_power = 0.0
        total_limit = 0.0

        for dev_id in self._device_ids:
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_used += mem_info.used / (1024 * 1024)
            total_mem += mem_info.total / (1024 * 1024)

            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            total_util += util.gpu

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            max_temp = max(max_temp, temp)

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                total_power += power
                total_limit += limit
            except pynvml.NVMLError:
                pass

        n = len(self._device_ids)
        return HardwareStats(
            memory_used_mb=total_used,
            memory_total_mb=total_mem,
            memory_utilization=total_used / total_mem if total_mem > 0 else 0.0,
            accelerator_utilization=total_util / (n * 100) if n > 0 else 0.0,
            temperature_c=max_temp,
            power_usage_w=total_power,
            power_limit_w=total_limit,
        )

    def get_aggregate_stats(self) -> AggregateHardwareStats:
        if not self._samples:
            return AggregateHardwareStats()

        peak_mem = max(s.memory_used_mb for s in self._samples)
        avg_mem = sum(s.memory_used_mb for s in self._samples) / len(self._samples)
        avg_util = sum(s.accelerator_utilization for s in self._samples) / len(self._samples)
        max_temp = max(s.temperature_c for s in self._samples)
        total_power = sum(s.power_usage_w for s in self._samples) / len(self._samples)

        return AggregateHardwareStats(
            peak_memory_used_mb=peak_mem,
            avg_memory_used_mb=avg_mem,
            avg_utilization=avg_util,
            max_temperature_c=max_temp,
            total_power_w=total_power,
            num_samples=len(self._samples),
        )

    def get_accelerator_info(self) -> AcceleratorInfo:
        if not self._initialized:
            return AcceleratorInfo(
                accelerator_type=AcceleratorType.GPU,
                name="unknown (pynvml unavailable)",
                count=len(self._device_ids),
            )

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_ids[0])
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_per_device = mem_info.total / (1024 * 1024)
        except pynvml.NVMLError:
            name = "unknown"
            mem_per_device = 0.0

        return AcceleratorInfo(
            accelerator_type=AcceleratorType.GPU,
            name=name,
            count=len(self._device_ids),
            memory_per_device_mb=mem_per_device,
            total_memory_mb=mem_per_device * len(self._device_ids),
        )
