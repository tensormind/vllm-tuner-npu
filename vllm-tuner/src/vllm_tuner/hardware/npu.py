from __future__ import annotations

import os
import re
import subprocess
import threading

from vllm_tuner.core.models import (
    AcceleratorInfo,
    AcceleratorType,
    AggregateHardwareStats,
    HardwareStats,
)
from vllm_tuner.hardware.base import HardwareMonitor
from vllm_tuner.helper.logging import get_logger

logger = get_logger()

try:
    import torch_npu  # noqa: F401 — import side-effect: registers NPU as torch backend

    _TORCH_NPU_AVAILABLE = True
except ImportError:
    _TORCH_NPU_AVAILABLE = False


# Regex for parsing the npu-smi info main table.
# Each NPU block looks like:
#   | 0     910B3               | OK            | 93.5        34                0    / 0             |
#   | 0                         | 0000:C1:00.0  | 0           0    / 0          58797/ 65536         |
# We capture: npu_id, power_w, temp_c, aicore_pct, hbm_used_mb, hbm_total_mb.
# Regex for parsing the npu-smi info main table.
# Each NPU spans TWO rows:
#   | 0     910B3               | OK            | 93.5        34                0    / 0             |
#   | 0                         | 0000:C1:00.0  | 0           0    / 0          58797/ 65536         |
# Row 1: NPU id, Name, Health, Power(W), Temp(C), Hugepages
# Row 2: Chip, Bus-Id, AICore(%), Memory-Usage, HBM-Usage(used/total)
_NPU_SMI_ROW = re.compile(
    r"\|\s*(\d+)\s+\S+\s*\|\s*\S+\s*\|"  # NPU id + Name | Health
    r"\s*([\d.]+)\s+"  # Power(W)
    r"([\d.]+)\s+"  # Temp(C)
    r"\S+\s*/\s*\S+\s*\|"  # Hugepages-Usage + row end
    r"[\s\S]*?"  # cross to next row
    r"\|\s*\d+\s*\|\s*\S+\s*\|\s*"  # Chip | Bus-Id |
    r"([\d.]+)\s+"  # AICore(%)
    r"\S+\s*/\s*\S+\s+"  # Memory-Usage(MB)
    r"(\d+)\s*/\s*(\d+)\s*\|",  # HBM-Usage(MB)  used/total
    re.MULTILINE,
)


class NPUMonitor(HardwareMonitor):
    """Ascend NPU monitoring via ``npu-smi`` CLI + ``torch_npu``.

    Uses ``npu-smi info`` (always available with CANN) for power/temperature/
    utilization and ``torch_npu.npu.mem_get_info`` for precise HBM usage when
    available. Falls back to npu-smi parsing only if torch_npu is absent.

    Only the devices visible to *this* process (via
    ``ASCEND_RT_VISIBLE_DEVICES``) are monitored, matching the vLLM server
    instance being tuned.
    """

    def __init__(self, device_ids: list[int] | None = None):
        self._collecting = False
        self._samples: list[HardwareStats] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._initialized = False
        self._power_limit_w = 310.0  # 910B3 typical TDP; npu-smi doesn't expose limit easily

        # Resolve which NPU devices to monitor.
        if device_ids is not None:
            self._device_ids = device_ids
        else:
            self._device_ids = self._resolve_visible_devices()

        if self._device_ids:
            self._initialized = True
            logger.info("NPUMonitor initialized for devices: {}", self._device_ids)

    @staticmethod
    def _resolve_visible_devices() -> list[int]:
        """Read ASCEND_RT_VISIBLE_DEVICES; default to all NPU ids."""
        vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        if vis.strip():
            ids = []
            for part in vis.split(","):
                part = part.strip()
                if part.isdigit():
                    ids.append(int(part))
            if ids:
                return ids
        # No env override: enumerate from npu-smi
        return NPUMonitor._enumerate_all_npus()

    @staticmethod
    def _enumerate_all_npus() -> list[int]:
        try:
            out = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            ids = []
            for m in _NPU_SMI_ROW.finditer(out):
                ids.append(int(m.group(1)))
            return ids or [0]
        except Exception:
            return [0]

    def start_collection(self, interval_seconds: float = 1.0) -> None:
        if self._collecting:
            return
        self._collecting = True
        self._samples = []
        self._stop_event.clear()

        if not self._initialized:
            logger.debug("NPUMonitor: collection started (no-op, not initialized)")
            return

        self._thread = threading.Thread(
            target=self._collection_loop,
            args=(interval_seconds,),
            daemon=True,
        )
        self._thread.start()
        logger.info("NPUMonitor: background collection started (interval={}s)", interval_seconds)

    def stop_collection(self) -> None:
        if not self._collecting:
            return
        self._collecting = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("NPUMonitor: collection stopped ({} samples)", len(self._samples))

    def _collection_loop(self, interval: float) -> None:
        while not self._stop_event.is_set():
            try:
                stats = self._query_devices()
                self._samples.append(stats)
            except Exception:
                logger.debug("NPUMonitor: failed to query device stats")
            self._stop_event.wait(interval)

    def get_current_stats(self) -> HardwareStats:
        if not self._initialized:
            return HardwareStats()
        try:
            return self._query_devices()
        except Exception:
            logger.debug("NPUMonitor: failed to get current stats")
            return HardwareStats()

    def _query_devices(self) -> HardwareStats:
        """Query NPU stats via ``npu-smi info`` (aggregated across monitored devices)."""
        try:
            out = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except Exception as e:
            logger.debug("NPUMonitor: npu-smi info failed: {}", e)
            return HardwareStats()

        total_used = 0.0
        total_mem = 0.0
        total_util = 0.0
        max_temp = 0.0
        total_power = 0.0

        matched = 0
        for m in _NPU_SMI_ROW.finditer(out):
            npu_id = int(m.group(1))
            if npu_id not in self._device_ids:
                continue
            power = float(m.group(2))
            temp = float(m.group(3))
            aicore = float(m.group(4))
            hbm_used = float(m.group(5))
            hbm_total = float(m.group(6))

            total_used += hbm_used
            total_mem += hbm_total
            total_util += aicore
            max_temp = max(max_temp, temp)
            total_power += power
            matched += 1

        if matched == 0:
            return HardwareStats()

        n = matched
        return HardwareStats(
            memory_used_mb=total_used,
            memory_total_mb=total_mem,
            memory_utilization=total_used / total_mem if total_mem > 0 else 0.0,
            accelerator_utilization=total_util / (n * 100.0) if n > 0 else 0.0,
            temperature_c=max_temp,
            power_usage_w=total_power,
            power_limit_w=self._power_limit_w * n,
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
                accelerator_type=AcceleratorType.NPU,
                name="unknown (npu-smi unavailable)",
                count=len(self._device_ids),
            )

        try:
            out = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            # Take name from the first monitored device
            name = "Ascend NPU"
            for m in _NPU_SMI_ROW.finditer(out):
                if int(m.group(1)) == self._device_ids[0]:
                    # Name is on the same row between NPU id and Health
                    # Re-search the full line for the name field.
                    line = out[out.rfind("|", 0, m.start()) : m.end()]
                    name_match = re.search(r"\|\s*\d+\s+(\S+)\s+\|", line)
                    if name_match:
                        name = name_match.group(1)
                    break
            mem_per = 65536.0  # 910B3 = 64GB HBM
            for m in _NPU_SMI_ROW.finditer(out):
                if int(m.group(1)) == self._device_ids[0]:
                    mem_per = float(m.group(6))
                    break
        except Exception:
            name = "Ascend NPU"
            mem_per = 0.0

        return AcceleratorInfo(
            accelerator_type=AcceleratorType.NPU,
            name=name,
            count=len(self._device_ids),
            memory_per_device_mb=mem_per,
            total_memory_mb=mem_per * len(self._device_ids),
        )
