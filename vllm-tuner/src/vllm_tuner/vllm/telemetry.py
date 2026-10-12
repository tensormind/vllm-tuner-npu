from __future__ import annotations

import re

from vllm_tuner.core.models import VLLMTelemetry
from vllm_tuner.helper.logging import get_logger


logger = get_logger()

# Regex patterns for vLLM log parsing
_OOM_PATTERN = re.compile(
    r"CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError"
    r"|NPU out of memory|acl.*OOM|HBM.*insufficient|ERR99999",
    re.IGNORECASE,
)
_KV_CACHE_PATTERN = re.compile(r"KV cache usage:\s*([\d.]+)%", re.IGNORECASE)
_PREEMPTION_PATTERN = re.compile(r"preempt(?:ion|ed)", re.IGNORECASE)
_SWAP_PATTERN = re.compile(r"swap(?:ped|ping)", re.IGNORECASE)


class VLLMTelemetryParser:
    """Parses vLLM server logs for telemetry data (KV cache, OOM, preemption, swap)."""

    def parse_logs(self, log_lines: list[str]) -> VLLMTelemetry:
        """Parse vLLM log output and extract telemetry."""
        logger.debug("Parsing {} lines of vLLM logs", len(log_lines))

        oom_detected = False
        kv_cache_usage: float | None = None
        num_preemptions = 0
        num_swaps = 0

        for line in log_lines:
            if _OOM_PATTERN.search(line):
                oom_detected = True

            match = _KV_CACHE_PATTERN.search(line)
            if match:
                kv_cache_usage = float(match.group(1)) / 100.0

            if _PREEMPTION_PATTERN.search(line):
                num_preemptions += 1

            if _SWAP_PATTERN.search(line):
                num_swaps += 1

        return VLLMTelemetry(
            kv_cache_usage=kv_cache_usage,
            num_preemptions=num_preemptions,
            num_swaps=num_swaps,
            oom_detected=oom_detected,
        )

    def detect_oom(self, log_lines: list[str]) -> bool:
        """Detect if any log line indicates an OOM error."""
        return any(_OOM_PATTERN.search(line) for line in log_lines)

    def get_kv_cache_usage(self, log_lines: list[str]) -> float | None:
        """Extract the last reported KV cache usage percentage."""
        last_usage = None
        for line in log_lines:
            match = _KV_CACHE_PATTERN.search(line)
            if match:
                last_usage = float(match.group(1)) / 100.0
        return last_usage
