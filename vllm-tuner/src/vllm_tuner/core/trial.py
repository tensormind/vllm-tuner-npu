from __future__ import annotations

import time
from typing import Any

from vllm_tuner.benchmarks.base import BenchmarkProvider
from vllm_tuner.core.models import BenchmarkConfig, TrialConfig, TrialResult, TrialStatus, VLLMTelemetry
from vllm_tuner.hardware.base import HardwareMonitor
from vllm_tuner.hardware.null import NullMonitor
from vllm_tuner.helper.logging import get_logger
from vllm_tuner.vllm.launcher import VLLMLauncher
from vllm_tuner.vllm.telemetry import VLLMTelemetryParser


logger = get_logger()


class TrialRunner:
    """Executes a single trial: start vLLM, benchmark, collect metrics.

    Wires together VLLMLauncher, HardwareMonitor, and BenchmarkProvider.
    """

    def __init__(
        self,
        launcher: VLLMLauncher | None = None,
        monitor: HardwareMonitor | None = None,
        benchmark_provider: BenchmarkProvider | None = None,
        benchmark_config: BenchmarkConfig | None = None,
        startup_timeout: float = 300.0,
        dashboard: Any | None = None,
    ):
        self._launcher = launcher
        self._monitor = monitor or NullMonitor()
        self._benchmark_provider = benchmark_provider
        self._benchmark_config = benchmark_config or BenchmarkConfig()
        self._startup_timeout = startup_timeout
        self._telemetry_parser = VLLMTelemetryParser()
        self._dashboard = dashboard

    def run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Execute a complete trial lifecycle."""
        start_time = time.monotonic()
        trial_num = trial_config.trial_number

        logger.info("Trial #{}: starting with params {}", trial_num, trial_config.parameters)

        # If no launcher or benchmark provider, run in dry-run mode
        if self._launcher is None or self._benchmark_provider is None:
            duration = time.monotonic() - start_time
            logger.info(
                "Trial #{}: dry-run completed in {:.2f}s (no launcher or benchmark provider)", trial_num, duration
            )
            return TrialResult(
                trial_number=trial_num,
                status=TrialStatus.COMPLETED,
                config=trial_config,
                duration_seconds=duration,
            )

        try:
            return self._execute_trial(trial_config, start_time)
        except Exception as e:
            duration = time.monotonic() - start_time
            logger.error("Trial #{}: failed with error: {}", trial_num, e)
            return TrialResult(
                trial_number=trial_num,
                status=TrialStatus.FAILED,
                config=trial_config,
                duration_seconds=duration,
                error_message=str(e),
            )

    def _execute_trial(self, trial_config: TrialConfig, start_time: float) -> TrialResult:
        """Core trial execution with real components."""
        trial_num = trial_config.trial_number
        telemetry = VLLMTelemetry()
        hardware_stats = None

        try:
            # Step 1: Start vLLM server
            logger.info("Trial #{}: starting vLLM server", trial_num)
            if self._dashboard:
                self._dashboard.on_server_starting("")
            self._launcher.start(trial_config)

            # Step 2: Wait for server readiness
            logger.info("Trial #{}: waiting for server readiness", trial_num)
            if not self._launcher.wait_until_ready(timeout=self._startup_timeout):
                logs = self._launcher.read_logs()
                telemetry = self._telemetry_parser.parse_logs(logs)

                tail_lines = logs[-20:] if logs else []
                error_lines = [
                    line
                    for line in tail_lines
                    if any(kw in line.lower() for kw in ("error", "exception", "fatal", "failed", "traceback"))
                ]
                last_error = (
                    error_lines[-1].strip()
                    if error_lines
                    else (tail_lines[-1].strip() if tail_lines else "No server logs captured")
                )

                if telemetry.oom_detected:
                    error = f"OOM detected during startup: {last_error}"
                else:
                    error = f"Server failed to start: {last_error}"

                logger.error(
                    "Trial #{}: {} | Last {} log lines:\n{}", trial_num, error, len(tail_lines), "\n".join(tail_lines)
                )

                if self._dashboard:
                    self._dashboard.on_server_failed(last_error)

                return TrialResult(
                    trial_number=trial_num,
                    status=TrialStatus.FAILED,
                    config=trial_config,
                    duration_seconds=time.monotonic() - start_time,
                    telemetry=telemetry,
                    error_message=error,
                )

            if self._dashboard:
                self._dashboard.on_server_ready()

            # Step 3: Start hardware monitoring
            logger.info("Trial #{}: starting hardware monitoring", trial_num)
            self._monitor.start_collection()

            # Step 4: Run benchmark
            logger.info("Trial #{}: running benchmark", trial_num)
            if self._dashboard:
                self._dashboard.on_benchmark_start(trial_num)
                self._benchmark_provider.log_callback = self._dashboard.on_server_log
            benchmark_result = self._benchmark_provider.run(
                self._launcher.server_url,
                self._benchmark_config,
            )

            # Detect empty benchmark result (provider failed silently)
            if benchmark_result.total_requests == 0 and benchmark_result.throughput_req_per_sec == 0.0:
                provider_error = getattr(self._benchmark_provider, "last_error", "")
                error_msg = provider_error or "Benchmark returned no results"
                if self._dashboard:
                    self._dashboard.on_benchmark_error(trial_num, error_msg)
                logs = self._launcher.read_logs()
                telemetry = self._telemetry_parser.parse_logs(logs)
                return TrialResult(
                    trial_number=trial_num,
                    status=TrialStatus.FAILED,
                    config=trial_config,
                    duration_seconds=time.monotonic() - start_time,
                    telemetry=telemetry,
                    error_message=error_msg,
                )

            # Step 5: Stop hardware monitoring
            self._monitor.stop_collection()
            hardware_stats = self._monitor.get_aggregate_stats()
            logger.info("Trial #{}: hardware stats collected ({} samples)", trial_num, hardware_stats.num_samples)

            # Step 6: Parse server logs for telemetry
            logs = self._launcher.read_logs()
            telemetry = self._telemetry_parser.parse_logs(logs)

            duration = time.monotonic() - start_time
            logger.info(
                "Trial #{}: completed in {:.2f}s (throughput={:.2f} req/s)",
                trial_num,
                duration,
                benchmark_result.throughput_req_per_sec,
            )

            return TrialResult(
                trial_number=trial_num,
                status=TrialStatus.COMPLETED,
                config=trial_config,
                benchmark=benchmark_result,
                telemetry=telemetry,
                hardware=hardware_stats,
                duration_seconds=duration,
            )
        finally:
            # Always stop server and monitor
            self._monitor.stop_collection()
            self._launcher.stop()
            logger.info("Trial #{}: cleanup complete", trial_num)
