from __future__ import annotations

from typing import Any

from vllm_tuner.benchmarks.base import BenchmarkProvider
from vllm_tuner.core.models import BenchmarkResult, StudyConfig, TrialConfig
from vllm_tuner.core.trial import TrialRunner
from vllm_tuner.hardware.base import HardwareMonitor
from vllm_tuner.hardware.null import NullMonitor
from vllm_tuner.helper.logging import get_logger
from vllm_tuner.vllm.launcher import VLLMLauncher


logger = get_logger()


class BaselineRunner:
    """Runs a pre-optimization baseline benchmark for comparison.

    Starts vLLM with default parameters (no tuning) and runs a single benchmark.
    """

    def __init__(
        self,
        benchmark_provider: BenchmarkProvider | None = None,
        monitor: HardwareMonitor | None = None,
        dashboard: Any | None = None,
    ):
        self._benchmark_provider = benchmark_provider
        self._monitor = monitor or NullMonitor()
        self._dashboard = dashboard

    def run_baseline(self, config: StudyConfig) -> BenchmarkResult:
        """Run baseline benchmark with default vLLM parameters.

        Skipped entirely when ``config.baseline.enabled`` is False — avoids
        launching a no-tuning vLLM server (and a full benchmark pass) that the
        user did not request.
        """
        if not config.baseline.enabled:
            logger.info("BaselineRunner: baseline disabled (baseline.enabled=False), skipping")
            return BenchmarkResult()

        logger.info(
            "BaselineRunner: running baseline for model '{}' with {} requests",
            config.model,
            config.baseline.num_requests,
        )

        if self._benchmark_provider is None:
            logger.info("BaselineRunner: no benchmark provider — returning empty result")
            return BenchmarkResult()

        if self._dashboard:
            self._dashboard.on_baseline_start()

        log_callback = self._dashboard.on_server_log if self._dashboard else None
        launcher = VLLMLauncher(model=config.model, log_callback=log_callback)
        baseline_config = TrialConfig(
            trial_number=0,
            parameters={},
            static_parameters=config.static_parameters,
        )

        runner = TrialRunner(
            launcher=launcher,
            monitor=self._monitor,
            benchmark_provider=self._benchmark_provider,
            benchmark_config=config.benchmark,
            dashboard=self._dashboard,
        )

        result = runner.run_trial(baseline_config)

        if result.benchmark is not None:
            logger.info(
                "BaselineRunner: baseline throughput={:.2f} req/s, p95={:.1f}ms",
                result.benchmark.throughput_req_per_sec,
                result.benchmark.p95_latency_ms,
            )
            return result.benchmark

        logger.warning("BaselineRunner: no benchmark data from baseline run")
        return BenchmarkResult()
