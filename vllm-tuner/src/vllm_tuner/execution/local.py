from __future__ import annotations

from typing import Any

from vllm_tuner.benchmarks.base import BenchmarkProvider
from vllm_tuner.core.models import BenchmarkConfig, TrialConfig, TrialResult
from vllm_tuner.core.trial import TrialRunner
from vllm_tuner.execution.base import ExecutionBackend, JobHandle
from vllm_tuner.hardware.base import HardwareMonitor
from vllm_tuner.hardware.null import NullMonitor
from vllm_tuner.helper.logging import get_logger
from vllm_tuner.vllm.launcher import VLLMLauncher


logger = get_logger()


class LocalExecutionBackend(ExecutionBackend):
    """Sequential trial execution on local machine via subprocess."""

    def __init__(
        self,
        model: str = "",
        host: str = "127.0.0.1",
        port: int = 8000,
        benchmark_provider: BenchmarkProvider | None = None,
        benchmark_config: BenchmarkConfig | None = None,
        monitor: HardwareMonitor | None = None,
        startup_timeout: float = 300.0,
        dashboard: Any | None = None,
    ):
        self._model = model
        self._host = host
        self._port = port
        self._benchmark_provider = benchmark_provider
        self._benchmark_config = benchmark_config or BenchmarkConfig()
        self._monitor = monitor or NullMonitor()
        self._startup_timeout = startup_timeout
        self._completed_results: dict[str, TrialResult] = {}
        self._active_launcher: VLLMLauncher | None = None
        self._dashboard = dashboard

    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        logger.info(
            "Local backend: running trial #{} with params: {}",
            trial_config.trial_number,
            trial_config.parameters,
        )
        handle = JobHandle(
            job_id=f"local-{trial_config.trial_number}",
            trial_number=trial_config.trial_number,
            backend=self.name,
        )

        # Synchronous execution — run trial right now
        log_callback = self._dashboard.on_server_log if self._dashboard else None
        launcher = VLLMLauncher(
            model=self._model,
            host=self._host,
            port=self._port,
            log_callback=log_callback,
        )
        self._active_launcher = launcher

        if self._dashboard:
            self._dashboard.on_trial_start(trial_config.trial_number, trial_config.parameters)

        runner = TrialRunner(
            launcher=launcher,
            monitor=self._monitor,
            benchmark_provider=self._benchmark_provider,
            benchmark_config=self._benchmark_config,
            startup_timeout=self._startup_timeout,
            dashboard=self._dashboard,
        )
        result = runner.run_trial(trial_config)
        self._active_launcher = None
        self._completed_results[handle.job_id] = result

        if self._dashboard:
            self._dashboard.on_trial_complete(result)

        return handle

    def poll_trials(self, handles: list[JobHandle]) -> tuple[list[TrialResult], list[JobHandle]]:
        results = []
        remaining = []
        for handle in handles:
            result = self._completed_results.pop(handle.job_id, None)
            if result is not None:
                results.append(result)
            else:
                remaining.append(handle)
        return results, remaining

    def cleanup(self) -> None:
        if self._active_launcher is not None:
            self._active_launcher.stop()
            self._active_launcher = None
        self._completed_results.clear()
        logger.info("Local backend: cleanup complete")

    @property
    def name(self) -> str:
        return "local"

    @property
    def supports_parallel(self) -> bool:
        return False
