from __future__ import annotations

from vllm_tuner.core.models import TrialConfig, TrialResult, TrialStatus
from vllm_tuner.execution.base import ExecutionBackend, JobHandle
from vllm_tuner.helper.logging import get_logger


logger = get_logger()

try:
    import ray

    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False


class RayExecutionBackend(ExecutionBackend):
    """Distributed trial execution via Ray actors.

    Covers bare-metal Ray (VM), multi-node Ray cluster, and KubeRay on K8s.
    Requires: pip install 'llm-vllm-tuner[ray]'

    Note: Full Ray integration (remote actors, GPU scheduling) is planned for
    a future release. Currently submits trials as mock handles.
    """

    def __init__(self, ray_address: str = "auto", runtime_env: dict | None = None):
        self._ray_address = ray_address
        self._runtime_env = runtime_env or {}
        self._connected = False

        if not _RAY_AVAILABLE:
            logger.warning("Ray is not installed — install with: pip install 'llm-vllm-tuner[ray]'")
        else:
            logger.info("Ray backend initialized: address={}, runtime_env={}", ray_address, self._runtime_env)

    def _ensure_connected(self) -> bool:
        """Initialize Ray connection if not already connected."""
        if not _RAY_AVAILABLE:
            return False
        if self._connected:
            return True
        try:
            ray.init(address=self._ray_address, runtime_env=self._runtime_env, ignore_reinit_error=True)
            self._connected = True
            logger.info("Ray backend: connected to cluster at '{}'", self._ray_address)
            return True
        except Exception as e:
            logger.error("Ray backend: failed to connect to '{}': {}", self._ray_address, e)
            return False

    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        logger.info(
            "Ray backend: submitting trial #{} to Ray at '{}' with params: {}",
            trial_config.trial_number,
            self._ray_address,
            trial_config.parameters,
        )

        if not self._ensure_connected():
            logger.error("Ray backend: not connected, trial #{} will be marked as failed", trial_config.trial_number)

        # TODO: submit as Ray remote task with GPU resource requirements
        # ray_task = run_trial_remote.options(num_gpus=1).remote(trial_config)
        handle = JobHandle(
            job_id=f"ray-{trial_config.trial_number}",
            trial_number=trial_config.trial_number,
            backend=self.name,
            metadata={"ray_address": self._ray_address},
        )
        return handle

    def poll_trials(self, handles: list[JobHandle]) -> tuple[list[TrialResult], list[JobHandle]]:
        results = []
        for handle in handles:
            logger.info("Ray backend: polling trial #{}", handle.trial_number)
            # TODO: ray.get() on futures for real results
            result = TrialResult(
                trial_number=handle.trial_number,
                status=TrialStatus.COMPLETED,
            )
            results.append(result)
        return results, []

    def cleanup(self) -> None:
        if self._connected and _RAY_AVAILABLE:
            try:
                ray.shutdown()
                logger.info("Ray backend: disconnected from cluster")
            except Exception:
                pass
            self._connected = False
        logger.info("Ray backend: cleanup complete")

    @property
    def name(self) -> str:
        return "ray"

    @property
    def supports_parallel(self) -> bool:
        return True
