import pytest

from vllm_tuner.core.models import TrialStatus
from vllm_tuner.execution.base import ExecutionBackend, JobHandle
from vllm_tuner.execution.local import LocalExecutionBackend
from vllm_tuner.execution.ray_backend import RayExecutionBackend


class TestJobHandle:
    def test_creation(self):
        handle = JobHandle(job_id="test-1", trial_number=1, backend="local")
        assert handle.job_id == "test-1"
        assert handle.trial_number == 1
        assert handle.metadata == {}


class TestExecutionBackendABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            ExecutionBackend()


class TestLocalExecutionBackend:
    def test_name(self):
        backend = LocalExecutionBackend()
        assert backend.name == "local"

    def test_not_parallel(self):
        backend = LocalExecutionBackend()
        assert backend.supports_parallel is False

    def test_submit_trial(self, sample_trial_config):
        backend = LocalExecutionBackend()
        handle = backend.submit_trial(sample_trial_config)
        assert handle.trial_number == sample_trial_config.trial_number
        assert handle.backend == "local"
        assert "local-" in handle.job_id

    def test_poll_trials(self, sample_trial_config):
        backend = LocalExecutionBackend()
        handle = backend.submit_trial(sample_trial_config)
        results, remaining = backend.poll_trials([handle])
        assert len(results) == 1
        assert len(remaining) == 0
        assert results[0].status == TrialStatus.COMPLETED

    def test_cleanup(self):
        backend = LocalExecutionBackend()
        backend.cleanup()  # should not raise


class TestRayExecutionBackend:
    def test_name(self):
        backend = RayExecutionBackend()
        assert backend.name == "ray"

    def test_parallel(self):
        backend = RayExecutionBackend()
        assert backend.supports_parallel is True

    def test_custom_address(self):
        backend = RayExecutionBackend(ray_address="ray://head:10001")
        assert backend._ray_address == "ray://head:10001"

    def test_submit_trial(self, sample_trial_config):
        backend = RayExecutionBackend()
        handle = backend.submit_trial(sample_trial_config)
        assert handle.trial_number == sample_trial_config.trial_number
        assert handle.backend == "ray"
        assert handle.metadata["ray_address"] == "auto"

    def test_poll_trials(self, sample_trial_config):
        backend = RayExecutionBackend()
        handle = backend.submit_trial(sample_trial_config)
        results, remaining = backend.poll_trials([handle])
        assert len(results) == 1
        assert len(remaining) == 0

    def test_cleanup(self):
        backend = RayExecutionBackend()
        backend.cleanup()
