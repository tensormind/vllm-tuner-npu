import pytest

from vllm_tuner.benchmarks.base import BenchmarkProvider
from vllm_tuner.benchmarks.guidellm import GuideLLMProvider
from vllm_tuner.benchmarks.http_client import HTTPBenchmarkProvider
from vllm_tuner.benchmarks.vllm_benchmark import VLLMBenchmarkProvider
from vllm_tuner.core.models import BenchmarkResult


class TestBenchmarkProviderABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BenchmarkProvider()


class TestGuideLLMProvider:
    def test_name(self):
        provider = GuideLLMProvider()
        assert provider.name == "guidellm"

    def test_supports_synthetic(self):
        assert GuideLLMProvider().supports_synthetic_workloads() is True

    def test_supports_real(self):
        assert GuideLLMProvider().supports_real_datasets() is True

    def test_run_without_guidellm_binary(self, sample_benchmark_config):
        """When guidellm is not installed, run returns empty result."""
        import shutil

        provider = GuideLLMProvider()
        if shutil.which("guidellm") is None:
            result = provider.run("http://localhost:8000", sample_benchmark_config)
            assert result.throughput_req_per_sec == 0.0


class TestVLLMBenchmarkProvider:
    def test_name(self):
        assert VLLMBenchmarkProvider().name == "vllm_benchmark"

    def test_run_without_vllm(self, sample_benchmark_config):
        """When vllm is not installed, run returns empty result."""
        provider = VLLMBenchmarkProvider()
        result = provider.run("http://localhost:8000", sample_benchmark_config)
        # Without vllm installed, subprocess fails—returns empty result
        assert isinstance(result, BenchmarkResult)


class TestHTTPBenchmarkProvider:
    def test_name(self):
        assert HTTPBenchmarkProvider().name == "http"

    def test_supports_synthetic(self):
        assert HTTPBenchmarkProvider().supports_synthetic_workloads() is True

    def test_supports_real(self):
        assert HTTPBenchmarkProvider().supports_real_datasets() is True

    def test_run_no_server(self, sample_benchmark_config):
        """When no server is running, returns empty result."""
        provider = HTTPBenchmarkProvider()
        result = provider.run("http://localhost:19999", sample_benchmark_config)
        assert result.throughput_req_per_sec == 0.0
