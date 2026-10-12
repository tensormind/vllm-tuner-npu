import pytest

from vllm_tuner.core.models import (
    BenchmarkConfig,
    BenchmarkResult,
    StudyConfig,
    TrialConfig,
    TrialResult,
    TrialStatus,
)


@pytest.fixture
def sample_trial_config():
    return TrialConfig(
        trial_number=1,
        parameters={"gpu_memory_utilization": 0.9, "max_num_seqs": 256},
        static_parameters={"tensor_parallel_size": 2, "max_model_len": 4096},
    )


@pytest.fixture
def sample_benchmark_result():
    return BenchmarkResult(
        throughput_req_per_sec=10.5,
        output_tokens_per_sec=2100.0,
        p50_latency_ms=85.0,
        p95_latency_ms=145.0,
        p99_latency_ms=210.0,
        ttft_ms=42.0,
        itl_ms=12.0,
        total_requests=50,
        successful_requests=50,
    )


@pytest.fixture
def sample_trial_result(sample_trial_config, sample_benchmark_result):
    return TrialResult(
        trial_number=1,
        status=TrialStatus.COMPLETED,
        config=sample_trial_config,
        benchmark=sample_benchmark_result,
        duration_seconds=120.5,
    )


@pytest.fixture
def sample_study_config():
    return StudyConfig(model="meta-llama/Llama-3.1-8B-Instruct")


@pytest.fixture
def sample_benchmark_config():
    return BenchmarkConfig()
