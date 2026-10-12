from vllm_tuner.core.models import (
    AcceleratorInfo,
    AcceleratorType,
    AggregateHardwareStats,
    BackendType,
    BenchmarkConfig,
    BenchmarkProviderType,
    BenchmarkResult,
    ConstraintSpec,
    CostReport,
    Direction,
    ExecutionConfig,
    HardwareConfig,
    HardwareStats,
    ObjectiveSpec,
    OptimizationConfig,
    ParameterSpec,
    SamplerType,
    StudyConfig,
    StudySettings,
    TrialConfig,
    TrialResult,
    TrialStatus,
    VLLMTelemetry,
)


class TestEnums:
    def test_trial_status_values(self):
        assert TrialStatus.PENDING == "pending"
        assert TrialStatus.RUNNING == "running"
        assert TrialStatus.COMPLETED == "completed"
        assert TrialStatus.FAILED == "failed"
        assert TrialStatus.PRUNED == "pruned"

    def test_direction_values(self):
        assert Direction.MINIMIZE == "minimize"
        assert Direction.MAXIMIZE == "maximize"

    def test_backend_type_values(self):
        assert BackendType.LOCAL == "local"
        assert BackendType.RAY == "ray"

    def test_accelerator_type_values(self):
        assert AcceleratorType.GPU == "gpu"
        assert AcceleratorType.TPU == "tpu"

    def test_sampler_type_values(self):
        assert SamplerType.TPE == "tpe"
        assert SamplerType.NSGA2 == "nsga2"


class TestHardwareModels:
    def test_hardware_stats_defaults(self):
        stats = HardwareStats()
        assert stats.memory_used_mb == 0.0
        assert stats.temperature_c == 0.0

    def test_hardware_stats_custom(self):
        stats = HardwareStats(memory_used_mb=8192.0, temperature_c=72.0)
        assert stats.memory_used_mb == 8192.0
        assert stats.temperature_c == 72.0

    def test_aggregate_hardware_stats_defaults(self):
        agg = AggregateHardwareStats()
        assert agg.num_samples == 0
        assert agg.peak_memory_used_mb == 0.0

    def test_accelerator_info(self):
        info = AcceleratorInfo(
            accelerator_type=AcceleratorType.GPU,
            name="NVIDIA L4",
            count=2,
            memory_per_device_mb=24576.0,
            total_memory_mb=49152.0,
        )
        assert info.accelerator_type == AcceleratorType.GPU
        assert info.count == 2


class TestBenchmarkModels:
    def test_benchmark_config_defaults(self):
        config = BenchmarkConfig()
        assert config.provider == BenchmarkProviderType.GUIDELLM
        assert config.prompt_tokens == 1500
        assert config.concurrent_requests == 50

    def test_benchmark_result_defaults(self):
        result = BenchmarkResult()
        assert result.throughput_req_per_sec == 0.0
        assert result.failed_requests == 0

    def test_benchmark_result_custom(self):
        result = BenchmarkResult(throughput_req_per_sec=12.5, p95_latency_ms=150.0)
        assert result.throughput_req_per_sec == 12.5
        assert result.p95_latency_ms == 150.0


class TestParameterModels:
    def test_parameter_spec_range(self):
        spec = ParameterSpec(name="gpu_memory_utilization", min=0.85, max=0.95, step=0.05)
        assert spec.name == "gpu_memory_utilization"
        assert spec.min == 0.85

    def test_parameter_spec_categorical(self):
        spec = ParameterSpec(name="kv_cache_dtype", options=["auto", "fp8"])
        assert spec.options == ["auto", "fp8"]
        assert spec.min is None

    def test_objective_spec(self):
        obj = ObjectiveSpec(metric="throughput_req_per_sec", direction=Direction.MAXIMIZE)
        assert obj.weight == 1.0

    def test_constraint_spec(self):
        c = ConstraintSpec(expression="max_num_batched_tokens >= max_num_seqs")
        assert "max_num_batched_tokens" in c.expression


class TestConfigModels:
    def test_study_settings_defaults(self):
        s = StudySettings()
        assert s.name == "vllm-tuning-study"
        assert "sqlite" in s.storage

    def test_optimization_config_defaults(self):
        opt = OptimizationConfig()
        assert opt.n_trials == 50
        assert opt.sampler == SamplerType.TPE
        assert len(opt.objectives) == 1

    def test_hardware_config_defaults(self):
        hw = HardwareConfig()
        assert hw.accelerator == AcceleratorType.GPU
        assert hw.device_ids == [0]

    def test_execution_config_defaults(self):
        ex = ExecutionConfig()
        assert ex.backend == BackendType.LOCAL
        assert ex.ray_address == "auto"

    def test_study_config_minimal(self):
        config = StudyConfig(model="test-model")
        assert config.model == "test-model"
        assert config.study.name == "vllm-tuning-study"
        assert config.execution.backend == BackendType.LOCAL
        assert config.baseline.enabled is True

    def test_study_config_full(self):
        config = StudyConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            optimization=OptimizationConfig(n_trials=100, sampler=SamplerType.NSGA2),
            execution=ExecutionConfig(backend=BackendType.RAY, ray_address="ray://head:10001"),
            parameters=[
                ParameterSpec(name="gpu_memory_utilization", min=0.85, max=0.95),
            ],
        )
        assert config.optimization.n_trials == 100
        assert config.execution.backend == BackendType.RAY
        assert len(config.parameters) == 1


class TestTrialModels:
    def test_trial_config(self):
        tc = TrialConfig(trial_number=5, parameters={"gpu_memory_utilization": 0.9})
        assert tc.trial_number == 5
        assert tc.parameters["gpu_memory_utilization"] == 0.9

    def test_trial_result_defaults(self):
        tr = TrialResult(trial_number=1)
        assert tr.status == TrialStatus.PENDING
        assert tr.benchmark is None
        assert tr.error_message is None

    def test_trial_result_completed(self, sample_trial_result):
        assert sample_trial_result.status == TrialStatus.COMPLETED
        assert sample_trial_result.benchmark is not None
        assert sample_trial_result.benchmark.throughput_req_per_sec == 10.5

    def test_vllm_telemetry(self):
        t = VLLMTelemetry(oom_detected=True, num_preemptions=3)
        assert t.oom_detected is True
        assert t.num_preemptions == 3
        assert t.kv_cache_usage is None

    def test_cost_report(self):
        report = CostReport(instances_needed=5, hourly_cost=9.0, monthly_cost=6570.0, perf_per_dollar=1.2)
        assert report.instances_needed == 5
        assert report.monthly_cost == 6570.0
