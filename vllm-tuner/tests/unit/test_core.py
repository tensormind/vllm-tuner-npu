from vllm_tuner.core.models import (
    ConstraintSpec,
    Direction,
    ObjectiveSpec,
    TrialResult,
    TrialStatus,
)
from vllm_tuner.core.optimizer import Optimizer
from vllm_tuner.core.trial import TrialRunner


class TestTrialRunner:
    def test_run_trial(self, sample_trial_config):
        runner = TrialRunner()
        result = runner.run_trial(sample_trial_config)
        assert result.trial_number == sample_trial_config.trial_number
        assert result.status == TrialStatus.COMPLETED
        assert result.duration_seconds > 0
        assert result.config is not None


class TestOptimizer:
    def test_compute_objective_values(self, sample_trial_result):
        objectives = [ObjectiveSpec(metric="throughput_req_per_sec", direction=Direction.MAXIMIZE)]
        optimizer = Optimizer(objectives=objectives)
        values = optimizer.compute_objective_values(sample_trial_result)
        assert len(values) == 1
        assert values[0] == 10.5

    def test_multi_objective(self, sample_trial_result):
        objectives = [
            ObjectiveSpec(metric="output_tokens_per_sec", direction=Direction.MAXIMIZE),
            ObjectiveSpec(metric="p95_latency_ms", direction=Direction.MINIMIZE),
        ]
        optimizer = Optimizer(objectives=objectives)
        values = optimizer.compute_objective_values(sample_trial_result)
        assert len(values) == 2
        assert values[0] == 2100.0
        assert values[1] == 145.0

    def test_unknown_metric(self, sample_trial_result):
        objectives = [ObjectiveSpec(metric="nonexistent_metric")]
        optimizer = Optimizer(objectives=objectives)
        values = optimizer.compute_objective_values(sample_trial_result)
        assert values[0] == 0.0

    def test_no_benchmark_results(self):
        objectives = [ObjectiveSpec(metric="throughput_req_per_sec")]
        optimizer = Optimizer(objectives=objectives)
        result = TrialResult(trial_number=1)
        values = optimizer.compute_objective_values(result)
        assert values == [0.0]

    def test_evaluate_constraints(self):
        constraints = [ConstraintSpec(expression="max_num_batched_tokens >= max_num_seqs")]
        optimizer = Optimizer(objectives=[], constraints=constraints)
        values = optimizer.evaluate_constraints({"max_num_batched_tokens": 8192, "max_num_seqs": 256})
        assert len(values) == 1
        assert values[0] >= 0  # satisfied
