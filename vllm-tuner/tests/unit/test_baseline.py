from vllm_tuner.baseline.runner import BaselineRunner
from vllm_tuner.core.models import BenchmarkResult


class TestBaselineRunner:
    def test_run_baseline_no_provider(self, sample_study_config):
        """Without a benchmark provider, returns empty result."""
        runner = BaselineRunner()
        result = runner.run_baseline(sample_study_config)
        assert isinstance(result, BenchmarkResult)
        assert result.throughput_req_per_sec == 0.0

    def test_run_baseline_with_mock_provider(self, sample_study_config):
        """With a mocked provider + launcher, runs the full pipeline."""
        from unittest.mock import MagicMock, patch

        from vllm_tuner.core.models import TrialResult, TrialStatus

        mock_provider = MagicMock()
        runner = BaselineRunner(benchmark_provider=mock_provider)

        with (
            patch("vllm_tuner.baseline.runner.VLLMLauncher"),
            patch("vllm_tuner.baseline.runner.TrialRunner") as MockRunner,
        ):
            MockRunner.return_value.run_trial.return_value = TrialResult(
                trial_number=0,
                status=TrialStatus.COMPLETED,
                benchmark=BenchmarkResult(throughput_req_per_sec=42.0, p95_latency_ms=100.0),
            )
            result = runner.run_baseline(sample_study_config)

        assert result.throughput_req_per_sec == 42.0
        assert result.p95_latency_ms == 100.0
