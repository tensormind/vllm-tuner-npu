import json

from vllm_tuner.core.models import BenchmarkResult, CostConfig
from vllm_tuner.reporting.cost_analysis import CostAnalyzer
from vllm_tuner.reporting.dashboard import TerminalDashboard
from vllm_tuner.reporting.export import ConfigExporter
from vllm_tuner.reporting.html import HTMLReportGenerator


class TestHTMLReportGenerator:
    def test_generate_creates_file(self, tmp_path, sample_trial_result):
        gen = HTMLReportGenerator()
        path = gen.generate([sample_trial_result], tmp_path, "test-study")
        assert path.exists()
        content = path.read_text()
        assert "test-study" in content
        assert "plotly" in content.lower()

    def test_generate_empty_results(self, tmp_path):
        gen = HTMLReportGenerator()
        path = gen.generate([], tmp_path)
        assert path.exists()

    def test_generate_with_baseline(self, tmp_path, sample_trial_result, sample_benchmark_result):
        gen = HTMLReportGenerator()
        baseline = BenchmarkResult(
            throughput_req_per_sec=5.0,
            output_tokens_per_sec=1000.0,
            p95_latency_ms=200.0,
        )
        path = gen.generate([sample_trial_result], tmp_path, "baseline-study", baseline=baseline)
        assert path.exists()
        content = path.read_text()
        assert "Baseline vs Best Trial" in content
        assert "5.00" in content  # baseline throughput


class TestConfigExporter:
    def test_export_yaml(self, tmp_path, sample_trial_result):
        exporter = ConfigExporter()
        path = exporter.export_yaml(sample_trial_result, tmp_path / "config.yaml")
        assert path.exists()
        content = path.read_text()
        assert "vllm_parameters" in content

    def test_export_json(self, tmp_path, sample_trial_result):
        exporter = ConfigExporter()
        path = exporter.export_json(sample_trial_result, tmp_path / "config.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert "vllm_parameters" in data
        assert "trial_number" in data

    def test_export_helm_values(self, tmp_path, sample_trial_result):
        exporter = ConfigExporter()
        path = exporter.export_helm_values(sample_trial_result, tmp_path / "helm_values.yaml")
        assert path.exists()
        content = path.read_text()
        assert "servingRuntime" in content
        assert "extraArgs" in content


class TestCostAnalyzer:
    def test_analyze_with_valid_config(self, sample_benchmark_result):
        analyzer = CostAnalyzer()
        config = CostConfig(
            target_throughput=100,
            cloud_provider="gcp",
            instance_type="g2-standard-4",
            pricing_mode="spot",
            price_per_hour=0.20,
        )
        report = analyzer.analyze(sample_benchmark_result, config)
        assert report.instances_needed > 0
        assert report.hourly_cost > 0
        assert report.monthly_cost > 0
        assert report.perf_per_dollar > 0

    def test_analyze_missing_config(self, sample_benchmark_result):
        analyzer = CostAnalyzer()
        report = analyzer.analyze(sample_benchmark_result, CostConfig())
        assert report.instances_needed == 0

    def test_analyze_unknown_instance(self, sample_benchmark_result):
        analyzer = CostAnalyzer()
        config = CostConfig(
            target_throughput=100,
            cloud_provider="gcp",
            instance_type="nonexistent-instance",
        )
        report = analyzer.analyze(sample_benchmark_result, config)
        assert report.instances_needed == 0


class TestTerminalDashboard:
    def test_show_trial_result(self, sample_trial_result, capsys):
        dashboard = TerminalDashboard()
        dashboard.show_trial_result(sample_trial_result)
        # Rich output goes to console, not easily captured; just verify no error

    def test_show_study_summary(self):
        dashboard = TerminalDashboard()
        dashboard.show_study_summary({"status": "completed", "trials": "50"})

    def test_show_progress(self):
        dashboard = TerminalDashboard()
        dashboard.show_progress(10, 50, best_value=12.5)

    def test_show_comparison(self, sample_benchmark_result):
        dashboard = TerminalDashboard()
        baseline = BenchmarkResult(
            throughput_req_per_sec=7.5,
            p95_latency_ms=200.0,
            p99_latency_ms=300.0,
            ttft_ms=60.0,
        )
        dashboard.show_comparison(baseline, sample_benchmark_result)
