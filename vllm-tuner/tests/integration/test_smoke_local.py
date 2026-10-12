"""Integration smoke test for Mac M1 (no CUDA, no vLLM binary).

Spins up a tiny mock HTTP server that mimics vLLM's /health and /v1/completions
endpoints, then exercises the full tuner pipeline:

  StudyController → TrialRunner → HTTPBenchmarkProvider → SQLiteStorage → HTMLReport

Run:
    python -m pytest tests/integration/test_smoke_local.py -v -s
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from vllm_tuner.benchmarks.http_client import HTTPBenchmarkProvider
from vllm_tuner.core.models import (
    BenchmarkConfig,
    BenchmarkProviderType,
    Direction,
    ObjectiveSpec,
    OptimizationConfig,
    ParameterSpec,
    StudyConfig,
    StudySettings,
    TrialConfig,
    TrialStatus,
)
from vllm_tuner.core.study_controller import StudyController
from vllm_tuner.core.trial import TrialRunner
from vllm_tuner.hardware.null import NullMonitor
from vllm_tuner.reporting.html import HTMLReportGenerator
from vllm_tuner.storage.sqlite import SQLiteStorage
from vllm_tuner.vllm.launcher import VLLMLauncher


# ──────────────────────────────────────────────
# Mock vLLM HTTP server
# ──────────────────────────────────────────────


class _MockVLLMHandler(BaseHTTPRequestHandler):
    """Mimics vLLM's health and completions endpoints."""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path == "/v1/models":
            body = json.dumps(
                {
                    "object": "list",
                    "data": [{"id": "mock-model", "object": "model"}],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/completions":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)
            body_data = json.loads(body_bytes) if body_bytes else {}
            # Simulate ~5ms inference latency
            time.sleep(0.005)

            if body_data.get("stream"):
                # Streaming SSE response
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                n_tokens = body_data.get("max_tokens", 20)
                try:
                    for i in range(n_tokens):
                        chunk = json.dumps(
                            {
                                "id": "cmpl-mock",
                                "object": "text_completion",
                                "choices": [{"text": "hello ", "index": 0, "finish_reason": None}],
                            }
                        )
                        self.wfile.write(f"data: {chunk}\n\n".encode())
                        self.wfile.flush()
                    # Final chunk
                    final = json.dumps(
                        {
                            "id": "cmpl-mock",
                            "object": "text_completion",
                            "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                        }
                    )
                    self.wfile.write(f"data: {final}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except BrokenPipeError:
                    pass
            else:
                # Non-streaming response
                body = json.dumps(
                    {
                        "id": "cmpl-mock",
                        "object": "text_completion",
                        "choices": [{"text": "Hello world! " * 10, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs during test


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock vLLM server on a random port."""
    server = HTTPServer(("127.0.0.1", 0), _MockVLLMHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", port
    server.shutdown()


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


class TestHTTPBenchmarkSmoke:
    """Directly test the HTTPBenchmarkProvider against the mock server."""

    def test_benchmark_returns_metrics(self, mock_server):
        url, _ = mock_server
        provider = HTTPBenchmarkProvider()

        config = BenchmarkConfig(
            provider=BenchmarkProviderType.HTTP,
            prompt_tokens=50,
            output_tokens=20,
            max_seconds=5,
            concurrent_requests=4,
        )
        result = provider.run(url, config)

        assert result.throughput_req_per_sec > 0, "Expected non-zero throughput"
        assert result.successful_requests > 0, "Expected successful requests"
        assert result.p50_latency_ms > 0, "Expected non-zero P50 latency"
        assert result.p95_latency_ms > 0, "Expected non-zero P95 latency"
        assert result.ttft_ms > 0, "Expected non-zero TTFT with streaming"
        print(f"\n  Throughput: {result.throughput_req_per_sec:.1f} req/s")
        print(f"  P50 latency: {result.p50_latency_ms:.1f} ms")
        print(f"  P95 latency: {result.p95_latency_ms:.1f} ms")
        print(f"  TTFT: {result.ttft_ms:.1f} ms")
        print(f"  Successful: {result.successful_requests}/{result.total_requests}")


class TestTrialRunnerSmoke:
    """Run a trial with a mock vLLM server (patching only the launcher)."""

    def test_full_trial_lifecycle(self, mock_server):
        url, port = mock_server

        launcher = VLLMLauncher(model="mock-model", host="127.0.0.1", port=port)
        provider = HTTPBenchmarkProvider()
        monitor = NullMonitor()
        benchmark_config = BenchmarkConfig(
            prompt_tokens=50,
            output_tokens=20,
            max_seconds=3,
            concurrent_requests=2,
        )

        runner = TrialRunner(
            launcher=launcher,
            monitor=monitor,
            benchmark_provider=provider,
            benchmark_config=benchmark_config,
            startup_timeout=5,
        )

        trial_config = TrialConfig(
            trial_number=1,
            parameters={"gpu_memory_utilization": 0.9, "max_num_seqs": 128},
        )

        # Patch launcher.start() to no-op (server is already running)
        # and launcher.stop() to no-op (we manage the mock server ourselves)
        with (
            patch.object(launcher, "start"),
            patch.object(launcher, "stop"),
            patch.object(launcher, "wait_until_ready", return_value=True),
            patch.object(launcher, "read_logs", return_value=[]),
        ):
            result = runner.run_trial(trial_config)

        assert result.status == TrialStatus.COMPLETED
        assert result.benchmark is not None
        assert result.benchmark.throughput_req_per_sec > 0
        assert result.duration_seconds > 0
        print(f"\n  Trial #{result.trial_number}: {result.status.value}")
        print(f"  Throughput: {result.benchmark.throughput_req_per_sec:.1f} req/s")
        print(f"  Duration: {result.duration_seconds:.2f}s")


class TestStudyControllerSmoke:
    """Run a 3-trial Optuna optimization loop with mock server."""

    def test_optimize_3_trials(self, mock_server, tmp_path):
        url, port = mock_server

        config = StudyConfig(
            model="mock-model",
            study=StudySettings(name="smoke-test"),
            optimization=OptimizationConfig(
                objectives=[ObjectiveSpec(metric="throughput_req_per_sec", direction=Direction.MAXIMIZE)],
                sampler="tpe",
                n_trials=3,
                n_startup_trials=2,
            ),
            parameters=[
                ParameterSpec(name="gpu_memory_utilization", min=0.5, max=0.95),
                ParameterSpec(name="max_num_seqs", min=32, max=512, step=32),
            ],
            benchmark=BenchmarkConfig(
                provider=BenchmarkProviderType.HTTP,
                prompt_tokens=50,
                output_tokens=20,
                max_seconds=3,
                concurrent_requests=2,
            ),
        )

        controller = StudyController(config)
        controller.create_study()

        launcher = VLLMLauncher(model="mock-model", host="127.0.0.1", port=port)
        provider = HTTPBenchmarkProvider()
        monitor = NullMonitor()

        runner = TrialRunner(
            launcher=launcher,
            monitor=monitor,
            benchmark_provider=provider,
            benchmark_config=config.benchmark,
            startup_timeout=5,
        )

        # Patch launcher lifecycle methods
        with (
            patch.object(launcher, "start"),
            patch.object(launcher, "stop"),
            patch.object(launcher, "wait_until_ready", return_value=True),
            patch.object(launcher, "read_logs", return_value=[]),
        ):
            controller.optimize(trial_runner=runner)

        assert len(controller.results) == 3
        for r in controller.results:
            assert r.status == TrialStatus.COMPLETED
            assert r.benchmark.throughput_req_per_sec > 0

        summary = controller.get_study_summary()
        assert summary["n_trials_completed"] == 3
        print(f"\n  Study summary: {summary}")

        # ── Storage round-trip ──
        db_path = str(tmp_path / "smoke.db")
        storage = SQLiteStorage(db_path)
        for r in controller.results:
            storage.save_trial("smoke-test", r)

        loaded = storage.load_trials("smoke-test")
        assert len(loaded) == 3
        print(f"  Storage: saved & loaded {len(loaded)} trials")

        # ── HTML report ──
        report_dir = tmp_path / "reports"
        report_dir.mkdir()
        generator = HTMLReportGenerator()
        report_path = generator.generate(controller.results, report_dir, "smoke-test")
        assert report_path.exists()
        html_content = report_path.read_text()
        assert "smoke-test" in html_content
        assert "throughput" in html_content.lower()
        print(f"  HTML report: {report_path} ({len(html_content)} bytes)")
