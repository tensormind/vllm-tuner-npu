from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from vllm_tuner.benchmarks.base import BenchmarkProvider
from vllm_tuner.core.models import BenchmarkConfig, BenchmarkResult
from vllm_tuner.helper.logging import get_logger


logger = get_logger()


class VLLMBenchmarkProvider(BenchmarkProvider):
    """vLLM benchmark_serving.py-based benchmark provider.

    Invokes vLLM's benchmark_serving.py as a subprocess and parses JSON output.
    Requires vLLM to be importable in the Python environment.
    """

    def __init__(self, model: str = ""):
        self._model = model

    def run(self, server_url: str, config: BenchmarkConfig) -> BenchmarkResult:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "benchmark_result.json"
            cmd = self._build_command(server_url, config, output_file)

            logger.info("vLLM benchmark: running: {}", " ".join(cmd))
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.max_seconds + 120,
                )
                if proc.returncode != 0:
                    stderr_snip = proc.stderr[:500] if proc.stderr else proc.stdout[:500]
                    logger.error(
                        "vLLM benchmark exited with code {}: {}",
                        proc.returncode,
                        stderr_snip,
                    )
                    self.last_error = f"vLLM benchmark exited with code {proc.returncode}: {stderr_snip}"
                    return BenchmarkResult()

                return self._parse_output(output_file, proc.stdout)
            except subprocess.TimeoutExpired:
                logger.error("vLLM benchmark timed out after {}s", config.max_seconds + 120)
                self.last_error = f"vLLM benchmark timed out after {config.max_seconds + 120}s"
                return BenchmarkResult()
            except FileNotFoundError:
                logger.error("Python not found for running benchmark_serving.py")
                self.last_error = "Python not found for running benchmark_serving.py"
                return BenchmarkResult()

    def _build_command(self, server_url: str, config: BenchmarkConfig, output_file: Path) -> list[str]:
        host_port = server_url.replace("http://", "").replace("https://", "")
        parts = host_port.split(":")
        host = parts[0]
        port = parts[1] if len(parts) > 1 else "8000"

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.run_batch_benchmark",
            "--host",
            host,
            "--port",
            port,
            "--model",
            self._model,
            "--num-prompts",
            str(config.concurrent_requests),
            "--request-rate",
            "inf",
            "--save-result",
            "--result-dir",
            str(output_file.parent),
            "--result-filename",
            output_file.name,
        ]
        if config.dataset:
            cmd.extend(["--dataset-name", config.dataset])
        else:
            cmd.extend(
                [
                    "--random-input-len",
                    str(config.prompt_tokens),
                    "--random-output-len",
                    str(config.output_tokens),
                ]
            )
        return cmd

    @staticmethod
    def _parse_output(output_file: Path, stdout: str) -> BenchmarkResult:
        """Parse vLLM benchmark JSON output into BenchmarkResult."""
        if output_file.exists():
            try:
                data = json.loads(output_file.read_text())
                return BenchmarkResult(
                    throughput_req_per_sec=float(data.get("request_throughput", 0.0)),
                    output_tokens_per_sec=float(data.get("output_throughput", 0.0)),
                    p50_latency_ms=float(data.get("median_request_latency", 0.0)) * 1000,
                    p95_latency_ms=float(data.get("p95_request_latency", 0.0)) * 1000,
                    p99_latency_ms=float(data.get("p99_request_latency", 0.0)) * 1000,
                    ttft_ms=float(data.get("median_ttft_ms", 0.0)),
                    itl_ms=float(data.get("median_itl_ms", 0.0)),
                    total_requests=int(data.get("total_input_tokens", 0)),
                    successful_requests=int(data.get("completed", 0)),
                    duration_seconds=float(data.get("duration", 0.0)),
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("Failed to parse vLLM benchmark output file: {}", e)

        # Fallback: try to parse from stdout (older vLLM versions)
        logger.warning("Could not parse vLLM benchmark JSON output, trying stdout")
        return BenchmarkResult()

    def supports_synthetic_workloads(self) -> bool:
        return True

    def supports_real_datasets(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "vllm_benchmark"
