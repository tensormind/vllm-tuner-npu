from __future__ import annotations

import json
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from vllm_tuner.benchmarks.base import BenchmarkProvider
from vllm_tuner.core.models import BenchmarkConfig, BenchmarkResult
from vllm_tuner.helper.logging import get_logger


logger = get_logger()

_GUIDELLM_AVAILABLE = shutil.which("guidellm") is not None


class GuideLLMProvider(BenchmarkProvider):
    """GuideLLM-based benchmark provider (v0.5+).

    Invokes the ``guidellm benchmark run`` CLI as a subprocess and parses
    the JSON output.
    """

    def run(self, server_url: str, config: BenchmarkConfig) -> BenchmarkResult:
        if not _GUIDELLM_AVAILABLE:
            self.last_error = "GuideLLM is not installed — install with: pip install 'llm-vllm-tuner[guidellm]'"
            logger.warning(self.last_error)
            return BenchmarkResult()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "benchmark.json"
            cmd = self._build_command(server_url, config, output_file)

            logger.info("GuideLLM: running benchmark: {}", " ".join(cmd))
            if self.log_callback:
                self.log_callback(f"[BENCHMARK] guidellm benchmark run --target {server_url}")
            try:

                def _preexec():
                    """Disable core dumps to suppress macOS crash dialog."""
                    try:
                        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                    except (ValueError, OSError):
                        pass

                preexec = _preexec if sys.platform == "darwin" else None

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    preexec_fn=preexec,
                )

                # Stream output in a reader thread
                output_lines: list[str] = []

                def _read_output():
                    assert proc.stdout is not None
                    try:
                        for line in proc.stdout:
                            stripped = line.rstrip("\n")
                            output_lines.append(stripped)
                            if self.log_callback:
                                self.log_callback(stripped)
                    except (ValueError, OSError):
                        pass

                reader = threading.Thread(target=_read_output, daemon=True)
                reader.start()

                try:
                    proc.wait(timeout=config.max_seconds + 120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    logger.error("GuideLLM timed out after {}s", config.max_seconds + 120)
                    self.last_error = f"GuideLLM timed out after {config.max_seconds + 120}s"
                    return BenchmarkResult()

                reader.join(timeout=5)

                if proc.returncode != 0:
                    tail = "\n".join(output_lines[-10:])
                    # Negative return code = killed by signal (e.g. -11 = SIGSEGV)
                    if proc.returncode < 0:
                        sig_num = -proc.returncode
                        sig_name = (
                            signal.Signals(sig_num).name
                            if sig_num in signal.Signals._value2member_map_
                            else f"signal {sig_num}"
                        )
                        self.last_error = (
                            f"GuideLLM crashed with {sig_name}. "
                            f"This often happens due to library conflicts (torch/tokenizers). "
                            f"Try using 'provider: http' in your config instead."
                        )
                    else:
                        self.last_error = f"GuideLLM exited with code {proc.returncode}: {tail}"
                    logger.error("GuideLLM failed: {}", self.last_error)
                    return BenchmarkResult()

                return self._parse_output(output_file, config)
            except FileNotFoundError:
                logger.error("GuideLLM binary not found")
                self.last_error = "GuideLLM binary not found"
                return BenchmarkResult()

    @staticmethod
    def _build_command(server_url: str, config: BenchmarkConfig, output_file: Path) -> list[str]:
        cmd = [
            "guidellm",
            "benchmark",
            "run",
            "--target",
            server_url,
            "--output-dir",
            str(output_file.parent),
            "--outputs",
            output_file.name,
            "--max-seconds",
            str(config.max_seconds),
            "--rate",
            str(config.concurrent_requests),
            "--profile",
            "concurrent",
            "--disable-console",
        ]
        if config.dataset:
            cmd.extend(["--data", config.dataset])
        else:
            # Synthetic data config: prompt_tokens ± stdev, fixed output_tokens
            synthetic = json.dumps(
                {
                    "prompt_tokens": config.prompt_tokens,
                    "prompt_tokens_stdev": config.prompt_tokens_stdev,
                    "output_tokens": config.output_tokens,
                }
            )
            cmd.extend(["--data", synthetic])
        return cmd

    @staticmethod
    def _parse_output(output_file: Path, config: BenchmarkConfig) -> BenchmarkResult:
        """Parse GuideLLM v0.5 JSON output into BenchmarkResult."""
        if not output_file.exists():
            logger.error("GuideLLM output file not found: {}", output_file)
            return BenchmarkResult()

        try:
            data = json.loads(output_file.read_text())
        except json.JSONDecodeError as e:
            logger.error("Failed to parse GuideLLM output: {}", e)
            return BenchmarkResult()

        benchmarks = data.get("benchmarks", [])
        if not benchmarks:
            logger.warning("GuideLLM output contains no benchmark results")
            return BenchmarkResult()

        metrics = benchmarks[0].get("metrics", {})

        # request_totals: {successful, errored, incomplete, total}
        totals = metrics.get("request_totals", {})
        total = int(totals.get("total", 0))
        successful = int(totals.get("successful", 0))
        failed = int(totals.get("errored", 0))

        def _stat_mean(metric_name: str) -> float:
            """Get mean from successful distribution."""
            m = metrics.get(metric_name, {})
            s = m.get("successful", {}) if isinstance(m, dict) else {}
            return float(s.get("mean", 0.0))

        def _stat_pct(metric_name: str, pct: str) -> float:
            """Get percentile from successful distribution."""
            m = metrics.get(metric_name, {})
            s = m.get("successful", {}) if isinstance(m, dict) else {}
            p = s.get("percentiles", {}) if isinstance(s, dict) else {}
            return float(p.get(pct, 0.0))

        return BenchmarkResult(
            throughput_req_per_sec=_stat_mean("requests_per_second"),
            output_tokens_per_sec=_stat_mean("output_tokens_per_second"),
            # request_latency is in seconds — convert to ms
            p50_latency_ms=_stat_pct("request_latency", "p50") * 1000,
            p95_latency_ms=_stat_pct("request_latency", "p95") * 1000,
            p99_latency_ms=_stat_pct("request_latency", "p99") * 1000,
            # time_to_first_token_ms is already in ms
            ttft_ms=_stat_pct("time_to_first_token_ms", "p50"),
            itl_ms=_stat_pct("inter_token_latency_ms", "p50"),
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
        )

    def supports_synthetic_workloads(self) -> bool:
        return True

    def supports_real_datasets(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "guidellm"
