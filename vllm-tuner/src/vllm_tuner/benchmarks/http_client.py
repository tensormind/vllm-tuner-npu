from __future__ import annotations

import asyncio
import json
import random
import statistics
import string
import time
from typing import Any

from vllm_tuner.benchmarks.base import BenchmarkProvider
from vllm_tuner.core.models import BenchmarkConfig, BenchmarkResult
from vllm_tuner.helper.logging import get_logger


logger = get_logger()

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


class HTTPBenchmarkProvider(BenchmarkProvider):
    """Async HTTP benchmark provider using httpx.

    Sends concurrent OpenAI-compatible chat/completion requests and measures
    throughput and latency percentiles.
    Requires: pip install 'llm-vllm-tuner[http]'
    """

    def run(self, server_url: str, config: BenchmarkConfig) -> BenchmarkResult:
        if not _HTTPX_AVAILABLE:
            self.last_error = "httpx is not installed — install with: pip install 'llm-vllm-tuner[http]'"
            logger.warning(self.last_error)
            return BenchmarkResult()

        self._log(
            "HTTP benchmark: {} concurrent against {} (max_seconds={})",
            config.concurrent_requests,
            server_url,
            config.max_seconds,
        )
        return asyncio.run(self._run_async(server_url, config))

    def _log(self, msg: str, *args: Any) -> None:
        """Log to both loguru and live dashboard callback."""
        formatted = msg.format(*args) if args else msg
        logger.info(formatted)
        if self.log_callback is not None:
            self.log_callback(formatted)

    def _discover_model(self, server_url: str) -> str | None:
        """Query /v1/models to get the served model name."""
        try:
            resp = httpx.get(f"{server_url}/v1/models", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    model_id = models[0].get("id", "")
                    self._log("Discovered model: {}", model_id)
                    return model_id
        except Exception as exc:
            self._log("Warning: could not discover model name: {}", exc)
        return None

    def _health_check(self, server_url: str) -> bool:
        """Verify server is reachable via /health before benchmarking."""
        try:
            resp = httpx.get(f"{server_url}/health", timeout=10)
            return resp.status_code == 200
        except Exception as exc:
            self._log("Health check failed: {}", exc)
            return False

    async def _run_async(self, server_url: str, config: BenchmarkConfig) -> BenchmarkResult:
        url = f"{server_url}/v1/completions"
        latencies: list[float] = []
        ttfts: list[float] = []
        total = 0
        successful = 0
        failed = 0
        total_output_tokens = 0
        token_counts: list[int] = []

        # Verify server is reachable before starting benchmark
        if not self._health_check(server_url):
            self.last_error = f"Server unreachable at {server_url}/health — server may have crashed"
            self._log("ERROR: {}", self.last_error)
            return BenchmarkResult()

        # Discover served model name (required by vLLM)
        model_name = self._discover_model(server_url)
        if not model_name:
            self.last_error = f"Could not discover model from {server_url}/v1/models"
            self._log("ERROR: {}", self.last_error)
            return BenchmarkResult()

        prompt_text = self._generate_prompt(config.prompt_tokens)
        start_time = time.monotonic()

        async with httpx.AsyncClient(timeout=httpx.Timeout(config.max_seconds)) as client:
            # Send requests in waves until max_seconds is reached
            active: set[asyncio.Task] = set()
            request_id = 0

            self._log("Sending initial wave of {} requests...", config.concurrent_requests)

            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= config.max_seconds:
                    self._log("Time limit reached ({:.0f}s), stopping", elapsed)
                    break

                # Fill up to concurrent_requests active tasks
                while len(active) < config.concurrent_requests:
                    if time.monotonic() - start_time >= config.max_seconds:
                        break
                    request_id += 1
                    task = asyncio.create_task(self._send_request(client, url, prompt_text, model_name, config))
                    active.add(task)

                if not active:
                    break

                # Wait for at least one task to complete
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    total += 1
                    try:
                        res = task.result()
                    except Exception:
                        failed += 1
                        continue
                    if res is None:
                        failed += 1
                        continue
                    successful += 1
                    latencies.append(res["latency"])
                    if res["ttft"] is not None:
                        ttfts.append(res["ttft"])
                    req_tokens = res.get("output_tokens", 0)
                    total_output_tokens += req_tokens
                    token_counts.append(req_tokens)
                    self._log(
                        "Request completed: {:.2f}s latency, {} tokens (ok={}/fail={}/total={})",
                        res["latency"],
                        req_tokens,
                        successful,
                        failed,
                        total,
                    )

            # Cancel remaining tasks on timeout
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
                self._log("Cancelled {} in-flight requests", len(active))

        duration = time.monotonic() - start_time
        self._log("Benchmark finished: {} successful, {} failed in {:.1f}s", successful, failed, duration)

        if token_counts:
            avg_tok = sum(token_counts) / len(token_counts)
            min_tok = min(token_counts)
            max_tok = max(token_counts)
            self._log(
                "Token distribution: avg={:.0f}, min={}, max={} (expected ~{})",
                avg_tok,
                min_tok,
                max_tok,
                config.output_tokens,
            )

        if not latencies:
            self.last_error = f"No successful requests ({failed} failed out of {total})"
            self._log("WARNING: {}", self.last_error)
            return BenchmarkResult(
                total_requests=total,
                failed_requests=failed,
                expected_output_tokens=config.output_tokens,
                expected_prompt_tokens=config.prompt_tokens,
            )

        latencies.sort()
        p50_idx = int(len(latencies) * 0.50)
        p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)

        throughput = successful / duration if duration > 0 else 0.0
        tokens_per_sec = total_output_tokens / duration if duration > 0 else 0.0

        avg_tok = sum(token_counts) / len(token_counts) if token_counts else 0.0
        min_tok = min(token_counts) if token_counts else 0
        max_tok = max(token_counts) if token_counts else 0

        return BenchmarkResult(
            throughput_req_per_sec=throughput,
            output_tokens_per_sec=tokens_per_sec,
            p50_latency_ms=latencies[p50_idx] * 1000,
            p95_latency_ms=latencies[p95_idx] * 1000,
            p99_latency_ms=latencies[p99_idx] * 1000,
            ttft_ms=statistics.median(ttfts) * 1000 if ttfts else 0.0,
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            duration_seconds=duration,
            avg_output_tokens_per_request=avg_tok,
            min_output_tokens_per_request=min_tok,
            max_output_tokens_per_request=max_tok,
            expected_output_tokens=config.output_tokens,
            expected_prompt_tokens=config.prompt_tokens,
        )

    @staticmethod
    async def _send_request(
        client: "httpx.AsyncClient",
        url: str,
        prompt: str,
        model: str,
        config: BenchmarkConfig,
    ) -> dict | None:
        """Send a single streaming request and measure latency + TTFT."""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": config.output_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        start = time.monotonic()
        ttft = None
        output_tokens = 0
        finish_reason = None
        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    await response.aread()
                    logger.debug("HTTP {}: {}", response.status_code, response.text[:200])
                    return None
                async for chunk in response.aiter_lines():
                    if not chunk.strip():
                        continue
                    if chunk.startswith("data: "):
                        data_str = chunk[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        if ttft is None:
                            ttft = time.monotonic() - start
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                output_tokens += 1
                                fr = choices[0].get("finish_reason")
                                if fr is not None:
                                    finish_reason = fr
                            usage = data.get("usage")
                            if usage and "completion_tokens" in usage:
                                output_tokens = usage["completion_tokens"]
                        except json.JSONDecodeError:
                            pass
            latency = time.monotonic() - start

            min_expected = max(1, int(config.output_tokens * 0.1))
            if output_tokens < min_expected:
                logger.debug(
                    "Discarding short response: {} tokens (expected >= {}, finish_reason={})",
                    output_tokens,
                    min_expected,
                    finish_reason,
                )
                return None

            return {
                "latency": latency,
                "ttft": ttft,
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
            }
        except httpx.ConnectError as exc:
            logger.warning("Server connection refused (server may have crashed): {}", exc)
            return None
        except Exception as exc:
            logger.debug("Request failed: {}", exc)
            return None

    @staticmethod
    def _generate_prompt(target_tokens: int) -> str:
        """Generate a synthetic prompt of approximately target_tokens length."""
        # Rough approximation: 1 token ≈ 4 characters
        chars_needed = target_tokens * 4
        words = []
        total = 0
        while total < chars_needed:
            word_len = random.randint(3, 8)
            word = "".join(random.choices(string.ascii_lowercase, k=word_len))
            words.append(word)
            total += word_len + 1
        return " ".join(words)

    def supports_synthetic_workloads(self) -> bool:
        return True

    def supports_real_datasets(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "http"
