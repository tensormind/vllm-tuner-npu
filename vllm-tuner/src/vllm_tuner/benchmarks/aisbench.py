from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from vllm_tuner.benchmarks.base import BenchmarkProvider
from vllm_tuner.core.models import BenchmarkConfig, BenchmarkResult
from vllm_tuner.helper.logging import get_logger


logger = get_logger()

_AIS_BENCH_AVAILABLE = shutil.which("ais_bench") is not None


# ─────────────────────────────────────────────────────────────────────────────
# API config template (ported from default_api.py).
# Placeholders are substituted by _render_api_template().
# ─────────────────────────────────────────────────────────────────────────────

_API_TEMPLATE = """\
from ais_bench.benchmark.models import {test_type_ref}

models = [
    dict(
        attr="service",
        type={test_type_ref},
        abbr='{test_abbr}',
        path="{model_path}",
        model="{model_name}",
        api_key="{api_key}",
        request_rate={request_rate},
        retry=2,
        host_ip="{host_ip}",
        host_port={host_port},
        max_out_len={output_len},
        batch_size={concurrency},
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
        )
    )
]
"""


# ─────────────────────────────────────────────────────────────────────────────
# Metric parsing (ported from save_file.py — behaviour identical to reference).
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_METRICS = {
    "current_time": None,
    "input_len": 99999,
    "output_len": 99999,
    "total_req": 99999,
    "max_cc": 99999,
    "cc": 99999,
    "rr": 0,
    "TTFT avg": 99999,
    "TTFT P90": 99999,
    "TPOT avg": 99999,
    "TPOT SLO_P90": 99999,
    "E2E_time": 99999,
    "output_throughput": 99999,
    "single_output_throughput": 99999,
    "E2E_throughput": 9999,
    "single_E2E_throughput": 9999,
    "qps": 9999,
    "qpm": 9999,
    "input_token_throughput": 9999,
    "prefill_token_throughput": 9999,
    "E2EL avg": 9999,
    "E2EL P90": 9999,
}

_METRIC_RULES = [
    ("E2EL", r"(\d+\.\d+)", 5, 1.0, None),
    ("TTFT", r"(\d+\.\d+)", 5, 1.0, None),
    ("TPOT", r"(\d+\.\d+)", 5, 1.0, None),
    ("Benchmark Duration", r"(\d+\.\d+)", 0, 1 / 1000, None),
    ("Concurrency", r"(\d+\.\d+)", 0, 1.0, None),
    ("Output Token Throughput", r"(\d+\.\d+)", 0, 1.0, lambda v, n: {
        "output_throughput": v, "single_output_throughput": v / n}),
    ("Input Token Throughput", r"(\d+\.\d+)", 0, 1.0, None),
    ("Total Token Throughput", r"(\d+\.\d+)", 0, 1.0, lambda v, n: {
        "E2E_throughput": v, "single_E2E_throughput": v / n}),
    ("InputTokens", r"(\d+\.?\d*)", 0, 1.0, None),
    ("OutputTokens", r"(\d+\.?\d*)", 0, 1.0, None),
    ("Total Requests", r"(\d+\.?\d*)", 0, 1.0, None),
    ("Request Throughput", r"(\d+\.\d+)", 0, 1.0, lambda v, n: {
        "qps": v, "qpm": v * 60}),
    ("Prefill Token Throughput", r"(\d+\.\d+)", 0, 1.0, None),
]


def _extract_value(line: str, pattern: str, index: int, scale: float):
    matches = re.findall(pattern, line)
    if not matches or index >= len(matches):
        return None
    try:
        return float(matches[index]) * scale
    except (ValueError, TypeError):
        return None


def _get_field_name(keyword: str) -> list[str]:
    name_map = {
        "E2EL": ["E2EL P90", "E2EL avg"],
        "TTFT": ["TTFT P90", "TTFT avg"],
        "TPOT": ["TPOT SLO_P90", "TPOT avg"],
        "Benchmark Duration": ["E2E_time"],
        "Concurrency": ["cc"],
        "Max Concurrency": ["max_cc"],
        "Output Token Throughput": ["output_throughput"],
        "Input Token Throughput": ["input_token_throughput"],
        "Total Token Throughput": ["E2E_throughput"],
        "InputTokens": ["input_len"],
        "OutputTokens": ["output_len"],
        "Total Requests": ["total_req"],
        "Request Throughput": ["qps"],
        "Prefill Token Throughput": ["prefill_token_throughput"],
    }
    return name_map.get(keyword, [])


def _parse_metrics(output_lines: list[str], npu_num: int = 1) -> dict:
    """Parse captured ais_bench stdout/log lines into a metrics dict.

    Faithful port of save_file.get_data() (minus CSV/log-file side effects).
    """
    metrics = dict(_DEFAULT_METRICS)
    for line in output_lines:
        if "Max Concurrency" in line:
            matches = re.findall(r"[\w']+", line)
            if matches:
                metrics["max_cc"] = matches[-1]
            continue
        for keyword, pattern, index, scale, extra_fn in _METRIC_RULES:
            if keyword in line:
                raw_value = _extract_value(line, pattern, index, scale)
                if raw_value is not None:
                    if extra_fn:
                        metrics.update(extra_fn(raw_value, npu_num))
                    else:
                        if keyword in ("E2EL", "TTFT", "TPOT"):
                            p90_key = f"{keyword} {'SLO_' if keyword == 'TPOT' else ''}P90"
                            avg_key = f"{keyword} avg"
                            if index == 5:
                                metrics[p90_key] = raw_value
                            avg_value = _extract_value(line, pattern, 0, scale)
                            if avg_value is not None:
                                metrics[avg_key] = avg_value
                        else:
                            field_names = _get_field_name(keyword)
                            if field_names:
                                metrics[field_names[0]] = raw_value
                break
    metrics["current_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Prefix cache hit-rate parsing (ported from cal_prefix_hit_rate.py).
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_metrics_text(host: str, port: int) -> str:
    import requests

    url = f"http://{host}:{port}/metrics"
    resp = requests.get(url, proxies={"http": None, "https": None}, timeout=30)
    resp.raise_for_status()
    return resp.text


def _get_prefix_cache_metrics(host: str, port: int, metric_type: str) -> tuple[dict, dict]:
    """Return (normal_stats, external_stats); keyed by engine_id (int)."""
    try:
        text = _fetch_metrics_text(host, port)
    except Exception as e:
        logger.warning("AISBench: failed to fetch /metrics for prefix cache: {}", e)
        return {}, {}

    lines = [
        l for l in text.split("\n")
        if "model_name" in l and f"prefix_cache_{metric_type}_total" in l
    ]
    normal_stats: dict[int, float] = {}
    external_stats: dict[int, float] = {}
    for line in lines:
        engine_match = re.search(r'engine="(\d+)"', line)
        if not engine_match:
            continue
        engine = int(engine_match.group(1))
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        if f"external_prefix_cache_{metric_type}_total" in line:
            external_stats[engine] = value
        elif f"vllm:prefix_cache_{metric_type}_total" in line:
            normal_stats[engine] = value
    return normal_stats, external_stats


def _snapshot_prefix_cache(host: str, port: int) -> tuple[dict, dict]:
    """Aggregate queries/hits across engines: returns (queries_total, hits_total)."""
    q_normal, q_external = _get_prefix_cache_metrics(host, port, "queries")
    h_normal, h_external = _get_prefix_cache_metrics(host, port, "hits")
    queries = sum(q_normal.values()) + sum(q_external.values())
    hits = sum(h_normal.values()) + sum(h_external.values())
    return {"queries": queries, "hits": hits}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _symlink_force(target: str, link_name: str) -> None:
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == getattr(os, "EEXIST", 17):
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise


def _resolve_host_port(server_url: str, fallback_host: str, fallback_port: int) -> tuple[str, int]:
    """Extract host/port from a server_url like http://127.0.0.1:8000."""
    if server_url:
        parsed = urlparse(server_url)
        host = parsed.hostname or fallback_host
        port = parsed.port or fallback_port
        return host, port
    return fallback_host, fallback_port


# ─────────────────────────────────────────────────────────────────────────────
# Provider
# ─────────────────────────────────────────────────────────────────────────────

class AISBenchProvider(BenchmarkProvider):
    """ais-bench CLI benchmark provider (chat/completions, GSM8K prompts).

    Adapts the aisbench_auto_tools_prefix validation method: invokes the
    ``ais_bench`` CLI as a subprocess, streams output, and parses the metric
    tables into a BenchmarkResult. Optionally measures prefix cache hit rate
    via the vLLM ``/metrics`` endpoint.
    """

    def __init__(
        self,
        model_path: str = "",
        model_name: str = "",
        aisbench_work_path: str = "",
        tools_path: str = "",
        dataset_dir: str = "",
        host: str = "127.0.0.1",
        port: int = 8000,
        prefix_cache: bool = False,
        npu_num: int = 1,
        summarizer: str = "default_perf",
    ):
        self._model_path = model_path
        self._model_name = model_name or (Path(model_path).name if model_path else "")
        self._work_path = aisbench_work_path
        self._tools_path = tools_path
        self._dataset_dir = dataset_dir
        self._host = host
        self._port = port
        self._prefix_cache = prefix_cache
        self._npu_num = npu_num
        self._summarizer = summarizer

    def run(self, server_url: str, config: BenchmarkConfig) -> BenchmarkResult:
        if not _AIS_BENCH_AVAILABLE:
            self.last_error = (
                "ais_bench CLI is not installed — install with: "
                "pip install ais-bench-benchmark"
            )
            logger.warning(self.last_error)
            return BenchmarkResult()

        if not self._work_path:
            self.last_error = (
                "AISBenchProvider requires 'aisbench.work_path' "
                "(the ais_bench install root containing ais_bench/benchmark/)"
            )
            logger.error(self.last_error)
            return BenchmarkResult()

        host, port = _resolve_host_port(server_url, self._host, self._port)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_dir = tmp / "aisbench_out"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Step 1: locate or generate dataset
            dataset_path = self._prepare_dataset(config, tmp)
            if dataset_path is None:
                # _prepare_dataset already set last_error
                return BenchmarkResult()

            # Step 2: render + symlink API config
            try:
                self._render_and_link_api(config, host, port)
            except Exception as e:
                self.last_error = f"AISBench: failed to render API config: {e}"
                logger.error(self.last_error)
                return BenchmarkResult()

            # Step 3: symlink dataset into ais_bench gsm8k dir
            try:
                self._link_dataset(dataset_path)
            except Exception as e:
                self.last_error = f"AISBench: failed to link dataset: {e}"
                logger.error(self.last_error)
                return BenchmarkResult()

            # Step 4: optional prefix-cache snapshot (before)
            cache_before = None
            if self._prefix_cache:
                cache_before = _snapshot_prefix_cache(host, port)

            # Step 5: build command
            cmd = self._build_command(output_dir)

            # Step 6: run subprocess with streamed output
            timeout = config.max_seconds * 3 + 300
            logger.info("AISBench: running: {}", cmd)
            if self.log_callback:
                self.log_callback(f"[BENCHMARK] {cmd}")

            output_lines: list[str] = []

            def _read_output(proc: subprocess.Popen):
                assert proc.stdout is not None
                try:
                    for line in proc.stdout:
                        stripped = line.rstrip("\n")
                        output_lines.append(stripped)
                        if self.log_callback:
                            self.log_callback(stripped)
                except (ValueError, OSError):
                    pass

            try:
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(tmp),
                )
            except FileNotFoundError:
                self.last_error = "ais_bench binary not found"
                logger.error(self.last_error)
                return BenchmarkResult()

            reader = threading.Thread(target=_read_output, args=(proc,), daemon=True)
            reader.start()

            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                reader.join(timeout=5)
                self.last_error = f"ais_bench timed out after {timeout}s"
                logger.error(self.last_error)
                return BenchmarkResult()

            reader.join(timeout=5)

            if proc.returncode != 0:
                tail = "\n".join(output_lines[-10:])
                self.last_error = f"ais_bench exited with code {proc.returncode}: {tail}"
                logger.error(self.last_error)
                # still attempt to parse — ais_bench may have printed metrics before failing

            # Step 7: optional prefix-cache snapshot (after) + delta
            prefix_hit_rate = 0.0
            if self._prefix_cache and cache_before is not None:
                cache_after = _snapshot_prefix_cache(host, port)
                q_delta = cache_after["queries"] - cache_before["queries"]
                h_delta = cache_after["hits"] - cache_before["hits"]
                if q_delta > 0:
                    prefix_hit_rate = h_delta / q_delta
                logger.info(
                    "AISBench: prefix cache hit rate={:.2%} (hits={}, queries={})",
                    prefix_hit_rate, h_delta, q_delta,
                )

            # Step 8: parse metrics
            return self._build_result(output_lines, config, prefix_hit_rate)

    # ── dataset ────────────────────────────────────────────────────────────

    def _prepare_dataset(self, config: BenchmarkConfig, tmp_dir: Path) -> str | None:
        """Return path to a JSONL dataset, or None on failure."""
        if config.dataset:
            if not os.path.exists(config.dataset):
                self.last_error = f"AISBench: dataset not found: {config.dataset}"
                logger.error(self.last_error)
                return None
            logger.info("AISBench: using prebuilt dataset: {}", config.dataset)
            return config.dataset

        # Generate via the reference project's generate_dataset module.
        if not self._tools_path:
            self.last_error = (
                "AISBench: no 'benchmark.dataset' and no 'aisbench.tools_path' "
                "for dataset generation"
            )
            logger.error(self.last_error)
            return None

        dataset_dir = self._dataset_dir or str(tmp_dir / "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        try:
            import sys

            if self._tools_path not in sys.path:
                sys.path.insert(0, self._tools_path)
            cwd = os.getcwd()
            os.chdir(self._tools_path)  # generate_dataset uses CWD-relative GSM8K.jsonl
            try:
                from generate_dataset import create_multi_prefix_dataset  # type: ignore

                prefix_flag = 1 if config.dataset_type == "prefix_cache" else 0
                repeat_rate = 0
                _, dataset_jsonl = create_multi_prefix_dataset(
                    self._model_path,
                    config.prompt_tokens,
                    config.data_num,
                    dataset_dir,
                    prefix_flag,
                    1,  # dp
                    repeat_rate,
                    1,  # seed
                    config.prefix_num,
                    None, None, None, None,  # variable-length params
                )
            finally:
                os.chdir(cwd)
            logger.info("AISBench: generated dataset: {}", dataset_jsonl)
            return dataset_jsonl
        except Exception as e:
            self.last_error = f"AISBench: dataset generation failed: {e}"
            logger.error(self.last_error)
            return None

    # ── API config ──────────────────────────────────────────────────────────

    def _render_and_link_api(self, config: BenchmarkConfig, host: str, port: int) -> None:
        if config.test_type == "text":
            test_type_ref = "VLLMCustomAPIChat"
            test_abbr = "vllm-api-general-chat"
        else:
            test_type_ref = "VLLMCustomAPIChatStream"
            test_abbr = "vllm-api-stream-chat"

        rendered = _API_TEMPLATE.format(
            test_type_ref=test_type_ref,
            test_abbr=test_abbr,
            model_path=self._model_path,
            model_name=self._model_name,
            api_key="",
            request_rate=config.request_rate,
            host_ip=host,
            host_port=port,
            output_len=config.output_tokens,
            concurrency=config.concurrent_requests,
        )

        temp_api = Path(self._work_path) / "ais_bench" / "benchmark" / "configs" / "models" / "vllm_api" / "vllm_api_chat_temp.py"
        temp_api.parent.mkdir(parents=True, exist_ok=True)
        # Write directly (rather than symlink) — simplest and most robust.
        temp_api.write_text(rendered, encoding="utf-8")
        logger.info("AISBench: wrote API config: {}", temp_api)

    def _link_dataset(self, dataset_path: str) -> None:
        gsm8k_dir = Path(self._work_path) / "ais_bench" / "datasets" / "gsm8k"
        gsm8k_dir.mkdir(parents=True, exist_ok=True)
        test_jsonl = gsm8k_dir / "test.jsonl"
        train_jsonl = gsm8k_dir / "train.jsonl"
        _symlink_force(os.path.abspath(dataset_path), str(test_jsonl))
        if not train_jsonl.exists():
            train_jsonl.touch()
        logger.info("AISBench: linked dataset {} -> {}", dataset_path, test_jsonl)

    # ── command ─────────────────────────────────────────────────────────────

    def _build_command(self, output_dir: Path) -> str:
        return (
            f"ais_bench --models vllm_api_chat_temp "
            f"--datasets gsm8k_gen_0_shot_cot_str_perf "
            f"--mode perf --summarizer {self._summarizer} "
            f"--work-dir {output_dir} --debug --num-warmups 0"
        )

    # ── result ──────────────────────────────────────────────────────────────

    def _build_result(
        self,
        output_lines: list[str],
        config: BenchmarkConfig,
        prefix_hit_rate: float,
    ) -> BenchmarkResult:
        metrics = _parse_metrics(output_lines, self._npu_num)
        logger.info("AISBench: parsed metrics: {}", metrics)

        def _num(key: str) -> float:
            v = metrics.get(key, 0.0)
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        total_req = _num("total_req")
        return BenchmarkResult(
            throughput_req_per_sec=_num("qps"),
            output_tokens_per_sec=_num("output_throughput"),
            p50_latency_ms=_num("E2EL avg"),
            p95_latency_ms=_num("E2EL P90"),
            p99_latency_ms=_num("E2EL P90"),
            ttft_ms=_num("TTFT avg"),
            itl_ms=_num("TPOT avg"),
            total_requests=int(total_req),
            successful_requests=int(total_req),
            failed_requests=0,
            duration_seconds=_num("E2E_time"),
            avg_output_tokens_per_request=_num("output_len"),
            expected_output_tokens=config.output_tokens,
            expected_prompt_tokens=config.prompt_tokens,
            prefix_cache_hit_rate=prefix_hit_rate,
        )

    # ── provider metadata ───────────────────────────────────────────────────

    def supports_synthetic_workloads(self) -> bool:
        return True

    def supports_real_datasets(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "aisbench"
