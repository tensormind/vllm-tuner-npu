"""Unit tests for the AISBenchProvider (no ais_bench CLI required).

Covers the pure-logic pieces ported from aisbench_auto_tools_prefix:
  - metric-table parsing (_parse_metrics)
  - API config template rendering (_render_and_link_api)
  - dataset symlink (_link_dataset)
  - command construction (_build_command)
  - host/port resolution
"""
import os
import tempfile
from pathlib import Path

import pytest

from vllm_tuner.benchmarks.aisbench import (
    AISBenchProvider,
    _parse_metrics,
    _resolve_host_port,
)
from vllm_tuner.core.models import BenchmarkConfig

# A realistic ais_bench perf-summary table fragment (same shape as the real log).
_SAMPLE_LINES = [
    "[ais_bench] Current exp folder: ./outputs/default/20260828_104135",
    "│ E2EL                     │ total   │ 1168853.8 ms     │ 264318.8 ms     │ 1840218.5 ms      │ 1311508.5 ms   │ 1574992.1 ms    │ 1628579.4 ms     │ 1816102.0 ms      │ 256 │",
    "│ TTFT                     │ total   │ 921703.2 ms      │ 7463.3 ms       │ 1575965.8 ms      │ 1051403.6 ms   │ 1322030.4 ms    │ 1363782.5 ms     │ 1554775.0 ms      │ 256 │",
    "│ TPOT                     │ total   │ 241.6 ms         │ 25.0 ms         │ 267.3 ms          │ 255.3 ms       │ 257.2 ms        │ 259.1 ms         │ 265.0 ms          │ 256 │",
    "│ Benchmark Duration      │ total   │ 2130899.6786 ms   │",
    "│ Total Requests          │ total   │ 256               │",
    "│ Concurrency             │ total   │ 140.4226          │",
    "│ Max Concurrency         │ total   │ 192               │",
    "│ Request Throughput      │ total   │ 0.1201 req/s      │",
    "│ Output Token Throughput │ total   │ 123.0203 token/s  │",
    "│ Total Token Throughput  │ total   │ 4060.8688 token/s │",
]


def test_parse_metrics_extracts_all_fields():
    m = _parse_metrics(_SAMPLE_LINES, npu_num=1)
    assert m["total_req"] == 256.0
    assert m["qps"] == pytest.approx(0.1201, rel=1e-3)
    assert m["output_throughput"] == pytest.approx(123.0203, rel=1e-3)
    assert m["E2E_throughput"] == pytest.approx(4060.8688, rel=1e-3)
    assert m["TTFT avg"] == pytest.approx(921703.2)
    assert m["TTFT P90"] == pytest.approx(1363782.5)
    assert m["TPOT avg"] == pytest.approx(241.6)
    assert m["E2E_time"] == pytest.approx(2130.8996786, rel=1e-3)
    assert m["cc"] == pytest.approx(140.4226)
    assert m["max_cc"] == "192"


def test_parse_metrics_empty_returns_defaults():
    m = _parse_metrics([], npu_num=1)
    # defaults are non-zero sentinels
    assert m["total_req"] == 99999
    assert m["qps"] == 9999


def test_resolve_host_port_from_url():
    host, port = _resolve_host_port("http://127.0.0.1:8000", "fallback", 9999)
    assert host == "127.0.0.1"
    assert port == 8000


def test_resolve_host_port_fallback():
    host, port = _resolve_host_port("", "10.0.0.1", 9000)
    assert host == "10.0.0.1"
    assert port == 9000


def test_render_and_link_api_writes_template():
    with tempfile.TemporaryDirectory() as wp:
        p = AISBenchProvider(
            model_path="/data/qwen3.5/Qwen3.5-9B",
            model_name="qwen3.5",
            aisbench_work_path=wp,
        )
        cfg = BenchmarkConfig(
            provider="aisbench",
            output_tokens=1024,
            concurrent_requests=32,
            request_rate=0,
            test_type="stream",
        )
        p._render_and_link_api(cfg, "127.0.0.1", 8000)
        api_file = Path(wp) / "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py"
        assert api_file.exists()
        text = api_file.read_text()
        assert "VLLMCustomAPIChatStream" in text
        assert 'path="/data/qwen3.5/Qwen3.5-9B"' in text
        assert 'model="qwen3.5"' in text
        assert 'host_ip="127.0.0.1"' in text
        assert "host_port=8000" in text
        assert "max_out_len=1024" in text
        assert "batch_size=32" in text


def test_render_api_text_mode():
    with tempfile.TemporaryDirectory() as wp:
        p = AISBenchProvider(model_path="/m", model_name="m", aisbench_work_path=wp)
        cfg = BenchmarkConfig(provider="aisbench", output_tokens=64, concurrent_requests=4, test_type="text")
        p._render_and_link_api(cfg, "1.2.3.4", 7000)
        text = (Path(wp) / "ais_bench/benchmark/configs/models/vllm_api/vllm_api_chat_temp.py").read_text()
        assert "VLLMCustomAPIChat\n" in text or "VLLMCustomAPIChat " in text or "VLLMCustomAPIChat," in text
        assert "vllm-api-general-chat" in text


def test_link_dataset_creates_symlink_and_train():
    with tempfile.TemporaryDirectory() as wp:
        p = AISBenchProvider(model_path="/m", model_name="m", aisbench_work_path=wp)
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write('{"question": "hi"}\n')
            ds = f.name
        try:
            p._link_dataset(ds)
            gsm8k = Path(wp) / "ais_bench/datasets/gsm8k"
            assert (gsm8k / "test.jsonl").is_symlink()
            assert os.readlink(str(gsm8k / "test.jsonl")) == os.path.abspath(ds)
            assert (gsm8k / "train.jsonl").exists()
        finally:
            os.unlink(ds)


def test_build_command_shape():
    p = AISBenchProvider(
        model_path="/m", model_name="m", aisbench_work_path="/wp",
        summarizer="stable_stage",
    )
    cmd = p._build_command(Path("/out/dir"))
    assert "ais_bench" in cmd
    assert "--models vllm_api_chat_temp" in cmd
    assert "--datasets gsm8k_gen_0_shot_cot_str_perf" in cmd
    assert "--mode perf" in cmd
    assert "--summarizer stable_stage" in cmd
    assert "--work-dir /out/dir" in cmd
    assert "--num-warmups 0" in cmd


def test_provider_metadata():
    p = AISBenchProvider(model_path="/m", model_name="qwen3.5", aisbench_work_path="/wp")
    assert p.name == "aisbench"
    assert p.supports_synthetic_workloads() is True
    assert p.supports_real_datasets() is True


def test_build_result_maps_metrics():
    p = AISBenchProvider(model_path="/m", model_name="qwen3.5", aisbench_work_path="/wp")
    cfg = BenchmarkConfig(provider="aisbench", prompt_tokens=16384, output_tokens=1024)
    result = p._build_result(_SAMPLE_LINES, cfg, prefix_hit_rate=0.42)
    assert result.total_requests == 256
    assert result.successful_requests == 256
    assert result.throughput_req_per_sec == pytest.approx(0.1201, rel=1e-3)
    assert result.output_tokens_per_sec == pytest.approx(123.0203, rel=1e-3)
    assert result.ttft_ms == pytest.approx(921703.2)
    assert result.itl_ms == pytest.approx(241.6)
    assert result.duration_seconds == pytest.approx(2130.8996786, rel=1e-3)
    assert result.expected_output_tokens == 1024
    assert result.expected_prompt_tokens == 16384
    assert result.prefix_cache_hit_rate == pytest.approx(0.42)
