import pytest

from vllm_tuner.core.models import TrialConfig
from vllm_tuner.vllm.launcher import VLLMLauncher
from vllm_tuner.vllm.telemetry import VLLMTelemetryParser


class TestVLLMLauncher:
    def test_server_url(self):
        launcher = VLLMLauncher(model="test-model", host="0.0.0.0", port=9000)
        assert launcher.server_url == "http://0.0.0.0:9000"

    def test_not_running_initially(self):
        launcher = VLLMLauncher(model="test-model")
        assert launcher.is_running is False

    def test_build_command(self):
        launcher = VLLMLauncher(model="meta-llama/Llama-3-8B", port=8000)
        config = TrialConfig(
            trial_number=1,
            parameters={"gpu_memory_utilization": 0.9},
            static_parameters={"tensor_parallel_size": 2},
        )
        cmd = launcher.build_command(config)
        assert "vllm" in cmd
        assert "serve" in cmd
        assert "meta-llama/Llama-3-8B" in cmd
        assert "--gpu-memory-utilization" in cmd
        assert "0.9" in cmd
        assert "--tensor-parallel-size" in cmd
        assert "2" in cmd

    def test_build_command_empty_params(self):
        launcher = VLLMLauncher(model="test-model", host="127.0.0.1", port=8000)
        config = TrialConfig(trial_number=1, parameters={})
        cmd = launcher.build_command(config)
        assert cmd == ["vllm", "serve", "test-model", "--host", "127.0.0.1", "--port", "8000"]

    def test_build_command_boolean_true_flag(self):
        launcher = VLLMLauncher(model="test-model", port=8000)
        config = TrialConfig(
            trial_number=1,
            parameters={},
            static_parameters={"enforce_eager": True},
        )
        cmd = launcher.build_command(config)
        assert "--enforce-eager" in cmd
        # Should NOT have "True" as a separate argument
        assert "True" not in cmd

    def test_build_command_boolean_false_flag(self):
        launcher = VLLMLauncher(model="test-model", port=8000)
        config = TrialConfig(
            trial_number=1,
            parameters={"enable_prefix_caching": "false"},
            static_parameters={},
        )
        cmd = launcher.build_command(config)
        assert "--no-enable-prefix-caching" in cmd
        assert "--enable-prefix-caching" not in [c for c in cmd if c == "--enable-prefix-caching"]

    def test_build_command_boolean_string_true(self):
        launcher = VLLMLauncher(model="test-model", port=8000)
        config = TrialConfig(
            trial_number=1,
            parameters={"enable_chunked_prefill": "true"},
            static_parameters={},
        )
        cmd = launcher.build_command(config)
        assert "--enable-chunked-prefill" in cmd
        assert "true" not in cmd

    def test_build_command_float_to_int(self):
        launcher = VLLMLauncher(model="test-model", port=8000)
        config = TrialConfig(
            trial_number=1,
            parameters={"max_num_seqs": 32.0, "max_num_batched_tokens": 768.0},
            static_parameters={},
        )
        cmd = launcher.build_command(config)
        # Should be "32" not "32.0"
        idx = cmd.index("--max-num-seqs")
        assert cmd[idx + 1] == "32"
        idx = cmd.index("--max-num-batched-tokens")
        assert cmd[idx + 1] == "768"

    def test_build_command_preserves_real_floats(self):
        launcher = VLLMLauncher(model="test-model", port=8000)
        config = TrialConfig(
            trial_number=1,
            parameters={"gpu_memory_utilization": 0.85},
            static_parameters={},
        )
        cmd = launcher.build_command(config)
        idx = cmd.index("--gpu-memory-utilization")
        assert cmd[idx + 1] == "0.85"

    def test_stop_when_not_running(self):
        """Stopping a server that was never started should be a no-op."""
        launcher = VLLMLauncher(model="test-model")
        launcher.stop()  # should not raise

    def test_health_check_no_server(self):
        """Health check when nothing is listening should return False."""
        launcher = VLLMLauncher(model="test-model", port=19999)
        assert launcher.health_check() is False

    def test_read_logs_empty(self):
        launcher = VLLMLauncher(model="test-model")
        assert launcher.read_logs() == []

    def test_pid_none_initially(self):
        launcher = VLLMLauncher(model="test-model")
        assert launcher.pid is None


class TestVLLMTelemetryParser:
    def test_empty_logs(self):
        parser = VLLMTelemetryParser()
        telemetry = parser.parse_logs([])
        assert telemetry.oom_detected is False
        assert telemetry.kv_cache_usage is None
        assert telemetry.num_preemptions == 0

    def test_oom_detection(self):
        parser = VLLMTelemetryParser()
        logs = [
            "INFO: Starting vLLM server",
            "ERROR: CUDA out of memory. Tried to allocate 2.00 GiB",
        ]
        telemetry = parser.parse_logs(logs)
        assert telemetry.oom_detected is True

    def test_oom_detection_torch(self):
        parser = VLLMTelemetryParser()
        logs = ["torch.cuda.OutOfMemoryError: CUDA out of memory"]
        assert parser.detect_oom(logs) is True

    def test_no_oom(self):
        parser = VLLMTelemetryParser()
        logs = ["INFO: Server started successfully"]
        assert parser.detect_oom(logs) is False

    def test_kv_cache_usage(self):
        parser = VLLMTelemetryParser()
        logs = [
            "INFO: KV cache usage: 45.2%",
            "INFO: KV cache usage: 67.8%",
        ]
        telemetry = parser.parse_logs(logs)
        assert telemetry.kv_cache_usage == pytest.approx(0.678)

    def test_kv_cache_usage_last(self):
        parser = VLLMTelemetryParser()
        logs = [
            "KV cache usage: 30.0%",
            "KV cache usage: 50.0%",
            "KV cache usage: 80.0%",
        ]
        usage = parser.get_kv_cache_usage(logs)
        assert usage == pytest.approx(0.80)

    def test_preemption_counting(self):
        parser = VLLMTelemetryParser()
        logs = [
            "WARNING: preemption occurred",
            "INFO: normal operation",
            "WARNING: request preempted due to memory",
        ]
        telemetry = parser.parse_logs(logs)
        assert telemetry.num_preemptions == 2

    def test_swap_counting(self):
        parser = VLLMTelemetryParser()
        logs = [
            "INFO: swapping blocks to CPU",
            "INFO: swapped 3 blocks",
        ]
        telemetry = parser.parse_logs(logs)
        assert telemetry.num_swaps == 2
