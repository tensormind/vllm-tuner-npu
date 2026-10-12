import pytest

from vllm_tuner.core.models import AcceleratorType
from vllm_tuner.hardware.base import HardwareMonitor
from vllm_tuner.hardware.null import NullMonitor
from vllm_tuner.hardware.nvml import NVMLMonitor
from vllm_tuner.hardware.tpu import TPUMonitor


class TestHardwareMonitorABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            HardwareMonitor()


class TestNullMonitor:
    def test_start_stop_no_error(self):
        monitor = NullMonitor()
        monitor.start_collection(interval_seconds=0.5)
        monitor.stop_collection()

    def test_current_stats_zeros(self):
        monitor = NullMonitor()
        stats = monitor.get_current_stats()
        assert stats.memory_used_mb == 0.0
        assert stats.temperature_c == 0.0

    def test_aggregate_stats_zeros(self):
        monitor = NullMonitor()
        agg = monitor.get_aggregate_stats()
        assert agg.num_samples == 0
        assert agg.peak_memory_used_mb == 0.0

    def test_accelerator_info(self):
        monitor = NullMonitor()
        info = monitor.get_accelerator_info()
        assert info.name == "none"
        assert info.count == 0


class TestNVMLMonitor:
    def test_gpu_type(self):
        monitor = NVMLMonitor(device_ids=[0])
        info = monitor.get_accelerator_info()
        assert info.accelerator_type == AcceleratorType.GPU

    def test_graceful_without_gpu(self):
        """On Mac M1 (no NVIDIA GPU), returns empty stats gracefully."""
        monitor = NVMLMonitor()
        stats = monitor.get_current_stats()
        from vllm_tuner.core.models import HardwareStats

        assert isinstance(stats, HardwareStats)

    def test_start_stop(self):
        monitor = NVMLMonitor()
        monitor.start_collection()
        assert monitor._collecting is True
        monitor.stop_collection()
        assert monitor._collecting is False

    def test_aggregate_stats_empty(self):
        monitor = NVMLMonitor()
        agg = monitor.get_aggregate_stats()
        from vllm_tuner.core.models import AggregateHardwareStats

        assert isinstance(agg, AggregateHardwareStats)


class TestTPUMonitor:
    def test_tpu_type(self):
        monitor = TPUMonitor(tpu_type="v6e-4")
        info = monitor.get_accelerator_info()
        assert info.accelerator_type == AcceleratorType.TPU
        assert "v6e-4" in info.name

    def test_stats_not_implemented(self):
        monitor = TPUMonitor()
        with pytest.raises(NotImplementedError):
            monitor.get_current_stats()
        with pytest.raises(NotImplementedError):
            monitor.get_aggregate_stats()

    def test_start_stop(self):
        monitor = TPUMonitor()
        monitor.start_collection()
        assert monitor._collecting is True
        monitor.stop_collection()
        assert monitor._collecting is False
