#!/usr/bin/env python3
"""验证 NPUMonitor 能否正确采集昇腾 NPU 硬件指标。"""
from vllm_tuner.hardware.npu import NPUMonitor


def main():
    print("=== 1. 初始化 NPUMonitor (全量8卡) ===")
    mon = NPUMonitor()
    print(f"  device_ids: {mon._device_ids}")
    print(f"  initialized: {mon._initialized}")

    print("\n=== 2. get_accelerator_info ===")
    info = mon.get_accelerator_info()
    print(f"  type: {info.accelerator_type}")
    print(f"  name: {info.name}")
    print(f"  count: {info.count}")
    print(f"  memory_per_device_mb: {info.memory_per_device_mb}")
    print(f"  total_memory_mb: {info.total_memory_mb}")

    print("\n=== 3. get_current_stats (单次快照) ===")
    stats = mon.get_current_stats()
    print(f"  memory_used_mb:        {stats.memory_used_mb:.1f}")
    print(f"  memory_total_mb:       {stats.memory_total_mb:.1f}")
    print(f"  memory_utilization:   {stats.memory_utilization:.4f}")
    print(f"  accelerator_utilization: {stats.accelerator_utilization:.4f}")
    print(f"  temperature_c:         {stats.temperature_c}")
    print(f"  power_usage_w:         {stats.power_usage_w:.1f}")
    print(f"  power_limit_w:         {stats.power_limit_w:.1f}")

    print("\n=== 4. 后台采集 3 秒 ===")
    mon.start_collection(interval_seconds=1.0)
    import time
    time.sleep(3)
    mon.stop_collection()

    print("\n=== 5. get_aggregate_stats ===")
    agg = mon.get_aggregate_stats()
    print(f"  peak_memory_used_mb:  {agg.peak_memory_used_mb:.1f}")
    print(f"  avg_memory_used_mb:   {agg.avg_memory_used_mb:.1f}")
    print(f"  avg_utilization:      {agg.avg_utilization:.4f}")
    print(f"  max_temperature_c:     {agg.max_temperature_c}")
    print(f"  total_power_w:        {agg.total_power_w:.1f}")
    print(f"  num_samples:          {agg.num_samples}")

    print("\n=== 6. ASCEND_RT_VISIBLE_DEVICES 限定测试 ===")
    import os
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
    mon2 = NPUMonitor()
    print(f"  device_ids (限定0-3): {mon2._device_ids}")
    stats2 = mon2.get_current_stats()
    print(f"  memory_total_mb: {stats2.memory_total_mb:.1f} (应为 4×65536=262144)")

    print("\n✅ NPUMonitor 验证通过" if agg.num_samples > 0 else "\n❌ 采集失败")


if __name__ == "__main__":
    main()
