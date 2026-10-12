#!/bin/sh
# Qwen3.5-9B 单机 8 卡 — DP=2 × TP=4 部署 (用于 prefix cache 测试, --dp 2)。
# 2 个数据并行副本, 各占 4 卡 (tensor-parallel-size=4), 共 8 卡。
# prefix cache 在 vllm V1 默认开启; 本服务暴露 /metrics 供 aisbench_test.py 读取命中率。
#
# 关键: 原生 dtype (bfloat16 权重 + float32 mamba SSM), 切勿加 --dtype float16
#        (否则 gated-delta-rule CANN 算子 EZ1009 崩溃)。
#
# 启动: cd /root/workspace/qwen && ./qwen35-9b-8npu-dp2.sh
#   先停旧服务: pkill -f 'vllm serve'  (并 kill 拖留 VLLM::EngineCore)
# 验证: curl -s http://127.0.0.1:1025/v1/models | python3 -m json.tool
# 指标: curl -s http://127.0.0.1:1025/metrics | grep -i prefix
# 端口 1025 对齐 aisbench_auto_tools_prefix/config.py 的 HOST_PORT (POD_INFO 为空时取此端点)。
export VLLM_USE_MODELSCOPE=True
export MODEL_PATH=/data/qwen3.5/Qwen3.5-9B

vllm serve $MODEL_PATH \
--host 0.0.0.0 \
--port 1025 \
--tensor-parallel-size 1 \
--data-parallel-size 8 \
--served-model-name qwen3.5 \
--max-num-seqs 64 \
--max-model-len 34816 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [2,4,8,16,32,64]}' \
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'

# ===== 说明 =====
# --tensor-parallel-size 4 --data-parallel-size 2 : TP=4×DP=2=8 卡。2 副本各自独立 KV/prefix cache。
# --max-num-seqs 64 : 单副本并发上限 64 (DP=2 → 在 NPU 上最多 128 路并行)。16K 下显存富余;
#                     32K×64 若 HBM 偏紧, vLLM 会自动少 admit, 跑时盯 npu-smi。
# --max-model-len 34816 : 兼容 32K(32778)+1024 输出; 16K 自然也兼容。
# prefix cache : vllm V1 默认开启, 无需额外 flag; /metrics 含 prefix cache 计数。
# 若 vllm-ascend 不支持单进程 TP×DP, 回退: 起 2 个进程 (各 TP=4, 端口 8000/8001) + 前端负载均衡。
