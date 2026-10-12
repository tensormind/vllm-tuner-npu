#!/bin/sh
# Qwen3.5-27B 单机 8 卡 (910B3 ×8) vLLM 启动脚本 —— output_throughput 摸高用。
# 与 9B 同架构 (Qwen3_5ForConditionalGeneration, hybrid 线性注意力 gated-delta-rule + 全注意力,
#   mamba_ssm_dtype=float32, MTP mtp_num_hidden_layers=1)。
#
# 关键: 原生 dtype (bfloat16 权重 + float32 mamba SSM)。
#   切勿加 --dtype float16 / --mamba-ssm-cache-dtype float16 ——
#   强制 float16 会触发 CANN 算子 aclnnChunkGatedDeltaRuleFwdH 的 EZ1009 dtype 不匹配崩溃。
#
# 显存预算 (TP=8): 权重 51.75GB/8 ≈ 6.5GB/卡; 0.90 利用率 → 单卡可用 ~58GB, 余 ~51GB 给 KV。
#   32K 全注意力 KV ≈ 8KB/token/卡 × 32768 ≈ 256MB/请求/卡 → 32K 下可放 ~190 路并发; 16K 可放 ~390 路。
#   故 max-num-seqs 192 对 32K 安全 (16K 更宽裕)。
#
# 启动: cd /root/workspace/qwen && nohup ./qwen35-27b-8npu.sh > 27b.log 2>&1 &
# 验证: curl -s http://127.0.0.1:1025/v1/models | python3 -m json.tool
# 指标: curl -s http://127.0.0.1:1025/metrics | grep -iE 'throughput|vllm'
# 停止: pkill -f 'vllm serve'; ps aux|grep EngineCore|grep -v grep|awk '{print $2}'|xargs -r kill -9

export VLLM_USE_MODELSCOPE=True
export MODEL_PATH=/data/qwen3.5/Qwen3.5-27B

vllm serve $MODEL_PATH \
--host 0.0.0.0 \
--port 1025 \
--tensor-parallel-size 8 \
--served-model-name qwen3.5 \
--max-num-seqs 192 \
--max-model-len 34816 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [2,4,8,16,32,64,128]}' \
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'

# ===== 调参说明 =====
# --tensor-parallel-size 8 : 模型按 8 卡切分, 单副本。权重 6.5GB/卡, KV 余量充足。
#                            若想换更高聚合吞吐的多副本拓扑, 见下方备选 (TP=4×DP=2 / TP=2×DP=4)。
# --max-num-seqs 192       : 32K 下显存安全上限约 190 路; 16K 可到 ~390。摸高扫描 cc 上限取 192。
# --max-model-len 34816    : = 32778(32K 实测 token) + 1024 输出 + 余量; 16K/32K 均兼容。
# cudagraph_capture_sizes  : 覆盖 8 卡下大 batch 档 (含 128), 过大占显存可按实测并发删减。
# 端口 1025 : 对齐 aisbench_auto_tools_prefix/config.py 的 HOST_PORT, 免改 config。
#
# ===== 备选: 多副本拓扑 (output_throughput 聚合更高, 无跨 8 卡 all-reduce) =====
# 27B output 吞吐受解码限; 多副本并行常给出更高聚合 output_throughput:
# # TP=4 × DP=2 (2 副本, 各 4 卡, 权重 12.9GB/卡, KV 余 ~45GB/卡):
# # --tensor-parallel-size 4 --data-parallel-size 2 --max-num-seqs 96
# # TP=2 × DP=4 (4 副本, 各 2 卡, 权重 25.9GB/卡, KV 余 ~32GB/卡):
# # --tensor-parallel-size 2 --data-parallel-size 4 --max-num-seqs 64
