#!/bin/sh
# Qwen3.5-27B vLLM LoRA 验证脚本 (910B3, TP=4) —— 已验证可用 (2026-09-10)
#
# 为什么是 TP=4 而不是 8:
#   1. Ascend sgmv_expand 内核要求 LoRA rank <= 输出切片/TP。最紧的模块是 GDN 的
#      in_proj_ba (MergedColumn [48,48]): TP=8 时每卡每 partition 仅 6 维, 且
#      --max-lora-rank 无 4 档 (只接受 1,8,16,...) -> TP=8 实际不可用;
#      TP=4 时切片 12 >= 8, rank 8 正常。
#   2. NPU6/7 被宿主机其他进程占用 32-42GB, TP=4 只用 0-3 卡, 避开污染。
#
# 前置: vllm-ascend 已打本地补丁 (见 vllm_ascend_lora_fix.patch.md),
#       否则启动即崩 "hidden in should be smaller than hidden out"。
#
# adapter: ./mylora (r=8, lora_B=0, 零输出影响, 冒烟用); 用 gen_lora.py 重新生成
# 启动: nohup ./qwen35-27b-lora-tp4.sh > 27b-lora.log 2>&1 &
# 验证: curl -s http://127.0.0.1:1025/v1/models | python3 -m json.tool
#       请求时 "model":"mylora" 走 LoRA, "model":"qwen3.5" 走基座, 可并存

export VLLM_USE_MODELSCOPE=True
export MODEL_PATH=/data/qwen3.5/Qwen3.5-27B
export LORA_PATH=/data/workspaces/qwen/mylora

vllm serve $MODEL_PATH \
--host 0.0.0.0 \
--port 1025 \
--tensor-parallel-size 4 \
--served-model-name qwen3.5 \
--max-num-seqs 8 \
--max-model-len 8192 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--enable-lora \
--lora-modules mylora=$LORA_PATH \
--max-lora-rank 8 \
--enforce-eager

# 冒烟通过后的可选调整:
#   - 去掉 --enforce-eager, 加回 cudagraph/ascend 编译参数 (参考 qwen35-27b-8npu.sh)
#   - 加回 MTP 投机解码 --speculative-config (LoRA×MTP 组合未验证, 需单独试)
#   - max-num-seqs / max-model-len 按 0-3 卡 KV (40GB+) 自行放大
