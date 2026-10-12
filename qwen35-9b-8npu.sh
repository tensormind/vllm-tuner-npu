#!/bin/sh
# Qwen3.5-9B 单机 8 卡 (910B3 ×8) vLLM 启动脚本。
# 在单卡 qwen35-9b-32kcap.sh 基础上: tensor-parallel-size=8, 并发与 cudagraph 档位按 8 卡调高。
#
# 关键: 使用模型原生 dtype (bfloat16 权重 + float32 mamba SSM)。
#   切勿加 --dtype float16 / --mamba-ssm-cache-dtype float16 ——
#   强制 float16 会触发 CANN 算子 aclnnChunkGatedDeltaRuleFwdH 的 EZ1009 dtype 不匹配错误。
#
# 启动: cd /root/workspace/qwen && ./qwen35-9b-8npu.sh
# 验证: curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
# 停止: pkill -f 'vllm serve'  (并 kill 残留 VLLM::EngineCore 子进程)

export VLLM_USE_MODELSCOPE=True
export MODEL_PATH=/data/qwen3.5/Qwen3.5-9B

# 如需限定使用某几张卡, 取消下行注释 (默认自动使用全部 8 张 NPU):
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve $MODEL_PATH \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 8 \
--served-model-name qwen3.5 \
--max-num-seqs 256 \
--max-model-len 34816 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [2,4,8,16,32,64]}' \
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'

# ===== 调参说明 =====
# --tensor-parallel-size 8 : 模型按 8 卡切分 (单机 8 卡默认即此)。单卡 64G 装 9B 富余, 若追求更高
#                            在线吞吐可改用数据并行 DP (8 副本各自单卡), 见下方备选。
# --max-num-seqs 256       : 8 卡下并发上限, 按单卡 32×8 估; 显存富余可再调高, 注意 32K×高并发 HBM 占用。
# --max-model-len 34816    : = 32768(32K 输入, 实测 32778 token) + 1024 输出 + 余量; 兼容 16K/32K 测试。
#                            仅 16K 测试可用 33792; 想跑 128K 长上下文可调到 131072 (8 卡显存足够, mamba KV 小)。
# cudagraph_capture_sizes  : 覆盖 8 卡下更大的常见 batch 档; 过大占显存, 按实际并发取舍。
# --gpu-memory-utilization 0.90 : 单卡预留 90% HBM; 8 卡各自独立预留。
#
# ===== 备选: 数据并行 DP (8 副本, 在线吞吐更高, 无 all-reduce 开销) =====
# vllm serve $MODEL_PATH --host 0.0.0.0 --port 8000 \
#   --data-parallel-size 8 --served-model-name qwen3.5 \
#   --max-num-seqs 32 --max-model-len 34816 --trust-remote-code \
#   --gpu-memory-utilization 0.90 \
#   --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
#   --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2,4,8,16]}' \
#   --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
