#!/bin/sh
# Qwen3.5-9B serve for AISBench 16K/32K context perf tests (single NPU, 910B3).
# max-model-len 33792 = 32768 (32K input) + 1024 headroom for 256 decode tokens,
# so both 16K-input and 32K-input tests fit. Native dtype (bfloat16 weights +
# float32 mamba ssm) — do NOT force float16 (breaks gated-delta-rule CANN kernel).
export VLLM_USE_MODELSCOPE=True
export MODEL_PATH=/data/qwen3.5/Qwen3.5-9B

vllm serve $MODEL_PATH \
--host 127.0.0.1 \
--port 1025 \
--tensor-parallel-size 1 \
--served-model-name qwen3.5 \
--max-num-seqs 32 \
--max-model-len 33792 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [2,4,8,16]}' \
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'
