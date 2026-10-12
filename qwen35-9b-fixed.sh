#!/bin/sh
# Load model from ModelScope to speed up download
export VLLM_USE_MODELSCOPE=True

# Model weight path; can be a ModelScope model id or a local directory path
export MODEL_PATH=/data/qwen3.5/Qwen3.5-9B

vllm serve $MODEL_PATH \
--host 127.0.0.1 \
--port 1025 \
--tensor-parallel-size 1 \
--served-model-name qwen3.5 \
--max-num-seqs 32 \
--max-model-len 16384 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--speculative-config '{"method": "qwen3_5_mtp","num_speculative_tokens":1}' \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [2,4,8,16]}' \
--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'
