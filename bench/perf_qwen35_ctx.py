#!/usr/bin/env python3
# AISBench performance config for Qwen3.5-9B on Ascend 910B3 (single NPU).
# Parametrized via environment variables so the same file serves 16K and 32K runs.
#
# Run:
#   INPUT_LEN=16384 OUTPUT_LEN=256 CONCURRENCY=32 NUM_PROMPTS=100 \
#   HOST_PORT=1025 TAG=16k ais_bench perf_qwen35_ctx.py --mode perf
#
# Input prompt is the literal string "A " repeated INPUT_LEN times, which the
# Qwen3.5 tokenizer maps 1:1 to INPUT_LEN tokens (calibrated), so the vLLM
# server receives exactly INPUT_LEN input tokens per request.
import os
import copy
from mmengine.config import read_base

from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    # default_perf summarizer -> csv/json/html performance report under work_dir
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    # base synthetic string dataset (provides reader/infer/eval cfg scaffolding)
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets as base_synthetic_datasets,
    )
    # base vLLM streaming API model client
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as base_vllm_api_stream_chat,
    )

INPUT_LEN = int(os.environ.get("INPUT_LEN", "16384"))
OUTPUT_LEN = int(os.environ.get("OUTPUT_LEN", "256"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "32"))
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "100"))
HOST_IP = os.environ.get("HOST_IP", "127.0.0.1")
HOST_PORT = int(os.environ.get("HOST_PORT", "1025"))
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.5")
TAG = os.environ.get("TAG", f"{INPUT_LEN // 1024}k")

assert INPUT_LEN <= 32768, "INPUT_LEN must be <= 32768 (server max-model-len cap)"

# --- model: streaming vLLM OpenAI-API client, concurrency=CONCURRENCY ---
model_cfg = copy.deepcopy(base_vllm_api_stream_chat[0])
model_cfg["abbr"] = f"qwen35-9b-{TAG}-bs{CONCURRENCY}"
model_cfg["model"] = MODEL_NAME
model_cfg["host_ip"] = HOST_IP
model_cfg["host_port"] = HOST_PORT
model_cfg["max_out_len"] = OUTPUT_LEN          # decode length per request
model_cfg["batch_size"] = CONCURRENCY           # concurrency (parallel in-flight requests)
model_cfg["request_rate"] = 0                   # 0 == send as fast as possible
model_cfg["stream"] = True
model_cfg["trust_remote_code"] = True
# ignore_eos=True + fixed max_out_len => every request decodes exactly OUTPUT_LEN tokens
model_cfg["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)
models = [model_cfg]

# --- dataset: synthetic, fixed input length (uniform min==max => constant) ---
ds = copy.deepcopy(base_synthetic_datasets[0])
ds["abbr"] = f"synthetic-{TAG}"
ds["config"] = {
    "Type": "string",
    "RequestCount": NUM_PROMPTS,
    "TrustRemoteCode": True,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": INPUT_LEN, "MaxValue": INPUT_LEN},
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": OUTPUT_LEN, "MaxValue": OUTPUT_LEN},
        },
    },
}
datasets = [ds]

# pair models[0] with datasets[0] only (no cartesian product)
model_dataset_combinations = [dict(models=models, datasets=datasets)]

work_dir = f"outputs/perf_{TAG}/"

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLApiInferTask)),
)
