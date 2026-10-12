#!/usr/bin/env python3
# AISBench performance config: Qwen3.5-9B, 16K input context, Ascend 910B3 (1 NPU).
# Input prompt = "A " repeated 16384 times -> Qwen3.5 tokenizer maps it 1:1 to 16384
# tokens, so the vLLM server receives exactly 16K input tokens per request.
# Output: fixed 256 decode tokens (ignore_eos=True) for stable TPOT measurement.
from mmengine.config import read_base

from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets as base_synthetic_datasets,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as base_vllm_api_stream_chat,
    )

INPUT_LEN = 16384
OUTPUT_LEN = 256
CONCURRENCY = 32
NUM_PROMPTS = 100

model_cfg = dict(base_vllm_api_stream_chat[0])
model_cfg["abbr"] = "qwen35-9b-16k-bs32"
model_cfg["model"] = "qwen3.5"
model_cfg["path"] = "/data/qwen3.5/Qwen3.5-9B"
model_cfg["host_ip"] = "127.0.0.1"
model_cfg["host_port"] = 1025
model_cfg["max_out_len"] = OUTPUT_LEN
model_cfg["batch_size"] = CONCURRENCY
model_cfg["request_rate"] = 0
model_cfg["stream"] = True
model_cfg["trust_remote_code"] = True
model_cfg["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)
models = [model_cfg]

ds = dict(base_synthetic_datasets[0])
ds["abbr"] = "synthetic-16k"
ds["config"] = {
    "Type": "string",
    "RequestCount": NUM_PROMPTS,
    "TrustRemoteCode": True,
    "StringConfig": {
        "Input": {"Method": "uniform", "Params": {"MinValue": INPUT_LEN, "MaxValue": INPUT_LEN}},
        "Output": {"Method": "uniform", "Params": {"MinValue": OUTPUT_LEN, "MaxValue": OUTPUT_LEN}},
    },
}
datasets = [ds]

model_dataset_combinations = [dict(models=models, datasets=datasets)]
work_dir = "outputs/perf_16k/"
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLApiInferTask)),
)
