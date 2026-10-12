from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="/data/qwen3.5/Qwen3.5-9B",
        model="qwen3.5",
        api_key="",
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1",
        host_port=1025,
        max_out_len=1024,
        batch_size=192,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
        )
    )
]
