#!/usr/bin/env python3
"""生成用于 vLLM LoRA 启动测试的随机初始化 LoRA adapter。

- lora_B 全零 → adapter 对模型输出零影响, 适合冒烟测试。
- 命名采用 peft/HF 标准格式 (base_model.model.model.language_model.layers.N.*),
  vLLM 的 hf_to_vllm_mapper 会自动映射到 language_model.model.layers.N.*。
- packed 模块 (qkv_proj/gate_up_proj) 在 adapter 中使用组件名 q/k/v_proj、gate/up_proj,
  vLLM 加载时自动合并。
- 仅覆盖全注意力层的 self_attn + 全部层的 MLP; linear_attn 层 vLLM LoRA 不支持, 跳过。
"""
import json
import os

import torch
from safetensors.torch import save_file

MODEL_CONFIG = "/data/qwen3.5/Qwen3.5-27B/config.json"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mylora")
R = 8          # LoRA rank, 须 <= vLLM --max-lora-rank (默认 16)
ALPHA = 16     # scaling = alpha / r = 2.0
SEED = 0
# 注意: Ascend sgmv_expand 内核要求 rank <= 输出切片/TP。最紧的模块是 GDN 的
# in_proj_ba (MergedColumn [48,48]) -> TP=8 时每卡每 partition 仅 6 维, rank 只能
# 用 1 (且 --max-lora-rank 无 4 档), dummy 预热即报 "hidden in should be smaller
# than hidden out"。TP=4 时切片 12 >= 8, rank 8 可用 -> LoRA 测试用 TP=4 拓扑。

cfg = json.load(open(MODEL_CONFIG))["text_config"]
hidden = cfg["hidden_size"]                      # 5120
n_layers = cfg["num_hidden_layers"]              # 64
head_dim = cfg["head_dim"]                       # 256
n_heads = cfg["num_attention_heads"]             # 24  -> q_proj out 6144
n_kv = cfg["num_key_value_heads"]                # 4   -> k/v_proj out 1024
inter = cfg["intermediate_size"]                 # 17408
layer_types = cfg["layer_types"]

torch.manual_seed(SEED)
tensors = {}
prefix = "base_model.model.model.language_model.layers"


def add(module_path: str, din: int, dout: int):
    tensors[f"{module_path}.lora_A.weight"] = torch.randn(R, din, dtype=torch.bfloat16) * 0.01
    tensors[f"{module_path}.lora_B.weight"] = torch.zeros(dout, R, dtype=torch.bfloat16)


for i in range(n_layers):
    p = f"{prefix}.{i}"
    # MLP: 每层都有
    for name, din, dout in (
        ("gate_proj", hidden, inter),
        ("up_proj", hidden, inter),
        ("down_proj", inter, hidden),
    ):
        add(f"{p}.mlp.{name}", din, dout)
    # self_attn: 仅全注意力层 (每 4 层 1 层)
    if layer_types[i] == "full_attention":
        for name, din, dout in (
            ("q_proj", hidden, n_heads * head_dim),
            ("k_proj", hidden, n_kv * head_dim),
            ("v_proj", hidden, n_kv * head_dim),
            ("o_proj", hidden, hidden),
        ):
            add(f"{p}.self_attn.{name}", din, dout)

adapter_config = {
    "base_model_name_or_path": "/data/qwen3.5/Qwen3.5-27B",
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "lora_alpha": ALPHA,
    "lora_dropout": 0.0,
    "peft_type": "LORA",
    "r": R,
    "rank_pattern": {},
    "alpha_pattern": {},
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "task_type": "CAUSAL_LM",
}

os.makedirs(OUT_DIR, exist_ok=True)
save_file(tensors, os.path.join(OUT_DIR, "adapter_model.safetensors"))
with open(os.path.join(OUT_DIR, "adapter_config.json"), "w") as f:
    json.dump(adapter_config, f, indent=2)

n_attn = sum(1 for t in layer_types if t == "full_attention")
n_params = sum(t.numel() for t in tensors.values())
size_mb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1024 / 1024
print(f"输出目录: {OUT_DIR}")
print(f"张量数量: {len(tensors)} (MLP {n_layers}层 × 3模块, self_attn {n_attn}层 × 4模块)")
print(f"参数总量: {n_params/1e6:.1f}M, 文件大小: {size_mb:.1f} MB")
print(f"rank={R}, alpha={ALPHA}, scaling={ALPHA/R}")
