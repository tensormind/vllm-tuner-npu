from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from vllm_tuner.helper.logging import get_logger


logger = get_logger()


# ──────────────────────────────────────
# Device type
# ──────────────────────────────────────


class DeviceType(StrEnum):
    GPU = "gpu"
    TPU = "tpu"


# ──────────────────────────────────────
# Result model
# ──────────────────────────────────────


class ModelAnalysis(BaseModel):
    """Result of analyzing a model's resource requirements."""

    device: DeviceType = DeviceType.GPU
    model_id: str = ""
    param_count: int = 0
    bytes_per_param: float = 2.0
    weights_memory_gb: float = 0.0
    is_quantized: bool = False
    quant_method: str | None = None
    is_moe: bool = False
    num_experts: int = 1

    # Recommended vLLM parameters
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    enforce_eager: bool = False
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    enable_expert_parallel: bool = False
    kv_cache_dtype: str = "auto"
    swap_space: float = 4.0
    block_size: int = 16
    cpu_offload_gb: float = 0.0
    dtype: str = "auto"

    # TPU-specific
    tpu_chip: str | None = None  # e.g. "v6e"

    # vLLM 0.17+ flags (set via --performance-mode, --scheduling-policy, etc.)
    # performance_mode: str = "auto"  # "balanced" | "interactivity" | "throughput" (vLLM 0.17+, PR #34936)
    # scheduling_policy: str = "fcfs"  # "fcfs" | "priority"
    # max_num_partial_prefills: int = 1  # concurrent partial prefills (1-4)
    # num_gpu_blocks_override: int | None = None  # manually set KV cache blocks

    can_fit: bool = True
    warnings: list[str] = []


# ──────────────────────────────────────
# TPU chip specifications
# ──────────────────────────────────────


@dataclass(frozen=True)
class TPUChipSpec:
    """Hardware specification for a TPU chip type."""

    name: str
    hbm_gb: float  # HBM per chip
    chips_per_host: int  # typical chips per host


TPU_CHIPS: dict[str, TPUChipSpec] = {
    "v4": TPUChipSpec(name="v4", hbm_gb=32.0, chips_per_host=4),
    "v5p": TPUChipSpec(name="v5p", hbm_gb=95.0, chips_per_host=4),
    "v5e": TPUChipSpec(name="v5e", hbm_gb=16.0, chips_per_host=8),
    "v6e": TPUChipSpec(name="v6e", hbm_gb=32.0, chips_per_host=8),
}


# ──────────────────────────────────────
# Quantization detection
# ──────────────────────────────────────


_QUANT_METHOD_BITS: dict[str, int] = {
    "awq": 4,
    "gptq": 4,
    "fp8": 8,
    "fbgemm_fp8": 8,
    "torchao": 4,
    "mxfp4": 4,
    "mxfp8": 8,
}


def _detect_quantization(config: dict[str, Any], model_id: str = "") -> tuple[bool, str | None, float]:
    """Detect quantization and return (is_quantized, method, bytes_per_param)."""
    quant_config = config.get("quantization_config", {})

    if quant_config:
        method = quant_config.get("quant_method", "")

        # Explicit bits in config
        bits = quant_config.get("bits")
        if bits is not None:
            return True, method or "unknown", bits / 8.0

        # bitsandbytes special cases
        if method.lower() == "bitsandbytes":
            if quant_config.get("load_in_4bit", False):
                return True, "bitsandbytes-4bit", 0.5
            if quant_config.get("load_in_8bit", False):
                return True, "bitsandbytes-8bit", 1.0

        # Known method → known bits
        bits = _QUANT_METHOD_BITS.get(method.lower())
        if bits is not None:
            return True, method, bits / 8.0

        return True, method or "unknown", 2.0

    # Check model ID for GGUF quantization patterns
    if model_id:
        model_upper = model_id.upper()
        match = re.search(r":(?:Q|UD-IQ|UD-Q)(\d+)_|UD-IQ(\d+)_", model_upper)
        if match:
            bits = int(match.group(1) or match.group(2))
            return True, "gguf", bits / 8.0

    # Default: FP16 (2 bytes per param)
    return False, None, 2.0


# ──────────────────────────────────────
# Config normalization
# ──────────────────────────────────────


def _resolve_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested text_config for multimodal models (Gemma-3, LLaVA, etc.).

    Returns a merged dict where text_config values take precedence for
    architecture fields, while preserving top-level fields like
    ``quantization_config`` and ``max_position_embeddings``.
    """
    text_cfg = config.get("text_config")
    if not text_cfg or not isinstance(text_cfg, dict):
        return config
    merged = {**config, **text_cfg}
    # Keep top-level overrides for non-architecture fields
    for key in ("quantization_config", "max_position_embeddings", "torch_dtype"):
        if key in config:
            merged[key] = config[key]
    return merged


# ──────────────────────────────────────
# Parameter estimation
# ──────────────────────────────────────


def _estimate_param_count(config: dict[str, Any]) -> int:
    """Estimate total parameter count from model architecture config."""
    # Try explicit count first
    for key in ("num_parameters", "num_params"):
        count = config.get(key)
        if count:
            return int(count)

    hidden_size = config.get("hidden_size", config.get("n_embd", 4096))
    num_layers = config.get("num_hidden_layers", 32)
    vocab_size = config.get("vocab_size", 32000)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    num_attention_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)

    embedding = vocab_size * hidden_size

    # GQA-aware attention: Q + K + V projections + output
    q_proj = hidden_size * (num_attention_heads * head_dim)
    k_proj = hidden_size * (num_kv_heads * head_dim)
    v_proj = hidden_size * (num_kv_heads * head_dim)
    o_proj = (num_attention_heads * head_dim) * hidden_size
    attention = num_layers * (q_proj + k_proj + v_proj + o_proj)

    # MLP: gate_proj + up_proj + down_proj (SwiGLU-style)
    single_expert_mlp = 3 * hidden_size * intermediate_size
    num_experts = config.get("num_local_experts", 1)
    mlp = num_layers * single_expert_mlp * num_experts

    # MoE router weights
    if num_experts > 1:
        mlp += num_layers * hidden_size * num_experts

    layernorm = num_layers * 2 * hidden_size  # RMSNorm / LayerNorm (pre-attn + pre-mlp)

    # 1.05× margin for tie weights, biases, and other small tensors
    return int((embedding + attention + mlp + layernorm) * 1.05)


def _estimate_kv_cache_per_token_gb(config: dict[str, Any]) -> float:
    """Estimate KV cache memory per token in GB."""
    num_layers = config.get("num_hidden_layers", 32)
    num_kv_heads = config.get("num_key_value_heads", config.get("num_attention_heads", 32))
    head_dim = config.get("head_dim")
    if head_dim is None:
        hidden_size = config.get("hidden_size", 4096)
        num_attention_heads = config.get("num_attention_heads", 32)
        head_dim = hidden_size // num_attention_heads

    # 2 (K+V) × layers × kv_heads × head_dim × 2 bytes (FP16)
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2
    return bytes_per_token / (1024**3)


# ──────────────────────────────────────
# Public API
# ──────────────────────────────────────


def analyze_model(
    config: dict[str, Any],
    total_vram_gb: float,
    num_gpus: int = 1,
    model_id: str = "",
) -> ModelAnalysis:
    """Analyze model resource requirements and recommend vLLM parameters.

    Args:
        config: Parsed HuggingFace ``config.json``.
        total_vram_gb: Total GPU VRAM in GB (sum across all GPUs).
        num_gpus: Number of GPUs to use.
        model_id: Optional model identifier (for GGUF quantization detection).

    Returns:
        ``ModelAnalysis`` with recommended parameters and warnings.

    """
    warnings: list[str] = []

    # Flatten multimodal configs (Gemma-3, LLaVA, etc.)
    config = _resolve_text_config(config)

    # Quantization
    is_quantized, quant_method, bytes_per_param = _detect_quantization(config, model_id)

    # Parameters
    param_count = _estimate_param_count(config)
    weights_gb = param_count * bytes_per_param / (1024**3)

    # MoE detection
    num_experts = config.get("num_local_experts", 1)
    is_moe = num_experts > 1

    # Memory budget
    per_gpu_vram = total_vram_gb / num_gpus
    hidden_size = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    activation_gb = max(0.3, hidden_size * num_layers / (1024**3) * 2)

    # ── Parallelism strategy ──
    # TP must be a power of 2, and ≤ num_gpus
    tp = 1
    if weights_gb > per_gpu_vram * 0.80:
        raw_tp = weights_gb / (per_gpu_vram * 0.70)
        # Round up to next power of 2
        tp = 1
        while tp < raw_tp:
            tp *= 2
        tp = min(tp, num_gpus)

    # EP for MoE models: use remaining GPUs after TP
    ep = 1
    if is_moe and num_gpus > tp:
        remaining = num_gpus // tp
        # EP should divide num_experts evenly
        ep = remaining
        while ep > 1 and num_experts % ep != 0:
            ep -= 1

    # DP: remaining GPUs after TP × EP
    dp = num_gpus // (tp * ep)

    per_gpu_weights = weights_gb / (tp * ep) if is_moe else weights_gb / tp

    # GPU memory utilization
    reserved_gb = max(1.0, per_gpu_vram * 0.10)
    gpu_mem_util = min(0.95, (per_gpu_vram - reserved_gb) / per_gpu_vram)
    gpu_mem_util = max(0.50, gpu_mem_util)

    # KV cache budget → max_model_len
    kv_per_token = _estimate_kv_cache_per_token_gb(config)
    available_for_kv = per_gpu_vram * gpu_mem_util - per_gpu_weights - activation_gb
    if kv_per_token > 0 and available_for_kv > 0:
        max_model_len = int(available_for_kv / kv_per_token)
    else:
        max_model_len = 512

    # Clamp to model's max position embeddings if available
    model_max_len = config.get("max_position_embeddings")
    if model_max_len:
        max_model_len = min(max_model_len, model_max_len)
    max_model_len = max(512, min(max_model_len, 131072))

    # max_num_seqs heuristic
    max_num_seqs = 8 if per_gpu_vram < 8 else 32 if per_gpu_vram < 24 else 64

    # max_num_batched_tokens — scale with VRAM
    if per_gpu_vram >= 70:
        max_num_batched_tokens = 8192
    elif per_gpu_vram >= 40:
        max_num_batched_tokens = 4096
    elif per_gpu_vram >= 24:
        max_num_batched_tokens = 2048
    else:
        max_num_batched_tokens = 1024

    # enforce_eager for quantized models on small GPUs
    enforce_eager = is_quantized and per_gpu_vram <= 16

    # Can it fit?
    can_fit = per_gpu_weights + activation_gb < per_gpu_vram * 0.95

    # Chunked prefill — good for throughput, disable for very small GPUs
    enable_chunked_prefill = per_gpu_vram >= 16

    # Prefix caching — generally beneficial
    enable_prefix_caching = True

    # Expert parallel flag — for MoE models with EP > 1
    enable_expert_parallel = is_moe and ep > 1

    # KV cache dtype — fp8 halves KV cache memory on ≥ Hopper GPUs
    kv_cache_dtype = "auto"
    if per_gpu_vram >= 40 and available_for_kv / per_gpu_vram < 0.3:
        kv_cache_dtype = "fp8"

    # Swap space — more for larger GPU servers
    swap_space = 4.0 if per_gpu_vram >= 24 else 2.0

    # Block size — 16 default, 32 for large context models
    block_size = 32 if max_model_len >= 32768 else 16

    # CPU offload — when model barely fits
    cpu_offload_gb = 0.0
    if not can_fit and per_gpu_vram >= 16:
        cpu_offload_gb = min(weights_gb - per_gpu_vram * tp * 0.85, 16.0)
        cpu_offload_gb = max(0.0, round(cpu_offload_gb, 1))

    # dtype recommendation
    if is_quantized:
        dtype = "auto"
    elif per_gpu_vram >= 40:
        dtype = "bfloat16"
    else:
        dtype = "auto"

    # Pipeline parallelism — only for very large models on many GPUs
    pp = 1

    # Can it fit?
    can_fit_with_offload = can_fit or cpu_offload_gb > 0

    if not can_fit:
        warnings.append(
            f"Model weights ({weights_gb:.1f} GB) + activations ({activation_gb:.1f} GB) "
            f"may exceed available VRAM ({total_vram_gb:.1f} GB across {num_gpus} GPU(s))"
        )
        if not is_quantized:
            warnings.append("Consider using a quantized version (AWQ, GPTQ, FP8)")
        if cpu_offload_gb > 0:
            warnings.append(f"Recommending CPU offload of {cpu_offload_gb:.1f} GB to help fit model")
        else:
            warnings.append("Consider using more GPUs for tensor parallelism")

    if enforce_eager:
        warnings.append("Using --enforce-eager to avoid torch.compile memory overhead on limited VRAM")

    if dp > 1:
        warnings.append(f"Model fits on {tp} GPU(s); using data parallelism (DP={dp}) for throughput")

    if ep > 1:
        warnings.append(f"MoE model with {num_experts} experts; using expert parallelism (EP={ep})")

    if kv_cache_dtype == "fp8":
        warnings.append("Using FP8 KV cache to maximize context length (requires Hopper+ GPU)")

    if not enable_chunked_prefill:
        warnings.append("Chunked prefill disabled due to limited VRAM")

    analysis = ModelAnalysis(
        device=DeviceType.GPU,
        model_id=model_id,
        param_count=param_count,
        bytes_per_param=bytes_per_param,
        weights_memory_gb=round(weights_gb, 2),
        is_quantized=is_quantized,
        quant_method=quant_method,
        is_moe=is_moe,
        num_experts=num_experts,
        gpu_memory_utilization=round(gpu_mem_util, 2),
        max_model_len=max_model_len,
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        expert_parallel_size=ep,
        pipeline_parallel_size=pp,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_prefix_caching=enable_prefix_caching,
        enable_expert_parallel=enable_expert_parallel,
        kv_cache_dtype=kv_cache_dtype,
        swap_space=swap_space,
        block_size=block_size,
        cpu_offload_gb=cpu_offload_gb,
        dtype=dtype,
        can_fit=can_fit_with_offload,
        warnings=warnings,
    )

    logger.info(
        "Model analysis for '{}': weights={:.1f}GB, quantized={}, tp={}, dp={}, ep={}, "
        "gpu_mem_util={}, max_model_len={}, max_num_batched_tokens={}, "
        "chunked_prefill={}, prefix_caching={}, kv_cache_dtype={}, can_fit={}",
        model_id or "unknown",
        weights_gb,
        is_quantized,
        tp,
        dp,
        ep,
        gpu_mem_util,
        max_model_len,
        max_num_batched_tokens,
        enable_chunked_prefill,
        enable_prefix_caching,
        kv_cache_dtype,
        can_fit_with_offload,
    )

    return analysis


def analyze_model_tpu(
    config: dict[str, Any],
    chip_type: str,
    num_chips: int = 0,
    model_id: str = "",
) -> ModelAnalysis:
    """Analyze model resource requirements for TPU and recommend vLLM parameters.

    Args:
        config: Parsed HuggingFace ``config.json``.
        chip_type: TPU chip type (v4, v5p, v5e, v6e).
        num_chips: Number of TPU chips. Defaults to chips_per_host for the chip type.
        model_id: Optional model identifier (for GGUF quantization detection).

    Returns:
        ``ModelAnalysis`` with recommended TPU parameters and warnings.

    Raises:
        ValueError: If *chip_type* is not recognised.

    """
    chip_type = chip_type.lower()
    if chip_type not in TPU_CHIPS:
        valid = ", ".join(sorted(TPU_CHIPS))
        raise ValueError(f"Unknown TPU chip type '{chip_type}'. Valid types: {valid}")

    chip = TPU_CHIPS[chip_type]
    if num_chips <= 0:
        num_chips = chip.chips_per_host

    warnings: list[str] = []

    # Flatten multimodal configs
    config = _resolve_text_config(config)

    # Quantization
    is_quantized, quant_method, bytes_per_param = _detect_quantization(config, model_id)

    # TPU only supports bfloat16; quantized models will still load but
    # the parameter bytes stay as detected for weight-size estimation.
    if is_quantized:
        warnings.append(
            "Quantized models have limited support on TPU; consider using a non-quantized bfloat16 checkpoint"
        )

    # Parameters
    param_count = _estimate_param_count(config)
    weights_gb = param_count * bytes_per_param / (1024**3)

    # MoE detection
    num_experts = config.get("num_local_experts", 1)
    is_moe = num_experts > 1

    # Memory budget
    per_chip_hbm = chip.hbm_gb
    total_hbm = per_chip_hbm * num_chips
    hidden_size = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    activation_gb = max(0.3, hidden_size * num_layers / (1024**3) * 2)

    # ── TP must match chips-per-host (or be a valid power-of-2 divisor) ──
    tp = 1
    if weights_gb > per_chip_hbm * 0.80:
        raw_tp = weights_gb / (per_chip_hbm * 0.70)
        tp = 1
        while tp < raw_tp:
            tp *= 2
        tp = min(tp, num_chips)

    # DP: remaining chips after TP
    dp = num_chips // tp

    per_chip_weights = weights_gb / tp

    # KV cache budget → max_model_len
    kv_per_token = _estimate_kv_cache_per_token_gb(config)
    # Reserve ~15% HBM for runtime overhead on TPU
    available_for_kv = per_chip_hbm * 0.85 - per_chip_weights - activation_gb
    if kv_per_token > 0 and available_for_kv > 0:
        max_model_len = int(available_for_kv / kv_per_token)
    else:
        max_model_len = 512

    model_max_len = config.get("max_position_embeddings")
    if model_max_len:
        max_model_len = min(max_model_len, model_max_len)
    max_model_len = max(512, min(max_model_len, 131072))

    # max_num_seqs — TPUs have high memory bandwidth, lean towards higher concurrency
    if per_chip_hbm >= 80:
        max_num_seqs = 64
    elif per_chip_hbm >= 32:
        max_num_seqs = 32
    else:
        max_num_seqs = 16

    # max_num_batched_tokens
    if per_chip_hbm >= 80:
        max_num_batched_tokens = 8192
    elif per_chip_hbm >= 32:
        max_num_batched_tokens = 4096
    else:
        max_num_batched_tokens = 2048

    # Can it fit?
    can_fit = per_chip_weights + activation_gb < per_chip_hbm * 0.85

    # Block size
    block_size = 32 if max_model_len >= 32768 else 16

    if not can_fit:
        warnings.append(
            f"Model weights ({weights_gb:.1f} GB) + activations ({activation_gb:.1f} GB) "
            f"may exceed available HBM ({total_hbm:.1f} GB across {num_chips} chip(s))"
        )
        if num_chips < chip.chips_per_host:
            warnings.append(f"Consider using all {chip.chips_per_host} chips per host for tensor parallelism")

    if dp > 1:
        warnings.append(f"Model fits on {tp} chip(s); using data parallelism (DP={dp}) for throughput")

    if is_moe:
        warnings.append("MoE expert parallelism is not yet supported on TPU by vLLM")

    analysis = ModelAnalysis(
        device=DeviceType.TPU,
        model_id=model_id,
        param_count=param_count,
        bytes_per_param=bytes_per_param,
        weights_memory_gb=round(weights_gb, 2),
        is_quantized=is_quantized,
        quant_method=quant_method,
        is_moe=is_moe,
        num_experts=num_experts,
        # TPU-specific defaults for GPU-only fields
        gpu_memory_utilization=0.0,
        enforce_eager=False,
        swap_space=0.0,
        kv_cache_dtype="auto",
        cpu_offload_gb=0.0,
        # Shared fields
        max_model_len=max_model_len,
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        expert_parallel_size=1,
        pipeline_parallel_size=1,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enable_expert_parallel=False,
        block_size=block_size,
        dtype="bfloat16",
        tpu_chip=chip_type,
        can_fit=can_fit,
        warnings=warnings,
    )

    logger.info(
        "TPU analysis for '{}': chip={}, chips={}, weights={:.1f}GB, quantized={}, tp={}, dp={}, "
        "max_model_len={}, max_num_batched_tokens={}, can_fit={}",
        model_id or "unknown",
        chip_type,
        num_chips,
        weights_gb,
        is_quantized,
        tp,
        dp,
        max_model_len,
        max_num_batched_tokens,
        can_fit,
    )

    return analysis
