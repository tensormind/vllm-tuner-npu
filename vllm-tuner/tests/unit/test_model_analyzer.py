from __future__ import annotations

import pytest

from vllm_tuner.utils.model_analyzer import (
    TPU_CHIPS,
    DeviceType,
    ModelAnalysis,
    _detect_quantization,
    _estimate_kv_cache_per_token_gb,
    _estimate_param_count,
    _resolve_text_config,
    analyze_model,
    analyze_model_tpu,
)


# ──────────────────────────────────────
# Quantization detection
# ──────────────────────────────────────


class TestDetectQuantization:
    def test_no_quantization(self):
        is_q, method, bpp = _detect_quantization({})
        assert is_q is False
        assert method is None
        assert bpp == 2.0

    def test_awq_4bit(self):
        config = {"quantization_config": {"quant_method": "awq", "bits": 4}}
        is_q, method, bpp = _detect_quantization(config)
        assert is_q is True
        assert method == "awq"
        assert bpp == 0.5

    def test_gptq_without_explicit_bits(self):
        config = {"quantization_config": {"quant_method": "gptq"}}
        is_q, method, bpp = _detect_quantization(config)
        assert is_q is True
        assert method == "gptq"
        assert bpp == 0.5  # 4 bits / 8

    def test_fp8_method(self):
        config = {"quantization_config": {"quant_method": "fp8"}}
        is_q, method, bpp = _detect_quantization(config)
        assert is_q is True
        assert method == "fp8"
        assert bpp == 1.0

    def test_bitsandbytes_4bit(self):
        config = {"quantization_config": {"quant_method": "bitsandbytes", "load_in_4bit": True}}
        is_q, method, bpp = _detect_quantization(config)
        assert is_q is True
        assert method == "bitsandbytes-4bit"
        assert bpp == 0.5

    def test_bitsandbytes_8bit(self):
        config = {"quantization_config": {"quant_method": "bitsandbytes", "load_in_8bit": True}}
        is_q, method, bpp = _detect_quantization(config)
        assert is_q is True
        assert method == "bitsandbytes-8bit"
        assert bpp == 1.0

    def test_unknown_method_with_bits(self):
        config = {"quantization_config": {"quant_method": "newmethod", "bits": 3}}
        is_q, method, bpp = _detect_quantization(config)
        assert is_q is True
        assert bpp == pytest.approx(3 / 8)

    def test_gguf_detection_from_model_id(self):
        is_q, method, bpp = _detect_quantization({}, model_id="org/Model-GGUF:Q4_0")
        assert is_q is True
        assert method == "gguf"
        assert bpp == 0.5

    def test_gguf_q8_from_model_id(self):
        is_q, method, bpp = _detect_quantization({}, model_id="org/Model-GGUF:Q8_0")
        assert is_q is True
        assert bpp == 1.0


# ──────────────────────────────────────
# Parameter estimation
# ──────────────────────────────────────


class TestEstimateParamCount:
    def test_explicit_num_parameters(self):
        config = {"num_parameters": 7_000_000_000}
        assert _estimate_param_count(config) == 7_000_000_000

    def test_architecture_based_estimation(self):
        config = {
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "intermediate_size": 14336,
        }
        count = _estimate_param_count(config)
        # Llama 7B-ish: should be in reasonable range (6-10B)
        assert 5e9 < count < 12e9

    def test_small_model(self):
        config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "vocab_size": 30522,
            "intermediate_size": 3072,
        }
        count = _estimate_param_count(config)
        # BERT-base-ish: ~110M
        assert 50e6 < count < 200e6

    def test_moe_model_counts_experts(self):
        config = {
            "hidden_size": 2880,
            "num_hidden_layers": 36,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "intermediate_size": 2880,
            "num_local_experts": 128,
        }
        count = _estimate_param_count(config)
        # GPT-oss-120B-style MoE: should be in 100B+ range
        assert count > 80e9


class TestResolveTextConfig:
    def test_plain_config_unchanged(self):
        config = {"hidden_size": 4096, "num_hidden_layers": 32}
        assert _resolve_text_config(config) == config

    def test_multimodal_flattens_text_config(self):
        config = {
            "model_type": "gemma3",
            "text_config": {
                "hidden_size": 5376,
                "num_hidden_layers": 62,
                "num_attention_heads": 32,
            },
        }
        resolved = _resolve_text_config(config)
        assert resolved["hidden_size"] == 5376
        assert resolved["num_hidden_layers"] == 62

    def test_top_level_quant_config_preserved(self):
        config = {
            "quantization_config": {"quant_method": "awq"},
            "text_config": {"hidden_size": 4096},
        }
        resolved = _resolve_text_config(config)
        assert resolved["quantization_config"] == {"quant_method": "awq"}
        assert resolved["hidden_size"] == 4096


# ──────────────────────────────────────
# KV cache estimation
# ──────────────────────────────────────


class TestEstimateKVCache:
    def test_returns_positive_value(self):
        config = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
        }
        kv_per_token = _estimate_kv_cache_per_token_gb(config)
        assert kv_per_token > 0

    def test_gqa_smaller_than_mha(self):
        base = {"num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096}
        mha = _estimate_kv_cache_per_token_gb({**base, "num_key_value_heads": 32})
        gqa = _estimate_kv_cache_per_token_gb({**base, "num_key_value_heads": 8})
        assert gqa < mha


# ──────────────────────────────────────
# Full analysis
# ──────────────────────────────────────


LLAMA_8B_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
}


class TestAnalyzeModel:
    def test_fits_on_80gb_gpu(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert isinstance(result, ModelAnalysis)
        assert result.can_fit is True
        assert result.tensor_parallel_size == 1
        assert result.gpu_memory_utilization > 0.5
        assert result.max_model_len >= 512
        assert result.max_model_len <= 8192

    def test_tight_vram_triggers_warnings(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=8.0, num_gpus=1)
        assert len(result.warnings) > 0

    def test_multi_gpu_increases_tp(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=32.0, num_gpus=4)
        # With 4x 8GB GPUs, should need TP > 1 for 8B model
        assert result.tensor_parallel_size >= 1

    def test_quantized_model_smaller_weights(self):
        quant_config = {**LLAMA_8B_CONFIG, "quantization_config": {"quant_method": "awq", "bits": 4}}
        fp16_result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=24.0)
        awq_result = analyze_model(quant_config, total_vram_gb=24.0)
        assert awq_result.weights_memory_gb < fp16_result.weights_memory_gb
        assert awq_result.is_quantized is True

    def test_small_gpu_enforce_eager(self):
        quant_config = {**LLAMA_8B_CONFIG, "quantization_config": {"quant_method": "awq", "bits": 4}}
        result = analyze_model(quant_config, total_vram_gb=16.0, num_gpus=1)
        assert result.enforce_eager is True

    def test_large_gpu_no_enforce_eager(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert result.enforce_eager is False

    def test_max_model_len_capped_by_config(self):
        config = {**LLAMA_8B_CONFIG, "max_position_embeddings": 2048}
        result = analyze_model(config, total_vram_gb=80.0, num_gpus=1)
        assert result.max_model_len <= 2048

    def test_model_analysis_model_id_stored(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, model_id="meta-llama/Llama-3-8B")
        assert result.model_id == "meta-llama/Llama-3-8B"

    def test_max_num_seqs_scales_with_vram(self):
        small = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=6.0, num_gpus=1)
        large = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert small.max_num_seqs <= large.max_num_seqs

    def test_tp_is_power_of_two(self):
        # 70B model on 8×24GB GPUs — TP should be power of 2
        config_70b = {
            **LLAMA_8B_CONFIG,
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 28672,
            "vocab_size": 128256,
        }
        result = analyze_model(config_70b, total_vram_gb=192.0, num_gpus=8)
        tp = result.tensor_parallel_size
        assert tp & (tp - 1) == 0, f"TP={tp} is not a power of 2"

    def test_extra_gpus_become_dp(self):
        # Small model on many GPUs → TP=1, DP > 1
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=640.0, num_gpus=8)
        assert result.tensor_parallel_size == 1
        assert result.data_parallel_size == 8

    def test_moe_model_gets_ep(self):
        moe_config = {
            "hidden_size": 2880,
            "num_hidden_layers": 36,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "intermediate_size": 2880,
            "num_local_experts": 128,
            "max_position_embeddings": 131072,
            "quantization_config": {"quant_method": "mxfp4"},
        }
        result = analyze_model(moe_config, total_vram_gb=640.0, num_gpus=8)
        assert result.is_moe is True
        assert result.expert_parallel_size > 1
        assert result.tensor_parallel_size * result.data_parallel_size * result.expert_parallel_size <= 8


# ──────────────────────────────────────
# New vLLM parameters
# ──────────────────────────────────────


class TestNewVllmParameters:
    def test_max_num_batched_tokens_scales_with_vram(self):
        small = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=16.0, num_gpus=1)
        large = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert small.max_num_batched_tokens <= large.max_num_batched_tokens

    def test_chunked_prefill_enabled_on_large_gpu(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert result.enable_chunked_prefill is True

    def test_chunked_prefill_disabled_on_small_gpu(self):
        config = {**LLAMA_8B_CONFIG, "quantization_config": {"quant_method": "awq", "bits": 4}}
        result = analyze_model(config, total_vram_gb=12.0, num_gpus=1)
        assert result.enable_chunked_prefill is False

    def test_prefix_caching_always_enabled(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=24.0, num_gpus=1)
        assert result.enable_prefix_caching is True

    def test_kv_cache_dtype_auto_by_default(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=24.0, num_gpus=1)
        assert result.kv_cache_dtype == "auto"

    def test_kv_cache_dtype_fp8_when_kv_tight(self):
        # Large model on 80GB GPU → tight KV budget → fp8
        config_70b = {
            **LLAMA_8B_CONFIG,
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "intermediate_size": 28672,
        }
        result = analyze_model(config_70b, total_vram_gb=80.0, num_gpus=1)
        # On 80GB GPU with 70B FP16 model, KV budget is tight
        assert result.kv_cache_dtype in ("auto", "fp8")

    def test_block_size_16_for_short_context(self):
        config = {**LLAMA_8B_CONFIG, "max_position_embeddings": 4096}
        result = analyze_model(config, total_vram_gb=80.0, num_gpus=1)
        assert result.block_size == 16

    def test_block_size_32_for_long_context(self):
        config = {**LLAMA_8B_CONFIG, "max_position_embeddings": 131072}
        result = analyze_model(config, total_vram_gb=80.0, num_gpus=1)
        assert result.block_size == 32

    def test_swap_space_larger_on_big_gpu(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert result.swap_space >= 4.0

    def test_swap_space_smaller_on_small_gpu(self):
        config = {**LLAMA_8B_CONFIG, "quantization_config": {"quant_method": "awq", "bits": 4}}
        result = analyze_model(config, total_vram_gb=12.0, num_gpus=1)
        assert result.swap_space <= 4.0

    def test_cpu_offload_zero_when_fits(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert result.cpu_offload_gb == 0.0

    def test_dtype_bfloat16_on_large_gpu(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert result.dtype == "bfloat16"

    def test_dtype_auto_on_small_gpu(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=24.0, num_gpus=1)
        assert result.dtype == "auto"

    def test_dtype_auto_for_quantized(self):
        config = {**LLAMA_8B_CONFIG, "quantization_config": {"quant_method": "awq", "bits": 4}}
        result = analyze_model(config, total_vram_gb=80.0, num_gpus=1)
        assert result.dtype == "auto"

    def test_enable_expert_parallel_for_moe(self):
        moe_config = {
            "hidden_size": 2880,
            "num_hidden_layers": 36,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "intermediate_size": 2880,
            "num_local_experts": 128,
            "max_position_embeddings": 131072,
            "quantization_config": {"quant_method": "mxfp4"},
        }
        result = analyze_model(moe_config, total_vram_gb=640.0, num_gpus=8)
        assert result.enable_expert_parallel is True

    def test_enable_expert_parallel_false_for_dense(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert result.enable_expert_parallel is False

    def test_pipeline_parallel_default_1(self):
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert result.pipeline_parallel_size == 1

    def test_max_num_batched_tokens_values(self):
        # Verify specific VRAM tiers
        r_small = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=16.0, num_gpus=1)
        r_24gb = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=24.0, num_gpus=1)
        r_48gb = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=48.0, num_gpus=1)
        r_80gb = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert r_small.max_num_batched_tokens == 1024
        assert r_24gb.max_num_batched_tokens == 2048
        assert r_48gb.max_num_batched_tokens == 4096
        assert r_80gb.max_num_batched_tokens == 8192


# ──────────────────────────────────────
# TPU analysis
# ──────────────────────────────────────


class TestTPUChipSpecs:
    def test_all_chips_have_positive_hbm(self):
        for name, chip in TPU_CHIPS.items():
            assert chip.hbm_gb > 0, f"{name} HBM should be positive"
            assert chip.chips_per_host > 0, f"{name} chips_per_host should be positive"

    def test_known_chip_types(self):
        assert set(TPU_CHIPS) == {"v4", "v5p", "v5e", "v6e"}


class TestAnalyzeModelTPU:
    def test_basic_v6e(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v6e")
        assert isinstance(result, ModelAnalysis)
        assert result.device == DeviceType.TPU
        assert result.tpu_chip == "v6e"
        assert result.dtype == "bfloat16"
        assert result.can_fit is True

    def test_device_field_set(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v5p")
        assert result.device == DeviceType.TPU

    def test_gpu_only_fields_zeroed(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v6e")
        assert result.gpu_memory_utilization == 0.0
        assert result.enforce_eager is False
        assert result.swap_space == 0.0
        assert result.cpu_offload_gb == 0.0

    def test_dtype_always_bfloat16(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v4")
        assert result.dtype == "bfloat16"

    def test_default_num_chips_from_spec(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v6e")
        # v6e has 8 chips per host; model fits on 1, so TP=1, DP=8
        assert result.tensor_parallel_size * result.data_parallel_size == TPU_CHIPS["v6e"].chips_per_host

    def test_custom_num_chips(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v6e", num_chips=4)
        assert result.tensor_parallel_size * result.data_parallel_size == 4

    def test_v5e_small_hbm_8b_model(self):
        # v5e has only 16GB HBM — 8B FP16 model (~16GB) is very tight
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v5e")
        assert result.tensor_parallel_size >= 1

    def test_v5p_large_hbm_8b_fits(self):
        # v5p has 95GB HBM — 8B model fits easily
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v5p")
        assert result.can_fit is True
        assert result.tensor_parallel_size == 1

    def test_tp_power_of_two(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v5e", num_chips=8)
        tp = result.tensor_parallel_size
        assert tp & (tp - 1) == 0, f"TP={tp} is not a power of 2"

    def test_max_model_len_capped(self):
        config = {**LLAMA_8B_CONFIG, "max_position_embeddings": 2048}
        result = analyze_model_tpu(config, chip_type="v5p")
        assert result.max_model_len <= 2048

    def test_model_id_stored(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v6e", model_id="meta-llama/Llama-3-8B")
        assert result.model_id == "meta-llama/Llama-3-8B"

    def test_quantized_model_warns(self):
        config = {**LLAMA_8B_CONFIG, "quantization_config": {"quant_method": "awq", "bits": 4}}
        result = analyze_model_tpu(config, chip_type="v6e")
        assert result.is_quantized is True
        assert any("quantized" in w.lower() for w in result.warnings)

    def test_moe_model_warns(self):
        moe_config = {
            "hidden_size": 2880,
            "num_hidden_layers": 36,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "vocab_size": 201088,
            "intermediate_size": 2880,
            "num_local_experts": 128,
            "max_position_embeddings": 131072,
        }
        result = analyze_model_tpu(moe_config, chip_type="v5p", num_chips=4)
        assert result.is_moe is True
        assert any("moe" in w.lower() for w in result.warnings)

    def test_invalid_chip_type_raises(self):
        with pytest.raises(ValueError, match="Unknown TPU chip type"):
            analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v99")

    def test_case_insensitive_chip_type(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="V6E")
        assert result.tpu_chip == "v6e"

    def test_block_size_32_for_long_context(self):
        config = {**LLAMA_8B_CONFIG, "max_position_embeddings": 131072}
        result = analyze_model_tpu(config, chip_type="v5p")
        assert result.block_size == 32

    def test_kv_cache_dtype_always_auto(self):
        result = analyze_model_tpu(LLAMA_8B_CONFIG, chip_type="v5p")
        assert result.kv_cache_dtype == "auto"

    def test_gpu_analysis_still_works(self):
        """Ensure GPU analysis is unaffected by TPU additions."""
        result = analyze_model(LLAMA_8B_CONFIG, total_vram_gb=80.0, num_gpus=1)
        assert result.device == DeviceType.GPU
        assert result.tpu_chip is None
