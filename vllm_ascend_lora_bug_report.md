# vLLM-Ascend LoRA `sgmv_expand` crashes / hangs at TP>1 with MergedColumn (QKV/GDN) modules

## Summary

On Ascend NPU (910B3), using tensor parallelism (`TP>1`) together with LoRA
makes `vllm_ascend/lora/punica_npu.py::add_expand` crash during the dummy LoRA
warmup, or hard-hang the device during decode. Both code paths are unreachable
in normal operation — they are triggered by an input-shape mismatch between
the all-gathered LoRA shrink buffer and the Ascend `sgmv_expand` kernel
contract. The net effect is that **TP>1 LoRA serving on Ascend is completely
unusable** without a local patch.

## Environment

| Component | Version |
|---|---|
| vLLM | 0.23.0 |
| vllm-ascend | 0.23.0 |
| CANN | 25.5.6 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post4 |
| Hardware | Ascend 910B3 ×8 |
| Model | Qwen3.5-27B (hybrid GDN architecture) |
| LoRA adapter | r=8, lora_B=0 (zero-output, smoke test) |

## Reproduction

```bash
vllm serve /data/qwen3.5/Qwen3.5-27B \
  --tensor-parallel-size 4 \
  --enable-lora \
  --lora-modules mylora=/data/workspaces/qwen/mylora \
  --max-lora-rank 8 \
  --enforce-eager \
  --trust-remote-code
```

### Observed result

- **TP=4** — During the dummy LoRA warmup phase, the engine aborts with:
  ```
  RuntimeError: hidden in should be smaller than hidden out
  ```
  If that check is bypassed, the server starts but **decode hard-hangs**: the
  NPU device freezes and HCCL collective communication hangs permanently.
- **TP=8** — Not recoverable by this patch. The tightest module is the GDN
  `in_proj_ba` (`MergedColumn [48,48]`): at TP=8 each per-rank partition is
  only 6 dims, which is `< max-lora-rank=8`, and `--max-lora-rank` has no 4
  step (accepts only 1/8/16/...), so TP=8 cannot satisfy the kernel contract.

### Expected behavior

LoRA loads and runs correctly at `TP>1`, without crashing or hanging, with
results mathematically equivalent to `TP=1`.

## Root cause

The LoRA shrink buffer for MCP/QKV-style layers is passed through
`tensor_model_parallel_all_gather` (default `dim=-1`). After the gather the
rank axis is a `[TP*rank]` concatenation — and because `lora_A` is replicated
across ranks, every chunk along that axis is **identical**.

The Ascend `sgmv_expand` kernel enforces
`TORCH_CHECK(x.size(1) <= slice_size)`. For small output slices this check is
**guaranteed to fail** because the gathered input width (`TP*rank`) exceeds
the per-rank slice width. The tightest module is the GDN `in_proj_ba`
(`MergedColumn [48,48]`): at TP=4 the per-card slice is `12 < TP*rank=32`.

This is **problem 1**: the kernel rejects a shape that is, mathematically,
correct — the extra gathered columns are redundant (identical replicas).

When problem 1 is bypassed by narrowing the input, the narrowed slice is a
**non-contiguous view**. The `sgmv_expand` kernel binds via a raw
`data_ptr()`, so the non-contiguous view causes an illegal memory access →
device freeze + HCCL hang during decode. This is **problem 2**.

## Fix

`vllm_ascend/lora/punica_npu.py::add_expand` — narrow `x` to `lora_B`'s
`hidden_in` and force contiguity:

```python
# Original
for slice_idx in range(len(lora_b_stacked)):
    self._apply_expand(
        y,
        x[slice_idx],
        lora_b_stacked[slice_idx],
        offset_left,
        output_slices[slice_idx],
        add_inputs=add_inputs,
    )
    offset_left += output_slices[slice_idx]

# Patched
for slice_idx in range(len(lora_b_stacked)):
    x_i = x[slice_idx]
    w_i = lora_b_stacked[slice_idx]
    if x_i.shape[-1] != w_i.shape[-1]:
        x_i = x_i[..., : w_i.shape[-1]].contiguous()
    self._apply_expand(
        y, x_i, w_i,
        offset_left, output_slices[slice_idx],
        add_inputs=add_inputs,
    )
    offset_left += output_slices[slice_idx]
```

### Why this works

- **Problem 1**: since every chunk is an identical replica of `lora_A`,
  narrowing `x` to `lora_B.hidden_in` is a mathematically equivalent
  transformation that satisfies `x.size(1) <= slice_size`.
- **Problem 2**: `.contiguous()` guarantees a contiguous allocation, so the
  raw `data_ptr()` read inside the kernel is valid.

### Verification

- TP=4 + Qwen3.5-27B + r=8 LoRA (zero-output): passes the dummy warmup.
- Request `model:"mylora"` routes through LoRA; `model:"qwen3.5"` routes
  through the base model — both coexist on the same server.
- Decode runs without device freeze or HCCL hang.

## Stack trace / log

```
RuntimeError: hidden in should be smaller than hidden out
  ... (sgmv_expand TORCH_CHECK, GDN in_proj_ba MergedColumn[48,48], TP=4)
# Aborts immediately during the dummy LoRA warmup phase at startup.
```

## Additional context

- **Trigger conditions**: `tensor_parallel_size > 1` + `--enable-lora` + a
  model containing `MergedColumn` (QKV) / GDN modules.
- **TP=8 is not workaroundable** by this patch: GDN `in_proj_ba` at TP=8 has
  only 6 dims per partition `< max-lora-rank=8`, and `--max-lora-rank` has no
  4 step, so TP=4 is the current practical ceiling.
- **Affected scope**: every Ascend-NPU deployment running vLLM 0.23.0 +
  vllm-ascend 0.23.0 with LoRA enabled at `TP>1`.
- The local patch is applied to
  `/vllm-workspace/vllm-ascend/vllm_ascend/lora/punica_npu.py` (`add_expand`,
  ~lines 245–256) and verified working.
- Suggested upstream report area: **TP>1 + MergedColumn/QKV LoRA modules** on
  the Ascend backend.


  ### What this PR does / why we need it?

#### Problem Description
When running vLLM on Ascend NPU with Tensor Parallelism (`TP > 1`) and `--enable-lora` enabled, models containing `MergedColumn` / `QKV` / `GDN` layers (such as `Qwen3.5-27B`) encounter two severe issues:
1. **Crash during Dummy Warmup**: The engine aborts with `RuntimeError: hidden in should be smaller than hidden out` in `vllm_ascend/lora/punica_npu.py::add_expand`.
2. **Device Freeze / Hard-hang during Decode**: If the assertion check is bypassed, the NPU device hard-hangs during decode, causing HCCL collective communication to lock up permanently.

#### Root Cause Analysis
1. **Shape Mismatch (Problem 1)**: The LoRA shrink buffer for `MergedColumn`/`QKV`-style layers passes through `tensor_model_parallel_all_gather` (dim=-1), resulting in a gathered input shape of `[TP * rank]`. Since `lora_A` is replicated across ranks, these chunks are identical replicas. The Ascend `sgmv_expand` kernel enforces `TORCH_CHECK(x.size(1) <= slice_size)`. For modules with small per-rank output slices (e.g., GDN `in_proj_ba` at TP=4 has a per-card slice width of 12 < `TP * rank = 32`), this assertion fails.
2. **Illegal Memory Access (Problem 2)**: Narrowing `x` without ensuring memory contiguity produces a non-contiguous Tensor view. Because the `sgmv_expand` kernel reads raw pointers via `data_ptr()`, non-contiguous views trigger illegal memory access, leading to device freezes and HCCL hangs.

#### Proposed Solution
In `vllm_ascend/lora/punica_npu.py::add_expand`:
- Narrow `x_i` to match `lora_B`'s `hidden_in` (`w_i.shape[-1]`) when shapes mismatch. Because all-gathered chunks are identical replicas of `lora_A`, this transformation is mathematically equivalent while satisfying `x.size(1) <= slice_size`.
- Enforce `.contiguous()` on `x_i` after narrowing to guarantee a contiguous memory allocation for raw `data_ptr()` access inside the kernel.

---

### Does this PR introduce _any_ user-facing change?

No. This PR only fixes internal kernel input shape alignment and memory contiguity for Ascend NPU LoRA execution, without modifying any public APIs or user-facing behavior.

---

### How was this patch tested?

#### Test Environment
- **Hardware**: Ascend 910B3 × 8
- **Software**: CANN 25.5.6, torch 2.1.0+cpu, torch_npu 2.1.0.post4, vLLM 0.23.0, vllm-ascend 0.23.0
- **Model**: Qwen3.5-27B
- **LoRA**: `r=8`, `lora_B=0`

#### Steps & Verification
1. Executed vLLM server with TP=4 and LoRA enabled:
   ```bash
   vllm serve /data/qwen3.5/Qwen3.5-27B \
     --tensor-parallel-size 4 \
     --enable-lora \
     --lora-modules mylora=/data/workspaces/qwen/mylora \
     --max-lora-rank 8 \
     --enforce-eager \
     --trust-remote-code

### 2. 中文版 (Chinese Version)

### What this PR does / why we need it?

#### 问题背景
在昇腾 NPU 上使用 vLLM 开启张量并行（`TP > 1`）并加载带有 `MergedColumn` / `QKV` / `GDN` 结构的模型（如 `Qwen3.5-27B`）运行 LoRA 时，服务会陷入不可用状态：
1. **启动阶段崩溃**：在 Dummy LoRA Warmup 阶段，`vllm_ascend/lora/punica_npu.py::add_expand` 会直接报 `RuntimeError: hidden in should be smaller than hidden out` 错误导致中断。
2. **Decode 阶段设备挂死**：即便绕过该校验，在推理 Decode 阶段会导致 NPU 芯片硬件级挂死（Hard-hang），HCCL 集合通信永久死锁。

#### 根因分析
1. **维度不匹配（问题 1）**：对于 `MergedColumn`/`QKV` 类型的图层，LoRA shrink buffer 在经过 `tensor_model_parallel_all_gather`（dim=-1）后，Gather 后的 Input 维度变为 `[TP * rank]`。由于 `lora_A` 在各个 Rank 间是完全复制（Replicated）的，这些分段存在冗余副本。而昇腾 `sgmv_expand` 内核强制校验 `TORCH_CHECK(x.size(1) <= slice_size)`。对于单卡切片较小的模块（例如 GDN `in_proj_ba` 在 TP=4 时单卡 Slice 宽度仅 12 < `TP * rank = 32`），直接触发断言崩溃。
2. **非连续内存非法访问（问题 2）**：若直接切片截取 `x` 而不处理内存连续性，得到的 Tensor 视图是非连续的。由于 `sgmv_expand` 算子底层通过 `data_ptr()` 直接读取物理内存地址，读取非连续内存会导致非法内存访问，最终引发 NPU 硬件锁死及 HCCL 死锁。

#### 修复方案
在 `vllm_ascend/lora/punica_npu.py` 的 `add_expand` 函数中：
- 当 `x_i` 与 `w_i`（`lora_B`）的最后一个维度不一致时，将其截断至 `w_i.shape[-1]`。由于 Gather 后的分段均为完全相同的副本，该截断在数学上完全等价，且满足 `x.size(1) <= slice_size` 约束。
- 对截断后的 Tensor 强制执行 `.contiguous()`，确保底层物理内存连续，保障内核 `data_ptr()` 指针访问的安全。

---

### Does this PR introduce _any_ user-facing change?

否。本 PR 仅涉及 vLLM Ascend 后端底层算子输入格式的整理与内存连续化，不影响任何对外 API、配置参数或接口行为。

---

### How was this patch tested?

#### 测试环境
- **硬件环境**：Ascend 910B3 × 8
- **软件环境**：CANN 25.5.6, torch 2.1.0+cpu, torch_npu 2.1.0.post4, vLLM 0.23.0, vllm-ascend 0.23.0
- **测试模型**：Qwen3.5-27B
- **LoRA 配置**：`r=8`, `lora_B=0`

#### 复现与验证步骤
1. 使用 TP=4 启动带 LoRA 的 vLLM 服务：
   ```bash
   vllm serve /data/qwen3.5/Qwen3.5-27B \
     --tensor-parallel-size 4 \
     --enable-lora \
     --lora-modules mylora=/data/workspaces/qwen/mylora \
     --max-lora-rank 8 \
     --enforce-eager \
     --trust-remote-code