### What this PR does / why we need it?

On Ascend NPU, using tensor parallelism (`TP>1`) together with LoRA makes
`vllm_ascend/lora/punica_npu.py::add_expand` crash during the dummy LoRA
warmup with `RuntimeError: hidden in should be smaller than hidden out`,
or hard-hang the device during decode (device freeze + HCCL hang). Both code
paths are unreachable in normal operation — they are triggered by an
input-shape mismatch between the all-gathered LoRA shrink buffer and the
Ascend `sgmv_expand` kernel contract. The net effect is that **TP>1 LoRA
serving on Ascend is completely unusable** without a local patch.

**Root cause:**

The LoRA shrink buffer for MCP/QKV-style layers is passed through
`tensor_model_parallel_all_gather` (default `dim=-1`). After the gather the
rank axis is a `[TP*rank]` concatenation — and because `lora_A` is replicated
across ranks, every chunk along that axis is **identical**.

The Ascend `sgmv_expand` kernel enforces
`TORCH_CHECK(x.size(1) <= slice_size)`. For small output slices this check is
**guaranteed to fail** because the gathered input width (`TP*rank`) exceeds
the per-rank slice width. The tightest module is the GDN `in_proj_ba`
(`MergedColumn [48,48]`): at TP=4 the per-card slice is `12 < TP*rank=32`.

When problem 1 is bypassed by narrowing the input, the narrowed slice is a
**non-contiguous view**. The `sgmv_expand` kernel binds via a raw
`data_ptr()`, so the non-contiguous view causes an illegal memory access →
device freeze + HCCL hang during decode.

**Fix:** narrow `x` to `lora_B`'s `hidden_in` and force contiguity in
`add_expand`. Narrowing is a mathematically equivalent transformation (the
extra gathered columns are redundant replicas of `lora_A`) and
`.contiguous()` guarantees a valid allocation for the kernel's raw
`data_ptr()` read.

### Does this PR introduce _any_ user-facing change?

No.

### How was this patch tested?

Tested with LoRA enabled under `TP>1` on Ascend 910B3:

```bash
vllm serve /data/qwen3.5/Qwen3.5-27B \
  --tensor-parallel-size 4 \
  --enable-lora \
  --lora-modules mylora=/data/workspaces/qwen/mylora \
  --max-lora-rank 8 \
  --enforce-eager \
  --trust-remote-code
```

- TP=4 + Qwen3.5-27B + r=8 LoRA (zero-output adapter): passes the dummy
  warmup (previously aborted with `RuntimeError: hidden in should be smaller
  than hidden out`).
- `model:"mylora"` routes through LoRA; `model:"qwen3.5"` routes through the
  base model — both coexist on the same server.
- Decode runs without device freeze or HCCL hang.

**Known limitation:** TP=8 is not workaroundable by this patch. The tightest
module (GDN `in_proj_ba`, `MergedColumn [48,48]`) at TP=8 has only 6 dims per
partition `< max-lora-rank=8`, and `--max-lora-rank` has no 4 step (accepts
only 1/8/16/...), so TP=4 is the current practical ceiling.

Refs: vLLM main commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`
