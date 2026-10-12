Hi @paulyu12 @vllm-project/ascend, gentle ping for review.

This PR fixes a hard crash/hang that makes LoRA completely unusable at `TP>1`
on Ascend (`RuntimeError: hidden in should be smaller than hidden out` during
warmup, or device freeze + HCCL hang during decode). The fix is a 18-line
change in `add_expand` that narrows the all-gathered shrink buffer to
`lora_B`'s `hidden_in` and forces contiguity — mathematically equivalent
since the extra gathered columns are identical replicas of `lora_A`.

- DCO sign-off has been added to the commit.
- This PR touches source code (`vllm_ascend/lora/punica_npu.py`), so
  precision/full tests are required before merge. Could a maintainer add the
  `ready-precise` label to trigger the selected-test suite? Happy to add a
  regression test if that's preferred.
- Pre-commit and cpu-ut are green; the change was verified end-to-end with
  Qwen3.5-27B + r=8 LoRA at TP=4 on 910B3.

Thanks!
