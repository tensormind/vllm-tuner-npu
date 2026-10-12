# vllm-ascend 0.23.0 LoRA 修复补丁 (本地应用)
# 文件: /vllm-workspace/vllm-ascend/vllm_ascend/lora/punica_npu.py  (add_expand)
# 备份: 原始文件可从 vllm-ascend 0.23.0 源码还原
#
# 问题 1: MCP/QKV LoRA 层的 shrink 缓冲经 tensor_model_parallel_all_gather
#   (默认 dim=-1) 后, rank 轴为 [TP*rank] 拼接 (lora_A 各卡复制, 各 chunk 相同)。
#   Ascend sgmv_expand 内核 TORCH_CHECK(x.size(1) <= slice_size) 对小输出切片
#   必然失败 (GDN in_proj_ba: MergedColumn [48,48], TP=4 时 slice=12 < TP*rank=32):
#   RuntimeError: hidden in should be smaller than hidden out
#   -> 启动即崩 (dummy LoRA 预热阶段)。
#   修复: 按 lora_B 的 hidden_in 收窄 x (各 chunk 相同, 数学等价)。
#
# 问题 2: 收窄切片为非连续视图, 内核绑定用裸 data_ptr() 读取导致非法内存访问,
#   decode 阶段设备僵死、HCCL 集合通信永久挂起。
#   修复: .contiguous()
#
# 原始代码:
#         for slice_idx in range(len(lora_b_stacked)):
#             self._apply_expand(
#                 y,
#                 x[slice_idx],
#                 lora_b_stacked[slice_idx],
#                 offset_left,
#                 output_slices[slice_idx],
#                 add_inputs=add_inputs,
#             )
#             offset_left += output_slices[slice_idx]
#
# 修复后:
#         for slice_idx in range(len(lora_b_stacked)):
#             x_i = x[slice_idx]
#             w_i = lora_b_stacked[slice_idx]
#             if x_i.shape[-1] != w_i.shape[-1]:
#                 x_i = x_i[..., : w_i.shape[-1]].contiguous()
#             self._apply_expand(
#                 y, x_i, w_i,
#                 offset_left, output_slices[slice_idx],
#                 add_inputs=add_inputs,
#             )
#             offset_left += output_slices[slice_idx]
#
# 建议向 vllm-ascend 上游报告 (TP>1 + MergedColumn/QKV LoRA 模块)。
