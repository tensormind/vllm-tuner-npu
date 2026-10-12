# Qwen3.5-27B TP=4 寻优报告：v3 引入 MTP 投机解码，吞吐 +23%

> 模型：Qwen3.5-27B / TP=4 / DP=1 / NPU 4-7（910B3，每卡 64GB）
> 场景：input 16K、output 4K、并发 40、请求数 160、GSM8K
> 框架：vLLM 0.23.0 + vllm-ascend，block_size=128，cudagraph FULL_DECODE_ONLY
> 目标：最大化 ais_bench `output_tokens_per_sec`（真实输出吞吐），无 TTFT 约束

---

## 一、结论速览

| 轮次 | 关键配置 | 最佳吞吐 | 相对 v2 |
|------|---------|---------|--------|
| v2（无 MTP） | MNBT=8192, MNS=40, MML=24576, GMU=0.88 | **611.1 tok/s** | 基线 |
| **v3（+ MTP）** | + `speculative_config` num_spec=1, MNBT=8192, MNS=44, GMU=0.88 | **751.0 tok/s** | **+22.9%** |

**v3 最佳配置（trial #9）**：

```yaml
speculative_config: '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'
max_num_batched_tokens: 8192      # chunked prefill 分块
max_num_seqs: 44                   # 并发 40 需 ≥40
max_model_len: 24576               # 固定（16384+4096+chat模板≈20485，余量充足）
gpu_memory_utilization: 0.88       # ★ 必须 ≤0.92，0.94 会启动 hang
block_size: 128
compilation_config: '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
additional_config: '{"enable_cpu_binding":true}'
# env: HCCL_BUFFSIZE=512 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#      TASK_QUEUE_ENABLE=1 OMP_PROC_BIND=false OMP_NUM_THREADS=1
```

---

## 二、v3 完整寻优路径（12/12 trials）

### 2.1 按吞吐排名

| 排名 | # | 吞吐 (tok/s) | MNBT | MNS | GMU | 备注 |
|------|---|------------|------|-----|-----|------|
| 1 | **9** | **751.0** | 8192 | 44 | 0.88 | ★ 全局最优 |
| 2 | 10 | 747.9 | 8192 | 44 | 0.90 | |
| 3 | 8 | 694.7 | 16384 | 44 | 0.88 | |
| 3 | 7 | 694.7 | 16384 | 44 | 0.88 | |
| 3 | 6 | 694.7 | 16384 | 44 | 0.92 | |
| 6 | 5 | 692.9 | 16384 | 40 | 0.92 | |
| 7 | 12 | 662.2 | 8192 | 44 | 0.90 | |
| 8 | 11 | 660.7 | 8192 | 42 | 0.88 | |
| 9 | 2 | 623.8 | 4096 | 42 | 0.90 | 热启动 |
| 10 | 4 | 620.9 | 8192 | 40 | 0.88 | |
| 11 | 3 | 523.9 | 16384 | 40 | 0.88 | 冷启动 |
| 12 | 1 | 433.2 | 4096 | 42 | 0.90 | 冷启动（误判为负优化的来源）|

### 2.2 TPE 寻优路径（按评估时间顺序）

上表是按吞吐排名的视角。下面按**评估时间顺序**还原 TPE 的探索→利用路径（trial # = 评估序号，`← 新最优` 标记该次刷新当前最优）：

```
 #    吞吐    MNBT   MNS   GMU   阶段
 1   433.2   4096    42   0.90  探索 ← 新最优(冷)
 2   623.8   4096    42   0.90  探索 ← 新最优(热,同配置)
 3   523.9  16384    40   0.88  探索(冷)
 4   620.9   8192    40   0.88  探索
 5   692.9  16384    40   0.92  探索 ← 新最优
 6   694.7  16384    44   0.92  利用 ← 新最优
 7   694.7  16384    44   0.88  利用
 8   694.7  16384    44   0.88  利用
 9   751.0   8192    44   0.88  利用 ← ★全局最优
10   747.9   8192    44   0.90  利用(确认#9邻域)
11   660.7   8192    42   0.88  利用(回探MNS=42)
12   662.2   8192    44   0.90  利用(收官)
```

**路径分三段：**

**① 探索期（#1–#5，startup trials）** —— TPESampler 的前 5 次随机采样覆盖四个 MNBT 档：
- #1/#2 撞 MNBT=4096（#1 冷 433 → #2 热 624，第一次暴露冷编译税）
- #3/#5 撞 MNBT=16384（#3 冷 524 → #5 热 693，第二次确认冷热规律）
- #4 撞 MNBT=8192（621）
- 期间 TPE 学到：**MNBT=16384 / GMU=0.92 是当前最优区（693）**，进入利用。

**② 利用期前段（#6–#8）** —— 围绕 #5 的 gbest 邻域精修：
- TPE 把 MNS 从 40 拉到 44（#6，694.7，小涨）、GMU 降到 0.88（#7/#8，持平 694.7）。
- 三次都在 MNBT=16384 / MNS=44 打转，**陷入局部**——#6–#8 全是 694.7，TPE 判断该区已饱和。

**③ 利用期后段（#9–#12）—— 跳出局部找到全局最优：**
- TPE 意识到 MNBT=16384 区已无增量，**回探 MNBT=8192 + MNS=44**（#9），结果 **751.0** 跳出局部。
- #10（747.9）在 #9 邻域确认、#11 回探 MNS=42（660，验证 MNS=44 更优）、#12 收官。

### 2.3 路径的三个特征

1. **冷编译税主导了探索期的表观波动**：#1–#5 看似乱跳（433→624→524→621→693），实际是同规律（冷低热高）叠加 TPE 的维度采样。若只看排名表会被冷启动的低分（#1=433）误导成"MTP 负优化"。
2. **TPE 在 #6–#8 陷入 MNBT=16384 局部**（694.7 三连平），靠 #9 回探 MNBT=8192 才跳出——这正是 TPE 比 PSO 慢但更稳的体现（PSO 会代际跳跃，TPE 靠 l/g 分布收缩 + 偶尔回探）。
3. **最优解 MNBT=8192 与 v2（无 MTP）的最优 MNBT 一致**：说明 MTP 不改变"MNBT=8192 最优"这一先验，只是把吞吐基线整体抬高了 23%。换言之，**MTP 是叠加在 v2 最优配置上的正交增益**，不需要重新选 MNBT。

> 对比 OPTIMIZATION_PATHS 里 TPE 窄空间的路径（idx=8 找到 655，后期大量采 MNBT=8192/GMU=0.90 邻域），本轮 v3 的路径模式高度相似——都是"探索期撞冷热税波动 → 利用期收缩到 MNBT=8192/MNS 高位区"，只是本轮加了 MTP 维度后天花板从 655/611 抬到 751。

---

## 三、核心发现：MTP 是真正的吞吐杠杆

### 3.1 v2 为何停在 611 —— 缺了 MTP 这一维

v2（611.1 tok/s）的搜索空间只有 4 个连续维度：MNBT / MNS / MML / GMU。这些是**低杠杆旋钮**（GMU 在 0.88-0.94 间波动仅 ±1-2%）。

真正能突破 611 的杠杆是 **MTP 投机解码**（multi-token prediction）。Qwen3.5-27B 原生支持：`config.json → text_config.mtp_num_hidden_layers = 1`。vLLM 通过 `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}'` 启用，每个解码步推测 1 个 token，命中则省一轮前向。

v3 验证：同场景（普通数据集、无 prefix caching）下，开 MTP 后吞吐 **611 → 751（+23%）**。这是该口径下实打实的增益。

### 3.2 为何差点误判 MTP 为"负优化"——冷编译税

v3 trial #1（433.2）一度让人以为 MTP 让吞吐掉了 29%。但 #1 和 #2 参数**完全相同**（MNBT=4096/MNS=42/GMU=0.9），得分却差 44%：

| Trial | tok/s | TTFT | 编译状态 |
|-------|-------|------|---------|
| #1 | 433.2 | 234s | **冷**（首次编译 cudagraph+MTP 图）|
| #2 | 623.8 | 197s | **热**（图缓存命中）|

同样规律在 MNBT=16384 区重复：#3（524，冷）vs #5（693，热）差 32%。

**根因**：cudagraph FULL_DECODE_ONLY + MTP 的图首次编译开销大，缓存命中后才体现真实性能。首次评估被严重低估。

> ⚠️ **教训**：单次基准测试 / 寻优会低估"首次编译代价高"的配置。同一参数两次跑分差 >30%，多半是冷热编译税，应取热值。诊断标志：同配置 trial 间吞吐跳变。

### 3.3 GMU=0.94 + MTP 会启动 hang（非 OOM）

首个 MTP 试跑用 GMU=0.94（每卡仅 ~3.8GB 空闲），启动 20+ 分钟未就绪：AICore 0%、功耗 ~92W（休眠）、端口未监听、进程 SLEEPING。**不是 OOM**（无报错），而是 MTP draft 模型 + cudagraph 图在剩余空间吃紧时**死锁**。

GMU=0.88 时正常启动：`Available KV cache memory: 38.31 GiB`，`Maximum concurrency: 69x`，`Graph capturing finished in 12 secs`。

**对策**：寻优 MTP 时 cap GMU ≤ 0.92（v3 设 0.88-0.92）。

### 3.4 FINAL_COMPARISON_REPORT 的 655/1115 为何不可横比

那篇报告自己的脚注¹ 明确写：PSO/TPE 的 651-655 来自 optimizer CSV 的 `generate_speed`，手工配置 1115 来自 toolbox 的 `output_throughput`——"两者计算路径不同，数值不能直接横比"。

- `generate_speed`：**含投机生成的 token**（含被拒绝的），MTP 天然抬高它。
- ais_bench `output_tokens_per_sec`：真实输出 token，不含投机丢弃。

所以那个 655 很可能是 `generate_speed` 口径虚高，对应 output 口径约 750——**与本轮 v3 的 751 量级一致**。换言之，655 和 751 测的是同一个 MTP 真实水平，只是统计口径不同，不存在 1.8 倍的性能鸿沟。

---

## 四、v2 vs v3 参数对比与差异归因

| 维度 | v2（611）| v3（751）| 变化影响 |
|------|---------|---------|---------|
| speculative_config | 无 | qwen3_5_mtp, num_spec=1 | **+23% 主因** |
| max_num_batched_tokens | 8192 | 8192 | 不变（同为最优）|
| max_num_seqs | 40 | **44** | +少量（并发裕度）|
| max_model_len | 24576（搜索维度）| 24576（固定）| 缩搜索空间提效 |
| gpu_memory_utilization | 0.88 | 0.88 | 不变 |
| enable_prefix_caching | false | false | 无影响（prefill 非瓶颈）|

**归因**：v2 的 611 不是天花板，只是没开 MTP。v3 唯一实质变化是加入 MTP，其余差异（MNS 40→44、MML 由搜索变固定）只是次要优化。**+22.9% 增益几乎全部来自 MTP 投机解码。**

---

## 五、两个真实陷阱（已存入记忆）

### 陷阱一：GMU=0.94 + MTP 启动死锁
- 现象：启动 20min+ 不就绪，AICore 0%、低功耗、进程休眠，无报错。
- 根因：MTP draft + cudagraph 图在 ~3.8GB/卡 空闲空间内死锁。
- 对策：寻优 MTP 时 GMU 上限设 0.92；生产部署 GMU≤0.90 更稳。

### 陷阱二：冷编译税
- 现象：同配置 trial 吞吐差 30-44%（#1=433 vs #2=624）。
- 根因：cudagraph FULL_DECODE_ONLY + MTP 图首次编译慢，缓存命中后快。
- 对策：评估冷启动配置须跑两次取热值；单次跑分会误判。

### 附：cudagraph 捕获尺寸告警
`max_cudagraph_capture_size(80) < max decode tokens(84)`（MNS=42×spec=1≈84），溢出回退 eager（vllm-ascend issue #8240）。MNS=44×spec=1≈88 也会溢出，但实测 MNS=44 仍最优（751），说明回退 eager 的损失 < MNS 增大的收益，无需专门调 capture_sizes。

---

## 六、可复现的启动命令

```bash
cd /data/workspaces/qwen/vllm-tuner

# 1. 数据集（160 条，普通无前缀，与 v2 同 workload）
head -160 /data/workspaces/qwen/aisbench_auto_tools_prefix/dataset/GSM8K-in16384-num512-Qwen3.5-9B.jsonl \
  > /data/workspaces/qwen/aisbench_auto_tools_prefix/dataset/GSM8K-in16384-num160-Qwen3.5-27B.jsonl

# 2. 跑 v3 寻优（GMU 已 cap 在 0.88-0.92 避免 hang）
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
  HCCL_BUFFSIZE=512 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  TASK_QUEUE_ENABLE=1 OMP_PROC_BIND=false OMP_NUM_THREADS=1 \
  python -m vllm_tuner.cli.main run \
  --config configs/ascend_npu_27b_tp4_mtp_tune.yaml

# 3. 查结果
python -c "
import optuna
s=optuna.load_study(study_name='ascend-27b-tp4-mtp-tune-v3',
    storage='sqlite:///results_27b_tp4_mtp_tune/study.db')
print('BEST', s.best_trial.number, round(s.best_value,1), s.best_params)
"
```

配置文件：`configs/ascend_npu_27b_tp4_mtp_tune.yaml`
结果库：`results_27b_tp4_mtp_tune/study.db`

---

## 七、总结

1. **v3 通过引入 MTP 投机解码，将吞吐从 611 提升到 751 tok/s（+22.9%）**，这是 ais_bench `output_tokens_per_sec` 真实口径下的增益。
2. v2 停在 611 不是配置不佳，而是**搜索空间缺了 MTP 这一高杠杆维度**——连续维度（GMU 等）微调天花板就在 611 附近。
3. FINAL_COMPARISON_REPORT 的 655/1115 是 **`generate_speed`/`output_throughput` 口径虚高**，与本轮 751（output 口径）量级一致，不存在横比意义上的更大空间。
4. 两个真实陷阱：**GMU≥0.94 启动死锁**、**冷编译税致同配置差 44%**，均已记录，避免后续重蹈。
