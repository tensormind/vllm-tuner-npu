==============================================================================
>>> 助手回答 (transcript 行 2149)
==============================================================================
因为这次的配置 `n_trials: 1`（只做一次验证），寻优路径只有 1 个点。下面分三层介绍怎么查寻优路径——从简单到深入。

## 1. vllm-tuner 自带命令（最简单）

```bash
# 表格汇总：每个 trial 的状态/吞吐/P95/耗时，best 高亮
python -m vllm_tuner.cli.main report \
  --study-name ascend-9b-aisbench \
  --storage sqlite:///results_ascend_aisbench/study.db \
  --format json   # 或 html 生成可视化报告
```

当前验证的输出就是 1 行：trial #0，COMPLETED，170.89 tok/s。

## 2. 直接查 Optuna study.db（看完整寻优细节）

寻优数据全在 [study.db](results_ascend_aisbench/study.db)，是 Optuna 的标准结构。刚查出来的结果：

```
trial #0  state=COMPLETE  params={'gpu_memory_utilization': 0.85}  objective=[170.8859]
    start=2026-09-15 04:54:21  complete=2026-09-15 05:13:14  (耗时 ~19min)
best trial #: 0
```

想看每个 trial 的中间采样值（比如逐步收敛曲线），查 `trial_intermediate_values` 表。更省事的是用 Optuna 自带 API：

```bash
python -c "
import optuna
s = optuna.load_study(study_name='ascend-9b-aisbench',
                      storage='sqlite:///results_ascend_aisbench/study.db')
for t in s.trials:
    print(f'#{t.number} {t.state} params={t.params} value={t.value} dur={t.datetime_complete-t.datetime_start}')
print('best:', s.best_trial.number, s.best_value, s.best_params)
"
```

## 3. 可视化寻优路径（多 trial 时最有用）

正式调参（`n_trials` 设大）后，Optuna 能画出参数→目标值的关系图，直接看出哪个参数区间好：

```bash
python -c "
import optuna
s = optuna.load_study(study_name='ascend-9b-aisbench',
                      storage='sqlite:///results_ascend_aisbench/study.db')
# 优化历史：每个 trial 的 objective 值 + best 累积线
fig = optuna.visualization.plot_optimization_history(s)
fig.write_html('optimization_history.html')
# 参数重要性（fANOVA）
fig2 = optuna.visualization.plot_param_importances(s)
fig2.write_html('param_importances.html')
# 单参数取值 vs 目标值（找好区间）
fig3 = optuna.visualization.plot_slice(s)
fig3.write_html('slice.html')
print('写好 3 个 html，浏览器打开')
"
```

- `optimization_history.html`：横轴 trial 编号、纵轴 objective，直接看寻优怎么爬升/收敛——**这就是"寻优路径"最直观的图**。
- `param_importances.html`：哪个参数对吞吐影响最大。
- `slice.html`：每个参数的采样分布 + 对应值，找甜区。

> 注意：单参数 / 单 trial 时 fANOVA 和 slice 画不出来（统计意义不够），需要 ≥5 个 trial 才有意义。

---

当前这次是**验证性单 trial**，所以路径只有一个点。要不要我把 `n_trials` 调大、加更多参数空间（比如 `gpu_memory_utilization` 0.80–0.92 + `max_num_seqs` 8–64 + `enable_chunked_prefill` 开关），跑一次真正的多 trial 寻优，这样你能看到完整的收敛曲线和参数重要性？

==============================================================================
>>> 助手回答 (transcript 行 2789)
==============================================================================
## DP=6 寻优 — 最终报告

### 结论：TTFT < 10s 约束可达，最优可行配置已找到

完整寻优路径（8 trials 目标，实际跑出结果如下）：

| Trial | 状态 | output tok/s | 可行 | max_num_seqs | max_model_len | gpu_mem | TTFT |
|---|---|---|---|---|---|---|---|
| #0-2 | COMPLETE | 0 | 否 | — | 24576 | 0.86-0.9 | OOM（前会话孤儿 worker 占卡） |
| #3 | RUNNING | — | 否 | 64 | 24576 | 0.92 | 僵尸（被杀的 run #1） |
| **#4** | **COMPLETE** | **469.9** | **✅** | **64** | **32768** | **0.80** | **4.8s** |
| #5 | COMPLETE | 99999(垃圾) | 否 | 64 | 32768 | 0.82 | degraded 服务器→手杀 |
| #6 | RUNNING | — | 否 | 32 | 32768 | 0.80 | crash/restart 循环 |

### 推荐配置（最优可行 trial #4）
```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 vllm serve /data/qwen3.5/Qwen3.5-9B \
  --tensor-parallel-size 1 --data-parallel-size 6 \
  --max-num-seqs 64 --max-model-len 32768 --gpu-memory-utilization 0.80 \
  --enforce-eager --enable-chunked-prefill --no-enable-prefix-caching \
  --dtype auto --trust-remote-code --served-model-name qwen3.5
```
- **output 吞吐 469.9 tok/s**（TP1 baseline 170.9 → DP6 ≈ 2.75×，`enforce_eager` 限制了单副本 decode 速度，未达理论 6×）
- **TTFT avg 4.8s**，距 10s 约束还有 5.2s 余量

### 过程中解决的关键问题
1. **所有 trial 一开始全 OOM/ERR99999** → 根因是前一会话遗留的孤儿 EngineCore worker 各占 57GB（NPU 0-3），`pgrep` 看不见，靠 `npu-smi info -t proc-mem` 找到 PID 清场后 DP=6 正常启动。
2. **constraint 机制全程验证有效**：对真 OOM（margin −1）、被手杀的 garbage（objective 99999 但 margin −89999）都正确标记 infeasible，TPE 搜索未被污染。这正是适配时修复的 metric-aware 约束的价值——否则 `ttft_ms<=10000` 会被静默忽略。

### 未能跑满 8 trials 的原因（非 vllm-tuner 问题）
vllm-ascend 0.23.0 的 DP=6 稳定性差：某些参数组合启动后只加载权重（21GB）不分配 KV cache，ais_bench 卡在 0/512 直到 35min 超时；部分组合进入 EngineCore crash/restart 循环。这是 vLLM-ascend 的 bug，不是 tuner 的问题。已存档为 [vllm-ascend-dp6-stability](memory/vllm-ascend-dp6-stability.md)。

### 如需继续探索更大吞吐
由于 TTFT 4.8s 还有 5.2s 余量，可放宽让 TPE 用 TTFT 换吞吐。建议：把配置里 `max_seconds: 600` 调到 300（缩短 stuck trial 等待）、`max_model_len` 去掉 32768（疑似触发不稳）、`n_trials` 加大到 16，重跑即可从 study.db 续跑（`load_if_exists`，trial #4 结果保留）。NPU 现已全部清空（6 卡均回 3.4GB 基线），随时可重启。

==============================================================================
>>> 助手回答 (transcript 行 2804)
==============================================================================
寻优路径（数据来自 `results_dp6_tune/study.db`，8 trials 目标，实际执行到 #6）：

```
Trial  state     objective      可行   max_num_seqs  max_model_len  gpu_mem  TTFT      用时
─────────────────────────────────────────────────────────────────────────────────────────────
#0     COMPLETE  0.0 tok/s      否①   48            24576          0.90     —         8.7min
#1     COMPLETE  0.0 tok/s      否①   32            24576          0.86     —         7.9min
#2     COMPLETE  0.0 tok/s      否①   16            24576          0.86     —         5.3min
#3     RUNNING   (无结果)        否②   64            24576          0.92     —         — (僵尸)
#4     COMPLETE  469.9 tok/s    ✅③   64            32768          0.80     4.80s    29.1min ★最优
#5     COMPLETE  99999(垃圾)     否④   64            32768          0.82     —         13.2min
#6     RUNNING   (无结果)        否⑤   32            32768          0.80     —         — (crash循环)
```

**注解：**
- ① **#0–#2 OOM infeasible**：这三条其实来自第一次启动（孤儿 worker 占 NPU 0-3 各 57GB）。`load_if_exists` 把它们带进同一 study，margin=−1 表示无 benchmark 指标。
- ② **#3 僵尸**：第一次 run 被手杀时正在 RUNNING，state 卡死。
- ③ **#4 唯一可行解（最优）**：`max_num_seqs=64 / max_model_len=32768 / gpu_mem=0.80` → **469.9 tok/s，TTFT 4.80s**（margin=+5200ms，距 10s 约束还有 5.2s 余量）。这是孤儿 worker 清场后第一次真正跑通的 trial。
- ④ **#5 垃圾 infeasible**：vLLM 启出 degraded 21GB 服务器（无 KV cache），ais_bench 卡 0/512，被手杀。parse 失败填入 DEFAULT_METRICS 的 99999 哨兵 → `ttft_ms=99999` → margin=10000−99999=**−89999**，被正确标 infeasible，**没有污染 TPE 搜索**。
- ⑤ **#6 crash/restart 循环**：vLLM 加载到 21GB 后在 KV cache profiling 阶段反复崩溃重启，永不完成。

**TPE 决策链：** 前三 startup trial（随机采样 24576）全部 infeasible → TPE 偏移搜索空间到 32768 区间 → #4 命中可行解 → #5 在 #4 基础上调高 gpu_mem 到 0.82 触发 degraded → #6 降 max_num_seqs 到 32 探索时撞上 crash 循环。

**结论**：在所探索的范围内，约束 `ttft_ms<=10000` 只有 **trial #4** 这一组参数同时满足可行 + 有真实吞吐，即推荐配置 `max_num_seqs=64, max_model_len=32768, gpu_memory_utilization=0.80`，对应 469.9 tok/s / TTFT 4.8s。

==============================================================================
>>> 助手回答 (transcript 行 3060)
==============================================================================
## DP=6 + block_size 寻优 — 最终报告

寻优全部完成（12/12 trials），全部满足 TTFT<10s 约束（除 1 个 infeasible）。

### 寻优路径
```
Trial  output tok/s  TTFT   max_num_seqs  max_model_len  block_size  gpu_mem
#0     828.1         7.7s   64            18432          64          0.80
#1     869.1         7.3s   48            32768          64          0.90
#2     895.5         8.6s   80            32768         128          0.86
#3     858.3         8.0s   96            32768          32          0.80
#4     868.6         8.3s  112            32768         128          0.84
#5     690.1         —      16            18432         128          0.88   ✗ infeasible
#6     960.8         7.1s  112            24576          32          0.92
#7     930.4         7.2s   48            24576          32          0.92
#8     929.6         7.7s   48            24576          32          0.82
#9     939.8         7.3s   80            24576          32          0.86
#10    966.5 ★最优   7.2s  128            18432          32          0.92
#11    962.2         7.7s  128            18432          32          0.92
```

### 推荐配置（最优 trial #10）
```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 vllm serve /data/qwen3.5/Qwen3.5-9B \
  --tensor-parallel-size 1 --data-parallel-size 6 \
  --max-num-seqs 128 --max-model-len 18432 --block-size 32 \
  --gpu-memory-utilization 0.92 \
  --enforce-eager --enable-chunked-prefill --no-enable-prefix-caching \
  --dtype auto --trust-remote-code --served-model-name qwen3.5
```

### 核心指标
| 指标 | 值 |
|---|---|
| **output 吞吐** | **966.5 tok/s** |
| **TTFT avg** | **7.23s**（约束 10s，余量 2.77s）✅ |
| TPOT (itl) | 121.4 ms |
| 并发 | 128 |
| 512/512 请求全部成功 | ✅ |
| 总耗时 | 542s |

### 关键发现
1. **新增 block_size 维度带来巨大提升**：本轮最优 966.5 vs 上一轮（无 block_size 寻优）469.9 → **翻了一倍**。`block_size=32`（小块）在 DP 场景下最优。
2. **TPE 收敛轨迹**：早期探索 `block_size=64/128`（828-895），中期发现 `mml=24576+block=32`（960.8），最终收敛到 `mml=18432+block=32+seqs=128+gmu=0.92`（966.5）。最优解在最后 2 个 trial 诞生。
3. **constraint 机制全程稳健**：11/12 trials 可行，仅 #5（max_num_seqs=16 太低导致排队）infeasible，TPE 正确避开。
4. **对比基线**：TP1 baseline 170.9 tok/s → DP6 最优 966.5 tok/s = **5.66× 加速**，接近理论 6×，TTFT 同时压在 7.2s。

### 顺手修复的框架 bug
[baseline/runner.py](src/vllm_tuner/baseline/runner.py) 的 `run_baseline` 之前不检查 `baseline.enabled`，即使配置禁用也无条件先跑一轮空参数 baseline（~30min 浪费）。已修复，每轮寻优省掉这一步。

### 产物
- 寻优路径 & 全部结果：`results_dp6_block_tune/study.db`
- 可视化报告：`results_dp6_block_tune/ascend-9b-dp6-block-tune_report.html`
- 12 个 trial JSON：`results_dp6_block_tune/study_data/ascend-9b-dp6-block-tune/trial_{0..11}.json`
Shell cwd was reset to /data/workspaces/qwen