# Qwen3.5-27B 4卡寻优对比实验总结报告

> 生成时间：2026-09-22 北京时间
> 模型：Qwen3.5-27B / TP=4 / NPU3-6（4×910B3 64GB）/ PD 混部
> 负载：aisbench_auto_tools_prefix / GSM8K 前缀数据集（16K入-4K出 / 并发40 / 160条 / 前缀命中率80%）
> 压测引擎：ais_bench 24e9ea8 / warmup=0（vllm 节点本地回环）

---

## 一、实验总表

| 方案 | 算法 | 总输出吞吐¹ | 单卡吞吐¹ | TTFT avg² | TPOT avg² | 最优配置要点 | 启动配置文件 |
|------|------|-----------|----------|-----------|-----------|-------------|-------------|
| 窄空间 PSO | PSO | **651.1 tok/s** | 162.8/卡 | 6.2s³ | 18.3ms | MNS=43 / MNBT=4096 / GMU=0.9 | `comparison/PSO_round/best_simulator.sh` |
| 窄空间 TPE | Optuna-TPE | **655.2 tok/s** | 163.8/卡 | 6.2s³ | 18.3ms | MNS=42 / MNBT=8192 / GMU=0.9 | `comparison/TPE_round/best_simulator.sh` |
| 扩展空间 TPE | Optuna-TPE | **651.6 tok/s** | 162.9/卡 | 6.3s³ | 18.3ms | MNS=47 / MNBT=8192 / GMU=0.88 | vllm 节点 `result/store/data_storage_20260921015126.csv` 最优行 |
| 窄空间基线 | 默认参数 | 507.4 tok/s | 126.9/卡 | 15.6s | 69.1ms | MNS=42 / MNBT=8192 / GMU=0.9（P0 修正前初始值） | — |
| 人工配置 | 人工调优 | 1115.2 tok/s⁴ | 278.8/卡⁴ | 6.8s⁴ | 48.7ms⁴ | MNS=128 / num_spec=2 / async / 融合算子 | `tools/manual_serve.sh` |

> ¹ 吞吐口径：PSO/TPE 来自优化器 CSV 的 `generate_speed`（优化器解析 ais_bench 输出后的聚合值）；人工配置来自工具箱 CSV 的 `output_throughput`（工具箱独立计算）。**两者计算路径不同，数值不能直接横比**——严格对比需统一工具+预热条件后重测。
> ² TTFT/TPOT 为优化器 CSV / ais_bench 各自口径，同上。
> ³ TPE 窄空间最优粒子的 TTFT/TPOT 值。
> ⁴ 人工配置通过工具箱 `aisbench_test.py` 直测（npu_num=4），160 条全部完成，进程在服务停止时未输出完整汇总行，数值来自逐请求明细表。

---

## 二、窄空间内 PSO vs TPE 算法对比

### 2.1 结果对比

| 指标 | PSO | Optuna-TPE | 结论 |
|------|-----|-----------|------|
| 最优吞吐 | 651.1 | 655.2 | 差异 +0.6% < 基线互验噪声（+4.2%），统计平局 |
| 失败评估 | 1/25 | **0/25** | TPE 容错更优 |
| 最优解出现 | 第 ~11 次 | 第 ~7 次 | TPE 更早进入高产区 |
| 收敛特征 | 第 2 代起向 GMU=0.9 脊线聚集 | 第 5 粒子后收缩至 MNS 42-43 邻域 | 均有效收敛 |

### 2.2 优化效率

| 效率维度 | PSO | TPE |
|---------|-----|-----|
| 首次达到 ≥600 tok/s | 第 9 次评估 | 第 6 次评估 |
| 最优解首次出现 | 第 ~11 次 | 第 ~7 次 |
| 预算内最终排名 | 第 1 名（PSO 自身） | 第 1 名（TPE 自身） |

### 2.3 结论

1. **窄空间内两算法统计平局**——最优差异（+0.6%）远小于环境噪声（+4.2%），两者都收敛到同一甜点脊线（MNBT 4096~8192 + GMU 0.9）
2. **TPE 略占优**：零失败评估 + 更早进入高产区（第 6 vs 9 次评估）——与 TPE 对低维昂贵黑盒的样本效率优势一致
3. **PSO 的重采样机制**在高 MNS + 大 capture 组合下发挥了避错作用（1 次失败被自动跳过）

---

## 三、搜索空间对寻优结果的约束（核心发现）

### 3.1 人工配置 vs skill 搜索空间的差异

| 差异维度 | skill 搜索空间 | 人工配置 | 差距原因（skill 环节） |
|---------|---------------|---------|----------------------|
| MNS 上界 | 40~48（R1 闭环钳制） | 128 | Phase 4 `calc_max_num_seqs_range` 经验钳制（MNS_HW_UPPER=32），未纳入命中率修正前 MNS 上界过窄 |
| num_spec 深度 | 1（固定） | 2 | Phase 4.2.10 MTP 投机深度未开放搜索（仅 mtp_num_hidden_layers=1 推导出固定值） |
| async-scheduling | 未开启 | 开启 | Phase 4 特性矩阵未收录该 flag |
| additional_config 融合算子 | 部分注入（P0-1 修复后） | 完整（4 项） | P0-1 修复了补丁加载，但 fuse_qknorm_rope 等融合开关需要 toml 补充（已补充） |
| capture 档位 | 密集 stride-4 至 228（P0-3 修复后） | 密集每 4 至 136 | P0-3 修复了覆盖洞（业务并发 40 不在列表中） |
| max-model-len | 22528（固定） | 262144 | Phase 4 推导为固定值，未开放搜索 |

### 3.2 P0/R1 修复的效果验证（对照 TPE 窄空间轮）

| 修复项 | 窄空间（修复前） | 扩展空间（修复后） | 验证结果 |
|--------|----------------|-------------------|---------|
| MNS 上界 | 40~44 | 40~256（命中率修正）→ 40~48（R1 闭环钳制） | MNS=114/159/254 粒子全部正常运行 ✅ |
| capture 覆盖 | 业务并发 40 ∉ 列表 | 密集 stride-4 覆盖 40/44/48 | 所有粒子服务正常启动 ✅ |
| 补丁加载 | hw 档缺失 → 整包丢失 | 家族级段部分加载 | additional_config/cpu_binding/family env 物化 ✅ |

---

## 四、启动配置使用说明

### 4.1 PSO 最优配置启动

```bash
# 文件位置（共享存储）
bash /home/projects/optimize-test/comparison/PSO_round/best_simulator.sh

# 或手动启动（等价命令）
source /root/mssprofiler/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=3,4,5,6
vllm serve /home/projects/qwen/Qwen3.5-27B \
    --served-model-name Qwen3.5-27B --host 127.0.0.1 --port 8000 \
    --max-num-batched-tokens 4096 --max-num-seqs 43 \
    --tensor-parallel-size 4 --max-model-len 101120 --block-size 128 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --enable-chunked-prefill --enable-prefix-caching \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'
```

### 4.2 TPE 最优配置启动

```bash
# 文件位置（共享存储）
bash /home/projects/optimize-test/comparison/TPE_round/best_simulator.sh

# 或手动启动（等价命令）
source /root/mssprofiler/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=3,4,5,6
vllm serve /home/projects/qwen/Qwen3.5-27B \
    --served-model-name Qwen3.5-27B --host 127.0.0.1 --port 8000 \
    --max-num-batched-tokens 8192 --max-num-seqs 42 \
    --tensor-parallel-size 4 --max-model-len 20608 --block-size 128 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --enable-chunked-prefill --enable-prefix-caching \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'
```

### 4.3 人工配置启动（供对照）

```bash
# 完整脚本（含全部环境变量）
bash /tmp/manual_serve.sh

# 核心差异参数（相对 PSO/TPE 最优）
--max-num-seqs 128            # vs 42-43
--max-model-len 262144        # vs 20608-101120
--gpu-memory-utilization 0.95 # vs 0.9
--async-scheduling            # PSO/TPE 均未开启
--speculative-config '{"num_speculative_tokens": 2, ...}'  # vs 1
--additional-config '{"enable_cpu_binding":true,"fuse_qknorm_rope":true,...}'
```

### 4.4 压测命令（三种配置通用）

```bash
source /root/mssprofiler/set_env.sh
cd /root/mssprofiler
/root/mssprofiler/bin/ais_bench --models vllm_api_stream_chat \
    --datasets gsm8k_gen_0_shot_cot_str_perf \
    --mode perf --num-prompts 160 \
    --work-dir /root/mssprofiler/result/benchmark/ais_bench
```

---

## 五、工程修复清单（全部已提交 git 可追溯）

| # | 修复 | git 提交 | 影响范围 | 分类 |
|---|------|---------|---------|------|
| 1 | python3.12 迁移 + 软链统一 | 基线 90fef55 | 全链路 | 环境 |
| 2 | ATB 环境固化（generate_config 模板） | 基线 | Phase 5/6 | 环境 |
| 3 | num-warmups 运行时探测 | 基线 | Phase 5/6 | 兼容 |
| 4 | phase2 安装管理 v2（wheelhouse 优先） | dfc0477 | Phase 2 | 环境 |
| 5 | Optuna-TPE 算法接入 + 算法工厂 | 2387b84 / 90fef55 | Phase 6 | 功能 |
| 6 | P0-1 补丁部分加载（家族级段解耦） | 38fcfdd | Phase 4/5 | 功能 |
| 7 | P0-2 命中率→MNS 上界 | 38fcfdd | Phase 4 | 功能 |
| 8 | P0-3 capture 稠密化+覆盖回填 | 38fcfdd | Phase 4 | 功能 |
| 9 | R1 闭环钳制（MNS≤cc+8） | 5fa1869 + bcf77de | Phase 4 | 功能 |
| 10 | PD 校验门控（#7944 仅 PD 分离） | 7290667 | Phase 5 | 修复 |
| 11 | pop/finalize 顺序修复 | 7290667 | Phase 4/5 | 修复 |
| 12 | phase7_report copy2→copy | 5fa1869 | Phase 7 | 修复 |
| 13 | qwen35_d_prefix_caching PD 门控 | 7290667 | Phase 5 | 修复 |
| 14 | set_env.sh ATB 环境块固化 | 5fa1869 | Phase 5/6 | 修复 |
| 15 | pip 镜像源切换（华为云 10×提速） | 环境 | 全链路 | 环境 |
| 16 | msmodeling 构建绕行（源码包分发） | 环境 | Phase 1 | 环境 |

---

## 六、后续建议

| 优先级 | 建议 | 预期收益 |
|--------|------|---------|
| P1 | num_spec 深度开放搜索（[1,2,3]） | 27B 案例人工配置用 num_spec=2，预期 TPOT 进一步降低 |
| P1 | async-scheduling 纳入特性矩阵（dense hybrid A2） | host 开销隐藏 |
| P1 | 源码枚举式特性矩阵（自动发现 vllm-ascend 新 flag） | TOPK/DENSE_OPTIMIZE/FUSED_MC2/NZ 等 |
| P2 | 源码枚举式 capture 覆盖自检（业务并发 ∈ 列表） | 避免 pad 浪费 |
| P2 | 多保真搜索（短压测淘汰+全长精测） | 提升预算效率 |
| P2 | 两阶段漏斗搜索（先结构维度后连续维度） | 结构性维度优先收敛 |
| P3 | topology.json 一等声明（控制/执行/共享存储/角色映射） | 消除硬编码路径 |
