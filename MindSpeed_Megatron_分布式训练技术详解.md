# MindSpeed / Megatron 大模型分布式训练技术详解

> 本文整理自对话记录，修复了全部 LaTeX 公式的排版问题：所有独立公式均用 `$$...$$` 包裹、行内公式用 `$...$` 包裹，并清除了公式中混入的 Markdown 加粗标记 `**` 与损坏的图片占位符。公式内容已按标准数学表达还原。

## 目录

1. [MindSpeed 与 Megatron / CANN 的关系](#sec01)
2. [MindSpeed 训练科学类模型是否有效](#sec02)
3. [基于 MindSpeed / Megatron 的大模型训练关键技术](#sec03)
4. [序列并行（SP）原理](#sec04)
5. [张量维度 [S, B, H] 的含义](#sec05)
6. [Attention 与 MLP 的四个投影](#sec06)
7. [为什么 LayerNorm、Dropout、残差连接在 S 维逐点独立](#sec07)
8. [序列并行（SP）是标配吗](#sec08)
9. [流水线并行（PP）原理](#sec09)
10. [流水线气泡占比公式原理](#sec10)
11. [PP 与 VPP 的命令行参数与调优策略](#sec11)
12. [上下文并行（CP）原理](#sec12)
13. [CP 与 Hybrid CP 的命令行参数与计算规则](#sec13)
14. [专家并行（EP）原理](#sec14)
15. [数据并行与分布式优化器（DP / ZeRO-1）](#sec15)
16. [基于 Qwen-27B 训练多学科科学 LLM 的总体策略](#sec16)
17. [两台昇腾 A3（2×16 卡）SFT 与 RL 工程架构建议](#sec17)
18. [MindSpeed/Megatron 与 torch/torch_npu 两种方案的关键区别](#sec18)
19. [通信与计算重叠（TP / CP / PP Overlap）](#sec19)

---

<a id="sec01"></a>
## 一、 MindSpeed 与 Megatron / CANN 的关系

是的，昇腾的 **MindSpeed** 本质上就是华为针对 **Megatron-LM** 开发的大模型分布式训练加速库，用于将 Megatron-LM 体系高效适配到昇腾 CANN 架构与 `torch_npu` 运行环境中。

### 架构分层与协作关系

- **上层（策略与算法）——Megatron-LM**：提供切片逻辑与分布式并行框架（张量并行 TP、流水线并行 PP、序列并行 SP、专家并行 EP、上下文并行 CP 等）。
- **中层（适配与加速）——MindSpeed**：通过非侵入式（Monkey Patch）机制插入 Megatron-LM，负责重写计算图、优化内存分配，并将 NV/CUDA 算子映射至昇腾 NPU 优化的计算逻辑。
- **下层（硬件与驱动）——`torch_npu` + CANN**：MindSpeed 调用的底层接口。CANN 提供 HCCL 通信库与 Ascend C 算子实现，最终在 NPU 芯片上完成硬件加速。

### MindSpeed 解决的核心适配问题

- **算子替换与融合**：将 Megatron-LM 原生的 CUDA/Cutlass 算子替换为基于 CANN/Ascend C 深度优化的 NPU 极致性能算子（例如 NPU 版 FlashAttention、Fused RMSNorm、Fused SwiGLU 等）。
- **通信与计算掩盖（Overlap）**：针对 HCCL 通信链路特点，重写 TP/SP/PP 的通信流水线，实现 Tensor-AllReduce/AllGather 与 Vector/Cube 计算的最佳调度掩盖。
- **特有长文本与 MoE 扩展**：在原生 Megatron 基础上，扩展并优化了适用于昇腾硬件的大文本 Context Parallel（如 Ring Attention 优化）以及 DeepSeek 等架构特有的稀疏 MoE 路由通信。
- **模型套件支持**：在 MindSpeed 核心库之上衍生了 **MindSpeed-LLM** 和 **MindSpeed-MM**，提供开箱即用的 Megatron 脚本，支持 Qwen、DeepSeek、LLaMA 等开源大模型的昇腾 NPU 训练。

---

<a id="sec02"></a>
## 二、 MindSpeed 训练科学类模型是否有效

MindSpeed 是否有效，取决于科学类 AI（AI4S）模型的**底层神经网络架构**与**计算图特征**。对于基于 Transformer 序列范式的模型极为高效，但对于包含大量图消息传递、3D 等变空间计算或非标注意力的模型，效果有限或需要深度定制。

### 高适配度：序列与 Transformer 架构（如 OligoFormer、ESM、Evo）

- **典型代表**：OligoFormer、RNA-FM、ESM-2/3、Evo 2 等 RNA/蛋白质序列大模型。
- **有效性分析**：**非常有效**。
- **核心加速逻辑**：此类模型与标准 LLM 架构高度同构，本质都是处理 1D 生物序列（核苷酸/氨基酸）。
  - **并行策略**：可以直接套用 MindSpeed 的张量并行（TP）、序列并行（SP）和上下文并行（CP），轻松突破单卡显存限制以支持超长 RNA/蛋白质序列训练。
  - **算子优化**：MindSpeed 提供的 NPU 版 FlashAttention、Fused RMSNorm/LayerNorm 及 SwiGLU 融合算子可无缝接入，显著提升计算密集型 MatMul 的 Cube 核心利用率。

### 适配受限：图网络与 3D 扩散模型（如 DiffSBDD、EGNN）

- **典型代表**：DiffSBDD、EGNN（等变图神经网络）、RFdiffusion 等 3D 分子生成与小分子对接模型。
- **有效性分析**：**直接作用有限**。
- **核心瓶颈分析**：
  - **计算范式不同**：此类模型的瓶颈不在于大尺度的密集矩阵乘法（MatMul），而在于 **3D 空间坐标更新、图消息传递（Message Passing）以及严重的 Scatter/Gather 访存开销**。
  - **MindSpeed 局限**：MindSpeed 基于 Megatron-LM 开发，其并行切片（如 TP/SP）是沿着 1D 序列或通道维度切分的，无法直接映射到 3D 空间坐标或不规则的分子图结构上。
  - **优化路径**：在昇腾 NPU 上优化这类模型，通常不依赖 MindSpeed，而是需要在 `torch_npu` 下使用 **Ascend C / Triton 自定义重写 Scatter/Gather 和图注意力融合算子**，解决 Vector 单元的访存瓶颈。

### 部分适配：复杂结构预测模型（如 AlphaFold2 / AlphaFold3）

- **典型代表**：AlphaFold2 (Evoformer)、AlphaFold3、OpenFold。
- **有效性分析**：**需要结合专有算子重构**。
- **核心优化拆解**：
  - **标准模块**：模型中用于处理 MSA 和序列特征的标准 Self-Attention / Cross-Attention 模块，可以享受到 MindSpeed 的分布式切分与 FlashAttention 加速。
  - **专用非标算子**：AlphaFold 核心的 **Triangle Attention（三角注意力）、Triangle Multiplicative Update（三角乘法更新）** 以及 **IPA（不变点注意力）**，并不属于 Megatron-LM 的标准算子，MindSpeed 无法对其自动切分。
  - **工程落地**：对 AlphaFold 类的加速，工程上通常需要专门使用 Ascend C 开发 Triangle/IPA 专属算子内核，并结合 MindSPONGE 或开源 OpenFold-NPU 适配工程，仅在标准 Transformer 骨干网络（Trunk）部分借用 MindSpeed 的并行思路。

### 总结选型建议

| 模型计算特征 | 代表模型 | 是否推荐使用 MindSpeed | 昇腾优化首选方案 |
| --- | --- | --- | --- |
| 1D 密集序列 / Transformer | OligoFormer, RNA-FM, ESM | **推荐** | MindSpeed + `torch_npu`（TP/SP + FlashAttention） |
| 3D 空间坐标 / 图神经网络 | DiffSBDD, EGNN | **不推荐** | `torch_npu` + Ascend C / Triton 融合算子优化 |
| 混合结构 / 非标注意力 | AlphaFold2/3, OpenFold | **部分推荐** | MindSpeed（Trunk）+ 自定义 Triangle/IPA 专用算子 |

判断的核心标准在于：**模型的主要计算瓶颈是标准的 Transformer 密集矩阵乘法，还是非标的 3D 图/几何结构计算。**

---

<a id="sec03"></a>
## 三、 基于 MindSpeed / Megatron 的大模型训练关键技术

基于 MindSpeed / Megatron-LM 的大语言模型分布式训练系统，其核心是通过**多维并行、显存控制、计算通信重叠**以及**深度算子融合**，解决千亿/万亿参数大模型在大规模集群上的显存瓶颈与通信开销问题。

### 一、 多维混合并行策略（Hybrid Parallelism）

- **张量并行（TP, Tensor Parallelism）**：将 Transformer 层中的 Linear 权重按列或按行切分（如 MLP 和 Attention 的 projection 矩阵），在前向/反向传播过程中通过 `AllReduce` 或 `AllGather` 同步。
- **序列并行（SP, Sequence Parallelism）**：在 TP 组内部，将原本在每张卡上重复计算且耗显存的 LayerNorm 和 Dropout 沿序列维度（Sequence Dimension）切分，降低激活值显存占用。
- **流水线并行（PP, Pipeline Parallelism）**：将模型的不同 Layer 纵向切分到不同 Stage/卡上。结合 **1F1B (One Forward, One Backward)** 或 **Interleaved 1F1B（Virtual Pipeline）** 调度算法，减少流水线气泡（Bubble）。
- **上下文并行（CP, Context Parallelism）**：针对超长文本（如 32k~1M context），沿序列维度切分 Self-Attention 中的 Key/Value。主要实现包括 **Ring-Attention**（环形传递 KV 块）和 **DeepSpeed-Ulysses**（通过 All-to-All 转置实现 Head 维度与 Sequence 维度的切换）。
- **专家并行（EP, Expert Parallelism）**：针对 MoE 混合专家架构，将不同 Expert 分配到不同 NPU/GPU 上，通过 `All-to-All` 算子实现 Token 在不同专家节点间的路由分发与聚合。
- **数据并行与分布式优化器（DP / ZeRO-1）**：在 DP 组内切分优化器状态（FP32 Master Weight、一阶/二阶动量）与梯度，大幅消除传统数据并行的显存冗余。

### 二、 显存与计算优化（Memory & Activation Optimizations）

- **选择性重算（Selective Recomputation）**：仅对显存开销极大但计算量相对较小的激活值（如 Self-Attention 中的 Attention Matrix、SwiGLU 激活函数等）进行 Recompute，在节省显存的同时减少不必要的计算重算开销。
- **全量重算（Full Checkpointing）**：在极限显存场景下，仅保存 Transformer Block 输入处的 Tensor，反向传播时重新计算整个 Layer。
- **Offload 机制**：将不频繁使用的激活值或优化器状态暂存至 Host CPU 内存，以时间换空间。

### 三、 通信与计算掩盖（Communication-Computation Overlap）

- **TP 通信重叠（TP Overlap）**：在 Tensor Parallel 的 Gemm（矩阵乘）与 AllReduce/AllGather 通信之间建立异步流水线，将向量通信时间隐藏在 Cube 核心的大矩阵计算之后。
- **CP 通信重叠（Ring CP Overlap）**：在 Ring Attention 迭代计算时，将当前 Block 的 Attention 计算与上一个 Block 的 KV 张量 P2P/Ring 传输重叠。
- **PP P2P 通信重叠**：将流水线 Stage 间的激活值与梯度点对点传输，与前向/反向的 Transformer Block 计算重叠。

### 四、 高性能算子融合与硬件适配（Kernel Fusion & NPU Acceleration）

- **FlashAttention / FlashDecoding**：引入 FlashAttention 算法（在昇腾下调用基于 Ascend C 优化的 FlashAttention 内核），将 Softmax 和 Attention Score 计算融合，显著削减 HBM/显存访存带宽需求。
- **融合算子库（Fused Kernels）**：将频繁触发 Vector 访存瓶颈的算子合并，例如 **Fused RMSNorm**、**Fused SwiGLU**、**Fused RoPE** 以及 **Fused Linear-CrossEntropy**，减少 Kernel Launch 延迟与碎片化访存。
- **混合精度训练（Mixed Precision Training）**：支持 BF16/FP16 混合精度以及 FP8 动态/静态 Scaling 训练，利用专有 Matrix/Cube 单元获得吞吐倍增。

### 五、 前沿架构专用适配

- **MLA（Multi-Head Latent Attention）优化**：针对 DeepSeek 系列模型的低秩压缩 Attention，进行 KV 缓存解压与并行切分重构。
- **MoE 无辅助损失路由与负载均衡**：优化动态 Token 路由的 All-to-All 通信与分组计算，防止部分 Expert 发生计算拖尾。

---

<a id="sec04"></a>
## 四、 序列并行（SP）原理

在基于 Megatron-LM 和 MindSpeed 的分布式训练体系中，**序列并行（Sequence Parallelism, 简称 SP）** 是一项极其重要的技术。它由 NVIDIA 在论文 *Reducing Activation Recomputation in Large Transformer Models* 中提出，旨在**消除张量并行（TP）组内重叠计算与显存冗余，且完全不增加任何额外的通信数据量**。

### 一、 背景：原生张量并行（TP）的显存痛点

在传统的 Megatron-LM 张量并行（TP）中，为了切分计算量，Transformer 层的矩阵乘法（MatMul）被分成了两部分：

1. **ColumnParallelLinear**：如 Attention 的 Q、K、V 投影和 MLP 的 Gate/Up 投影，按列切分权重。
2. **RowParallelLinear**：如 Attention 的 O 投影和 MLP 的 Down 投影，按行切分权重。

但是，在这些矩阵乘法**之外**的区域——包括 **LayerNorm / RMSNorm、Dropout、残差连接（Residual Connection）**——原生 TP 并没有对其进行切分。

这意味着在同一个 TP 组内的所有 NPU/GPU 上：

- 每一个节点都保存了一份**完整大小**的输入激活值（Shape 为 $[S, B, H]$，其中 $S$ 为序列长度，$B$ 为 Batch Size，$H$ 为 Hidden Size）。
- 每一个节点都在重复计算完全相同的 LayerNorm 和 Dropout。

随着模型隐层维度 $H$ 和序列长度 $S$ 的增加，**这些非 MatMul 区域占用的激活值显存（Activation Memory）逐渐成为了显存爆炸的主要元凶**。

### 二、 SP 的核心思想

序列并行（SP）的逻辑非常直接：**既然 LayerNorm、Dropout 和残差连接在 Token 维度（序列维度 $S$）上是独立逐点计算（Element-wise）的，为什么不将它们沿 $S$ 维度切分到 TP 组内的各个卡上计算呢？**

在 SP 启用后：

- 在非 MatMul 区域，每张卡只存 $1/TP$ 长度的序列，即 $[S/TP, B, H]$。
- 在需要进行 MatMul 区域前，通过通信拉取完整序列，完成矩阵乘法。

### 三、 SP 的数据流与通信算子重构

Megatron/MindSpeed 实现 SP 的精髓在于对通信算子的巧妙重构。它将原生 TP 中的 `AllReduce` 拆解为 `ReduceScatter` 和 `AllGather`。

#### 1. 传统 TP 与 TP+SP 数据流对比

以 Transformer 中的一个子块（如 Self-Attention 或 MLP）为例：

```
【原生 TP 流程】
输入 [S, B, H] (全卡冗余)
  │
  ├──> LayerNorm ──────> [S, B, H] (全卡冗余计算)
  ├──> ColumnParallel ──> [S, B, H/TP]
  ├──> RowParallel ─────> [S, B, H] (局部部分和 Partial Sum)
  ├──> AllReduce ──────> [S, B, H] (通信聚合)
  └──> Dropout + Residual ─> [S, B, H] (全卡冗余计算)
──────────────────────────────────────────────────────────
【TP + SP 流程】
输入 [S/TP, B, H] (沿序列维度切分)
  │
  ├──> LayerNorm ───────────> [S/TP, B, H] (本地只算 1/TP)
  ├──> AllGather ───────────> [S, B, H] (通信拼回完整序列)
  ├──> ColumnParallel ──────> [S, B, H/TP]
  ├──> RowParallel ──────────> [S, B, H] (局部部分和 Partial Sum)
  ├──> ReduceScatter ───────> [S/TP, B, H] (求和并沿 S 维度切分分发)
  └──> Dropout + Residual ─────> [S/TP, B, H] (本地只算 1/TP)
```

#### 2. 通信变换的核心细节

在原生 TP 中：

- `RowParallel` 的输出是一个 Partial Sum（部分和），必须做一次 **`AllReduce`** 才能让所有卡得到完整的累加结果 $[S, B, H]$。

在 TP + SP 中：

- `RowParallel` 的输出依然是 Partial Sum，但此时不再调用 `AllReduce`，而是调用 **`ReduceScatter`**。
- `ReduceScatter` 算子在规约求和的同时，把结果按 $S$ 维度散布（Scatter）到 TP 组的各张卡上。此时每张卡只拿到属于自己的 $[S/TP, B, H]$ 正确结果。
- 随后在进入下一个 MatMul 块之前，调用 **`AllGather`** 沿 $S$ 维度收集全量序列 $[S, B, H]$，供下一个 `ColumnParallel` 计算。

### 四、 为什么 SP 的通信开销为 0？

看似增加了通信频次（从 1 次 `AllReduce` 变成了 1 次 `ReduceScatter` + 1 次 `AllGather`），但**总通信数据量完全相等**。

根据分布式通信算法（如 Ring 算法）：

- 一个传输数据量为 $M$ 的 **`AllReduce`** 通信量为：

$$\text{Data Volume}_{\text{AllReduce}} = 2 \times \frac{TP - 1}{TP} \times M$$

- **`ReduceScatter`** 传输量为：

$$\text{Data Volume}_{\text{ReduceScatter}} = \frac{TP - 1}{TP} \times M$$

- **`AllGather`** 传输量为：

$$\text{Data Volume}_{\text{AllGather}} = \frac{TP - 1}{TP} \times M$$

因此：

$$\text{ReduceScatter} + \text{AllGather} = \frac{TP - 1}{TP} \times M + \frac{TP - 1}{TP} \times M = 2 \times \frac{TP - 1}{TP} \times M = \text{AllReduce}$$

**结论：SP 以零额外通信传输量的代价，完美剥离了非 MatMul 区域的显存冗余。**

### 五、 显存收益定量分析

在标准 Transformer Layer 中，保存用于反向传播的激活值显存公式如下（假设 Batch Size 为 $B$，Seq Len 为 $S$，Hidden Dim 为 $H$，Attention Heads 为 $N$）。

不开启 SP 时，非 MatMul 区域（LayerNorm, Dropout, Residual）的激活值显存开销为：

$$\text{Memory}_{\text{non-MatMul}} = 2 \cdot B \cdot S \cdot H \ (\text{LayerNorm}) + 1 \cdot B \cdot S \cdot H \ (\text{Dropout}) + \cdots$$

开启 SP（切分为 $TP$ 份）后：

- LayerNorm 激活值：$B \times S \times H \longrightarrow \dfrac{B \times S \times H}{TP}$
- Dropout 掩码与激活值：$B \times S \times H \longrightarrow \dfrac{B \times S \times H}{TP}$
- 残差连接：$B \times S \times H \longrightarrow \dfrac{B \times S \times H}{TP}$

对于 $TP = 8$ 的集群，这直接将大头非 MatMul 激活值显存削减了 $1 - 1/8 = 87.5\%$，大幅降低了开启全量重算（Full Recomputation）的必要性，从而间接提升了训练 Overall TFLOPS。

### 六、 补充：Megatron SP 与 Context Parallel（CP）的区别

经常容易与 Sequence Parallelism 混淆的是 **Context Parallelism（CP）**（如 DeepSpeed-Ulysses 或 Ring-Attention），二者虽然都是沿着序列维度切分，但应用层级不同：

| 维度 | Megatron 序列并行（SP） | 上下文并行（CP） |
| --- | --- | --- |
| **作用范围** | **TP 组内部**（$TP \le 8$） | **DP / CP 组**（可扩展至几十/上百卡） |
| **解决目标** | 消除 TP 组内 LayerNorm/Dropout 的显存冗余 | 解决单卡超长上下文（100k+）下 Attention 显存与计算瓶颈 |
| **Attention 计算** | 每张卡拿到完整序列 $S$，在 Head 维度做并行 Attention | 在 $S$ 维度切分 Attention 本身（通过 Ring-Attention 或 All-to-All 转置） |
| **通信机制** | `AllGather` + `ReduceScatter` | `All-to-All`（Ulysses）或 P2P 环形传递 KV（Ring-Attention） |

在现代大模型训练（如 MindSpeed 训练 32k/128k 长文本）中，通常会将 **SP 与 CP 结合使用**：在 TP 组内开启 Megatron SP 压榨单节点显存，在跨节点组开启 CP 扩展超长上下文。

---

<a id="sec05"></a>
## 五、 张量维度 [S, B, H] 的含义

在大模型（Transformer 架构）训练与推理中，张量维度 $[S, B, H]$ 各字母分别代表：

- **S（Sequence Length / 序列长度）**：单个样本包含的 Token（词/字/片段）数量。例如，输入的文本被切分为 2048 个 Token，则 $S = 2048$。
- **B（Batch Size / 批次大小）**：单次前向或反向传播中同时并发处理的文本样本（句/段）数量。
- **H（Hidden Size / 隐藏层维度）**：模型内部隐层向量（Embedding / Feature Vector）的特征维度长度。例如，LLaMA-7B / Qwen-7B 的 $H = 4096$，70B 级别的模型通常为 $H = 8192$。

### 补充说明（为什么是 $[S, B, H]$ 而不是 $[B, S, H]$）

在主流 PyTorch 框架中，经常看到 $[B, S, H]$（Batch First）的形式；但在 Megatron-LM 和 MindSpeed 等分布式大模型加速库中，通常默认采用 **$[S, B, H]$（Sequence First）**：

1. **便于序列切分**：在序列并行（Sequence Parallelism）或上下文并行（Context Parallelism）中，需要沿 $S$ 维度对张量进行切分与通信。将 $S$ 放在第 0 维，在内存中是连续排列的，进行切片（Slice）、拼接（Concat）或通信（`AllGather`/`ReduceScatter`）时速度最快，无需额外的重拼装（Re-stride）或转置开销。
2. **加速矩阵乘法**：Transformer 中的矩阵乘法（如 ColumnParallel 和 RowParallel）可以直接将 $[S \times B, H]$ 视为二维矩阵处理，极大地简化了底层 BLAS/CANN 算子的调用逻辑。

---

<a id="sec06"></a>
## 六、 Attention 与 MLP 的四个投影

这四个投影（Projection）本质上都是**全连接层/矩阵乘法（Linear Layer）**，它们分别构成了现代 Transformer 大模型（如 LLaMA、Qwen、DeepSeek 等）中 **Self-Attention 模块**与 **MLP（FFN）模块**的输入与输出变换。

### 一、 Attention 中的 Q、K、V 投影 与 O 投影

Self-Attention 模块的作用是计算序列中 Token 之间的相关性权重并聚合特征。

#### 1. Q、K、V 投影（输入端 / 映射到多头空间）

- **位置**：Self-Attention 模块的入口。
- **计算过程**：

$$Q = XW_q, \quad K = XW_k, \quad V = XW_v$$

其中输入 $X \in \mathbb{R}^{S \times B \times H}$（隐藏层维度 $H$）。

- **作用**：
  - **Q（Query / 查询）**：代表当前 Token“想要寻找”什么信息。
  - **K（Key / 键）**：代表当前 Token“能提供”什么信息。
  - **V（Value / 值）**：代表当前 Token 实际包含的“特征内容”。
- **TP 类型**：**ColumnParallelLinear（按列切分）**。将 $W_q, W_k, W_v$ 沿输出维度（即 Attention Heads 维度）切分，各卡/NPU 独立计算自己负责的几个 Head，**无卡间通信**。

#### 2. O 投影（Output Projection，输出端 / 融合降维）

- **位置**：Self-Attention 模块的出口。
- **计算过程**：

$$\text{Output}_{\text{attn}} = \text{Attn\_Out} \cdot W_o$$

- **作用**：在多头注意力计算完成后，将各个 Head 算出的局部特征拼接起来，通过 $W_o$ 进行**特征融合**，并映射回统一的隐藏层维度 $H$，以便进行后续的残差相加（Residual Connection）。
- **TP 类型**：**RowParallelLinear（按行切分）**。输入 `Attn_Out` 在各卡上持有部分 Head 的结果，$W_o$ 沿输入维度按行切分。每张卡算出部分和（Partial Sum），最后通过 **`AllReduce`**（开启 SP 时为 **`ReduceScatter`**）聚合出完整向量。

### 二、 MLP 中的 Gate、Up 投影 与 Down 投影

现代 LLM 普遍采用了 **SwiGLU** 架构（替代了传统 GPT-2 的 GELU MLP），因此 MLP 层从原来的 2 个线性层变为了 3 个线性层（Gate、Up、Down）。

#### 1. Gate 投影 与 Up 投影（输入端 / 升维与门控）

- **位置**：MLP（前馈神经网络）模块的入口。
- **计算过程**：

$$\text{Gate} = XW_{\text{gate}}, \quad \text{Up} = XW_{\text{up}}$$

$$\text{Inter} = \text{Swish}(\text{Gate}) \odot \text{Up}$$

- **作用**：
  - **Up 投影**：将输入向量从隐藏层维度 $H$ 升维到更高的中间维度 $H_{\text{ffn}}$（通常 $H_{\text{ffn}} \approx 4H$ 或 $\dfrac{8}{3}H$），拓展特征表达空间。
  - **Gate 投影**：计算一个同等维度的门控信号，配合激活函数（Swish）对 Up 升维后的特征进行按元素（Element-wise）的**动态筛选与过滤**（类似逻辑门控制信息流）。
- **TP 类型**：**ColumnParallelLinear（按列切分）**。$W_{\text{gate}}$ 和 $W_{\text{up}}$ 均按列切分，各卡各自完成升维和 SwiGLU 点乘，**无卡间通信**。

#### 2. Down 投影（输出端 / 降维回归）

- **位置**：MLP 模块的出口。
- **计算过程**：

$$\text{Output}_{\text{mlp}} = \text{Inter} \cdot W_{\text{down}}$$

- **作用**：将高维空间 $H_{\text{ffn}}$ 中经过门控过滤后的特征，重新**降维**映射回原始的隐藏层维度 $H$，输出给下一个 Transformer Block。
- **TP 类型**：**RowParallelLinear（按行切分）**。$W_{\text{down}}$ 按行切分，各卡将本地的局部中间特征降维后，调用 **`AllReduce`**（开启 SP 时为 **`ReduceScatter`**）完成全局累加。

### 三、 维度与张量并行（TP）映射总结

| 投影名称 | 所在模块 | 维度变换（输入 → 输出） | TP 矩阵切分类型 | 通信触发点 |
| --- | --- | --- | --- | --- |
| Q、K、V 投影 | Attention | $H \to H_{q/k/v}$ | **ColumnParallel**（按列） | 无 |
| O 投影 | Attention | $H \to H$ | **RowParallel**（按行） | 触发 `AllReduce` / `ReduceScatter` |
| Gate / Up 投影 | MLP（SwiGLU） | $H \to H_{\text{ffn}}$ | **ColumnParallel**（按列） | 无 |
| Down 投影 | MLP（SwiGLU） | $H_{\text{ffn}} \to H$ | **RowParallel**（按行） | 触发 `AllReduce` / `ReduceScatter` |

---

<a id="sec07"></a>
## 七、 为什么 LayerNorm、Dropout、残差连接在 S 维逐点独立

LayerNorm、Dropout 和残差连接之所以在 Token 维度（序列维度 $S$）上完全独立、互不依赖，是由它们的**数学计算定义**与 Transformer 的**架构设计哲学**共同决定的。

### 一、 逐个算子的数学计算原理分析

假设输入张量的形状为 $[S, B, H]$，对于其中某一个特定的样本 $b$ 和特定的 Token 位置 $s$，其对应的特征向量为 $x_{s,b} \in \mathbb{R}^H$。

#### 1. LayerNormalization（层归一化 / RMSNorm）

**计算只发生在 $H$ 维（隐藏层特征维度），不同 $s$ 之间不发生统计量共享。**

LayerNorm 的公式为：

$$\mu_{s,b} = \frac{1}{H} \sum_{h=1}^{H} x_{s,b,h}, \qquad \sigma_{s,b}^{2} = \frac{1}{H} \sum_{h=1}^{H} \left( x_{s,b,h} - \mu_{s,b} \right)^{2}$$

$$\text{LN}(x_{s,b}) = \frac{x_{s,b} - \mu_{s,b}}{\sqrt{\sigma_{s,b}^{2} + \epsilon}} \odot \gamma + \beta$$

- **关键点**：均值 $\mu_{s,b}$ 和方差 $\sigma_{s,b}^{2}$ 的求和汇总（Reduction）**完全是在单个 Token 的 $H$ 个 Channel 内部完成的**。
- **对比 BatchNorm**：BatchNorm 是沿着 Batch 维和 Sequence 维做归一化（依赖其他 Token 的统计量）；而 LayerNorm 对每一个 Token 独立计算一套均值和方差。因此 Token $s_1$ 的归一化结果完全不依赖 Token $s_2$。

#### 2. Dropout（随机失活）

**计算是按元素（Element-wise）独立采样的。**

Dropout 的公式为：

$$\text{Dropout}(x_{s,b,h}) = \frac{1}{1-p} \cdot x_{s,b,h} \cdot m_{s,b,h}$$

其中 $m_{s,b,h} \sim \text{Bernoulli}(1-p)$ 是独立同分布采样的 0/1 掩码（Mask）。

- **关键点**：每个位置的元素是否被丢弃，仅取决于当前位置随机生成的掩码比特，其他 Token 的值和掩码对其毫无影响。

#### 3. 残差连接（Residual Connection）

**计算是逐位置的向量直接相加。**

残差连接的公式为：

$$y_{s,b} = x_{s,b} + \text{SubLayer}(x)_{s,b}$$

- **关键点**：向量加法满足按元素独立性，即第 $s$ 个 Token 的输出，仅由第 $s$ 个 Token 的输入与该位置经过子层（如 Attention 或 FFN）计算后的输出相加得到，无需跨位置寻址或聚合。

### 二、 Transformer 的架构设计哲学：职责分离

从模型设计的角度看，Transformer 将**跨 Token 交互**与**单 Token 特征演化**进行了非常清晰的模块化拆分：

- **跨 Token 关联与信息融合（跨 $S$ 维）**：**完全交给 Self-Attention 机制**（以及部分架构中的 MoE Token 路由、全局/局部卷积）。只有 Attention 需要计算 $QK^{\top}$ 矩阵，让 Token $s_1$ 去关注并聚合 Token $s_2$ 的特征。
- **单 Token 内部特征映射与正则化（单 $H$ 维）**：**交由 LayerNorm、MLP（FFN）、Dropout 和残差连接**。它们的作用是在不打乱上下文关联的前提下，对每一个 Token 独立的 $H$ 维语义向量进行高维投影、非线性激活、数值稳定化和防止过拟合。

这种"职责分离"的设计使得 Transformer 模块高度规范化：除了 Attention 之外的所有层，都可以被视为对序列中每个 Token 施加的同等映射函数。

### 三、 对分布式并行（序列并行 SP）的工程价值

正因为这三个算子在 $S$ 维度上具备**无跨位置依赖（No Cross-Token Dependency）**的特性，才使得 Megatron / MindSpeed 的**序列并行（Sequence Parallelism）**成为可能：

- 当我们把序列 $S$ 均匀切分到 $TP$ 张卡上，每张卡拿到一个长度为 $S/TP$ 的子张量时：
  - **LayerNorm**：每张卡在本地对自己的 $S/TP$ 个 Token 分别求 $H$ 维均值方差并归一化，**结果与单卡算全量 $S$ 完全一致**。
  - **Dropout**：每张卡在本地生成属于自己 $S/TP$ 区域的掩码，**无需跨卡同步随机数**。
  - **残差相加**：每张卡在本地做本地 Tensor 的加法，**零通信开销**。

这就是为什么 SP 可以把非 MatMul 区域沿 $S$ 维切分来压榨显存，而不需要引入任何额外的卡间通信。

---

<a id="sec08"></a>
## 八、 序列并行（SP）是标配吗

是的。在现今基于 Megatron-LM 与昇腾 MindSpeed 的大模型训练工程中，**只要开启了张量并行（$TP > 1$），序列并行（SP）几乎就是毫无争议的"标配"（De Facto Standard）**。

在实际生产环境的训练脚本中，只要设置了 `--tensor-model-parallel-size > 1`，工程师必定会同时带上 `--sequence-parallel` 参数。

### 为什么 SP 能成为"免费的午餐"与行业标配？

在分布式训练的各种优化技术中，绝大多数都是"以时间换空间"或"以通信换显存"。但 Megatron SP 属于极少数**毫无副作用的硬核优化**，总结为"三个零"：

1. **零额外通信量（Zero Extra Communication）**
   如前所述，将原本 TP 的 1 次 `AllReduce` 拆解为 1 次 `ReduceScatter` 和 1 次 `AllGather`，总通信数据量在数学上**完全相等**（都等于 $2 \times \dfrac{TP - 1}{TP} \times M$）。
2. **零精度损失（Zero Accuracy Loss）**
   由于 LayerNorm、Dropout、残差连接都是按 Token 逐点独立计算的，切分到各卡上计算与单卡全量计算的数值结果完全一致。
3. **零额外开发/侵入成本**
   在 MindSpeed 与 Megatron-LM 中，SP 已经深度集成进框架底层的 `ColumnParallelLinear` 和 `RowParallelLinear` 算子中，开启只需一行命令行参数。

**正是因为 SP 在"不增加通信传输时间、不降低模型精度"的前提下，直接削减了 TP 组内非矩阵乘法区域 80%+ 的激活值显存，因此没有任何理由关闭它。**

### 唯一不开启 SP 的特例场景

虽然 SP 是标配，但在以下特殊情况下它不会被使用或失效：

1. **未开启张量并行（$TP = 1$）**
   SP 是寄生在 TP 组内部的逻辑。如果模型采用单卡运行，或者仅使用数据并行（DP）、流水线并行（PP）或上下文并行（CP）而将 TP 设为 1，此时序列没有被 TP 组切分的基础，SP 自动失效。
2. **极少数旧版本自定义算子的兼容性问题**
   如果用户自己在 Megatron/MindSpeed 外层挂载了未经 SP 适配的第三方自定义单卡 C++/CUDA/Ascend C 算子（没有正确处理输入从 $[S, B, H]$ 变成 $[S/TP, B, H]$ 的情况），可能会短暂关闭 SP 进行调试，但这属于工程避坑的临时手段。
3. **纯推理场景（Inference）**
   推理时不存在反向传播，不需要保存 LayerNorm/Dropout 的中间激活值用于求导，显存瓶颈主要在 KV Cache 上。因此推理加速框架（如 vLLM、MindIE）使用的是针对 KV Cache 的 PagedAttention 和 CP/TP，而不是训练意义上的 Megatron SP。

### 总结

在 MindSpeed / Megatron-LM 体系下：

- **$TP > 1$ + 开启 SP** = 现代大模型训练的默认标准姿态。
- **超长文本场景** = TP 组内开 SP + 跨节点组开 CP（Context Parallel）组合拳。

---

<a id="sec09"></a>
## 九、 流水线并行（PP）原理

流水线并行（Pipeline Parallelism, 简称 PP）是一种**模型层间纵向切分**的分布式并行技术。它将深度神经网络按层（Layer）依次切分到不同的卡或节点（称为 Stage）上，并将一个 Batch 的数据拆分为多个**微批次（Micro-Batch）**，以流水线重叠（Overlapping）的方式实现跨卡并行训练。

### 一、 核心切分逻辑与数据流

流水线并行将模型的深度（Depth）维度拆开，并结合数据切分：

1. **模型层切分**：假设模型共有 32 层 Transformer Block，设置 $PP = 4$，则 Rank 0（Stage 0）负责 1~8 层，Rank 1（Stage 1）负责 9~16 层，以此类推。
2. **微批次拆分（Micro-Batching）**：如果一个训练步（Step）的总 Batch Size 为 $B$，流水线会将其拆分为 $m$ 个更小的 Micro-Batch（大小为 $b = B / m$）。
3. **流水化传递**：Stage 0 算完 Micro-Batch 1 的前向传播后，立即将激活值（Activation）点对点发送给 Stage 1，同时 Stage 0 接着计算 Micro-Batch 2。数据像工厂装配线一样依次向下游传输。

### 二、 流水线气泡（Pipeline Bubble）

由于下游 Stage 必须等待上游 Stage 算完第一个 Micro-Batch 后才能启动，且上游 Stage 必须等待下游反向传播完成后才能拿到梯度，这就不可避免地产生了硬件空闲等待区间，即**流水线气泡（Bubble）**。

传统的流水线气泡占比（Bubble Ratio）公式为：

$$F_{\text{bubble}} = \frac{PP - 1}{m + PP - 1}$$

其中 $PP$ 为流水线级数（Stage 数量），$m$ 为 Micro-Batch 的数量。

- **优化方向**：可以通过增大 $m$（增加微批次数量）来稀释气泡占比；但 $m$ 过大会导致整体 Batch Size 变大，影响模型收敛。

### 三、 三大核心调度算法演进

为了控制气泡占比并降低激活值显存占用，流水线调度算法经历了三次关键演进：

#### 1. GPipe（F-then-B 模式）

- **调度逻辑**：先连续完成所有 $m$ 个 Micro-Batch 的前向传播（Forward），再连续完成所有 $m$ 个 Micro-Batch 的反向传播（Backward）。
- **缺点**：反向传播开始之前，卡上需要**暂存全部 $m$ 个 Micro-Batch 的中间激活值**，显存开销随着 $m$ 呈线性增长，极易导致显存溢出（OOM）。

#### 2. 1F1B（One Forward, One Backward）

Megatron-LM 和 MindSpeed 中最常用的标准调度策略：

- **调度逻辑**：
  1. **Warmup 阶段**：流水线先进行有限个 Micro-Batch 的前向计算（填充流水线）。
  2. **Steady 阶段**：进入稳定状态后，**每计算 1 个 Micro-Batch 的前向，紧接着计算 1 个 Micro-Batch 的反向**（1 Forward, 1 Backward）。
- **核心收益**：将卡上存留的最大激活值 Micro-Batch 数量锁死在 $\sim PP$ 级别的常数，**显存占用与总微批次量 $m$ 解耦**，大幅释放显存空间。

#### 3. Interleaved 1F1B（虚拟流水线 VPP, Virtual Pipeline）

Megatron-LM 提出的高级交错调度算法：

- **调度逻辑**：将单张物理卡（Rank）切分为 $v$ 个非连续的虚拟阶段（Virtual Stage）。例如 $PP = 4, v = 2$ 时，Rank 0 同时管理 Stage 0（1~4 层）和 Stage 4（17~20 层）。
- **核心收益**：Micro-Batch 在卡间交替跳跃传输，将流水线气泡尺寸缩小为原来的 $1/v$：

$$F_{\text{bubble}}^{\text{VPP}} \approx \frac{PP - 1}{v \cdot m}$$

在仅增加微量点对点通信成本的情况下，显著提升了设备利用率（TFLOPS）。

### 四、 PP 的通信特征与部署优势

- **点对点通信（P2P）**：不同于张量并行（TP）在每层都需要调用全卡同步算子（`AllReduce`/`AllGather`），PP 仅在相邻 Stage 的边界层之间触发 `Send` / `Recv` 点对点通信。
- **跨节点部署友好**：PP 传递的数据量仅为激活值/梯度张量（形状为 $[S, B, H]$），通信频次极低（每个 Micro-Batch 仅通信一次）。因此，**PP 通常用于跨机/跨节点（Inter-Node）的集群扩展**，可以使用低带宽的跨节点网络（如普通 InfiniBand 或 RoCE）连接。

### 五、 总结：PP 的显存与计算收益

- **模型权重与优化器显存**：直接降为单卡的 $1/PP$。
- **激活值显存**：结合 1F1B 调度后，峰值激活显存被限制在 $PP$ 个 Micro-Batch 规模。
- **硬件适用性**：与 TP（节点内/高带宽）和 DP/ZeRO（全局数据并行）互补，构成千亿/万亿大模型训练中不可或缺的纵向扩展维度。

---

<a id="sec10"></a>
## 十、 流水线气泡占比公式原理

流水线气泡占比公式的推导核心在于**比对整个集群在一次迭代（Iteration）中的"实际有效计算时间"与"理论最大总可用时间"**。

### 一、 核心概念与假设定义

设一个训练 Step 包含：

- $PP$：流水线级数（Stage 数量，即物理卡数/节点数）。
- $m$：切分的 Micro-Batch 数量。
- $t_f$：1 个 Micro-Batch 在 1 个 Stage 上完成前向传播（Forward）所需的时间。
- $t_b$：1 个 Micro-Batch 在 1 个 Stage 上完成反向传播（Backward）所需的时间。

为了推导方便，定义 1 个 Micro-Batch 的完整计算周期时间为：

$$t_{\text{step}} = t_f + t_b$$

### 二、 流水线总执行时间 $T_{\text{total}}$ 推导

观察从 **Stage 0 开始计算第 1 个 Micro-Batch** 到 **Stage 0 完成最后一个 Micro-Batch 反向计算**的完整过程：

1. **流水线填充延迟（Warmup Phase）**：
   Stage 0 算完第一个 Micro-Batch 的前向后，传给 Stage 1，直到传递到最后一个 Stage（$PP - 1$）。数据流过 $PP - 1$ 个 Stage 需要经过 $PP - 1$ 次前向传递延迟，时间消耗为 $(PP - 1) \cdot t_f$。
2. **稳态计算（Steady State Phase）**：
   最后一个 Stage（$PP - 1$）收到数据后，连续完成所有 $m$ 个 Micro-Batch 的前向与反向计算。该 Stage 的总工作时间为 $m \cdot (t_f + t_b) = m \cdot t_{\text{step}}$。
3. **流水线排空延迟（Drain Phase）**：
   当 Stage $PP - 1$ 完成最后一个 Micro-Batch 的反向传播后，梯度需要反向传递回 Stage 0。跨越 $PP - 1$ 个 Stage 返回需要经过 $PP - 1$ 次反向传递延迟，时间消耗为 $(PP - 1) \cdot t_b$。

将这三阶段累加，得到单次 Iteration 的**总运行时间 $T_{\text{total}}$**：

$$T_{\text{total}} = (PP - 1) \cdot t_f + m \cdot (t_f + t_b) + (PP - 1) \cdot t_b$$

$$T_{\text{total}} = (m + PP - 1) \cdot (t_f + t_b) = (m + PP - 1) \cdot t_{\text{step}}$$

### 三、 气泡占比 $F_{\text{bubble}}$ 推导

有了单次 Iteration 的总运行时间，即可计算整个集群的资源利用率与气泡占比：

1. **集群总可用时间容量（Total Hardware Capacity）**：$PP$ 张卡在 $T_{\text{total}}$ 时间内能够提供的最大计算容量为：

$$\text{Capacity}_{\text{total}} = PP \cdot T_{\text{total}} = PP \cdot (m + PP - 1) \cdot t_{\text{step}}$$

2. **集群实际有效计算时间（Actual Work Done）**：所有 $PP$ 张卡都需要对这 $m$ 个 Micro-Batch 进行一次完整的 Forward 和 Backward 计算，实际有效计算时间为：

$$\text{Work}_{\text{actual}} = PP \cdot m \cdot (t_f + t_b) = PP \cdot m \cdot t_{\text{step}}$$

3. **集群总气泡时间（Total Bubble Time）**：集群在填充和排空阶段闲置的总时间即为可用容量与实际工作的差值：

$$\text{Time}_{\text{bubble}} = \text{Capacity}_{\text{total}} - \text{Work}_{\text{actual}} = PP \cdot (PP - 1) \cdot t_{\text{step}}$$

4. **气泡占比公式（Bubble Ratio $F_{\text{bubble}}$）**：气泡时间占总可用时间的比例为：

$$F_{\text{bubble}} = \frac{\text{Time}_{\text{bubble}}}{\text{Capacity}_{\text{total}}} = \frac{PP \cdot (PP - 1) \cdot t_{\text{step}}}{PP \cdot (m + PP - 1) \cdot t_{\text{step}}} = \frac{PP - 1}{m + PP - 1}$$

### 四、 公式表达的物理意义

- **分子 $(PP - 1)$**：代表流水线在**建立阶段**与**结束阶段**所必然付出的"阶梯排空与填充"延迟项。流水线级数越多（$PP$ 越大），等待数据透传到末端的时间就越长。
- **分母 $(m + PP - 1)$**：代表包含了等待开销与有效工作在内的**总有效时间步数**。
- **稀释效应**：只要**增加 Micro-Batch 数量 $m$**，分母就会被迅速拉大。当 $m \gg PP$ 时，$F_{\text{bubble}} \to 0$，气泡占比趋近于 0。

### 五、 常见流水线调优对气泡公式的影响

| 流水线调度算法 | 气泡占比公式 $F_{\text{bubble}}$ | 说明 |
| --- | --- | --- |
| **GPipe（F-then-B）** | $\dfrac{PP - 1}{m + PP - 1}$ | 标准公式，但中间激活值显存随着 $m$ 线性增加。 |
| **1F1B** | $\dfrac{PP - 1}{m + PP - 1}$ | **气泡占比与 GPipe 一致**，但通过交叉前向反向，将显存占用限制在常数级。 |
| **Interleaved 1F1B（VPP）** | $\approx \dfrac{PP - 1}{v \cdot m}$ | 每张卡拆分为 $v$ 个虚拟 Stage，将单个气泡块拆碎，使总气泡占比缩小约 $v$ 倍。 |

---

<a id="sec11"></a>
## 十一、 PP 与 VPP 的命令行参数与调优策略

在 **Megatron-LM** 和昇腾 **MindSpeed** 框架中，流水线并行（Pipeline Parallelism, PP）与虚拟流水线（Virtual Pipeline Parallelism, VPP / Interleaved 1F1B）是在大规模 NPU/GPU 集群上横向扩展模型层数、压榨吞吐效率的关键技术。

### 一、 核心命令行参数解析

在启动脚本中，控制 PP 和 VPP 行为的核心参数如下：

#### 1. 并行维度控制参数

| 参数名称 | 简写/类型 | 说明与典型值 |
| --- | --- | --- |
| `--pipeline-model-parallel-size` | PP (int) | 流水线阶段（Stage）数量。即模型在深度方向切分为多少份。 |
| `--num-layers-per-virtual-pipeline-stage` | $L_v$ (int) | **开启 VPP 的核心开关**。指定**每个虚拟 Stage 包含的 Transformer 层数**。如果不设置，则默认使用传统 1F1B 流水线。 |
| `--tensor-model-parallel-size` | TP (int) | 张量并行大小（通常在节点内高带宽互联）。 |
| `--sequence-parallel` | Flag | 开启序列并行（与 TP 配合，建议必开）。 |

#### 2. 批次与流水线气泡控制参数

| 参数名称 | 类型 | 说明 |
| --- | --- | --- |
| `--global-batch-size` | GBS (int) | 单次训练 Step 的全局总批次大小（以 Token 数或 Sentence 数计）。 |
| `--micro-batch-size` | MBS (int) | 单张卡每次前向计算处理的微批次大小。 |

> **隐式计算衍生值**：
>
> - 数据并行度：$\text{DP} = \dfrac{\text{World\_Size}}{\text{TP} \times \text{PP} \times \text{CP} \times \text{EP}}$
> - 每个流水线步中的 Micro-Batch 数量：$m = \dfrac{\text{GBS}}{\text{MBS} \times \text{DP}}$

#### 3. 显存与重算控制参数

| 参数名称 | 选项 | 说明 |
| --- | --- | --- |
| `--recompute-granularity` | `selective` / `full` | 重算粒度：选择性重算（Selective）或全量重算（Full）。 |
| `--recompute-method` | `block` / `uniform` | 全量重算时的分配方式。 |
| `--recompute-num-layers` | int | 指定重算的层数。 |

### 二、 参数约束与数学硬性法则

配置 VPP（$L_v$）时，必须严格遵守以下**整除性几何约束**，否则框架在初始化模型切片时会直接断言报错（`AssertionError`）：

#### 1. 层数整除性法则

假设模型总层数为 $N$（如 LLaMA-70B 的 $N = 80$），流水线级数为 PP，每个虚拟 Stage 的层数为 $L_v$：

- **单卡虚拟块数量（$v$）计算**：

$$v = \frac{N}{PP \times L_v}$$

- **硬性约束**：总层数 $N$ **必须能被 $(PP \times L_v)$ 整除**：

$$N \bmod (PP \times L_v) == 0$$

> **示例**：对于 $N = 80$ 的模型，若设置 $PP = 4$，则 $PP \times L_v$ 必须能整除 80：
>
> - 设置 $L_v = 5 \Rightarrow v = \dfrac{80}{4 \times 5} = 4$（每张物理卡管理 4 个虚拟 Stage，每个包含 5 层）。
> - 设置 $L_v = 2 \Rightarrow v = \dfrac{80}{4 \times 2} = 10$（每张物理卡管理 10 个虚拟 Stage，每个包含 2 层）。

#### 2. 气泡率（Bubble Ratio）与 $m$ 的平衡关系

开启 VPP 后的气泡率公式为：

$$F_{\text{bubble}}^{\text{VPP}} \approx \frac{PP - 1}{v \cdot m} = \frac{(PP - 1) \cdot PP \cdot L_v}{N \cdot m}$$

- **结论**：增加 $v$（即减小 $L_v$）或增加 $m$（即增大 GBS / 减小 MBS），都可以**按比例压缩流水线气泡**。

### 三、 VPP（Interleaved 1F1B）数据流与映射逻辑

当配置了 `--num-layers-per-virtual-pipeline-stage $L_v$` 后，框架会将原本连续的 Stage 交叉拆解：

假设 $N = 16$, $PP = 4$, $L_v = 1 \Rightarrow v = 16 / (4 \times 1) = 4$：

- **Chunk 编号**：模型被拆分为 16 个 Chunk（Chunk 0 ~ Chunk 15）。
- **Rank 分配映射**：
  - **Rank 0（Stage 0）**：持有 Chunk 0, Chunk 4, Chunk 8, Chunk 12（共 4 层）
  - **Rank 1（Stage 1）**：持有 Chunk 1, Chunk 5, Chunk 9, Chunk 13（共 4 层）
  - **Rank 2（Stage 2）**：持有 Chunk 2, Chunk 6, Chunk 10, Chunk 14（共 4 层）
  - **Rank 3（Stage 3）**：持有 Chunk 3, Chunk 7, Chunk 11, Chunk 15（共 4 层）

数据流向前传播时，前向计算会在 Rank 0 → Rank 1 → Rank 2 → Rank 3 → Rank 0 → Rank 1 ... 之间**循环跳跃传递**。

### 四、 实战调优策略与最佳实践

#### 策略 1：硬件拓扑与并行维度的映射规则

在昇腾 NPU 集群（如昇腾 910B/910C 组网）上，通信带宽架构为：**片间/板内 HCCS 高速互联 > 跨节点 RoCE / InfiniBand 网络**。

- **TP**：限制在**单节点内**（通常 $TP \le 8$），充分利用节点内的高带宽 HCCS 拓扑。
- **PP**：部署在**跨节点间**。由于 PP 仅在 Stage 边界通过点对点（P2P）传递 $[S/TP, B, H]$ 张量，通信量极小，非常适合跨 RoCE 网络扩展。
- **VPP（$v$）**：**建议首选 $v = 2$ 或 $v = 4$**。
  - $v = 1$（不开启 VPP）：气泡较大，但通信频次最低。
  - $v = 2$：气泡直接**减半**，P2P 通信次数翻倍。在大规模训练中通常是**综合吞吐最高、性价比最佳的平衡点**。
  - $v \ge 4$：气泡进一步变小，但频繁的跨节点 P2P 小包通信可能引发通信延迟开销（Communication Overhead），抵消气泡收益。

#### 策略 2：通过 $m$ / PP 比例控制气泡

为了将气泡率控制在 **10% 以下**：

- 尽量保持每一个流水线步中的 Micro-Batch 数量 $m \ge 4 \times PP$。
- 如果受限于总 Batch Size（GBS），导致 $m$ 较小（例如 $m = 2 \times PP$），此时**强烈建议开启 VPP（$v = 2$ 或 $v = 4$）**来强行压缩气泡。

#### 策略 3：解决首尾 Stage 显存倾斜（Embedding 与 Head 均衡）

由于 Stage 0 保存 Embedding 层，最后一个 Stage 保存 LM Head 和 Loss 计算，首尾 Stage 容易发生 OOM：

1. **Standalone Embedding 机制**：通过参数将 Embedding 层从 Stage 0 中剥离，独立分配或分散（若框架版本支持）。
2. **结合选择性重算（Selective Recomputation）**：不要盲目全量重算。开启 `--recompute-granularity selective` 仅重算 Attention 中的 FlashAttention / Softmax 部分，能以不到 2% 的计算开销释放 60%+ 的激活显存，避免首尾 Stage 显存溢出。

#### 策略 4：MindSpeed 专属通信重叠加速（P2P Communication Overlap）

在 MindSpeed 运行昇腾 CANN 架构时，可以开启 P2P 通信与计算重叠，隐藏跨节点传输延迟：

```bash
# 开启 P2P 异步通信重叠 (MindSpeed 特有优化)
--overlap-p2p-communication
```

### 五、 典型生产环境 Shell 配置脚本示例

以下为一个 80 层 70B 级别大模型在 32 卡昇腾 NPU 集群（4 节点 × 8 卡）上，采用 **TP=4, PP=4, DP=2, VPP（$v=2$）** 的典型配置参考：

```bash
#!/bin/bash
# 硬件拓扑与分布式配置
WORLD_SIZE=32
TP_SIZE=4
PP_SIZE=4
CP_SIZE=1
DP_SIZE=2

# 模型参数 (80 层 Transformer)
NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_HEADS=64

# VPP 配置: 每卡管理 v=2 个虚拟 Stage, 每个虚拟 Stage 包含 10 层
# 满足约束: NUM_LAYERS (80) % (PP_SIZE (4) * LAYERS_PER_VIRTUAL_STAGE (10)) == 0
LAYERS_PER_VIRTUAL_STAGE=10

# 批次配置
# m = GBS / (MBS * DP) = 128 / (2 * 2) = 32 个 Micro-Batches
# 气泡率 F_bubble ≈ (4 - 1) / (2 * 32) ≈ 4.69% (极低)
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=2

# 命令行运行参数
EXEC_ARGS="
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers-per-virtual-pipeline-stage ${LAYERS_PER_VIRTUAL_STAGE} \
    --sequence-parallel \
    \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_HEADS} \
    \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    \
    --recompute-granularity selective \
    --use-flash-attn \
    --overlap-p2p-communication \
    --use-distributed-optimizer \
"
python3 pretrain_gpt.py $EXEC_ARGS
```

---

<a id="sec12"></a>
## 十二、 上下文并行（CP）原理

**上下文并行（Context Parallelism, 简称 CP）** 是专门针对超长文本（如 32k、128k、1M+ Tokens）训练与推理而设计的序列切分并行技术。

在传统并行方案中，即使开启了张量并行（TP）与序列并行（SP），**Self-Attention 模块内部依然需要拉回全量序列 $S$ 进行计算**。随着文本长度 $S$ 呈二次方（$O(S^2)$）增长，单卡的显存与计算能力会迅速崩溃。

CP 的核心突破在于：**它将 Self-Attention 计算本身沿着序列维度 $S$ 完全切分到不同的节点/卡上，使每张卡真正只承担 $1/CP$ 的序列计算与显存开销。**

### 一、 为什么有了序列并行（SP），还需要上下文并行（CP）？

很多工程师容易混淆 SP 与 CP，二者的本质差异在于 Attention 模块内部的处理方式：

| 维度 | Megatron 序列并行（SP） | 上下文并行（CP） |
| --- | --- | --- |
| **作用组界限** | 局限于 **TP 组**内部（通常为节点内 $TP \le 8$） | 作用于 **CP 组**（可跨节点扩展至数十/上百卡） |
| **Non-Attention 区域** | 沿 $S$ 切分（每卡 $S/TP$） | 沿 $S$ 切分（每卡 $S/CP$） |
| **Attention 内部机制** | **强行拉回全量序列**：调用 `AllGather` 收集全量 $S$ 的 K、V，单卡计算完整 $S \times S$ 注意力矩阵 | **序列彻底切分**：通过算法拆解，单卡仅保留 $S/CP$ 的 Query，**无需在任何时刻拼出全量 $S$ 的注意力矩阵** |
| **显存瓶颈** | 无法解决 Attention $O(S^2)$ 的显存与计算爆炸 | **显存与计算量随 CP 线性降低**（降为 $1/CP$） |

### 二、 CP 的两大主流实现范式

目前在 Megatron-LM、MindSpeed 以及 DeepSpeed 中，CP 主要有两种核心实现方案：**DeepSpeed-Ulysses** 与 **Ring-Attention**。

#### 范式 1：DeepSpeed-Ulysses（基于 All-to-All 转置的 CP）

DeepSpeed-Ulysses 的设计非常直观优雅，它利用两次 **`All-to-All`** 通信算子，在"序列维度切分"与"注意力头维度切分"之间进行转置切换。

##### 1. 数据流与维度变换过程

假设序列长度为 $S$，隐层维度为 $H$，注意力头数为 $N$，单头维度为 $D = H / N$，CP 组大小为 $P$：

```
输入序列张量 [S/P, B, N, D] (沿 S 切分)
          │
          ▼  【第 1 次 All-to-All 通信】
   按 S 维聚合全量序列 S，同时按 Head 维切分 N/P
          │
          ▼
转置后张量 [S, B, N/P, D] (沿 Head 维切分)
          │
          ▼  【本地 Attention 计算】
各卡使用标准的 FlashAttention，独立计算自己负责的 N/P 个 Head
（计算的是完整的 S 长度，但 Head 数只有 N/P）
          │
          ▼
输出张量 [S, B, N/P, D]
          │
          ▼  【第 2 次 All-to-All 通信】
按 Head 维聚合全量 N，同时按 S 维重新切分 S/P
          │
          ▼
还原后张量 [S/P, B, N, D] (沿 S 切分)
```

##### 2. 优缺点分析

- **优点**：极易集成，本地可以直接无缝调用昇腾 NPU 或 CUDA 上的标准 FlashAttention 算子。
- **致命局限**：**受限于 Head 数量**。CP 组大小 $P$ 不能超过注意力头数 $N$（$P \le N$）。尤其在现代大模型普遍采用 **GQA（Grouped Query Attention）** 架构下，KV Head 数量 $N_{kv}$ 很小（如 4 或 8），导致 Ulysses 的 CP 并行度很难继续扩大。

#### 范式 2：Ring-Attention / Megatron-CP（基于环形 P2P 传输的 CP）

为了突破 Head 数量的限制，UC Berkeley 提出了 **Ring-Attention**（Megatron-LM 和 MindSpeed 默认的 CP 方案）。它的核心在于**结合 Online Softmax（FlashAttention 的数学原理），将 K、V 张量像接力棒一样在卡间环形传递**。

##### 1. 核心运行逻辑

在 Ring-Attention 中，每张卡始终持有属于自己的 $S/CP$ 长度的 **$Q_i$（Query 留在本地不动）**，而 $K_i, V_i$ 则在 CP 环（Ring）上传输：

```
[Card 0] (持有 Q0, K0, V0) ──P2P Send K0,V0──> [Card 1] (持有 Q1, K1, V1)
   ▲                                               │
   │                                               │ P2P Send K1,V1
P2P Send K3,V3                                     ▼
[Card 3] (持有 Q3, K3, V3) <──P2P Send K2,V2── [Card 2] (持有 Q2, K2, V2)
```

- **Step 0**：每张卡利用本地的 $Q_i$ 和本地的 $K_i, V_i$，计算局部 Attention Block，并保存当前局部 Softmax 的分母统计量（Max 与 Sum）。
- **Step 1**：每张卡通过异步 P2P 通信，将自己的 K、V 块发送给下一个邻居 Rank，同时接收上一个 Rank 发来的 $K', V'$ 块。
- **Step 2**：每张卡利用本地不变的 $Q_i$ 与刚收到的 $K', V'$ 计算下一块 Attention，并利用 **Online Softmax 分区更新公式**，动态修正并累加当前的 Output 与 Rescale 因子。
- **Step $CP - 1$**：当 K、V 块在环上顺时针旋转一圈（经过 $CP$ 次迭代）后，所有卡均完成了全局 $Q \cdot K^{\top}$ 的计算，得到最终精确的 Attention 输出。

##### 2. 通信与计算掩盖（Ring CP Overlap）

Ring-Attention 的极致工程优化在于**通信与计算重叠**：在第 $k$ 步计算 $Q_i \cdot K_{i-k}^{\top}$ 的同时，硬件的通信引擎（如昇腾 HCCL / NVLink）在后台**异步传输第 $k+1$ 步所需的 $K_{i-k-1}, V_{i-k-1}$**。

由于单次 K、V 块的 P2P 传输量较小，**P2P 通信时间可以被 100% 隐藏在 Cube 核心的 Attention 计算之后**，实现近乎零通信损耗的线性加速。

##### 3. 优缺点分析

- **优点**：**完全不受 Head 数量限制**（即使 GQA 的 KV Head 为 1 也能正常切分）；配合 FlashAttention 可做到极高的通信重叠效率。
- **缺点**：需要底层 FlashAttention 内核支持传入并更新 Online Softmax 中间状态（MindSpeed 与 CANN 已对此进行了深度定制适配）。

### 三、 混合上下文并行（Hybrid CP: Ulysses + Ring）

在实际的千卡级大模型训练（如 MindSpeed 训练 128k 文本）中，通常会将两种 CP 结合使用，形成 **Hybrid CP** 方案：

- **节点内（Intra-node）**：采用 **Ulysses CP**。利用节点内板间 HCCS / NVLink 的超高带宽，高效执行 `All-to-All` 转置。
- **节点间（Inter-node）**：采用 **Ring CP**。利用跨节点 P2P 传输量小的特点，结合跨机 RoCE 网络进行环形掩盖传输。

这种组合既规避了 Ulysses 的 Head 数限制，又消除了 Ring-Attention 跨节点步骤过多的问题。

### 四、 总结：四大并行维度（TP / SP / PP / CP）协同图景

在大模型分布式训练体系中，四大并行维度的分工极其明确：

```
┌─────────────── 大模型分布式训练并行体系 ───────────────┐
                     │                                                      │
             ┌───────┴───────┐                              ┌───────────────┴───────┐
             │  模型切分维度  │                              │  数据/序列切分维度     │
             └───────┬───────┘                              └───────────────┬───────┘
                     │                                                      │
         ┌───────────┴───────────┐                              ┌───────────┴───────────┐
         ▼                       ▼                              ▼                       ▼
【张量并行 TP】          【流水线并行 PP】              【序列并行 SP】          【上下文并行 CP】
· 切分：单 Block 权重   · 切分：模型层数 (Layer)        · 切分：TP 组内 LN/Res   · 切分：Attention 序列 S
· 范围：节点内 (8卡)     · 范围：跨节点                 · 范围：TP 组内          · 范围：跨节点/CP组
· 目的：解决大权重显存   · 目的：扩展总层数/节点间通信   · 目的：消除 TP 显存冗余  · 目的：解决 100k+ 超长文本
```

---

<a id="sec13"></a>
## 十三、 CP 与 Hybrid CP 的命令行参数与计算规则

在昇腾 **MindSpeed** 框架中，上下文并行（Context Parallelism, CP）是解决 32k、128k 甚至 1M+ 超长文本训练显存与计算瓶颈的关键技术。由于长文本下 Attention 的计算与显存开销呈二次方（$O(S^2)$）爆炸，合理配置 CP、Ulysses 和 Ring-Attention 的混合参数至关重要。

### 一、 MindSpeed 核心 CP 命令行参数详解

在 MindSpeed 启动脚本中，控制 CP 行为的主要参数如下：

| 命令行参数 | 类型/选项 | 说明 |
| --- | --- | --- |
| `--context-parallel-size` | int（默认 `1`） | **总 CP 大小（$CP_{\text{total}}$）**。即上下文并行组的总卡数。 |
| `--context-parallel-algo` | string | **CP 算法类型**，可选：`ulysses_cp_algo`（纯 Ulysses 模式）、`megatron_cp_algo`（纯 Ring-Attention / Megatron-CP 模式）、`hybrid_cp_algo`（Ulysses + Ring 混合 CP 模式） |
| `--ulysses-degree-in-cp` | int（默认 `None`） | **仅在 `hybrid_cp_algo` 模式下生效**。指定 Hybrid CP 中 **Ulysses 维度的大小（$CP_{\text{ulysses}}$）**。 |
| `--use-cp-send-recv-overlap` | Flag | 开启 Ring-Attention 中 P2P 通信与计算的异步重叠（推荐必开）。 |

### 二、 三大 CP 模式及选型原则

#### 1. 纯 Ulysses 模式（`ulysses_cp_algo`）

- **原理**：基于 `All-to-All` 通信，在序列维度和 Head 维度之间做转置。
- **参数要求**：只需设置 `--context-parallel-size N` 和 `--context-parallel-algo ulysses_cp_algo`。
- **适用场景**：$CP \le 8$ 且模型 Head 数量较多的场景（如全量 Self-Attention 模型）。

#### 2. 纯 Ring 模式（`megatron_cp_algo`）

- **原理**：基于环形 P2P 通信，将 KV 块在卡间环形传递，结合 Online Softmax 动态更新。
- **参数要求**：只需设置 `--context-parallel-size N` 和 `--context-parallel-algo megatron_cp_algo`。
- **适用场景**：超长序列（如 128k+）、大 CP 规模（如 $CP \ge 16$）或 GQA/MQA（KV Head 数较少）的模型。

#### 3. 混合 CP 模式（`hybrid_cp_algo`）

- **原理**：**节点内使用 Ulysses**（吃满节点内高带宽 HCCS/NVLink），**节点间使用 Ring**（吃跨节点 RoCE 网络）。
- **参数要求**：
  - `--context-parallel-algo hybrid_cp_algo`
  - `--context-parallel-size $CP_{\text{total}}$`
  - `--ulysses-degree-in-cp $CP_{\text{ulysses}}$`
  - 隐式推导得 Ring 维度：

$$CP_{\text{ring}} = \frac{CP_{\text{total}}}{CP_{\text{ulysses}}}$$

### 三、 参数计算规则与四大硬性约束法则

在配置 CP 参数时，必须满足以下四个数学与硬件拓扑约束，否则会导致框架初始化报错（`AssertionError`）或性能大幅衰减：

#### 约束 1：总集群规模（World Size）整除法则

集群总卡数（World Size）在多维并行维度间满足：

$$\text{World Size} = TP \times PP \times CP_{\text{total}} \times DP \times EP$$

由此可推导出数据并行度 DP：

$$DP = \frac{\text{World Size}}{TP \times PP \times CP_{\text{total}} \times EP}$$

#### 约束 2：Attention Head 数量整除法则（Ulysses 维度硬约束）

在 Ulysses 算法中，`All-to-All` 通信需要将 Head 维度均匀切分到 $TP \times CP_{\text{ulysses}}$ 组内。

- **对于多头注意力（MHA）**：注意力头数 $N_{\text{head}}$ 必须能被 $(TP \times CP_{\text{ulysses}})$ 整除：

$$N_{\text{head}} \bmod (TP \times CP_{\text{ulysses}}) == 0$$

- **对于分组查询注意力（GQA / MQA）**：因为 Key/Value 的 Head 数量 $N_{\text{kv\_head}}$ 远小于 Query 的 Head 数量，为了避免 KV 在切分时出现不均，$N_{\text{kv\_head}}$ 必须被 $(TP \times CP_{\text{ulysses}})$ 整除：

$$N_{\text{kv\_head}} \bmod (TP \times CP_{\text{ulysses}}) == 0$$

> **关键推论**：如果一个模型的 $N_{\text{kv\_head}} = 8$（如 LLaMA-3-70B），在 $TP = 2$ 时，$CP_{\text{ulysses}}$ **最大只能设置为 4**（因为 $TP \times CP_{\text{ulysses}} = 8 \le 8$）。此时如果想进一步扩大 CP 到 16，就**必须采用 Hybrid CP**，将多出来的 4 倍放置到 Ring 维度（即 $CP_{\text{ring}} = 4$）。

#### 约束 3：序列长度 $S$ 整除法则

输入序列长度 $S$ 在切分后必须能被各个并行维度均匀切分：

- $S$ 必须能被 $(CP_{\text{total}} \times TP)$ 整除。
- 对于 Ring-Attention，由于存在 Causal Mask（因果掩码）三角矩阵的切块平衡算法，通常要求：

$$S \bmod (2 \times CP_{\text{total}} \times TP \times MBS) == 0$$

#### 约束 4：硬件拓扑与 HCCS 节点内绑定原则（Hybrid CP 最佳实践）

昇腾 910B/910C 服务器单节点通常包含 **8 张 NPU**，节点内通过 HCCS 超高速互联，跨节点通过 RoCE / InfiniBand 网络互联。

- **节点内并行空间**：单个节点内 8 张卡应优先给 TP 和 $CP_{\text{ulysses}}$ 瓜分：

$$TP \times CP_{\text{ulysses}} \le 8$$

- **最佳拓扑搭配**：
  - 若 $TP = 2$，则节点内 $CP_{\text{ulysses}}$ 建议设为 4（$2 \times 4 = 8$，正好吃满单节点 8 卡）。
  - 若 $TP = 4$，则节点内 $CP_{\text{ulysses}}$ 建议设为 2（$4 \times 2 = 8$）。
  - 跨节点的 CP 需求全量交给 $CP_{\text{ring}}$ 处理。

### 四、 实战配置与 Shell 脚本示例

#### 场景：在 64 卡昇腾 NPU 集群（8 节点 × 8 卡）上训练 128k 超长文本模型

- **模型参数**：$N_{\text{layer}} = 80$, $N_{\text{head}} = 64$, $N_{\text{kv\_head}} = 8$（GQA）。
- **并行组合**：$TP = 2$, $PP = 4$, $CP_{\text{total}} = 8$, $DP = 1$。
- **Hybrid CP 拆解**：
  - 单节点 8 卡限制 $TP \times CP_{\text{ulysses}} \le 8 \Rightarrow CP_{\text{ulysses}} = 4$（满足 $N_{\text{kv\_head}} = 8$ 可被 $2 \times 4$ 整除）。
  - 跨节点 $CP_{\text{ring}} = CP_{\text{total}} / CP_{\text{ulysses}} = 8 / 4 = 2$。

```bash
#!/bin/bash
# 硬件拓扑
WORLD_SIZE=64
NNODES=8
NPUS_PER_NODE=8

# 并行维度设置
TP_SIZE=2
PP_SIZE=4
CP_SIZE=8           # CP_total = 8
ULYSSES_DEGREE=4    # CP_ulysses = 4 (推导得 CP_ring = 2)

# 模型结构 (128k 文本)
SEQ_LENGTH=131072
GLOBAL_BATCH_SIZE=8
MICRO_BATCH_SIZE=1

EXEC_ARGS="
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    \
    --context-parallel-size ${CP_SIZE} \
    --context-parallel-algo hybrid_cp_algo \
    --ulysses-degree-in-cp ${ULYSSES_DEGREE} \
    --use-cp-send-recv-overlap \
    \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --num-query-groups 8 \
    \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    \
    --use-flash-attn \
    --sequence-parallel \
    --recompute-granularity selective \
    --use-distributed-optimizer \
"
torchrun --nproc_per_node=${NPUS_PER_NODE} --nnodes=${NNODES} pretrain_gpt.py $EXEC_ARGS
```

### 五、 调优与避坑总结

1. **GQA 报 `AssertionError`**：如果使用的是 DeepSeek 或 LLaMA-3 等 GQA 模型，且发生了 CP 切分报错，第一时间检查 $N_{\text{kv\_head}} \bmod (TP \times CP_{\text{ulysses}}) == 0$。如果超界，请**降低 `ulysses-degree-in-cp`，并将数值转移至 $CP_{\text{ring}}$**。
2. **流水线气泡与 CP**：CP 主要是切分序列 $S$，不增加流水线 Stage。因此 CP 变大不会放大流水线气泡（PP Bubble），非常适合与小 PP 结合扩展长文本。
3. **通信重叠标记**：务必带上 `--use-cp-send-recv-overlap`，MindSpeed 会调用昇腾 CANN 底层的异步 HCCL 通信算子，将 Ring CP 的 P2P 传输隐藏在 FlashAttention 的计算背后。

---

<a id="sec14"></a>
## 十四、 专家并行（EP）原理

**专家并行（Expert Parallelism, EP）** 是大语言模型从单体密集模型（Dense Model）向稀疏混合专家模型（Mixture-of-Experts, MoE）演进过程中不可或缺的核心并行技术。

### 一、 为什么需要专家并行（EP）？

在传统的 Dense 模型中，增加参数量意味着每一次前向传播都要计算所有参数。而 **MoE 模型** 将标准的 前馈网络（FFN）替换为包含多个独立"专家"（Experts）的 MoE 层（如 8、64 甚至 256 个专家），每个 Token 在运行时通过路由算法仅激活其中 $K$ 个专家（如 Top-1 或 Top-2）。

![EP 与 DP 结合时的 Token 路由与 All-to-All 通信流程，AI 生成](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3Nj6mXnZAwMSg4tY2fY-ymkgDRiVkebsE9l6G_mCXeapiJY8H8bCTUAM&s=10)

EP 与 DP 结合时的 Token 路由与 All-to-All 通信流程。**来源：Perplexity Research**

- **显存瓶颈**：随着专家数量暴增，单张 GPU 的显存无法放下所有专家的权重矩阵。
- **核心思想**：EP 是一种**模型并行（Model Parallelism）**策略，它**将不同的专家权重分布放置到集群中不同的计算节点/GPU 上**。每张 GPU 只存一部分专家，通过高效的网络通信在设备间路由 Token。

### 二、 专家并行的完整前向工作流

在结合数据并行（DP）或张量并行（TP）的典型拓扑中，MoE 层的执行逻辑可以划分为以下四个步骤：

```
[ 输入 Token ] (基于 DP/CP 分布在各卡)
       │
       ▼
1. [ Router 路由计算 ] ──► 计算每个 Token 的 Top-K 目标专家 ID
       │
       ▼
2. [ Dispatch 阶段 ]  ──► 第一次 All-to-All 通信（按专家分布重新打散分发 Token）
       │
       ▼
3. [ 局部专家计算 ]   ──► 各卡独立计算发送给本卡专家的 Token (FFN/SwiGLU)
       │
       ▼
4. [ Combine 阶段 ]   ──► 第二次 All-to-All 通信（将专家输出特征发回原始卡）
       │
       ▼
[ 加权聚合输出 ]      ──► 恢复原始 Token 顺序，进入下一层
```

#### 1. 路由计算（Router / Gating）

对于输入矩阵 $X \in \mathbb{R}^{S \times D}$（$S$ 为 Token 数，$D$ 为隐藏层维度），路由门控网络计算 Token 与各个专家的匹配得分：

$$G(x) = \text{Softmax}\left( \text{TopK}\left( x \cdot W_g, K \right) \right)$$

路由输出两项内容：

1. 目标专家的设备索引（应该发送给哪张卡）。
2. 对应的路由权重 $\alpha$（用于最后的加权合并）。

#### 2. 分发阶段（Dispatch: 第一次 `All-to-All` 通信）

卡 0 上的某个 Token 可能需要送往卡 3 上的专家，而卡 3 上的 Token 可能需要发往卡 0。

- 此时触发**第一次 `All-to-All` 集合通信**：集群中所有卡交换数据，根据路由决策把 Token 重新整理打包，发往放置对应目标专家的 GPU。

#### 3. 局部专家计算（Expert Execution）

当 GPU 收到跨卡发来的所有 Token 后，将其按专家组织为 Batch，并行调用本卡上的 FFN/SwiGLU 专家进行矩阵乘法计算：

$$Y_{\text{expert}} = \text{FFN}_i(X_{\text{routed}})$$

#### 4. 聚合阶段（Combine: 第二次 `All-to-All` 通信）

专家计算完毕后，必须将结果返回给发起请求的原始 GPU：

- 触发**第二次 `All-to-All` 集合通信**：把计算结果回发给原始卡。
- 原始卡收到来自不同专家的输出后，按照路由权重加权求和，汇总恢复出原始序列维度：

$$Y_o = \sum_{j \in \text{TopK}} \alpha_j \cdot Y_j$$

### 三、 核心通信原语：`All-to-All` 剖析

与大家熟知的 `All-Reduce`（归并求和）或 `All-Gather`（广播收集）不同，专家并行的核心是 **`All-to-All`（个性化全交换通信）**。

| 通信原语 | 特点 | 在 EP 中的作用 |
| --- | --- | --- |
| **`All-Gather`** | 每张卡将自己的数据广播给所有人，每个人收到的内容完全一致。 | 用于 TP 或 DP 阶段收集完整权重/特征。 |
| **`All-to-All`** | 卡 $i$ 向卡 $j$ 发送专门属于 $j$ 的数据块 $D_{i \to j}$。不同卡之间收发的数据切片互不相同。 | **Token 路由重组**：把处于不同卡上的 Token 准确送到目标专家卡上。 |

- **数据维度变化**：
  - 通信前：`[Cards, Tokens_per_Card, Hidden_Dim]`（按数据/序列切分）
  - 通信后：`[Experts_per_Card, Assigned_Tokens, Hidden_Dim]`（按专家切分）

### 四、 EP 面临的核心挑战与前沿优化

虽然 EP 能将参数量线性扩展至数百亿甚至万亿，但它在工程落地上面临两大痛点：

#### 1. 专家负载不均衡（Load Imbalance）

- **问题**：真实自然语言中存在"热门词"（如标点、高频词），会导致大量 Token 涌向少数"热门专家"（Hot Experts），造成部分 GPU 显存溢出与算力卡顿，而"冷门专家"所在的 GPU 则处于饥饿等待状态。
- **解决方案**：
  - **辅助损失（Auxiliary Loss）**：在损失函数中增加均衡惩罚项，强制 Router 均匀分布 Token。
  - **专家容量限制（Capacity Factor）**：限定每个专家一次最多处理的 Token 数量，溢出的 Token 被直接丢弃（Drop Token）或跳过 MoE 层。
  - **无辅助损失动态均衡（Auxiliary-Loss-Free Load Balancing）**：如 DeepSeek-V3 采用动态偏置项（Bias Adjustment）实时微调路由偏好，既保证了专家负载均衡，又避免了辅助损失损害模型效果。

#### 2. `All-to-All` 通信瓶颈（Communication Bottleneck）

- **问题**：在跨节点大集群中，跨机网络带宽（如 InfiniBand / RoCE）远低于节点内高带宽互联（如 NVLink / HCCS），频繁的 `All-to-All` 交换极其容易导致 GPU 严重等待网络传输。
- **解决方案**：
  - **通信与计算重叠（Overlap）**：将 Token 按照 Micro-batch 进一步拆细，利用流水线（如 DeepSeek DualPipe 或 Megatron 异步通信算子）做到**通信传输下一步 Token 的同时，Tensor Core 计算当前 Token 的专家 FFN**。
  - **拓扑感知路由（Topology-Aware Routing）**：优先将经常共同调用的专家放置在同一节点内，尽量降低跨节点的 Token 路由比例。

### 五、 专家并行与多维并行的组合拓扑（3D/4D Parallelism）

在实际的大模型训练与推理框架（如 Megatron-LM、DeepSeek-V3/R1、MindSpeed）中，EP 很少单独使用，而是与其他并行模式深度配合：

1. **Attention 层走 DP/CP，MoE 层走 EP**：
   自注意力机制（Attention）参数量小但 KV Cache 大，采用数据并行（DP）或上下文并行（CP）切分；到达 FFN（MoE）层时，通过 `All-to-All` 转换为 EP 模式进行计算，计算完成后再转换回 DP/CP。
2. **EP + TP（专家内张量并行）**：
   当单个 Expert 的参数量极高（例如单个专家大小超过 14B）时，可以在 EP 的基础上对**单个专家内部叠加 TP**（如 $EP = 8, TP = 2$），将单专家切分到多张卡上进一步降低单卡显存压力。

---

<a id="sec15"></a>
## 十五、 数据并行与分布式优化器（DP / ZeRO-1）

在传统的数据并行（Standard DP / PyTorch DDP）中，为了实现多卡并行训练，每个 GPU 都会保存一份**全量的模型参数、全量的梯度以及全量的优化器状态**。随着大语言模型（LLM）参数量的剧增，**优化器状态（Optimizer States）占据了绝大部分显存**，成为了限制单卡训练模型规模的主要瓶颈。

微软提出的 **ZeRO-1（Zero Redundancy Optimizer Stage 1，优化器状态切分 $P_{os}$）** 就是为了消除这一显存冗余而生的技术。

### 一、 传统混合精度训练的显存开销分析

在回答 ZeRO-1 原理前，首先需要明确在现代 LLM 训练（以经典的 **FP16/BF16 混合精度 + AdamW 优化器** 为例）中，显存到底被谁消耗掉了。

假设模型参数量为 $\Psi$：

| 数据类型 | 精度/格式 | 字节数 / 参数 | 总显存开销（Bytes） | 属于哪一部分 |
| --- | --- | --- | --- | --- |
| **模型参数（Parameters）** | FP16 / BF16 | 2 Bytes | $2\Psi$ | 基础模型 |
| **梯度（Gradients）** | FP16 / BF16 | 2 Bytes | $2\Psi$ | 反向传播生成 |
| **主参数（Master Weights）** | FP32 | 4 Bytes | $4\Psi$ | 优化器状态（$P_{os}$） |
| **一阶动量（$m$, Momentum）** | FP32 | 4 Bytes | $4\Psi$ | 优化器状态（$P_{os}$） |
| **二阶动量（$v$, Variance）** | FP32 | 4 Bytes | $4\Psi$ | 优化器状态（$P_{os}$） |

- **静态显存汇总**：
  - **模型 + 梯度**：$2\Psi + 2\Psi = 4\Psi$ 字节。
  - **优化器状态（$P_{os}$）**：$4\Psi + 4\Psi + 4\Psi = 12\Psi$ 字节。
  - **单卡总静态显存** $= 16\Psi$ 字节！

> **结论**：优化器状态（$12\Psi$）占据了静态显存的 **75%**！例如一个 7B 参数的模型，优化器状态就需要占用 $7 \times 12 = 84$ GB 显存，远远超过单张 80GB A100/H100 或昇腾 910B 的显存容量。

### 二、 ZeRO-1（$P_{os}$）的核心切分原理

ZeRO-1 的核心思想是：**"计算时各自冗余，更新时按卡分工"**。

在数据并行组（DP 组，卡数为 $N_{dp}$）内：

1. **模型参数 $\Psi$ 和梯度 $g$** 仍然在每张 GPU 上保持全量复制（Replicated，各占 $2\Psi$ 字节）。
2. **优化器状态 $P_{os}$** 被平均切分为 $N_{dp}$ 份（Partitioned），每张 GPU **只保留与更新自己的 $1/N_{dp}$ 份优化器状态**（对应 $1/N_{dp}$ 份 FP32 主参数、一阶动量 $m$ 和二阶动量 $v$）。

![ZeRO 各阶段显存占比与通信量对比 (论文原图)，AI 生成](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTCMKn-TY34hZ9PXRgxQsSRRCLN3S3hST7rTSMN9JjZGz3fo8yUAb5j1e9q&s=10)

ZeRO 各阶段显存占比与通信量对比（论文原图）。**来源：DeepSpeed**

### 三、 显存节省公式推导

经过 ZeRO-1 切分后，每张 GPU 上的静态显存开销变为：

$$\text{Memory}_{\text{ZeRO-1}} = \underbrace{2\Psi}_{\text{FP16 参数}} + \underbrace{2\Psi}_{\text{FP16 梯度}} + \underbrace{\frac{12\Psi}{N_{dp}}}_{\text{FP32 优化器状态}} = 4\Psi + \frac{12\Psi}{N_{dp}}$$

- **当 $N_{dp} = 1$（单卡）**：显存为 $16\Psi$。
- **当 $N_{dp} = 8$（单节点 8 卡）**：显存降至 $4\Psi + \dfrac{12\Psi}{8} = 5.5\Psi$（**显存降低约 65%**）。
- **当 $N_{dp} = 64$（大集群）**：显存降至 $4\Psi + \dfrac{12\Psi}{64} \approx 4.19\Psi$（**显存降低接近 74%**）。

### 四、 ZeRO-1 的完整执行流程与通信原语

ZeRO-1 在打破优化器状态冗余的同时，巧妙设计了通信机制，使其**完全没有增加额外的总通信带宽开销**。

```
[ 前向传播 Forward ] ──► 各卡独立计算，输入不同 Batch 数据
       │
       ▼
[ 反向传播 Backward ] ──► 各卡独立计算，生成全量 FP16 梯度
       │
       ▼
[ Reduce-Scatter 阶段 ] ──► 替换传统的 All-Reduce：
                           各卡将梯度规约归并，但每张卡只接收属于自己的 1/N 梯度 slice
       │
       ▼
[ 局部优化器更新 Step ] ──► 每张卡用自己分到的 1/N 梯度，更新自己持有的 1/N FP32 优化器状态，
                           并更新出对应的 1/N FP16 最新参数 slice
       │
       ▼
[ All-Gather 阶段 ]    ──► 各卡将更新后的 1/N FP16 参数 slice 广播给 DP 组内的所有人，
                           重新拼凑出完整的 FP16 模型参数，进入下一轮迭代
```

#### 关键通信原语对比：

- **传统 DDP 模式**：
  - 执行一次 **`All-Reduce`** 聚合梯度。
  - 通信数据量：$2 \times \dfrac{N_{dp} - 1}{N_{dp}} \times 2\Psi \approx 4\Psi$ 字节。

- **ZeRO-1 模式**：
  1. **`Reduce-Scatter` 梯度**：每张卡得到 $1/N_{dp}$ 的规约后梯度，通信量为 $\dfrac{N_{dp} - 1}{N_{dp}} \times 2\Psi$。
  2. **`All-Gather` 参数**：更新完毕后同步最新的 FP16 参数，通信量为 $\dfrac{N_{dp} - 1}{N_{dp}} \times 2\Psi$。
  - **总通信数据量** $= 2\Psi + 2\Psi = 4\Psi$ 字节！

> **重要结论**：**ZeRO-1 实现了"免费的午餐"**，它在将优化器显存开销降低最多 4 倍的同时，**总通信量与传统 DDP 完全一致**。

### 五、 总结与局限性

#### 优势：

1. **零通信惩罚**：通信量与标准 DP 完全相等，没有任何额外传输等待。
2. **零侵入性与极高稳定性**：由于前向传播与反向传播期间，参数和梯度都是全量的，完全不改变算子执行逻辑，极易集成到现有框架（如 PyTorch DDP、Megatron-LM、DeepSpeed、MindSpeed 等）。

#### 局限性：

尽管优化器状态被切分了，但 **FP16 参数（$2\Psi$）和梯度（$2\Psi$）依然保持全量冗余**。当模型参数量超过 13B~30B 时，单卡仅存放参数和梯度就会面临 OOM，此时需要进一步升级至 **ZeRO-2（切分梯度 $P_g$）** 或 **ZeRO-3（切分参数 $P_p$）**。

---

<a id="sec16"></a>
## 十六、 基于 Qwen-27B 训练多学科科学 LLM 的总体策略

基于 Qwen 27B 级别基座模型训练涵盖数学证明、分子化学、基因生命科学等多学科的科学大模型，核心突破点在于**解决不同学科间表征模态差异大、符号推导严密性要求高以及科学幻觉敏感**的问题。

总体训练策略应遵循"**表征重构 → 增量预训练（CPT）→ 过程显式化 SFT → 规则/编译器驱动的强化学习（GRPO/RL）→ 科学工具链协同**"的五阶段路径。

### 一、 词表扩展与数据表征工程（Representation Engineering）

通用 Tokenizer 在处理基因序列（DNA/RNA）、分子 SMILES 或 Lean 4 证明代码时，极易发生"分词过碎"或"语义切分错误"，导致序列长度暴增及 Perplexity 异常。

- **词表扩展（Tokenizer Expansion）**：
  - **分子/化学**：引入高频官能团（Functional Groups）与 SMILES/SELFIES 专用 Token。
  - **基因/蛋白**：引入 k-mer 短语（如 3-mer / 6-mer DNA Token）与常见氨基酸模体（Motifs）。
  - **数学/逻辑**：补全 LaTeX 复杂算子与 Lean 4 / Coq 形式化语言关词。
- **多模态文本化表达（Multi-View Textualization）**：
  - 对同一个分子/蛋白质，构建"SMILES + SELFIES + 2D/3D 拓扑图文本描述（Graph-to-Text）+ IUPAC 名称"的多重对照表示，增强模型对结构与特性的映射感知。

### 二、 领域增量预训练（Continual Pre-training, CPT）

为防止模型在吸收专业领域知识时发生语言能力衰退（Catastrophic Forgetting），必须严格设计数据混合比例（Data Mixture）。

**推荐 CPT 数据配比矩阵**：

| 数据类型 | 占比 | 数据来源与内容 | 核心作用 |
| --- | --- | --- | --- |
| **通用基座/代码/数学** | 35% - 40% | Web 优质文本、Python/C++ 代码、通用数学题 | 保留推理基座与语言表达能力 |
| **科学文献与专利** | 25% - 30% | ArXiv, PubMed, CrossRef, PubChem 专利 | 建立跨学科通用科学语料与推理范式 |
| **结构化科学数据库** | 20% | UniProt, PDB, NCBI, PubChem, ChEBI（转换为 NL 描述） | 注入精确的结构、序列、反应与通路事实知识 |
| **形式化语言与符号计算** | 10% - 15% | Lean 4 Mathlib, Coq, Isabelle, RDKit/SymPy 交互日志 | 强化严格推导逻辑与符号计算敏感度 |

- **长上下文支持**：采用 32k~64k 的上下文窗口与 Packing 策略，保证完整的长论文、大型基因组序列段落或长定理证明能一次性喂入。

### 三、 过程显式化 SFT（Process-Oriented SFT）

摒弃传统的"输入问题 → 输出最终答案"的数据形式，全面转向**思维链（CoT）与逻辑步步推导**。

- **数学证明**：
  - 采用"自然语言思路分析 → 步骤化定理拆解 → Lean 4 形式化证明代码"的三段式格式。
- **化学/分子设计**：
  - **逆合成分析**：按反应机理逐步倒推断键与保护基团选择。
  - **性质预测**：要求模型先给出分子结构特征归因（如"存在强电子吸引基团"），再给出定量/定性预测。
- **基因/生信**：
  - **变异解读**：按照"DNA 突变 → 氨基酸改变 → 蛋白质 3D 构象变化 → 通路影响/疾病关联"的完整因果链训练。

### 四、 规则与编译器驱动的强化学习（Verifier-Driven RL / GRPO）

科学领域的最大优势在于**存在大量的可确定性验证外部环境（Executables & Verifiers）**。可采用类似 DeepSeek-R1 的 **GRPO（Group Relative Policy Optimization）** 架构，结合外部验证工具提供绝对客观的 Reward。

```
┌─── Lean 4 编译器 ───► 编译通过/失败 (0/1)
                              │
[ Prompt ] ──► Qwen 27B ───► ┼─── RDKit 工具 ─────► 分子合法性/化合价检查 (Score)
  (多领域)     (生成多条 Chain)│
                              ├─── SymPy 引擎 ─────► 符号等价性验证 (0/1)
                              │
                              └─── PRM 过程赏罚模型 ──► 逐步打分 (Step Rewards)
```

1. **数学 Reward**：接入 Lean 4 / Coq 编译器，通过定理证明校验器返回 0/1 判定；结合过程奖励模型（PRM）对每一步推理打分。
2. **化学 Reward**：接入 RDKit，校验生成分子的 SMILES 合法性、化合价约束、合成可及性分数（SA Score）及类药性（QED）。
3. **基因/蛋白 Reward**：基于与 ESM/AlphaFold 的预测比对得分（如 pLDDT）、BLAST 序列匹配度或自由能变化作为 Reward 信号。

### 五、 工具增强与专用 Adapter 对齐（Tool-Augmented System）

27B LLM 不应独立完成耗费巨大算力的数值计算（如 3D 分子动力学模拟或精确量子化学计算），而应定位为"科学计算控制中枢"。

- **Tool-Calling 训练**：精调模型熟练使用 Python 科学计算库（`RDKit`, `Biopython`, `SymPy`, `OpenMM`）以及数据库查询 API。
- **模态 Adapter 对齐（轻量化扩展）**：
  - 如果需要输入真实的 3D 分子结构或高维基因表达谱，无需重新训练全量 LLM。
  - 可在 LLM 侧边挂载预训练好的图神经网络（如 EGNN）或基因大模型（如 Evo 2 / ESM-2）作为 Encoder，通过 MLP Projector 将结构特征映射至 Qwen 27B 的 Embedding 空间。

### 六、 27B 模型分布训练工程架构建议

对于 27B 参数规模，在典型 8 卡/16 卡/32 卡 NPU/GPU 集群上，推荐配置如下分布式并行拓扑：

- **并行组合**：
  - **张量并行（TP）**：设为 `2` 或 `4`（限制在单节点 HCCS/NVLink 高速互联内）。
  - **上下文并行（CP）**：对于 32k/64k 文本，采用 **Hybrid CP**（`ulysses-degree-in-cp` 设为 2~4，剩余维度走 Ring CP），隐藏长文本 Attention 通信开销。
  - **数据并行/优化器切分（DP/ZeRO-1/ZeRO-2）**：在剩余卡间进行 DP 切分，使用 ZeRO-1 切分 FP32 优化器状态，结合 FlashAttention 确保极高的计算吞吐。

---

<a id="sec17"></a>
## 十七、 两台昇腾 A3（2×16 卡）SFT 与 RL 工程架构建议

针对 **2 台 16 卡昇腾 A3（共 32 卡，每卡 64G 显存）** 训练 27B 规模多学科大模型，拓扑结构的核心特征是：**单节点内拥有 16 卡 HCCS 高速互联**，而**跨节点通信走 RoCE 网络**。

64G 显存对于 27B 模型相对充裕，架构设计的目标是**最大限度提升算力利用率（MFU）**、**减少跨节点通信**，并**避免流水线气泡（PP Bubble）**。

### 一、 SFT 分布式训练工程架构建议

在 32k~64k 长上下文 SFT 阶段，推荐采用 **TP=4 + CP=2 + PP=1 + DP=4（ZeRO-2）** 的并行组合。

#### 1. 并行维度拆解与拓扑映射

- **张量并行（$TP = 4$）**：Qwen 27B 的 KV Head 通常为 8（GQA），$TP = 4$ 满足 $N_{\text{kv\_head}} \bmod TP == 0$。4 卡组内通信量大，**必须锁死在单节点内的 HCCS 高速环上**。
- **上下文并行（$CP = 2$）**：采用 **Ulysses CP**。单节点内 $TP \times CP = 4 \times 2 = 8$ 卡。单节点 16 卡刚好可以容纳 2 个（$TP=4, CP=2$）组，**实现 CP 与 TP 通信 100% 走节点内 HCCS，零跨节点 CP 开销**。
- **流水线并行（$PP = 1$）**：显存完全够用，**彻底放弃 PP**，消除流水线气泡与 Stage 间的通信同步等待。
- **数据并行 / 分布式优化器（$DP = 4$, ZeRO-2）**：
  - 总卡数 $32 / (TP 4 \times CP 2 \times PP 1) = DP 4$。
  - 采用 **ZeRO-2（切分优化器状态 + 梯度）**。跨节点的 2 个机器之间仅进行 DP 的 `Reduce-Scatter` 与 `All-Gather` 通信，该通信可以完美重叠（Overlap）在前向/反向计算中。

#### 2. SFT 显存预算与拓扑分布

**单卡 64GB 显存分配预算（TP=4, DP=4, ZeRO-2）**：

- **FP16 模型参数**：$\dfrac{54\ \text{GB}}{TP\ 4} = 13.5$ GB
- **ZeRO-2 梯度**：$\dfrac{13.5\ \text{GB}}{DP\ 4} \approx 3.4$ GB
- **ZeRO-2 优化器状态（AdamW）**：$\dfrac{12 \times 27B}{TP\ 4 \times DP\ 4} \approx 20.3$ GB
- **静态显存小计**：$\approx 37.2$ GB
- **动态激活显存**：剩余 $\approx 26.8$ GB，搭配开启选择性重算（Selective Recompute）和 FlashAttention，可轻松支持 32k~64k 的长文本 Packing 训练。

#### 3. MindSpeed 核心参数配置

```bash
# 32卡 SFT 配置 (2节点 x 16卡)
--tensor-model-parallel-size 4 \
--context-parallel-size 2 \
--context-parallel-algo ulysses_cp_algo \
--pipeline-model-parallel-size 1 \
--use-distributed-optimizer \          # 开启 ZeRO-2
--overlap-grad-reduce \                 # 梯度规约与反向计算重叠
--use-flash-attn \
--sequence-parallel \                   # 开启序列并行进一步压缩显存
--recompute-granularity selective       # 选择性重算
```

### 二、 强化学习（GRPO）分布式训练工程架构建议

对于数学证明、化学与基因科学，建议采用类似 DeepSeek-R1 的 **GRPO（Group Relative Policy Optimization）** 算法。由于科学领域存在严格的外部校验器（Lean 4、RDKit、SymPy），RL 架构采用 **"同宿部署（Colocated）+ 异步 CPU 验证池"** 模式。

#### 1. 节点与任务部署架构（Colocated 方案）

在 32 卡规模下，推荐采用 **MindSpeed-RL + vLLM-Ascend 同宿轮换模式**，避免显存跨卡搬运的巨大开销。训练流程分为三个交替阶段：

```
[ 阶段 1: Rollout 采样 ] ──► 32 卡全部切为 vLLM-Ascend 推理引擎 (TP=4 x 8 实例)
                               │  生成 G=8 个 Candidate 响应
                               ▼
[ 阶段 2: 外部规则 Reward ] ──► 抛给 CPU 异步计算集群 (执行 Lean4 编译 / RDKit 校验)
                               │  返回 0/1 或定量的 Reward 标量
                               ▼
[ 阶段 3: Policy 更新 ]   ──► 32 卡切回 MindSpeed 训练引擎 (TP=4, CP=2, DP=4)
                                  计算 Policy Loss & Ref KL 散度并更新权重
```

#### 2. RL 关键模块与硬件分配

| 模块名称 | 部署位置 / 卡数分配 | 运行模式与并行策略 | 核心说明 |
| --- | --- | --- | --- |
| **Actor Policy（训练）** | 32 NPUs（全量） | TP=4, CP=2, DP=4, ZeRO-2 | 计算 Log Standard Probs，更新梯度。 |
| **Actor Rollout（推理）** | 32 NPUs（全量） | vLLM-Ascend, TP=4（共 8 个 Engine 实例） | 专用于高吞吐 Batch 生成，利用 PagedAttention 预留 25GB+ KV Cache。 |
| **Ref Model（参考模型）** | 32 NPUs（共享显存） | 与 Actor 共用权重或静态加载 FP16 权重 | 计算 KL 散度。可开启 **Offload to CPU RAM**，仅在计算 Logprob 时加载。 |
| **Reward Verifiers（编译器/工具）** | **Host CPU 进程池** | Python 多进程 / Docker 隔离容器 | 运行 Lean 4 校验器、RDKit 分子结构检查、SymPy 代数化简。**不占用 NPU 显存**。 |

#### 3. GRPO 显存与通信优化技巧

1. **Rollout 采样吞吐优化**：
   将 32 卡划分为 8 个独立的 **TP=4** vLLM 推理实例。由于 **TP=4** 锁在单节点 HCCS 内，8 个实例可以并行处理 Prompt，生成采样序列（如每个 Prompt 生成 $G = 8$ 个 Completion）。
2. **Ref Model 内存开销压缩**：
   采用 **LoRA 模式进行 RL 训练**（Policy 仅训练 Target Modules），此时 Ref Model 与 Policy Model **共享 Base 模型权重**，只在 Policy 侧挂载 Adapter，使 RL 阶段的额外显存增加量降低到 < 2GB。
3. **CPU 外部验证器异步化（Async Reward Pipeline）**：
   由于 Lean 4 形式化证明编译或分子动力学模拟可能耗时较长（数百毫秒至数秒），必须建立 CPU 多进程 Worker 队列。NPU 完成 Rollout 后将文本送入 CPU 队列，同时预先加载下一 Batch 数据，实现 NPU 计算与 CPU 编译校验并行。

### 三、 总结：集群工程落地最佳实践清单

1. **网络拓扑对齐**：检查单节点 16 卡的 HCCS 连线图，确保每一个 **TP=4** 组内的 4 张卡属于同一个 HCCS 物理 Sub-ring（子环），避免节点内跨 Bus 传输。
2. **通信与计算重叠**：在 SFT 和 RL 训练中，开启 MindSpeed 的 `--overlap-grad-reduce` 和 `--use-cp-send-recv-overlap`，隐藏跨节点的 DP/ZeRO-2 通信。
3. **数据预处理 Packaging**：针对长文本 SFT，提前将数学证明与基因序列拼接（Concat & Pack）至固定的 32k 长度，并添加 `position_ids` 块掩码，防止预处理成为训练瓶颈。

---

<a id="sec18"></a>
## 十八、 MindSpeed/Megatron 与 torch/torch_npu 两种方案的关键区别

在昇腾（Ascend）NPU 生态中，**基于 MindSpeed / Megatron** 与 **基于原生 PyTorch / `torch_npu`** 代表了两种完全不同的技术路线和开发范式。

简单来说：

- **`torch/torch_npu`** 是**底层的硬件适配层与基础算子库**，目的是让标准的 PyTorch 代码能够无缝运行在昇腾 NPU（如 910B/910C）上。
- **`MindSpeed/Megatron`** 是**针对大语言模型（LLM）的超大规模并行训练加速框架**，它在 Megatron-LM 的基础上，深度绑定 CANN 和 `torch_npu`，通过定制化的 Ascend C 高性能算子与通信重叠技术，榨干昇腾芯片的极致性能。

### 一、 核心维度对比一览

| 维度 | 基于 MindSpeed / Megatron | 基于原生 PyTorch / `torch_npu` |
| --- | --- | --- |
| **定位与目标** | 高吞吐、大规模 LLM 预训练 / 长文本 SFT / RL | 快捷开发、模型迁移验证、轻量微调/推理 |
| **并行支持能力** | **原生 3D/4D/5D 高阶并行**（TP + PP + CP + EP + SP + ZeRO） | 主要依赖 **DP / DeepSpeed ZeRO-1/2/3** 或 **PyTorch FSDP**，高阶并行（如 TP/CP）支持较弱 |
| **长文本（CP）支持** | 支持 **Ulysses、Ring-Attention 及 Hybrid CP**，极致优化通信 | 缺乏原生高效 CP 实现，超长上下文易 OOM 或性能急剧下降 |
| **专家并行（EP）** | 支持 **DeepSeek/MoE 专家并行**，集成高性能 `All-to-All` 重叠 | 难以进行高效的跨节点 MoE 路由与通信重叠 |
| **算子与通信融合** | 深度集成 **CANN/Ascend C 定制融合算子**，显存与计算重叠（Overlap）极为激进 | 依赖 PyTorch 基础算子组合，算子调度开销（Launch Overhead）相对较高 |
| **代码与权重生态** | 需要按照 Megatron 结构改写模型代码，且需进行 **HF ↔ Megatron 权重转换** | **直接兼容 Hugging Face 生态**（`AutoModelForCausalLM`），几乎零代码修改即可运行 |
| **硬件算力利用率（MFU）** | **高（通常 40% - 60%+）** | **中/低（通常 20% - 35%）** |

### 二、 四大关键技术差异详解

#### 1. 并行架构与扩展上限（Scale Limit）

- **原生 `torch/torch_npu` + DeepSpeed / FSDP**：
  - **模式**：主要采用**数据并行（DP）**与 **ZeRO 显存切分**。
  - **瓶颈**：当模型参数量突破 20B/70B，或者单条 Token 长度达到 32k/64k 以上时，单纯依靠 ZeRO-3 会产生极大的跨节点 `All-Gather` 通信开销。单靠 FSDP 无法支撑超长序列（缺少 CP）和 MoE 架构（缺少 EP）。
- **`MindSpeed / Megatron`**：
  - **模式**：提供完整的**多维混合并行（3D/4D Parallelism）**。
  - **优势**：可以将张量并行（TP）锁定在节点内 HCCS 高速互联，将上下文并行（CP）与流水线并行（PP）拓展至跨节点。通过切分模型结构本身（而非仅仅切分优化器/梯度），使得 100B~1000B+ 参数的模型以及 100k+ 序列能够在上千卡集群中近乎线性扩展。

#### 2. 通信与计算的隐式重叠（Overlap Optimization）

- **原生 `torch/torch_npu`**：
  - 算子发射按照 PyTorch 动态图（Eager Mode）执行，通信（如 `All-Reduce`）与前向/反向计算的重叠主要依赖 DeepSpeed 或 DDP 默认的 Bucket 机制，粒度较粗。
- **`MindSpeed / Megatron`**：
  - **深度重叠**：MindSpeed 在底层改写了 Megatron 的 Transformer Layer 执行流，实现了：
    - **`CP Send-Recv Overlap`**：在 Ring-Attention 计算当前 KV 块时，异步传输下一个 KV 块。
    - **`TP / SP All-Gather / Reduce-Scatter Overlap`**：将序列并行的通信隐藏在 Linear 层矩阵乘法（GEMM）内部。
    - **`EP All-to-All Overlap`**：在 MoE 路由分发 Token 的同时进行专家计算（如 DualPipe 思想）。

#### 3. 算子融合与显存优化（Memory & Operator Fusion）

- **原生 `torch/torch_npu`**：
  - 通常使用标准的 PyTorch LayerNorm、Rotary Embedding、SwiGLU 组合。即便开启 PyTorch 2.0 `torch.compile`，在 NPU 上的图编译（TorchAir/CANN Graph）仍处于不断演进阶段，融合效率不及手动优化的 C++ / Ascend C 内核。
- **`MindSpeed / Megatron`**：
  - 专门集成了针对昇腾 NPU 架构优化的 **Ascend C 融合算子**（如 `FlashAttention-2/3`、`RMSNorm` 融合算子、`SwiGLU` 融合算子、`Rotary Positional Embedding (RoPE)` 融合算子）。
  - 引入了 **Selective Activation Recomputation（选择性激活重算）**，只重算显存占比大但计算量小的算子（如 Attention Dropout/Softmax），大幅降低显存占用同时减少计算开销。

#### 4. 开发成本与模型生态迁移

- **原生 `torch/torch_npu`（低迁移门槛）**：
  - 只需要在代码头部加上 `import torch_npu`，并将 `.cuda()` 替换为 `.npu()`，即可直接加载 Hugging Face 的代码和 `safetensors` 权重。适合快速验证算法正确性、做小型 Dataset 的 LoRA 微调。
- **`MindSpeed / Megatron`（高架构门槛）**：
  - 必须将 Hugging Face 格式的模型结构重构成 Megatron 的 `ParallelTransformerLayer` 语法；
  - 训练前必须运行权重转换脚本（把单个 checkpoint 按照 TP/PP 切分为多个分片），训练完成后还需将分片合并回 Hugging Face 格式。

### 三、 选型决策指南

```
┌─► 参数量 ≤ 14B，且文本长度 ≤ 8k？
                            │   ├─► 是 ──► 【方案 A】原生 torch/torch_npu + DeepSpeed / HF Trainer
                            │   └─► 否 ──┐
[ 业务需求与技术选型 ] ─────┤            │
                            │            ▼
                            └─► 参数量 ≥ 27B / 长文本 (≥32k) / MoE 架构 / 大规模预训练？
                                └──────► 【方案 B】MindSpeed / Megatron 架构
```

#### 适用场景推荐：

1. **选择 `torch/torch_npu`（+ Hugging Face / DeepSpeed）的场景**：
   - **快速验证（PoC）**：刚拿到新模型，需要在 1~2 天内快速跑通推理或 SFT。
   - **参数量较小**：训练 1.8B、7B、14B 规模的模型，且单卡显存足够放下的场景。
   - **轻量化微调**：使用 LoRA / QLoRA 等冻结基座权重的微调方案。
2. **选择 `MindSpeed / Megatron` 的场景**：
   - **超大规模预训练 / 增量预训练（CPT）**：模型规模在 27B、70B 甚至数百 B。
   - **长文本训练**：上下文在 32k、64k、128k 以上，必须依赖 Ulysses / Ring CP 解决 OOM 问题。
   - **MoE / DeepSeek 架构**：涉及复杂的专家路由与跨卡 `All-to-All` 高并发通信。
   - **追求极致吞吐与算力性价比**：需要将昇腾 NPU 的 MFU 榨干至 50% 以上，严格控制算力成本的大型 AI 团队。

---

<a id="sec19"></a>
## 十九、 通信与计算重叠（TP / CP / PP Overlap）

在大语言模型分布式训练中，**通信与计算重叠（Communication-Computation Overlap）** 是将 GPU/NPU 算力利用率（MFU）提升至 50%~60%+ 的核心技术。

其底层物理基础在于：**AI 芯片的矩阵计算单元（如 NPU Cube 核心 / GPU Tensor Core）与网络通信引擎（如 HCCL/NCCL 控制器、RDMA 网卡）在硬件结构上是独立的**。只要调度得当，两者的流水线就可以并行运转。

在 **MindSpeed / Megatron-LM** 中，TP、CP、PP 三种并行的通信重叠具体实现机制与配置方式如下：

### 一、 TP 通信重叠（TP Comm Overlap / SP Overlap）

在开启序列并行（Sequence Parallelism, SP）的 Megatron/MindSpeed 中，TP 模块包含两类典型的计算-通信组合：

1. **ColumnParallelLinear（前向）**：`AllGather` 通信 → 矩阵乘（Gemm）。
2. **RowParallelLinear（前向）**：矩阵乘（Gemm）→ `ReduceScatter`（或 `AllReduce`）通信。

传统执行是串行的（先通信完再算 Gemm，或者先算完 Gemm 再通信），TP Overlap 的目标是将通信掩盖在 Gemm 计算之后。

#### 核心实现原理：

1. **细粒度 Micro-Chunk 切分（粗粒度重叠）**：将输入张量在序列长度维度 $S$（或 Batch 维度）切分为 $N$ 个更小的 Chunk（如 Chunk A 和 Chunk B）。
   - **RowParallel 流程**：
     - 步骤 1：计算 Chunk A 的 Gemm。
     - 步骤 2：**异步发起 Chunk A 的 `ReduceScatter` 通信**，同时 NPU/GPU **并行计算 Chunk B 的 Gemm**。
     - 步骤 3：等待 Chunk A 通信完成，异步发起 Chunk B 的 `ReduceScatter` 通信。
2. **底层融合算子（MC2 / Ascend C 算子级重叠）**：在昇腾 **MindSpeed** 中，借助 CANN 底层的 **MC2（MindSpeed Communication Component）** 机制，提供融合算子（如 `npu_mm_all_reduce_base` 或 `npu_all_gather_base_mm`）：
   - 芯片在执行 M/N/K 矩阵乘法的 Tile 计算块时，一旦某个 Tile 计算完成，直接触发片上 DMA / 网络引擎将该 Tile 结果发送出去，**实现指令级/Tile 级的 Gemm 与 HCCL 通信重叠**。

#### MindSpeed 命令行参数与配置：

```bash
# 开启 TP/SP 通信重叠
--tp-comm-overlap \
--use-fused-ring-attention-update \   # 搭配融合算子
# 在 MindSpeed 中指定 TP 通信重叠算法策略 (如 mc2 或 bulk)
--tp-comm-overlap-cfg ${PATH_TO_CONFIG_JSON}
```

### 二、 Ring CP 通信重叠（Ring Context Parallel Overlap）

在超长文本训练中，Ring-Attention 将序列 $S$ 切分为 $CP$ 个 Block 分发给各卡。Ring CP 需要在 $CP$ 个 Step 的环形迭代中，不断将当前卡的 Key/Value（KV）张量 P2P 发送给下一张卡，并接收上一张卡发来的 KV 张量。

#### 核心实现原理：双缓冲区机制（Double Buffering）

MindSpeed/Megatron 在显存中开辟两套 KV 缓存区（Buffer 0 和 Buffer 1）：

```
[ Step t ]
├── NPU Cube 核心 ──► 计算 Buffer 0 (KV_t) 与 Local Q 的 FlashAttention (计算开销: O(S_block²))
└── HCCL/RDMA 引擎 ──► 异步 P2P 传输: 发送 KV_t, 接收 KV_{t+1} 存入 Buffer 1 (通信开销: O(S_block))
[ Step t+1 ]
├── NPU Cube 核心 ──► 计算 Buffer 1 (KV_{t+1}) 与 Local Q 的 FlashAttention
└── HCCL/RDMA 引擎 ──► 异步 P2P 传输: 发送 KV_{t+1}, 接收 KV_{t+2} 存入 Buffer 0
```

#### 为什么掩盖效果极佳？

- **计算复杂度**：$T_{\text{compute}} = O(S_{\text{block}}^2 \times D)$（FlashAttention 二次方复杂度）。
- **通信复杂度**：$T_{\text{comm}} = O(S_{\text{block}} \times D)$（P2P 传输线性复杂度）。
- 当分块大小 $S_{\text{block}} \ge 2048$ 时，$T_{\text{compute}} \gg T_{\text{comm}}$，通信时间可以被 **100% 完美隐藏**在 FlashAttention 的计算时间之后。

#### MindSpeed 命令行参数与配置：

```bash
--context-parallel-algo megatron_cp_algo \   # 或 hybrid_cp_algo
--context-parallel-size 8 \
--use-cp-send-recv-overlap                  # 开启 Ring CP P2P 双缓冲异步重叠
```

### 三、 PP P2P 通信重叠（Pipeline Parallelism P2P Overlap）

在流水线并行（PP）中，相邻 Stage 之间（Stage $k \to$ Stage $k+1$）需要传递前向激活值（Activations）和反向激活梯度（Gradients）。

#### 核心实现原理：

1. **Interleaved 1F1B（虚拟流水线 VPP）**：
   - 将单张卡上的物理层数切分为 $v$ 个虚拟 Stage（例如 32 层模型，$PP = 4$，原本每卡 8 层；若 $v = 2$，则每卡分到 2 个 Chunk，每个 Chunk 4 层）。
   - **重叠逻辑**：当卡 $K$ 完成虚拟 Chunk 1 的前向计算后，**非阻塞（Non-blocking）异步发起 P2P 发送**；同时**立即投入到虚拟 Chunk 2 的前向计算（或反向计算）中**，无需原地等待 P2P 通信完成。
2. **Boundary Layer Overlap（边界层计算重叠）**：
   - 在最新的 Megatron/MindSpeed（如 DeepSeek DualPipe 思想实现）中，进一步将每个 Chunk 的首尾 Transformer Layer 拆分。
   - 尾层（Last Layer）的计算与下一 Stage 的 P2P 发送重叠，首层（First Layer）的计算与上一 Stage 的 P2P 接收重叠。

```
[ GPU k 计算流 ] ──► [ Layer 1..N-1 计算 ] ──► [ Layer N 计算 ] ──────────────────────► [ 下一 Stage 算子 ]
                                                     │ (触发异步 P2P Send)
[ 网络传输流 ]  ────────────────────────────────────┴────────► [ P2P 传输 Activations 至 GPU k+1 ]
```

#### MindSpeed 命令行参数与配置：

```bash
--pipeline-model-parallel-size 4 \
--num-layers-per-virtual-pipeline-stage 2 \   # 开启 Interleaved 1F1B (VPP)
--use-p2p-communication-async \               # 开启 PP P2P 异步非阻塞通信
--p2p-comm-overlap                            # 开启边界层通信重叠
```

### 四、 三种重叠技术对比与协同工作汇总

在实际生产环境中（如 27B / 70B / 100B+ 大模型训练），这三种重叠技术是**同时开启、层层嵌套协同工作**的：

| 通信重叠类型 | 触发的通信原语 | 参与的硬件模块 | 核心掩盖机制 | 依赖的关键条件 |
| --- | --- | --- | --- | --- |
| **TP Overlap** | `ReduceScatter` / `AllGather` | Cube (Gemm) + HCCL/Vector | 序列/Batch 维度切 Chunk，Gemm 与通信交叉流水线 | 单 Block 的 M、N、K 足够的计算密度 |
| **CP Overlap** | P2P（`Send` / `Recv`） | Cube (FlashAttn) + HCCL/RDMA | 双缓冲区（Double Buffering）循环交替收发与计算 | $S_{\text{block}}$ 足够大（$T_{FA} > T_{P2P}$） |
| **PP Overlap** | P2P（`Send` / `Recv`） | Cube (Transformer Layer) + RDMA | VPP 虚拟流水线交错，上个 Chunk 通信与下个 Chunk 计算重叠 | Micro-batch 数量 $M \gg PP$ 规模 |

通过同时配置这三者，MindSpeed 可以将集群网络传输造成的 GPU/NPU 停顿（Stall）降至接近于 0，使大型集群获得接近线性的扩展效率。
