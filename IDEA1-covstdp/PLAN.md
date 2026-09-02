- [IDEA1: Conv-STDP 卷积前端 for VPRTempo —— 总体方案与实验设计（主控文档）](#idea1-conv-stdp-卷积前端-for-vprtempo--总体方案与实验设计主控文档)
  - [0. 论文定位与核心论点（先想清楚故事，再动手）](#0-论文定位与核心论点先想清楚故事再动手)
    - [0.1 问题](#01-问题)
    - [0.2 主张（claim）](#02-主张claim)
    - [0.3 论证策略（双轨评测，本文方法论核心）](#03-论证策略双轨评测本文方法论核心)
    - [0.4 对照组设计逻辑（为什么必须是 B0–B5 六个 + 外部参照）](#04-对照组设计逻辑为什么必须是-b0b5-六个--外部参照)
    - [0.5 竞争态势与相关工作定位（写作前必须核实，此处为骨架）](#05-竞争态势与相关工作定位写作前必须核实此处为骨架)
    - [0.6 增量验证框架（改进阶梯）——让论文是"论文"而不是消融报告](#06-增量验证框架改进阶梯让论文是论文而不是消融报告)
  - [1. 全局架构改动总览](#1-全局架构改动总览)
    - [1.1 数据流（改动前 → 改动后）](#11-数据流改动前--改动后)
    - [1.2 架构决策记录（ADR）—— 动工前必须拍板的三件事](#12-架构决策记录adr-动工前必须拍板的三件事)
      - [ADR-1 维度链与空间下采样（强制项，不是可选项）](#adr-1-维度链与空间下采样强制项不是可选项)
      - [ADR-2 层接口与框架接入：在两个遍历点加显式分发，不伪装接口](#adr-2-层接口与框架接入在两个遍历点加显式分发不伪装接口)
      - [ADR-3 conv-STDP 更新数学：blitnet 五步骤逐项裁决 + 向量化实现](#adr-3-conv-stdp-更新数学blitnet-五步骤逐项裁决--向量化实现)
  - [2. 阶段划分与依赖图](#2-阶段划分与依赖图)
  - [3. 阶段 1：基础设施与数据通路](#3-阶段-1基础设施与数据通路)
    - [前置步骤（第 -1 步：main 分支参照跑）](#前置步骤第--1-步main-分支参照跑)
    - [S1.1 分支策略与实验配置系统](#s11-分支策略与实验配置系统)
    - [S1.2 PatchNorm 开关化](#s12-patchnorm-开关化)
    - [S1.3 seed 管理与可复现性](#s13-seed-管理与可复现性)
    - [S1.4 轨 B 评测脚本 `eval_retrieval.py`](#s14-轨-b-评测脚本-eval_retrievalpy)
    - [S1.5 B0 基线双轨复现（阶段 1 的出口，Gate 0）](#s15-b0-基线双轨复现阶段-1-的出口gate-0)
  - [4. 阶段 2：ConvSNNLayer 实现（核心算法）](#4-阶段-2convsnnlayer-实现核心算法)
    - [S2.1 前向传播（单步幅度域）](#s21-前向传播单步幅度域)
    - [S2.2 WTA 竞争机制（三变体，消融变量）](#s22-wta-竞争机制三变体消融变量)
    - [S2.3 卷积 STDP 更新规则（本 idea 的核心公式）](#s23-卷积-stdp-更新规则本-idea-的核心公式)
    - [S2.4 卷积版 ITP 阈值可塑性](#s24-卷积版-itp-阈值可塑性)
    - [S2.5 接入逐层训练框架](#s25-接入逐层训练框架)
    - [S2.6 B1：Random Conv 冻结对照](#s26-b1random-conv-冻结对照)
    - [S2.7 B3：两层 Conv-STDP](#s27-b3两层-conv-stdp)
    - [S2.8 B4：同结构 CNN + BP 参照](#s28-b4同结构-cnn--bp-参照)
    - [S2.9 B5：手工 Gabor 滤波器组前端](#s29-b5手工-gabor-滤波器组前端)
  - [5. 阶段 3：实验](#5-阶段-3实验)
    - [S3.1 实验 1.2：卷积核可视化与量化分析（Figure 2）](#s31-实验-12卷积核可视化与量化分析figure-2)
    - [S3.2 实验 1.1：主表（Table 1）](#s32-实验-11主表table-1)
    - [S3.3 实验 1.3 / 1.4：消融](#s33-实验-13--14消融)
    - [S3.4 实验 1.5：效率表（Table 4）](#s34-实验-15效率表table-4)
    - [S3.5 补充实验（审稿预案驱动）](#s35-补充实验审稿预案驱动)
  - [6. 决策门与风险（提前承诺，防止事后移动球门）](#6-决策门与风险提前承诺防止事后移动球门)
  - [7. 篇幅与时间线](#7-篇幅与时间线)
  - [8. Fork 会话指引（给未来的实现会话）](#8-fork-会话指引给未来的实现会话)
    - [Fork 前一次性锁定的决策清单](#fork-前一次性锁定的决策清单改动需回到本文档修订并记录原因)
    - [总检查清单](#总检查清单)

# IDEA1: Conv-STDP 卷积前端 for VPRTempo —— 总体方案与实验设计（主控文档）

> **本文档的用途**：这是整个 idea/paper 的"主控设计文档"。后续会 fork 出多个实现会话，
> 每个 fork 会话只负责本文档中的一个步骤卡片（S1.x / S2.x / S3.x）。
> 因此每个步骤卡片都写成自包含的：**背景与动机（为什么）→ 前置依赖 → 详细操作（干什么）→ 验收标准**。
>
> 代码引用基于 main 分支当前版本。关键已核实锚点：
> `blitnet.py`（SNNLayer / addWeights / clamp_spikes / calc_stdp）、
> `VPRTempoTrain.py`（train_model 循环 / _anneal_learning_rate / train_new_model）、
> `dataset.py`（ProcessImage / PatchNormalisePad）、`metrics.py:134`（recallAtK）。

---

## 0. 论文定位与核心论点（先想清楚故事，再动手）

### 0.1 问题
VPRTempo 把图像经 `ProcessImage` 展平成 [H*W] 向量后直接进全连接 SNN（论文配置 28×28：`feature_layer: 784→1568`，稀疏随机连接 p=[0.1,0.5]）。**空间局部结构被完全丢弃**：feature_layer 的每个神经元看的是随机 10% 像素的线性组合，没有任何平移共享和局部感受野。

### 0.2 主张（claim）
在 spike 编码与 feature_layer 之间插入一个**无监督 Conv-STDP 前端**：
- 卷积核提供空间归纳偏置（局部感受野 + 权值共享）；
- 学习规则是**单步幅度域 STDP**（与 VPRTempo 的单步编码自洽），不是 SpikingJelly/bindsnet 那种多步 trace-based LIF 仿真——无多步开销；
- 全程无 BP。

### 0.3 论证策略（双轨评测，本文方法论核心）
光看端到端 Recall 无法定位增益/损失来自哪里，所以每个变体都跑两条轨：

- **轨 A（经 output layer）**：完整训练 → `run_inference` → Recall@K。回答："对完整系统有没有帮助？"
- **轨 B（raw feature retrieval）**：前端输出 → flatten → cosine 最近邻 → `recallAtK`。绕开 spike-forcing 读出，回答："encoder 本身学没学到好特征？"

两条轨的四种组合对应四种结论，其中"轨B升、轨A不升"是最有分析价值的结果（特征有效但与 spike-forcing 读出失配），这是预留的退路故事（见 §6）。

### 0.4 对照组设计逻辑（为什么必须是 B0–B5 六个 + 外部参照）
| 编号 | 前端 | 训练 | 排除的备择解释 |
|---|---|---|---|
| B0 | 无（原 VPRTempo） | — | 基线 |
| B1 | Random Conv | 冻结 | "随便加卷积都涨"（随机特征经常是强基线，必须排除） |
| B2 | Conv-STDP 1层 | 无监督 STDP | 核心处理组 |
| B3 | Conv-STDP 2层 | 逐层无监督 | "增益只是多了一层深度" |
| B4 | 同结构 CNN | BP | 性能上界参照系，回答"无 BP 代价多大" |
| B5 | **手工 Gabor 滤波器组** | 无需训练（手工设计，冻结） | "既然核收敛出 Gabor 结构，直接放固定 Gabor 不就完了"——审稿人必问，必须有这一行 |

B1 vs B2 是 Gate 1 的关键对比：**两者用同一随机初始化、同一结构，唯一差别是 STDP 训没训**。

**B5 的设计依据**：conv-STDP 文献里手工滤波器前端是有先例的——Kheradpisheh et al. 2018 的第一层就是手工 DoG 滤波器组（只有后层用 STDP 学）。所以"学习 vs 手工"的对比不是稻草人，是该领域真实的路线分歧。B5 的实现：4 方向 × 2 频率 × 2 相位的 Gabor 组（补齐到 C=32 通道），冻结，其余通路与 B2 完全相同。三种结果都有话写：B2 > B5 → "学习能适配数据统计特性，超越手工设计"；B2 ≈ B5 → "单步规则以零设计成本达到手工滤波器水平"（仍是卖点，省掉人工调参）；B2 < B5 → 亮红灯，说明规则没学到东西或塌缩（呼应 S2.3 的直流塌缩防线）。

**外部参照（不重跑，引用已发表数字）**：主表脚注或正文一段引用 Nordland 上的已发表结果——VPRTempo 原论文数字、VPRSNN 系（Hussaini et al. RA-L 2022 / ICRA 2023）、经典描述子（CoHOG / NetVLAD）的公开数字。注意这些多为 3300 地/全数据集配置，与我们 500 地单模块**不可直接比**，只作定性参照并注明划分差异（Gate 0 的判据是同仓库同配置先验跑，不是这些数字，见 §6）。

### 0.5 竞争态势与相关工作定位（写作前必须核实，此处为骨架）

**该领域已做到什么程度**（fork 写作会话前需逐条核对原文数字）：

| 路线 | 代表工作 | 做到了什么 | 与本文的差距（我们的位置） |
|---|---|---|---|
| 全连接 SNN + STDP（识别） | Diehl & Cook 2015（Front. Comput. Neurosci.） | MNIST ~95%，全连接、无卷积、无空间结构 | 我们补空间归纳偏置 |
| 卷积 SNN + STDP（识别） | Kheradpisheh et al. 2018（Neural Networks 99:56-67）；Mozafari et al. 2018/2019（reward-modulated）；Tavanaei & Maida 2016/2017 | 深层 conv SNN 做物体识别；**多步时延编码 + trace-based STDP**；第一层常为手工 DoG | 我们是单步幅度域、无多步开销；且任务是 VPR 不是识别 |
| VPR 专用 SNN（全连接） | Hussaini et al. RA-L 2022（VPRSNN, arXiv:2109.06452）；Modular SNNs ICRA 2023；arXiv:2311.13186；**VPRTempo ICRA 2024（本文基座）** | 全连接 SNN 做 VPR，权重神经元分配 / 模块化 / 时序编码。**已核实：VPRSNN 与 VPRTempo 同为 28×28 输入 + 7×7 patch norm**（QVPR 系口径一致，外部对比干净） | 无空间归纳偏置，我们补卷积前端 |
| 卷积 SNN for VPR（BP 路线） | **LoCS-Net**（Akçal et al. 2025, Frontiers in Neurorobotics, 10.3389/fnbot.2024.1490267） | 端到端卷积 SNN 做 VPR：56×56 灰度输入、3 conv + FC place 层；**BP 训练**（rate 近似 LIF → 推理转脉冲 LIF 的 ANN-to-SNN 转换路线，含多步仿真）；Nordland P@100%R 78.6%（vs SNN SOTA 73.0%）、ORC 45.7%（vs 20.2%）；Kapoho Bay 神经形态部署 | "卷积 SNN 做 VPR"的位置已被占，但走的是 BP + 多步路线。**我们的空位精确化为：第一个无 BP 局部可塑性的 VPR 卷积前端**；LoCS-Net 同时是外部对比行和 56×56 附录对照的存在理由 |
| 并发工作预警 | arXiv:2607.13584（2026-07，rate 编码 + snnTorch 离散 STDP 做 VPR） | 100 地规模、rate 编码、多步 | 需要读一遍，在 related work 里划界：多步 rate vs 单步幅度；规模与编码方式均不同 |

**一句话定位**：第一条把"卷积前端 + **无 BP** 局部可塑性"带进 VPR 任务、并以双轨协议隔离 encoder 与读出贡献的工作（LoCS-Net 已占据"卷积 SNN + BP"位置，本文不与其拼性能上界，而是占据"无 BP 可行边界 + 机制可解释性"位置）。竞争态势结论：**空位真实存在但比预想窄**，措辞必须精确到"无 BP"；窗口期有限（并发工作已出现），时间线不宜拖。

### 0.6 增量验证框架（改进阶梯）——让论文是"论文"而不是消融报告

**问题**：B0–B5 矩阵是"平行对比"，读起来像消融报告。论文的叙事脊柱应该是**累积改进路径**：从 B0 出发，每引入一个机制都要回答"这一步对 VPRTempo 改进了多少"，形成一张阶梯表。

**阶梯设计（Table 0 / 正文第一张三栏表）**：

| 阶梯 | 配置 | 引入的机制 | 必须回答的问题 |
|---|---|---|---|
| R0 | B0 | —（原 VPRTempo） | 基线锚点 |
| R1 | B1 | + 卷积结构（随机核） | 纯空间归纳偏置值多少？ |
| R2 | B1 + 训 STDP（无 WTA） | + 无监督可塑性 | 学习本身值多少？（与 Table 2 的 WTA=none 格**同一配置，跑一次两处引用**；更新策略见 S2.2 none 条） |
| R3 | R2 + local WTA | + 竞争稀疏 | 竞争机制值多少？ |
| R4 | R3 + ITP | + 阈值可塑性 | 通道分化值多少？（**= B2 主组合**：STDP + local WTA + ITP 全开） |

**配置钉死**：主表里的 B2 ≡ R4 全配置（C=32, k=5, conv_epoch=2, WTA=local(4×4), agg=mean, pre_mode=centered, **ITP=on, E/I=on, homeostasis=off**）；R2/R3 是阶梯专用独立配置名，不与 B2 混用，避免阶梯表与主表对不上。

每升一阶，双轨各报一次 Recall@K，**边际增益必须为正或至少有解释**（为负也要诚实报，那是分析素材）。这张表同时就是消融逻辑——区别只在于叙事顺序：消融是"从全配置往下拆"，阶梯是"从基线往上搭"。论文用阶梯叙事，附录用消融矩阵。

**执行含义**：S2.x 的实现顺序就按阶梯走，每完成一个机制立刻跑 R_n 双轨小表（单 seed 即可），不要等到 S3.2 才第一次看数字——这是"每次引入新机制都验证对 VPRTempo 的改进"的具体落实，也能最早暴露直流塌缩这类问题。

---

## 1. 全局架构改动总览

### 1.1 数据流（改动前 → 改动后）

```
【B0 现状】（论文配置：28×28 输入，main.py:368 注释 / pixi.toml 会议任务一致）
RGB图 → ProcessImage(灰度→gamma→resize→PatchNorm→uint8→spike编码)
      → 展平 [1, 784] → feature_layer(784→1568) → output_layer(→500)

【B2/B3 改动后】（维度数字以 28×28 输入、C=32, k=5, local WTA 4×4 主组合为例，依据 ADR-1）
RGB图 → ProcessImage(同上) → 展平 [1, 784]
      → 【新增】reshape 回 [1,1,28,28]
      → conv_layer_1: Conv2d(1→C1, k×k) + 阈值 + WTA + STDP/ITP   ← 无监督训练后冻结
      → (B3: conv_layer_2: Conv2d(C1→C2, k×k), 同上)
      → 【ADR-1 强制】空间下采样：local WTA 的块 winner 图即 4×4 max-pool → [1,C,6,6]
      → flatten [1, 1152]
      → feature_layer(1152 → 2304) → output_layer(→500)
```

> **输入尺寸拍板：主实验统一 28×28 + patches=7**（VPRTempo 论文/会议任务配置，非代码默认的 56×56）。
> 理由：①与会议规模（3300 地）配置一致，500→3300 扩展只差模块数，故事干净；②更小输入 = 更快迭代。
> 56×56 留一组对照（附录），说明结论对分辨率不敏感。

### 1.2 架构决策记录（ADR）—— 动工前必须拍板的三件事

这三条不是"注意事项"，而是**不定下来就无法写代码的硬决策**。每条给出数据、候选方案、取舍与结论。后续 S2.x 卡片只引用 ADR 编号，不再重复论证。

---

#### ADR-1 维度链与空间下采样（强制项，不是可选项）

**先把账算清楚**（28×28 输入、k=5、无 padding → H'=W'=24，C=32）：

| 方案 | conv 输出 flatten | feature_layer 权重（dense fp32） | 可行性 |
|---|---|---|---|
| 不下采样，直接 flatten | 32×24×24 = 18,432 | 18,432 × 36,864 ≈ 6.8×10⁸ 参数 ≈ **2.7 GB** | ❌ 不采用：存储在 24 GB 卡上勉强能跑，真正的否决理由是**参数量 ~30× 于 B0（1.2M），可比性破产** + `calc_stdp` 每样本 tile 同尺寸矩阵极慢 |
| 4×4 空间下采样后 flatten | 32×6×6 = **1,152** | 1,152 × 2,304 ≈ 2.65×10⁶ ≈ 10.6 MB | ✅ 与 B0（784×1568=1.2M）同量级 |

（56×56 对照组：输入像素是 4 倍，但 feature_layer 参数量约 22 倍——不下采样 60 GB 完全不可行，下采样后 234 MB 勉强可行——进一步支持主实验用 28×28。）

注意 blitnet 的权重是**稠密存储**（稀疏连接体现在值上，矩阵本身 dense），且 `calc_stdp` 每样本还会 tile 出 [in,out] 全矩阵（blitnet.py:540-541）——不池化没有任何变通余地。**结论：conv 与 feature_layer 之间必须有空间下采样，这是架构的强制组成部分。**

**候选下采样方案对比**：

| 方案 | 说明 | 取舍 |
|---|---|---|
| (a) conv 加 stride | conv 本身 stride=4 | 感受野跳格采样，丢细节；STDP 的 patch 提取也变 stride 对齐，复杂度上升 |
| (b) 独立 max-pool 层 | conv → clamp → 4×4 max-pool | 多一个模块，但与 WTA 解耦，WTA=none 时也必须用它 |
| (c) **复用 local WTA 的块结构** | local WTA(4×4) 本来就是"每块取 winner"——块 winner 值拼起来就是一张 4×4 max-pooled 图 | ✅ 零额外计算，WTA 与池化同一操作，叙事优雅："竞争即池化" |

**拍板**：主组合用 **(c)**——local WTA(4×4) 的 winner 值重排为 [1,C,6,6] 后 flatten（1,152 维）送 feature_layer；WTA=none 的消融格子用 **(b)** 独立 4×4 max-pool 补足同一维度链（保证消融只变 WTA 一个变量）；WTA=(a) stride 不采用。

**WTA 模式与下游维度的耦合（必须写进论文设计说明）**：

| WTA 模式 | 送 feature_layer 的张量 | 维度 | 风险 |
|---|---|---|---|
| global | 每通道 1 个 winner 值 → [C] 向量 | 32 | ⚠️ 信息瓶颈：整张图压成 32 个数，大概率掉点；若 global 在消融中意外胜出，需加 C 或改池化后再进主表 |
| local(4×4) | 块 winner 图 [C,6,6] flatten | 1,152 | 主组合，与 B0（784）同量级 |
| none | 全图 4×4 max-pool 后 flatten | 1,152 | 与 local 同维度，隔离 WTA 变量 |

**feature = 2×input 规则保留**（VPRTempoTrain.py:162）：下采样后 input=1,152 → feature=2,304，保持与 B0 相同的层宽比，可比性优先。

---

#### ADR-2 层接口与框架接入：在两个遍历点加显式分发，不伪装接口

**问题**：框架里有两处"对所有层一视同仁"的遍历，它们假设层是线性层：

1. `train_model` 的 prev_layers 冻结前向（VPRTempoTrain.py:429-434）：`spikes = self.forward(spikes, prev_layer)` → 内部是 `layer.w(spikes)`（VPRTempoTrain.py:506），即 `F.linear`；
2. `VPRTempo.py` 推理前向：同样的线性假设。

**候选方案**：

| 方案 | 做法 | 取舍 |
|---|---|---|
| (a) 伪装接口 | 给 ConvSNNLayer 包一个 `.w` 模块，forward 内部做 reshape→conv2d→flatten，让 `layer.w(spikes)` 原样成立 | 框架零改动；但 `clamp_spikes` 紧接着用 `layer.thr` 广播（blitnet.py:385）——conv 的 thr 是 [1,C,1,1] 每通道一个，flatten 后是 [1, C·H'·W']，**要么把 thr 复制 H'·W' 份（破坏"每通道一个阈值"的 ITP 语义），要么 clamp 语义错位**。伪装的代价是把空间语义压平，得不偿失 |
| (b) **显式 isinstance 分发** | 在上述两个遍历点各加一个小分支：`isinstance(layer, ConvSNNLayer)` → 走 conv 前向路径（reshape→conv→减thr→clamp→WTA→池化→flatten）；否则走原路径 | 框架改动 = 2 处小分支，原路径一行不动；语义诚实，调试时一眼看清数据形状 |

**拍板**：**(b)**。ConvSNNLayer 内部全程保持空间张量 [1,C,H',W']，**flatten 只发生在 conv→feature_layer 边界上的一次**。reshape [1,H*W]→[1,1,H,W] 发生在 conv 层入口（DataLoader 送来的仍是 `ProcessImage` 的平向量，dataset.py:437-439）。推理侧（VPRTempo.py）镜像同一分支，inference=True 的 ConvSNNLayer 只保留 w/thr（对齐 blitnet.py:101-111）。

**连带决策**：
- `train_new_model` 层循环（VPRTempoTrain.py:674）同样加 isinstance 分发 → 调 `train_conv_layer`（见 S2.5）；
- 维度重算按 ADR-1 的池化后维度：`self.input = C × (H'/4) × (W'/4)`，在 `__init__` 里由 dims/k/C/pool 参数算出，硬编码零容忍；
- 保存/加载：conv 层注册为 nn.Module 即自动进 state_dict（VPRTempoTrain.py:531-534）；推理侧构造层序必须与训练侧一致（同 layer_dict 顺序、同维度），模型命名加 `_CONVC<C>K<k>` 标记防覆盖。

---

#### ADR-3 conv-STDP 更新数学：blitnet 五步骤逐项裁决 + 向量化实现

**blitnet `calc_stdp` 实际是五个步骤的捆绑**（blitnet.py:418-636），卷积版不能笼统说"照搬"，逐项裁决：

| # | blitnet 步骤 | 卷积版裁决 | 理由 |
|---|---|---|---|
| 1 | STDP 兴奋更新 `(0.5−post)·Θ(pre)·Θ(post)`（:555-557） | **移植并改造**：作用对象从全连接矩阵元素 → winner 的核-patch 对；pre 端默认幅度加权（见 S2.3 的 Θ 失效分析） | 核心公式 |
| 2 | STDP 抑制更新（:566-568） | **合并进通道级 E/I**：抑制通道的核用同一公式、反向学习率 | 卷积核权值共享，按元素分 E/I 会破坏感受野结构（S2.1） |
| 3 | 符号钳制 [1e-6,10]（:581-584） | **改造**：[0,10]/[-10,0]，防止 0 被顶成 1e-6 破坏稀疏 | S2.3 已注明 |
| 4 | ITP Δθ=η(Θ(x)−f)（:597-606） | **移植**：每神经元 → 每通道 | S2.4 |
| 5 | Homeostasis 抑制稳态缩放（:608+，公式 4） | **默认关，留消融开关** | blitnet 需要它是因为全连接版**没有 WTA**、无发放上限控制；卷积版已有三重稳定机制（WTA 稀疏 + 每核保范数归一化 + ITP），先验证无 homeostasis 是否稳定，不稳定再开 |

**向量化是硬性实现要求**：local WTA(4×4) 下每张图 winner 数 = C×(H'/4)×(W'/4) = 32×36 = **1,152 个**；Python 逐 winner 循环 × 1,000 图 × 2 epoch ≈ 230 万次迭代，不可行。

*更进一步的观察*：把 (0.5−post) 写回 winner 位置、其余置零得到响应图 M，则每通道更新量 `ΔK_c = Σ_winners (0.5−post)·patch` **正是 pre_img 与 M 的互相关——即卷积层权重梯度的定义**。因此整个更新一行解决，走 cuDNN，无索引体操：

```python
M = torch.zeros_like(post_map)
M[winner_mask] = (0.5 - post_map[winner_mask])          # [1,C,H',W']
dK = torch.nn.grad.conv2d_weight(pre_img, layer.w.weight.shape, M)  # [C,1,k,k]，= agg'sum'
layer.w.weight.data += layer.eta_stdp * dK
# agg='mean'：dK / winner_count_per_channel.view(C,1,1,1).clamp(min=1)
# E/I 分组学习率：eta 做成 [C,1,1,1] 张量按通道乘（兴奋正、抑制负）
```

注意 `pre_mode='centered'` 时传入 `pre_img - 0.5` 即可，公式形式不变。Python 循环版仍保留在玩具测试里做逐元素对拍（两版输出必须一致），但正式训练只走 `conv2d_weight` 路径——这省掉 unfold/gather/index_add 方案里的大部分调试时间。

**内存对比（卖点素材，写给实验 1.5）**：全连接版每样本 tile 出 [in,out] = [1152, 2304] 全矩阵做更新；卷积版每样本只动 N_win×k² = 1152×25 个元素——**单样本更新量差 input 维度数量级**，这是"卷积 STDP 比全连接 STDP 便宜"的量化论据。

---

**三条 ADR 的相互依赖**：ADR-1 决定 feature_layer 维度（S2.5 的维度重算照此实现）；ADR-2 决定框架改动范围（两处 isinstance 分发 + 一处层循环分发，共 3 个小分支）；ADR-3 决定 `calc_stdp_conv` 的函数签名与性能预算。fork 会话实现 S2.x 时若发现与 ADR 冲突，**先回来改 ADR 并记录原因，再动代码**。

---

## 2. 阶段划分与依赖图

```
阶段1（基础设施，不依赖任何新算法）
  S1.1 分支与实验配置系统 ──┐
  S1.2 PatchNorm 开关化      ├─→ S1.4 eval_retrieval.py（轨B）
  S1.3 seed 管理            ─┘         │
                              S1.5 B0 基线双轨复现 ←──┘
阶段2（核心算法，依赖阶段1的数据通路）
  S2.1 ConvSNNLayer 前向 → S2.2 WTA → S2.3 卷积STDP → S2.4 卷积ITP
        → S2.5 接入逐层训练 → S2.6 B1对照 → S2.7 B3两层 → S2.8 B4 CNN参照 → S2.9 B5 Gabor组
阶段3（实验，依赖阶段2全部变体）
  S3.1 核可视化(实验1.2) → S3.2 主表(实验1.1) → S3.3 消融(1.3/1.4)
        → S3.4 效率表(1.5) → S3.5 ORC复跑 + SpikingJelly参照
```

**关键路径**：S1.2 → S1.4 → S1.5 → S2.1→S2.5 → S3.2。一切围绕先打通 B0 双轨基线。

---

## 3. 阶段 1：基础设施与数据通路

### 前置步骤（第 -1 步：main 分支参照跑）

**为什么需要它**：main 分支没有 `--seed`，自身两次运行结果都不同，"B0 逐比特回归"需要两边都有种子。因此参照跑分两层：

- **-1a 冒烟跑**（不改任何代码，在 main 上）：100 地 spring,fall→summer、28×28/patches=7、skip=0、1 次。产出：环境/数据通路验证 + 单 run 墙钟（决定后续跑本地 CPU 还是工作站 GPU）+ Recall 量级 sanity（对照论文数字定性）。
- **-1b 种子化参照跑**（S1.3 完成后，回到 main 分支）：用 S1.3 的 `seeded_ref_run.py` 驱动脚本（先固定种子再调 main 入口，不依赖任何新参数）跑 500 地正式参照——**这才是 Gate 0 与 B0 回归的真参照**；随后在 feat 分支同配置复跑，两边必须逐比特一致。

### S1.1 分支策略与实验配置系统

**背景与动机**：实验矩阵是 ~10 配置 × 3 seed × 2 轨 × 2 数据集，手工敲 CLI 参数必然出错且不可复现。论文级实验需要"配置即代码"。

**详细操作**：
1. 建分支 `feat/convstdp-base`（从 main 切出）。所有阶段 1 改动在此分支，完成后合回 main。
2. 在 `IDEA1-covstdp/phase1/configs/` 建配置目录，每个实验一个 JSON，字段对应 main.py 的全部 CLI 参数（**含 `skip`、`filter`、`data_dir`**——它们是数据通路的一部分，B0–B5、双轨、将来 3300 地扩展都靠它们对齐）+ 新增参数占位（`seed`, `patch_norm`, `frontend`, `wta_mode`, `wta_block`, `agg_mode`, `conv_channels`, `conv_kernel`, `conv_epoch`）。机器相关值（`data_dir` 等）放 `phase1/configs/local_override.json`（gitignore），不进版本化配置模板。
3. 写薄封装 `IDEA1-covstdp/experiments/run_exp.py`：读配置 → 构造 `argparse.Namespace` → **函数调用 `from main import initialize_and_run_model`（main.py:199，入口逻辑已是函数，无需大重构）** → stdout/结果表存入 `IDEA1-covstdp/results/<exp_id>/seed_<n>/`。**禁止 subprocess 拼命令行**。三个坑必须处理：①`check_pretrained_model`（VPRTempoTrain.py:549）在模型已存在时 `input()` 交互询问——批量跑会卡死，用唯一模型名（exp_id + seed）规避；②相对路径（`'./vprtempo/models'`）要求 cwd 为仓库根，run_exp.py 显式断言；③全局 logging 在多次函数调用下的重复 handler 问题。
4. 结果文件统一命名 `<exp_id>__seed<n>__trackA.json` / `__trackB.json`，内含 Recall@1/5/10/15/20/25 + recall@100%precision + 配置全文 + device + 墙钟（自描述，防止日后对不上号）。

**验收**：用一个配置跑通 B0 的 `pixi run eval` 等效流程，结果落盘且可重复（同一 seed 两次运行结果一致）。

### S1.2 PatchNorm 开关化

**背景与动机**：`ProcessImage.__call__`（dataset.py:352）第 4 步固定调用 `PatchNormalisePad`（dataset.py:422-424），输出值域 [-1,1] 再映射到 uint8。PatchNorm 本身是**局部高通滤波**（局部 Z-score 去掉局部均值），提取的正是边缘/纹理——这和 Conv-STDP 要学的东西高度重叠。所以 PatchNorm on/off 与前端类型的交互（实验 1.4）是本文最独特的一张表，前提是 PatchNorm 必须可关。

**详细操作**：
1. main.py 加参数 `--patch_norm`（on/off 选择，**默认 on**——不改变现有行为）。
2. `ProcessImage.__init__` 增加 `patch_norm=True` 参数；`__call__` 中第 4 步包条件分支：
   - on：现状不变（PatchNorm → [-1,1] → uint8 映射，dataset.py:429）。
   - off：跳过 PatchNorm，直接把 resize 后的图 clamp 到 [0,255] 转 uint8，接第 6 步 spike 编码。**注意**：off 分支要跳过第 5 步的 `(1+x)/2` 映射（那是为 [-1,1] 值域设计的）。
3. 调用链上传参：main.py → VPRTempoTrain/VPRTempo 构造 → `train_new_model`/`run_inference` 里的 `ProcessImage(model.dims, model.patches, ...)`（VPRTempoTrain.py:623-625 和 VPRTempo.py 对应位置）。
4. 模型文件名中加入 patch_norm 标记，避免 on/off 模型互相覆盖（参考现有命名约定 `<dirs>_VPRTempo_IN..._FN..._DB....pth`）。

**验收**：①同配置 on/off 各跑一次 eval，Recall@K 出现可记录差异；off 时 B0 不退化到不可用（若退化严重，本身即是实验 1.4 的一个发现）。②**on/off 各存一张输入像素值直方图到 `results/`**（on 背景应 ≈127.5、off 保留暗背景）——这是 Table 3b 直流塌缩分析的前置证据，事后无法补采。

### S1.3 seed 管理与可复现性

**背景与动机**：结论判据是"B2 − B1 > 3-seed 联合标准差"，std 的质量决定结论可信度。当前代码有多处隐式随机源：`addWeights` 里 `np.random.rand` 稀疏化（blitnet.py:308）、`torch.normal_` 初始化（blitnet.py:290）、`DataLoader(shuffle=True)`（VPRTempoTrain.py:727-733）。**不固定 seed，3-seed 实验毫无意义。**

**详细操作**：
1. 加 `--seed` 参数；入口处 `torch.manual_seed / np.random.seed / random.seed` 三件套 + `generator=torch.Generator().manual_seed(seed)` 传给 DataLoader。
2. **补两个易漏的随机源**：`DataLoader(num_workers>0)` 时配 `worker_init_fn`（按 seed+worker_id 派生每个 worker 的种子，否则 worker 内 numpy/torch 随机序列不可控）；环境变量 `PYTHONHASHSEED` 写进 run_exp.py 的启动封装。
3. 三个实验 seed 固定为 {0, 1, 2}，写死进实验配置。
4. 注意 blitnet.py:308 用的是 `np.random`（不是 torch），必须一起固定。
5. CUDA 下接受非完全确定性（cudnn benchmark），在论文里注明；seed 固定到"初始化与数据顺序可复现"级别即可。
6. **种子注入驱动脚本 `experiments/seeded_ref_run.py`**：在做任何事之前先固定三件套种子，再调 main 入口。**不依赖任何新参数**，因此可在 main 分支上跑（第 -1b 步参照）、也可在 feat 分支上跑（B0 回归对拍）——"逐比特一致"的回归测试只有两边都有种子才成立。

**验收**：同一配置同一 seed 连跑两次（CPU），模型参数一致或 Recall 完全一致。

### S1.4 轨 B 评测脚本 `eval_retrieval.py`

**背景与动机**：这是双轨方法论的半壁江山，也是审稿人眼中"分析严谨性"的来源。轨 A 的 Recall 混入了 spike-forcing 读出层的影响；轨 B 把前端输出直接做最近邻检索，干净隔离 encoder 质量。**必须对每个变体（B0–B5、每个消融格子）都能跑。**

**指标决策（全文统一，提前拍板）**：
- **主指标 Recall@1/5/10/25**（`recallAtK`）——与 VPRTempo 原论文口径一致，外部可比；
- **互补主指标 recall@100%precision + precision@100%recall + PR 曲线**（`metrics.py` 的 `recallAt100precision` / `createPR` 现成）——threshold-free，衡量相似度矩阵本身的可分性。两个方向都要：R@100%P 衡量零误报下的找回能力；**P@100%R 是 LoCS-Net 的口径**（其 Nordland 78.6% 即此指标），采用它可直接与最强外部对手比数字。R@1 只看 top-1 猜没猜对，且实测对规模钝化（500→1000 地 R@1 只掉 3 点，R@100%P 从 0.587 塌到 0.180）；
- 不再堆其他性能指标（AUC/F1 在 VPR 社区不流行）；效率指标归 Table 4，核分析指标（Gabor R² 等）是机制证据，三者分工清晰。

**详细操作**：
1. 新文件 `IDEA1-covstdp/experiments/eval_retrieval.py`，结构镜像 `run_inference`（VPRTempo.py:669）：建数据集 → 加载模型 → 遍历前向。差别在最后一步。
2. **特征提取点**（每个变体一个指定层，写进配置；**统一取池化后 flatten 的 1,152 维向量**——即 feature_layer 实际看到的那个向量，不是池化前的 18,432 维大图）：
   - B0：feature_layer 输出（clamp 后）。
   - B1/B2/B3/B5：conv 前端最后一层**池化后**输出 → flatten。**同时**记录 feature_layer 输出作为第二个特征点（附录分析用）。
   - B4：CNN backbone 输出（同样取送分类头前的 flatten 向量）。
3. 特征 L2 归一化 → 计算 Q×D cosine 相似度矩阵 S。
4. GT 构造**必须与轨 A 逐比特一致**：复用 `run_inference` 中的 GT 逻辑（单位矩阵 + `GT_tolerance` 对角膨胀 + `skip` 偏移）——最稳妥的做法是把那段 GT 构造抽成一个共用函数两边调用，而不是抄一遍（抄会漂移）。
5. 调 `recallAtK(S, GT, K)`（metrics.py:134），K ∈ {1,5,10,15,20,25}；同时输出 `recallAt100precision` 与 PR 曲线数据，PrettyTable 输出 + JSON 落盘。
6. 效率红利：轨 B 不需要训练 output_layer，conv 前端训完即可评——**调 conv 超参时先只看轨 B**，迭代周期大幅缩短。

**验收**：对 B0 跑轨 B，Recall@K 量级合理（raw 特征检索通常显著低于轨 A，正常）；同一特征矩阵手工抽查若干最近邻正确。

### S1.5 B0 基线双轨复现（阶段 1 的出口，Gate 0）

**背景与动机**：在任何新代码落地前，必须有可信的 B0 双轨基线数字。它是所有后续对比的分母，也是回归参照。

**详细操作**：
1. 数据：Nordland，参考 = spring + fall（`--database_dirs spring,fall`，现有代码的 `location_repeat=2` 机制，VPRTempoTrain.py:177-179），查询 = summer；`--database_places 500 --query_places 500`（500 地 = 单模块，不触发 max_module 拆分，先避开多模块复杂度；500 地是作者官方预训练模型的规模，README.md:45）。**输入配置与论文对齐：`--dims 28,28 --patches 7`**（main.py:368 注释：论文 28×28 / 7×7，代码默认 56×56 是开发默认值，不用）。
2. 3 seeds × （轨A: 训练 + eval）× （轨B: S1.4 脚本）。
3. 结果填入 `results/table_baseline_b0.md`；**每格记录训练/推理墙钟**，基线跑完后**外推全实验矩阵总耗时**（B0–B5 × 消融 × 3 seeds × 双轨），形成本地/工作站分工建议写入 `results/time_budget.md`——CPU-only 环境下这不是可选项，是整个时间表的关键数据。
4. PatchNorm on/off 各一组（共 2×3=6 格），提前给实验 1.4 的第一列。

**验收**：轨 A Recall@1 与**第 -1b 步的种子化 main 参照**一致（±2 点内），且 feat 分支默认参数运行与参照**逐比特一致**（真·B0 回归）。论文报告数字是 3300 地 3 模块的，与 500 地单模块不可比，只作量级 sanity。若明显偏低，先排查环境/数据问题——**不要带着坏基线进入阶段 2**。

---

## 4. 阶段 2：ConvSNNLayer 实现（核心算法）

> 新文件：`IDEA1-covstdp/src/conv_snn_layer.py`（`ConvSNNLayer` 类 + `calc_stdp_conv` 函数 + `train_conv_layer` 函数）。
> 风格对齐 blitnet.py：MIT 头、模块级注释块、行级注释。
> **设计总原则：接口对齐 SNNLayer，内部数学独立。** 对齐的部分：设备处理、thr/fire_rate 初始化、state_dict 可序列化；独立的部分：前向是 conv2d 不是 linear，STDP 是 patch-wise 不是 tile。

### S2.1 前向传播（单步幅度域）

**背景与动机**：VPRTempo 的"spike"其实是单步幅度值（SetImageAsSpikes 输出 [0,1] 幅度，clamp 上限 0.9，blitnet.py:385），没有膜电位时间演化。conv 前端必须遵守同一约定——**单步卷积 + 减阈值 + clamp**。这是与 SpikingJelly/bindsnet 多步 trace-based LIF 的本质区别，也是 related work 的立足点，绝不能为了"更 SNN"而引入多步仿真。

**详细操作**：
1. `ConvSNNLayer(nn.Module)`，构造参数对齐 SNNLayer 风格：`in_channels, out_channels, kernel_size, thr_range=[0,0.5], fire_rate=[0.2,0.9], ip_rate=0.15, stdp_rate=0.005, p=[p_exc,p_inh], device, inference`。
2. 权重：`self.w = nn.Conv2d(in_ch, out_ch, k, bias=False)`。E/I 结构照搬 blitnet 思想但**按核（输出通道）分配兴奋/抑制角色**而非按元素：初始化时按 p_exc 比例随机指定若干通道为兴奋核、其余为抑制核，掩码形状 [C_out]。理由：卷积核权值共享，按元素稀疏会破坏感受野内的结构学习；按通道分 E/I 保留 BLiTNet 的 E/I 平衡思想且适配卷积。**这个对 blitnet 的偏离要在论文里写明并给理由。**
3. 初始化用 `addWeights` 的卷积版：正态采样（mean=范围中点，std=跨度/6，3σ 原则，blitnet.py:277-280）→ 按符号裁剪（blitnet.py:298-301）→ **每核 L1 归一化**（对应 blitnet.py:319-325，归一化单位从"列"变"核"）。
4. 前向：
   ```python
   def forward(self, x):            # x: [1, 1, H, W]（由调用方从 [1, H*W] reshape）
       z = self.w(x)                # [1, C, H', W']
       z = z - self.thr             # thr: [1, C, 1, 1]，每通道一个阈值
       z = torch.clamp(z, 0.0, 0.9) # 对齐 clamp_spikes（blitnet.py:385）
       return z
   ```
5. 推理模式（inference=True）只保留 w 和 thr，对齐 SNNLayer 推理分支（blitnet.py:101-111）。

**验收**：随机初始化的层前向输出形状正确；发放比例合理（不恒 0 不恒饱和，否则调 thr_range）；sanity 脚本固化在 `experiments/sanity_conv_forward.py`。

### S2.2 WTA 竞争机制（三变体，消融变量）

**背景与动机**：为什么需要 WTA？若没有竞争，feature map 上所有位置都发放、都触发 STDP，核会被所有位置的 patch 平均成"全局均值模板"，学不出选择性。WTA 制造稀疏发放（呼应 BLiTNet 稀疏哲学），保证每次更新只来自最有信心的位置。做三个变体是因为"全局唯一 winner 太稀疏 vs 局部 winner 密度适中"孰优无法先验判断——这是 Table 2 的消融轴。

**详细操作**：
1. 参数 `wta_mode ∈ {'global','local','none'}`，`wta_block=4`。
2. **global**：每通道 argmax，仅该位置保留原值，其余置零（flat argmax → scatter 成 mask → `z * mask`）。
3. **local**：不重叠 4×4 块，每块一个 winner。实现用 reshape/unfold 分块 → 块内 argmax → 置零其余 → 拼回。H',W' 不被 4 整除时**裁掉右/下余数边缘**（简单、无 padding 伪影，论文注明）。主配置 28 输入 k=5 → H'=24，24/4=6 整除，无此问题（56 对照 k=5 → 52/4=13 同样整除；仅 k=3 等组合需要裁边）。
4. **none**：不置零（对照组）。此模式下 STDP 更新用**稠密 M = (0.5−post) 全图**（`conv2d_weight` 公式不变，M 不置零即可，ADR-3）——这也是阶梯 R2 与 Table 2 none 格的实现方式，**同一配置跑一次两处引用**。
5. winner 的 (通道, y, x) 坐标列表要返回/缓存，供 S2.3 patch 提取。
6. mask 作用于 clamp 后的活动，保证下游 feature_layer 看到的也是稀疏表征（否则 WTA 只影响学习不影响表征，实验逻辑不干净）。

**验收**：global 模式每通道非零数 == 1；local == (H'/4)·(W'/4)；none 不变。存一张 mask 前后 feature map 对比图。

**与 ADR-1 的耦合（实现时必须一起处理）**：WTA 模式决定送 feature_layer 的张量形态——local 模式直接把块 winner 值重排为 [1,C,H'/4,W'/4] 再 flatten（"竞争即池化"，零额外计算）；none 模式要补一个独立 4×4 max-pool 凑同一维度链；global 模式输出 [C] 向量（信息瓶颈风险已记录在 ADR-1，若消融中 global 意外胜出需重新设计下游维度）。

### S2.3 卷积 STDP 更新规则（本 idea 的核心公式）

**背景与动机**：blitnet 普通 STDP（calc_stdp 分支二，blitnet.py:555-557）：
`ΔW = η·(0.5−post)·Θ(pre)·Θ(post)`，按元素作用于全连接矩阵。卷积版要回答三个新问题：①更新谁（答：只更新 winner 感受野对应的核-patch 对，由 S2.2 保证）；②多 winner 的更新怎么合并（答：聚合，消融变量）；③E/I 与归一化怎么迁移（答：符号钳制 + 每核归一化）。`(0.5−post)` 项的含义（post<0.5 增强、>0.5 减弱，blitnet.py:550-552）保留——它让 winner 响应趋向中等幅度，自稳定防爆。

**关键设计决策（pre 端形式：中心化是必要条件，不是可选项）**：

*问题分析（直流塌缩）*：spike 编码后的输入**处处 ≥ 0**。实测分布（S1.2 直方图证据 `results/input_histogram_patchnorm.png`，50 张 Nordland spring）：PatchNorm on 时 z-score 被 clip 到 [-1,1]（dataset.py:191），映射后近似均匀分布 + 0/1 边界尖峰，mean≈0.485（"背景恒为 0.5"的直觉不准确——平坦区域只占约 3.5%）；off 时为 gamma 校正后的钟形分布，mean≈0.496，**也无近零暗背景**。在幅度加权规则 `dK = η·(0.5−post)·pre_patch` 下，`pre_patch` 每元素 ≥ 0、`(0.5−post)` 对标量同号——k² 个权重每次同向移动、移动量正比于 pre 幅度。配合保范数归一化，该规则的**不动点是"该通道所有获胜 patch 的归一化加权均值"**，其中含强直流分量（两种模式下都是）：核趋向"均值模板"，Gabor 拟合 R² 会变差，且兴奋核 clamp(min=0) 使权重无法取负，拼不出中心-外周的正负瓣结构。

*裁决*：`pre_mode ∈ {'centered', 'amp', 'heaviside'}` 三选一，**默认 `'centered'`**：
```python
pre_term = (pre_patch - 0.5) if pre_mode == 'centered' else \
           pre_patch if pre_mode == 'amp' else (pre_patch > 0.5).float()
```
- `centered`：不动点变为"获胜 patch 相对背景的平均偏离模式"——那才是边缘/纹理结构。低于背景的位置获得负更新（被 clamp 压向 0），高于背景的位置持续增强 → 稀疏化 + 方向性自然涌现。
- `amp` / `heaviside`：消融对照，用于在 **Table 3b**（S3.3 新增：B2 限定 patch_norm{on,off} × pre_mode{centered,amp} 2×2）里量化"直流塌缩"假设的解释力。修正后的预期（依据实测直方图）：**amp 在 on/off 下都应差于 centered**——off 模式实测也无近零暗背景（gamma 校正抬升暗部），不再"近似中心化"；on/off 的差异更多体现在 patch 内容结构上而非直流。
- 语义说明：blitnet 的 spike forcing 分支本就用幅度而非 Θ(pre)（blitnet.py:484/494），中心化偏离在论文中注明理由（PatchNorm 值域偏移所致）。

**详细操作**：
1. 新函数 `calc_stdp_conv(pre_img, post_map, winners, layer)`：
   - `pre_img`: [1,1,H,W]；`post_map`: WTA 后 [1,C,H',W']；`winners`: S2.2 坐标列表。
2. 对每个 winner (c, y, x)：
   ```python
   pre_patch = pre_img[0, 0, y:y+k, x:x+k]   # [k,k]，无 padding 时感受野直接对齐
   post = post_map[0, c, y, x]                # 标量
   pre_term 按 pre_mode 三选一（见上方设计决策，默认 centered）
   dK = layer.eta_stdp * (0.5 - post) * pre_term * float(post > 0)
   ```
3. **聚合**（`agg_mode`，消融变量）：同通道所有 winner 的 dK 取 **mean**（首选：更新幅度不随 winner 数漂移）或 **sum**（对照），加到 `layer.w.weight[c, 0]`。
4. **符号钳制**：语义照搬 blitnet.py:581-584，按 S2.1 的通道级掩码。细则偏离：blitnet 把兴奋权重 clamp 到 [1e-6, 10]，会把 0 顶成 1e-6、破坏稀疏；卷积版改为兴奋核 `clamp(min=0, max=10)`、抑制核 `clamp(min=-10, max=0)`，只保证符号不翻转。**偏离处在论文注明理由。**
5. **归一化**：每次更新后，被更新的核做**保范数 L1 归一化**（恢复到更新前的核范数），防止幅度漂移（对应 addWeights 的归一化思想）。
6. **Homeostasis**：默认**关**，留消融开关 `conv_homeostasis`。裁决理由见 ADR-3：blitnet 需要它是因为全连接版没有 WTA；卷积版已有 WTA 稀疏 + 保范数归一化 + ITP 三重稳定。若玩具测试发现核范数/发放率失控，再打开。
7. 发放率调制（blitnet.py:480-484 的 `mpre = pre / fire_rate`）：默认**关**，留作附录消融。
8. **向量化实现（硬性要求，ADR-3）**：正式路径用 `torch.nn.grad.conv2d_weight` 一行完成全通道更新（构造响应图 M = winner 处置 (0.5−post)，其余为 0；更新即 pre_img 与 M 的互相关 = 卷积权重梯度），走 cuDNN；agg='mean' 时除以每通道 winner 数。Python 循环版仅作玩具测试的逐元素对拍。

**验收**：玩具测试（固定输入竖直边缘图 + 若干自然图，训练若干步）：
1. 对应核在边缘位置权重显著增大（数值断言）；
2. 更新只发生在 winner 通道；
3. 1000 步内核范数曲线稳定不发散；
4. **直流防线**：每核 **DC/AC 比** `|mean(K)| / (std(K) + ε)` 随训练**下降**（centered 模式必须满足；amp 模式预期不满足——两相对照本身就是设计决策的实证）。**不要用 `|mean|/L1`**：兴奋核 clamp(min=0) 后所有元素 ≥0，此时 Σw=Σ|w|，`|mean|/L1 ≡ 1/k²` 是与核形状无关的常数——平坦亮斑和稀疏边缘核算出同一个数，断言永远"通过"，防线形同虚设。DC/AC 比则区分度明确：平坦核 std→0 比值爆表，结构核 std 大比值小；
5. **去相关防线**：核间两两余弦相似度均值随训练**不上升**（防止所有核塌向同一模板）。
断言 4/5 在写任何大规模训练代码之前就能判定规则对不对——玩具测试不过，不进 S2.5。

### S2.4 卷积版 ITP 阈值可塑性

**背景与动机**：blitnet 的 ITP（Δθ = η_ITP·(Θ(x)−f)，blitnet.py:597-606）是每神经元的，配套每神经元不同目标发放率（blitnet.py:164-167 线性分配；理论依据：低发放率神经元学稀疏特异特征、高发放率学泛化特征）。卷积层没有"每神经元"，自然映射为**每通道一个阈值、每通道一个目标发放率**：通道级发放率分化让不同通道变成"稀疏特异检测器"和"通用纹理检测器"。没有 ITP，WTA+STDP 容易导致少数通道垄断 winner（死通道）。

**详细操作**：
1. `fire_rate` 形状 [1,C,1,1]，[0.2, 0.9] 线性分配（对齐 blitnet.py:164-167 的 fstep 逻辑）。
2. 每个训练样本（**observed 必须在 WTA 之前的 clamp 图上统计**）：
   ```python
   observed = (pre_wta_map > 0).float().mean(dim=(2,3), keepdim=True)  # 每通道实际发放率（WTA 前！）
   layer.thr.data += layer.eta_ip * (observed - layer.fire_rate)
   layer.thr.data.clamp_(min=0)   # 对齐 blitnet.py:606
   ```
   **为什么不能在 WTA 后统计**：local WTA 下每通道恒有 (H'/4)×(W'/4) = 36 个 winner，post-WTA 图上每通道发放率被结构钉死为常数 36/576 ≈ 6.25%——`observed − f` 对所有通道同号同值，ITP 的差异化调节完全失效。global WTA 同理（恒 1 个 winner）。
3. 死通道监控（信号必须换）：local WTA 下 winner 计数被结构钉死，"0-winner 报警"永远不会触发。改用两个有效信号——**每通道 winner 平均幅度 ≈ 0**（块内全零时 argmax 选出的也是 0）+ **pre-WTA 发放率长期贴 0**。任一触发即报警（ITP 失效信号，调 η_ITP 或 f 范围）。

**验收**：训练后各通道 **pre-WTA** 经验发放率与目标 f 的秩相关显著为正；无 winner 均幅 ≈ 0 的死通道。

### S2.5 接入逐层训练框架

**背景与动机**：不重写训练框架，而是**插入**它——`train_new_model` 的 layer_dict 排序（VPRTempoTrain.py:667）+ prev_layers 冻结前向（VPRTempoTrain.py:429-434）已是逐层训练的完整实现。改框架收益低风险高（B0 回归），插入收益相同风险低。

**详细操作**：
1. `VPRTempoTrain.__init__` 加分支：`args.frontend == 'conv_stdp'` 时，在 add feature_layer **之前** add conv 层（layer_dict 顺序 0,1,2 → 自动先训 conv）。
2. **维度重算（照 ADR-1）**：`self.input = C × (H'/pool) × (W'/pool)`（主组合 = 32×6×6 = 1,152），由 dims/k/C/pool 参数在 `__init__` 算出，硬编码零容忍；feature_layer dims 随之改（VPRTempoTrain.py:161-162），feature = 2×input 规则保留。`--dims` 语义不变（仍是图像尺寸）。
3. **框架改动范围（照 ADR-2，共 3 处小分支，原路径一行不改）**：
   - `train_new_model` 层循环（VPRTempoTrain.py:674）：`isinstance(layer, ConvSNNLayer)` → 调 `train_conv_layer`；
   - `train_model` 的 prev_layers 冻结前向（VPRTempoTrain.py:429-434）：conv 层走 conv 前向路径（reshape→conv→减thr→clamp→WTA→池化→flatten）；
   - `VPRTempo.py` 推理前向：同一分支镜像，inference=True 加载。
4. 新函数 `train_conv_layer(train_loader, layer, model)`（结构镜像 train_model，VPRTempoTrain.py:327-485）：
   - epoch 数用 `conv_epoch`（独立参数，默认 2）而非 `self.epoch`；
   - 每样本：reshape [1,H*W]→[1,1,H,W] → 前向（S2.1）→ WTA（S2.2）→ `calc_stdp_conv` 向量化路径（S2.3）→ ITP（S2.4）→ 退火；
   - 退火复用 `_anneal_learning_rate`，但 **T 单独 = 数据库图像数 × conv_epoch**（conv 层步数与 feature/output 层不同，共用 self.T 会退火错误；self.T 的计算见 VPRTempoTrain.py:181-184）。
5. **多模块共享前端**：conv 层只在 module 0 上训一次，权重拷给其余模块（`load_state_dict`）。理由：①省算力；②叙事"通用视觉特征"，模块间特征空间一致，feature_layer 学到的东西跨模块可比；③若每模块各训，模块边界特征空间跳变，无依据地伤害输出拼接。**论文明确写此设计决策。**（500 地主实验是单模块，此条先为大规模实验预留。）
6. 保存：save_model 的 state_dict 机制（VPRTempoTrain.py:509-540）自动覆盖新层；推理侧构造层序必须与训练侧一致（同 layer_dict 顺序、同维度）；模型命名加 `_CONVC<C>K<k>` 标记，防覆盖 B0 模型。

**验收**：端到端跑通训练 + 推理（C=16, k=3, conv_epoch=1, 500 地）；保存/加载往返后推理一致；**B0 回归**：不带 `--frontend` 时行为与 main 逐比特一致。

### S2.6 B1：Random Conv 冻结对照

**背景与动机**：随机卷积 + 只训读出层是惊人强的基线（random features 文献）。若 B2 ≈ B1，"STDP 学到东西"不成立，Gate 1 失败。B1 与 B2 **必须共享同一随机初始化**（同 seed 同 init），唯一变量是"训没训"。

**详细操作**：`--frontend random_conv`：结构同 B2，但给 ConvSNNLayer 加 `frozen=True` 标志，train_new_model 分发时直接跳过其训练（进 trained_layers）。配置固定 B1/B2 同 init seed。

**验收**：B1 可前向可双轨评测；同 seed 下 B1 初始核与 B2 训前初始核逐元素一致。

### S2.7 B3：两层 Conv-STDP

**背景与动机**：回答"增益是不是只是加了深度"。两层逐层无监督（conv1 训完冻结 → conv2）是 BLiTNet 逐层哲学的自然延伸，layer_dict 机制再次免费复用。

**详细操作**：`--frontend conv_stdp2`：conv1: 1→C1；conv2: C1→C2。维度链照 ADR-1 逐层算清并写进配置，**拍板方案（28×28 主配置）**：

```
conv1（k=5, padding=2）：28 → 28（保持分辨率；STDP 的 conv2d_weight 调用须传相同 padding 保持对齐）
conv2（k=5, 无 padding）：28 → 24
4×4 池化（local WTA 块 winner 图）：24/4 = 6 → flatten = C2×6×6 = 32×36 = 1,152 ✓ 与 B2 严格同维
```

注意：两层都不 padding 的话是 28→24→20，池化后 5×5=25×C2=800 ≠ 1152，B2/B3 下游维度就不可比了——**conv1 的 padding=2 是同维的关键**，不是可选项。**禁止先写代码后算维度**——feature_layer 参数量随 flatten 维度平方增长（ADR-1 的账）。类型分发天然支持多层（按 layer_dict 顺序各训各的）。

**验收**：双层训练完成、维度链正确；轨 B 数字可进主表。

### S2.8 B4：同结构 CNN + BP 参照

**背景与动机**：审稿人必问"为什么不用端到端 BP"。标准回答："本文研究无 BP 的可行边界"——需要 B4 作上界参照系量化"边界离上界多远"。不声称超越 CNN。

**详细操作**：
1. `experiments/train_cnn_ref.py`：同数据通路（同 ProcessImage、同 500 地），Conv(k,C)→ReLU→(可选第二层)→flatten→Linear(500 类)，交叉熵 + Adam + 早停。
2. 评测以**轨 B 协议为主**（backbone 特征 → cosine → recallAtK）保证与 B0–B3 可比；其监督读出准确率作为"轨 A 等价"附注，论文里说明协议差异。
3. 训练时间/FLOPs/参数量一并记录（实验 1.5 需要 B4 列）。

**验收**：B4 显著优于 B0（否则 CNN 参照本身有问题，先排查）；数字进主表。

### S2.9 B5：手工 Gabor 滤波器组前端

**背景与动机**：回应"学到 Gabor 就直接用手工 Gabor"的必问（§0.4）。手工滤波器前端在 conv-STDP 文献有先例（Kheradpisheh et al. 2018 第一层即手工 DoG），这不是稻草人。B5 还兼作 B2 的"目标形态参照"：S3.1 的核可视化可以直接和 B5 的真 Gabor 并排。

**详细操作**：
1. `IDEA1-covstdp/src/gabor_frontend.py`：生成 Gabor 组（4 方向 {0,45,90,135°} × 2 频率 × 2 相位 × 2 尺度，凑齐 C=32；k 与 B2 主组合一致），参数覆盖应大致均匀采样方向-频率空间，不要手工调到"看起来好"（那是过拟合先验）。
2. 复用 ConvSNNLayer 结构，权重直接载入 Gabor 组并 `frozen=True`（同 S2.6 机制），其余通路（WTA、池化、feature/output 层训练）与 B2 完全相同——**唯一变量是核的来源**。
3. **负瓣保护（易踩的坑）**：Gabor 核自带负瓣（这正是它的表达力来源），而 ConvSNNLayer 的兴奋通道 clamp(min=0)。**载入 Gabor 权重时必须跳过符号钳制与保范数归一化**——frozen 标志要同时旁路这两条路径，仅确认初始化不经过 addWeights 的钳制是不够的，训练中每个 update 后的 clamp/renorm 也必须对 B5 关闭。
4. **不对称性声明（叙事素材，不是缺陷）**：B5 用带符号核、B2 兴奋核非负——论文中显式说明，并正好给 ON/OFF 双通道编码（S3.3 附录探索）提供动机："无 BP 规则下负瓣由 OFF 通道表达"。
5. 双轨评测，进 Table 1。

**验收**：B5 双轨数字落在合理区间（预期 ≥ B1，因为它是知情设计）；若 B5 < B1 需排查 Gabor 参数覆盖。

---

## 5. 阶段 3：实验

### S3.1 实验 1.2：卷积核可视化与量化分析（Figure 2）

**为什么先做它**：成本最低（训完的核直接画图）、反馈最快——**它是 STDP 是否工作的第一道定性证据**。若核没有结构，不必烧主表算力，直接回头查 S2.3。

**详细操作**：
1. `experiments/viz_kernels.py`：加载 B2（及 B3 各层）模型 → `conv_layer.w.weight` [C,1,k,k] → 每核归一化到 [0,1] → matplotlib grid；并排画同 seed 随机初始核（B1）。
2. 量化三指标：
   - **Gabor 拟合优度**：每核拟合 2D Gabor（scipy.optimize 最小二乘），报 R² 分布（中位数 + IQR）；B2 应显著高于 B1（Mann-Whitney）。
   - **方向选择性**：拟合成功核的方向角直方图 + 合成长度（resultant vector length）。
   - **稀疏度**：|w| < 0.1·max 的元素占比，B1 vs B2。
3. 输出：`results/fig2_kernels.png` + `results/table_kernel_stats.json`。

**验收**：图可直接进论文草稿；B2 vs B1 统计差异可见（完全没有 → 亮红灯回查 S2.3）。

### S3.2 实验 1.1：主表（Table 1）

**详细操作**：
1. 矩阵：B0–B5 × 3 seeds × 双轨 × PatchNorm=on（off 归实验 1.4）。**B2 主组合钉死（= 阶梯 R4）**：C=32, k=5, conv_epoch=2, WTA=local(4×4), agg=mean, pre_mode=centered, **ITP=on, E/I=on, homeostasis=off**。B5（Gabor 滤波器组）无需训练，3 seeds 只影响下游 feature/output 层初始化。
2. 指标：Recall@1/5/10/25 为主 + recall@100%precision 互补（指标决策见 S1.4），PR 曲线图进正文。
3. 执行：run_exp.py 批量 → `experiments/make_table1.py` 汇总 mean±std。
4. 统计判据（提前承诺）：B2−B1（轨B）、B2−B0（轨A）报差值 ± 联合 std；3 seed 太少不做强显著性声明，以效应量为主，措辞谨慎。
5. **B2 vs B5 单段分析**（主表自带的小节素材）：学习 vs 手工设计，三种结果的叙事预案见 §0.4。

**验收**：Table 1 完整，Gate 1 / Gate 1.5 / Gate 2 判定明确（§6）。

### S3.3 实验 1.3 / 1.4：消融

**详细操作**：
1. **Table 2（WTA × 聚合）**：{global, local, none} × {mean, sum}，固定 C=32/k=5/B2，双轨。意义：WTA 稀疏度控制 STDP 样本效率，聚合方式控制更新尺度——两者都可能"差到不可用"，必须扫。
2. **C × k 主组合**：{16,32,64} × {3,5,7} 选 6–9 组（对角线选法：16/3, 32/3, 32/5, 32/7, 64/5, 64/7），轨 B 优先（便宜），代表组合补轨 A。
   **最优 k 的预先承诺选择程序（防止事后挑数）**：
   - 固定 C=32、其余默认，先在**轨 B** 上扫 k∈{3,5,7}，按 3-seed 平均 Recall@K 选主组合；
   - 用两个辅助证据交叉验证而非只看一个数字：核可视化的 Gabor 拟合优度（k=3 在 28×28 上可能学不出完整周期结构；k=7 感受野占比过大、样本效率差）；效率表（FLOPs 按 k² 涨）；
   - 选定的 k 在轨 A 与 ORC 上**复核**，复核不成立如实报告，不换数。
   - 理论锚点（写进方法论）：k 应与 PatchNorm 窗口同量级或更小——28×28 下 patches=7，k=5 ≈ 0.7×7 合理；k=1 退化为逐点变换（无空间结构），k=H 退化为全局模板匹配（≈全连接），最优值必在中间，{3,5,7} 覆盖低/中/高段。
   - **主文坚持单一 k**：多尺度并联（Inception 式 3/5/7）使前端算力 ×3、叙事变浑，且低分辨率 VPR 图上多尺度收益有限；若审稿人要求，作为附录消融（3/5/7 并联、通道三等分保持总 C 不变），届时基础设施齐全，成本约一天算力。
3. **Table 3（PatchNorm 2×4，本文最独特分析）**：PatchNorm {on,off} × {B0,B1,B2,B3}，双轨。假设：on 时 Conv-STDP 增益收窄（局部高通已提取边缘，conv 没新东西可学）、off 时增益放大。**两个方向都有结论可写**：收窄 → "VPRTempo 预处理已隐式完成卷积前端的工作"；放大 → "conv 前端与朴素编码互补"。按实际方向组织叙事。
4. **Table 3b（直流塌缩的直接证据格）**：B2 限定，patch_norm {on,off} × pre_mode {centered,amp} 2×2，双轨。修正后的预期（依据 S1.2 实测直方图）：amp 在 on/off 下都应明显差于 centered（off 模式实测无近零暗背景，两种模式输入都处处 ≥0）；on/off 差异更多体现 patch 内容结构差异。这是 S2.3 设计决策的量化验证，也是审稿人问"为什么 centered"时的数据答案。
5. **conv epoch ∈ {1,2,4}**（附录）：验证 STDP 收敛与过拟合（无监督也会过拟合：核塌缩 / winner 垄断）。
6. **E/I 通道拆分消融**（Table 2 附属行，一个开关）：{E/I 拆分（默认） vs 全兴奋核}，固定 B2 主组合，双轨。背景：blitnet 中抑制的三个经典角色在卷积前端里有两个已被替代——竞争由显式 WTA 接管（Diehl&Cook/Kheradpisheh 用侧抑制实现 WTA）、稳态由 WTA 稀疏 + 保范数归一化 + ITP 承担（homeostasis 默认关）；剩下唯一角色是"反对比度模式检测"的特征多样性，且 centered pre-term 下兴奋核更新已带符号，抑制核边际贡献未经验证。两种结果都可写：全兴奋 ≈ E/I → 显式 WTA 接管了抑制的经典角色（有意思的发现）；E/I 明显更好 → BLiTNet 的 E/I 思想在卷积域同样承重。
6. **编码探索（附录，可选）**：主实验一律用原始单步幅度编码（`SetImageAsSpikes`，像素/255），理由：①B0–B3 可比性，唯一变量是 conv 前端；②换多步编码（发放率/时延）会摧毁"无多步仿真"卖点——那是 SpikingJelly 参照行（S3.5）的领地。可选探索：**ON/OFF 双通道编码**（签名信号拆正/负两通道，类 DoG 中心-外周），与 conv-STDP 预期的中心-外周核结构天然契合；做则只跑 B2 主配置轨 B，作为附录一小节。注意它会改变输入通道数（1→2），不进主表。

**验收**：三张表齐全，每张至少一段可直接写进论文的观察。

### S3.4 实验 1.5：效率表（Table 4）

**背景与动机**：VPRTempo 的卖点是快，审稿人对"加 conv 前端后还快不快"极度敏感。B0 / B2 / B4 三列：训练墙钟时间（同硬件）、单查询前向 FLOPs、峰值内存、参数量。

**详细操作**：`experiments/benchmark.py`：FLOPs 手工算（conv: 2·C_in·C_out·k²·H'·W'；linear: 2·in·out）或 torch.profiler；训练时间从 run_exp 日志提取；显存 `torch.cuda.max_memory_allocated`。

**验收**：Table 4 三列齐全；B2 相对 B0 的开销增幅有量化数字（叙事素材："x% 开销换 y 点 Recall"）。

### S3.5 补充实验（审稿预案驱动）

1. **Oxford RobotCar 450 地复跑主表核心配置**（B0/B1/B2 + PatchNorm on/off，3 seeds）。回答"只有一个数据集？"。约 1 天算力，用现有 `orc-*.csv` 通路。
2. **SpikingJelly conv-STDP 参照行**：回答"和现有 SNN 库的区别"。用阶段 2 原型改最小多步 LIF conv-STDP 前端（T=10 步），只跑轨 B 主配置 1 seed。不追求赢它，追求"单步规则以 ~1/T 开销达到可比性能"的对比叙事。
3. 两项进附录，主文各一句话引用。

**写作层面的两个命名/定位决策（不用跑实验，但决定审稿风险）**：

1. **不要笼统叫"STDP"**。单步幅度域没有 spike timing，`ΔK = η·(0.5−post)·pre` 在数学上是带自稳定项的归一化 Hebb / Oja 型规则。审稿人里只要有一个神经形态背景的，"你的 STDP 里 timing 在哪"就是必中评论。论文措辞定为：*single-step competitive Hebbian plasticity*，并显式说明它是 BLiTNet/VPRTempo STDP 公式在单步极限下的退化形式（继承关系写清楚，既诚实又自然衔接 VPRTempo 系文献）。代码里模块名保留 `conv_stdp` 无妨（实现血缘），正文命名要严谨。
2. **"学到的不就是 Gabor 吗，直接用手工滤波器组不就行了"**——这正是 B5 存在的理由（§0.4），主文用 B2 vs B5 一行数据 + Figure 2 的核对比回应，不要在 related work 里空辩。

---

## 6. 决策门与风险（提前承诺，防止事后移动球门）

| 门 | 判据 | 通过含义 | 不通过的动作 |
|---|---|---|---|
| Gate 0（阶段1出口） | B0 与第 -1b 步种子化 main 参照一致（±2 点）且 feat 默认参数逐比特回归通过 | 基础设施可信 | 不进阶段 2，先排查 |
| Gate 1 | 轨B：B2 > B1，差值 > 3-seed 联合 std | 可塑性学到结构（核可视化佐证） | 回查 S2.3/S2.4（先查直流防线断言 4/5）；仍不过则整个 idea 重估 |
| Gate 1.5 | 轨B：B2 ≥ B5（Gabor 手工组）− 1×std | 学习达到/超越手工滤波器 | B2 明显 < B5 → 规则没学到结构，回查 pre_mode 与 WTA；若 B5 反而最强，故事改为"手工前端 + SNN 读出"并弱化学习叙事 |
| Gate 2 | 轨A：B2 或 B3 > B0 | 空间归纳偏置帮助完整系统 | **退路启动**：改分析型故事 |

**退路剧本（Gate 1 过、Gate 2 不过）**：故事改为分析型——"无 BP 卷积可塑性学到何种特征、为何与 spike-forcing 读出失配"。轨 B 正结果 + 核可视化 + PatchNorm 交互分析仍完整成立，失配本身是贡献（顺势引出创新点 2"读出适配"）。投稿目标从 ICRA/RA-L 转向 Frontiers/期刊。

---

## 7. 篇幅与时间线

- 目标：ICRA / RA-L，6 页 + 参考文献。正文：实验 1.1（Table 1）+ 1.2（Figure 2）+ 1.3/1.4 精选各一张表；其余附录。
- 时间线（全职）：

| 阶段 | 内容 | 时长 |
|---|---|---|
| 1 | S1.1–S1.5 基础设施 + 双轨基线 | 3–4 周 |
| 2 | S2.1–S2.8 ConvSNNLayer + 全部变体 | 4–6 周 |
| 3 | S3.1–S3.5 实验 | 4 周 |
| — | 写作 | 3 周 |
| 合计 | | **3.5–4.5 个月** |

---

## 8. Fork 会话指引（给未来的实现会话）

每个 fork 会话开工时，把对应步骤卡片贴给它，并遵守：

1. **一次只实现一个 S 编号卡片**；卡片的"验收标准"是该会话的完成定义。
2. 新代码放 `IDEA1-covstdp/src/` 或 `experiments/`；对 `vprtempo/` 的修改必须**默认行为不变**（B0 回归：不带新参数时输出与 main 一致）。
3. 每个卡片完成后：更新本文件清单勾选 + `results/` 落盘 + 提交到功能分支，**不直接合 main**（main 只收阶段级合并）。
4. 分支命名：`feat/convstdp-s<编号>-<短名>`，如 `feat/convstdp-s21-conv-forward`。
5. 所有实验必须能由 `experiments/run_exp.py` + 配置复现，禁止"手工跑了一次"的孤儿结果进表。

### Fork 前一次性锁定的决策清单（改动需回到本文档修订并记录原因）

| # | 事项 | 锁定值 |
|---|---|---|
| 1 | B2 主组合完整配置 | C=32, k=5, conv_epoch=2, WTA=local(4×4), agg=mean, pre_mode=centered, **ITP=on, E/I=on, homeostasis=off**（= 阶梯 R4，S3.2 已同步） |
| 2 | 轨 B 特征点 | 池化后 flatten 的 1,152 维（feature_layer 实际看到的向量），全变体统一 |
| 3 | Gate 0 判据 | 第 -1b 步种子化 main 参照 ±2 点 + feat 默认参数逐比特回归（论文 3300 地数字只作量级 sanity） |
| 4 | 调参协议 | 轨 B 单 seed 粗调 → 锁定写入配置文件 → 3 seed 正式跑（防止隐式调参泄漏进主表） |
| 5 | seed 细节 | 三件套 + `worker_init_fn`（num_workers>0 时）+ `PYTHONHASHSEED`（S1.3） |
| 6 | 诊断标准化 | 每个训练 run 固定落盘 JSON：pre-WTA 发放率、winner 平均幅度、核范数曲线、thr 曲线、DC/AC 比、核间余弦——并入 S2.3/S2.4 验收 |
| 7 | 时间线瘦身预案 | 窗口期紧张时的砍单顺序：编码探索（S3.3 第 7 条）→ B3 → ORC（B5 便宜且叙事价值高，最后砍） |

### 总检查清单
- [x] S1.1 配置系统与 run_exp.py
- [x] S1.2 PatchNorm 开关化
- [x] S1.3 seed 三件套
- [x] S1.4 eval_retrieval.py（轨 B）
- [x] S1.5 B0 双轨基线（Gate 0）✅ 通过（results/table_baseline_b0.md）
- [x] S2.1 ConvSNNLayer 前向（src/conv_snn_layer.py，sanity 15/15 通过）
- [x] S2.2 WTA 三变体（winner_mask 结构性断言 + results/wta_mask_demo.png）
- [ ] S2.3 calc_stdp_conv（聚合 + 钳制 + 归一化）
- [ ] S2.4 卷积 ITP
- [ ] S2.5 接入 train_new_model + 推理侧 + 共享前端
- [ ] S2.6 B1 Random Conv 对照
- [ ] S2.7 B3 两层
- [ ] S2.8 B4 CNN 参照
- [ ] S2.9 B5 Gabor 滤波器组前端
- [ ] S3.1 核可视化（Figure 2）
- [ ] S3.2 主表（Table 1，Gate 1/2 判定）
- [ ] S3.3 消融（Table 2/3/3b + 附录）
- [ ] S3.4 效率（Table 4）
- [ ] S3.5 ORC 复跑 + SpikingJelly 参照
