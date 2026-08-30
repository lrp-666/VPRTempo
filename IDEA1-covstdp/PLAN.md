# IDEA1: Conv-STDP 卷积前端 for VPRTempo —— 详细实施规划

> 目标：在 VPRTempo 的 spike 编码之后、`feature_layer` 之前插入无监督 Conv-STDP 卷积前端，
> 用**双轨评测**隔离验证"空间归纳偏置 + 无 BP 可塑性"对 VPR 的实际贡献。
> 代码引用基于 main 分支当前版本（VPRTempoTrain.py / blitnet.py / metrics.py / dataset.py）。

---

## 0. 创新点一句话

VPRTempo 把图像展平成向量后做全连接 SNN，丢失了空间结构；本文证明一个**单步幅度域 Conv-STDP 前端**（无多步仿真、无 BP）能学到可解释的 oriented-edge 卷积核，并通过双轨评测精确定位其增益来源与失配环节。

---

## 阶段 1.1：基础设施与数据通路（3–4 周）

### 1.1.1 分支与目录策略
- 本目录 `IDEA1-covstdp/` 为实验工作区：`src/`（新模块代码）、`experiments/`（配置与脚本）、`results/`（结果表与图）。
- 功能开发在 `feat/convstdp-*` 分支进行，稳定后合回 main；不动 `vprtempo/` 原有类的默认行为，所有新行为走开关参数。

### 1.1.2 数据通路（主实验）
- **Nordland 500 地**：参考 = spring + fall（拼接为数据库），查询 = summer。
- 沿用现有 CLI 参数：`--dataset nordland --database_places 500 --query_places 500`，CSV 标注用 `vprtempo/dataset/nordland-*.csv`。
- **3 个 seed**（{0,1,2} 或 {42,123,2024}，固定写入实验配置），每个格子报 mean±std。
- 补充数据集（应对审稿）：Oxford RobotCar 子集 450 地，重跑主表，成本约 1 天。

### 1.1.3 PatchNorm 开关化
- 位置：`vprtempo/src/dataset.py:422`（`PatchNormalisePad(self.patches)` 调用处）。
- 加 `--patch_norm {on,off}` 参数（默认 on，保持现有行为不变）；off 时跳过步骤 4，直接缩放编码。这是实验 1.4 的前提。

### 1.1.4 轨 B 评测脚本：`eval_retrieval.py`（新写）
- 流程：加载训练好的前端 → 对数据库/查询图像提取特征（conv 前端输出或 feature_layer 输出）→ flatten → cosine 相似度矩阵 S → 复用 `metrics.py:134` 的 `recallAtK(S, GT, K)`。
- 复用现有 GT 构造逻辑（单位矩阵 + `GT_tolerance` 对角膨胀 + `skip` 偏移），保证与轨 A 完全可比。
- 输出 PrettyTable：Recall@1/5/10/15/20/25，格式与轨 A 一致。

---

## 阶段 1.2：ConvSNNLayer 实现（4–6 周）

新类 `ConvSNNLayer`（建议放 `IDEA1-covstdp/src/conv_snn_layer.py`，成熟后再考虑并入 `vprtempo/src/`）。

### 1.2.1 前向传播（单步幅度域）
- 输入：spike 编码后的图像幅度（[1,H,W]，与 `SetImageAsSpikes` 输出对齐）。
- `F.conv2d` 一步卷积 → 加兴奋/抑制贡献 → 与阈值 `thr` 比较 → Θ(·) 得到 [C_out, H', W'] 二值（或幅度）活动。
- **不做多步 LIF 仿真**，与 VPRTempo 单步编码自洽——这是与 SpikingJelly/bindsnet conv-STDP 的核心差异（related work 论点）。

### 1.2.2 兴奋/抑制通道结构
- 照搬 blitnet 的 E/I 思想：核权重拆兴奋掩码（正）与抑制掩码（负），比例沿用 blitnet 默认。
- 初始化：随机核（与 B1 的 Random Conv 用**同一初始化分布**，保证 B1 vs B2 只差"训没训"）。

### 1.2.3 WTA 竞争（消融变量，两变体都实现）
- **Global WTA**：每通道全图 argmax，仅 1 个 winner。
- **Local WTA**：feature map 切不重叠 4×4 块，每块 1 个 winner。
- **无 WTA**：对照组。
- winner 位置之外的后层活动置零（mask），保证 STDP 只发生在 winner 的 receptive field 上。

### 1.2.4 卷积 STDP 更新规则
- 对每个 winner：取 receptive field 对应前层 patch `pre_patch ∈ [1,1,k,k]` 与 winner 后层幅度 `post`（标量），按 blitnet 原规则：

  ```
  ΔK_c = η_STDP · (0.5 − post) · Θ(pre_patch) · Θ(post)
  ```

- **多 winner 聚合**（消融变量）：同通道多个 winner 的 ΔK_c 取 mean（首选）或 sum，更新共享卷积核。
- **符号钳制**：照搬 `blitnet.py:581-584` —— 兴奋核 clamp 到 [1e-6, 10]，抑制核 clamp 到 [-10, -1e-6]。
- **归一化**：每次更新后对每通道核做 L1 或 L2 归一化（对应 blitnet 权重归一化思想），防止核幅度漂移。

### 1.2.5 ITP 阈值可塑性（卷积版）
- 把 `blitnet.py:597-606` 的 Δθ = η_ITP·(Θ(x) − f) 从"每神经元"改为"**每通道**"：用该通道 feature map 平均发放率代替单神经元活动。
- 目标发放率 f 每通道一个，在 [0.2, 0.9] 线性分配（沿用 `blitnet.py:164-167` 的思想）。
- 阈值保持非负（`blitnet.py:606`）。

### 1.2.6 接入逐层训练框架
- 训练顺序：conv_layer（无监督 STDP，遍历数据库若干 epoch）→ **冻结** → feature_layer → output_layer。
- 复用 `train_new_model()`（`VPRTempoTrain.py:591`）的 `layer_dict` 排序机制（`VPRTempoTrain.py:667` 按序逐层训练）：conv 层 `add_layer` 在最前，`trained_layers` 机制天然支持"前层冻结只做前向"。
- 学习率退火复用 `_anneal_learning_rate`（`VPRTempoTrain.py:275`），conv 层的 T 单独定义：**T = 数据库图像数 × conv_epoch**。
- **多模块决策**：conv 前端在所有模块间共享一份（用全部数据库图像训一次），不每模块各训——省算力，且叙事更合理（"通用视觉特征"）。论文中明确写出此设计。

---

## 阶段 1.3：评测协议与实验矩阵

### 1.3.1 统一双轨评测（每个变体都做）
| 轨道 | 路径 | 指标 |
|---|---|---|
| 轨 A | 完整训练 → `run_inference` 现有流程 | Recall@1/5/10/25（`metrics.py:134`） |
| 轨 B | 前端输出 → flatten → cosine 最近邻（`eval_retrieval.py`） | 同上 |

轨 B 绕开 output layer 的 spike-forcing 读出，是**干净隔离 encoder 质量**的关键。

### 1.3.2 主实验矩阵（实验 1.1，Table 1；每配置 × 3 seeds）
| 编号 | 前端 | 前端训练 | 评测 |
|---|---|---|---|
| B0 | 无（原 VPRTempo） | — | 轨A + 轨B |
| B1 | Random Conv | 冻结不训 | 轨A + 轨B |
| B2 | Conv-STDP（1 层） | 无监督 STDP | 轨A + 轨B |
| B3 | Conv-STDP（2 层） | 逐层无监督 | 轨A + 轨B |
| B4 | 同结构 CNN | BP（经 output layer 反传或线性读出） | 轨A + 轨B |

主组合：C=32, k=5 起步；通道数 {16,32,64} × 核大小 {3,5,7} 选 6–9 组，不做全网格。

### 1.3.3 消融（实验 1.3 / 1.4）
- **WTA**：global / local(4×4) / 无 WTA（Table 2）。
- **winner 聚合**：mean / sum（Table 2）。
- **PatchNorm 交互**（实验 1.4，Table 3，本篇最独特分析）：2×4 设计 = PatchNorm {on,off} × 前端 {B0,B1,B2,B3}。
  - 假设：PatchNorm on 时 Conv-STDP 增益收窄（局部高通已提取边缘），off 时增益放大。两个方向都有结论可写。
- **conv epoch** ∈ {1,2,4}（附录）。

### 1.3.4 决策门（提前承诺判据，写进实验计划）
- **Gate 1**：轨 B 上 B2 > B1（超出 3-seed 标准差）→ STDP 确实学到结构。
- **Gate 2**：轨 A 上 B2/B3 > B0 → 空间归纳偏置对 VPRTempo 整体有帮助。
- **退路**：Gate 1 过、Gate 2 不过 → 论文转向分析型定位："STDP 特征有效但与 spike-forcing 读出失配"，顺势进入创新点 2（读出适配），投稿转向 Frontiers/期刊。

---

## 实验 1.2：卷积核可视化（Figure 2，低成本高说服力）
- 训练后 [C_out,1,k,k] 核画成 grid，与随机初始化核并排。
- 预期：STDP 核呈 oriented edge / 中心-外周结构。即使量化提升不大，"核出现可解释结构"本身是 Gate 1 的定性证据。
- 量化指标：Gabor 拟合优度（R²）、方向选择性分布（方向直方图集中度）、稀疏度（核近零元素比例）。

## 实验 1.5：效率数据（Table 4，必须有）
- 训练时间、单查询前向 FLOPs / 内存、参数量，对比 B0/B2/B4。
- VPRTempo 系读者看重效率；强调 Conv-STDP 单步规则 vs 多步 trace-based LIF 的开销差异。

---

## 审稿人必问 + 预先准备
1. **"和 SpikingJelly/bindsnet 的 conv-STDP 有什么区别？"**
   → related work 一小节：那些是多步 trace-based LIF STDP；本文是单步幅度域规则，与 VPRTempo 编码自洽，无多步仿真开销。实验上加一行 SpikingJelly conv-STDP 前端参照（用阶段 1.2 之前的原型代码，不白做）。
2. **"为什么不用端到端 BP？"**
   → 论文定位就是研究无 BP 的可行边界，B4 就是参照系；不声称超越 CNN。
3. **"只有一个数据集？"**
   → Oxford RobotCar 子集（450 地）重跑主表，成本约 1 天。

---

## 篇幅与时间线
- 目标会议：ICRA / RA-L，6 页 + 参考文献。实验 1.1 + 1.2 为核心，1.3 / 1.4 挑进正文，其余附录。
- 时间线（全职投入）：
  | 阶段 | 内容 | 时长 |
  |---|---|---|
  | 1.1 | 基础设施 + 数据通路 + eval_retrieval.py | 3–4 周 |
  | 1.2 | ConvSNNLayer 实现与调通 | 4–6 周 |
  | 1.3 | 主实验 + 消融 | 4 周 |
  | — | 写作 | 3 周 |
  | 合计 | | **约 3.5–4.5 个月** |

- 风险备案：Gate 2 不过但 Gate 1 过 → 改分析型故事（见 1.3.4 退路），投 Frontiers/期刊更稳。

---

## 执行检查清单（按序）
- [ ] 1.1.3 PatchNorm 开关化（dataset.py，加 `--patch_norm`）
- [ ] 1.1.4 `eval_retrieval.py`（轨 B：特征 → cosine → `recallAtK`）
- [ ] 1.1.2 Nordland 500 地 spring,fall→summer 通路跑通 B0 基线（3 seeds）
- [ ] 1.2.1–1.2.5 `ConvSNNLayer`（前向 / E-I 掩码 / WTA×3 / STDP+聚合 / ITP）
- [ ] 1.2.6 接入 `train_new_model` 逐层训练，conv 前端跨模块共享
- [ ] B1（Random Conv 冻结）通路验证
- [ ] 实验 1.2 核可视化脚本（grid + Gabor 拟合 + 方向分布 + 稀疏度）
- [ ] 实验 1.1 主表 B0–B4 × 3 seeds（双轨）
- [ ] 实验 1.3 / 1.4 消融表
- [ ] 实验 1.5 效率表
- [ ] Oxford RobotCar 450 地复跑主表
- [ ] SpikingJelly conv-STDP 参照行
