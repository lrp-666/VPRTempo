# Kimi Work 调研任务提示词（S2.11 新学习规则探索）

> 2026-09-04 定稿，供 Kimi Work 新任务直接粘贴。调研报告回来后交母会话评审，
> 报告入库为 `IDEA1-covstdp/docs/lit_review_single_step_rules.md`。

```
你是一名神经形态计算与脉冲神经网络方向的学术调研助手。请做一次系统的文献调研，产出一份中文调研报告。

【背景】
我在做一个视觉场景识别（VPR）的脉冲神经网络工作，基座是 VPRTempo（ICRA 2024，基于 BLiTNet）。
网络是单步幅度域的：所谓"spike"是 [0, 0.9] 的幅度值（clamp 后），没有膜电位时间演化、没有多步仿真。
当前使用的无监督学习规则继承自 BLiTNet，是全连接版的逐元素规则：
    ΔW = η · (0.5 − post) · pre_term · Θ(post)
其中 post 是后层神经元幅度，pre_term 是前层输入（我们用中心化版本 pre − 0.5），
配合 WTA 竞争（只有 winner 位置更新）和每通道 ITP 阈值可塑性（Δθ = η_ITP·(发放率 − f)）。
我已把它卷积化（共享核 + winner patch 更新 + 保范数 L1 归一化）。
注意：严格说这不是 STDP（没有时间维度），而是单步幅度域的竞争 Hebbian 规则。

【调研问题】
找出文献中所有可能适用于此设置的**无监督局部学习规则**，重点是：
1. 经典规则族及其性质：Hebb/Oja（归一化防发散）、BCM（滑动静息点）、
   Krotov & Hopfield 2019（竞争 + ReLU 高次项）、SoftHebb（softmax 竞争 + 贝叶斯解释）、
   triplet STDP、homeostatic Hebbian、anti-Hebbian 组合等；
2. SNN 社区的 conv-STDP 规则：Diehl & Cook 2015、Kheradpisheh 2018、Mozafari 2018/2019、
   Tavanaei、以及 SpikingJelly/BindsNET 里的实现——它们的规则形式和竞争机制；
3. 每个候选规则给出：数学公式、稳定性机制（怎么防权重发散/塌缩）、
   与我们 WTA + ITP 框架的兼容性、改造成"单步幅度域"需要做什么；
4. 特别回答：BLiTNet 的 (0.5 − post) 自稳定项相比 Oja 式归一化/BCM 滑动阈值的优劣；
   有没有规则天然更擅长学出 Gabor 样方向选择性结构（我们现在学出的是中心-外周团块，
   想要 oriented edge）；
5. 若能找到"在好初始化（如 Gabor）上用局部规则微调"的相关先例，也一并收集。

【产出要求】
- 一份结构化报告：每个候选规则一节（公式 + 出处 + 优缺点 + 与我们的适配难度）；
- 末尾给出 2–3 个"最值得尝试"的推荐，附改造伪代码骨架；
- 所有文献给出可核查的引用（标题、作者、年份、venue/arXiv 号），不要编造；
- 不确定的文献标注"未核实"。
```
