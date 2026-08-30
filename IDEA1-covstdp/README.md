# IDEA1: Conv-STDP 卷积前端 for VPRTempo

为 VPRTempo 增加无监督 Conv-STDP 卷积前端的论文实验工作区。
**完整方案与实验设计见 [PLAN.md](PLAN.md)** —— 主控文档，每个步骤卡片自包含
（背景动机 / 前置依赖 / 详细操作 / 验收标准），供 fork 出的实现会话直接使用。

## 核心设计（速览）

- **创新点**：spike 编码后、`feature_layer` 前插入单步幅度域 Conv-STDP 前端（无 BP、无多步仿真）。
- **双轨评测**：
  - 轨 A：完整训练 → 现有 `run_inference` → Recall@K；
  - 轨 B：`eval_retrieval.py` 提特征 → cosine → `recallAtK`（`metrics.py:134`），隔离 encoder 质量。
- **对照组**：B0 原模型 / B1 Random Conv 冻结 / B2 Conv-STDP 1层 / B3 Conv-STDP 2层 / B4 CNN+BP 上界。
- **决策门**：Gate 0 基线复现；Gate 1 = 轨B B2>B1（超 3-seed std）；Gate 2 = 轨A B2/B3>B0。
- **主数据**：Nordland 500 地 spring,fall→summer，每格 3 seeds × mean±std；补 Oxford RobotCar 450 地。

## 目录结构

```
IDEA1-covstdp/
├── PLAN.md         # 主控设计文档（阶段 1–3、步骤卡片 S1.1–S3.5、决策门、时间线、fork 指引）
├── src/            # ConvSNNLayer 等新模块代码（成熟后考虑并入 vprtempo/src/）
├── experiments/    # 实验配置、run_exp.py、eval_retrieval.py、分析与可视化脚本
└── results/        # 结果表格（JSON/MD）与图（Figure 2 等）
```

## 关键代码锚点（main 分支，已核实）

| 用途 | 位置 |
|---|---|
| STDP 更新（全连接版，公式 2/6） | `vprtempo/src/blitnet.py:418-557`（`calc_stdp`） |
| 符号钳制 [1e-6,10] / [-10,-1e-6] | `vprtempo/src/blitnet.py:581-584` |
| ITP 阈值可塑性 | `vprtempo/src/blitnet.py:597-606` |
| 目标发放率线性分配 [0.2,0.9] | `vprtempo/src/blitnet.py:164-167` |
| 权重初始化（3σ 正态 + 符号裁剪 + L1 归一化） | `vprtempo/src/blitnet.py:250-327`（`addWeights`） |
| clamp_spikes 到 [0, 0.9] | `vprtempo/src/blitnet.py:377-387` |
| 单层训练循环（prev_layers 冻结前向） | `vprtempo/VPRTempoTrain.py:327-485`（`train_model`） |
| 学习率退火 (1−t/T)² | `vprtempo/VPRTempoTrain.py:275-316` |
| 逐层训练总控（layer_dict 排序） | `vprtempo/VPRTempoTrain.py:591-764`（`train_new_model`） |
| 图像预处理流水线 | `vprtempo/src/dataset.py:345-441`（`ProcessImage`） |
| PatchNorm 开关点 | `vprtempo/src/dataset.py:422-424` |
| 推理总控 | `vprtempo/VPRTempo.py:669`（`run_inference`） |
| Recall@K | `vprtempo/src/metrics.py:134`（`recallAtK`） |
