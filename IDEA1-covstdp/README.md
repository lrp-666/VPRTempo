# IDEA1: Conv-STDP 卷积前端 for VPRTempo

为 VPRTempo 增加无监督 Conv-STDP 卷积前端的论文实验工作区。
完整实施步骤与实验规划见 [PLAN.md](PLAN.md)。

## 核心设计（速览）

- **创新点**：spike 编码后、`feature_layer` 前插入单步幅度域 Conv-STDP 前端（无 BP、无多步仿真）。
- **双轨评测**：
  - 轨 A：完整训练 → 现有 `run_inference` → Recall@K；
  - 轨 B：`eval_retrieval.py` 提特征 → cosine → `recallAtK`（`metrics.py:134`），隔离 encoder 质量。
- **决策门**：Gate 1 = 轨B 上 Conv-STDP > Random Conv；Gate 2 = 轨A 上 B2/B3 > B0。
- **主数据**：Nordland 500 地 spring,fall→summer，每格 3 seeds × mean±std；补 Oxford RobotCar 450 地。

## 目录结构

```
IDEA1-covstdp/
├── PLAN.md         # 详细实施规划（阶段 1.1–1.3，实验 1.1–1.5，审稿预案，时间线）
├── src/            # ConvSNNLayer 等新模块代码（成熟后考虑并入 vprtempo/src/）
├── experiments/    # 实验配置与运行脚本（含 eval_retrieval.py 规划）
└── results/        # 结果表格、核可视化图
```

## 关键代码锚点（main 分支）

| 用途 | 位置 |
|---|---|
| STDP 更新 / 符号钳制 | `vprtempo/src/blitnet.py:581-584` |
| ITP 阈值可塑性 | `vprtempo/src/blitnet.py:597-606` |
| 目标发放率线性分配 | `vprtempo/src/blitnet.py:164-167` |
| 逐层训练框架 | `vprtempo/VPRTempoTrain.py:591`（`train_new_model`） |
| 学习率退火 | `vprtempo/VPRTempoTrain.py:275` |
| Recall@K | `vprtempo/src/metrics.py:134` |
| PatchNorm 开关点 | `vprtempo/src/dataset.py:422` |
