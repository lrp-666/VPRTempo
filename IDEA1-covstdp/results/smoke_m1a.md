# 第 -1a 步冒烟跑结果（main 分支，未改动代码）

日期：2026-09-02｜机器：本机 CPU-only（无 CUDA/MPS）｜分支：main @ 2e4adfc

## 配置

```
--dataset nordland --database_places 100 --query_places 100 --max_module 100
--database_dirs spring,fall --query_dir summer --dims 28,28 --patches 7
--skip 0 --filter 8 --epoch 4 --data_dir /mnt/d/Data/datasets/vpr/Nordland
```

## 结果

| 项 | 值 |
|---|---|
| 训练墙钟 | **29 s**（800 步 = 100 地 × 2 季 × 4 epoch；feature 层 ~21s + output 层 ~5s） |
| 推理墙钟 | 数秒级（100 查询） |
| Recall@1/5/10/15/20/25 | 1.0 / 1.0 / 1.0 / 1.0 / 1.0 / 1.0 |
| R@100%precision | 1.0（PR_curve_data.json 计算） |
| 模型文件 | `vprtempo/models/springfall_VPRTempo_IN784_FN1568_DB100.pth` |

## 结论

1. **环境与数据通路验证通过**：pixi 环境、/mnt/d 数据集路径、CSV 标注全部正常。
2. **CPU 可行性确认**：100 地训练 29s → 500 地（4000 步）估计 ~2.5 min/格。阶段 1 的 6 格 B0 基线（≈20 min）**全部本地完成，无需工作站**。
3. **发现并记录一个坑**：`database_places=100` 必须显式 `--max_module 100`——默认 500 时 `remainder=100≠0 且 num_modules=1` 触发 `ZeroDivisionError`（main.py:245）。后续 run_exp.py 需自动处理 max_module。
4. 100 地 Recall=1.0 说明小规模下任务过易——500 地基线才是有意义的参照；100 地仅用于通路验证，不进任何表格。
