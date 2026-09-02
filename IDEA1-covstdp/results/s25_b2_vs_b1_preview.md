# S2.5 出口：B2 vs B1 轨 B 单 seed 粗比（Gate 1 预演）

日期：2026-09-03｜分支：feat/convstdp-s25-integration｜配置：Nordland 500 地 spring,fall→summer、28×28/patches=7、seed=0、单 seed（无误差棒，方向性参考）

## 结果（轨 B：conv 池化后 1152 维特征 → cosine → recallAtK）

| 变体 | R@1 | R@5 | R@10 | R@15 | R@20 | R@25 | R@100%P | P@100%R |
|---|---|---|---|---|---|---|---|---|
| B1（random_conv 冻结，同 seed 同初始化） | 0.736 | 0.868 | 0.900 | 0.922 | 0.928 | 0.938 | 0.1277 | 0.7390 |
| B2（conv_stdp 训练，conv_epoch=2） | **0.772** | 0.866 | **0.908** | **0.932** | **0.942** | **0.954** | **0.1684** | **0.7751** |
| 差值（B2−B1） | **+0.036** | −0.002 | +0.008 | +0.010 | +0.014 | +0.016 | **+0.041** | **+0.036** |

参考：B0 轨 B（feature_layer 输出，同 500 地 seed0）R@1 = 0.948 / R@100%P = 0.587（见 table_baseline_b0.md）。

## 观察

1. **Gate 1 预演信号为正**：B2 在 R@1 / R@100%P / P@100%R 三个主指标上全部超过 B1（R@100%P 相对提升 +32%）。STDP 训练相对同初始化的随机核有真实增益——但单 seed，正式判定待 3 seed 主表。
2. **诚实记录**：B1/B2 的轨 B 均明显低于 B0 的轨 B（0.77 vs 0.95）。注意口径差异——B0 轨 B 用的是**训练过的 feature_layer 输出**，B1/B2 用的是 conv 池化后直接输出（未经 feature_layer）。conv 前端是否真正改善系统，需看 B2 的 feature_layer 输出 vs B0（第二特征点，S3.2 补）以及 3300 地规模（B0 encoder 天花板更低处，见 scale_check.md）。
3. 冒烟（100 地）数字：B2 轨 A R@1=0.47 / 轨 B R@1=0.93——轨 B ≫ 轨 A 的"读出瓶颈"模式在小规模同样出现，与 3300 地规模发现一致。

## 工程验证（S2.5 验收项）

- 端到端训练管线：conv_layer →（冻结）→ feature_layer → output_layer 逐层训练跑通（100 地 56s、500 地 ~300s）；
- 保存/加载往返：训练保存 → run_inference 加载 → 推理正常；
- B0 回归：frontend='none' 路径 smoke100 训练+评估正常（R@1=0.99，与阶段 1 一致）；
- B1 frozen 机制：random_conv 跳过 conv 训练直接进下游，同 seed 下与 B2 共享初始化；
- 坑已修：VPRTempo.py 评估阶段 plt.show() 在交互式后端挂死 → run_exp.py 强制 MPLBACKEND=Agg。
