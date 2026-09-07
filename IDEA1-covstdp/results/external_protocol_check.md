# 外部对比口径核查（LoCS-Net / VPRTempo / VPRSNN vs 本文）

> 日期：2026-09-07 ｜ 目的：核实"B5 轨B P@100%R@3300 = 0.802 超 LoCS-Net 78.6%"这一对外声明的合法性
> 来源：LoCS-Net 全文（Frontiers Neurorobotics 2025, Table 1/2 + §4.1/4.2）、VPRTempo 论文 PDF（papers/VPRTempo.pdf §IV）、VPRSNN（arXiv:2109.06452v2 §V）

## 1. 四方 Nordland 协议对照

| 项 | LoCS-Net (2025) | VPRTempo (ICRA 2024) | VPRSNN (RA-L 2022) | 本文（3300 地正式档） |
|---|---|---|---|---|
| 参考 traverse | spring + fall | spring + fall | spring + fall | spring + fall ✓ |
| 查询 traverse | summer | summer | summer | summer ✓ |
| 采样 | 每 8 帧取 1（filter=8） | 每 8 秒（≈100m，filter=8） | 每 8 秒，**取前 1/4 段** | 每 8 秒 ✓ |
| 参考地点数 | **3,072** | 3,300 | 100 地分段 ×15（可扩展性实验最大 400） | 3,300（3 模块 × 1100） |
| 查询数 | 3,072（**不含 skip**） | 2,700（**省略前 20% 参考图像**，因 VPRSNN 用其做校准） | 每段 100 | 2,700（skip=4800 ≈ 前 600 地）✓ 与 VPRTempo 同 |
| GT 容差 | 精确地点标签（分类式） | 精确匹配，零容差 | 精确匹配，零容差 | 零容差 ✓ |
| 主指标 | **P@100%R** + AUC + R@N | R@N + PR | **R@100%P**（注意方向相反）+ AUC | Recall@K + R@100%P + **P@100%R** ✓ |
| 输入 | 56×56 灰度 | 28×28 | 28×28（7×7 patch norm） | 28×28 ✓ |

## 2. LoCS-Net Table 2 的 Nordland P@100%R 格局（照录）

| 方法 | 类型 | P@100%R |
|---|---|---|
| MixVPR | ANN | 94.6% |
| Conv-AP | ANN | 91.3% |
| EigenPlaces | ANN | 80.2% |
| **LoCS-Net（GPU，off-chip）** | SNN（BP 训练） | **78.6%** |
| LoCS-Net（Loihi on-chip） | SNN | 71.1% |
| **VPRTempo** | SNN（无 BP） | **73.0%**（LoCS-Net 论文实测/引用值） |
| Ensemble SNNs | SNN | 66.9% |
| VPRSNN (WNA) | SNN | 0.3%（在 3072 地规模崩塌——VPRSNN 不扛规模的实证） |

## 3. 本文数字的可比性判定

**可比的部分**：
- 指标方向一致（P@100%R）✓；季节划分一致（spring+fall→summer）✓；采样率一致（filter=8）✓；GT 零容差 ✓。

**必须声明的差异**：
1. **参考集规模**：本文 3,300（更难）vs LoCS-Net 3,072；**查询集**：本文省略前 600 地（2,700 查询，VPRTempo 口径），LoCS-Net 用全部 3,072。
2. **管线性质**：本文 0.802 是**轨 B（conv 特征 cosine 检索）**，LoCS-Net 的 78.6% 是其完整系统输出。严格同口径对比应用本文**轨 A P@100%R**（主表只报了轨 A R@1——轨 A 的 P@100%R 可从已保存的相似度矩阵离线补算，成本极低，进正文前必须补）。
3. LoCS-Net 的 78.6% 未注明多种子平均（其敏感性分析提到 5 seeds，但 Table 2 主数字口径未明）；本文数字为 3 seeds mean±std。

**措辞建议**：声明写成"在 Nordland 会议规模协议（spring,fall→summer，filter=8，零容差）下，本文冻结 Gabor 前端的描述子级检索 P@100%R 达 80.2%，超过 BP 训练的 LoCS-Net 报告的 78.6%（其参考集 3,072 地，本文 3,300 地且查询不含前 600 地，协议更严）"——**不要**简化为"全面超越 LoCS-Net"（其完整系统指标是分类器输出，且 SpikeVPR 2026 事件系工作 R@1 88.2% 存在，需划界）。

## 4. 待办

- [ ] 补算主表各变体**轨 A 的 P@100%R**（从已存 S 矩阵离线算，metrics.py createPR 现成）——轨 A 才是与 LoCS-Net 完整系统对应的行；
- [ ] ORC 侧：LoCS-Net ORC 协议为 grid 离散化 2,500 标签（与 VPRTempo/本文的 450 地 sun,rain→dusk 完全不同），**ORC 数字与 LoCS-Net 不可比，只能与 VPRTempo 系比**；
- [ ] VPRSNN 只作历史性引用（其主实验是 100 地分段 + R@100%P，与本表不同口径，不进对比表）。
