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

- [x] ~~补算轨 A P@100%R~~ **已完成（2026-09-07）**：主表 JSON 中 trackA 的 precisionAt100recall 字段已在，离线聚合即可（`experiments/agg_p100r_trackA.py` → `results/table1_p100r_trackA.md`）。
- [ ] ORC 侧：LoCS-Net ORC 协议为 grid 离散化 2,500 标签（与 VPRTempo/本文的 450 地 sun,rain→dusk 完全不同），**ORC 数字与 LoCS-Net 不可比，只能与 VPRTempo 系比**；
- [ ] VPRSNN 只作历史性引用（其主实验是 100 地分段 + R@100%P，与本表不同口径，不进对比表）。

## 5. 补算结果：轨 A P@100%R（2026-09-07，离线聚合，无重跑）

**3300 地轨 A P@100%R（3 seeds）**：B5+BCM 0.727 > B5 0.719 > DCT 0.712 > Log-Gabor 0.685 > B6a 0.639 > B2+BCM 0.619 > B2 0.600 = B0 0.600 > freesign 0.573 > B1 0.531 > B1+ITP 0.437。
（注意：轨 A 是"强制每查询必匹配"的分类式系统，P@100%R ≈ R@1，阈值化无收益——这本身说明 spike-forcing 输出的分数分布没有可利用的余量。）

**关键判定——对外声明的最终口径**：

| 声明 | 数字 | 成立性 |
|---|---|---|
| 描述子级（轨 B）：B5 > LoCS-Net | 0.802 vs 0.786 | ✅ 成立（且我方协议更严：3300 参考地、查询省略前 600 地） |
| 完整系统级（轨 A）：B5 > LoCS-Net | 0.719 vs 0.786 | ❌ **不成立**——差 6.7 点 |
| 完整系统级：B5 vs VPRTempo（LoCS-Net 表内 73.0%） | 0.719 vs 0.730 | ❌ 差 1.1 点（但注意该 73.0% 是 3072 地无 skip 协议，与本文 3300 地协议不同） |
| 完整系统级：B5 vs 本文 B0 复现 | 0.719 vs 0.600 | ✅ +11.9 点（同协议同指标，最干净的对比） |
| 轨 A R@1：B5 vs VPRTempo 原论文报告值 | 0.717 vs 0.56 | ✅ +15.7 点（协议差异见 §1，作定性参照） |

**结论**：摘要级 hook 只能用**描述子级**口径；系统级我们与 LoCS-Net 的差距（−6.7 点）恰好就是读出瓶颈的量化证据（轨 B 0.802 vs 轨 A 0.719 = 8.3 点被读出层丢掉）——这不是弱点，是创新点 2（读出适配）的直接动机。

## 6. VPRTempo 复现核对（本文 B0 vs 原论文）

| 口径 | 原论文/引用值 | 本文 B0 复现 | 判定 |
|---|---|---|---|
| Nordland 3300 地轨 A R@1（VPRTempo 论文 Table II） | **56%** | **59.7% ± 0.6**（3 seeds） | ✅ 同量级且略高（协议逐参数一致：3300 地/2700 查询/skip 前 20%/filter 8/零容差/28×28） |
| Nordland P@100%R（LoCS-Net 表内的 VPRTempo） | 73.0% | 轨 A P@100%R 60.0% | ⚠️ 不可直接比：LoCS-Net 协议为 3072 地且无 skip；且其 PR 计算的分数分布细节未知。论文中引用此值时须注明协议差异 |
| ORC 450 地 R@1（VPRTempo 论文） | 37% | 待 S3.5 复跑 | 待办 |

**写作纪律**：对外对比表引用 LoCS-Net Table 2 原值并注明协议；对内结论一律用同协议数字（B0 复现列）。
