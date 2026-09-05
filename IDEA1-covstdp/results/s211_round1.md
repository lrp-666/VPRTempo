# S2.11 规则锦标赛 Round 1 结果（迭代档）

- 日期：2026-09-05；分支 feat/convstdp-s210-b6 @ 8476368（代码）→ 结果提交见本文件 commit
- 协议：PLAN.md S2.11 + §8 第 4 条迭代档（500 地 × seed 0）；工作站容器（RTX 4090 D，`CUDA_VISIBLE_DEVICES=1`）
- 基座：B2 主组合新工作点 **conv_epoch=4 + conv_stdp_rate=0.01**（§8 锁定清单第 1 条，S3.2a 调参窗锁定）
- 执行：每格 `run_exp.py --train --eval`（轨 A）+ `eval_retrieval.py`（轨 B），串行 + 3s 错秒（logger 撞目录防护）
- 诊断：`results/s211_round1_diag.json`（Part A 训后核形态 + Part B 训练曲线重放，含 R1 θ_M 曲线）+ `results/s211_round1_kernels.png`（五格核网格图）
- 玩具门槛：`results/s211_toy_rules.json`（对拍 66/66 全过；动力学断言 R1/R2/R3 全过，**R4 未过**，见下）

## 玩具门槛（进真数据前的预注册判据）

| 变体 | 范数稳定 | DC/AC 下降 | 余弦不升 | 门槛 |
|---|---|---|---|---|
| R0（全关对照） | ✅ 1.000 | ✅ 3.10→0.85 | ✅ 0.901→0.708 | 过 |
| R1 bcm_gate | ✅ 1.000 | ✅ 3.10→1.09 | ✅ 0.901→0.573 | 过 |
| R2 rank_push | ✅ 1.000 | ✅ 3.10→0.89 | ✅ 0.901→0.780 | 过 |
| R3 oja_decay | ✅ 1.410 | ✅ 3.10→0.93 | ✅ 0.901→0.657 | 过 |
| R4 attractor | ✅ 1.000 | ❌ 3.13→**22.33** | ❌ 0.903→**1.000** | **未过** |

R4 失败机制明确：pre_term 从 (pre−0.5) 换成 (patch−K) 后，更新含未中心化 patch 的直流分量（正是 S2.3 centered 防线针对的直流塌缩），所有核收敛到同一个近常数原型（余弦→1.0）。**按预注册 R4 无 Round 2 资格；下表真数据数字仅为诊断留档**（实现正确性已由 Part A 对拍 66/66 保证，含 R4 向量化 vs 朴素循环版逐元素一致）。

## 主表（500 地 × seed 0）

| 格 | 轨B R@1 | 轨B R@100%P | 轨B P@100%R | 轨A R@1 | ΔR@1 vs R0（轨B） |
|---|---|---|---|---|---|
| **R0 锚**（新工作点首测） | 0.890 | 0.4787 | 0.8918 | 0.92 | — |
| R1 BCM 滑动阈值 | **0.894** | 0.3803 | **0.903** | 0.90 | +0.4 点 |
| R2 名次反推（δ=0.4, k=2） | 0.868 | 0.4217 | 0.8697 | 0.88 | −2.2 点 |
| R3 Oja 衰减 | 0.882 | 0.4694 | 0.8838 | 0.91 | −0.8 点 |
| R4 弹性项（玩具门槛未过，诊断留档） | 0.836 | 0.4737 | 0.8479 | 0.71 | −5.4 点 |
| 对照：B5 手工 Gabor | 0.984 | 0.7927 | 0.9899 | 0.98 | — |
| 对照：freesign | 0.978 | 0.7178 | 0.978 | 0.89 | — |
| 对照：B2 旧工作点（epoch2,stdp0.005） | 0.878 | 0.4601 | 0.8851 | 0.84 | — |
| 参考：tune 单维 epoch4@0.005 | 0.884 | 0.4819 | 0.8858 | 0.88 | — |
| 参考：tune 单维 stdp0.01@epoch2 | 0.884 | 0.5000 | 0.8858 | 0.88 | — |

注：0.884 两个 tune 格是逐维扫描的单维格（见 s32a_tuning.md），本表 R0 是 **epoch4+stdp0.01 组合工作点的首测**（0.890，比旧工作点 +1.2 点，组合有微弱协同）。

**预注册胜出判据核对**：轨 B R@1 > R0 + 2 点 = 0.910。**四格无一达到**；R1 +0.4 点在迭代档噪声内，R2/R3 负向，R4 门槛未过且真数据同样显著负向。

## 每格诊断要点

- **R1（BCM）**：θ_M 均值从 0.25 单调降至 0.0187（逐通道末态 0.006–0.048，无振荡——α=0.001 的慢 EMA 与 ITP 无共振迹象）。动力学含义：post² 的全图均值被大量未发放位置压低，θ_M 迅速远小于 0.25，门控 (θ_M−post) 整体变负、**更新的自稳定项减弱**（等效于降低了"去选择性"回拉）。训练曲线 DC/AC 终值 0.69 高于 R0 的 0.52，核更稠（稀疏度 0.56 vs 0.72）。轨 B +0.4 点 / 轨 A −2 点，信号弱且方向不一致。
- **R2（名次反推）**：终态核间余弦 0.566，**高于** R0 的 0.492——廉价反推没有去相关，反而让第 2 名通道向其 winner patch 反向靠拢，核群更聚拢；轨 B −2.2 点，四格中（除 R4 外）最差。形态与 R0 几乎无差（R² 0.465 vs 0.456）。
- **R3（Oja）**：核 L1 范数 1.00→1.071 后自动饱和（+7%，不衰减不发散，软归一化不动点存在）；thr 终值略高（0.509 vs 0.468）。轨 B −0.8 点（噪声边缘）。形态略优：Gabor R² 中位 0.480、16/32 拟合成功（五格最高），但仍为团块/脊状而非条纹。
- **R4（弹性项，门槛未过）**：真数据确认玩具结论——训后 DC/AC 中位 31.4、核间余弦 1.000、Gabor R² 0.022、0/32 拟合成功、稀疏度 0（所有核塌缩为同一近常数模板）。即使如此轨 B 仍有 0.836（块池化后的 DC 地形仍带位置信息），但轨 A 崩到 0.71（spike-forcing 读出对塌缩特征更敏感）。
- **R0 对照**：新工作点训练曲线健康（发放率 0.877→0.442，DC/AC 3.12→0.52，余弦 0.907→0.492）。

## 形态注记（哪个变体最接近条状）

**没有任何变体学出条纹/棒状形态**。Gabor 拟合 R² 中位数全部在 0.44–0.48（R4 除外，0.02），拟合成功核 11–16/32，方向合成长度 ≤0.26（无方向偏好）；核网格图（s211_round1_kernels.png）上 R0/R1/R2/R3 均为团块/拉长脊，R1 更稠密、R4 全平。相对最好的是 R3（R² 0.480、16/32），但与 R0 的差异在 IQR 内，不构成形态改进。与"形态/性能解耦"（freesign 证据）一致：本轮性能微扰不伴随形态改变。

## 结论（数据层面，不越权判 Round 2）

- 按预注册判据（轨 B R@1 > R0+2 点），**Round 1 无胜出格**；四个单机制手术在迭代档都不带来超噪声的收益。
- R1 是唯一正向微扰（+0.4 点）且机制曲线干净（θ_M 单调、无振荡），若母会话考虑 Round 2，R1 是唯一没有负面证据的候选；R2/R3 为负/平，R4 门槛出局。
- 另一读数：R0 组合工作点 0.890 本身比 tune 单维格（0.884）略高，新锚点可信。

## 坑（本轮遇到）

1. **容器 NVML 失效**（宿主机 nvidia-smi 正常、容器内 `Failed to initialize NVML`）——本机有 `vpr-host` ssh 别名可达宿主机且 ps 在 docker 组，`docker restart vpr-tempo-Li_Ruipeng` + 重启容器内 sshd 即恢复，无需动 docker  daemon（避免影响他人容器 coxgraph）。
2. 宿主机 GPU0 被他人容器占 583 MiB——本轮全程 `CUDA_VISIBLE_DEVICES=1` 锁 4090 D。
3. GitHub 推送 HTTP2 framing/超时抖动——`git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=500 -c http.lowSpeedTime=60 push` 重试一次即成功。

## 复现

```bash
# 工作站容器（需 local_override.json: {"data_dir": "/home/ps/datasets/Nordland"}）
for c in r0_wp r1_bcm r1_rank r1_oja r1_attractor; do
  pixi run --environment cuda python IDEA1-covstdp/experiments/run_exp.py IDEA1-covstdp/phase2/configs/$c.json --train --eval --seed 0
  sleep 3
  pixi run --environment cuda python IDEA1-covstdp/experiments/eval_retrieval.py IDEA1-covstdp/phase2/configs/$c.json --seed 0
  sleep 3
done
pixi run --environment cuda python IDEA1-covstdp/experiments/diag_s211_round1.py
```
