# feat-BCM 变体网格实验报告（feature_layer BCM 滑动阈值，G1–G4）

> 分支 feat/convstdp-bcm-feature（tip a7d611b）。机制：feature_layer 普通 STDP 分支的
> 固定门控 (0.5 − post) → 每神经元滑动阈值 (θ_M − post)，θ_M = EMA of post²
> （α=0.001，初值 0.25=0.5²，普通属性不进 state_dict；先用旧 θ_M 门控本步再 EMA 更新，
> 机制照抄 conv 版 bcm_gate，conv_learning.py Step 1）。spk_force 分支（output_layer）未动。
> **本报告只做汇总，判定回母会话。**

## 实现清单

| 文件 | 位置 | 内容 |
|---|---|---|
| `vprtempo/src/blitnet.py` | :60-61 | SNNLayer 构造参数 `bcm_gate=False, bcm_alpha=0.001` |
| `vprtempo/src/blitnet.py` | :194-202 | 训练分支创建 `theta_m`（[1, dims[-1]]，初值 0.25，普通属性） |
| `vprtempo/src/blitnet.py` | :567-570 | 普通 STDP 分支：`gate = tile(θ_M)`（与 post 同形广播），先 tile 出旧 θ_M 门控、再 `θ_M ← (1−α)θ_M + α·spikes²`；两条权重更新用 `(gate − post)` |
| `main.py` | :526-530 | `--feat_bcm`（store_true）+ `--feat_bcm_alpha`（默认 0.001） |
| `vprtempo/VPRTempoTrain.py` | :224-225 | 仅 feature_layer 的 add_layer 透传；output_layer 不传 |
| `IDEA1-covstdp/experiments/test_feat_bcm.py` | 新 | 玩具测试 5 项断言 |
| `IDEA1-covstdp/phase3/configs/bcmfeat/` | 新 | 12 配置（4 变体 × 500 单文件 + 3300 train/eval 双文件），与主表对应行逐键一致（脚本校验 ✔） |
| `IDEA1-covstdp/experiments/run_bcmfeat.sh` | 新 | 批处理（照 run_zoo_formal.sh：槽位队列/断点续跑/错秒启动/进度日志） |

## 正确性验证

- **玩具测试 5/5 通过**（`pixi run python IDEA1-covstdp/experiments/test_feat_bcm.py`）：
  ①关闭（默认/显式 False）5 步 calc_stdp 后权重与阈值 torch.equal 逐比特一致；
  ②单步更新手工复现（gate=旧 θ_M + 符号钳制 + Homeostasis）逐元素一致，EMA 时序正确；
  ③冻结权重+恒定输入下 θ_M 2000 步逐神经元单调收敛到 post²（终值误差 4.8e-6），无振荡；
  ④state_dict 键 = [thr, w.weight]，无 theta_m；⑤开启后发放率分布变化
  （300 步随机输入：mean 0.5317→0.5254，逐神经元 |Δfr| 均值 0.0141）。
- **B0 回归（工作站 GPU，seed0，500 地）**：默认关闭时重跑 `table1/b0_500.json`，
  与既有 `t1_b0_500/seed_0` 结果**逐点一致**：recallAtK {1:0.93, 5:0.97, 10:0.98,
  15:0.99, 20:0.99, 25:0.99}、R@100%P=0.1771058315334773、P@100%R 全部 MATCH。
  （本机 CPU 冒烟同配置 R@1=0.92，属 CPU/GPU 浮点差异，非回归失败。）

## 结果

### 500 地迭代档（seed0，单 seed）

| 变体 | 轨A R@1 | 轨B R@1 | 轨B R@100%P | 主表对应行 seed0（轨A / 轨B R@1 / R@100%P） |
|---|---|---|---|---|
| G1: B0 + feat_bcm | 0.540 | 0.794（feature_layer） | 0.265 | B0: 0.930 / 0.950 / 0.613 |
| G2: B2 + feat_bcm | 0.200 | 0.890（conv） | 0.479 | B2: 0.910 / 0.890 / 0.479 |
| G3: B2+BCM + feat_bcm | 0.310 | 0.894（conv） | 0.380 | B2+BCM: 0.900 / 0.894 / 0.380 |
| G4: B5 + feat_bcm | 0.050 ⚠️ | 0.984（conv） | 0.793 | B5: 0.980 / 0.984 / 0.793 |

### 3300 地正式档（3 seeds，mean ± std）

| 变体 | 轨A R@1 | 轨B R@1 | 轨B R@100%P | 轨B P@100%R | 主表对应行（轨A / 轨B R@1） |
|---|---|---|---|---|---|
| G1: B0 + feat_bcm | 0.000 ± 0.000 ⚠️ | 0.325 ± 0.008（feature_layer） | 0.054 ± 0.002 | 0.327 ± 0.008 | B0: 0.597±0.006 / 0.716±0.003 |
| G2: B2 + feat_bcm | 0.007 ± 0.006 ⚠️ | 0.637 ± 0.011（conv） | 0.027 ± 0.001 | 0.638 ± 0.011 | B2: 0.600±0.000 / 0.637±0.011 |
| G3: B2+BCM + feat_bcm | 0.007 ± 0.006 ⚠️ | 0.676 ± 0.009（conv） | 0.094 ± 0.045 | 0.678 ± 0.009 | B2+BCM: 0.620±0.000 / 0.676±0.009 |
| G4: B5 + feat_bcm | 0.133 ± 0.049 ⚠️ | 0.801 ± 0.003（conv） | 0.386 ± 0.009 | 0.802 ± 0.003 | B5: 0.717±0.006 / 0.801±0.003 |

⚠️ = 3300 地轨A R@1 < 0.2，按纪律标注**待复核**（不丢弃）。G1–G4 的 3300 轨A 全部落入该区间。

### 完整性检查

- 16/16 格 DONE，0 FAIL（`bcmfeat_progress.log`）；轨A eval JSON + 轨B JSON + logfile 全部齐全。
- **锚点校验（正面证据）**：G2/G3/G4 的轨B（conv 特征点，不经过 feature_layer）与主表
  B2/B2+BCM/B5 行逐位一致（3300：0.637±0.011 / 0.676±0.009 / 0.801±0.003 完全相同；
  500 seed0：0.890/0.894/0.984 及 R@100%P 均相同）——feat_bcm 只影响 feature_layer 训练，
  conv 前端与轨B-conv 口径未受污染，塌缩定位在 feature_layer 内部及其读出。
- 500 地轨A：G4=0.050 < 0.2 同属可疑区间（500 档纪律未硬性规定，一并标注待复核）。

## 观察（非判定，供母会话参考）

1. **轨A 系统性塌缩、规模越大越重**：500 地 0.05–0.54，3300 地 0.000–0.133；
   对照行 0.597–0.717。四个变体方向完全一致，与前端类型无关。
2. **G1 的轨B（feature_layer 特征点）也大幅下降**（500：0.794 vs 0.950；
   3300：0.325 vs 0.716）——FC 域 BCM 门控不仅破坏 spike-forcing 读出，
   也削弱了 feature_layer 表征本身；而 conv 域 bcm_gate（R1）当年是正面增益。
3. 机制假说（待复核方向，非结论）：FC 层 post ∈ [0,0.9] 且发放稀疏，
   θ_M = E[post²] ≪ 0.5（初值 0.25 已低于 0.5），门控 (θ_M − post) 对绝大多数
   发放神经元长期为负 → 净抑制 → 特征层输出幅度塌缩 → 下游 spike forcing 失锚。
   conv 版 θ_M 是"全图均方响应"且只对 WTA winner 位置更新，口径不同，
   这或是同一机制在两域效果相反的原因之一。θ_M 曲线未在本批留诊断（可在迭代档补）。

## 母会话判定（2026-09-08）

**负面结果成立，机制层面而非实现层面**：①玩具测试 5/5 + B0 回归逐点一致（实现正确）；
②锚点校验——G2/G3/G4 的轨 B（conv 特征点）与主表对应行逐位一致（污染隔离证明）；
③塌缩模式（轨 A 系统性崩塌、轨 B conv 点不变、G1 的 feature 层特征点同步退化）
与"θ_M 长期为负 → 净抑制 → 幅度塌缩"假说自洽。

**论文可用的结论**：BCM 滑动阈值的收益是**位置特异**的——竞争选择层（conv 前端 WTA+winner
更新）适用，广播式更新层（feature 层普通 STDP）有害。这恰好是 Zenke & Gerstner (2017)
"快/慢补偿过程分工"的实证案例：两层各需不同的稳定机制，(0.5−post) 定点门控在 feature
层的设计合理性被反向印证。进消融/分析章节，不进主表。

## 墙钟

- 批处理总墙钟：2026-09-08 00:14:37 → 01:16:40（约 62 min，双 RTX 4090 槽位并行）。
- 500 地每格 45–118s；3300 地每格 189–193s（G1，无 conv）/ 641–659s（G2–G4）。
- 单格 JSON 内 wall_time_s（train+eval 分段）与逐格日志在 `bcmfeat_logs/`。

## 坑（本次踩到）

1. 工作站无全局 pixi（`~/.pixi/bin` 不存在）；项目本地环境直通：
   `PY_CMD="$PWD/.pixi/envs/cuda/bin/python"`（run_bcmfeat.sh 支持 PY_CMD 覆盖）。
2. 工作站 `local_override.json` 的 data_dir（/home/ps/datasets/Nordland）已失效，
   实际数据集在 /mnt/sda2/Li_Ruipeng/datasets/Nordland——已就地修正（gitignore 文件）。
3. bundle 检出被 9 个 untracked 结果文件阻挡（它们在本地已入库）：
   备份至 /tmp/ws_untracked_backup 后 `checkout -f`，JSON 逐一 diff 全部 SAME，无数据丢失。
