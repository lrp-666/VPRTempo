# VPRTempo 代码与论文对照学习指南（3天 × 6小时）

> **目标**：三天后达到"代码↔理论"完全对应，能独立借助AI工具修改代码、设计消融实验，为发表改进型小论文奠定基础。

---

## 零、工程与论文关系总览（阅读此节 30分钟）

### 0.1 两层关系

| 论文 | 角色 | 与代码的关系 |
|------|------|-------------|
| **BLiTNet** (Stratton et al.) | **理论基础** | 提出了 STUNN（Spike-Timing Unsupervised Neural Network）的核心机制：STDP + ITP + 可塑性抑制平衡 + Spike Forcing + 时序编码。VPRTempo 的 `blitnet.py` 几乎是这篇论文公式的直接翻译。 |
| **VPRTempo** (Hines et al., ICRA 2024) | **应用与工程化** | 将 BLiTNet 从 MNIST 分类迁移到视觉场景识别（VPR），并做了三处关键工程创新：(1) 时序编码替代速率编码；(2) 模块化专家网络（modular engrams）实现规模化；(3) 完全用 PyTorch 张量重写，支持 GPU 并行。代码仓库即该论文的官方实现。 |

### 0.2 核心创新链

```
BLiTNet (理论基础)
    ├── 解决"脉冲消失问题" (vanishing spike)
    ├── STDP + ITP + Homeostasis
    └── Spike Forcing 监督读出
            ↓
VPRTempo (工程迁移)
    ├── 时序编码：像素强度 → 单脉冲幅度 → 抽象 theta 振荡相位
    ├── 模块化：每模块 ≤1000 地点，训练/推理可并行
    └── PyTorch 张量化：训练 O(n)、查询 O(log n)
            ↓
本代码仓库 (官方实现)
    ├── blitnet.py      ←→ BLiTNet 公式 (2)-(6)
    ├── dataset.py      ←→ VPRTempo 公式 (8) + Patch Normalization
    ├── VPRTempo*.py   ←→ VPRTempo 章节 III-A~C
    └── metrics.py      ←→ VPRTempo 章节 IV-C
```

---

## 第一天：理论骨架 ↔ 代码骨架（6小时）

### 上午（3h）：精读 BLiTNet 核心理论

**材料**：`papers/BLiTNet.pdf` 的 Introduction + Methods（Signal Propagation + MNIST Networks）

**必须理解的概念清单**：
1. **Vanishing Spike Problem**：为什么标准前馈 SNN 的信号会消失或雪崩？（Fig. 1d-e）
2. **E/I Balance**：可塑性抑制连接如何近似抵消兴奋连接？（公式4、6中的负权重规则）
3. **STDP 规则**：pre-post 时序如何决定突触增强/抑制？（公式2）
4. **ITP（Intrinsic Threshold Plasticity）**：阈值如何根据实际发放率和目标发放率自适应调整？（公式5）
5. **Spike Forcing**：在输出层如何人为强制发放并计算 delta 误差来引导学习？（BLiTNet 中类似 delta rule）
6. **不同目标发放率的作用**：为什么特征层神经元要有 `f ∈ [0.2, 0.9]` 的分布？（Fig. 5）

**笔记任务**：
- 手写推导：从公式 (2) 到代码 `calc_stdp()` 中 `(0.5 - post) * (pre > 0) * (post > 0)` 的对应关系。
- 思考：为什么 `0.5` 这个 magic number 出现？（提示：spike forcing 中 `x_force = 0.5`）

---

### 下午（2.5h）：精读 VPRTempo 论文 + 建立文件映射

**材料**：`papers/VPRTempo.pdf` 的 III-A ~ III-C + IV-A

**关键映射表（必须手抄或打字整理）**：

| 论文公式/章节 | 代码位置 | 对应代码行/函数 |
|--------------|---------|----------------|
| 公式 (1) 神经元状态 | `blitnet.py` | `layer.w(spikes)` → `add_input` → `clamp_spikes` (即 `clamp(input - thr, 0, 0.9)`) |
| 公式 (2) STDP | `blitnet.py` | `calc_stdp()` 第 208-221 行（`else` 分支，非 spk_force） |
| 公式 (3) 学习率退火 | `VPRTempoTrain.py` | `_anneal_learning_rate()` 第 138-147 行 |
| 公式 (4) Homeostasis | `blitnet.py` | `calc_stdp()` 第 235-240 行（`inhW` 归一化） |
| 公式 (5) ITP | `blitnet.py` | `calc_stdp()` 第 228-232 行（`layer.thr.data += ...`） |
| 公式 (6) Spike Forcing STDP | `blitnet.py` | `calc_stdp()` 第 175-205 行（`if layer.spk_force`） |
| 公式 (8) Gamma 校正 | `dataset.py` | `ProcessImage.__call__()` 第 138-142 行 |
| 公式 (9) 匹配规则 | `VPRTempo.py` | `evaluate()` 中通过 `argmax` 思想取最大输出 spike |
| 表 I 超参数 | `VPRTempoTrain.py` `__init__` | `thr_range`, `fire_rate`, `ip_rate`, `stdp_rate`, `p` |
| III-B 模块化 | `main.py` | `initialize_and_run_model()` 第 87-102 行（模块数计算） |
| III-C 并行张量 | `VPRTempo.py` | `forward()` 中 `torch.cat(outputs, dim=1)` |

**笔记任务**：
- 画出一张 "数据流图"：从原始图像 `read_image` → `ProcessImage` → `SNNLayer.forward` → `calc_stdp` → 输出 spikes → `evaluate` 中的相似度矩阵。

---

### 晚上（0.5h）：环境检查与代码初览

**执行命令**：
```bash
pixi install          # 安装依赖
pixi run demo         # 若未下载数据，会触发 download.py 自动下载 (~600MB)
```

**观察目标**：
- 看控制台输出的 ASCII banner 和 Logger 信息（来自 `loggers.py`）。
- 运行成功后，看 `vprtempo/output/` 下生成的日志文件结构。
- 用 `ls -lh vprtempo/models/` 确认预训练模型权重文件。

---

## 第二天：逐行精读核心模块（6小时）

### 上午（3h）：`blitnet.py` —— 心脏模块逐行解剖

**阅读策略**：打开论文公式 (1)-(6) 与 `blitnet.py` 左右分屏，逐函数对照。

#### 2.1 `SNNLayer.__init__`（训练模式）
- **权重初始化**（`addWeights`）：
  - 正态分布 `normal_(mean=Wmn, std=Wsd)` → 对应论文"normally distribute random weights"。
  - 根据 `W_range` 裁剪符号（兴奋 >0，抑制 <0）。
  - `np.random.rand(nrow,ncol) > p` → 实现稀疏连接概率 `P_exc=0.1, P_inh=0.5`。
  - `torch.linalg.norm(..., ord=1, axis=0)` → **L1 归一化到常数 k**，对应公式 (4) 上方的归一化规则。
- **发放率序列化**：`fstep = (fire_rate[1]-fire_rate[0])/dims[-1]` → 实现论文 Fig. 5 中"分布的目标发放率"。
- **Constant Input**：`self.const_inp` → 对应公式 (1) 中的 `C=0.1`。

#### 2.2 `calc_stdp` —— 全代码最关键函数

**分支一：`layer.spk_force == True`（输出层）**
```python
# 对应公式 (6)
xdiff = layer.x.index_fill_(-1, idx_sel, 0.5) - spikes  # 0.5 即 x_force
mpre = prespike / prev_layer.fire_rate                   # 发放率调制学习率
layer.w.weight.data += ((pre * post * layer.havconnCombinedExc.T) * layer.eta_stdp).T
```
- 思考：`pre * post` 就是 `x_pre * (x_force - x_post)`，与公式 (6) 完全一致。
- `fire_rate` 在分母的作用：低发放率神经元获得更高有效学习率。

**分支二：普通 STDP（特征层）**
```python
layer.w.weight.data += (((0.5 - post) * (pre > 0) * (post > 0) * layer.havconnCombinedExc.T) * layer.eta_stdp).T
```
- 对应公式 (2)：若 pre>0 且 post>0，则权重变化为 `η_STDP * (0.5 - x_post)`。
- `0.5` 的意义：使 post spike 向中等幅度收敛，防止饱和。

**Homeostasis（抑制权重平衡）**
```python
inhW = layer.w.weight.data.T.clone()
inhW[inhW>0] = 0
layer.w.weight.data += (torch.mul(noclp, inhW) * layer.eta_stdp*50).T
```
- 对应公式 (4)：当净输入为正（`noclp > 0`）时，负权重被进一步减小（更负），增强抑制；反之减弱。

**ITP**
```python
layer.thr.data += layer.eta_ip * (layer.x - layer.fire_rate)
```
- 对应公式 (5)：若实际发放 `layer.x` 高于目标 `fire_rate`，阈值增加，降低发放概率。

**笔记任务**：
- 在 `blitnet.py` 的每一行关键代码旁，用中文注释标出对应的论文公式编号。（练习：实际动手改代码注释）

---

### 下午（2.5h）：训练流程 + 数据流水线

#### 2.3 `VPRTempoTrain.py` 训练流程

**逐层训练（Layer-wise Training）**：
```python
for layer_name, _ in sorted(models[0].layer_dict.items(), ...):
    for i, model in enumerate(models):
        model.train_model(train_loader, layer, model, i, prev_layers=trained_layers)
    trained_layers.append(layer_name)
```
- **为什么逐层？** 这是 BLiTNet 的核心训练方式：先训练 feature_layer 提取特征，再固定特征层训练 output_layer。这与反向传播完全不同！
- **前向传递**：`spikes = self.forward(spikes, layer)` → 仅当前层做 `layer.w(spikes)`。
- **已训练层固定**：`with torch.no_grad(): for prev_layer_name in prev_layers...` → 前面层参数冻结。

**Spike Forcing 实现细节**：
- `idx = torch.round((labels - idx_scale) / self.filter)` → 计算当前图像应强制激活的输出神经元索引。
- 注意 `idx_scale`：不同 module 负责不同地理范围，输出神经元索引需要偏移校正。

#### 2.4 `dataset.py` 图像处理流水线

对照 VPRTempo 论文 IV-A：
```
RGB Image
  ↓ [ torchvision.io.read_image ]
Grayscale: 0.299R + 0.587G + 0.114B
  ↓ [ Gamma Correction ]
ρ_norm = ρ^γ, where γ = ln(0.5*255) / ln(mean)
  ↓ [ F.interpolate ]
Resize to 56×56 (default, 论文中用 28×28)
  ↓ [ PatchNormalisePad ]
局部 Z-score: (pixel - μ_patch) / σ_patch, clip to [-1, 1]
  ↓ [ 量化到 uint8 ]
img = 255 * (1 + im_norm) / 2
  ↓ [ SetImageAsSpikes ]
Spike amplitude = pixel_intensity / 255  (即 x ∈ [0, 1])
```

**时序编码的理解**：
- 代码中并未显式模拟时间步内的 theta 振荡相位。
- 实际上，**幅度即时间**：`x=1.0` 表示脉冲在 timestep 最早时刻发放，`x=0.1` 表示较晚时刻发放。
- 论文 Fig. 1A-iii 的抽象：一个 timestep 内，高幅度脉冲先到达，低幅度后到达，从而 STDP 能感知"谁先谁后"。

**笔记任务**：
- 画一张流程图：从 `CustomImageDataset.__getitem__` 到最终 `spikes` 张量的完整变换链。
- 修改 `ProcessImage` 中的 `dims` 或 `patches`，观察 `SetImageAsSpikes` 输出 shape 的变化。

---

### 晚上（0.5h）：推理与评估模块速览

#### 2.5 `VPRTempo.py` 推理
- `nn.Sequential(model.feature_layer.w, model.output_layer.w)` → 推理时完全剥离 STDP/ITP，只做矩阵乘法。
- 多模块：`torch.cat(outputs, dim=1)` → 将所有 module 的输出拼接，形成完整数据库的相似度向量。

#### 2.6 `metrics.py`
- `recallAtK`：对每一查询列，取相似度最高的 K 个行，若包含 GT=1 则算命中。
- `createPR`：对相似度矩阵 S 设定不同阈值，计算 Precision/Recall。

---

## 第三天：动手实验 + 改进点挖掘（6小时）

### 上午（2.5h）：运行完整训练/评估闭环

**实验 A：使用默认参数跑通训练**
```bash
pixi run train --dataset nordland --database_dirs spring,fall \
  --query_dir summer --database_places 500 --query_places 500 \
  --train_new_model --epoch 4
```
- 观察 `Module 1` 进度条，注意 `T = max_module * location_repeat * epoch = 500 * 2 * 4 = 4000` 步。
- 观察 Logger 输出的网络结构信息。

**实验 B：评估刚才训练的模型**
```bash
pixi run eval --dataset nordland --database_dirs spring,fall \
  --query_dir summer --database_places 500 --query_places 500 \
  --PR_curve --sim_mat
```
- 记录 Recall@1, @5, @10。
- 若运行了 `--PR_curve`，观察 `vprtempo/output/DDMMYY-HH-MM-SS/PR_curve_data.json`。

**实验 C：量化模型对比**
```bash
pixi run train_quant --dataset nordland --database_dirs spring,fall \
  --query_dir summer --database_places 500 --query_places 500 \
  --train_new_model --epoch 4

pixi run eval_quant --dataset nordland --database_dirs spring,fall \
  --query_dir summer --database_places 500 --query_places 500
```
- 对比 fp32 与 int8 的 Recall@1 差距。

---

### 下午（2.5h）：消融实验 —— 修改关键超参

**以下实验请逐一执行并记录结果，建立你的"实验记录表"**：

| 实验编号 | 修改项 | 命令/代码位置 | 预期效果 | 实际 Recall@1 |
|---------|--------|-------------|---------|--------------|
| D1 | 修改 `stdp_rate` | `VPRTempoTrain.py` 第 99 行：`stdp_rate=0.005 → 0.01` | 学习更快，可能过拟合 | |
| D2 | 修改 `ip_rate` | 同上，第 98 行：`ip_rate=0.15 → 0.3` | 阈值适应更快 | |
| D3 | 修改 `fire_rate` | 第 97 行：`[0.2, 0.9] → [0.1, 0.5]` | 特征更稀疏 | |
| D4 | 修改 `p`（连接稀疏度） | 第 100 行：`[0.1, 0.5] → [0.05, 0.3]` | 更稀疏的连接 | |
| D5 | 修改图像尺寸 | `--dims 28,28` | 更小输入，更快但可能掉精度 | |
| D6 | 修改 Patch 大小 | `--patches 7` | 更局部的归一化 | |
| D7 | 修改 epoch | `--epoch 2` / `--epoch 8` | 欠拟合 / 过拟合观察 | |
| D8 | 修改 `max_module` | `--max_module 250` | 更多模块，每个模块地点更少 | |

**关键观察问题**：
1. 哪个超参对 Recall@1 影响最大？（通常是 `stdp_rate` 或 `fire_rate`）
2. 减少 `max_module`（增加模块数）是否提高精度？（模块化论文中的权衡）
3. `--epoch 2` 与 `--epoch 4` 的差距大吗？这说明了什么？（BLiTNet 声称只需 1/50 训练轮数）

---

### 晚上（1h）：整理 "可改进点清单"（为发小 Paper 做准备）

基于前三天的理解，以下是一份**可直接用于头脑风暴的改进方向清单**。你的任务是在每个方向后写下 1-2 句自己的初步想法：

#### 方向 1：架构层面
- **增加特征层深度**：当前仅 2 层（LI→LF→LO），能否增加中间层？如何保持逐层训练的可行性？
- **卷积化**：当前是全连接稀疏层，能否改为局部连接（Local Connected）或稀疏卷积？这会破坏哪些 BLiTNet 的假设？
- **残差连接**：在 SNN 中加入 skip connection 是否有助于缓解梯度/脉冲传播问题？

#### 方向 2：编码与输入层面
- **多尺度输入**：同时输入 56×56 和 28×28，分别走不同 feature_layer 后融合。
- **更丰富的时序编码**：当前仅用单脉冲幅度编码，能否使用相位编码或多脉冲burst编码？
- **数据增强**：当前无数据增强。对 VPR 任务，添加随机光照、对比度变化、轻微裁剪是否能提升跨季节鲁棒性？

#### 方向 3：学习规则层面
- **自适应 STDP 学习率**：当前 `η_STDP` 全局退火，能否根据每个神经元的收敛情况局部调整？
- **输出层替代方案**：Spike Forcing 本质上是 delta rule，能否换成更柔软的监督信号（如软标签、温度缩放的 forcing）？
- **遗忘与持续学习**：VPR 机器人需要 lifelong learning，能否加入突触巩固（synaptic consolidation）机制防止灾难性遗忘？

#### 方向 4：系统与评估层面
- **动态模块路由**：当前查询时所有模块并行计算，能否训练一个轻量级"路由器"只激活最相关的 1-2 个模块？这将大幅提升大尺度地图的推理速度。
- **跨数据集泛化**：在 Nordland 训练，在 Oxford RobotCar 查询（zero-shot VPR），需要哪些域自适应修改？
- **更激进的量化**：当前是 INT8，能否结合 PyTorch 量化到 INT4 甚至二值化？对 Recall@1 的影响如何？

#### 方向 5：应用层面
- **事件相机（Event Camera）输入**：VPRTempo 的时序编码天然适合事件数据，如何修改 `dataset.py` 读取 DVS 事件流？
- **SLAM 回环检测集成**：将 VPRTempo 作为 ORB-SLAM3 或 RTAB-Map 的回环检测模块，需要满足哪些实时性接口？

---

## 附录：关键代码速查表

### A. 训练启动链路
```
main.py:parse_network()
  ↓
initialize_and_run_model()
  ↓ 计算模块数、生成模型名
VPRTempoTrain.__init__() / VPRTempoQuantTrain.__init__()
  ↓ 定义 feature_layer, output_layer (blitnet.SNNLayer)
train_new_model() / train_new_model_quant()
  ↓ 逐层循环
model.train_model() → bn.calc_stdp() → bn.clamp_spikes()
  ↓
model.save_model()
```

### B. 推理启动链路
```
main.py:parse_network()
  ↓
initialize_and_run_model()
  ↓
VPRTempo.__init__() (inference=True)
  ↓
run_inference() → DataLoader → model.evaluate()
  ↓ 构建 nn.Sequential → forward → torch.cat
metrics.recallAtK() / metrics.createPR()
```

### C. 图像处理链路
```
CustomImageDataset.__getitem__()
  ↓
ProcessImage.__call__()
  ├── Grayscale (RGB weights)
  ├── Gamma correction
  ├── F.interpolate (resize)
  ├── PatchNormalisePad (local Z-score)
  └── SetImageAsSpikes (uint8 → [0,1] amplitude)
```

### D. 超参数默认值与论文表 I 对照

| 参数 | 论文表 I | 代码默认值 | 代码位置 |
|------|---------|-----------|---------|
| θ_max | 0.5 | `thr_range=[0, 0.5]` | `VPRTempoTrain.py:96` |
| η_init_STDP | 0.005 | `stdp_rate=0.005` | `VPRTempoTrain.py:99` |
| η_init_ITP | 0.15 | `ip_rate=0.15` | `VPRTempoTrain.py:98` |
| f_min, f_max | [0.2, 0.9] | `fire_rate=[0.2, 0.9]` | `VPRTempoTrain.py:97` |
| P_exc | 0.1 | `p=[0.1, 0.5]` 第一个元素 | `VPRTempoTrain.py:100` |
| P_inh | 0.5 | `p=[0.1, 0.5]` 第二个元素 | `VPRTempoTrain.py:100` |
| C | 0.1 | `const_inp=[0, 0.099]` 近似 | `SNNLayer.__init__` 默认 `const_inp=[0,0]`，代码中未显式传入 |

> **注意**：代码中 `const_inp` 默认 `[0,0]`，但 `add_input()` 会加上 `layer.const_inp`。VPRTempo 论文表 I 给出 C=0.1，但在 `VPRTempoTrain.py` 中未向 `feature_layer` 传入 `const_inp` 参数。这是**一个值得深究的差异点**：是工程简化，还是有意为之？（提示：看 BLiTNet 论文，C 主要对 XOR/NOT 等小网络关键，对大型图像网络可能不那么关键）。

---

## 学习检查清单（第三天结束前自测）

- [ ] 我能不看论文，向他人解释 BLiTNet 的四个核心机制（STDP、ITP、Homeostasis、Spike Forcing）。
- [ ] 我能指出 `calc_stdp()` 中每一行对应 BLiTNet 的哪个公式。
- [ ] 我能解释为什么 VPRTempo 用"幅度编码时间"，而不是真的模拟时间步内的 theta 振荡。
- [ ] 我能独立运行 `pixi run train` 和 `pixi run eval`，并读懂输出日志。
- [ ] 我能修改至少 3 个超参数，观察对 Recall@1 的影响，并给出合理的物理解释。
- [ ] 我已经整理出一份"个人改进点清单"，其中至少包含 2 个我认为"最有可能出小 paper"的方向。

---

> **下一步（三天后）**：从你的"改进点清单"中挑选 1 个方向，用 AI 辅助修改代码，设计 3-5 组消融实验，跑通数据并画图。这就是你小 Paper 的实验章节雏形。
