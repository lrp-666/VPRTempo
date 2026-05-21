# VPRTempo 论文公式 ↔ 代码逐行对照学习文档

> 本文档将 VPRTempo (ICRA 2024) 和 BLiTNet 的核心理论公式，精确映射到代码文件的函数与行号。
> 点击每个代码链接可直接跳转到对应文件位置。

---

## 总览映射表

| 论文公式/章节 | 代码文件 | 代码位置（行号） | 核心函数/代码段 |
|--------------|---------|-----------------|----------------|
| **公式 (1)** 神经元状态 | [`blitnet.py`](../../vprtempo/src/blitnet.py) | [L91-L102](../../vprtempo/src/blitnet.py#L91-L102) | `SNNLayer.__init__` 推理模式分支：权重 + 阈值初始化 |
| | [`blitnet.py`](../../vprtempo/src/blitnet.py) | [L338-L346](../../vprtempo/src/blitnet.py#L338-L346) | `add_input(spikes, layer)`：恒定输入 C |
| | [`blitnet.py`](../../vprtempo/src/blitnet.py) | [L367-L377](../../vprtempo/src/blitnet.py#L367-L377) | `clamp_spikes(spikes, layer)`：ReLU 钳制 [0, 0.9] |
| | [`VPRTempoTrain.py`](../../vprtempo/VPRTempoTrain.py) | [L375-L377](../../vprtempo/VPRTempoTrain.py#L375-L377) | `forward(spikes, layer)`：矩阵乘法 y = x·W |
| **公式 (2)** STDP | [`blitnet.py`](../../vprtempo/src/blitnet.py) | [L513-L550](../../vprtempo/src/blitnet.py#L513-L550) | `calc_stdp()` 普通 STDP 分支（`else`） |
| **公式 (3)** 学习率退火 | [`VPRTempoTrain.py`](../../vprtempo/VPRTempoTrain.py) | [L203-L224](../../vprtempo/VPRTempoTrain.py#L203-L224) | `_anneal_learning_rate()` |
| **公式 (4)** Homeostasis | [`blitnet.py`](../../vprtempo/src/blitnet.py) | [L608-L619](../../vprtempo/src/blitnet.py#L608-L619) | `calc_stdp()` 末段：抑制权重动态平衡 |
| **公式 (5)** ITP | [`blitnet.py`](../../vprtempo/src/blitnet.py) | [L579-L588](../../vprtempo/src/blitnet.py#L579-L588) | `calc_stdp()` 中阈值更新 |
| **公式 (6)** Spike Forcing STDP | [`blitnet.py`](../../vprtempo/src/blitnet.py) | [L429-L499](../../vprtempo/src/blitnet.py#L429-L499) | `calc_stdp()` Spike Forcing 分支（`if layer.spk_force`） |
| **公式 (8)** Gamma 校正 | [`dataset.py`](../../vprtempo/src/dataset.py) | [L275-L282](../../vprtempo/src/dataset.py#L275-L282) | `ProcessImage.__call__()` 中 gamma 计算 |
| **公式 (9)** 匹配规则 | [`VPRTempo.py`](../../vprtempo/VPRTempo.py) | [L219-L248](../../vprtempo/VPRTempo.py#L219-L248) | `evaluate()` 中相似度矩阵 → argmax 匹配 |
| **表 I** 超参数 | [`VPRTempoTrain.py`](../../vprtempo/VPRTempoTrain.py) | [L93-L111](../../vprtempo/VPRTempoTrain.py#L93-L111) | `__init__()` 中 `feature_layer` / `output_layer` 定义 |
| **III-B** 模块化 | [`main.py`](../../main.py) | [L101-L145](../../main.py#L101-L145) | `initialize_and_run_model()` 模块数计算与划分 |
| **III-C** 并行张量 | [`VPRTempo.py`](../../vprtempo/VPRTempo.py) | [L327-L352](../../vprtempo/VPRTempo.py#L327-L352) | `forward()` 中 `torch.cat(outputs, dim=1)` |

---

## 公式 (1)：神经元状态

**论文原文（VPRTempo III-A）**

> $$x_j^n = \sum_{i} x_i^m \left(W_{ji}^{+} - W_{ji}^{-}\right) + C - \theta_j^n$$

**代码实现拆解**

该公式被拆分到 4 个代码位置实现：

### ① 权重矩阵初始化（推理模式）
- **文件**：[`vprtempo/src/blitnet.py`](../../vprtempo/src/blitnet.py)
- **行号**：[L91-L102](../../vprtempo/src/blitnet.py#L91-L102)

```python
# 推理模式：合并权重 W = W⁺ + W⁻（W⁻ 已为负数）
self.w = nn.Linear(dims[0], dims[1], bias=False)
self.w.to(device)

# 阈值 θ ∈ [0, θ_max]
self.thr = nn.Parameter(torch.zeros([1, dims[-1]],
                                    device=self.device).uniform_(thr_range[0],
                                                                thr_range[1]))
```

### ② 恒定输入 C
- **文件**：[`vprtempo/src/blitnet.py`](../../vprtempo/src/blitnet.py)
- **行号**：[L338-L346](../../vprtempo/src/blitnet.py#L338-L346)

```python
def add_input(spikes, layer):
    spikes += layer.const_inp   # + C 项
    return spikes
```

### ③ 减去阈值 + ReLU 钳制
- **文件**：[`vprtempo/src/blitnet.py`](../../vprtempo/src/blitnet.py)
- **行号**：[L367-L377](../../vprtempo/src/blitnet.py#L367-L377)

```python
def clamp_spikes(spikes, layer):
    # [Σ x_i W_i + C - θ]₊， clamp 到 [0, 0.9]
    spikes = torch.clamp(torch.sub(spikes, layer.thr), min=0.0, max=0.9)
    return spikes
```

### ④ 矩阵乘法（前向传播）
- **文件**：[`vprtempo/VPRTempoTrain.py`](../../vprtempo/VPRTempoTrain.py)
- **行号**：[L375-L377](../../vprtempo/VPRTempoTrain.py#L375-L377)

```python
def forward(self, spikes, layer):
    spikes = layer.w(spikes)   # y = x · W^T
    return spikes
```

---

## 公式 (2)：普通 STDP（特征层）

**论文原文（VPRTempo III-A）**

> $$\Delta W_{ji}^{nm}(t) = \frac{\eta_{\text{STDP}}(t)}{f_j^n} \cdot \Theta\left(x_i^m(t-1)\right) \cdot \Theta\left(x_j^n(t)\right) \cdot \Big(0.5 - x_j^n(t)\Big)$$

**代码实现**
- **文件**：[`vprtempo/src/blitnet.py`](../../vprtempo/src/blitnet.py)
- **行号**：[L513-L550](../../vprtempo/src/blitnet.py#L513-L550)

```python
else:  # 普通 STDP 分支（feature_layer）
    shape = layer.w.weight.data.shape
    
    # Tile：将向量扩展为矩阵以逐元素更新权重
    pre = torch.tile(torch.reshape(prespike, (shape[1], 1)), (1, shape[0]))
    post = torch.tile(spikes, (shape[1], 1))
    
    # 兴奋权重更新：对应公式 (2)
    # (0.5 - post) 驱动脉冲趋向中等幅度
    # (pre > 0) * (post > 0) 实现 Heaviside Θ(·)
    layer.w.weight.data += (((0.5 - post) * (pre > 0) * (post > 0) *
                              layer.havconnCombinedExc.T) * layer.eta_stdp).T
    
    # 抑制权重更新（方向相反）
    layer.w.weight.data += (((0.5 - post) * (pre > 0) *
                              (post > 0) * layer.havconnCombinedInh.T) * 
                              (layer.eta_stdp * -1)).T
```

**关键对应说明**

| 数学符号 | 代码表达 | 说明 |
|---------|---------|------|
| $\eta_{\text{STDP}}(t)$ | `layer.eta_stdp` | 当前时刻学习率张量 |
| $\Theta(x_i^m(t-1))$ | `(pre > 0)` | 前层脉冲的 Heaviside 函数 |
| $\Theta(x_j^n(t))$ | `(post > 0)` | 后层脉冲的 Heaviside 函数 |
| $0.5 - x_j^n(t)$ | `(0.5 - post)` | 防止饱和的调制因子 |
| $/f_j^n$ | **代码中省略** | 工程简化，避免学习率不均衡 |

---

## 公式 (3)：学习率退火

**论文原文（VPRTempo III-A）**

> $$\eta_{\text{STDP}}(t) = \eta_{\text{STDP}}^{\text{init}} \left(1 - \frac{t}{T}\right)^2$$

**代码实现**
- **文件**：[`vprtempo/VPRTempoTrain.py`](../../vprtempo/VPRTempoTrain.py)
- **行号**：[L203-L224](../../vprtempo/VPRTempoTrain.py#L203-L224)

```python
def _anneal_learning_rate(self, layer, mod, itp, stdp):
    if np.mod(mod, 100) == 0:  # 每 100 步更新一次
        # pt = (1 - mod/T)^2，mod=0 时 pt=1；mod=T 时 pt=0
        pt = pow(float(self.T - mod) / self.T, 2)
        
        layer.eta_ip = torch.mul(itp, pt)     # ITP 退火
        layer.eta_stdp = torch.mul(stdp, pt)  # STDP 退火
    return layer
```

**物理意义**：训练初期满学习率快速收敛，后期逐渐归零精细调整，防止权重震荡。

---

## 公式 (4)：Homeostasis（抑制权重动态平衡）

**论文原文（VPRTempo III-A）**

> $$\hat{W}_{ji}^{-}(t) \leftarrow W_{ji}^{-}(t) \cdot \left[1 - \eta_{\text{STDP}}(t) \cdot \Theta\left(\sum_i x_i(t)\right)\right]$$

**代码实现**
- **文件**：[`vprtempo/src/blitnet.py`](../../vprtempo/src/blitnet.py)
- **行号**：[L608-L619](../../vprtempo/src/blitnet.py#L608-L619)

```python
if torch.any(layer.w.weight.data).item() and layer.eta_stdp != 0:
    # 提取抑制权重（仅保留负值）
    inhW = layer.w.weight.data.T.clone()
    inhW[inhW > 0] = 0
    
    # 根据未钳制的净输入 noclp 调整抑制权重
    # noclp > 0 → 净输入为正 → 乘积 < 0 → 权重更负（抑制增强）
    # noclp < 0 → 净输入为负 → 乘积 > 0 → 权重趋零（抑制减弱）
    layer.w.weight.data += (torch.mul(noclp, inhW) * layer.eta_stdp * 50).T
```

---

## 公式 (5)：ITP（Intrinsic Threshold Plasticity）

**论文原文（VPRTempo III-A）**

> $$\Delta \theta_j^n(t) = \eta_{\text{ITP}}(t) \cdot \left[\Theta\left(x_j^n(t)\right) - f_j^n\right]$$

**代码实现**
- **文件**：[`vprtempo/src/blitnet.py`](../../vprtempo/src/blitnet.py)
- **行号**：[L579-L588](../../vprtempo/src/blitnet.py#L579-L588)

```python
if layer.have_rate and layer.eta_ip > 0.0:
    # θ_new = θ_old + η_ITP * (实际发放指示 - 目标发放率)
    layer.thr.data += layer.eta_ip * (layer.x - layer.fire_rate)
    
    # 阈值不能为负
    layer.thr.data[layer.thr.data < 0] = 0
```

---

## 公式 (6)：Spike Forcing STDP（输出层监督信号）

**论文原文（VPRTempo III-A）**

> $$\Delta W_{ji}^{nm}(t) = \frac{\eta_{\text{STDP}}(t)}{f_j^n} \cdot \left[x_i^m(t-1) \cdot \left(x_{\text{force}} - x_j^n(t)\right)\right]$$

**代码实现**
- **文件**：[`vprtempo/src/blitnet.py`](../../vprtempo/src/blitnet.py)
- **行号**：[L429-L499](../../vprtempo/src/blitnet.py#L429-L499)

```python
if layer.spk_force:  # 输出层分支
    shape = layer.w.weight.data.shape
    idx_sel = torch.arange(int(idx[0]), int(idx[0]) + 1,
                           device=layer.device, dtype=int)
    
    # xdiff = x_force(0.5) - x_actual
    layer.x = torch.full_like(layer.x, 0)
    xdiff = layer.x.index_fill_(-1, idx_sel, 0.5) - spikes
    xdiff.clamp(min=0.0, max=0.9)
    
    # 发放率调制：低发放率的前层神经元获得更高有效学习率
    if prev_layer.fire_rate == None:
        mpre = prespike
    else:
        mpre = prespike / prev_layer.fire_rate
    
    # Tile 后逐元素更新
    pre = torch.tile(torch.reshape(mpre, (shape[1], 1)), (1, shape[0]))
    post = torch.tile(xdiff, (shape[1], 1))
    
    # 兴奋/抑制权重分别更新
    layer.w.weight.data += ((pre * post * layer.havconnCombinedExc.T) * layer.eta_stdp).T
    layer.w.weight.data += ((-pre * post * layer.havconnCombinedInh.T) * (layer.eta_stdp * -1)).T
```

---

## 公式 (8)：Gamma 校正

**论文原文（VPRTempo IV-A）**

> $$\rho_i^{\text{norm}} = \rho_i^\gamma, \quad \gamma = \frac{\ln(0.5 \times 255)}{\ln(\bar{\rho}_i)}$$

**代码实现**
- **文件**：[`vprtempo/src/dataset.py`](../../vprtempo/src/dataset.py)
- **行号**：[L275-L282](../../vprtempo/src/dataset.py#L275-L282)

```python
# Gamma 校正（对应公式 8）
mid = 0.5
mean = torch.mean(img.float())
# γ = ln(0.5 * 255) / ln(mean)
# 偏暗图像 gamma < 1（增强暗部）；偏亮图像 gamma > 1（压缩亮部）
gamma = math.log(mid * 255) / math.log(mean)
img = torch.pow(img, gamma).clip(0, 255)
```

---

## 公式 (9)：匹配规则

**论文原文（VPRTempo IV-C）**

> $$\hat{p} = \arg\max_i x_i$$

**代码实现**
- **文件**：[`vprtempo/VPRTempo.py`](../../vprtempo/VPRTempo.py)
- **行号**：[L219-L248](../../vprtempo/VPRTempo.py#L219-L248)

```python
# 对每一查询列，取相似度最高的行索引 → argmax 匹配
# out 形状：[database_places, query_places]
# out[i, j] 表示查询 j 与数据库地点 i 的相似度
# 外部通过 recallAtK(out, GT, K=1) 验证： top-1 是否正确

# 实际上，evaluate() 不直接调用 argmax，而是将 out 传入 metrics.py：
# recallAtK(out, GT, K=1) 对每列取 top-1，检查是否在 GT 正例中
```

更直接的匹配在推理时隐含发生：每个查询图像的输出脉冲向量中，**幅度最大的神经元索引**即被识别为匹配地点。

---

## 表 I：超参数默认值

**论文原文（VPRTempo IV-A）**

| 参数 | 论文值 | 代码默认值 |
|------|--------|-----------|
| $\theta_{\max}$ | 0.5 | `thr_range=[0, 0.5]` |
| $\eta_{\text{STDP}}^{\text{init}}$ | 0.005 | `stdp_rate=0.005` |
| $\eta_{\text{ITP}}^{\text{init}}$ | 0.15 | `ip_rate=0.15` |
| $f_{\min}, f_{\max}$ | [0.2, 0.9] | `fire_rate=[0.2, 0.9]` |
| $P_{\text{exc}}$ | 0.1 | `p=[0.1, 0.5]` 首元素 |
| $P_{\text{inh}}$ | 0.5 | `p=[0.1, 0.5]` 次元素 |
| $C$ | 0.1 | 代码中默认 `[0,0]`（工程差异） |

**代码实现**
- **文件**：[`vprtempo/VPRTempoTrain.py`](../../vprtempo/VPRTempoTrain.py)
- **行号**：[L93-L111](../../vprtempo/VPRTempoTrain.py#L93-L111)

```python
self.add_layer(
    'feature_layer',
    dims=[self.input, self.feature],
    thr_range=[0, 0.5],      # θ_max = 0.5
    fire_rate=[0.2, 0.9],    # f_min, f_max
    ip_rate=0.15,            # η_init_ITP
    stdp_rate=0.005,         # η_init_STDP
    p=[0.1, 0.5],            # P_exc, P_inh
    device=self.device
)
self.add_layer(
    'output_layer',
    dims=[self.feature, self.output],
    ip_rate=0.15,
    stdp_rate=0.005,
    p=[1.0, 1.0],            # 全连接
    spk_force=True,          # 启用 Spike Forcing
    device=self.device
)
```

---

## III-B：模块化（Modular Place Representation）

**论文原文（VPRTempo III-B）**

> 将大数据集拆分为多个专家网络（Expert Modules），每个模块负责不重叠的地点子集。
> 正式描述：$U = \bigcup_{i=1}^{|N|} N_i$，其中 $N_i \cap N_j = \emptyset \; \forall i \neq j$

**代码实现**
- **文件**：[`main.py`](../../main.py)
- **行号**：[L101-L145](../../main.py#L101-L145)

```python
def initialize_and_run_model(args, dims):
    # 1. 计算模块数量
    places = args.database_places
    num_modules = 1
    while places > args.max_module:
        places -= args.max_module
        num_modules += 1

    # 2. 处理非整除情况
    remainder = args.database_places % args.max_module
    if remainder != 0:
        out_dim = int((args.database_places - remainder) / (num_modules - 1))
        final_out_dim = remainder
    else:
        out_dim = int(args.database_places / num_modules)
        final_out_dim = out_dim

    # 3. 创建模块列表并训练/推理
    # ...（fp32 训练 / QAT 训练 / fp32 推理 / QAT 推理 四条分支）
```

**模块化优势（论文总结）**：
1. 小网络训练更快、更精确
2. 不同模块的随机种子异质性降低误匹配率
3. 可线性扩展到 27k+ 地点

---

## III-C：并行张量与高效推理

**论文原文（VPRTempo III-C）**

> 使用 3D 权重张量 $T \in \mathbb{R}^{|N| \cdot |L_i| \cdot |L_j|}$ 和图像张量 $I \in \mathbb{R}^{|N| \cdot 1 \cdot |L_i|}$，
> 利用并行计算同时处理所有模块，查询时间复杂度为 $O(\log n)$。

**代码实现**
- **文件**：[`vprtempo/VPRTempo.py`](../../vprtempo/VPRTempo.py)
- **行号**：[L327-L352](../../vprtempo/VPRTempo.py#L327-L352)

```python
def forward(self, spikes):
    in_spikes = spikes.detach().clone()
    outputs = []

    # 遍历所有 module，独立计算输出
    for inference in self.inferences:
        out_spikes = inference(in_spikes)
        outputs.append(out_spikes)

    # 在维度 1（输出神经元维度）上拼接
    # 例如 3 个模块各输出 [1, 1100] → 拼接为 [1, 3300]
    concatenated_output = torch.cat(outputs, dim=1)
    
    return concatenated_output
```

**推理加速关键**：训练时用逐层 `clamp_spikes`，推理时用 `nn.Sequential` 连乘，跳过所有中间钳制。

---

## 附录 A：完整数据流图

```
[原始图像 RGB]
    ↓ [read_image]
[vprtempo/src/dataset.py: CustomImageDataset.__getitem__]
    ↓ [ProcessImage]
Step 1: Grayscale (0.299R + 0.587G + 0.114B)
    ↓
Step 2: Gamma 校正 [公式 8] → (dataset.py L275)
    ↓
Step 3: Resize → 56×56
    ↓
Step 4: PatchNormalisePad → 局部 Z-score
    ↓
Step 5: uint8 量化 → [0, 255]
    ↓
Step 6: SetImageAsSpikes → 幅度 x ∈ [0, 1]
    ↓
[vprtempo/src/blitnet.py: SNNLayer]
训练时：
    y = layer.w(spikes)              [公式 1: Σ x_i W_i]
    y = add_input(y, layer)          [公式 1: + C]
    y = clamp_spikes(y, layer)       [公式 1: - θ, ReLU]
    ↓
    calc_stdp(pre, post, noclp, ...) [公式 2/4/5/6]
    _anneal_learning_rate            [公式 3]
推理时：
    nn.Sequential(feature.w, output.w)  [III-C 高效连乘]
    ↓
[vprtempo/VPRTempo.py: evaluate]
    torch.cat(outputs, dim=1)        [III-C 并行拼接]
    ↓
    相似度矩阵 S [database × query]
    ↓
    recallAtK(S, GT, K=1,5,10...)    [IV-C 评估]
```

---

## 附录 B：文件与行号速查

| 想看的理论 | 打开文件 | 跳到行号 |
|-----------|---------|---------|
| 神经元状态公式 (1) | `vprtempo/src/blitnet.py` | [L91](../../vprtempo/src/blitnet.py#L91) |
| STDP 公式 (2) | `vprtempo/src/blitnet.py` | [L513](../../vprtempo/src/blitnet.py#L513) |
| 退火公式 (3) | `vprtempo/VPRTempoTrain.py` | [L203](../../vprtempo/VPRTempoTrain.py#L203) |
| Homeostasis 公式 (4) | `vprtempo/src/blitnet.py` | [L608](../../vprtempo/src/blitnet.py#L608) |
| ITP 公式 (5) | `vprtempo/src/blitnet.py` | [L579](../../vprtempo/src/blitnet.py#L579) |
| Spike Forcing 公式 (6) | `vprtempo/src/blitnet.py` | [L429](../../vprtempo/src/blitnet.py#L429) |
| Gamma 校正 公式 (8) | `vprtempo/src/dataset.py` | [L275](../../vprtempo/src/dataset.py#L275) |
| 匹配规则 公式 (9) | `vprtempo/VPRTempo.py` | [L219](../../vprtempo/VPRTempo.py#L219) |
| 超参数 表 I | `vprtempo/VPRTempoTrain.py` | [L93](../../vprtempo/VPRTempoTrain.py#L93) |
| 模块化 III-B | `main.py` | [L101](../../main.py#L101) |
| 并行张量 III-C | `vprtempo/VPRTempo.py` | [L327](../../vprtempo/VPRTempo.py#L327) |
