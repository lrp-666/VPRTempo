# MIT License
#
# Copyright (c) 2023 Adam Hines, Peter Stratton, Michael Milford, Tobias Fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
================================================================================
Part 1: 导入模块 (Imports)
================================================================================
本文件是 VPRTempo 的核心脉冲神经网络层实现，直接对应 BLiTNet 论文的
公式 (1)-(6) 以及 VPRTempo 论文的 III-A 节。

关键理论对应关系：
- SNNLayer 类      → BLiTNet 神经元层 + VPRTempo 网络架构
- addWeights()     → 权重初始化（含稀疏连接 + L1 归一化）
- add_input()      → 公式 (1) 中的恒定输入 C
- clamp_spikes()   → 公式 (1) 中的脉冲钳制（防止消失/爆炸）
- calc_stdp()      → 公式 (2) 普通 STDP + 公式 (6) Spike Forcing + 公式 (5) ITP + 公式 (4) Homeostasis
================================================================================
"""
import torch
import torch.nn as nn
import numpy as np


# ================================================================================
# Part 2: SNNLayer 类 —— 脉冲神经网络层的核心定义
# ================================================================================
# 该类实现了 BLiTNet 的核心神经元层，支持两种模式：
#   1. 训练模式 (inference=False): 包含完整的 STDP、ITP、Homeostasis 机制
#   2. 推理模式 (inference=True) : 仅保留权重矩阵和阈值，用于前向传播
# ================================================================================
class SNNLayer(nn.Module):
    def __init__(self, 
                 dims=[0,0],  # 输入维度和输出维度，如 [3136, 6272] 表示输入 56x56=3136 像素，输出为输入的 2 倍（VPRTempo 默认架构）
                 thr_range=[0,0], # 初始发放阈值的均匀分布范围 [min, max]，论文表 I 中 θ_max=0.5，即 thr_range=[0, 0.5]
                 fire_rate=[0,0], # 目标发放率范围 [f_min, f_max]，论文表 I 中 [0.2, 0.9]。特征层神经元会被赋予从低到高线性分布的目标发放率
                 ip_rate=0, # ITP (Intrinsic Threshold Plasticity) 学习率 η_ITP，论文表 I 中初始值为 0.15
                 stdp_rate=0, # STDP 学习率 η_STDP，论文表 I 中初始值为 0.005
                 const_inp=[0,0], # 恒定输入 C 的范围 [min, max]，对应公式 (1) 中的 C。BLiTNet 论文中 C≈0.1，但 VPRTempo 代码中默认 [0,0]
                 p=[1,1],  # 连接概率 [P_exc, P_inh]，论文表 I 中 [0.1, 0.5] TODO 所以这里是全连接吗？
                 spk_force=False,# 是否启用 Spike Forcing（仅在输出层 LO 使用），对应 VPRTempo 论文 III-A 末段和公式 (6)
                 device=None, # 计算设备 (cuda:0 / mps / cpu)，默认为 None 表示自动选择
                 inference=False, # True 表示推理模式，False 表示训练模式。推理模式下不包含学习机制，仅保留前向传播所需的权重和阈值。
                 args=None # 额外的命令行参数（预留）
                 ):
        """
        ================================================================================
        函数层说明：SNNLayer 的构造函数
        ================================================================================
        参数说明：
            dims      : [输入维度, 输出维度]，如 [3136, 6272] 表示输入 56x56=3136 像素，
                        特征层输出为输入的 2 倍（VPRTempo 默认架构）
            thr_range : 初始发放阈值的均匀分布范围 [min, max]，论文表 I 中 θ_max=0.5，
                        即 thr_range=[0, 0.5]
            fire_rate : 目标发放率范围 [f_min, f_max]，论文表 I 中 [0.2, 0.9]。
                        特征层神经元会被赋予从低到高线性分布的目标发放率
            ip_rate   : ITP (Intrinsic Threshold Plasticity) 学习率 η_ITP，
                        论文表 I 中初始值为 0.15
            stdp_rate : STDP 学习率 η_STDP，论文表 I 中初始值为 0.005
            const_inp : 恒定输入 C 的范围 [min, max]，对应公式 (1) 中的 C。
                        BLiTNet 论文中 C≈0.1，但 VPRTempo 代码中默认 [0,0]
            p         : 连接概率 [P_exc, P_inh]，论文表 I 中 [0.1, 0.5]
            spk_force : 是否启用 Spike Forcing（仅在输出层 LO 使用），
                        对应 VPRTempo 论文 III-A 末段和公式 (6)
            device    : 计算设备 (cuda:0 / mps / cpu)
            inference : True 表示推理模式，False 表示训练模式
            args      : 额外的命令行参数（预留）
        ================================================================================
        """
        super(SNNLayer, self).__init__()

        # ----------------------------------------
        # 逐行说明：设备设置
        # ----------------------------------------
        self.device = device  # 存储当前层运行的设备地址

    # ================================================================================
    # 分支 A：推理模式 (inference=True)
    # ================================================================================
        # 推理时不需要 STDP/ITP 等学习机制，仅需前向传播所需的权重和阈值。
        # VPRTempo.py 和 VPRTempoQuant.py 在构建推理模型时会使用此分支。
        # ================================================================================
        if inference:
            # 创建一个无偏置的线性层作为合并后的权重矩阵 W = W⁺ + W⁻
            # 对应公式 (1) 中的 (W⁺_ji - W⁻_ji)
            self.w = nn.Linear(dims[0], dims[1], bias=False)
            self.w.to(device)  # 将权重张量迁移到指定设备

            # 初始化发放阈值 θ，在 [thr_range[0], thr_range[1]] 上均匀分布
            # 对应公式 (1) 中的 θ ∈ [0, θ_max]
            self.thr = nn.Parameter(torch.zeros([1, dims[-1]],
                                                device=self.device).uniform_(thr_range[0],
                                                                            thr_range[1]))

    # ================================================================================
    # 分支 B：训练模式 (inference=False)
    # ================================================================================
    # 训练模式需要完整的神经元状态、学习率、权重、发放率等参数。
    # 这是 BLiTNet 理论的核心实现所在。
    # ================================================================================
        else:
        # ----------------------------------------
        # 逐行说明：标量参数转列表（健壮性处理）
        # ----------------------------------------
            # 如果用户传入的是标量而非列表，自动转换为等值列表
            if np.isscalar(thr_range): thr_range = [thr_range, thr_range]
            if np.isscalar(fire_rate): fire_rate = [fire_rate, fire_rate]
            if np.isscalar(const_inp): const_inp = [const_inp, const_inp]

        # ----------------------------------------
        # 逐行说明：初始化神经元状态张量
        # ----------------------------------------
            # self.x: 当前层神经元的输出状态（脉冲幅度），形状 [1, 输出维度]
            #         对应公式 (1) 中的 x_j^n(t)
            self.x = torch.zeros([1, dims[-1]], device=self.device)

            # self.eta_ip: ITP 学习率张量，对应公式 (5) 中的 η_ITP(t)
            #              初始为固定值，训练过程中会被 _anneal_learning_rate 退火
            self.eta_ip = torch.tensor(ip_rate, device=self.device)

            # self.eta_stdp: STDP 学习率张量，对应公式 (2)(6) 中的 η_STDP(t)
            #                同样会在训练中被退火（公式 3）
            self.eta_stdp = torch.tensor(stdp_rate, device=self.device)

        # ----------------------------------------
        # 逐行说明：初始化可学习参数
        # ----------------------------------------
            # self.thr: 发放阈值 θ，nn.Parameter 表示这是需要被保存到 state_dict 的参数
            #           形状 [1, 输出维度]，在 [0, θ_max] 上均匀初始化
            self.thr = nn.Parameter(torch.zeros([1, dims[-1]],
                                                device=self.device).uniform_(thr_range[0],
                                                                            thr_range[1]))

            # self.fire_rate: 目标发放率 f，形状 [1, 输出维度]
            #                 初始在 [f_min, f_max] 上均匀分布
            self.fire_rate = torch.zeros([1, dims[-1]], device=self.device).uniform_(fire_rate[0], fire_rate[1])

            # ----------------------------------------
            # 逐行说明：目标发放率的序列化分配
            # ----------------------------------------
            # BLiTNet 论文 Fig. 5 发现：让不同神经元具有不同的目标发放率，
            # 可以使网络同时学习到稀疏的特异性特征（低发放率）和
            # 泛化的分布式特征（高发放率），从而获得最佳性能。
            # 这里将 [f_min, f_max] 线性分配给每个输出神经元。
            # ----------------------------------------
            if not torch.all(self.fire_rate == 0).item():  # 如果目标发放率不全为 0
                fstep = (fire_rate[1] - fire_rate[0]) / dims[-1]  # 计算线性步长
                for i in range(dims[-1]):
                    self.fire_rate[:, i] = fire_rate[0] + fstep * (i + 1)

            # self.have_rate: 布尔标志，表示该层是否有非零的目标发放率
            #                 用于控制是否启用 ITP（ITP 需要目标发放率作为参考）
            self.have_rate = torch.any(self.fire_rate[:, 0] > 0.0).to(self.device)

            # self.const_inp: 恒定输入 C，对应公式 (1) 中的 C
            #                 形状 [1, 输出维度]，在 [const_inp[0], const_inp[1]] 上均匀分布
            self.const_inp = torch.zeros([1, dims[-1]], device=self.device).uniform_(const_inp[0], const_inp[1])

            # 保存连接概率和维度信息供后续使用
            self.p = p
            self.dims = dims

            # ----------------------------------------
            # 逐行说明：额外状态变量
            # ----------------------------------------
            self.set_spks = []   # 预留：用于存储强制发放的脉冲记录
            self.sspk_idx = 0    # 预留：强制发放的脉冲索引计数器
            self.spikes = torch.empty([], dtype=torch.float64)  # 预留：脉冲存储张量
            self.spk_force = spk_force  # 是否启用 Spike Forcing 的标志位

            # ----------------------------------------
            # 逐行说明：创建兴奋性权重 W⁺
            # ----------------------------------------
            # 兴奋性权重初始范围 [0, 1]，连接概率 p[0]=P_exc（论文表 I 为 0.1）
            # 对应公式 (1) 中的 W⁺_ji
            self.exc = nn.Linear(dims[0], dims[1], bias=False)
            self.exc.weight = self.addWeights(dims=dims,
                                              W_range=[0, 1],   # 兴奋权重：非负
                                              p=p[0],            # 兴奋连接概率 P_exc
                                              device=device)

            # ----------------------------------------
            # 逐行说明：创建抑制性权重 W⁻
            # ----------------------------------------
            # 抑制性权重初始范围 [-1, 0]，连接概率 p[-1]=P_inh（论文表 I 为 0.5）
            # 对应公式 (1) 中的 W⁻_ji
            self.inh = nn.Linear(dims[0], dims[1], bias=False)
            self.inh.weight = self.addWeights(dims=dims,
                                              W_range=[-1, 0],  # 抑制权重：非正
                                              p=p[-1],           # 抑制连接概率 P_inh
                                              device=device)

            # ----------------------------------------
            # 逐行说明：建立连接掩码（Mask）
            # ----------------------------------------
            # havconnExc: 布尔张量，标记哪些位置是兴奋性连接（权重 > 0）
            # havconnInh: 布尔张量，标记哪些位置是抑制性连接（权重 < 0）
            # 这些掩码在 calc_stdp 中用于防止权重在学习过程中改变符号
            self.havconnExc = self.exc.weight > 0
            self.havconnInh = self.inh.weight < 0

            # ----------------------------------------
            # 逐行说明：合并兴奋性和抑制性权重
            # ----------------------------------------
            # 为了前向传播时只需一次矩阵乘法，将 W⁺ 和 W⁻ 相加为合并权重 W：
            #   W = W⁺ + W⁻
            # 公式 (1) 中的净输入为：Σ x_i (W⁺_ji - W⁻_ji)
            # 注意：这里 W⁻ 已经是负数，所以相加等价于 W⁺ - |W⁻|
            self.w = nn.Linear(dims[0], dims[1], bias=False)
            self.w.weight = nn.Parameter(torch.add(self.exc.weight, self.inh.weight))

            # 对合并后的权重同样建立兴奋/抑制掩码
            # 用于后续 STDP 更新时区分正权重和负权重的更新规则
            self.havconnCombinedExc = self.w.weight > 0
            self.havconnCombinedInh = self.w.weight < 0

            # 删除独立的 exc/inh 对象，释放内存
            # 此后所有前向传播和权重更新都通过 self.w 进行
            del self.exc, self.inh

    # ================================================================================
    # 函数：addWeights
    # ================================================================================
    # 功能：按照 BLiTNet 论文方法初始化稀疏连接权重矩阵
    # 理论依据：
    #   1. 正态分布初始化 + 符号裁剪 → 确保兴奋/抑制权重的极性
    #   2. 连接概率 p → 实现稀疏连接（对应论文 P_exc=0.1, P_inh=0.5）
    #   3. L1 归一化 → 对应公式 (4) 上方的权重归一化规则：
    #      "The total excitatory weight to each postsynaptic neuron j is normalised
    #       to a constant k each timestep"
    # ================================================================================
    def addWeights(self, W_range=[0,0], p=[0,0], dims=[0,0], device=None):
        """
        逐层说明：
            本函数实现了 BLiTNet 论文中的权重初始化策略。
            输入：
                W_range: 权重初始范围的 [min, max]，如兴奋性 [0,1]、抑制性 [-1,0]
                p      : 连接概率，控制稀疏度。只有 (nrow × ncol × p) 个连接被保留
                dims   : [输入维度, 输出维度]
                device : 目标计算设备
            输出：
                一个经过初始化、裁剪、稀疏化、L1 归一化后的 nn.Parameter 权重矩阵
        """
        # ----------------------------------------
        # 逐行说明：参数校验
        # ----------------------------------------
        device = device  # 冗余赋值，保持接口一致性
        if np.isscalar(W_range): W_range = [W_range, W_range]  # 标量转列表

        # ----------------------------------------
        # 逐行说明：确定权重矩阵维度
        # ----------------------------------------
        nrow = dims[1]   # 输出神经元数量（行数）
        ncol = dims[0]   # 输入神经元数量（列数）

        # ----------------------------------------
        # 逐行说明：计算正态分布参数
        # ----------------------------------------
        # Wmn (mean): 范围中点，如 [0,1] 对应 0.5，[-1,0] 对应 -0.5
        # Wsd (std) : 范围跨度 / 6，对应正态分布的 3σ 原则（覆盖 99.7% 概率）
        Wmn = (W_range[0] + W_range[1]) / 2.0  # 范围中点作为均值
        Wsd = (W_range[1] - W_range[0]) / 6.0  # 范围跨度的 1/6 作为标准差

        # ----------------------------------------
        # 逐行说明：初始化空张量（历史遗留，实际未使用此维度） TODO 可以删除
        # ----------------------------------------
        W = torch.empty((0, nrow, ncol), device=device)

        # ----------------------------------------
        # 逐行说明：从正态分布采样权重
        # ----------------------------------------
        W = torch.empty(nrow, ncol, device=device).normal_(mean=Wmn, std=Wsd)

        # ----------------------------------------
        # 逐行说明：根据符号裁剪权重
        # ----------------------------------------
        # 若 W_range[-1] != 0（即上限不为 0，如 [0,1]），则为兴奋性权重，删除负值
        # 否则（如 [-1,0]），则为抑制性权重，删除正值
        # 这保证了兴奋权重始终 ≥0，抑制权重始终 ≤0
        if W_range[-1] != 0:
            W[W < 0] = 0.0    # 兴奋性权重裁剪：负值置零
        else:
            W[W > 0] = 0.0    # 抑制性权重裁剪：正值置零

        # ----------------------------------------
        # 逐行说明：根据连接概率稀疏化权重
        # ----------------------------------------
        # 生成 [0,1) 均匀随机矩阵，大于 p 的位置被置零
        # 保留概率约为 p，实现论文中的稀疏连接
        setzero = np.random.rand(nrow, ncol) > p
        if setzero.any():
           W[setzero] = 0.0   # 将未连接的位置置零

        # ----------------------------------------
        # 逐行说明：L1 归一化（列归一化）
        # ----------------------------------------
        # 对每一列（每个输入神经元到所有输出神经元的连接）计算 L1 范数
        # ord=1 表示 L1 范数（绝对值之和），axis=0 表示沿列方向
        # 这对应 BLiTNet 论文中的公式：W_ji ← k * W_ji / Σ_i W_ji
        # 其中 k 是归一化常数，确保每个输入神经元的总 outgoing 权重恒定
        nrm = torch.linalg.norm(W[len(W)-1], ord=1, axis=0)

        # 防止除以零：如果某列全为零（无连接），将其范数设为 1.0
        nrm[nrm == 0.0] = 1.0

        # 执行归一化：W_ij = W_ij / ||W_·j||_1
        W = nn.Parameter(W / nrm)

        return W


# ================================================================================
# Part 3: 辅助函数 —— 神经元动力学与学习规则
# ================================================================================
# 以下三个函数实现了 BLiTNet/VPRTempo 论文中的核心神经元操作：
#   add_input()    → 公式 (1) 中的恒定输入 C
#   clamp_spikes() → 公式 (1) 中的脉冲钳制（防止脉冲消失/爆炸）
#   calc_stdp()    → 公式 (2) STDP + 公式 (5) ITP + 公式 (4) Homeostasis + 公式 (6) Spike Forcing
# ================================================================================

# ================================================================================
# 函数：add_input
# ================================================================================
# 功能：为神经元状态添加恒定输入（Constant Input / Bias）
# 理论对应：BLiTNet 公式 (1) 中的 C 项：
#          x_j^n = Σ_i x_i^m (W⁺_ji - W⁻_ji) + C - θ_j^n
# 注意：在 VPRTempo 的默认配置中，const_inp 被初始化为 [0,0] 附近的小值，
#      因此此函数的实际影响较小。BLiTNet 原文中 C≈0.1 对小网络（XOR/NOT）更关键。
# ================================================================================
def add_input(spikes, layer):
    # spikes: 当前神经元的输入/状态张量，形状通常为 [batch, 输出维度]
    # layer : SNNLayer 实例，包含 const_inp 参数

    # 将恒定输入加到当前状态上
    # 对应公式 (1) 中的 + C 项
    spikes += layer.const_inp

    return spikes


# ================================================================================
# 函数：clamp_spikes
# ================================================================================
# 功能：计算神经元发放状态并钳制到合理范围
# 理论对应：BLiTNet 公式 (1) 的核心运算：
#          x_j^n = [Σ_i x_i^m (W⁺_ji - W⁻_ji) + C - θ_j^n]₊
#          其中 [·]₊ 表示 max(·, 0)，即 ReLU
# 代码实现：
#          1. torch.sub(spikes, layer.thr) → 减去阈值 (input - θ)
#          2. torch.clamp(..., min=0.0, max=0.9) → 钳制到 [0, 0.9]
# 为什么 max=0.9 而不是 1.0？
#          BLiTNet 论文指出：若 mean spike amplitude 设为 1.0，则一半脉冲会被裁剪到最大值。
#          使用 0.9 作为上限，给脉冲幅度留出动态余量，防止饱和。
# 为什么需要钳制？
#          这是 BLiTNet 解决"脉冲消失/爆炸问题"（Vanishing Spike Problem）的关键：
#          - 下限 0.0 保证脉冲非负（无脉冲 = 0）
#          - 上限 0.9 防止脉冲幅度无限增长导致网络饱和（avalanche）
# ================================================================================
def clamp_spikes(spikes, layer):
    # spikes: 当前神经元在减去阈值前的净输入张量 TODO 应该是包含了这个C吧？ 是的
    # layer : SNNLayer 实例，包含 thr（阈值）参数

    # Step 1: 减去阈值 θ（对应公式 1 中的 - θ 项）
    # Step 2: 钳制到 [0.0, 0.9]
    #   - min=0.0: ReLU 操作，负值置零（未超过阈值则不发放）
    #   - max=0.9: 防止脉冲幅度过大导致雪崩
    spikes = torch.clamp(torch.sub(spikes, layer.thr), min=0.0, max=0.9)

    return spikes


# ================================================================================
# 函数：calc_stdp —— 全文件最核心的函数
# ================================================================================
# 功能：根据 STDP 规则更新权重，同时执行 ITP 阈值调整和 Homeostasis 抑制平衡
# 理论对应：
#   1. 若 layer.spk_force=True（输出层）：
#      执行 VPRTempo 公式 (6) 的 Spike Forcing STDP：
#      ΔW_ji^n^m(t) = η_STDP(t) / f_j^n · [x_i^m(t-1) · (x_force - x_j^n(t))]
#
#   2. 若 layer.spk_force=False（特征层）：
#      执行 VPRTempo 公式 (2) 的普通 STDP：
#      ΔW_ji^n^m(t) = η_STDP(t) / f_j^n · Θ(x_i^m(t-1)) · Θ(x_j^n(t)) · (0.5 - x_j^n(t))
#
#   3. ITP（公式 5）：
#      Δθ_j^n(t) = η_ITP(t) · [Θ(x_j^n(t)) - f_j^n]
#
#   4. Homeostasis（公式 4）：
#      当净输入为正时，增强抑制权重（使其更负），维持 E/I 平衡
#
# 输入参数说明：
#   prespike   : 前一层（pre-synaptic）的脉冲输出 x_i^m(t-1)
#   spikes     : 当前层（post-synaptic）的脉冲输出 x_j^n(t)（经过 clamp 后的）
#   noclp      : 当前层在 clamp 前的原始净输入（未减去阈值、未钳制）
#                用于 Homeostasis，判断净输入的正负
#   layer      : 当前要更新的 SNNLayer 实例
#   idx        : 当前样本对应的输出神经元索引（仅用于 Spike Forcing）
#   prev_layer : 前一层的 SNNLayer 实例（用于获取其 fire_rate 以调制学习率）
# ================================================================================
def calc_stdp(prespike, spikes, noclp, layer, idx, prev_layer=None):
    """
    逐层说明：
        本函数实现了 BLiTNet/VPRTempo 的完整学习规则，包括：
        - Spike Forcing（输出层监督信号）
        - 标准 STDP（特征层无监督特征学习）
        - 权重符号保持（兴奋/抑制不翻转）
        - ITP（阈值可塑性）
        - Homeostasis（抑制权重动态平衡）
    """

    # ================================================================================
    # 分支一：Spike Forcing（输出层监督学习）
    # ================================================================================
    # 适用场景：仅在 output_layer（LO）使用，spk_force=True
    # 理论来源：VPRTempo 论文 III-A "Spike forcing" 小节，公式 (6)
    # 核心思想：
    #   对于当前输入图像对应的地点 p_i，强制其分配的输出神经元 n_i 产生幅度为 0.5 的脉冲。
    #   然后计算实际输出与强制输出之间的差异（delta error），用这个误差来驱动 STDP。
    #   这本质上是一种 delta learning rule / supervised readout。
    # ================================================================================
    if layer.spk_force:

        # ----------------------------------------
        # 逐行说明：获取权重矩阵维度
        # ----------------------------------------
        # shape[0] = 输入维度 (ncol)，shape[1] = 输出维度 (nrow)
        shape = layer.w.weight.data.shape

        # ----------------------------------------
        # 逐行说明：确定需要强制发放的神经元索引
        # ----------------------------------------
        # idx 是当前训练样本对应的地点标签经换算后的输出神经元索引
        # idx_sel 将其转换为整数张量，用于 index_fill_ 操作
        idx_sel = torch.arange(int(idx[0]), int(idx[0]) + 1,
                               device=layer.device,
                               dtype=int)

        # ----------------------------------------
        # 逐行说明：计算强制脉冲与实际脉冲的差异
        # ----------------------------------------
        # layer.x 初始化为全零张量，形状 [1, 输出维度]
        # index_fill_(-1, idx_sel, 0.5)：在第 idx[0] 个位置填入 0.5（即 x_force）
        # xdiff = x_force - x_actual，即公式 (6) 中的 (x_force - x_j^n(t))
        layer.x = torch.full_like(layer.x, 0)
        xdiff = layer.x.index_fill_(-1, idx_sel, 0.5) - spikes
        xdiff.clamp(min=0.0, max=0.9)  # 将差异也钳制到合理范围

        # ----------------------------------------
        # 逐行说明：发放率调制的前层脉冲
        # ----------------------------------------
        # BLiTNet 论文指出：低发放率的前层神经元应该具有更高的有效学习率。
        # 原因：TODO 如果一个前层神经元很少发放，那么当它确实发放时，其携带的信息更珍贵，
        #      应该对后层权重产生更大的影响。
        # 实现：mpre = prespike / prev_layer.fire_rate
        #      当 fire_rate 低时，除法结果大，学习率等效增强。
        if prev_layer.fire_rate == None:
            mpre = prespike  # 如果前层无目标发放率，不做调制
        else:
            # 发放率调制：低发放率 → 高有效学习率
            mpre = prespike / prev_layer.fire_rate

        # ----------------------------------------
        # 逐行说明：张量广播（Tile）以匹配权重矩阵维度
        # ----------------------------------------
        # 为了对权重矩阵进行逐元素更新，需要将 pre 和 post 从向量扩展为矩阵：
        #   pre : [输入维度, 1] → tile → [输入维度, 输出维度]
        #   post: [1, 输出维度]  → tile → [输入维度, 输出维度]
        # 这样 pre[i,j] 表示从输入 i 到输出 j 的前层脉冲
        #     post[i,j] 表示从输入 i 到输出 j 的后层误差 就是xdiff
        pre = torch.tile(torch.reshape(mpre, (shape[1], 1)), (1, shape[0]))
        post = torch.tile(xdiff, (shape[1], 1))

        # ----------------------------------------
        # 逐行说明：应用 Spike Forcing STDP 权重更新（公式 6）
        # ----------------------------------------
        # 兴奋性权重更新（正权重）：
        #   ΔW⁺ = pre * post * η_STDP
        #   其中 pre = x_pre / f_pre（或 x_pre），post = (x_force - x_post)
        #   即 ΔW⁺ ∝ x_pre * (x_force - x_post) * η_STDP
        #   若 x_post < x_force（实际输出不足），post > 0，权重增强（potentiation）
        #   若 x_post > x_force（实际输出过度），post < 0，权重减弱（depression）
        layer.w.weight.data += ((pre * post * layer.havconnCombinedExc.T) *
                                       layer.eta_stdp).T

        # 抑制性权重更新（负权重）：
        #   ΔW⁻ = -pre * post * (-η_STDP) = pre * post * η_STDP
        #   注意负号：抑制权重的更新方向与兴奋权重相反
        #   这是因为抑制输入对输出的作用是负的：更强的抑制连接会减小输出。
        #   如果希望输出增加（post > 0），需要减弱抑制（使负权重趋近于零）。
        layer.w.weight.data += ((-pre * post * layer.havconnCombinedInh.T) *
                                       (layer.eta_stdp * -1)).T

    # ================================================================================
    # 分支二：普通 STDP（特征层无监督学习）
    # ================================================================================
    # 适用场景：feature_layer（LF），spk_force=False
    # 理论来源：VPRTempo 论文 III-A "Weight updates and learning rules"，公式 (2)
    # 核心思想：
    #   如果前层神经元在 t-1 时刻发放（pre > 0），且后层神经元在 t 时刻发放（post > 0），
    #   则增强连接；增强的幅度取决于 (0.5 - post)，即若 post 已接近饱和（0.9），
    #   则增量变小，防止权重无限增长。
    #   若 pre>0 但 post=0，则权重不变（代码中通过 (pre>0)*(post>0) 掩码实现）。
    #   若 pre=0，则无论 post 如何，权重不变。
    # ================================================================================
    else:

        # ----------------------------------------
        # 逐行说明：获取权重矩阵维度
        # ----------------------------------------
        shape = layer.w.weight.data.shape

        # ----------------------------------------
        # 逐行说明：张量广播（Tile）
        # ----------------------------------------
        # 与 Spike Forcing 分支类似，将 pre/post 向量扩展为矩阵
        pre = torch.tile(torch.reshape(prespike, (shape[1], 1)), (1, shape[0]))
        post = torch.tile(spikes, (shape[1], 1))

        # ----------------------------------------
        # 逐行说明：兴奋性权重更新（公式 2）
        # ----------------------------------------
        # 更新量 = (0.5 - post) * (pre > 0) * (post > 0) * η_STDP
        # 物理意义拆解：
        #   (pre > 0)    : Heaviside 函数 Θ(x_i^m(t-1))，前层必须发放
        #   (post > 0)   : Heaviside 函数 Θ(x_j^n(t))，后层必须发放
        #   (0.5 - post) : 调制因子。若 post 较小（如 0.1），则增量 +0.4，大幅增强；
        #                  若 post 较大（如 0.9），则增量 -0.4，反而减弱。
        #                  这使得后层脉冲趋向于中等幅度（≈0.5），防止饱和。
        #   η_STDP       : 当前时刻的学习率
        #   havconnCombinedExc.T : 兴奋连接掩码，确保只有正权重被更新
        layer.w.weight.data += (((0.5 - post) * (pre > 0) * (post > 0) *
                                  layer.havconnCombinedExc.T) 
                                  * layer.eta_stdp).T

        # ----------------------------------------
        # 逐行说明：抑制性权重更新
        # ----------------------------------------
        # 与兴奋性权重更新公式相同，但作用在抑制连接掩码上
        # 且学习率乘以 -1，使得更新方向相反：
        #   当 (0.5 - post) > 0（post 不足）时，兴奋权重增加，抑制权重减小（趋近于 0）
        #   这是因为抑制输入对输出有负向贡献，要增加输出就应该减弱抑制。
        layer.w.weight.data += (((0.5 - post) * (pre > 0) * (post > 0) 
                                 * layer.havconnCombinedInh.T) 
                                 * (layer.eta_stdp * -1)).T

    # ================================================================================
    # 步骤三：权重符号保持（Sign Clamping）
    # ================================================================================
    # 理论依据：BLiTNet 论文强调兴奋性和抑制性权重必须保持符号不变。
    #          如果权重在训练中改变符号，则将其重置为极小的 ±1e-6。
    # 代码实现：
    #   - 兴奋权重（havconnCombinedExc 标记的位置）钳制在 [1e-6, 10]
    #   - 抑制权重（havconnCombinedInh 标记的位置）钳制在 [-10, -1e-6]
    # 为什么需要这一步？
    #   因为 STDP 更新可能导致小权重越过零点。保持符号确保 E/I 平衡的结构不被破坏。
    # ================================================================================
    # 兴奋权重保持为正：最小 1e-6（防止变零或变负），最大 10（防止过大）
    layer.w.weight.data[layer.havconnCombinedExc] = layer.w.weight.data[layer.havconnCombinedExc].clamp(min=1e-06, max=10)
    # 抑制权重保持为负：最小 -10（防止过大负值），最大 -1e-6（防止变零或变正）
    layer.w.weight.data[layer.havconnCombinedInh] = layer.w.weight.data[layer.havconnCombinedInh].clamp(min=-10, max=-1e-06)

    # ================================================================================
    # 步骤四：ITP (Intrinsic Threshold Plasticity) —— 公式 (5)
    # ================================================================================
    # 理论来源：VPRTempo 公式 (5)：
    #          Δθ_j^n(t) = η_ITP(t) · [Θ(x_j^n(t)) - f_j^n]
    # 物理意义：
    #   - 若神经元实际发放了（Θ(x) = 1）且目标发放率 f 较低，则阈值 θ 增加 → 下次更难发放
    #   - 若神经元未发放（Θ(x) = 0）且目标发放率 f > 0，则阈值 θ 减小 → 下次更易发放
    #   - 这样使得每个神经元的实际发放率趋向于其目标发放率 f
    # 适用条件：layer.have_rate=True（有非零目标发放率）且 η_ITP > 0
    # ================================================================================
    if layer.have_rate and layer.eta_ip > 0.0:

        # 更新阈值：θ_new = θ_old + η_ITP * (实际发放指示 - 目标发放率)
        # 注意：代码中使用 layer.x（存储的脉冲状态）而非 Θ(spikes)
        #       但在实际运行中，layer.x 通常在前向传播时被设置为 spikes 的值
        layer.thr.data += layer.eta_ip * (layer.x - layer.fire_rate)

        # 阈值不能为负：若 θ < 0 则重置为 0
        # 原因：阈值为负意味着任何正输入都会触发脉冲，失去调节意义
        layer.thr.data[layer.thr.data < 0] = 0

    # ================================================================================
    # 步骤五：Homeostasis —— 公式 (4)
    # ================================================================================
    # 理论来源：VPRTempo 公式 (4)：
    #          Ŵ⁻_ji(t) ← W⁻_ji(t) · [1 - η_STDP(t) · Θ(Σ_i x_i(t))]
    # 物理意义（BLiTNet 核心机制）：
    #   - 当净输入为正（Σ x_i · W_i > 0，即 noclp > 0）时，说明兴奋输入占主导，
    #     需要增强抑制（使抑制权重更负）来恢复 E/I 平衡。
    #   - 当净输入为负（noclp < 0）时，说明抑制过强，需要减弱抑制（使抑制权重趋近于 0）。
    # 代码实现解析：
    #   inhW: 克隆权重矩阵的转置，并将所有正值置零 → 仅保留抑制权重（负值）
    #   torch.mul(noclp, inhW): 逐元素相乘
    #       - 若 noclp > 0（净输入为正）且 inhW < 0（抑制权重），乘积 < 0
    #       - 加到原权重上：W += (负数) * η_STDP * 50 → 权重变得更负（抑制增强）
    #       - 若 noclp < 0（净输入为负）且 inhW < 0，乘积 > 0
    #       - 加到原权重上：W += (正数) * η_STDP * 50 → 权重变得不那么负（抑制减弱）
    #   系数 50 是一个经验放大因子，加速 Homeostasis 的收敛。
    # ================================================================================
    if torch.any(layer.w.weight.data).item() and layer.eta_stdp != 0:

        # 提取抑制权重部分（仅保留负值）
        inhW = layer.w.weight.data.T.clone()
        inhW[inhW > 0] = 0  # 屏蔽所有兴奋性（正）权重

        # Homeostasis 更新：根据未钳制的净输入 noclp 调整抑制权重
        # 转置 .T 是因为 noclp 的形状为 [1, 输出维度]，需要与 inhW 对齐
        layer.w.weight.data += (torch.mul(noclp, inhW) * layer.eta_stdp * 50).T

        # 注：下方被注释掉的代码是一个更激进的替代方案——直接将任何变为正的权重重置为极小负值
        # layer.w.weight.data[layer.w.weight.data > 0.0] = -1e-06

    return layer
