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
Part 1: 模块说明 (Imports & Overview)
================================================================================
本文件实现 IDEA1 的 ConvSNNLayer —— VPRTempo 的卷积脉冲前端（PLAN.md S2.1/S2.2）。

设计依据（详见 IDEA1-covstdp/PLAN.md）：
- S2.1 前向传播（单步幅度域）：conv2d → 减阈值 → clamp[0, 0.9]，无多步仿真
- S2.2 WTA 竞争（global / local / none 三变体）：winner 外置零
- ADR-1 维度链：local WTA 的块 winner 图即 4×4 max-pool（"竞争即池化"），
  none 模式用独立 max-pool 凑同一维度链，global 模式输出 [C] 向量
- ADR-2 接口：内部全程保持空间张量，flatten 只发生在 conv→feature_layer 边界

本 fork（s2122）只实现前向路径；STDP（S2.3）/ ITP（S2.4）在下一个 fork 实现。
为它们预留的接口（见 ConvLayerOutput）：
  pre_wta     —— WTA 之前的 clamp 响应图（ITP 必须用它统计发放率，不能用 WTA 后的）
  winner_mask —— winner 位置布尔掩码（STDP 构造响应图 M 用）
  pooled_flat —— 池化后 flatten 向量（送 feature_layer 的表征）
================================================================================
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ================================================================================
# Part 2: ConvLayerOutput —— 前向输出的统一约定
# ================================================================================
# 下游（本 fork 的 sanity、下一个 fork 的 STDP/ITP、S2.5 的框架接入）一律从
# 这个 dataclass 取数据，不要各自再算一遍。字段含义见上方模块注释。
# ================================================================================
@dataclass
class ConvLayerOutput:
    pre_wta: torch.Tensor      # [1, C, H', W'] WTA 前的 clamp 响应图（ITP 用）
    post_wta: torch.Tensor     # [1, C, H', W'] WTA 后的稀疏响应图（下游表征的稀疏版）
    winner_mask: torch.Tensor  # [1, C, H', W'] bool，winner 位置为 True（STDP 构造 M 用）
    pooled_flat: torch.Tensor  # [1, D] 池化并 flatten 后的向量（送 feature_layer）
    winner_values: torch.Tensor  # [N] winner 位置上的响应值（诊断/STDP 用）


# ================================================================================
# Part 3: ConvSNNLayer —— 卷积脉冲层
# ================================================================================
class ConvSNNLayer(nn.Module):
    def __init__(self,
                 input_dims=[28, 28],  # 输入图像尺寸 [H, W]（reshape 平向量用）
                 in_channels=1,        # 输入通道数（灰度 spike 图为 1）
                 out_channels=32,      # 卷积核个数 C
                 kernel_size=5,        # 卷积核边长 k
                 padding=0,            # B3 的 conv1 需要 padding=2（S2.7），其余为 0
                 thr_range=[0, 0.5],   # 初始阈值均匀分布范围（对齐 blitnet θ_max=0.5）
                 fire_rate=[0.2, 0.9], # 每通道目标发放率范围（S2.4 ITP 用，线性分配）
                 ip_rate=0.15,         # ITP 学习率（下一个 fork 使用，先存起来）
                 stdp_rate=0.005,      # STDP 学习率（下一个 fork 使用，先存起来）
                 p_exc=0.5,            # 兴奋通道比例（通道级 E/I，S2.1 卡片）
                 wta_mode='local',     # {'global','local','none'}
                 wta_block=4,          # local WTA 的块边长（即池化倍率）
                 device=None,
                 inference=False,      # True 只保留 w/thr（对齐 blitnet.py:101-111）
                 frozen=False,         # B1/B5 冻结前端（S2.6/S2.9）：跳过训练，且
                                       # calc_stdp_conv 入口直接返回——符号钳制与
                                       # 保范数归一化均不经过（负瓣保护，见
                                       # conv_learning.py 的 frozen 守卫）
                 free_sign=False,      # free-sign 消融（S2.10 B6b / S3.3-8）：True 时
                                       # 前向跳过 ON/OFF 取负（z=conv(x) 直通，因为抑制
                                       # 通道取负依赖权重≤0 假设，free-sign 下不再成立），
                                       # STDP 的 Step 6 统一 +η、Step 7 改为幅度安全钳
                                       # clamp(-10,10)（见 conv_learning.py）
                 # ---- S2.11 规则锦标赛 Round 1 开关（默认全关 = B2 行为不变）----
                 bcm_gate=False,       # R1 BCM 滑动阈值：(0.5−post) → (θ_M,c − post)，
                                       # θ_M,c 为每通道 EMA of post²（α=bcm_alpha，
                                       # 初值 0.25=0.5²，与现规则不动点兼容）
                 bcm_alpha=0.001,      # θ_M 滑动速率（须慢于权重学习 10–50×，防与 ITP 振荡）
                 rank_push=False,      # R2 名次反推：块内跨通道第 rank_k 名通道在其
                                       # winner 位置给予 −rank_delta 倍更新（KH 式去相关）
                 rank_delta=0.4,       # R2 反推强度 δ
                 rank_k=2,             # R2 反推名次 k
                 oja_decay=False,      # R3 Oja 衰减：关 Step 8 保范数归一化，更新里加
                                       # −post²·K_c 衰减项（幅度安全钳保留）
                 attractor=False,      # R4 弹性项：pre_term (pre−0.5) → (patch − 当前核)
                                       # （重构式吸引子；dK = corr(pre_img,M) − (ΣM_c)·K_c）
                 ):
        super(ConvSNNLayer, self).__init__()
        self.device = device
        self.inference = inference
        self.frozen = frozen
        self.free_sign = free_sign
        # S2.11 Round 1 规则开关（默认全关 = B2 主组合行为不变）
        self.bcm_gate = bcm_gate
        self.bcm_alpha = float(bcm_alpha)
        self.rank_push = rank_push
        self.rank_delta = float(rank_delta)
        self.rank_k = int(rank_k)
        self.oja_decay = oja_decay
        self.attractor = attractor
        self.wta_mode = wta_mode
        self.wta_block = wta_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.input_dims = list(input_dims)

        # ----------------------------------------
        # 派生维度（ADR-1：硬编码零容忍，全部现场计算）
        # ----------------------------------------
        H, W = self.input_dims
        self.H_out = (H + 2 * padding - kernel_size) // 1 + 1
        self.W_out = (W + 2 * padding - kernel_size) // 1 + 1
        b = wta_block
        # local WTA 裁右/下边缘（S2.2 卡片）；池化后尺寸
        self.H_pool = self.H_out // b
        self.W_pool = self.W_out // b
        if wta_mode in ('local', 'none'):
            self.flat_dim = out_channels * self.H_pool * self.W_pool
        elif wta_mode == 'global':
            self.flat_dim = out_channels
        else:
            raise ValueError(f"Unknown wta_mode: {wta_mode}")

        # ----------------------------------------
        # 权重与阈值（训练/推理共用）
        # ----------------------------------------
        self.w = nn.Conv2d(in_channels, out_channels, kernel_size,
                           padding=padding, bias=False)
        self.w.to(device)
        self.thr = nn.Parameter(torch.zeros([1, out_channels, 1, 1],
                                            device=device).uniform_(thr_range[0],
                                                                    thr_range[1]))
        # 通道级 E/I 掩码 [C]：按 p_exc 比例随机指定兴奋（ON）通道
        # 推理前向同样需要它做 ON/OFF 符号翻转，故在 inference 早退之前注册
        # （卷积核权值共享，按元素稀疏会破坏感受野结构，故按通道分 E/I —— S2.1 卡片）
        n_exc = int(round(out_channels * p_exc))
        havconnExc = torch.zeros(out_channels, dtype=torch.bool, device=device)
        havconnExc[torch.randperm(out_channels, device=device)[:n_exc]] = True
        self.register_buffer('havconnExc', havconnExc)
        if inference:
            # 推理模式只保留 w/thr（+ havconnExc 用于 ON/OFF 前向）
            return

        # ----------------------------------------
        # 训练模式：初始化 + 学习率/目标发放率
        # ----------------------------------------
        if np.isscalar(thr_range): thr_range = [thr_range, thr_range]
        if np.isscalar(fire_rate): fire_rate = [fire_rate, fire_rate]

        # 学习率张量（下一个 fork 的退火机制会更新它们）
        self.eta_ip = torch.tensor(ip_rate, device=device)
        self.eta_stdp = torch.tensor(stdp_rate, device=device)

        # 每通道目标发放率 [1,C,1,1]，[f_min,f_max] 线性分配（对齐 blitnet.py:164-167）
        # 注意：用普通属性而非 buffer —— 与 SNNLayer 一致（fire_rate 由配置确定性派生，
        # 不进 state_dict，避免训练态/推理态加载时键不匹配）
        self.fire_rate = torch.zeros([1, out_channels, 1, 1], device=device)
        fstep = (fire_rate[1] - fire_rate[0]) / out_channels
        for i in range(out_channels):
            self.fire_rate[:, i] = fire_rate[0] + fstep * (i + 1)

        # 卷积版权重初始化（addWeights 的卷积版，见下方函数）
        self.w.weight = self._add_conv_weights()

        # R1 BCM 滑动阈值 θ_M [1,C,1,1]（S2.11）：普通属性而非 buffer/parameter
        # —— 推理不需要（calc_stdp_conv 仅在训练路径调用），不进 state_dict，
        # 训练态/推理态加载键不受影响。初值 0.25 = 0.5²，与现规则不动点兼容。
        if self.bcm_gate:
            self.theta_m = torch.full([1, out_channels, 1, 1], 0.25, device=device)

    # ================================================================================
    # 函数：_add_conv_weights —— addWeights 的卷积版（S2.1 卡片第 3 条）
    # ================================================================================
    # 对齐 blitnet.py:250-327 的初始化思想，单位从"列"变"核"：
    #   1. 正态采样（mean=范围中点，std=跨度/6，3σ 原则）
    #   2. 按通道 E/I 角色做符号裁剪（兴奋核负值置零，抑制核正值置零）
    #   3. 每核 L1 归一化（防止不同核初始量级漂移）
    # ================================================================================
    def _add_conv_weights(self):
        C = self.out_channels
        k = self.kernel_size
        W = torch.empty(C, self.in_channels, k, k, device=self.device)

        if self.free_sign:
            # free-sign（S2.10 B6b / S3.3-8）：符号约束在初始化同步放开——所有通道
            # N(0, 1/6) 带符号采样，不按 E/I 角色裁剪。若沿用符号裁剪，抑制通道在
            # 直通前向（无 ON/OFF 取负）下对非负输入恒输出 ≤0，永久性死通道
            # （无 winner → 无 STDP 更新），消融格将失去判别力。
            W.normal_(mean=0.0, std=1.0 / 6.0)
        else:
            exc = self.havconnExc
            # 兴奋核：N(+0.5, 1/6)，负值裁剪为 0
            W[exc] = torch.empty(exc.sum(), self.in_channels, k, k,
                                 device=self.device).normal_(mean=0.5, std=1.0 / 6.0)
            W[exc] = W[exc].clamp(min=0.0)
            # 抑制核：N(-0.5, 1/6)，正值裁剪为 0
            inh = ~exc
            if inh.any():
                W[inh] = torch.empty(inh.sum(), self.in_channels, k, k,
                                     device=self.device).normal_(mean=-0.5, std=1.0 / 6.0)
                W[inh] = W[inh].clamp(max=0.0)

        # 每核 L1 归一化（范数为 0 的核置 1 防除零）
        nrm = torch.linalg.norm(W.flatten(1), ord=1, dim=1).view(C, 1, 1, 1)
        nrm[nrm == 0.0] = 1.0
        return nn.Parameter(W / nrm)

    # ================================================================================
    # 函数：reshape_input —— 平向量 [1, H*W] → [1, C_in, H, W]
    # ================================================================================
    # DataLoader 送来的是 ProcessImage 的平向量（dataset.py:437-439），
    # reshape 发生在 conv 层入口（ADR-2）。
    # ================================================================================
    def reshape_input(self, x_flat):
        H, W = self.input_dims
        return x_flat.view(1, self.in_channels, H, W)

    # ================================================================================
    # 函数：forward —— 单步幅度域前向 + WTA + 池化（S2.1/S2.2）
    # ================================================================================
    # 输入 x: [1, C_in, H, W]（平向量请先过 reshape_input）
    # 返回 ConvLayerOutput（字段约定见 Part 2 注释）
    # ================================================================================
    def forward(self, x):
        z = self.w(x)                                        # [1, C, H', W']
        # ON/OFF 双通路（S2.1 设计修订）：兴奋（ON）通道响应正相关；
        # 抑制（OFF）通道响应负相关 —— 负号约束核若直接过 clamp(min=0) 会先天性死亡
        # （实测发放率精确 0），取负后成为 OFF 检测器（反相对比模式检测）。
        # free-sign（S2.10 B6b / S3.3-8）：跳过取负，z = conv(x) 直通——取负依赖
        # 抑制核 ≤0 的假设，free-sign 下权重符号放开，该假设不再成立。
        if not self.free_sign:
            exc = self.havconnExc.view(1, -1, 1, 1)
            z = torch.where(exc, z, -z)
        pre_wta = torch.clamp(z - self.thr, min=0.0, max=0.9)  # 对齐 clamp_spikes

        post_wta, winner_mask = self._apply_wta(pre_wta)
        pooled_flat = self._pool_flat(pre_wta, post_wta)
        winner_values = pre_wta[winner_mask] if winner_mask.any() else \
            torch.zeros(0, device=x.device)

        return ConvLayerOutput(pre_wta=pre_wta,
                               post_wta=post_wta,
                               winner_mask=winner_mask,
                               pooled_flat=pooled_flat,
                               winner_values=winner_values)

    # ================================================================================
    # 函数：_apply_wta —— WTA 竞争三变体（S2.2）
    # ================================================================================
    # 返回 (post_wta, winner_mask)。mask 作用于 clamp 后活动，保证下游看到的
    # 也是稀疏表征。H',W' 不被 wta_block 整除时裁右/下边缘（边缘位置 mask=False）。
    # ================================================================================
    def _apply_wta(self, pre_wta):
        mode = self.wta_mode
        if mode == 'none':
            # 无竞争：不置零。winner_mask 全 True 表示"所有位置参与"
            # （STDP 的 none 模式用稠密 M，见 PLAN S2.2 none 条）
            return pre_wta, torch.ones_like(pre_wta, dtype=torch.bool)

        mask = torch.zeros_like(pre_wta, dtype=torch.bool)

        if mode == 'global':
            # 每通道全图 argmax，仅 1 个 winner
            C, H, W = pre_wta.shape[1], pre_wta.shape[2], pre_wta.shape[3]
            flat_idx = pre_wta.view(1, C, -1).argmax(dim=-1)     # [1, C]
            ys, xs = flat_idx // W, flat_idx % W
            mask[0, torch.arange(C, device=pre_wta.device), ys[0], xs[0]] = True

        elif mode == 'local':
            # 不重叠 b×b 块，每块一个 winner；裁右/下边缘
            b = self.wta_block
            C = pre_wta.shape[1]
            Hc, Wc = self.H_pool * b, self.W_pool * b
            x = pre_wta[:, :, :Hc, :Wc]
            # [1,C,Hb,b,Wb,b] → [1,C,Hb,Wb,b*b]，块内 argmax
            blocks = x.view(1, C, self.H_pool, b, self.W_pool, b) \
                      .permute(0, 1, 2, 4, 3, 5).reshape(1, C, self.H_pool, self.W_pool, b * b)
            idx = blocks.argmax(dim=-1, keepdim=True)            # [1,C,Hb,Wb,1]
            blk_mask = torch.zeros_like(blocks).scatter_(-1, idx, True)
            blk_mask = blk_mask.view(1, C, self.H_pool, self.W_pool, b, b) \
                               .permute(0, 1, 2, 4, 3, 5).reshape(1, C, Hc, Wc)
            mask[:, :, :Hc, :Wc] = blk_mask

        return pre_wta * mask, mask

    # ================================================================================
    # 函数：_pool_flat —— 池化并 flatten（ADR-1 的 WTA-维度耦合表）
    # ================================================================================
    # local ：post_wta 只有 winner 非零，max_pool(b) 即块 winner 图（"竞争即池化"）
    # none  ：独立 4×4 max-pool 凑同一维度链
    # global：全图 max → [C] 向量
    # ================================================================================
    def _pool_flat(self, pre_wta, post_wta):
        b = self.wta_block
        if self.wta_mode == 'global':
            pooled = post_wta.amax(dim=(2, 3))                   # [1, C]
        elif self.wta_mode == 'local':
            Hc, Wc = self.H_pool * b, self.W_pool * b
            pooled = torch.nn.functional.max_pool2d(
                post_wta[:, :, :Hc, :Wc], kernel_size=b)         # 值 ≥0，max=winner 值
        else:  # none
            Hc, Wc = self.H_pool * b, self.W_pool * b
            pooled = torch.nn.functional.max_pool2d(
                pre_wta[:, :, :Hc, :Wc], kernel_size=b)
        return pooled.reshape(1, -1)


# ================================================================================
# Part 4: ConvFrontendModule —— 推理链适配器（S2.5 框架接入）
# ================================================================================
# VPRTempo.evaluate 的推理链是 nn.Sequential(feature_layer.w, output_layer.w)，
# 输入是平向量 [1, H*W]。本适配器把 conv 前端包装成 nn.Module：
#   forward(x_flat) → reshape → ConvSNNLayer → pooled_flat
# 使 nn.Sequential(ConvFrontendModule(conv), feature_layer.w, output_layer.w) 直接成立，
# 推理侧改动最小（ADR-2 的拍板：显式分发，不伪装进 SNNLayer 接口）。
# ================================================================================
class ConvFrontendModule(nn.Module):
    def __init__(self, conv_layer):
        super().__init__()
        self.conv_layer = conv_layer

    def forward(self, x_flat):
        out = self.conv_layer(self.conv_layer.reshape_input(x_flat))
        return out.pooled_flat
