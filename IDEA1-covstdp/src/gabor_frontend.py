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
本文件实现 IDEA1 的 B5 手工 Gabor 滤波器组前端（PLAN.md S2.9）。

设计要点（S2.9 卡片）：
- 滤波器组：4 方向 {0°,45°,90°,135°} × 2 频率 × 2 相位 × 2 尺度 = 32 通道，
  k=5 对齐 B2 主组合。参数按方向-频率空间大致均匀采样（见下方取值注释），
  不做"看起来好"式的手工调优（那是过拟合先验）。
- 每核先减均值（去直流——Gabor 标准做法；输入非负，残直流会让核退化为
  亮度计），再按每核 L1 范数归一化到 1（对齐 ConvSNNLayer._add_conv_weights
  的每核 L1=1 约定，使 B5 与 B1/B2 的响应量级约定一致）。
- 负瓣保护（S2.9 卡片第 3 条，易踩的坑）：Gabor 核自带负瓣（表达力来源），
  而 STDP 更新路径有符号钳制（兴奋核 clamp(min=0)）与保范数归一化。
  载入 Gabor 后层置 frozen=True —— frozen 在 calc_stdp_conv 入口直接返回，
  clamp/renorm 路径完全不经过（见 conv_learning.py 的 frozen 守卫）；
  训练分发侧 VPRTempoTrain.py 同样跳过 frozen 层的训练循环。
- 其余通路（ON/OFF 双通路、WTA、池化、下游 feature/output 层训练）与 B2
  完全相同——唯一变量是核的来源（S2.9 卡片第 2 条）。OFF 通路取负等效于
  相位 +π，配合自带的 {0, π/2} 相位，有效相位覆盖 {0, π/2, π, 3π/2}。

核公式（标准 2D Gabor）：
    x' =  x·cosθ + y·sinθ,  y' = −x·sinθ + y·cosθ
    g(x,y) = exp(−(x'² + γ²y'²) / (2σ²)) · cos(2π·x'/λ + ψ)
================================================================================
"""
import math

import torch


# ================================================================================
# 函数：gabor_kernel_bank —— 生成 [32, 1, k, k] Gabor 滤波器组
# ================================================================================
# 参数取值依据（大致均匀采样方向-频率空间，非手工调优）：
#   orientations: {0, π/4, π/2, 3π/4}     —— [0, π) 均匀 4 等分
#   wavelengths : {2.0, 3.0} 像素          —— k=5 下保持 ≥1.5 个完整周期且不欠采样
#                                            的仅有的两个整数值（λ<2 欠采样，λ>5
#                                            不足一个周期退化为梯度）
#   phases      : {0, π/2}                 —— 偶/奇对称对（标准取法）
#   sigmas      : {1.0, 2.0} 像素          —— 紧/松两档包络
#   gamma       : 0.5（固定）              —— 空间纵横比，文献常用中值
# 全程确定性，无随机数（保证推理侧重建与训练侧逐元素一致）。
# ================================================================================
def gabor_kernel_bank(kernel_size=5, device=None):
    """返回 [32, 1, k, k] 的 Gabor 核张量（去直流 + 每核 L1=1）"""
    k = kernel_size
    half = k // 2
    # 坐标网格：y 行向、x 列向（与 conv 权重 [..., ky, kx] 布局一致）
    ys, xs = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing='ij')

    orientations = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    wavelengths = [2.0, 3.0]
    phases = [0.0, math.pi / 2]
    sigmas = [1.0, 2.0]
    gamma = 0.5

    kernels = []
    for theta in orientations:
        ct, st = math.cos(theta), math.sin(theta)
        xr = xs * ct + ys * st          # 旋转后沿条纹方向的坐标
        yr = -xs * st + ys * ct
        for lam in wavelengths:
            for psi in phases:
                carrier = torch.cos(2 * math.pi * xr / lam + psi)
                for sigma in sigmas:
                    envelope = torch.exp(-(xr ** 2 + gamma ** 2 * yr ** 2)
                                         / (2 * sigma ** 2))
                    g = envelope * carrier
                    g = g - g.mean()                     # 去直流
                    nrm = g.abs().sum()                  # 每核 L1=1（对齐 B2 约定）
                    if nrm > 0:
                        g = g / nrm
                    kernels.append(g)

    bank = torch.stack(kernels).unsqueeze(1)             # [32, 1, k, k]
    return bank.to(device)


# ================================================================================
# 函数：load_gabor_weights —— 把 Gabor 组载入 ConvSNNLayer 并冻结（B5）
# ================================================================================
# frozen 的语义（S2.9 卡片第 3 条）：不只是"跳过训练"，而是连 STDP 更新后的
# 符号钳制与保范数归一化都不能经过——守卫实现在 conv_learning.py 的
# calc_stdp_conv 入口（frozen 直接 return）。
# ================================================================================
def load_gabor_weights(layer):
    """载入 Gabor 组到 layer.w.weight 并置 frozen=True，返回 layer"""
    bank = gabor_kernel_bank(kernel_size=layer.kernel_size,
                             device=layer.w.weight.device)
    assert bank.shape == layer.w.weight.shape, (
        f"Gabor 组形状 {tuple(bank.shape)} 与层权重 "
        f"{tuple(layer.w.weight.shape)} 不匹配（B5 要求 C=32）")
    with torch.no_grad():
        layer.w.weight.copy_(bank)
    layer.frozen = True
    # B5 专用（S2.9 修订）：权重冻结但保留 ITP 阈值自适应——Gabor 零均值带符号核的
    # 响应量级与非负随机核系统性不同（直流承载 vs 纯边缘响应），初始阈值 U[0,0.5] 下
    # 实测 25/32 通道死亡（pre-WTA 发放率均值 1.6%）。ITP 是层本身的工作点自适应
    # 机制（B2 主组合 ITP=on），只调阈值不触碰手工核，"学习 vs 手工"的对比变量
    # 仍是核的来源。纯冻结（无 ITP）数字存档于
    # results/s29_b5_frozen_noitp_diagnostic/。
    layer.itp_on_frozen = True
    return layer


# ================================================================================
# 函数：load_gabor_weights_decomposed —— ON/OFF 分解载入（B6a，S2.10）
# ================================================================================
# bank 排布为 θ(4) × λ(2) × ψ(2) × σ(2)（σ 最内层），σ=1.0 的 16 个模式在
# 偶数索引 0,2,...,30 上——保持 4 方向 × 2 频率 × 2 相位全覆盖（不取前 16 个，
# 前 16 个只含 θ∈{0,45°} 两个方向）。
#
# 分解装载（OFF 通道符号链见 conv_snn_layer.py forward：OFF 输出 = −conv(x, w_off)）：
#   ON  通道 i   ← G⁺ = clamp(G, min=0)          （满足兴奋钳制 [0,10]）
#   OFF 通道 i+16 ← w_off = −G⁻，G⁻=clamp(−G,min=0)≥0 （w_off ≤ 0，满足抑制钳制 [-10,0]）
# 联合响应 ON_out − OFF_out = conv(x,G⁺) − conv(x,G⁻) ≡ conv(x,G)——整流形式精确
# 编码带符号 Gabor 响应，符号钳制全程无需放松。
#
# 通道配对改写 havconnExc 为确定性排布（前 16 ON / 后 16 OFF，通道 i 与 i+16 配对），
# 不用随机 randperm——B6a 的配对语义必须确定。分解后不做任何额外归一化
# （否则 ON−OFF ≠ conv(x,G)，一致性断言会破）。
# frozen=False：STDP + ITP 正常跑（B6 问的就是"局部可塑性在手工先验上有无增量"）。
# ================================================================================
def load_gabor_weights_decomposed(layer):
    """16 个 σ=1.0 Gabor 模式分解为 32 通道（16 ON + 16 OFF 配对），frozen=False"""
    bank = gabor_kernel_bank(kernel_size=layer.kernel_size,
                             device=layer.w.weight.device)
    modes = bank[0::2]                                    # σ=1.0 的 16 个模式
    n = modes.shape[0]
    assert n * 2 == layer.w.weight.shape[0], (
        f"B6a 要求 C = 2×{n}（当前 {layer.w.weight.shape[0]}）")
    g_pos = modes.clamp(min=0.0)                          # G⁺
    g_neg = (-modes).clamp(min=0.0)                       # G⁻ ≥ 0
    with torch.no_grad():
        layer.w.weight[:n] = g_pos                        # ON 通道装 G⁺
        layer.w.weight[n:] = -g_neg                       # OFF 通道装 −G⁻ ≤ 0
        # 确定性通道配对：前 16 ON、后 16 OFF（i 与 i+16 同属一个 Gabor 模式）
        havconn = torch.zeros_like(layer.havconnExc)
        havconn[:n] = True
        layer.havconnExc.copy_(havconn)
    layer.frozen = False
    return layer


# ================================================================================
# 函数：load_gabor_weights_signed —— signed Gabor 原样载入（B6b，S2.10）
# ================================================================================
# 与 B5 同一份带符号核，但 frozen=False + free_sign=True：放开符号钳制训练
# （Step 7 变为幅度安全钳 clamp(-10,10)，保范数归一化保留），前向跳过 ON/OFF
# 取负（z = conv(x) 直通）。与 S3.3-8 的 free-sign 消融（随机初始化 + 放开钳制）
# 互为对照：区分"条纹来自初始化还是来自放开约束后的学习"。
# ================================================================================
def load_gabor_weights_signed(layer):
    """signed Gabor 组原样载入（负瓣保留），frozen=False，free_sign 由 dispatch 设置"""
    bank = gabor_kernel_bank(kernel_size=layer.kernel_size,
                             device=layer.w.weight.device)
    assert bank.shape == layer.w.weight.shape, (
        f"Gabor 组形状 {tuple(bank.shape)} 与层权重 "
        f"{tuple(layer.w.weight.shape)} 不匹配（B6b 要求 C=32）")
    with torch.no_grad():
        layer.w.weight.copy_(bank)
    layer.frozen = False
    return layer
