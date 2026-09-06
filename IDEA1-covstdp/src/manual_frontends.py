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
本文件实现 IDEA1 的"手工特征动物园"三个新冻结前端（S33 下午批，PLAN.md 手工
先验分解对照）：

  dog      —— DoG 中心-外周核组（无方向性：径向对称）。对照问题：Gabor 的优势
              来自"带通"还是"方向选择性"？DoG 有频率选择（带通）无方向。
  loggabor —— 对数频率域 Gabor（无直流问题：频域径向包络在 f=0 处精确为 0）。
              与 Gabor 组参数覆盖对齐（4 方向 × 2 频率 × 2 相位 × 2 带宽），
              对照问题：Gabor 的直流残差/带宽形状是否关键。
  dct      —— DCT-II 基组（有频率无方向的关键对照）。注意：5×5 DCT-II 只有
              25 个基函数，去直流后 24 < 32，无法凑满 C=32 —— 采用 8×8 DCT-II
              频率字典在 5×5 窗口上求值（超完备采样余弦，u,v ∈ {0..7} 在该
              窗口上无混叠重复），按频率序 (u+v, u) 取去直流后的前 32 个。

共同约定（与 gabor_frontend.py 完全一致）：
- k=5，C=32，全程确定性（无随机数，训练/推理两侧逐元素一致）；
- 每核减均值（去直流）后按 L1 范数归一化到 1（对齐 B2/B5 的每核 L1=1 约定）；
- 载入机制复用 frozen + itp_on_frozen（负瓣保护：frozen 守卫在
  calc_stdp_conv 入口旁路符号钳制与保范数归一化；ITP 只调阈值不触碰手工核）。
================================================================================
"""
import math

import torch


def _grid(kernel_size):
    """以核中心为原点的坐标网格（y 行向、x 列向，与 conv 权重 [..., ky, kx] 一致）"""
    half = kernel_size // 2
    ys, xs = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing='ij')
    return ys, xs


def _normalize(k):
    """去直流 + 每核 L1=1（与 gabor_kernel_bank 同一约定）"""
    k = k - k.mean()
    nrm = k.abs().sum()
    if nrm > 0:
        k = k / nrm
    return k


# ================================================================================
# 函数：dog_kernel_bank —— DoG 中心-外周核组（径向对称，无方向性）
# ================================================================================
# 参数网格（确定性、大致均匀采样尺度空间，非手工调优）：
#   σ_c ∈ {0.6, 0.8, 1.0, 1.2} 像素           —— 中心尺度 4 档
#   σ_s/σ_c ∈ {1.5, 2.0, 2.5, 3.0}            —— 外周/中心比 4 档
#   sign ∈ {+1, −1}                           —— on-center / off-center 两相位
#   共 4×4×2 = 32 核（σ_c 最外层，sign 最内层）。
# DoG = G(σ_c) − G(σ_s)，两高斯各自离散体积归一化（sum=1）——差值直流精确为 0，
# 径向对称由构造保证（仅依赖 x²+y²）。
# ================================================================================
def dog_kernel_bank(kernel_size=5, device=None):
    """返回 [32, 1, k, k] 的 DoG 核张量（径向对称、零直流、每核 L1=1）"""
    ys, xs = _grid(kernel_size)
    r2 = xs ** 2 + ys ** 2

    def gauss(sigma):
        g = torch.exp(-r2 / (2 * sigma ** 2))
        return g / g.sum()                      # 体积归一化 → DoG 直流精确为 0

    sigmas_c = [0.6, 0.8, 1.0, 1.2]
    ratios = [1.5, 2.0, 2.5, 3.0]

    kernels = []
    for sc in sigmas_c:
        gc = gauss(sc)
        for r in ratios:
            d = gc - gauss(sc * r)              # 中心 − 外周
            for s in (1.0, -1.0):
                kernels.append(_normalize(s * d))

    bank = torch.stack(kernels).unsqueeze(1)    # [32, 1, k, k]
    return bank.to(device)


# ================================================================================
# 函数：loggabor_kernel_bank —— 对数频率域 Gabor 核组（无直流）
# ================================================================================
# 在 5×5 DFT 频率网格上构造传递函数 H(u,v) = 径向 log-Gabor × 方向高斯，
# 空间核 = Re[IFFT(H·e^{iψ})]（ψ=0 偶对称 / ψ=π/2 奇对称，标准求法）。
# log-Gabor 径向包络 exp(−(log(f/f0))²/(2β²)) 在 f=0 处精确为 0 —— 无直流问题
# （普通 Gabor 的高斯包络在 f=0 处非零，需手工减均值）。
#
# 参数与 Gabor 组对齐（gabor_frontend.py：4 方向 × 2 频率 × 2 相位 × 2 尺度）：
#   orientations: {0, π/4, π/2, 3π/4}
#   f0          : {1/3, 1/2} 周期/像素  —— 与 Gabor λ∈{3,2} 像素一一对应
#   phases      : {0, π/2}              —— 偶/奇对称对
#   beta        : {0.5, 0.75}           —— log 带宽两档（对应 Gabor 紧/松包络档）
#   sigma_theta : 0.5 rad（固定）       —— 方向带宽，≈±30° 半宽
# 循环顺序镜像 gabor_kernel_bank：方向 × 频率 × 相位 × 带宽（带宽最内层）。
# ================================================================================
def loggabor_kernel_bank(kernel_size=5, device=None):
    """返回 [32, 1, k, k] 的 Log-Gabor 核张量（零直流、每核 L1=1）"""
    k = kernel_size
    f1 = torch.fft.fftfreq(k, dtype=torch.float32)      # 周期/像素
    fy, fx = torch.meshgrid(f1, f1, indexing='ij')
    f = torch.sqrt(fx ** 2 + fy ** 2)                   # 径向频率
    ang = torch.atan2(fy, fx)                           # 方向角（f=0 处为 0，不用）
    nz = f > 0

    orientations = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
    f0s = [1.0 / 3.0, 1.0 / 2.0]
    phases = [0.0, math.pi / 2]
    betas = [0.5, 0.75]
    sigma_theta = 0.5

    kernels = []
    for th0 in orientations:
        # 方向高斯（包裹角距离；只取 θ0 一支不含对跖支 → IFFT 为复数，
        # 实部/虚部即偶/奇正交对）
        dth = torch.atan2(torch.sin(ang - th0), torch.cos(ang - th0))
        angular = torch.exp(-dth ** 2 / (2 * sigma_theta ** 2))
        for f0 in f0s:
            for psi in phases:
                phase = torch.exp(torch.complex(
                    torch.zeros((), dtype=torch.float32),
                    torch.tensor(psi, dtype=torch.float32)))
                for beta in betas:
                    radial = torch.zeros_like(f)
                    radial[nz] = torch.exp(
                        -(torch.log(f[nz] / f0)) ** 2 / (2 * beta ** 2))
                    H = radial * angular                  # f=0 处精确为 0 → 无直流
                    sp = torch.fft.ifft2(H.to(torch.complex64)) * phase
                    kernels.append(_normalize(sp.real))

    bank = torch.stack(kernels).unsqueeze(1)            # [32, 1, k, k]
    return bank.to(device)


# ================================================================================
# 函数：dct_kernel_bank —— DCT-II 基组（有频率无方向的关键对照）
# ================================================================================
# 注意（坑）：5×5 DCT-II 只有 25 个基函数，去直流后 24 < 32，无法凑满 C=32。
# 采用 8×8 DCT-II 频率字典在 5×5 窗口上求值（超完备采样余弦）：
#   B(u,v)[y,x] = cos(π(2y+1)v/16) · cos(π(2x+1)u/16)，  u,v ∈ {0..7}, x,y ∈ {0..4}
# 在该窗口上 u 与 16−u 呈符号翻转混叠（16−u ∈ {9..15} 在取值范围外），
# {0..7} 内无重复。按频率序 (u+v, u) 排列，去掉直流 (0,0)，取前 32 个。
# 低 (u+v) 的基是各向同性的"频率片"——有频率选择性、无方向选择性（方向性
# 只来自 u/v 比例的偶然组合，不按方向系统覆盖），作为 Gabor 方向先验的对照。
# ================================================================================
def dct_kernel_bank(kernel_size=5, device=None, basis_n=8):
    """返回 [32, 1, k, k] 的 DCT-II 核张量（去直流、每核 L1=1）"""
    k = kernel_size
    n = basis_n
    xy = torch.arange(k, dtype=torch.float32)
    # 1D 基：basis1d[f, x] = cos(π(2x+1)f / (2n))，f ∈ {0..n-1}
    basis1d = torch.cos(math.pi * (2 * xy.view(1, -1) + 1)
                        * torch.arange(n, dtype=torch.float32).view(-1, 1) / (2 * n))

    order = sorted(((u + v, u, v) for u in range(n) for v in range(n)))
    kernels = []
    for s, u, v in order:
        if u == 0 and v == 0:
            continue                                    # 去直流
        g = torch.outer(basis1d[v], basis1d[u])         # [k, k]，行向 y↔v
        kernels.append(_normalize(g))
        if len(kernels) == 32:
            break
    assert len(kernels) == 32, f"DCT 字典去直流后不足 32 个（{len(kernels)}）"

    bank = torch.stack(kernels).unsqueeze(1)            # [32, 1, k, k]
    return bank.to(device)


# ================================================================================
# 函数：load_manual_weights —— 把手工核组载入 ConvSNNLayer 并冻结（动物园通用）
# ================================================================================
# 与 load_gabor_weights（gabor_frontend.py）同一机制：frozen=True（STDP 的符号
# 钳制与保范数归一化完全被 calc_stdp_conv 入口的 frozen 守卫旁路，负瓣保护）
# + itp_on_frozen=True（权重冻结但保留 ITP 阈值自适应——带符号零均值核的响应
# 量级与非负随机核系统性不同，无 ITP 时大量通道死亡，见 s29 诊断）。
# ================================================================================
BANKS = {
    'dog': dog_kernel_bank,
    'loggabor': loggabor_kernel_bank,
    'dct': dct_kernel_bank,
}


def load_manual_weights(layer, kind):
    """载入 kind ∈ {dog, loggabor, dct} 核组到 layer.w.weight 并冻结，返回 layer"""
    assert kind in BANKS, f"未知手工前端: {kind}（可选 {sorted(BANKS)}）"
    bank = BANKS[kind](kernel_size=layer.kernel_size,
                       device=layer.w.weight.device)
    assert bank.shape == layer.w.weight.shape, (
        f"{kind} 组形状 {tuple(bank.shape)} 与层权重 "
        f"{tuple(layer.w.weight.shape)} 不匹配（要求 C=32）")
    with torch.no_grad():
        layer.w.weight.copy_(bank)
    layer.frozen = True
    layer.itp_on_frozen = True
    return layer
