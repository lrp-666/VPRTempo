# MIT License
#
# Copyright (c) 2023 Adam Hines, Peter G Stratton, Michael Milford, Tobias Fischer
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
【模块级注释】dataset.py —— 图像数据集加载与脉冲编码预处理流水线
================================================================================
位置：vprtempo/src/dataset.py
归属：VPRTempo 脉冲神经网络视觉场景识别系统

【核心职责】
  本模块负责将磁盘上的原始 RGB 图像转换为可供 SNN（脉冲神经网络）直接输入的
  脉冲幅度张量。整个流程包含数据加载、灰度化、Gamma 校正、尺寸缩放、
  局部块归一化（Patch Normalization）以及时序脉冲编码。

【理论对应（VPRTempo 论文）】
  - 公式 (8) Gamma 校正      → ProcessImage.__call__() Step 2
  - IV-A 数据预处理          → ProcessImage 完整流程
  - III-A 时序编码           → SetImageAsSpikes（幅度编码单脉冲）
  - Patch Normalization     → PatchNormalisePad（局部 Z-score）

【包含类一览】
  1. GetPatches2D      — 2D 滑动窗口提取器（底层工具类）
  2. PatchNormalisePad — 局部块归一化（核心预处理）
  3. SetImageAsSpikes  — 图像 → 脉冲幅度编码（时序编码）
  4. ProcessImage      — 完整图像预处理流水线（组合上述步骤）
  5. CustomImageDataset— PyTorch Dataset，负责从 CSV 加载图像并应用流水线

【调用关系】
  main.py / VPRTempoTrain.py
      └─> CustomImageDataset(transform=ProcessImage(...))
              └─> __getitem__() 调用 read_image() 读取原始图
              └─> transform(img) 进入 ProcessImage.__call__()
                    ├─> Step 1: RGB → Grayscale
                    ├─> Step 2: Gamma 校正
                    ├─> Step 3: Resize
                    ├─> Step 4: PatchNormalisePad
                    ├─> Step 5: uint8 量化
                    └─> Step 6: SetImageAsSpikes
                          └─> 输出 spikes [H*W]，输入 SNNLayer
================================================================================
"""

import os           # 【行级】操作系统接口：拼接文件路径、判断文件存在性
import math         # 【行级】数学库：提供 log()、pow()，用于 Gamma 校正的自适应 gamma 计算
import torch        # 【行级】PyTorch 核心库：张量运算、nn.functional、quantization

import pandas as pd # 【行级】Pandas：读取 CSV 标注文件，处理图像路径与标签表格
import numpy as np  # 【行级】NumPy：配合 torch 处理数值计算，如 errstate 控制除零警告
import torch.nn.functional as F  # 【行级】PyTorch 函数式 API：提供 interpolate()、pad()

from torchvision.io import read_image      # 【行级】读取图像为 torch.Tensor [C, H, W]
from torch.utils.data import Dataset       # 【行级】PyTorch 数据集基类，需实现 __len__ 和 __getitem__


# ================================================================================
# 【模块级注释】类：GetPatches2D —— 2D 滑动窗口提取器
# ================================================================================
# 【功能定位】
#   本类是 PatchNormalisePad 的底层工具，负责使用 PyTorch 的 unfold 操作
#   从填充后的图像张量中提取所有局部滑动窗口（patches）。
#
# 【为什么需要它】
#   Patch Normalization 要求对每个像素的邻域计算均值和标准差。
#   如果逐像素用循环提取邻域，在 Python 中效率极低。
#   unfold 是 PyTorch 的向量化操作，可在 GPU 上高效完成滑动窗口提取。
#
# 【核心机制】
#   unfold(dim, size, step) 沿指定维度以步长 step 滑动，提取长度为 size 的窗口。
#   先在 H 维度 unfold，再在 W 维度 unfold，最后 reshape 为 [patch_area, num_patches]。
# ================================================================================
class GetPatches2D: 
    def __init__(self, patch_size, image_pad):
        # ---- 行级：保存 patch 的尺寸，例如 (15, 15) ----
        self.patch_size = patch_size
        # ---- 行级：保存经过 F.pad 填充后的图像张量，边缘已用 NaN 填充 ----
        self.image_pad = image_pad

    def __call__(self, img):
        # ================================================================================
        # 【函数级注释】__call__ —— 提取所有滑动窗口 patches
        # ================================================================================
        # 输入：img —— 原始图像（仅用于推导行列数，实际数据来自 self.image_pad）
        # 输出：patches —— 形状为 [patch_size[0]*patch_size[1], num_patches] 的张量
        #       每一列是一个 patch 展平后的像素值向量
        # ================================================================================

        # ---- 行级：第 1 次 unfold —— 在 H 维度（dim=0）上以步长 1 滑动，窗口高 patch_size[0] ----
        # ---- 行级：第 2 次 unfold —— 在 W 维度（dim=1）上以步长 1 滑动，窗口宽 patch_size[1] ----
        # 结果 unfolded 的形状近似：[H, W, patch_h, patch_w]（实际需考虑 stride=1 的滑动窗口数）
        unfolded = self.image_pad.unfold(0, self.patch_size[0], 1).unfold(1, self.patch_size[1], 1)

        # ---- 行级：permute 重排维度，将 patch_h, patch_w 放到前面，H, W 放到后面 ----
        # ---- 行级：contiguous() 确保内存连续，view 将后两维展平为 num_patches ----
        # ---- 行级：最终形状 [patch_h * patch_w, (H-patch_h+1)*(W-patch_w+1)]，
        #           由于前面做了 pad，(H-patch_h+1) 恰好等于原图 H ----
        patches = unfolded.permute(2, 3, 0, 1).contiguous().view(self.patch_size[0]*self.patch_size[1], -1)
        return patches


# ================================================================================
# 【模块级注释】类：PatchNormalisePad —— 局部块归一化（Patch Normalization）
# ================================================================================
# 【功能定位】
#   对图像每个像素的局部邻域（patch）做 Z-score 归一化：
#       im_norm = (pixel - μ_patch) / σ_patch
#
# 【与全局归一化的区别】
#   全局归一化用整幅图像的均值/方差，无法适应局部光照突变（如阴影、云层）。
#   Patch Normalization 以每个像素为中心取邻域，只对该邻域做归一化，
#   因此对跨季节、跨天气的视觉场景识别（VPR）更鲁棒。
#
# 【理论来源】
#   VPRTempo 论文 IV-A "patch normalization"
# ================================================================================
class PatchNormalisePad:
    def __init__(self, patches):
        # ---- 行级：patch 边长，例如 15，表示取 15×15 的邻域 ----
        self.patches = patches

    def nanstd(self, input_tensor, dim=None, unbiased=True):
        # ================================================================================
        # 【函数级注释】nanstd —— 支持 NaN 的标准差计算
        # ================================================================================
        # 背景：PyTorch 标准库没有提供 nanstd（忽略 NaN 的标准差）。
        # 由于 PatchNormalisePad 用 NaN 填充图像边缘，边缘像素的 patch 含有 NaN，
        # 必须用 nanstd 才能正确计算局部标准差。
        #
        # 输入：input_tensor —— 可能含 NaN 的张量
        #       dim —— 沿哪个维度计算，None 则全局计算
        #       unbiased —— 是否做 Bessel 校正（除以 n-1 而非 n）
        # 输出：标准差张量
        # ================================================================================

        if dim is not None:
            # ---- 行级：统计非 NaN 元素个数，用于后续均值和方差计算 ----
            valid_count = torch.sum(~torch.isnan(input_tensor), dim=dim, dtype=torch.float)
            # ---- 行级：nansum 忽略 NaN 求和，除以有效个数得到均值 μ ----
            mean = torch.nansum(input_tensor, dim=dim) / valid_count
            # ---- 行级：unsqueeze 保持维度对齐，便于广播减法 ----
            diff = input_tensor - mean.unsqueeze(dim)
            # ---- 行级：计算方差 = Σ(diff²) / n（先算有偏方差）----
            variance = torch.nansum(diff * diff, dim=dim) / valid_count
            # ---- 行级：若 unbiased=True，乘以 n/(n-1) 做 Bessel 校正，得到样本方差 ----
            if unbiased:
                variance = variance * (valid_count / (valid_count - 1))
        else:
            # ---- 行级：dim=None 分支：对整个张量计算单个均值和标准差 ----
            valid_count = torch.sum(~torch.isnan(input_tensor), dtype=torch.float)
            mean = torch.nansum(input_tensor) / valid_count
            diff = input_tensor - mean
            variance = torch.nansum(diff * diff) / valid_count
            if unbiased:
                variance = variance * (valid_count / (valid_count - 1))
        # ---- 行级：返回标准差 = sqrt(方差) ----
        return torch.sqrt(variance)

    def __call__(self, img):
        # ================================================================================
        # 【函数级注释】__call__ —— Patch Normalization 主流程
        # ================================================================================
        # 输入：img [1, H, W] 或 [H, W] —— 单通道灰度图（通常来自 ProcessImage Step 1）
        # 输出：im_norm [H, W] —— 局部归一化后的图像，值域约 [-1, 1]
        #
        # 步骤：
        #   1. squeeze 去掉可能的单通道维度
        #   2. 计算 patch 半径（半边长）
        #   3. F.pad 用 NaN 填充边缘，避免边界 patch 引入虚假零值
        #   4. GetPatches2D 提取所有 patches
        #   5. 逐 patch 计算均值 μ 和标准差 σ
        #   6. (img - μ) / σ 得到 Z-score 归一化图像
        #   7. clip 到 [-1, 1]，NaN 置 0
        # ================================================================================

        # ---- 行级：去掉 batch 或通道维，确保 img 是 [H, W] ----
        img = torch.squeeze(img, 0)
        # ---- 行级：构造 patch 尺寸元组，例如 (15, 15) ----
        patch_size = (self.patches, self.patches)
        # ---- 行级：计算半边长，15→7，用于对称填充 ----
        patch_half_size = [int((p-1)/2) for p in patch_size]

        # ---- 行级：根据半边长构造 F.pad 的参数 (left, right, top, bottom) ----
        # 注意：F.pad 的顺序是 (W_left, W_right, H_top, H_bottom)
        if isinstance(patch_half_size, int):
            pad = (patch_half_size, patch_half_size, patch_half_size, patch_half_size)
        else:
            pad = (patch_half_size[1], patch_half_size[1], patch_half_size[0], patch_half_size[0])

        # ---- 行级：用 NaN（非数字）填充图像边缘。
        # 原因：边界像素的邻域会超出图像范围，用 NaN 标记这些越界位置，
        #       后续 nanmean/nanstd 会自动忽略它们，避免用 0 填充导致的偏差。----
        image_pad = F.pad(img, pad, mode='constant', value=float('nan'))

        # ---- 行级：记录原图尺寸，用于后续 reshape ----
        nrows = img.shape[0]
        ncols = img.shape[1]
        # ---- 行级：实例化滑动窗口提取器 ----
        patcher = GetPatches2D(patch_size, image_pad)
        # ---- 行级：提取所有 patches，形状 [225, H*W]（假设 15×15）----
        patches = patcher(img)

        # ---- 行级：沿 dim=0（patch 像素维度）计算每个 patch 的均值 μ ----
        mus = torch.nanmean(patches, dim=0)
        # ---- 行级：沿 dim=0 计算每个 patch 的标准差 σ ----
        stds = self.nanstd(patches, dim=0)

        # ---- 行级：Z-score 归一化：重塑 μ 和 σ 回 [H, W]，逐像素做 (x-μ)/σ ----
        # np.errstate 忽略除以 0 或无效值的警告（std=0 时会产生 NaN，下一步处理）----
        with np.errstate(divide='ignore', invalid='ignore'):
            im_norm = (img - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)

        # ---- 行级：后处理 1：std=0 的区域（如纯黑/纯白块）会产生 NaN，将其置 0 ----
        im_norm[torch.isnan(im_norm)] = 0.0
        # ---- 行级：后处理 2：截断到 [-1, 1]，防止极端值 ----
        im_norm[im_norm < -1.0] = -1.0
        im_norm[im_norm > 1.0] = 1.0

        return im_norm


# ================================================================================
# 【模块级注释】类：SetImageAsSpikes —— 将图像编码为脉冲幅度（时序编码）
# ================================================================================
# 【功能定位】
#   将经过预处理的 uint8 灰度图像转换为脉冲幅度张量，作为 SNN 的输入。
#
# 【理论来源：时序编码（Temporal Coding）】
#   传统 SNN 使用速率编码（rate coding）：像素越亮，发放脉冲越多。
#   VPRTempo 使用时序编码：每个输入神经元在单个时间步内只发放一个脉冲，
#   脉冲的"幅度"（amplitude）∈ [0, 1] 对应像素强度 / 255。
#   在 theta 振荡的框架下，这等价于"发放时刻"的编码：
#     - 幅度 1.0 → 最早发放（最强刺激）
#     - 幅度 0.1 → 最晚发放（最弱刺激）
#
# 【量化支持（QAT）】
#   训练时：FakeQuantize 模拟 INT8 量化的前向/反向，观察统计值范围。
#   推理时：执行真正的 INT8 量化（quantize_per_tensor），降低内存和计算量。
# ================================================================================
class SetImageAsSpikes:
    def __init__(self, intensity=255, test=True):
        # ---- 行级：最大像素值，uint8 图像为 255，用于归一化分母 ----
        self.intensity = intensity

        # ---- 行级：实例化 PyTorch 伪量化模块 FakeQuantize ----
        # observer=MovingAverageMinMaxObserver：滑动窗口记录输入的最小/最大值
        # quant_min=0, quant_max=255：模拟 uint8 的量化范围
        # dtype=torch.quint8：无符号 8 位整型
        # qscheme=torch.per_tensor_affine：每张量仿射量化（含 scale 和 zero_point）
        self.fake_quantize = torch.quantization.FakeQuantize(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False
        )

    def train(self):
        # ---- 行级：切换到训练模式，FakeQuantize 会记录统计信息并模拟量化误差 ----
        self.fake_quantize.train()

    def eval(self):
        # ---- 行级：切换到评估模式，后续 __call__ 会执行真正的 INT8 量化 ----
        self.fake_quantize.eval()

    def __call__(self, img_tensor):
        # ================================================================================
        # 【函数级注释】__call__ —— 图像 → 脉冲幅度编码
        # ================================================================================
        # 输入：img_tensor [1, H, W] —— uint8 格式的灰度图（来自 ProcessImage Step 5）
        # 输出：spikes [1, H*W] —— 每个像素对应一个输入神经元的脉冲幅度 [0, 1]
        #       若处于 eval 模式，输出为 torch.quint8 量化张量
        #
        # 步骤：
        #   1. view 展平空间维度 [1, H, W] → [1, 1, H*W]
        #   2. 除以 255 归一化到 [0, 1]
        #   3. squeeze 去掉中间维度 → [1, H*W]
        #   4. FakeQuantize（训练时观察，推理时真量化）
        # ================================================================================

        # ---- 行级：解包输入形状，N=1（batch），W=宽，H=高 ----
        N, W, H = img_tensor.shape
        # ---- 行级：view 展平 H 和 W，保留 N 和 1 个中间通道维，形状变为 [1, 1, H*W] ----
        reshaped_batch = img_tensor.view(N, 1, -1)

        # ---- 行级：除以 intensity（255）将像素值归一化到 [0, 1]，
        #           这就是脉冲的"幅度"，越亮越强 ----
        normalized_batch = reshaped_batch / self.intensity
        # ---- 行级：去掉中间那个单通道维，形状变为 [1, H*W] ----
        normalized_batch = torch.squeeze(normalized_batch, 0)

        # ---- 行级：通过伪量化器。训练模式下记录 min/max 并模拟量化；
        #           评估模式下准备真正的量化参数 ----
        spikes = self.fake_quantize(normalized_batch)

        # ---- 行级：若处于评估/推理模式（非 training），执行真正的 INT8 量化 ----
        # calculate_qparams() 根据观察到的 min/max 计算 scale 和 zero_point
        # quantize_per_tensor 将浮点张量转换为 torch.quint8 量化张量
        if not self.fake_quantize.training:
            scale, zero_point = self.fake_quantize.calculate_qparams()
            spikes = torch.quantize_per_tensor(spikes, float(scale), int(zero_point), dtype=torch.quint8)

        return spikes


# ================================================================================
# 【模块级注释】类：ProcessImage —— 完整图像预处理流水线
# ================================================================================
# 【功能定位】
#   将原始 RGB 图像通过一系列可微/张量运算转换为 SNN 输入脉冲。
#   对应 VPRTempo 论文 IV-A 的完整预处理链。
#
# 【处理流程概览】
#   RGB [3, H_orig, W_orig]
#     → Grayscale [1, H_orig, W_orig]
#     → Gamma 校正 [1, H_orig, W_orig]
#     → Resize [1, 56, 56]
#     → Patch Normalization [56, 56]（值域 [-1, 1]）
#     → uint8 量化 [1, 56, 56]
#     → Spike 编码 [56*56=3136]
#
# 【为什么用 Gamma 校正】
#   不同季节/天气下，同一场景的图像可能整体偏暗（阴天）或偏亮（晴天）。
#   自适应 Gamma 根据图像均值调整对比度，使不同条件下的图像具有相似的动态范围。
# ================================================================================
class ProcessImage:  #TODO VPRTempoTrain常用
    def __init__(self, dims, patches):
        # ---- 行级：目标缩放尺寸 [H, W]，默认 [56, 56]，决定 SNN 输入神经元数量 ----
        self.dims = dims
        # ---- 行级：Patch Normalization 的窗口大小，如 15，需为奇数以保证中心对称 ----
        self.patches = patches

    def __call__(self, img):
        # ================================================================================
        # 【函数级注释】__call__ —— 完整图像预处理流水线
        # ================================================================================
        # 输入：img [3, H_orig, W_orig] —— torchvision.io.read_image 读取的 RGB 图像，值域 [0, 255]
        # 输出：spikes [H*W] —— 可直接输入 SNNLayer 的脉冲幅度张量
        #
        # 步骤详解：
        # ------------------------------------------------------------------------
        # Step 1: RGB → Grayscale
        #   使用标准 ITU-R BT.601 权重：0.299R + 0.587G + 0.114B
        #   原因：VPR 任务对颜色不敏感，灰度图减少 2/3 输入维度，降低计算量。
        # ------------------------------------------------------------------------
        # Step 2: Gamma 校正（对应论文公式 8）
        #   公式：ρ_norm = ρ^γ
        #   其中 γ = ln(0.5 * 255) / ln(mean)
        #   物理意义：自适应 gamma。
        #     - 若图像偏暗（mean 小），gamma < 1，暗部被拉伸（变亮）。
        #     - 若图像偏亮（mean 大），gamma > 1，亮部被压缩（变暗）。
        #     - 若 mean = 127.5（适中），gamma = 1（无变化）。
        #   这使得不同季节/光照条件下的图像具有相似的动态范围。
        # ------------------------------------------------------------------------
        # Step 3: Resize
        #   使用双线性插值（bilinear）缩放到目标尺寸（默认 56×56）。
        #   align_corners=False 是 PyTorch 推荐的标准设置。
        # ------------------------------------------------------------------------
        # Step 4: Patch Normalization
        #   对每个像素的局部邻域做 Z-score，适应局部光照变化（阴影、云层）。
        # ------------------------------------------------------------------------
        # Step 5: 量化到 uint8
        #   公式：img = 255 * (1 + im_norm) / 2
        #   将 PatchNorm 输出的 [-1, 1] 线性映射到 [0, 255]，便于后续 spike 编码。
        # ------------------------------------------------------------------------
        # Step 6: Spike 编码（时序编码）
        #   调用 SetImageAsSpikes，将 uint8 像素转为 [0, 1] 幅度。
        # ================================================================================

        # ==================== Step 1: RGB → Grayscale ====================
        # ---- 行级：判断输入是否为 3 通道 RGB 图 ----
        if img.shape[0] == 3:
            # ---- 行级：按 ITU-R BT.601 标准加权求和，得到单通道灰度图 [H, W] ----
            img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        # ---- 行级：增加通道维，变为 [1, H, W]，统一后续处理 ----
        img = img.unsqueeze(0)

        # ==================== Step 2: Gamma 校正（对应 VPRTempo 公式 8）====================
        # ---- 行级：mid=0.5 是目标中值比例（对应 0.5*255=127.5）----
        mid = 0.5
        # ---- 行级：计算当前图像的像素均值（浮点），作为自适应 gamma 的依据 ----
        mean = torch.mean(img.float())
        # ---- 行级：计算自适应 gamma。
        #   推导：我们希望校正后图像的均值为 mid*255=127.5。
        #   设校正后均值为 (mean^gamma) = 127.5，则 gamma = ln(127.5)/ln(mean)。
        #   这里 mean 是 [0,255] 范围内的实际均值。----
        gamma = math.log(mid * 255) / math.log(mean)
        # ---- 行级：逐元素幂运算并截断到 [0, 255]，防止数值溢出 ----
        img = torch.pow(img, gamma).clip(0, 255)

        # ==================== Step 3: Resize ====================
        # ---- 行级：若 img 是 [1, H, W]，增加 batch 维变为 [1, 1, H, W]，
        #           因为 F.interpolate 需要 4D 输入 (N, C, H, W) ----
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        # ---- 行级：双线性插值缩放到目标尺寸 self.dims（如 [56, 56]）----
        img = F.interpolate(img, size=self.dims, mode='bilinear', align_corners=False)
        # ---- 行级：去掉 batch 维，恢复为 [1, H, W] ----
        img = img.squeeze(0)

        # ==================== Step 4: Patch Normalization ====================
        # ---- 行级：实例化局部归一化器 ----
        patch_normaliser = PatchNormalisePad(self.patches)
        # ---- 行级：执行归一化，输出值域约 [-1, 1] ----
        im_norm = patch_normaliser(img)

        # ==================== Step 5: 映射到 uint8 ====================
        # ---- 行级：将 [-1, 1] 线性映射到 [0, 255]：
        #           -1 → 0，0 → 127.5，1 → 255 ----
        img = (255.0 * (1 + im_norm) / 2.0).to(dtype=torch.uint8)
        # ---- 行级：增加通道维，变为 [1, H, W]，匹配 SetImageAsSpikes 输入要求 ----
        img = torch.unsqueeze(img, 0)

        # ==================== Step 6: Spike 编码（时序编码）====================
        # ---- 行级：实例化脉冲编码器 ----
        spike_maker = SetImageAsSpikes()
        # ---- 行级：将 uint8 图像编码为 [0,1] 幅度脉冲，形状 [1, H*W] ----
        img = spike_maker(img)
        # ---- 行级：去掉 batch 维，最终输出 [H*W]，直接送入 SNN 输入层 ----
        img = torch.squeeze(img, 0)

        return img


# ================================================================================
# 【模块级注释】类：CustomImageDataset —— 自定义 PyTorch 数据集
# ================================================================================
# 【功能定位】
#   从 CSV 标注文件加载图像路径和标签，配合 DataLoader 实现批量加载。
#   支持训练/测试两种模式，支持多目录合并、子采样、跳过、模块划分等高级功能。
#
# 【CSV 文件格式】
#   每行两列：[图像文件名, 全局标签]
#   例如：image_0001.jpg, 0
#         image_0002.jpg, 1
#
# 【关键参数说明】
#   - filter：子采样步长（对应论文"filter images every 8 seconds"），每隔 filter 帧取一帧
#   - skip：跳过前 skip 帧，去除起始段不稳定数据
#   - max_samples：限制最大样本数，用于快速测试
#   - img_range：[start, end] 索引范围，用于多模块（multi-module）划分
#
# 【多模块支持】
#   当数据库图像数 > max_module（默认 500）时，VPRTempo 会将数据库拆分为多个模块。
#   每个模块独立训练一个模型子网络。img_range 参数用于限定本 Dataset 只加载某个模块
#   对应的图像索引范围。
# ================================================================================
class CustomImageDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 base_dir, img_dirs, 
                 transform=None, 
                 target_transform=None,
                 filter=1, 
                 skip=0, 
                 max_samples=None, 
                 test=True, 
                 img_range=None):
        # ---- 行级：图像预处理流水线实例（通常是 ProcessImage）----
        self.transform = transform
        # ---- 行级：标签预处理函数（本项目通常不使用）----
        self.target_transform = target_transform
        # ---- 行级：子采样步长，例如 filter=8 表示每 8 帧取 1 帧 ----
        self.filter = filter
        # ---- 行级：模块内图像索引范围 [start, end]，用于多模块切分 ----
        self.img_range = img_range
        # ---- 行级：跳过前 skip 张图像 ----
        self.skip = skip

        # ================================================================================
        # 【函数级注释】__init__ —— 加载并处理 CSV 标注
        # ================================================================================
        # 背景：database_dirs 可能包含多个季节/天气目录（如 ["spring", "fall"]）。
        # 每个目录对应一个 CSV 文件，需要分别加载后合并。
        #
        # 处理顺序（对每个 CSV）：
        #   1. 读取 CSV 为 DataFrame
        #   2. 拼接完整文件路径（base_dir / img_dir / filename）
        #   3. 应用 img_range（模块划分）
        #   4. 应用 skip（跳过前段）
        #   5. 应用 filter（子采样）
        #   6. 应用 max_samples（截断）
        #   7. 合并所有 DataFrame
        # ================================================================================

        # ---- 行级：初始化图像标签列表 ----
        self.img_labels = []
        # ---- 行级：若传入的是单个字符串而非列表，包装为列表统一处理 ----
        if not isinstance(annotations_file, list):
            annotations_file = [annotations_file]
        # ---- 行级：遍历每个 CSV 标注文件（每个对应一个 img_dir）----
        for idx, annotation in enumerate(annotations_file):
            # ---- 行级：用 Pandas 读取 CSV，默认按逗号分隔 ----
            img_labels = pd.read_csv(annotation)
            # ---- 行级：新增 file_path 列，将相对文件名拼接为绝对路径。
            #           img_dirs[idx] 如 "spring"，row.iloc[0] 如 "image_0001.jpg" ----
            img_labels['file_path'] = img_labels.apply(
                lambda row: os.path.join(base_dir, img_dirs[idx], row.iloc[0]), axis=1
            )
            # ---- 行级：若指定了 img_range，截取该索引范围内的行（含 end）----
            if self.img_range is not None:
                img_labels = img_labels.iloc[self.img_range[0]:self.img_range[1]+1]
            # ---- 行级：若指定了 skip，跳过前 skip 行 ----
            if self.skip > 0:
                img_labels = img_labels.iloc[self.skip:]
            # ---- 行级：子采样：每隔 filter 行取一行（::filter 是 Python 切片语法）----
            img_labels = img_labels.iloc[::filter]
            # ---- 行级：若指定了 max_samples，只保留前 max_samples 行 ----
            if max_samples is not None:
                img_labels = img_labels.iloc[:max_samples]

            # ---- 行级：test=True（测试/推理模式）：直接赋值给 self.img_labels。
            #           test=False（训练模式）：将多个 DataFrame 累积到列表，稍后合并。----
            if test:
                self.img_labels = img_labels
            else:
                self.img_labels.append(img_labels)

        # ---- 行级：训练模式下，self.img_labels 是列表，用 pd.concat 纵向合并所有 DataFrame。
        #           ignore_index=True 重新生成连续整数索引。----
        if isinstance(self.img_labels, list):
            self.img_labels = pd.concat(self.img_labels, ignore_index=True)

    def __len__(self):
        # ================================================================================
        # 【函数级注释】__len__ —— 返回数据集样本总数
        # ================================================================================
        # PyTorch DataLoader 通过此函数知道数据集大小，用于计算 epoch 迭代次数。
        # ================================================================================
        return len(self.img_labels)

    def __getitem__(self, idx):
        # ================================================================================
        # 【函数级注释】__getitem__ —— 获取单个样本
        # ================================================================================
        # 输入：idx —— 整数索引（由 DataLoader 的 Sampler 决定，可能经过 shuffle）
        # 输出：tuple (image, label)
        #   image: 经过 ProcessImage 处理后的脉冲张量，形状 [H*W]
        #   label: 全局图像索引（整数），用于训练时计算 Spike Forcing 的目标神经元位置
        #
        # 工作流程：
        #   1. 从 DataFrame 中取出第 idx 行的 file_path
        #   2. 检查文件是否存在（防错）
        #   3. read_image() 读取为 torch.Tensor [C, H, W]，值域 [0, 255]
        #   4. 取第 2 列（索引 1）作为标签
        #   5. 应用 transform（ProcessImage 流水线）
        #   6. 返回 (image, label)
        # ================================================================================

        # ---- 行级：从已处理的 DataFrame 中取出第 idx 行的图像路径 ----
        img_path = self.img_labels.iloc[idx]['file_path']
        # ---- 行级：防御性检查：若文件不存在立即抛出异常，避免后续 read_image 报错难定位 ----
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No file found for index {idx} at {img_path}.")

        # ---- 行级：使用 torchvision.io.read_image 读取图像。
        #           输出为 torch.uint8，形状 [C, H, W]，值域 [0, 255]。
        #           支持 JPEG、PNG 等常见格式，底层使用 libpng/libjpeg。----
        image = read_image(img_path)
        # ---- 行级：取第 2 列（iloc 索引 1）作为标签，通常是图像的全局序号或场景 ID ----
        label = self.img_labels.iloc[idx, 1]

        # ---- 行级：若定义了图像变换（如 ProcessImage），将原始图转换为脉冲张量 ----
        if self.transform:
            image = self.transform(image)
        # ---- 行级：若定义了标签变换（本项目通常不使用），对标签做转换 ----
        if self.target_transform:
            label = self.target_transform(label)

        # ---- 行级：返回 (图像脉冲张量, 标签) 元组，DataLoader 会将多个样本堆叠为 batch ----
        return image, label
