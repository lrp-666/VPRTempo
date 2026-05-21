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
dataset.py —— 图像数据集与预处理流水线
================================================================================
理论对应：
  - 公式 (8) Gamma 校正      → ProcessImage.__call__() 第 138-142 行
  - IV-A 数据预处理          → ProcessImage 完整流程
  - III-A 时序编码           → SetImageAsSpikes（幅度编码单脉冲）
  - Patch Normalization      → PatchNormalisePad（局部 Z-score）
================================================================================
"""

import os
import math
import torch

import pandas as pd
import numpy as np
import torch.nn.functional as F

from torchvision.io import read_image
from torch.utils.data import Dataset


# ================================================================================
# 类：GetPatches2D —— 2D 滑动窗口提取器
# ================================================================================
# 功能：使用 PyTorch unfold 提取图像的局部滑动窗口（patches）。
# 这是 PatchNormalisePad 的底层工具，用于计算每个像素邻域的均值和标准差。
# ================================================================================
class GetPatches2D:
    def __init__(self, patch_size, image_pad):
        self.patch_size = patch_size   # 例如 (15, 15)
        self.image_pad = image_pad     # 经过 F.pad 填充后的图像张量
    
    def __call__(self, img):
        # 使用 unfold 在 H 和 W 维度上提取滑动窗口
        # unfold(dim, size, step) 沿指定维度以 step=1 滑动，提取 size 大小的窗口
        unfolded = self.image_pad.unfold(0, self.patch_size[0], 1).unfold(1, self.patch_size[1], 1)
        
        # reshape: 将窗口展平为 [patch_size*patch_size, num_patches]
        # 这样每个列向量是一个 patch 的像素值
        patches = unfolded.permute(2, 3, 0, 1).contiguous().view(self.patch_size[0]*self.patch_size[1], -1)
        return patches


# ================================================================================
# 类：PatchNormalisePad —— 局部块归一化
# ================================================================================
# 理论来源：VPRTempo 论文 IV-A "patch normalization"
# 功能：对每个像素的局部邻域（patch）做 Z-score 归一化：
#       im_norm = (pixel - μ_patch) / σ_patch
# 这与全局归一化不同，它能适应局部光照变化，对跨季节 VPR 更鲁棒。
# ================================================================================
class PatchNormalisePad:
    def __init__(self, patches):
        self.patches = patches  # patch 边长，例如 15

    def nanstd(self, input_tensor, dim=None, unbiased=True):
        """
        ================================================================================
        函数层说明：自定义 nanstd（支持 NaN 的 std）
        ================================================================================
        PyTorch 标准库没有 nanstd，此处手动实现：
          1. 统计非 NaN 元素个数
          2. 计算非 NaN 元素的均值
          3. 计算方差（支持 Bessel 校正的无偏估计）
        ================================================================================
        """
        if dim is not None:
            valid_count = torch.sum(~torch.isnan(input_tensor), dim=dim, dtype=torch.float)
            mean = torch.nansum(input_tensor, dim=dim) / valid_count
            diff = input_tensor - mean.unsqueeze(dim)
            variance = torch.nansum(diff * diff, dim=dim) / valid_count
            if unbiased:
                variance = variance * (valid_count / (valid_count - 1))
        else:
            valid_count = torch.sum(~torch.isnan(input_tensor), dtype=torch.float)
            mean = torch.nansum(input_tensor) / valid_count
            diff = input_tensor - mean
            variance = torch.nansum(diff * diff) / valid_count
            if unbiased:
                variance = variance * (valid_count / (valid_count - 1))
        return torch.sqrt(variance)
   
    def __call__(self, img):
        """
        ================================================================================
        函数层说明：Patch Normalization 主流程
        ================================================================================
        输入：img [1, H, W] 或 [H, W] —— 单通道灰度图
        输出：im_norm [H, W] —— 局部归一化后的图像，值域约 [-1, 1]
        步骤：
          1. squeeze 去掉通道维
          2. 计算 padding 大小（patch 半径）
          3. F.pad 用 NaN 填充边缘
          4. 提取所有 patches
          5. 对每个 patch 计算 μ 和 σ
          6. (img - μ) / σ 得到归一化图像
          7. clip 到 [-1, 1]，NaN 置 0
        ================================================================================
        """
        img = torch.squeeze(img, 0)
        patch_size = (self.patches, self.patches)
        patch_half_size = [int((p-1)/2) for p in patch_size]
        
        # 计算 padding：left, right, top, bottom
        if isinstance(patch_half_size, int):
            pad = (patch_half_size, patch_half_size, patch_half_size, patch_half_size)
        else:
            pad = (patch_half_size[1], patch_half_size[1], patch_half_size[0], patch_half_size[0])

        # 用 NaN 填充边缘，这样边界像素的 patch 不会引入虚假零值
        image_pad = F.pad(img, pad, mode='constant', value=float('nan'))

        nrows = img.shape[0] 
        ncols = img.shape[1]
        patcher = GetPatches2D(patch_size, image_pad)
        patches = patcher(img)
        
        # 对每个 patch（列向量）计算均值和标准差
        mus = torch.nanmean(patches, dim=0)
        stds = self.nanstd(patches, dim=0)
        
        # Z-score 归一化：重塑回图像形状
        with np.errstate(divide='ignore', invalid='ignore'):
            im_norm = (img - mus.reshape(nrows, ncols)) / stds.reshape(nrows, ncols)
        
        # 后处理：NaN→0（通常是 std=0 的均匀区域），clip 到 [-1, 1]
        im_norm[torch.isnan(im_norm)] = 0.0
        im_norm[im_norm < -1.0] = -1.0
        im_norm[im_norm > 1.0] = 1.0
        
        return im_norm


# ================================================================================
# 类：SetImageAsSpikes —— 将图像编码为脉冲幅度（时序编码）
# ================================================================================
# 理论来源：VPRTempo 论文 III-A "Temporal coding for visual place recognition"
# 核心思想：像素强度不决定脉冲数量（速率编码），而是决定单个脉冲的幅度。
#          幅度 ∈ [0, 1] 被抽象为 theta 振荡的一个时间步内的发放时刻：
#          - 幅度 1.0 → 最早发放
#          - 幅度 0.1 → 最晚发放
# 量化支持：FakeQuantize 用于 QAT（量化感知训练），观察统计值范围。
# ================================================================================
class SetImageAsSpikes:
    def __init__(self, intensity=255, test=True):
        self.intensity = intensity  # 最大像素值（uint8 为 255）
        
        # QAT 伪量化器：模拟 INT8 量化对梯度的影响
        self.fake_quantize = torch.quantization.FakeQuantize(
            observer=torch.quantization.MovingAverageMinMaxObserver, 
            quant_min=0, 
            quant_max=255, 
            dtype=torch.quint8, 
            qscheme=torch.per_tensor_affine, 
            reduce_range=False
        )
        
    def train(self):
        self.fake_quantize.train()

    def eval(self):
        self.fake_quantize.eval()    
    
    def __call__(self, img_tensor):
        """
        ================================================================================
        函数层说明：图像 → 脉冲幅度编码
        ================================================================================
        输入：img_tensor [1, H, W] —— uint8 格式的灰度图
        输出：spikes [1, H*W] —— 每个像素对应一个输入神经元的脉冲幅度 [0, 1]
        步骤：
          1. view 展平为 [1, 1, H*W]
          2. 除以 255 归一化到 [0, 1]
          3. FakeQuantize（训练时观察，推理时真量化）
        ================================================================================
        """
        N, W, H = img_tensor.shape
        reshaped_batch = img_tensor.view(N, 1, -1)
        
        # 归一化到 [0, 1]：spike 幅度 = 像素强度 / 255
        normalized_batch = reshaped_batch / self.intensity
        normalized_batch = torch.squeeze(normalized_batch, 0)

        # 伪量化（QAT 支持）
        spikes = self.fake_quantize(normalized_batch)
        
        # 推理模式下，执行真正的 INT8 量化
        if not self.fake_quantize.training:
            scale, zero_point = self.fake_quantize.calculate_qparams()
            spikes = torch.quantize_per_tensor(spikes, float(scale), int(zero_point), dtype=torch.quint8)

        return spikes


# ================================================================================
# 类：ProcessImage —— 完整图像预处理流水线
# ================================================================================
# 对应 VPRTempo 论文 IV-A 的完整预处理链：
#   RGB → Grayscale → Gamma 校正 → Resize → Patch Normalization → uint8 → Spike 编码
# ================================================================================
class ProcessImage:
    def __init__(self, dims, patches):
        self.dims = dims       # 缩放目标尺寸 [H, W]，如 [56, 56]
        self.patches = patches # Patch Normalization 窗口大小，如 15
        
    def __call__(self, img):
        """
        ================================================================================
        函数层说明：完整图像预处理流水线
        ================================================================================
        输入：img [3, H_orig, W_orig] —— torchvision.io.read_image 读取的 RGB 图像
        输出：spikes [H*W] —— 可直接输入 SNN 的脉冲幅度张量
        
        步骤详解：
        ------------------------------------------------------------------------
        Step 1: RGB → Grayscale
            使用标准 ITU-R BT.601 权重：0.299R + 0.587G + 0.114B
            原因：VPR 任务对颜色不敏感，灰度图减少 2/3 输入维度。
        ------------------------------------------------------------------------
        Step 2: Gamma 校正（对应公式 8）
            公式：ρ_norm = ρ^γ
            其中 γ = ln(0.5 * 255) / ln(mean)
            物理意义：自适应 gamma。若图像偏暗（mean 小），gamma 变小，增强暗部；
                     若图像偏亮（mean 大），gamma 变大，压缩亮部。
            这使得不同季节/光照条件下的图像具有相似的动态范围。
        ------------------------------------------------------------------------
        Step 3: Resize
            使用双线性插值缩放到目标尺寸（默认 56×56）
        ------------------------------------------------------------------------
        Step 4: Patch Normalization
            对每个像素的局部邻域做 Z-score，适应局部光照变化
        ------------------------------------------------------------------------
        Step 5: 量化到 uint8
            img = 255 * (1 + im_norm) / 2
            将 [-1, 1] 映射到 [0, 255]，便于 spike 编码
        ------------------------------------------------------------------------
        Step 6: Spike 编码
            调用 SetImageAsSpikes，将 uint8 像素转为 [0, 1] 幅度
        ================================================================================
        """
        # Step 1: RGB → Grayscale
        if img.shape[0] == 3:
            img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        img = img.unsqueeze(0)

        # Step 2: Gamma 校正（对应 VPRTempo 公式 8）
        mid = 0.5
        mean = torch.mean(img.float())
        # γ = ln(mid * 255) / ln(mean)
        # 当 mean = 0.5*255 = 127.5 时，gamma = 1（无变化）
        # 当 mean < 127.5（偏暗），gamma < 1，暗部被拉伸
        # 当 mean > 127.5（偏亮），gamma > 1，亮部被压缩
        gamma = math.log(mid * 255) / math.log(mean)
        img = torch.pow(img, gamma).clip(0, 255)
        
        # Step 3: Resize
        if len(img.shape) == 3:
            img = img.unsqueeze(0)  # 增加 batch 维 [1, 1, H, W]
        img = F.interpolate(img, size=self.dims, mode='bilinear', align_corners=False)
        img = img.squeeze(0)  # [1, H, W]
        
        # Step 4: Patch Normalization
        patch_normaliser = PatchNormalisePad(self.patches)
        im_norm = patch_normaliser(img)  # 输出约 [-1, 1]
        
        # Step 5: 映射到 uint8
        img = (255.0 * (1 + im_norm) / 2.0).to(dtype=torch.uint8)
        img = torch.unsqueeze(img, 0)  # [1, H, W]
        
        # Step 6: Spike 编码（时序编码）
        spike_maker = SetImageAsSpikes()
        img = spike_maker(img)
        img = torch.squeeze(img, 0)  # 去掉 batch 维，输出 [H*W]

        return img


# ================================================================================
# 类：CustomImageDataset —— 自定义数据集
# ================================================================================
# 功能：从 CSV 标注文件加载图像路径和标签，支持：
#   - filter：子采样步长（每 filter 帧取一帧）
#   - skip：跳过前 skip 帧
#   - max_samples：限制最大样本数
#   - img_range：只加载指定索引范围的样本（用于多模块划分）
# ================================================================================
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, base_dir, img_dirs, transform=None, target_transform=None, 
                 filter=1, skip=0, max_samples=None, test=True, img_range=None):
        self.transform = transform
        self.target_transform = target_transform
        self.filter = filter        # 子采样步长（对应论文"filter images every 8 seconds"）
        self.img_range = img_range  # 模块内图像索引范围 [start, end]
        self.skip = skip            # 跳过前 skip 张图像
        
        # ----------------------------------------
        # 逐行说明：加载并处理 CSV 标注
        # ----------------------------------------
        # CSV 格式：每行包含 [图像文件名, 全局标签]
        # 若 database_dirs 有多个（如 spring,fall），会分别加载后合并
        self.img_labels = []
        if not isinstance(annotations_file, list):
            annotations_file = [annotations_file]
        for idx, annotation in enumerate(annotations_file):
            img_labels = pd.read_csv(annotation)
            # 构造完整文件路径：base_dir / img_dir / filename
            img_labels['file_path'] = img_labels.apply(
                lambda row: os.path.join(base_dir, img_dirs[idx], row.iloc[0]), axis=1
            )
            # 应用 img_range（模块划分）
            if self.img_range is not None:
                img_labels = img_labels.iloc[self.img_range[0]:self.img_range[1]+1]
            # 应用 skip
            if self.skip > 0:
                img_labels = img_labels.iloc[self.skip:]
            # 应用 filter（子采样）
            img_labels = img_labels.iloc[::filter]
            # 限制最大样本数
            if max_samples is not None:
                img_labels = img_labels.iloc[:max_samples]
            
            # test=True 时直接赋值；test=False 时累积到列表后合并
            if test:
                self.img_labels = img_labels
            else:
                self.img_labels.append(img_labels)
        
        # 训练模式下合并多个 DataFrame
        if isinstance(self.img_labels, list):
            self.img_labels = pd.concat(self.img_labels, ignore_index=True)
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """
        ================================================================================
        函数层说明：获取单个样本
        ================================================================================
        输出：(image, label)
          image: 经过 ProcessImage 处理后的脉冲张量 [H*W]
          label: 全局图像索引（整数），用于计算 Spike Forcing 的神经元位置
        ================================================================================
        """
        img_path = self.img_labels.iloc[idx]['file_path']
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No file found for index {idx} at {img_path}.")
            
        image = read_image(img_path)  # [C, H, W]，值域 [0, 255]
        label = self.img_labels.iloc[idx, 1]  # 全局标签
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
