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
Part 1: 模块说明
================================================================================
本文件实现 IDEA1 的卷积学习规则（PLAN.md S2.3 / S2.4）：
  calc_stdp_conv           —— 卷积 STDP 权重更新（正式向量化路径，ADR-3）
  calc_stdp_conv_reference —— Python 循环参考版（仅用于对拍，不用于训练）
  apply_itp_conv           —— 卷积版 ITP 阈值可塑性（observed 必须在 WTA 前统计！）

核心公式（blitnet 公式 2 的卷积版）：
    ΔK_c = η · (0.5 − post) · pre_term(patch)，只在 winner 位置聚合触发（local/global WTA）；
    none 模式为稠密 M = (0.5 − post) 全图（PLAN S2.2 none 条）。

向量化关键（ADR-3）：ΔK_c = Σ_winners (0.5−post)·patch 恰为 pre 与响应图 M 的互相关，
即卷积层权重梯度 → torch.nn.grad.conv2d_weight 一步完成，走 cuDNN，无索引体操。

pre_mode（S2.3 关键设计决策，直流塌缩防线）：
  centered（默认）: pre − 0.5，不动点为"获胜 patch 相对背景的平均偏离" → 边缘结构；
  amp             : pre 原幅度，不动点含强直流分量（对照，Table 3b）；
  heaviside       : Θ(pre > 0.5)（对照）。
================================================================================
"""
import torch


# ================================================================================
# 函数：calc_stdp_conv —— 卷积 STDP 权重更新（正式向量化路径）
# ================================================================================
def calc_stdp_conv(pre_img,   # [1, C_in, H, W] 输入 spike 图（已 reshape）
                   out,       # ConvLayerOutput（forward 的返回，含 pre_wta / winner_mask）
                   layer,     # ConvSNNLayer 实例
                   pre_mode='centered',   # {'centered','amp','heaviside'}
                   agg_mode='mean',       # {'mean','sum'}
                   ):
    """
    ================================================================================
    【函数级注释】卷积 STDP 更新（ADR-3 向量化路径）
    ================================================================================
    流程：
        1. 构造响应图 M：winner 处填 (0.5−post)，其余 0；none 模式为稠密全图
        2. pre 端三选一（centered 默认，直流塌缩防线）
        3. dK = conv2d_weight(pre_term, w_shape, M)（互相关 = 卷积权重梯度，天然 sum 聚合）
        4. agg_mode='mean' 时除以每通道 winner 数
        5. 保范数归一化用：更新前记录每核 L1 范数
        6. 权重更新：E/I 分组学习率（兴奋 +η、抑制 −η，对齐 blitnet 抑制反向更新）
        7. 符号钳制 [0,10]/[-10,0]（S2.3 第 4 条；偏离 blitnet 的 [1e-6,10] 已在 PLAN 注明）
        8. 保范数 L1 归一化（仅本次有更新的通道，防止核幅度漂移）
    ================================================================================
    """
    with torch.no_grad():
        W = layer.w.weight.data
        C = W.shape[0]
        device = W.device

        # ---- Step 1: 构造响应图 M ----
        if layer.wta_mode == 'none':
            M = 0.5 - out.pre_wta                                   # 稠密（S2.2 none 条）
        else:
            M = torch.zeros_like(out.pre_wta)
            M[out.winner_mask] = 0.5 - out.pre_wta[out.winner_mask]

        # ---- Step 2: pre 端三选一 ----
        if pre_mode == 'centered':
            x = pre_img - 0.5
        elif pre_mode == 'amp':
            x = pre_img
        elif pre_mode == 'heaviside':
            x = (pre_img > 0.5).float()
        else:
            raise ValueError(f"Unknown pre_mode: {pre_mode}")

        # ---- Step 3: 全通道一步 dK（互相关 = 卷积权重梯度）----
        dK = torch.nn.grad.conv2d_weight(
            x, layer.w.weight.shape, M,
            stride=1, padding=layer.padding, dilation=1, groups=1)  # [C, C_in, k, k]

        # ---- Step 4: mean 聚合时除以每通道 winner 数 ----
        if agg_mode == 'mean':
            if layer.wta_mode == 'none':
                cnt = torch.full((C,), float(M.shape[2] * M.shape[3]), device=device)
            else:
                cnt = out.winner_mask.float().sum(dim=(0, 2, 3))    # [C]
            dK = dK / cnt.view(C, 1, 1, 1).clamp(min=1.0)
        elif agg_mode != 'sum':
            raise ValueError(f"Unknown agg_mode: {agg_mode}")

        # ---- Step 5: 更新前记录每核 L1 范数（保范数归一化用）----
        nrm_before = torch.linalg.norm(W.flatten(1), ord=1, dim=1)  # [C]

        # ---- Step 6: 权重更新（E/I 分组学习率）----
        sign = torch.where(layer.havconnExc, 1.0, -1.0).view(C, 1, 1, 1)
        W += layer.eta_stdp * sign * dK

        # ---- Step 7: 符号钳制（兴奋核 [0,10]，抑制核 [-10,0]）----
        exc = layer.havconnExc.view(C, 1, 1, 1)
        layer.w.weight.data = torch.where(exc, W.clamp(min=0.0, max=10.0),
                                          W.clamp(min=-10.0, max=0.0))

        # ---- Step 8: 保范数 L1 归一化（仅本次有更新且范数非零的通道）----
        updated = (dK.abs().flatten(1).sum(dim=1) > 0)
        nrm_after = torch.linalg.norm(layer.w.weight.data.flatten(1), ord=1, dim=1)                      # [C]
        scale = torch.ones_like(nrm_before)
        valid = updated & (nrm_before > 0) & (nrm_after > 0)
        scale[valid] = nrm_before[valid] / nrm_after[valid]
        layer.w.weight.data *= scale.view(C, 1, 1, 1)


# ================================================================================
# 函数：calc_stdp_conv_reference —— Python 循环参考版（仅用于对拍，不用于训练）
# ================================================================================
# 与 calc_stdp_conv 语义完全一致的朴素实现：逐通道、逐 winner 循环，直接按公式计算。
# 仅用于玩具测试中的对拍（两版输出必须逐元素一致），验证向量化路径没有引入实现错误。
# ================================================================================
def calc_stdp_conv_reference(pre_img, out, layer, pre_mode='centered', agg_mode='mean'):
    with torch.no_grad():
        W = layer.w.weight.data.clone()
        C = W.shape[0]
        k = layer.kernel_size
        pad = layer.padding

        # pre 端三选一
        if pre_mode == 'centered':
            x = pre_img - 0.5
        elif pre_mode == 'amp':
            x = pre_img
        else:
            x = (pre_img > 0.5).float()

        # winner 坐标（none 模式 = 所有位置）
        if layer.wta_mode == 'none':
            Hp, Wp = out.pre_wta.shape[2], out.pre_wta.shape[3]
            coords = [(c, y, xx) for c in range(C) for y in range(Hp) for xx in range(Wp)]
        else:
            nz = out.winner_mask[0].nonzero(as_tuple=False)        # [N, 3] (c, y, x)
            coords = [(int(r[0]), int(r[1]), int(r[2])) for r in nz]

        # 每通道聚合
        dK = torch.zeros_like(W)
        counts = torch.zeros(C, device=W.device)
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad)) if pad > 0 else x
        for (c, y, xx) in coords:
            post = out.pre_wta[0, c, y, xx]                        # winner 响应（标量）
            patch = x_pad[0, 0, y:y + k, xx:xx + k]                # 感受野 patch（含 padding 对齐）
            dK[c, 0] += (0.5 - post) * patch
            counts[c] += 1
        if agg_mode == 'mean':
            dK = dK / counts.view(C, 1, 1, 1).clamp(min=1.0)

        # E/I 分组学习率 + 更新 + 钳制 + 保范数归一化（与正式版同序）
        sign = torch.where(layer.havconnExc, 1.0, -1.0).view(C, 1, 1, 1).to(W.device)
        nrm_before = torch.linalg.norm(W.flatten(1), ord=1, dim=1)
        W += layer.eta_stdp * sign * dK
        exc = layer.havconnExc.view(C, 1, 1, 1)
        layer.w.weight.data = torch.where(exc, W.clamp(min=0.0, max=10.0),
                                          W.clamp(min=-10.0, max=0.0))
        nrm_after = torch.linalg.norm(layer.w.weight.data.flatten(1), ord=1, dim=1)
        updated = (counts > 0)
        scale = torch.ones_like(nrm_before)
        valid = updated & (nrm_before > 0) & (nrm_after > 0)
        scale[valid] = nrm_before[valid] / nrm_after[valid]
        layer.w.weight.data *= scale.view(C, 1, 1, 1)


# ================================================================================
# 函数：apply_itp_conv —— 卷积版 ITP 阈值可塑性（S2.4）
# ================================================================================
def apply_itp_conv(out, layer):
    """
    ================================================================================
    【函数级注释】ITP：Δθ = η_ITP·(observed − f)，observed 为每通道 WTA 前发放率
    ================================================================================
    注意：observed 必须用 pre_wta（WTA 前）统计——WTA 后每通道 winner 数被结构
    钉死（local 恒 36 个），post-WTA 发放率是常数，ITP 会失效（PLAN S2.4）。
    ================================================================================
    """
    with torch.no_grad():
        observed = (out.pre_wta > 0).float().mean(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        layer.thr.data += layer.eta_ip * (observed - layer.fire_rate)
        layer.thr.data.clamp_(min=0)                                          # 对齐 blitnet.py:606


# ================================================================================
# 函数：train_conv_layer —— conv 层的训练循环（S2.5）
# ================================================================================
# 结构镜像 VPRTempoTrain.train_model（VPRTempoTrain.py:327-485），但有三点不同：
#   1. epoch 数用 model.conv_epoch（独立参数），T = 图像数 × conv_epoch（独立退火，
#      不共用 model.T —— conv 层步数与 feature/output 层不同，共用会退火错误）；
#   2. 每样本：reshape → forward（含 WTA/池化）→ calc_stdp_conv → apply_itp_conv；
#   3. 无 Spike Forcing / idx（无监督，不需要标签）。
# ================================================================================
def train_conv_layer(train_loader, layer, model, model_num=0):
    """
    ================================================================================
    【函数级注释】训练单个模块的 conv 前端（无监督 STDP + ITP）
    ================================================================================
    参数：
        train_loader — DataLoader，每次返回 (spikes [1, H*W], labels)
        layer        — ConvSNNLayer 实例（models[i].conv_layer）
        model        — 当前 VPRTempoTrain 实例（取 device / conv_epoch / pre_mode / agg_mode）
        model_num    — 模块序号（进度条显示用）
    ================================================================================
    """
    from tqdm import tqdm

    n_imgs = len(train_loader.dataset)
    T = int(n_imgs * model.conv_epoch)          # 独立退火总步数（S2.5 卡片）
    pbar = tqdm(total=T, desc=f"Module {model_num + 1} conv_layer", position=0)

    # 保存初始学习率（对齐 train_model 的 detach 副本做法）
    init_itp = layer.eta_ip.detach()
    init_stdp = layer.eta_stdp.detach()
    mod = 0
    pre_mode = getattr(model, 'pre_mode', 'centered')
    agg_mode = getattr(model, 'agg_mode', 'mean')

    for _ in range(model.conv_epoch):
        for spikes, _ in train_loader:
            spikes = spikes.to(model.device)
            x = layer.reshape_input(spikes)              # [1,1,H,W]
            out = layer(x)                               # 前向 + WTA + 池化
            calc_stdp_conv(x, out, layer, pre_mode=pre_mode, agg_mode=agg_mode)
            apply_itp_conv(out, layer)

            # 学习率退火 (1−t/T)²，每 100 步一次（镜像 _anneal_learning_rate 但用 conv 的 T）
            if mod % 100 == 0:
                pt = pow(float(T - mod) / T, 2)
                layer.eta_ip = torch.mul(init_itp, pt)
                layer.eta_stdp = torch.mul(init_stdp, pt)
            mod += 1
            pbar.update(1)
    pbar.close()
