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

S2.11 规则锦标赛 Round 1 开关（layer 属性携带，默认全关 = B2 行为逐比特不变）：
  bcm_gate   : Step 1 的固定 0.5 → 每通道滑动阈值 θ_M,c（EMA of post²，α=bcm_alpha，
               初值 0.25；θ_M 存在 layer.theta_m 上，普通属性不进 state_dict）；
  rank_push  : Step 1 后名次手术（仅 local WTA）——块内跨通道按 winner 响应排序，
               第 rank_k 名通道在其 winner 位置的 M 值改为 −rank_delta·(gate−post)；
  oja_decay  : Step 5/8 保范数归一化关闭，Step 6 更新里加 −η·post²·K_c 衰减项
               （post² 按 winner 位置聚合，mean 模式下与 dK 同口径除 winner 数）；
  attractor  : Step 2 pre_term 由 (pre−0.5) 换成 (patch − 当前核)：
               dK = conv2d_weight(pre_img, w_shape, M) − (Σ_{y,x} M_c)·K_c，
               即逐 winner 更新 (0.5−post)·(patch−K_c) 的向量化形式。
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
    # frozen 层（B1 random_conv / B5 Gabor，S2.6/S2.9）：完全旁路权重更新——
    # 含 Step 7 符号钳制与 Step 8 保范数归一化（负瓣保护：Gabor 核的负瓣一旦
    # 经过 clamp(min=0) 即被摧毁，frozen 必须连这两条路径都不经过）。
    if getattr(layer, 'frozen', False):
        return

    with torch.no_grad():
        W = layer.w.weight.data
        C = W.shape[0]
        device = W.device

        # ---- Step 1: 构造响应图 M ----
        # R1（bcm_gate）：固定 0.5 → 每通道滑动阈值 θ_M,c（EMA of post²）。
        # 先用当前 θ_M 构造 M，再更新 θ_M（标准 BCM 时序：用旧阈值门控本步）。
        bcm_on = getattr(layer, 'bcm_gate', False)
        gate = layer.theta_m.clone() if bcm_on else 0.5
        if layer.wta_mode == 'none':
            M = gate - out.pre_wta                                  # 稠密（S2.2 none 条）
        else:
            M = torch.zeros_like(out.pre_wta)
            M[out.winner_mask] = (gate - out.pre_wta)[out.winner_mask] \
                if bcm_on else 0.5 - out.pre_wta[out.winner_mask]

        # θ_M 更新：θ ← (1−α)θ + α·E[post²]（每通道全图均方响应，含未发放位置）
        if bcm_on:
            post2 = out.pre_wta.pow(2).mean(dim=(0, 2, 3), keepdim=True)  # [1,C,1,1]
            layer.theta_m.mul_(1.0 - layer.bcm_alpha).add_(layer.bcm_alpha * post2)

        # ---- Step 1.5: R2 名次反推（rank_push，仅 local WTA）----
        # 块内跨通道按 winner 响应排序：rank 1（最高）正常更新（现有行为）；
        # 第 rank_k 名通道在其自身 winner 位置的 M 值改为 −δ·(gate − post_rankk)。
        if getattr(layer, 'rank_push', False):
            if layer.wta_mode != 'local':
                raise ValueError("rank_push 仅在 wta_mode='local' 下定义")
            b = layer.wta_block
            Hc, Wc = layer.H_pool * b, layer.W_pool * b
            blk = out.pre_wta[:, :, :Hc, :Wc]
            C_, Hb, Wb = blk.shape[1], layer.H_pool, layer.W_pool
            # 块内每通道 winner 值 = 块 max（winner 即块内 argmax，见 _apply_wta）
            winv = blk.view(1, C_, Hb, b, Wb, b).permute(0, 1, 2, 4, 3, 5) \
                      .reshape(1, C_, Hb, Wb, b * b).amax(dim=-1)           # [1,C,Hb,Wb]
            ch_k = winv.argsort(dim=1, descending=True)[:, layer.rank_k - 1:layer.rank_k]
            # 该通道在其块内的 winner 位置（argmax 索引 → (y,x) 坐标）
            pos = blk.view(1, C_, Hb, b, Wb, b).permute(0, 1, 2, 4, 3, 5) \
                     .reshape(1, C_, Hb, Wb, b * b).argmax(dim=-1)          # [1,C,Hb,Wb]
            pos_k = pos.gather(1, ch_k).flatten()                           # [Hb*Wb]
            ch_k = ch_k.flatten()
            val_k = winv.gather(1, ch_k.view(1, 1, Hb, Wb)).flatten()
            g_k = (gate.flatten()[ch_k] if bcm_on
                   else torch.full_like(val_k, 0.5))
            hb = torch.arange(Hb, device=device).view(Hb, 1).expand(Hb, Wb).reshape(-1)
            wb = torch.arange(Wb, device=device).view(1, Wb).expand(Hb, Wb).reshape(-1)
            ys = hb * b + torch.div(pos_k, b, rounding_mode='floor')
            xs = wb * b + pos_k % b
            M[0, ch_k, ys, xs] = -layer.rank_delta * (g_k - val_k)

        # ---- Step 2: pre 端三选一（R4 attractor：换成原始 pre，弹性项在 Step 3.5 扣）----
        if getattr(layer, 'attractor', False):
            x = pre_img                    # (patch − K) 的 patch 部分用原始输入
        elif pre_mode == 'centered':
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

        # ---- Step 3.5: R4 弹性项 −(Σ_{y,x} M_c)·K_c ----
        # Σ_w (0.5−post)·(patch − K_c) = conv2d_weight(pre,M) − (Σ_w (0.5−post))·K_c
        if getattr(layer, 'attractor', False):
            dK = dK - W * M.sum(dim=(0, 2, 3)).view(C, 1, 1, 1)

        # ---- Step 4: mean 聚合时除以每通道 winner 数 ----
        cnt = None
        if agg_mode == 'mean':
            if layer.wta_mode == 'none':
                cnt = torch.full((C,), float(M.shape[2] * M.shape[3]), device=device)
            else:
                cnt = out.winner_mask.float().sum(dim=(0, 2, 3))    # [C]
            dK = dK / cnt.view(C, 1, 1, 1).clamp(min=1.0)
        elif agg_mode != 'sum':
            raise ValueError(f"Unknown agg_mode: {agg_mode}")

        # ---- Step 4.5: R3 Oja 衰减项 −post²·K_c（与 dK 同聚合口径）----
        oja = None
        if getattr(layer, 'oja_decay', False):
            if layer.wta_mode == 'none':
                pw2 = out.pre_wta.pow(2).sum(dim=(0, 2, 3))         # [C]
            else:
                pw2 = (out.pre_wta.pow(2) * out.winner_mask).sum(dim=(0, 2, 3))
            if cnt is not None:
                pw2 = pw2 / cnt.clamp(min=1.0)
            oja = pw2.view(C, 1, 1, 1) * W

        # ---- Step 5: 更新前记录每核 L1 范数（保范数归一化用；R3 下跳过）----
        if not getattr(layer, 'oja_decay', False):
            nrm_before = torch.linalg.norm(W.flatten(1), ord=1, dim=1)  # [C]

        # ---- Step 6: 权重更新（E/I 分组学习率）----
        # free-sign（S2.10 B6b / S3.3-8）：统一 +η（分组符号依赖"抑制核 ≤0"的语义，
        # 符号放开后不再成立）
        # R3：Oja 衰减 −η·post²·K_c 直接作用于 W（不做 E/I 符号翻转——
        # 衰减是保范数替代，方向恒指向 0，与 Hebb 项的分组符号无关）
        if getattr(layer, 'free_sign', False):
            upd = layer.eta_stdp * dK
        else:
            sign = torch.where(layer.havconnExc, 1.0, -1.0).view(C, 1, 1, 1)
            upd = layer.eta_stdp * sign * dK
        if oja is not None:
            upd = upd - layer.eta_stdp * oja
        W += upd

        # ---- Step 7: 符号钳制（兴奋核 [0,10]，抑制核 [-10,0]）----
        # free-sign：放开符号钳制，仅保留幅度安全钳 clamp(-10,10) 防爆（S3.3-8）
        if getattr(layer, 'free_sign', False):
            layer.w.weight.data = W.clamp(min=-10.0, max=10.0)
        else:
            exc = layer.havconnExc.view(C, 1, 1, 1)
            layer.w.weight.data = torch.where(exc, W.clamp(min=0.0, max=10.0),
                                              W.clamp(min=-10.0, max=0.0))

        # ---- Step 8: 保范数 L1 归一化（仅本次有更新且范数非零的通道）----
        # R3（oja_decay）：显式保范数归一化关闭，由 Step 6 的 −post²·K_c 衰减项
        # 软归一化替代（幅度安全钳仍在 Step 7 生效，防发散）
        if getattr(layer, 'oja_decay', False):
            return
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
    # frozen 守卫与正式版保持一致（见 calc_stdp_conv 头部注释）
    if getattr(layer, 'frozen', False):
        return
    with torch.no_grad():
        W = layer.w.weight.data.clone()
        C = W.shape[0]
        k = layer.kernel_size
        pad = layer.padding

        # R1 BCM 滑动阈值（与正式版同时序：先用旧 θ 构造更新，再 EMA 更新 θ）
        bcm_on = getattr(layer, 'bcm_gate', False)

        # pre 端三选一（R4 attractor：patch 用原始输入，弹性项在循环内 −K_c）
        attractor = getattr(layer, 'attractor', False)
        if attractor:
            x = pre_img
        elif pre_mode == 'centered':
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

        # R2 名次反推（仅 local）：块内跨通道第 rank_k 名 → 其 winner 位置更新取 −δ
        # rank_neg: {(c, y, x): True} 标记被反推的 winner 位置
        rank_neg = set()
        if getattr(layer, 'rank_push', False):
            if layer.wta_mode != 'local':
                raise ValueError("rank_push 仅在 wta_mode='local' 下定义")
            b = layer.wta_block
            Hb, Wb = layer.H_pool, layer.W_pool
            # 块内每通道 winner 值与位置
            winval = {}
            winpos = {}
            for (c, y, xx) in coords:
                winval[(c, y // b, xx // b)] = max(
                    winval.get((c, y // b, xx // b), -1.0),
                    float(out.pre_wta[0, c, y, xx]))
                cur = winpos.get((c, y // b, xx // b))
                # 严格大于才换位置：与 argmax 的"取首个最大值"语义一致
                if cur is None or float(out.pre_wta[0, c, y, xx]) > cur[0]:
                    winpos[(c, y // b, xx // b)] = (float(out.pre_wta[0, c, y, xx]), y, xx)
            for hb in range(Hb):
                for wb in range(Wb):
                    order = sorted(range(C),
                                   key=lambda c: winval.get((c, hb, wb), 0.0),
                                   reverse=True)
                    ck = order[layer.rank_k - 1]
                    if (ck, hb, wb) in winpos:
                        _, y, xx = winpos[(ck, hb, wb)]
                        rank_neg.add((ck, y, xx))

        # 每通道聚合
        dK = torch.zeros_like(W)
        counts = torch.zeros(C, device=W.device)
        x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad)) if pad > 0 else x
        for (c, y, xx) in coords:
            post = out.pre_wta[0, c, y, xx]                        # winner 响应（标量）
            g = float(layer.theta_m[0, c, 0, 0]) if bcm_on else 0.5
            patch = x_pad[0, 0, y:y + k, xx:xx + k]                # 感受野 patch（含 padding 对齐）
            coef = (g - post)
            if (c, y, xx) in rank_neg:
                coef = -layer.rank_delta * coef
            upd = coef * (patch - W[c, 0]) if attractor else coef * patch
            dK[c, 0] += upd
            counts[c] += 1
        if agg_mode == 'mean':
            dK = dK / counts.view(C, 1, 1, 1).clamp(min=1.0)

        # R1：θ_M EMA 更新（在构造完本步更新之后，与正式版同时序）
        if bcm_on:
            post2 = out.pre_wta.pow(2).mean(dim=(0, 2, 3), keepdim=True)
            layer.theta_m.mul_(1.0 - layer.bcm_alpha).add_(layer.bcm_alpha * post2)

        # R3 Oja 衰减项（与正式版同口径：post² 按 winner 位置聚合，mean 时除 winner 数）
        oja = None
        if getattr(layer, 'oja_decay', False):
            if layer.wta_mode == 'none':
                pw2 = out.pre_wta.pow(2).sum(dim=(0, 2, 3))
            else:
                pw2 = (out.pre_wta.pow(2) * out.winner_mask).sum(dim=(0, 2, 3))
            if agg_mode == 'mean':
                pw2 = pw2 / counts.clamp(min=1.0)
            oja = pw2.view(C, 1, 1, 1) * W

        # E/I 分组学习率 + 更新 + 钳制 + 保范数归一化（与正式版同序）
        # free-sign 同步：统一 +η、幅度安全钳 clamp(-10,10)（见 calc_stdp_conv Step 6/7）
        free_sign = getattr(layer, 'free_sign', False)
        sign = torch.where(layer.havconnExc, 1.0, -1.0).view(C, 1, 1, 1).to(W.device)
        oja_off = getattr(layer, 'oja_decay', False)
        nrm_before = torch.linalg.norm(W.flatten(1), ord=1, dim=1)
        upd_w = layer.eta_stdp * (dK if free_sign else sign * dK)
        if oja is not None:
            upd_w = upd_w - layer.eta_stdp * oja
        W += upd_w
        if free_sign:
            layer.w.weight.data = W.clamp(min=-10.0, max=10.0)
        else:
            exc = layer.havconnExc.view(C, 1, 1, 1)
            layer.w.weight.data = torch.where(exc, W.clamp(min=0.0, max=10.0),
                                              W.clamp(min=-10.0, max=0.0))
        if oja_off:
            return
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
    conv_epoch = int(getattr(model, 'conv_epoch', 2))   # 防御性默认（旧配置可能缺 conv 字段）
    T = int(n_imgs * conv_epoch)          # 独立退火总步数（S2.5 卡片）
    pbar = tqdm(total=T, desc=f"Module {model_num + 1} conv_layer", position=0)

    # 保存初始学习率（对齐 train_model 的 detach 副本做法）
    init_itp = layer.eta_ip.detach()
    init_stdp = layer.eta_stdp.detach()
    mod = 0
    pre_mode = getattr(model, 'pre_mode', 'centered')
    agg_mode = getattr(model, 'agg_mode', 'mean')

    for _ in range(conv_epoch):
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
