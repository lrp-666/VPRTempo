#!/usr/bin/env python
# ================================================================================
# viz_kernels.py — IDEA1 S3.1 实验 1.2：卷积核可视化与量化分析（Figure 2）
#
# 功能：
#   1. 三方并排 grid 图 → results/fig2_kernels.png
#      左：随机初始化核（seed 0 现场重建，与 b1_500__seed0.pth 逐元素一致，代表"未训练"）
#      中：B2 训练后的核（vprtempo/models/b2_500_block2__seed0.pth 的 conv_layer.w.weight）
#      右：B5 真 Gabor 核（gabor_frontend.gabor_kernel_bank 确定性生成）
#      每核独立对称归一化（红=正、蓝=负），按 B2 的 havconnExc 掩码分 ON/OFF 两组标注。
#   2. 量化三指标 → results/table_kernel_stats.json + 控制台汇总
#      a. Gabor 拟合优度：每核拟合 2D Gabor（torch LBFGS 多起点最小二乘），
#         报 R² 中位数 + IQR，B1(init) vs B2 Mann-Whitney U 检验；
#      b. 方向选择性：拟合成功核（R²>0.5）的方向角分布 + 合成长度 R（倍角合成）；
#      c. 稀疏度：每核 |w| < 0.1·max|w| 的元素占比，B1 vs B2。
#
# 依赖说明：不引入 scipy——pixi 环境无 scipy 且纪律要求不改依赖清单。
#   Gabor 拟合用 torch.optim.LBFGS（8 个固定确定性起点），Mann-Whitney 用
#   手工 U 统计量 + 连续性/并列校正的正态近似（n1=n2=32，近似足够）。
#
# 已核实的坑（2026-09-03）：
#   - b2_500_block2__seed0.pth 在 GPU（RTX 4090）训练，b1_500 在 CPU 训练。
#     CPU/GPU RNG 不同 ⇒ B1 存核 ≠ B2 训前初始化（逐元素意义上）。但 seed-0 CPU
#     现场重建与 B1 存核逐元素一致（max diff 0.0，已验证），故左图/B1 组用
#     重建核（=B1 核），作为"同 seed 同结构未训练随机对照"，统计比较不受影响。
#   - B2 训练有符号钳制（兴奋核 ≥0、抑制核 ≤0）与每核 L1=1 保范数
#     （conv_learning.py），故拟合模型含幅度 + 直流偏置两个自由参数吸收量级。
#
# 用法（必须从仓库根目录运行）：
#   pixi run python IDEA1-covstdp/experiments/viz_kernels.py
# ================================================================================
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "IDEA1-covstdp" / "results"
B2_MODEL = REPO_ROOT / "vprtempo" / "models" / "b2_500_block2__seed0.pth"
B1_MODEL = REPO_ROOT / "vprtempo" / "models" / "b1_500__seed0.pth"

SEED = 0
R2_THRESHOLD = 0.5          # 方向选择性统计的拟合成功阈值（PLAN S3.1 建议值）
N_THETA_BINS = 6            # 方向角直方图分箱（[0,π) 六等分 = 30°/bin）


# ================================================================================
# IDEA1 模块按路径懒加载（目录名含连字符，对齐 conv_frontend.py 的做法）
# ================================================================================
def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ================================================================================
# 核获取：随机初始化（现场重建）/ B2 训练后 / B5 Gabor
# ================================================================================
def regenerate_random_init(seed=SEED):
    """复刻 main.py 播种 + VPRTempoTrain.__init__ 的 RNG 消耗顺序（构造前无其他消耗），
    返回 (W [C,1,k,k], havconnExc [C] bool)。已验证与 b1_500__seed0.pth 逐元素一致。"""
    csl = _load("idea1_conv_snn_layer", "IDEA1-covstdp/src/conv_snn_layer.py")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    layer = csl.ConvSNNLayer(input_dims=[28, 28], in_channels=1, out_channels=32,
                             kernel_size=5, thr_range=[0, 0.5], fire_rate=[0.2, 0.9],
                             wta_mode='local', wta_block=2, device='cpu',
                             inference=False, frozen=False)
    return layer.w.weight.detach().clone(), layer.havconnExc.clone()


def load_b2_kernels():
    """从组合 state_dict 提取 conv 层权重与 ON/OFF 掩码"""
    sd = torch.load(B2_MODEL, map_location='cpu', weights_only=True)['model_0']
    return (sd['conv_layer.w.weight'].detach().clone(),
            sd['conv_layer.havconnExc'].clone())


def load_gabor_kernels():
    gf = _load("idea1_gabor_frontend", "IDEA1-covstdp/src/gabor_frontend.py")
    return gf.gabor_kernel_bank(kernel_size=5, device='cpu').detach().clone()


# ================================================================================
# 2D Gabor 最小二乘拟合（torch LBFGS，多确定性起点）
#   模型: g(x,y) = amp·exp(−(xr²+γ²yr²)/(2σ²))·cos(2π·xr/λ+ψ) + dc
#   xr = (x−x0)cosθ + (y−y0)sinθ,  yr = −(x−x0)sinθ + (y−y0)cosθ
#   坐标约定与 gabor_frontend.py 一致（x 列向、y 行向，原点为核中心）。
#   参数变换（无约束 LBFGS 友好）：
#     x0,y0 = 1.5·tanh(·)；σ = 0.3+4·sigmoid(·)；λ = 1.5+5·sigmoid(·)；
#     γ = 0.2+1.3·sigmoid(·)；θ、ψ、amp、dc 自由。
# ================================================================================
_K = 5
_HALF = _K // 2
_YS, _XS = torch.meshgrid(torch.arange(-_HALF, _HALF + 1, dtype=torch.float64),
                          torch.arange(-_HALF, _HALF + 1, dtype=torch.float64),
                          indexing='ij')


def _logit(p):
    return math.log(p / (1.0 - p))


def _gabor_model(p):
    amp, dc, x0r, y0r, theta, sigr, lamr, gamr, psi = p
    x0 = 1.5 * torch.tanh(x0r)
    y0 = 1.5 * torch.tanh(y0r)
    sigma = 0.3 + 4.0 * torch.sigmoid(sigr)
    lam = 1.5 + 5.0 * torch.sigmoid(lamr)
    gamma = 0.2 + 1.3 * torch.sigmoid(gamr)
    xc = _XS - x0
    yc = _YS - y0
    ct, st = torch.cos(theta), torch.sin(theta)
    xr = xc * ct + yc * st
    yr = -xc * st + yc * ct
    envelope = torch.exp(-(xr ** 2 + gamma ** 2 * yr ** 2) / (2 * sigma ** 2))
    carrier = torch.cos(2 * math.pi * xr / lam + psi)
    return amp * envelope * carrier + dc


def fit_gabor(kernel_2d):
    """拟合单个 [k,k] 核，返回 dict(R2, theta, lam, sigma, gamma, psi, sse)。
    SST≈0（死核/平坦核）时 R2=0、theta=None。全程确定性（8 个固定起点）。"""
    y = kernel_2d.to(torch.float64)
    sst = ((y - y.mean()) ** 2).sum().item()
    if sst < 1e-12:
        return {"R2": 0.0, "theta": None, "lam": None, "sigma": None,
                "gamma": None, "psi": None, "sse": 0.0}

    y_std, y_mean = y.std().item(), y.mean().item()
    best = None
    for theta0 in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        for psi0 in (0.0, math.pi / 2):
            p = torch.tensor([y_std, y_mean, 0.0, 0.0, theta0,
                              _logit((2.0 - 0.3) / 4.0),   # σ0 = 2.0
                              _logit((3.0 - 1.5) / 5.0),   # λ0 = 3.0
                              _logit((0.5 - 0.2) / 1.3),   # γ0 = 0.5
                              psi0], dtype=torch.float64, requires_grad=True)
            opt = torch.optim.LBFGS([p], max_iter=200, history_size=10,
                                    line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                loss = ((_gabor_model(p) - y) ** 2).sum()
                loss.backward()
                return loss

            opt.step(closure)
            with torch.no_grad():
                sse = ((_gabor_model(p) - y) ** 2).sum().item()
            if best is None or sse < best[0]:
                best = (sse, p.detach().clone())

    sse, p = best
    r2 = max(0.0, 1.0 - sse / sst)
    amp, dc, x0r, y0r, theta, sigr, lamr, gamr, psi = [v.item() for v in p]
    return {
        "R2": r2,
        "theta": float(theta % math.pi),                    # 方向角归一到 [0, π)
        "lam": 1.5 + 5.0 / (1.0 + math.exp(-lamr)),
        "sigma": 0.3 + 4.0 / (1.0 + math.exp(-sigr)),
        "gamma": 0.2 + 1.3 / (1.0 + math.exp(-gamr)),
        "psi": float(psi % (2 * math.pi)),
        "sse": sse,
    }


# ================================================================================
# Mann-Whitney U 检验（手工实现：U 统计量 + 并列校正 + 连续性校正的正态近似，
# 双侧 p；n1=n2=32 时近似误差可忽略。Φ 用 math.erf）
# ================================================================================
def mannwhitneyu(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1, n2 = len(x), len(y)
    allv = np.concatenate([x, y])
    order = allv.argsort(kind='stable')
    ranks = np.empty(n1 + n2, dtype=np.float64)
    # 平均秩（处理并列）
    sv = allv[order]
    i = 0
    tie_sum = 0.0
    while i < n1 + n2:
        j = i
        while j + 1 < n1 + n2 and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0   # 1-based 平均秩
        t = j - i + 1
        if t > 1:
            tie_sum += t ** 3 - t
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2.0
    n = n1 + n2
    sigma = math.sqrt(n1 * n2 / 12.0 * (n + 1 - tie_sum / (n * (n - 1))))
    z = (u - mu + 0.5) / sigma                      # 连续性校正（u ≤ mu）
    p_two = math.erfc(abs(z) / math.sqrt(2.0))
    return {"U": float(u), "z": float(z), "p_two_sided": float(p_two)}


# ================================================================================
# 指标计算
# ================================================================================
def _median_iqr(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {"median": float(np.median(v)),
            "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
            "n": int(len(v))}


def analyze_group(name, W):
    """W: [C,1,k,k] torch 张量 → 拟合 + 稀疏度逐核结果"""
    fits, sparsity = [], []
    for c in range(W.shape[0]):
        k2d = W[c, 0]
        fits.append(fit_gabor(k2d))
        mx = k2d.abs().max().item()
        sparsity.append(float((k2d.abs() < 0.1 * mx).float().mean().item())
                        if mx > 0 else 1.0)
    r2s = [f["R2"] for f in fits]

    # 方向选择性：拟合成功核的 θ（倍角合成长度，θ∈[0,π) 为轴向量）
    good = [f for f in fits if f["theta"] is not None and f["R2"] > R2_THRESHOLD]
    thetas = [f["theta"] for f in good]
    if thetas:
        t = np.asarray(thetas)
        resultant = float(abs(np.exp(2j * t).mean()))
        hist = np.histogram(t % math.pi, bins=N_THETA_BINS,
                            range=(0.0, math.pi))[0].tolist()
    else:
        resultant, hist = 0.0, [0] * N_THETA_BINS

    return {
        "name": name,
        "r2": _median_iqr(r2s),
        "r2_values": [round(v, 4) for v in r2s],
        "n_good_fit_R2_gt_0.5": len(good),
        "orientation": {
            "n_kernels": len(thetas),
            "theta_deg": [round(math.degrees(v), 1) for v in thetas],
            "hist_6bins_0to180deg": hist,
            "resultant_vector_length": resultant,
        },
        "sparsity": _median_iqr(sparsity),
        "sparsity_values": [round(v, 4) for v in sparsity],
        "fits": [{k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in f.items()} for f in fits],
    }


# ================================================================================
# Figure 2：三方并排 grid
# ================================================================================
def make_figure(groups, havconn_exc, out_path):
    """groups: [(title, W [C,1,k,k])]; 按 havconn_exc 分 ON(上)/OFF(下) 两组排序。"""
    exc_idx = np.where(havconn_exc)[0]
    inh_idx = np.where(~havconn_exc)[0]
    order = np.concatenate([exc_idx, inh_idx])
    n_exc = len(exc_idx)
    rows, cols = 8, 4                        # 32 核 = 8 行 × 4 列（ON 4 行 + OFF 4 行）

    fig = plt.figure(figsize=(13.5, 10.0))
    outer = fig.add_gridspec(1, 3, wspace=0.22, left=0.06, right=0.98,
                             top=0.90, bottom=0.06)
    for gi, (title, W) in enumerate(groups):
        gs = outer[gi].subgridspec(rows, cols, hspace=0.10, wspace=0.10)
        Ws = W[order, 0].numpy()
        for i in range(rows * cols):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            k = Ws[i]
            v = np.abs(k).max()
            ax.imshow(k, cmap='RdBu_r', vmin=-v, vmax=v,
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if i == n_exc:                   # ON/OFF 分界：加粗上边框
                for s in ('top',):
                    ax.spines[s].set_linewidth(2.2)
                    ax.spines[s].set_color('black')
        # 面板标题与 ON/OFF 组标注
        ax_title = fig.add_subplot(gs[:])
        ax_title.set_frame_on(False)
        ax_title.set_xticks([]); ax_title.set_yticks([])
        ax_title.set_title(title, fontsize=13, pad=14)
        if gi == 0:
            ax_title.text(-0.16, 0.75, "ON", transform=ax_title.transAxes,
                          fontsize=12, fontweight='bold', va='center', rotation=90)
            ax_title.text(-0.16, 0.25, "OFF", transform=ax_title.transAxes,
                          fontsize=12, fontweight='bold', va='center', rotation=90)
        ax_title.text(0.5, -0.035,
                      f"top 4 rows: ON ({n_exc} ch) | bottom 4 rows: OFF "
                      f"({rows * cols - n_exc} ch)",
                      transform=ax_title.transAxes, fontsize=9,
                      ha='center', color='0.35')

    fig.suptitle("Conv front-end kernels (C=32, k=5) — per-kernel symmetric "
                 "normalization (red = positive, blue = negative)",
                 fontsize=14, y=0.965)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[viz_kernels] 图已写入 {out_path}")


# ================================================================================
# main
# ================================================================================
def main():
    assert Path.cwd() == REPO_ROOT, f"必须从仓库根目录运行（当前 {Path.cwd()}）"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- 核获取 ----------------
    w_init, mask_init = regenerate_random_init(SEED)
    w_b2, mask_b2 = load_b2_kernels()
    w_gabor = load_gabor_kernels()

    # 一致性自检：重建 init 必须逐元素等于 B1 存核（否则说明 RNG 复刻失效，报错）
    sd_b1 = torch.load(B1_MODEL, map_location='cpu', weights_only=True)['model_0']
    w_b1 = sd_b1['conv_layer.w.weight']
    diff = (w_init - w_b1).abs().max().item()
    assert diff == 0.0, f"重建 init 与 b1_500 存核不一致（max diff {diff}）"
    assert torch.equal(mask_init, mask_b2), "B1/B2 的 ON/OFF 掩码不一致"
    print(f"[viz_kernels] 自检通过：重建 init == b1_500 存核（max diff {diff}）；"
          f"B1/B2 ON/OFF 掩码一致（ON {int(mask_b2.sum())} / OFF {int((~mask_b2).sum())}）")

    groups = [
        ("Random init (seed 0, untrained = B1)", w_init),
        ("B2: Conv-STDP trained (2 ep)", w_b2),
        ("B5: hand-crafted Gabor bank", w_gabor),
    ]

    # ---------------- Figure 2 ----------------
    make_figure(groups, mask_b2.numpy(), RESULTS_DIR / "fig2_kernels.png")

    # ---------------- 量化三指标 ----------------
    stats_init = analyze_group("B1_random_init", w_init)
    stats_b2 = analyze_group("B2_conv_stdp_trained", w_b2)
    stats_gabor = analyze_group("B5_gabor_bank", w_gabor)

    mw_r2 = mannwhitneyu(stats_init["r2_values"], stats_b2["r2_values"])
    mw_sp = mannwhitneyu(stats_init["sparsity_values"], stats_b2["sparsity_values"])

    table = {
        "experiment": "S3.1 Exp 1.2 — kernel visualization & quantitative analysis (Figure 2)",
        "date": "2026-09-03",
        "seed": SEED,
        "models": {
            "B1_random_init": str(B1_MODEL.relative_to(REPO_ROOT))
                              + " (CPU-trained; regen init bit-identical, verified)",
            "B2_trained": str(B2_MODEL.relative_to(REPO_ROOT))
                          + " (GPU-trained; its CUDA-RNG init differs element-wise "
                            "from the CPU init — B1 serves as same-seed same-arch "
                            "untrained random control)",
            "B5_gabor": "gabor_frontend.gabor_kernel_bank(k=5), deterministic",
        },
        "gabor_fit": {
            "model": "amp*exp(-(xr^2+g^2*yr^2)/(2*sigma^2))*cos(2*pi*xr/lam+psi)+dc",
            "optimizer": "torch LBFGS, 8 deterministic starts (4 theta x 2 psi)",
            "B1_r2": stats_init["r2"],
            "B2_r2": stats_b2["r2"],
            "B5_r2_sanity": stats_gabor["r2"],
            "mannwhitney_B1_vs_B2": mw_r2,
        },
        "orientation_selectivity": {
            "r2_threshold": R2_THRESHOLD,
            "B1": stats_init["orientation"],
            "B2": stats_b2["orientation"],
            "B5_sanity": stats_gabor["orientation"],
        },
        "sparsity": {
            "definition": "per-kernel fraction of |w| < 0.1*max|w|",
            "B1": stats_init["sparsity"],
            "B2": stats_b2["sparsity"],
            "B5": stats_gabor["sparsity"],
            "mannwhitney_B1_vs_B2": mw_sp,
        },
        "per_kernel": {
            "B1": stats_init["fits"],
            "B2": stats_b2["fits"],
            "B5": stats_gabor["fits"],
        },
    }
    out_json = RESULTS_DIR / "table_kernel_stats.json"
    with open(out_json, "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print(f"[viz_kernels] 指标已写入 {out_json}")

    # ---------------- 控制台汇总 ----------------
    print("\n================ S3.1 实验 1.2 汇总 ================")
    print(f"Gabor R² 中位数 [IQR]  B1: {stats_init['r2']['median']:.3f} "
          f"[{stats_init['r2']['q25']:.3f}, {stats_init['r2']['q75']:.3f}] | "
          f"B2: {stats_b2['r2']['median']:.3f} "
          f"[{stats_b2['r2']['q25']:.3f}, {stats_b2['r2']['q75']:.3f}] | "
          f"B5(sanity): {stats_gabor['r2']['median']:.3f}")
    print(f"Mann-Whitney R²  B1 vs B2: U={mw_r2['U']:.1f}, "
          f"z={mw_r2['z']:.2f}, p={mw_r2['p_two_sided']:.4g}")
    print(f"拟合成功核数 (R²>{R2_THRESHOLD}): B1 {stats_init['orientation']['n_kernels']}"
          f"/32, B2 {stats_b2['orientation']['n_kernels']}/32")
    print(f"方向合成长度 R: B1 {stats_init['orientation']['resultant_vector_length']:.3f} | "
          f"B2 {stats_b2['orientation']['resultant_vector_length']:.3f} | "
          f"B5 {stats_gabor['orientation']['resultant_vector_length']:.3f}")
    print(f"方向直方图(6bin) B1: {stats_init['orientation']['hist_6bins_0to180deg']} "
          f"B2: {stats_b2['orientation']['hist_6bins_0to180deg']}")
    print(f"稀疏度中位数 [IQR]  B1: {stats_init['sparsity']['median']:.3f} "
          f"[{stats_init['sparsity']['q25']:.3f}, {stats_init['sparsity']['q75']:.3f}] | "
          f"B2: {stats_b2['sparsity']['median']:.3f} "
          f"[{stats_b2['sparsity']['q25']:.3f}, {stats_b2['sparsity']['q75']:.3f}]")
    print(f"Mann-Whitney 稀疏度 B1 vs B2: U={mw_sp['U']:.1f}, "
          f"z={mw_sp['z']:.2f}, p={mw_sp['p_two_sided']:.4g}")


if __name__ == "__main__":
    main()
