#!/usr/bin/env python
# ================================================================================
# diag_s211_round1.py — S2.11 规则锦标赛 Round 1 诊断（PLAN S2.11"每格留诊断 JSON"条
# + §8 锁定清单第 6 条诊断标准化）
#
# 对象：r0_wp / r1_bcm / r1_rank / r1_oja / r1_attractor（phase2/configs/，seed 0）
#
# Part A（训后核形态诊断，post-hoc 从 .pth 读核）：
#   每格训后核做 Gabor 拟合（复用 viz_kernels.analyze_group），报 R² 中位数/IQR、
#   拟合成功核数（R²>0.5）、方向合成长度、稀疏度、DC/AC 比分布 + 核间余弦；
#   另出五格并排核网格图 results/s211_round1_kernels.png（形态注记用）。
#
# Part B（训练动力学曲线，诊断性重放，口径同 diag_b6_kernels Part B）：
#   同配置同 seed 重放 conv 层训练（仅 conv 前端），每 25 步快照 §8 标准化字段：
#   pre-WTA 发放率、winner 平均幅度、核 L1 范数、thr 统计、DC/AC 比（中位数）、
#   核间余弦（非对角均值）；R1（bcm_gate）额外记录 θ_M 曲线（均值 + 逐通道）。
#   注意：重放的 RNG 消耗序列与真实训练不完全一致，曲线是同分布诊断轨迹，
#   不与存盘模型逐元素对应。
#
# 输出：results/s211_round1_diag.json + results/s211_round1_kernels.png
#
# 用法（必须从仓库根目录运行）：
#   pixi run --environment cuda python IDEA1-covstdp/experiments/diag_s211_round1.py
# ================================================================================
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_exp import load_config, RESULTS_DIR  # noqa: E402

CONFIGS = {
    "r0_wp": "IDEA1-covstdp/phase2/configs/r0_wp.json",
    "r1_bcm": "IDEA1-covstdp/phase2/configs/r1_bcm.json",
    "r1_rank": "IDEA1-covstdp/phase2/configs/r1_rank.json",
    "r1_oja": "IDEA1-covstdp/phase2/configs/r1_oja.json",
    "r1_attractor": "IDEA1-covstdp/phase2/configs/r1_attractor.json",
}
SNAP_EVERY = 25
SEED = 0


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


viz = _load("idea1_viz_kernels", "IDEA1-covstdp/experiments/viz_kernels.py")


def dc_ac(K):
    """每核 DC/AC 比 |mean(K)| / (std(K) + eps)（PLAN S2.3 直流防线定义）"""
    kf = K.flatten(1).double()
    return (kf.mean(dim=1).abs() / (kf.std(dim=1) + 1e-12))


def pairwise_cos(K):
    """核间余弦（非对角均值，绝对值）"""
    kf = torch.nn.functional.normalize(K.flatten(1).double(), dim=1)
    csm = (kf @ kf.T).abs()
    n = csm.shape[0]
    return float((csm.sum() - csm.diag().sum()) / (n * (n - 1)))


def load_trained_kernels(exp_id, seed=SEED):
    p = REPO_ROOT / "vprtempo" / "models" / f"{exp_id}__seed{seed}.pth"
    sd = torch.load(p, map_location="cpu", weights_only=True)["model_0"]
    return sd["conv_layer.w.weight"].detach().clone(), \
        sd["conv_layer.havconnExc"].clone()


# ================================================================================
# Part A：训后核形态统计 + 并排核网格图
# ================================================================================
def make_grid_figure(groups, mask, out_path):
    """groups: [(title, W [C,1,k,k])]；按 ON/OFF 分组排序，单行长条并排。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exc_idx = np.where(mask)[0]
    inh_idx = np.where(~mask)[0]
    order = np.concatenate([exc_idx, inh_idx])
    n_exc = len(exc_idx)
    rows, cols = 8, 4
    n_groups = len(groups)

    fig = plt.figure(figsize=(4.6 * n_groups, 9.5))
    outer = fig.add_gridspec(1, n_groups, wspace=0.22, left=0.04, right=0.985,
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
            if i == n_exc:
                ax.spines['top'].set_linewidth(2.2)
                ax.spines['top'].set_color('black')
        ax_title = fig.add_subplot(gs[:])
        ax_title.set_frame_on(False)
        ax_title.set_xticks([]); ax_title.set_yticks([])
        ax_title.set_title(title, fontsize=12, pad=12)

    fig.suptitle("S2.11 Round 1 trained kernels (C=32, k=5; top 4 rows ON / "
                 "bottom 4 rows OFF) — per-kernel symmetric normalization",
                 fontsize=13, y=0.965)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[PartA] 核网格图已写入 {out_path}")


def part_a():
    groups = {}
    fig_groups = []
    mask = None
    for exp_id in CONFIGS:
        W, mask = load_trained_kernels(exp_id)
        stats = viz.analyze_group(exp_id, W)
        da = dc_ac(W)
        stats["dc_ac"] = {"median": float(da.median()),
                          "q25": float(da.quantile(0.25)),
                          "q75": float(da.quantile(0.75)),
                          "values": [round(v, 4) for v in da.tolist()]}
        stats["pairwise_cos_mean"] = pairwise_cos(W)
        stats["kernel_l1_median"] = float(W.flatten(1).abs().sum(dim=1).median())
        groups[exp_id] = stats
        fig_groups.append((exp_id, W))
        print(f"[PartA] {exp_id}: R² median={stats['r2']['median']:.3f} "
              f"[{stats['r2']['q25']:.3f},{stats['r2']['q75']:.3f}], "
              f"good(R²>0.5)={stats['n_good_fit_R2_gt_0.5']}/{W.shape[0]}, "
              f"DC/AC median={stats['dc_ac']['median']:.3f}, "
              f"cos={stats['pairwise_cos_mean']:.3f}")

    make_grid_figure(fig_groups, mask.numpy(),
                     RESULTS_DIR / "s211_round1_kernels.png")
    return groups


# ================================================================================
# Part B：训练动力学曲线（诊断性重放，含 R1 的 θ_M 曲线）
# ================================================================================
def replay_curves(exp_id, device):
    import random

    from torchvision import transforms

    from vprtempo.src import conv_frontend as cf
    from vprtempo.src.dataset import CustomImageDataset, ProcessImage

    cfg = load_config(str(REPO_ROOT / CONFIGS[exp_id]), SEED)
    dims = [int(v) for v in cfg["dims"].split(",")]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    ns = argparse.Namespace(**cfg)
    layer = cf.build_conv_layer(ns, dims, device, inference=False)

    image_transform = transforms.Compose([
        ProcessImage(dims, cfg["patches"],
                     patch_norm=cfg.get("patch_norm", "on") == "on")])
    db_csvs = [os.path.join("./vprtempo/dataset", f'{cfg["dataset"]}-{d}.csv')
               for d in cfg["database_dirs"].split(",")]
    dataset = CustomImageDataset(
        annotations_file=db_csvs, base_dir=cfg["data_dir"],
        img_dirs=cfg["database_dirs"].split(","), transform=image_transform,
        filter=cfg["filter"], skip=cfg["skip"], test=False,
        img_range=[0, (cfg["max_module"] - 1) * cfg["filter"]],
        max_samples=cfg["database_places"])
    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4,
                        persistent_workers=False, generator=g)

    n_imgs = len(dataset)
    conv_epoch = int(cfg.get("conv_epoch", 2))
    T = int(n_imgs * conv_epoch)
    init_itp = layer.eta_ip.detach().clone()
    init_stdp = layer.eta_stdp.detach().clone()

    curve = {"steps": [], "fire_rate_pre_wta": [], "winner_mean_amp": [],
             "kernel_l1_median": [], "thr_mean": [], "dc_ac_median": [],
             "pairwise_cos_mean": []}
    has_theta = hasattr(layer, 'theta_m')
    if has_theta:
        curve["theta_m_mean"] = []
        curve["theta_m_per_channel"] = []
    mod = 0
    for _ in range(conv_epoch):
        for spikes, _ in loader:
            spikes = spikes.to(device)
            x = layer.reshape_input(spikes)
            out = layer(x)
            cf.calc_stdp_conv(x, out, layer,
                              pre_mode=cfg.get("pre_mode", "centered"),
                              agg_mode=cfg.get("agg_mode", "mean"))
            cf.apply_itp_conv(out, layer)
            if mod % 100 == 0:
                pt = pow(float(T - mod) / T, 2)
                layer.eta_ip = torch.mul(init_itp, pt)
                layer.eta_stdp = torch.mul(init_stdp, pt)
            if mod % SNAP_EVERY == 0 or mod == T - 1:
                with torch.no_grad():
                    K = layer.w.weight.data
                    da = dc_ac(K)
                    curve["steps"].append(mod)
                    curve["fire_rate_pre_wta"].append(
                        float((out.pre_wta > 0).float().mean()))
                    curve["winner_mean_amp"].append(
                        float(out.winner_values.mean()) if out.winner_values.numel()
                        else 0.0)
                    curve["kernel_l1_median"].append(
                        float(K.flatten(1).abs().sum(dim=1).median()))
                    curve["thr_mean"].append(float(layer.thr.data.mean()))
                    curve["dc_ac_median"].append(float(da.median()))
                    curve["pairwise_cos_mean"].append(pairwise_cos(K))
                    if has_theta:
                        curve["theta_m_mean"].append(float(layer.theta_m.mean()))
                        curve["theta_m_per_channel"].append(
                            [round(v, 5) for v in
                             layer.theta_m.flatten().tolist()])
            mod += 1
    curve["n_steps"] = T
    curve["dc_ac_final_per_kernel"] = [round(v, 4) for v in
                                       dc_ac(layer.w.weight.data).tolist()]
    return curve


def part_b(device):
    curves = {}
    for exp_id in CONFIGS:
        t0 = time.time()
        curves[exp_id] = replay_curves(exp_id, device)
        c = curves[exp_id]
        extra = (f" | θ_M {c['theta_m_mean'][0]:.4f} → {c['theta_m_mean'][-1]:.4f}"
                 if "theta_m_mean" in c else "")
        print(f"[PartB] {exp_id}: {c['n_steps']} 步重放 {time.time() - t0:.0f}s | "
              f"DC/AC {c['dc_ac_median'][0]:.3f} → {c['dc_ac_median'][-1]:.3f} | "
              f"发放率 {c['fire_rate_pre_wta'][0]:.3f} → "
              f"{c['fire_rate_pre_wta'][-1]:.3f}{extra}")
    return curves


def main():
    ap = argparse.ArgumentParser(description="S2.11 Round 1 诊断")
    ap.add_argument("--skip-curves", action="store_true", help="只跑 Part A")
    ap.add_argument("--skip-kernels", action="store_true", help="只跑 Part B")
    cli = ap.parse_args()

    groups = part_a() if not cli.skip_kernels else None
    curves = None
    if not cli.skip_curves:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        curves = part_b(device)

    out = RESULTS_DIR / "s211_round1_diag.json"
    payload = {"experiment": "S2.11 Round 1 diagnostics "
                             "(kernel morphology + training replay curves; "
                             "replay caveat: 与存盘模型不逐元素对应)",
               "seed": SEED, "snap_every": SNAP_EVERY}
    if groups is not None:
        payload["kernel_stats"] = groups
    if curves is not None:
        payload["curves"] = curves
    # 分段运行时合并已有文件，避免互相覆盖
    if out.exists():
        with open(out) as f:
            old = json.load(f)
        old.update(payload)
        payload = old
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[diag] 已写入 {out}")


if __name__ == "__main__":
    main()
