#!/usr/bin/env python
# ================================================================================
# diag_b6_kernels.py — S2.10 / S3.3-8 诊断（PLAN S2.10"诊断防线"条 + §8 清单第 6 条）
#
# Part A（训后核结构诊断，post-hoc 从 .pth 读核）：
#   对 b1_500_block2 / b2_500_block2 / b6a_500 / b6b_500 / freesign_500（seed 0）
#   的训后核做 Gabor 拟合（复用 viz_kernels.fit_gabor），报 R² 中位数/IQR、
#   拟合成功核数（R²>0.5）、方向统计、稀疏度、DC/AC 比；B5 手工组作 sanity。
#   B6a 的核取 ON/OFF 联合核 J = w_on + w_off（= 训前 G，带符号），
#   拟合在 16 个联合核上进行；其余组直接拟合 32 个核。
#   → results/s210_b6_kernel_diag.json
#
# Part B（DC/AC 曲线，诊断性重放）：
#   以同配置同 seed 重放 conv 层训练（仅 conv 前端，不训 feature/output 层），
#   每 25 步快照 §8 标准化字段：pre-WTA 发放率、winner 平均幅度、核 L1 范数、
#   thr 统计、DC/AC 比（中位数）、核间余弦（非对角均值）。
#   注意：重放的 RNG 消耗序列与真实训练不完全一致（模型构造还消耗了别的随机数），
#   曲线是同一分布下的诊断轨迹，不与存盘模型逐元素对应。
#   → results/s210_b6_dcac_curves.json
#
# 用法（必须从仓库根目录运行）：
#   pixi run --environment cuda python IDEA1-covstdp/experiments/diag_b6_kernels.py
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
    "b1_500_block2": "IDEA1-covstdp/phase2/configs/b1_500_block2.json",
    "b2_500_block2": "IDEA1-covstdp/phase2/configs/b2_500_block2.json",
    "b6a_500": "IDEA1-covstdp/phase2/configs/b6a_500.json",
    "b6b_500": "IDEA1-covstdp/phase2/configs/b6b_500.json",
    "freesign_500": "IDEA1-covstdp/phase2/configs/freesign_500.json",
}
# Part B 曲线重放组（b1 权重冻结无曲线意义，排除）
CURVE_EXPS = ["b2_500_block2", "b6a_500", "b6b_500", "freesign_500"]
SNAP_EVERY = 25
SEED = 0


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


viz = _load("idea1_viz_kernels", "IDEA1-covstdp/experiments/viz_kernels.py")


# ================================================================================
# 核的"带符号化"约定：B6a 取 ON/OFF 联合核 J = w_on + w_off（前 16 ON / 后 16 OFF，
# 确定性配对，见 gabor_frontend.load_gabor_weights_decomposed）；其余直接用原始核
# ================================================================================
def signed_kernels(exp_id, W):
    if exp_id == "b6a_500":
        n = W.shape[0] // 2
        return W[:n] + W[n:]
    return W


def dc_ac(K):
    """每核 DC/AC 比 |mean(K)| / (std(K) + eps)（PLAN S2.3 直流防线定义）"""
    kf = K.flatten(1).double()
    return (kf.mean(dim=1).abs() / (kf.std(dim=1) + 1e-12))


def load_trained_kernels(exp_id, seed=SEED):
    p = REPO_ROOT / "vprtempo" / "models" / f"{exp_id}__seed{seed}.pth"
    sd = torch.load(p, map_location="cpu", weights_only=True)["model_0"]
    return sd["conv_layer.w.weight"].detach().clone(), \
        sd["conv_layer.havconnExc"].clone()


# ================================================================================
# Part A：训后核 Gabor 拟合 R² + DC/AC
# ================================================================================
def part_a():
    groups = {}
    for exp_id in CONFIGS:
        W, mask = load_trained_kernels(exp_id)
        K = signed_kernels(exp_id, W)
        stats = viz.analyze_group(exp_id, K)
        da = dc_ac(K)
        stats["dc_ac"] = {"median": float(da.median()),
                          "q25": float(da.quantile(0.25)),
                          "q75": float(da.quantile(0.75)),
                          "values": [round(v, 4) for v in da.tolist()]}
        if exp_id == "b6a_500":
            # B6a 配对 sanity：state_dict 中的 havconnExc 必须是前 16 ON / 后 16 OFF
            n = W.shape[0] // 2
            assert bool(mask[:n].all()) and bool((~mask[n:]).all()), \
                "b6a_500 存盘模型的 ON/OFF 配对排布异常"
        groups[exp_id] = stats
        print(f"[PartA] {exp_id}: R² median={stats['r2']['median']:.3f} "
              f"[{stats['r2']['q25']:.3f},{stats['r2']['q75']:.3f}], "
              f"good(R²>0.5)={stats['n_good_fit_R2_gt_0.5']}/{K.shape[0]}, "
              f"DC/AC median={stats['dc_ac']['median']:.3f}")

    # B5 手工组 sanity（确定性，无需模型文件）
    gf = _load("idea1_gabor_frontend", "IDEA1-covstdp/src/gabor_frontend.py")
    bank = gf.gabor_kernel_bank(kernel_size=5, device="cpu")
    stats_b5 = viz.analyze_group("B5_gabor_bank", bank)
    groups["b5_gabor_bank"] = stats_b5
    print(f"[PartA] B5 sanity: R² median={stats_b5['r2']['median']:.3f}")

    out = RESULTS_DIR / "s210_b6_kernel_diag.json"
    with open(out, "w") as f:
        json.dump({"experiment": "S2.10/S3.3-8 kernel diagnostics",
                   "seed": SEED, "groups": groups}, f, indent=2, ensure_ascii=False)
    print(f"[PartA] 已写入 {out}")
    return groups


# ================================================================================
# Part B：DC/AC 等 §8 字段的训练曲线（诊断性重放）
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
                    K = signed_kernels(exp_id, layer.w.weight.data)
                    da = dc_ac(K)
                    kf = K.flatten(1)
                    kf_n = torch.nn.functional.normalize(kf, dim=1)
                    csm = kf_n @ kf_n.T
                    off = csm[~torch.eye(csm.shape[0], dtype=bool,
                                         device=csm.device)].mean()
                    curve["steps"].append(mod)
                    curve["fire_rate_pre_wta"].append(
                        float((out.pre_wta > 0).float().mean()))
                    curve["winner_mean_amp"].append(
                        float(out.winner_values.mean()) if out.winner_values.numel()
                        else 0.0)
                    curve["kernel_l1_median"].append(
                        float(kf.abs().sum(dim=1).median()))
                    curve["thr_mean"].append(float(layer.thr.data.mean()))
                    curve["dc_ac_median"].append(float(da.median()))
                    curve["pairwise_cos_mean"].append(float(off))
            mod += 1
    curve["n_steps"] = T
    curve["dc_ac_final_per_kernel"] = [round(v, 4) for v in
                                       dc_ac(signed_kernels(
                                           exp_id, layer.w.weight.data)).tolist()]
    return curve


def part_b(device):
    curves = {}
    for exp_id in CURVE_EXPS:
        t0 = time.time()
        curves[exp_id] = replay_curves(exp_id, device)
        c = curves[exp_id]
        print(f"[PartB] {exp_id}: {c['n_steps']} 步重放 {time.time() - t0:.0f}s | "
              f"DC/AC {c['dc_ac_median'][0]:.3f} → {c['dc_ac_median'][-1]:.3f} | "
              f"发放率 {c['fire_rate_pre_wta'][0]:.3f} → "
              f"{c['fire_rate_pre_wta'][-1]:.3f}")
    out = RESULTS_DIR / "s210_b6_dcac_curves.json"
    with open(out, "w") as f:
        json.dump({"experiment": "S2.10/S3.3-8 DC/AC + §8 diagnostic curves "
                                 "(diagnostic replay, see header caveat)",
                   "seed": SEED, "snap_every": SNAP_EVERY,
                   "curves": curves}, f, indent=2, ensure_ascii=False)
    print(f"[PartB] 已写入 {out}")


def main():
    ap = argparse.ArgumentParser(description="S2.10/S3.3-8 核诊断 + DC/AC 曲线")
    ap.add_argument("--skip-curves", action="store_true", help="只跑 Part A")
    cli = ap.parse_args()

    part_a()
    if not cli.skip_curves:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        part_b(device)


if __name__ == "__main__":
    main()
