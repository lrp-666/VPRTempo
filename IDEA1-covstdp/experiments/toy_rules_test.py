#!/usr/bin/env python
# ================================================================================
# toy_rules_test.py — S2.11 规则锦标赛 Round 1 验收（PLAN.md S2.11 预注册判据第 1 条）
#
# 四个规则手术开关（bcm_gate / rank_push / oja_decay / attractor，默认全关 = B2 不变）
# 进真数据前的门槛测试，断言口径与 toy_stdp_test.py 完全一致：
#
#   Part A 对拍：calc_stdp_conv（向量化）vs calc_stdp_conv_reference（循环版）
#               逐元素一致 —— 每个开关单独开启 + 四开关全开的组合，
#               × agg_mode {mean,sum} × pre_mode {centered,amp,heaviside}
#               （rank_push 仅 local WTA 有定义，global/none 下跳过该开关组合）。
#               R4（attractor）的弹性项 −(ΣM_c)·K_c 在此与朴素循环版对拍。
#   Part B 玩具断言（真实 Nordland 子集 50 张 × 1000 步，同 toy_stdp_test Part B）：
#     1. 核范数不发散（0.5 < 末/初比值 < 1.5）；
#     2. DC/AC 比随训练下降；
#     3. 核间余弦不上升（+0.05 容差）。
#     每个变体（含 R0 全关对照）独立跑一遍。
#
#   诊断落盘：results/s211_toy_rules.json
#
# 用法：pixi run python IDEA1-covstdp/experiments/toy_rules_test.py
# ================================================================================
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"
sys.path.insert(0, str(REPO_ROOT))

import importlib.util

def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

conv_mod = load_mod("conv_snn_layer", "IDEA1-covstdp/src/conv_snn_layer.py")
learn_mod = load_mod("conv_learning", "IDEA1-covstdp/src/conv_learning.py")
ConvSNNLayer = conv_mod.ConvSNNLayer

DEVICE = "cpu"
RESULTS = {}
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ================================================================================
# Part A: 对拍（向量化 vs 循环参考版）—— 开关单开 + 全开组合
# ================================================================================
print("=" * 60)
print("Part A: 对拍 calc_stdp_conv vs reference（S2.11 开关）")

SWITCH_SETS = {
    "bcm":       {"bcm_gate": True},
    "rank":      {"rank_push": True},
    "oja":       {"oja_decay": True},
    "attractor": {"attractor": True},
    "all":       {"bcm_gate": True, "rank_push": True,
                  "oja_decay": True, "attractor": True},
}

n_pairs = 0
for sw_name, sw in SWITCH_SETS.items():
    wta_modes = ['local'] if sw.get("rank_push") else ['local', 'global', 'none']
    for wta_mode in wta_modes:
        for agg_mode in ['mean', 'sum']:
            for pre_mode in ['centered', 'amp', 'heaviside']:
                torch.manual_seed(hash((sw_name, wta_mode, agg_mode, pre_mode)) % 2**31)
                l1 = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                                  wta_mode=wta_mode, device=DEVICE, **sw)
                l2 = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                                  wta_mode=wta_mode, device=DEVICE, **sw)
                l2.load_state_dict(l1.state_dict())
                x = torch.rand(1, 28 * 28)
                o1 = l1(l1.reshape_input(x))
                o2 = l2(l2.reshape_input(x))
                learn_mod.calc_stdp_conv(l1.reshape_input(x), o1, l1,
                                         pre_mode=pre_mode, agg_mode=agg_mode)
                learn_mod.calc_stdp_conv_reference(l2.reshape_input(x), o2, l2,
                                                   pre_mode=pre_mode, agg_mode=agg_mode)
                same = torch.allclose(l1.w.weight.data, l2.w.weight.data, atol=1e-6)
                if sw.get("bcm_gate"):  # θ_M 同步更新，也应一致
                    same = same and torch.allclose(l1.theta_m, l2.theta_m, atol=1e-9)
                if not same:
                    print(f"  MISMATCH @ {sw_name}/{wta_mode}/{agg_mode}/{pre_mode}: "
                          f"max diff {(l1.w.weight.data - l2.w.weight.data).abs().max().item():.2e}")
                n_pairs += 1
                PASS.append(bool(same))
print(f"对拍 {n_pairs} 组全部一致: {all(PASS[-n_pairs:])}")


# ================================================================================
# Part B: 玩具断言（真实 Nordland 子集，口径同 toy_stdp_test Part B）
# ================================================================================
print("=" * 60)
print("Part B: 真实数据子集训练动力学（每变体独立）")

import pandas as pd
from torchvision.io import read_image
from vprtempo.src.dataset import ProcessImage

data_dir = json.loads((REPO_ROOT / "IDEA1-covstdp/phase1/configs/local_override.json").read_text())["data_dir"]
df = pd.read_csv(REPO_ROOT / "vprtempo/dataset/nordland-spring.csv")
proc = ProcessImage([28, 28], 7)
imgs = []
for i in range(0, 400, 8):  # 对齐仓库 filter=8 的子采样
    img = read_image(str(Path(data_dir) / "spring" / df.iloc[i, 0])).float()
    imgs.append(proc(img).view(1, -1))
print(f"真实图子集: {len(imgs)} 张（spring, filter=8）")


def dc_ac(W):
    """每核 DC/AC 比 |mean|/(std+eps) 的均值"""
    flat = W.flatten(1)
    return (flat.mean(dim=1).abs() / (flat.std(dim=1) + 1e-8)).mean().item()


def inter_kernel_cosine(W):
    flat = W.flatten(1)
    flat = torch.nn.functional.normalize(flat, dim=1)
    sim = (flat @ flat.T).abs()
    n = sim.shape[0]
    off = sim.sum() - sim.diag().sum()
    return (off / (n * (n - 1))).item()


def train_run(steps=1000, **sw):
    torch.manual_seed(3)
    layer = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                         wta_mode='local', device=DEVICE, **sw)
    hist = {"step": [], "nrm": [], "dcac": [], "cos": [], "fire": []}
    if sw.get("bcm_gate"):
        hist["theta_m"] = []
    for step in range(steps):
        x = imgs[step % len(imgs)]
        out = layer(layer.reshape_input(x))
        learn_mod.calc_stdp_conv(layer.reshape_input(x), out, layer, pre_mode='centered')
        learn_mod.apply_itp_conv(out, layer)
        if step % 100 == 0 or step == steps - 1:
            W = layer.w.weight.data
            hist["step"].append(step)
            hist["nrm"].append(torch.linalg.norm(W.flatten(1), ord=1, dim=1).mean().item())
            hist["dcac"].append(dc_ac(W))
            hist["cos"].append(inter_kernel_cosine(W))
            hist["fire"].append((out.pre_wta > 0).float().mean().item())
            if sw.get("bcm_gate"):
                hist["theta_m"].append(layer.theta_m.mean().item())
    return layer, hist


VARIANTS = {
    "r0_off":      {},
    "r1_bcm":      {"bcm_gate": True},
    "r1_rank":     {"rank_push": True},
    "r1_oja":      {"oja_decay": True},
    "r1_attractor": {"attractor": True},
}

for name, sw in VARIANTS.items():
    layer, hist = train_run(**sw)
    RESULTS[name] = hist
    nrm_ratio = hist["nrm"][-1] / hist["nrm"][0]
    check(f"{name} 断言1: 1000 步核范数稳定（0.5 < 比值 < 1.5）",
          0.5 < nrm_ratio < 1.5, f"(比值 {nrm_ratio:.3f})")
    check(f"{name} 断言2: DC/AC 随训练下降",
          hist["dcac"][-1] < hist["dcac"][0],
          f"({hist['dcac'][0]:.3f} → {hist['dcac'][-1]:.3f})")
    check(f"{name} 断言3: 核间余弦不上升",
          hist["cos"][-1] <= hist["cos"][0] + 0.05,
          f"({hist['cos'][0]:.3f} → {hist['cos'][-1]:.3f})")
    if "theta_m" in hist:
        print(f"[INFO] {name} θ_M: {hist['theta_m'][0]:.4f} → {hist['theta_m'][-1]:.4f}")

# ================================================================================
# 汇总 + 落盘
# ================================================================================
out_path = REPO_ROOT / "IDEA1-covstdp" / "results" / "s211_toy_rules.json"
RESULTS['pass_count'] = sum(PASS)
RESULTS['total'] = len(PASS)
out_path.write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
print(f"\n{'='*60}\n{sum(PASS)}/{len(PASS)} 断言通过，诊断已存 {out_path}")
if not all(PASS):
    sys.exit(1)
print("S2.11 Round 1 玩具测试全部通过 ✅")
