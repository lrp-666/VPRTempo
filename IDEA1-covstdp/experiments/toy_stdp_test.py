#!/usr/bin/env python
# ================================================================================
# toy_stdp_test.py — S2.3/S2.4 验收（PLAN.md 玩具测试 5 断言 + 对拍）
#
# 结构：
#   Part A 对拍：calc_stdp_conv（向量化）vs calc_stdp_conv_reference（循环版）
#               逐元素一致（3 种 wta_mode × 2 种 agg_mode × 3 种 pre_mode 全组合）；
#   Part B 玩具测试 5 断言（合成数据 + 真实 Nordland 子集）：
#     1. 见过竖直亮条纹后，对应核在条纹列的权重大于两侧（学习信号生效）；
#     2. 无 winner 的通道不更新（M 构造/掩码正确）；
#     3. 真实图训 1000 步核范数不发散（归一化/自稳定生效）；
#     4. DC/AC 比 |mean(K)|/(std(K)+ε) 随训练下降（centered 必须满足；amp 对照预期不满足）；
#     5. 核间余弦相似度均值不上升（防塌缩到单一模板）。
#   诊断落盘：results/s2324_toy_test.json（锁定决策清单 #6）。
#
# 用法：pixi run python IDEA1-covstdp/experiments/toy_stdp_test.py
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

torch.manual_seed(0)
DEVICE = "cpu"
RESULTS = {}
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ================================================================================
# Part A: 对拍（向量化 vs 循环参考版）
# ================================================================================
print("=" * 60)
print("Part A: 对拍 calc_stdp_conv vs reference")
n_pairs = 0
for wta_mode in ['local', 'global', 'none']:
    for agg_mode in ['mean', 'sum']:
        for pre_mode in ['centered', 'amp', 'heaviside']:
            torch.manual_seed(hash((wta_mode, agg_mode, pre_mode)) % 2**31)
            l1 = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                              wta_mode=wta_mode, device=DEVICE)
            l2 = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                              wta_mode=wta_mode, device=DEVICE)
            l2.load_state_dict(l1.state_dict())
            x = torch.rand(1, 28 * 28)
            o1 = l1(l1.reshape_input(x))
            o2 = l2(l2.reshape_input(x))
            learn_mod.calc_stdp_conv(l1.reshape_input(x), o1, l1,
                                     pre_mode=pre_mode, agg_mode=agg_mode)
            learn_mod.calc_stdp_conv_reference(l2.reshape_input(x), o2, l2,
                                               pre_mode=pre_mode, agg_mode=agg_mode)
            same = torch.allclose(l1.w.weight.data, l2.w.weight.data, atol=1e-6)
            if not same:
                print(f"  MISMATCH @ {wta_mode}/{agg_mode}/{pre_mode}: "
                      f"max diff {(l1.w.weight.data - l2.w.weight.data).abs().max().item():.2e}")
            n_pairs += 1
            PASS.append(bool(same))
print(f"对拍 {n_pairs} 组全部一致: {all(PASS[-n_pairs:])}")


# ================================================================================
# Part B: 玩具测试 5 断言
# ================================================================================
print("=" * 60)
print("Part B: 玩具测试")

# ---- 断言 1：竖直亮条纹 → 核学到条纹结构 ----
# 合成图：背景 0.5（PatchNorm on 的平坦背景），中央竖直亮条纹 1.0
def make_stripe_img(col=14, bg=0.5, fg=1.0, noise=0.02):
    img = torch.full((28, 28), bg)
    img[:, col] = fg
    img += torch.randn(28, 28) * noise
    return img.clamp(0, 1).view(1, -1)

torch.manual_seed(1)
layer = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                     wta_mode='local', device=DEVICE)
layer.eta_stdp = torch.tensor(0.05, device=DEVICE)   # 玩具测试加速收敛（真实训练用默认 0.005 + 退火）
W_init = layer.w.weight.data.clone()
for step in range(1000):
    x = make_stripe_img(col=14)          # 固定位置条纹（PLAN：固定输入竖直边缘图）
    out = layer(layer.reshape_input(x))
    learn_mod.calc_stdp_conv(layer.reshape_input(x), out, layer, pre_mode='centered')
    learn_mod.apply_itp_conv(out, layer)

W_exc = layer.w.weight.data[layer.havconnExc]          # [n_exc, 1, 5, 5]
# 逐核计算"主列能量 / 边缘列能量"，取最强核（卡片语义："对应核在条纹位置权重显著增大"）
col_energy = W_exc.flatten(1).abs().view(W_exc.shape[0], 5, 5).mean(dim=1)  # [n_exc,5] 列能量
ratio_per_kernel = (col_energy.max(dim=1).values /
                    col_energy[:, [0, 4]].mean(dim=1).clamp(min=1e-8))
best_ratio = ratio_per_kernel.max().item()
best_kernel = ratio_per_kernel.argmax().item()
check("断言1: 最强兴奋核主列能量 > 2× 边缘列能量",
      best_ratio > 2.0, f"(ch{best_kernel} 比值 {best_ratio:.2f})")

# ---- 断言 2：无 winner 的通道不更新 ----
torch.manual_seed(2)
layer = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                     wta_mode='local', device=DEVICE)
x = torch.rand(1, 28 * 28)
out = layer(layer.reshape_input(x))
# 人为清空 ch0 的 winner_mask
out.winner_mask[0, 0] = False
W_before = layer.w.weight.data.clone()
learn_mod.calc_stdp_conv(layer.reshape_input(x), out, layer, pre_mode='centered')
check("断言2: winner_mask 清空的通道权重不变",
      torch.equal(W_before[0], layer.w.weight.data[0]))

# ---- 断言 3/4/5：真实 Nordland 子集（50 张）上的训练动力学 ----
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
    """每核 DC/AC 比 |mean|/(std+eps) 的均值（S2.3 断言 4 的正确指标）"""
    flat = W.flatten(1)
    return (flat.mean(dim=1).abs() / (flat.std(dim=1) + 1e-8)).mean().item()


def inter_kernel_cosine(W):
    flat = W.flatten(1)
    flat = torch.nn.functional.normalize(flat, dim=1)
    sim = (flat @ flat.T).abs()
    n = sim.shape[0]
    off = sim.sum() - sim.diag().sum()
    return (off / (n * (n - 1))).item()


def train_run(pre_mode, steps=1000):
    torch.manual_seed(3)
    layer = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                         wta_mode='local', device=DEVICE)
    hist = {"step": [], "nrm": [], "dcac": [], "cos": [], "fire": []}
    for step in range(steps):
        x = imgs[step % len(imgs)]
        out = layer(layer.reshape_input(x))
        learn_mod.calc_stdp_conv(layer.reshape_input(x), out, layer, pre_mode=pre_mode)
        learn_mod.apply_itp_conv(out, layer)
        if step % 100 == 0 or step == steps - 1:
            W = layer.w.weight.data
            hist["step"].append(step)
            hist["nrm"].append(torch.linalg.norm(W.flatten(1), ord=1, dim=1).mean().item())
            hist["dcac"].append(dc_ac(W))
            hist["cos"].append(inter_kernel_cosine(W))
            hist["fire"].append((out.pre_wta > 0).float().mean().item())
    return layer, hist


layer_c, hist_c = train_run('centered')
layer_a, hist_a = train_run('amp')
RESULTS['centered'] = hist_c
RESULTS['amp'] = hist_a

# 断言 3：核范数不发散（1000 步后 < 初值 × 1.5，且不塌缩到 0）
nrm_ratio = hist_c["nrm"][-1] / hist_c["nrm"][0]
check("断言3: centered 1000 步核范数稳定（0.5 < 比值 < 1.5）",
      0.5 < nrm_ratio < 1.5, f"(比值 {nrm_ratio:.3f})")

# 断言 4：centered 的 DC/AC 下降；amp 的 DC/AC 不下降（对照）
check("断言4a: centered DC/AC 随训练下降",
      hist_c["dcac"][-1] < hist_c["dcac"][0],
      f"({hist_c['dcac'][0]:.3f} → {hist_c['dcac'][-1]:.3f})")
print(f"[INFO] amp DC/AC: {hist_a['dcac'][0]:.3f} → {hist_a['dcac'][-1]:.3f}"
      f"（对照，预期不下降或降幅明显更小）")

# 断言 5：核间余弦不上升
check("断言5: centered 核间余弦不上升",
      hist_c["cos"][-1] <= hist_c["cos"][0] + 0.05,
      f"({hist_c['cos'][0]:.3f} → {hist_c['cos'][-1]:.3f})")

# ITP 有效性（S2.4 验收附带）：pre-WTA 发放率与目标 f 的秩相关（手写 Spearman，不依赖 scipy）
def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm() + 1e-12)).item()

fire_obs = []
torch.manual_seed(4)
for x in imgs[:50]:
    out = layer_c(layer_c.reshape_input(x))
    fire_obs.append((out.pre_wta > 0).float().mean(dim=(2, 3)).flatten())
fire_mean = torch.stack(fire_obs).mean(dim=0)
rho = spearman(fire_mean, layer_c.fire_rate.flatten())
RESULTS['itp_spearman'] = float(rho)
check("S2.4: pre-WTA 发放率与目标 f 秩相关显著为正", rho > 0.5, f"(ρ={rho:.3f})")

# 死通道信号（诊断信息）：winner 平均幅度
win_amp = []
for x in imgs[:20]:
    out = layer_c(layer_c.reshape_input(x))
    win_amp.append(out.winner_values.mean().item() if len(out.winner_values) else 0.0)
RESULTS['winner_mean_amp'] = sum(win_amp) / len(win_amp)
print(f"[INFO] 训练后 winner 平均幅度 = {RESULTS['winner_mean_amp']:.3f}（≈0 则报警）")

# ================================================================================
# 汇总 + 落盘
# ================================================================================
out_path = REPO_ROOT / "IDEA1-covstdp" / "results" / "s2324_toy_test.json"
RESULTS['pass_count'] = sum(PASS)
RESULTS['total'] = len(PASS)
out_path.write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
print(f"\n{'='*60}\n{sum(PASS)}/{len(PASS)} 断言通过，诊断已存 {out_path}")
if not all(PASS):
    sys.exit(1)
print("S2.3/S2.4 玩具测试全部通过 ✅")
