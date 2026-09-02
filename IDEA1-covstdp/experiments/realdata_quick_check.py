#!/usr/bin/env python
# ================================================================================
# realdata_quick_check.py — S2.3/S2.4 真数据快检（fork B 出口）
#
# 用 ~100 张真实 Nordland spring 图训练 ConvSNNLayer（centered, local WTA, ITP 全开），
# 验证核是否出现边缘/方向结构（最早的定性信号，比 S3.1 早一个阶段）。
#
# 输出：
#   results/s2324_kernels_quick.png  —— 训练前后 32 核 grid 对比
#   results/s2324_quick_metrics.json —— DC/AC、核间余弦、发放率曲线
#
# 数据说明：使用 Nordland spring 前 100 张子采样图（filter=8），
#           原图示例：spring/images-00202.png（640×360 铁轨场景）。
#
# 用法：pixi run python IDEA1-covstdp/experiments/realdata_quick_check.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import read_image
from vprtempo.src.dataset import ProcessImage

torch.manual_seed(0)
DEVICE = "cpu"

# ---- 数据：Nordland spring 前 100 张（filter=8 子采样）----
data_dir = json.loads((REPO_ROOT / "IDEA1-covstdp/phase1/configs/local_override.json").read_text())["data_dir"]
df = pd.read_csv(REPO_ROOT / "vprtempo/dataset/nordland-spring.csv")
proc = ProcessImage([28, 28], 7)
imgs = [proc(read_image(str(Path(data_dir) / "spring" / df.iloc[i, 0])).float()).view(1, -1)
        for i in range(0, 800, 8)]
print(f"训练图: {len(imgs)} 张（spring, filter=8）")

# ---- 训练 ----
layer = conv_mod.ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                              wta_mode='local', device=DEVICE)
W_init = layer.w.weight.data.clone().cpu()
hist = {"step": [], "dcac": [], "cos": [], "fire": []}
STEPS = 1000
for step in range(STEPS):
    x = imgs[step % len(imgs)]
    out = layer(layer.reshape_input(x))
    learn_mod.calc_stdp_conv(layer.reshape_input(x), out, layer, pre_mode='centered')
    learn_mod.apply_itp_conv(out, layer)
    if step % 100 == 0 or step == STEPS - 1:
        W = layer.w.weight.data
        flat = W.flatten(1)
        dcac = (flat.mean(dim=1).abs() / (flat.std(dim=1) + 1e-8)).mean().item()
        fn = torch.nn.functional.normalize(flat, dim=1)
        sim = (fn @ fn.T).abs()
        n = sim.shape[0]
        cos = ((sim.sum() - sim.diag().sum()) / (n * (n - 1))).item()
        hist["step"].append(step)
        hist["dcac"].append(dcac)
        hist["cos"].append(cos)
        hist["fire"].append((out.pre_wta > 0).float().mean().item())

print(f"DC/AC: {hist['dcac'][0]:.3f} → {hist['dcac'][-1]:.3f}")
print(f"核间余弦: {hist['cos'][0]:.3f} → {hist['cos'][-1]:.3f}")
print(f"发放率: {hist['fire'][0]:.3f} → {hist['fire'][-1]:.3f}")

# ---- 核可视化：训练前 vs 训练后（兴奋/抑制分组显示）----
W_after = layer.w.weight.data.cpu()
exc = layer.havconnExc.cpu()

def norm_vis(W):
    Wn = W.clone()
    for i in range(Wn.shape[0]):
        m = Wn[i].abs().max()
        if m > 0:
            Wn[i] = Wn[i] / m
    return Wn

fig, axes = plt.subplots(4, 16, figsize=(20, 5.5))
for r in range(4):
    for c in range(16):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
order = torch.cat([exc.nonzero().flatten(), (~exc).nonzero().flatten()])
for i, ch in enumerate(order):
    r, c = (i // 16) * 2, i % 16
    axes[r, c].imshow(norm_vis(W_init)[ch, 0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[r + 1, c].imshow(norm_vis(W_after)[ch, 0], cmap="RdBu_r", vmin=-1, vmax=1)
    if r == 0:
        axes[r, c].set_title(f"ch{ch}{'E' if exc[ch] else 'I'}", fontsize=7)
axes[0, 0].set_ylabel("init", fontsize=9)
axes[1, 0].set_ylabel("trained", fontsize=9)
axes[2, 0].set_ylabel("init", fontsize=9)
axes[3, 0].set_ylabel("trained", fontsize=9)
fig.suptitle("ConvSNNLayer kernels: init vs trained (1000 steps, 100 Nordland spring imgs) — E=兴奋/ON, I=抑制/OFF", fontsize=10)
fig.tight_layout()
out_png = REPO_ROOT / "IDEA1-covstdp" / "results" / "s2324_kernels_quick.png"
fig.savefig(out_png, dpi=110)
print(f"saved {out_png}")

out_json = REPO_ROOT / "IDEA1-covstdp" / "results" / "s2324_quick_metrics.json"
out_json.write_text(json.dumps(hist, indent=2))
print(f"saved {out_json}")
