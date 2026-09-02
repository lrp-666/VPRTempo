#!/usr/bin/env python
# ================================================================================
# sanity_conv_forward.py — ConvSNNLayer 前向 + WTA 的验收脚本（PLAN.md S2.1/S2.2）
#
# 完成定义（两张卡片的验收全过才算 fork A 完成）：
#   1. 形状断言：28×28/k=5 → pre_wta [1,32,24,24]、pooled_flat [1,1152]；
#      56×56 也要能构造（ADR-1 维度纪律）；
#   2. 发放比例合理：不恒 0、不恒饱和；
#   3. 初始化约束：兴奋核逐元素 ≥0、抑制核 ≤0、每核 L1 ≈ 1；
#   4. WTA 三模式非零计数断言（global==1、local==36@28×28k5、none 不变）；
#   5. 真实 Nordland 图的 mask 前后 feature map 对比图 → results/wta_mask_demo.png。
#
# 用法：pixi run python IDEA1-covstdp/experiments/sanity_conv_forward.py
# 注意：必须从仓库根目录运行（vprtempo 的相对路径约定）。
# ================================================================================
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"

# 直接按路径加载（目录名含连字符，不能用常规 import 语法）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "conv_snn_layer", REPO_ROOT / "IDEA1-covstdp" / "src" / "conv_snn_layer.py")
conv_snn_layer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv_snn_layer)
ConvSNNLayer = conv_snn_layer.ConvSNNLayer

torch.manual_seed(0)  # 断言可复现
DEVICE = "cpu"
PASS = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    PASS.append(cond)
    print(f"[{status}] {name}")


# ================================================================================
# 1. 构造与形状（28×28 主配置 + 56×56 对照配置）
# ================================================================================
layer = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                     wta_mode='local', device=DEVICE)
x_flat = torch.rand(1, 28 * 28, device=DEVICE)          # 模拟 spike 编码后的平向量
out = layer(layer.reshape_input(x_flat))

check("28x28: pre_wta 形状 [1,32,24,24]", tuple(out.pre_wta.shape) == (1, 32, 24, 24))
check("28x28: pooled_flat 形状 [1,1152]", tuple(out.pooled_flat.shape) == (1, 1152))
check("28x28: flat_dim 元数据一致", layer.flat_dim == 1152)

layer56 = ConvSNNLayer(input_dims=[56, 56], out_channels=32, kernel_size=5,
                       wta_mode='local', device=DEVICE)
out56 = layer56(layer56.reshape_input(torch.rand(1, 56 * 56, device=DEVICE)))
check("56x56: pre_wta 形状 [1,32,52,52]", tuple(out56.pre_wta.shape) == (1, 32, 52, 52))
check("56x56: pooled_flat 形状 [1,5408]", tuple(out56.pooled_flat.shape) == (1, 5408))

# ================================================================================
# 2. 发放比例合理（不恒 0、不恒饱和）
# ================================================================================
frac_pos = (out.pre_wta > 0).float().mean().item()
frac_sat = (out.pre_wta >= 0.9).float().mean().item()
check(f"发放比例 0 < {frac_pos:.3f} < 1", 0.0 < frac_pos < 1.0)
check(f"饱和比例 {frac_sat:.3f} < 0.5", frac_sat < 0.5)

# ================================================================================
# 3. 初始化约束（通道级 E/I + 每核 L1 归一化）
# ================================================================================
W = layer.w.weight.data
exc = layer.havconnExc
check(f"兴奋通道数 = 16（p_exc=0.5 × 32）", int(exc.sum()) == 16)
check("兴奋核逐元素 >= 0", bool((W[exc] >= 0).all()))
check("抑制核逐元素 <= 0", bool((W[~exc] <= 0).all()))
l1 = torch.linalg.norm(W.flatten(1), ord=1, dim=1)
check("每核 L1 范数 ≈ 1", bool(torch.allclose(l1, torch.ones_like(l1), atol=1e-5)))

# ================================================================================
# 4. WTA 三模式断言
# ================================================================================
# 结构性断言数 winner_mask 的计数（掩码逻辑）；零值 winner（块内全零时 argmax
# 选出的也是 0）只作诊断信息——(post>0) 计数会低估，结构性 winner 数应看 winner_mask。
# ================================================================================
def winner_count_per_channel(wta_mode):
    l = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                     wta_mode=wta_mode, device=DEVICE)
    o = l(l.reshape_input(x_flat))
    return o.winner_mask.float().sum(dim=(2, 3)), o

cnt_global, _ = winner_count_per_channel('global')
check("global: 每通道 winner_mask 计数 == 1", bool((cnt_global == 1).all()))

cnt_local, out_local = winner_count_per_channel('local')
check("local: 每通道 winner_mask 计数 == 36（24/4 × 24/4）", bool((cnt_local == 36).all()))

# 零值 winner 比例（块内全零时 argmax 选出的也是 0——诊断信息，不作硬断言）
l_local = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                       wta_mode='local', device=DEVICE)
o_local = l_local(l_local.reshape_input(x_flat))
zero_winner_frac = (o_local.winner_values == 0).float().mean().item()
print(f"[INFO] local 模式零值 winner 比例 = {zero_winner_frac:.3f}（死块比例，S2.4 死通道监控的信号之一）")

l_none = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                      wta_mode='none', device=DEVICE)
o_none = l_none(l_none.reshape_input(x_flat))
check("none: post_wta 与 pre_wta 相同", bool(torch.equal(o_none.post_wta, o_none.pre_wta)))
check("none: pooled_flat 形状 [1,1152]", tuple(o_none.pooled_flat.shape) == (1, 1152))

# ================================================================================
# 5. 真实 Nordland 图的 WTA mask 前后对比图
# ================================================================================
def save_real_image_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.io import read_image

    override_path = REPO_ROOT / "IDEA1-covstdp" / "phase1" / "configs" / "local_override.json"
    if not override_path.exists():
        print("[SKIP] local_override.json 不存在，跳过真实图像对比图")
        return
    data_dir = json.loads(override_path.read_text())["data_dir"]

    from vprtempo.src.dataset import ProcessImage
    import pandas as pd
    csv_path = REPO_ROOT / "vprtempo" / "dataset" / "nordland-spring.csv"
    df = pd.read_csv(csv_path)                       # 首行是表头
    fname = df.iloc[0, 0]
    img = read_image(str(Path(data_dir) / "spring" / fname)).float()
    spikes = ProcessImage([28, 28], 7)(img)          # [784] 平向量

    torch.manual_seed(0)
    l = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                     wta_mode='local', device=DEVICE)
    o = l(l.reshape_input(spikes.unsqueeze(0)))

    fig, axes = plt.subplots(3, 9, figsize=(18, 6))
    axes[0, 0].imshow(spikes.view(28, 28).numpy(), cmap="gray")
    axes[0, 0].set_title("input", fontsize=8)
    for i in range(8):
        axes[0, 1 + i].axis("off")
    for r, (mp, tag) in enumerate([(o.pre_wta, "pre-WTA"), (o.post_wta, "post-WTA")], start=1):
        for i in range(8):
            axes[r, i].imshow(mp[0, i].detach().numpy(), cmap="viridis")
            axes[r, i].set_title(f"ch{i} {tag}", fontsize=8)
        axes[r, 8].axis("off")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    out_path = REPO_ROOT / "IDEA1-covstdp" / "results" / "wta_mask_demo.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"[ OK ] 对比图已存 {out_path}")


save_real_image_figure()

# ================================================================================
# 汇总
# ================================================================================
print(f"\n{'='*60}\n{sum(PASS)}/{len(PASS)} 断言通过")
if not all(PASS):
    sys.exit(1)
print("S2.1/S2.2 sanity 全部通过 ✅")
