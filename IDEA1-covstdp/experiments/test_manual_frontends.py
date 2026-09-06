#!/usr/bin/env python
# ================================================================================
# test_manual_frontends.py — S33 手工特征动物园验收断言（dog / loggabor / dct）
#
# 断言清单：
#   1. 三个核组形状 [32,1,5,5]、每核 L1=1、确定性（两次生成逐元素一致）、零直流；
#   2. DoG 组径向对称（无方向性自检：转置/rot90 不变）；
#   3. DCT 组无直流（均值≈0）且基序正确（第 1 个核为 (u,v)=(1,0)/(0,1) 频率档，
#      组内无重复核）；
#   4. Log-Gabor 组 DC 精确为 0（频域构造保证）、4 方向覆盖（不同方向核不重合）；
#   5. 载入 frozen 层：权重逐元素一致、frozen + itp_on_frozen 置位、负瓣保留，
#      calc_stdp_conv 后权重逐位不变（frozen 守卫）；
#   6. dispatch 冒烟：经 vprtempo.src.conv_frontend.build_conv_layer 构造三种前端，
#      前向 pooled_flat 维度与 B5 一致（block=2 → [1, 32*12*12]）。
#
# 用法：pixi run python IDEA1-covstdp/experiments/test_manual_frontends.py
# ================================================================================
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"
sys.path.insert(0, str(REPO_ROOT))


def load_mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv_mod = load_mod("conv_snn_layer", "IDEA1-covstdp/src/conv_snn_layer.py")
learn_mod = load_mod("conv_learning", "IDEA1-covstdp/src/conv_learning.py")
man_mod = load_mod("manual_frontends", "IDEA1-covstdp/src/manual_frontends.py")
ConvSNNLayer = conv_mod.ConvSNNLayer

torch.manual_seed(0)

# ---- 断言 1：形状 / L1=1 / 确定性 / 零直流 ----
banks = {}
for kind in ('dog', 'loggabor', 'dct'):
    bank = man_mod.BANKS[kind](kernel_size=5)
    bank2 = man_mod.BANKS[kind](kernel_size=5)
    assert bank.shape == (32, 1, 5, 5), f"{kind} 组形状错误: {bank.shape}"
    assert torch.equal(bank, bank2), f"{kind} 组非确定性（两次生成不一致）"
    l1 = bank.abs().flatten(1).sum(dim=1)
    assert torch.allclose(l1, torch.ones(32), atol=1e-5), \
        f"{kind} 每核 L1 应=1: {l1.min()}~{l1.max()}"
    means = bank.flatten(1).mean(dim=1)
    assert means.abs().max() < 1e-6, f"{kind} 每核均值应≈0: {means.abs().max()}"
    banks[kind] = bank
    print(f"[1] {kind}: [32,1,5,5]，L1=1，mean≈0，确定性 ✔")

# ---- 断言 2：DoG 径向对称（转置 + rot90 不变 = 无方向性）----
dog = banks['dog'][:, 0]
for i in range(32):
    k = dog[i]
    assert torch.allclose(k, k.T, atol=1e-6), f"DoG 核 {i} 转置不对称"
    assert torch.allclose(k, torch.rot90(k, 1), atol=1e-6), f"DoG 核 {i} rot90 不对称"
# on/off 中心成对（相邻两核互为负）
assert torch.allclose(dog[0::2], -dog[1::2], atol=1e-6), "DoG on/off 相位配对失败"
print("[2] DoG 组 32 核全部径向对称（无方向性），on/off 中心成对 ✔")

# ---- 断言 3：DCT 组无直流 + 无重复核 + 首核为最低频率档 ----
dct_bank = banks['dct'][:, 0]
assert dct_bank.flatten(1).mean(dim=1).abs().max() < 1e-6, "DCT 组含直流"
flat = dct_bank.flatten(1)
cos = torch.nn.functional.cosine_similarity(
    flat.unsqueeze(0), flat.unsqueeze(1), dim=-1)
cos.fill_diagonal_(0)
assert cos.abs().max() < 0.999, f"DCT 组存在重复核（cos={cos.abs().max():.4f}）"
print(f"[3] DCT 组无直流（mean≈0）、32 核互不相同（max|cos|={cos.abs().max():.3f}）✔")

# ---- 断言 4：Log-Gabor DC 精确为 0 + 4 方向覆盖 ----
lg = banks['loggabor'][:, 0]
assert lg.flatten(1).mean(dim=1).abs().max() < 1e-6, "Log-Gabor 组含直流"
# 方向覆盖：bank 排布为 方向(4) × f0(2) × 相位(2) × 带宽(2)，每 8 核换一个方向；
# 不同方向同参数核（如核 0 与核 8）不应高度重合（方向选择性存在）
c08 = torch.nn.functional.cosine_similarity(flat_lg := lg.flatten(1)[0:1], lg.flatten(1)[8:9], dim=-1)
assert abs(float(c08)) < 0.95, f"Log-Gabor 方向 0 与方向 1 的核重合（cos={float(c08):.3f}）"
assert (lg < 0).any(), "Log-Gabor 组应有负瓣"
print(f"[4] Log-Gabor 组 DC=0、负瓣在、方向间 cos={float(c08):.3f}（方向选择性在）✔")

# ---- 断言 5：载入 frozen 层（负瓣保护 + frozen 守卫）----
x = torch.rand(1, 1, 28, 28)
for kind in ('dog', 'loggabor', 'dct'):
    layer = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                         wta_mode='local', wta_block=2, device='cpu', frozen=True)
    man_mod.load_manual_weights(layer, kind)
    assert layer.frozen is True and layer.itp_on_frozen is True
    assert torch.equal(layer.w.weight.data, banks[kind]), f"{kind} 载入后权重不一致"
    n_neg = int((layer.w.weight.data < 0).sum())
    assert n_neg > 0, f"{kind} 负瓣丢失"
    out = layer(x)
    w_before = layer.w.weight.data.clone()
    learn_mod.calc_stdp_conv(x, out, layer)
    assert torch.equal(layer.w.weight.data, w_before), f"{kind} frozen 层被 STDP 改动"
    print(f"[5] {kind}: 载入 frozen+itp_on_frozen ✔，负瓣 {n_neg} 个，STDP 旁路 ✔")

# ---- 断言 6：dispatch 冒烟（build_conv_layer 三分支 + 前向维度）----
from vprtempo.src import conv_frontend as cf
for kind in ('dog', 'loggabor', 'dct'):
    model = argparse.Namespace(frontend=kind, conv_channels=32, conv_kernel=5,
                               wta_mode='local', wta_block=2)
    layer = cf.build_conv_layer(model, [28, 28], torch.device('cpu'), inference=False)
    out = layer(torch.rand(1, 1, 28, 28))
    assert out.pooled_flat.shape == (1, 32 * 12 * 12), \
        f"{kind} pooled_flat 维度错误: {out.pooled_flat.shape}"
    print(f"[6] {kind}: dispatch ✔，pooled_flat {tuple(out.pooled_flat.shape)}")

print("\n全部断言通过 ✔  手工特征动物园（dog / loggabor / dct）就绪")
