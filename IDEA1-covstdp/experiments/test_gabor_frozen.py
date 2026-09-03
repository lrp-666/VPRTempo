#!/usr/bin/env python
# ================================================================================
# test_gabor_frozen.py — S2.9 验收断言（B5 Gabor 前端 + frozen 负瓣保护）
#
# 断言清单（S2.9 卡片第 2/3 条）：
#   1. Gabor 组形状 [32,1,5,5]、每核 L1=1、均值≈0，且自带负瓣（负元素非零）；
#   2. 载入 frozen 层后，权重与 Gabor 组逐元素一致——负值元素保持非零
#      （初始化路径未经过符号钳制）；
#   3. frozen 层调用 calc_stdp_conv / calc_stdp_conv_reference 后权重逐位不变
#      （更新后的 clamp/renorm 路径被 frozen 守卫旁路）；
#   4. 回归：非 frozen 层（B2 默认路径）calc_stdp_conv 正常更新且兴奋核无负值
#      （frozen 守卫不改变默认行为）；
#   5. 前向冒烟：gabor 层 pooled_flat 维度与 B1/B2 一致（[1, 1152]）。
#
# 用法：pixi run python IDEA1-covstdp/experiments/test_gabor_frozen.py
# ================================================================================
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
gabor_mod = load_mod("gabor_frontend", "IDEA1-covstdp/src/gabor_frontend.py")
ConvSNNLayer = conv_mod.ConvSNNLayer

torch.manual_seed(0)

# ---- 断言 1：Gabor 组的形状 / 归一化 / 负瓣 ----
bank = gabor_mod.gabor_kernel_bank(kernel_size=5)
assert bank.shape == (32, 1, 5, 5), f"Gabor 组形状错误: {bank.shape}"
l1 = bank.abs().flatten(1).sum(dim=1)
assert torch.allclose(l1, torch.ones(32), atol=1e-5), f"每核 L1 应=1: {l1.min()}~{l1.max()}"
means = bank.flatten(1).mean(dim=1)
assert means.abs().max() < 1e-6, f"每核均值应≈0（去直流）: {means.abs().max()}"
assert (bank < 0).any(), "Gabor 组应有负瓣"
print(f"[1/5] Gabor 组 [32,1,5,5]，L1=1，mean≈0，负元素占比 {(bank < 0).float().mean():.3f}")

# ---- 断言 2：载入 frozen 层后负值保持非零 ----
layer = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                     wta_mode='local', wta_block=4, device='cpu', frozen=True)
gabor_mod.load_gabor_weights(layer)
assert layer.frozen is True
assert layer.itp_on_frozen is True, "B5 应保留 ITP 阈值自适应（S2.9 修订）"
assert torch.equal(layer.w.weight.data, bank), "载入后权重应与 Gabor 组逐元素一致"
n_neg = int((layer.w.weight.data < 0).sum())
assert n_neg > 0, "负瓣保护失败：载入后负值元素为零"
print(f"[2/5] frozen 层载入后负值元素 {n_neg} 个保持非零（未经过符号钳制）")

# ---- 断言 3：frozen 层的 STDP 路径完全旁路（clamp/renorm 不经过）----
x = torch.rand(1, 1, 28, 28)
out = layer(x)
w_before = layer.w.weight.data.clone()
learn_mod.calc_stdp_conv(x, out, layer)
learn_mod.calc_stdp_conv_reference(x, out, layer)
assert torch.equal(layer.w.weight.data, w_before), "frozen 层权重被 STDP 路径改动"
assert (layer.w.weight.data < 0).sum() == n_neg, "frozen 层负瓣被 clamp 摧毁"
print("[3/5] frozen 层 calc_stdp_conv（正式版+参考版）后权重逐位不变，负瓣完好")

# ---- 断言 4：回归——非 frozen 层默认行为不变（正常更新 + 符号钳制生效）----
layer_b2 = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                        wta_mode='local', wta_block=4, device='cpu')
out_b2 = layer_b2(x)
w_b2_before = layer_b2.w.weight.data.clone()
learn_mod.calc_stdp_conv(x, out_b2, layer_b2)
assert not torch.equal(layer_b2.w.weight.data, w_b2_before), "非 frozen 层未更新（异常）"
exc = layer_b2.havconnExc.view(32, 1, 1, 1)
assert (layer_b2.w.weight.data[exc.expand_as(layer_b2.w.weight.data)] >= 0).all(), \
    "非 frozen 层兴奋核出现负值（clamp 失效）"
print("[4/5] 非 frozen 层正常更新，兴奋核符号钳制生效（默认行为不变）")

# ---- 断言 5：前向冒烟——pooled_flat 维度与 B1/B2 一致 ----
assert out.pooled_flat.shape == (1, 32 * 6 * 6), f"pooled_flat 维度错误: {out.pooled_flat.shape}"
assert (out.pre_wta >= 0).all() and (out.pre_wta <= 0.9).all()
print(f"[5/5] 前向冒烟通过：pooled_flat {tuple(out.pooled_flat.shape)}，响应在 [0, 0.9]")

print("\n全部断言通过 ✔  B5 Gabor 前端 + frozen 负瓣保护就绪")
