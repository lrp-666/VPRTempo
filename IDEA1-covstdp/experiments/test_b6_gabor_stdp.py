#!/usr/bin/env python
# ================================================================================
# test_b6_gabor_stdp.py — S2.10 / S3.3-8 验收断言（B6 Gabor 初始化 + STDP；free-sign）
#
# 断言清单（S2.10 卡片"验收"条 + S3.3-8 卡片）：
#   1. B6a 分解一致性：随机输入 x，ON_out − OFF_out 与 signed Gabor 前向逐元素一致
#      （atol=1e-6）。OFF 通道符号链：OFF 输出 = −conv(x, w_off)，w_off = −G⁻，
#      故联合响应 = conv(x,G⁺) − conv(x,G⁻) ≡ conv(x,G)；
#   2. 16 模式方向覆盖：σ=1.0 的 16 个模式覆盖全部 4 方向（θ∈{0,45,90,135°}，
#      每方向与一个该方向模板余弦相似度 >0.999）；
#   3. free-sign：训若干步后核出现负值（Step 7 符号钳制已放开）；非 free-sign 层
#      无负值越界（回归——ON ≥0、OFF ≤0 的钳制仍然生效）；
#   4. B6a 训 50 步后 ON/OFF 对仍近似互补（漂移监控：每对联合核 J=w_on+w_off 与
#      初始 G 的余弦相似度，min > 0.5）；
#   5. B0 回归：不带 --frontend 跑 smoke100 配置，R@1 = 0.99 不变。
#
# 用法：pixi run python IDEA1-covstdp/experiments/test_b6_gabor_stdp.py
# ================================================================================
import importlib.util
import json
import math
import subprocess
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
DEVICE = 'cpu'

# ---- 断言 1：B6a 分解一致性——ON_out − OFF_out ≡ signed Gabor 前向 ----
bank = gabor_mod.gabor_kernel_bank(kernel_size=5, device=DEVICE)
modes = bank[0::2]                                       # σ=1.0 的 16 个模式
layer_b6a = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                         wta_mode='local', wta_block=2, device=DEVICE)
gabor_mod.load_gabor_weights_decomposed(layer_b6a)
assert layer_b6a.frozen is False, "B6a 应 frozen=False（STDP+ITP 正常跑）"
assert bool(layer_b6a.havconnExc[:16].all()) and \
       bool((~layer_b6a.havconnExc[16:]).all()), "B6a 通道配对应为前 16 ON / 后 16 OFF"

x = torch.rand(1, 1, 28, 28)
raw = layer_b6a.w(x)                                     # [1,32,24,24] 原始 conv 输出
exc = layer_b6a.havconnExc.view(1, -1, 1, 1)
z = torch.where(exc, raw, -raw)                          # forward 的 ON/OFF 取负后响应
joint = z[0, :16] - z[0, 16:]                            # ON_out − OFF_out → [16,24,24]
ref = torch.nn.functional.conv2d(x, modes)[0]            # signed Gabor 前向
max_diff = (joint - ref).abs().max().item()
assert max_diff < 1e-6, f"分解一致性失败：max |ON−OFF − conv(x,G)| = {max_diff}"
print(f"[1/5] B6a 分解一致性：ON_out−OFF_out ≡ conv(x,G)（max diff {max_diff:.2e}）")

# ---- 断言 2：16 模式方向覆盖（4 方向齐全）----
half = 2
ys, xs = torch.meshgrid(torch.arange(-half, half + 1, dtype=torch.float32),
                        torch.arange(-half, half + 1, dtype=torch.float32),
                        indexing='ij')
covered = []
for theta in [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]:
    ct, st = math.cos(theta), math.sin(theta)
    xr = xs * ct + ys * st
    yr = -xs * st + ys * ct
    env = torch.exp(-(xr ** 2 + 0.25 * yr ** 2) / (2 * 1.0 ** 2))   # σ=1.0, γ=0.5
    tmpl = env * torch.cos(2 * math.pi * xr / 2.0)                  # λ=2.0, ψ=0
    tmpl = tmpl - tmpl.mean()
    tmpl = tmpl / tmpl.abs().sum()
    sims = torch.nn.functional.cosine_similarity(
        modes.flatten(1), tmpl.flatten(0).unsqueeze(0), dim=1)
    best = sims.max().item()
    assert best > 0.999, f"方向 θ={math.degrees(theta):.0f}° 无覆盖（最佳余弦 {best:.4f}）"
    covered.append(math.degrees(theta))
print(f"[2/5] 16 模式方向覆盖齐全：{covered}（各方向模板余弦 >0.999）")

# ---- 断言 3：free-sign 训后出现负值；非 free-sign 回归无越界 ----
def train_steps(layer, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        xi = torch.rand(1, 1, 28, 28, generator=g)
        out = layer(xi)
        learn_mod.calc_stdp_conv(xi, out, layer)

# free-sign：先把核整体取绝对值（无非负……即初值全 ≥0），训后必须出现负值
layer_fs = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                        wta_mode='local', wta_block=2, device=DEVICE,
                        free_sign=True)
with torch.no_grad():
    w0 = layer_fs.w.weight.data.abs()
    w0 = w0 / w0.flatten(1).abs().sum(dim=1).view(32, 1, 1, 1)      # 每核 L1=1
    layer_fs.w.weight.data = w0
assert (layer_fs.w.weight.data >= 0).all()
train_steps(layer_fs, 20)
n_neg_fs = int((layer_fs.w.weight.data < 0).sum())
assert n_neg_fs > 0, "free-sign 层训后未出现负值（Step 7 放开失效？）"
assert (layer_fs.w.weight.data.abs() <= 10.0).all(), "free-sign 幅度安全钳失效"

# 回归：非 free-sign 层（B2 默认路径）训后 ON 无负值、OFF 无正值
layer_b2 = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                        wta_mode='local', wta_block=2, device=DEVICE)
train_steps(layer_b2, 20, seed=2)
exc_m = layer_b2.havconnExc.view(32, 1, 1, 1).expand_as(layer_b2.w.weight.data)
assert (layer_b2.w.weight.data[exc_m] >= 0).all(), "回归失败：非 free-sign ON 核出现负值"
assert (layer_b2.w.weight.data[~exc_m] <= 0).all(), "回归失败：非 free-sign OFF 核出现正值"
print(f"[3/5] free-sign 训后出现负值（{n_neg_fs} 个）；非 free-sign 符号钳制回归正常")

# ---- 断言 4：B6a 训 50 步后 ON/OFF 对近似互补（漂移监控）----
torch.manual_seed(3)
layer_drift = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                           wta_mode='local', wta_block=2, device=DEVICE)
gabor_mod.load_gabor_weights_decomposed(layer_drift)
train_steps(layer_drift, 50, seed=3)
with torch.no_grad():
    w = layer_drift.w.weight.data
    joint_after = w[:16] + w[16:]                        # 每对联合核 J = w_on + w_off
    cos = torch.nn.functional.cosine_similarity(
        joint_after.flatten(1), modes.flatten(1), dim=1)  # vs 初始 G
assert (w[:16] >= 0).all() and (w[16:] <= 0).all(), "B6a 训后符号钳制应仍生效"
assert cos.min().item() > 0.5, \
    f"B6a 训 50 步后互补结构漂移过大：min cos(J,G) = {cos.min():.3f}"
print(f"[4/5] B6a 训 50 步漂移监控：cos(J,G) min={cos.min():.3f} "
      f"median={cos.median():.3f}（ON/OFF 符号钳制仍生效）")

# ---- 断言 5：B0 回归——smoke100（frontend='none'）R@1 = 0.99 不变 ----
cmd = [sys.executable, "IDEA1-covstdp/experiments/run_exp.py",
       "IDEA1-covstdp/phase1/configs/smoke100.json", "--eval", "--seed", "0"]
print(f"[5/5] 运行 B0 回归: {' '.join(cmd)}")
res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
assert res.returncode == 0, f"B0 回归运行失败:\n{res.stdout}\n{res.stderr}"
out_json = (REPO_ROOT / "IDEA1-covstdp/results/smoke100/seed_0"
            / "smoke100__seed0__eval.json")
with open(out_json) as f:
    r1 = json.load(f)["recallAtK"]["1"]
assert r1 == 0.99, f"B0 回归失败：smoke100 R@1 = {r1}（应为 0.99）"
print(f"[5/5] B0 回归通过：smoke100 R@1 = {r1}（frontend='none' 行为不变）")

print("\n全部断言通过 ✔  B6a/B6b Gabor 初始化 + STDP 与 free-sign 消融就绪")
