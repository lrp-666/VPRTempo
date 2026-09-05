#!/usr/bin/env python
# ================================================================================
# test_conv_thr_min.py — fork G 任务 1 验收：conv_thr_min（ITP 阈值地板）
#
# 背景（results/s210_b6_preview.md §3）：free-sign 直通前向下带符号核对非负输入
# 只有正半响应可过 clamp(min=0)，可达发放率结构性偏低（≈3–5%）；ITP 追目标
# 发放率把阈值推到 0 后被 clamp(min=0) 卡死 → 工作点崩塌（B6b 轨 A 0.13）。
#
# 验证内容（合成数据，CPU，100 步）：
#   1. 对照组 thr_min=0.0（默认）：free-sign 低发放工况下 thr 触底贴 0（复现病因）；
#   2. 修复组 thr_min=-0.5：100 步后 thr 分布不再贴 0（出现负值），且不破地板
#      （min ≥ -0.5）；
#   3. 回归：默认构造的层 thr_min == 0.0，apply_itp_conv 仍在 0 处截断
#      （vprtempo 默认行为不变）；
#   4. 接线：build_conv_layer 透传 conv_thr_min / itp_on_frozen（B1+ITP 行）。
#
# 用法：pixi run python IDEA1-covstdp/experiments/test_conv_thr_min.py
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
ConvSNNLayer = conv_mod.ConvSNNLayer

DEVICE = "cpu"
STEPS = 100
PASS = []


def check(name, cond, detail=""):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def run_freesign(thr_min, seed=0):
    """free-sign 层（带符号随机初始化，对齐 conv_stdp_freesign）训 STEPS 步，返回 thr 副本"""
    torch.manual_seed(seed)
    layer = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                         wta_mode='local', wta_block=2, free_sign=True,
                         thr_min=thr_min, device=DEVICE)
    for _ in range(STEPS):
        x = torch.rand(1, 28 * 28)              # 非负输入（对齐 spike 编码值域）
        xi = layer.reshape_input(x)
        out = layer(xi)
        learn_mod.calc_stdp_conv(xi, out, layer, pre_mode='centered', agg_mode='mean')
        learn_mod.apply_itp_conv(out, layer)
    return layer.thr.data.flatten().clone()


def thr_stats(t):
    qs = torch.quantile(t, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    return (f"min={qs[0]:.4f} q25={qs[1]:.4f} median={qs[2]:.4f} "
            f"q75={qs[3]:.4f} max={qs[4]:.4f} mean={t.mean():.4f} "
            f"frac_at_floor(<1e-6)={(t.abs() < 1e-6).float().mean():.2f}")


print("=" * 60)
print("free-sign 工况（带符号随机核 + 非负输入），train 100 步后的 thr 分布")

thr_default = run_freesign(thr_min=0.0)
print(f"  对照 thr_min= 0.0 : {thr_stats(thr_default)}")
check("对照组复现病因：thr 触底（min 贴 0）",
      float(thr_default.min()) < 1e-6, f"(min={float(thr_default.min()):.2e})")

thr_fixed = run_freesign(thr_min=-0.5)
print(f"  修复 thr_min=-0.5 : {thr_stats(thr_fixed)}")
check("修复组 thr 不再贴 0（出现负值）",
      float(thr_fixed.min()) < -1e-3, f"(min={float(thr_fixed.min()):.4f})")
check("修复组不破地板（min >= -0.5）",
      float(thr_fixed.min()) >= -0.5 - 1e-6, f"(min={float(thr_fixed.min()):.4f})")

print("=" * 60)
print("回归：默认行为不变")
torch.manual_seed(0)
layer = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5, device=DEVICE)
check("默认构造 thr_min == 0.0", layer.thr_min == 0.0)
layer.thr.data.fill_(0.1)
out = layer(layer.reshape_input(torch.rand(1, 28 * 28)))
out.pre_wta.zero_()                            # observed=0，强制 ITP 向下推
layer.fire_rate.fill_(1.0)
for _ in range(50):
    learn_mod.apply_itp_conv(out, layer)
check("默认层 apply_itp_conv 仍在 0 处截断",
      float(layer.thr.data.min()) == 0.0, f"(min={float(layer.thr.data.min()):.2e})")

print("=" * 60)
print("接线：build_conv_layer 透传")
from vprtempo.src import conv_frontend as cf
ns = argparse.Namespace(frontend='conv_stdp_freesign', conv_thr_min=-0.5)
l = cf.build_conv_layer(ns, [28, 28], DEVICE, inference=False)
check("conv_thr_min=-0.5 透传到层", l.thr_min == -0.5 and l.free_sign)
ns = argparse.Namespace(frontend='conv_stdp')
l = cf.build_conv_layer(ns, [28, 28], DEVICE, inference=False)
check("缺省 conv_thr_min=0.0", l.thr_min == 0.0)
ns = argparse.Namespace(frontend='random_conv', itp_on_frozen=True)
l = cf.build_conv_layer(ns, [28, 28], DEVICE, inference=False)
check("random_conv + itp_on_frozen=True 生效（B1+ITP 行）",
      l.frozen and getattr(l, 'itp_on_frozen', False))
ns = argparse.Namespace(frontend='random_conv')
l = cf.build_conv_layer(ns, [28, 28], DEVICE, inference=False)
check("random_conv 默认仍完全冻结（无 ITP，B1 不变）",
      l.frozen and not getattr(l, 'itp_on_frozen', False))

print("=" * 60)
print(f"总计 {sum(PASS)}/{len(PASS)} 断言通过")
sys.exit(0 if all(PASS) else 1)
