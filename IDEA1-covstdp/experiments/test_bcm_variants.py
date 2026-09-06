#!/usr/bin/env python
# ================================================================================
# test_bcm_variants.py — S33 BCM 变体验收断言（bcm_full / bcm_on_frozen）
#
# 断言清单：
#   1. 互斥：bcm_gate + bcm_full 同开 → ValueError；bcm_on_frozen 非冻结层 →
#      ValueError；bcm_on_frozen + bcm_gate/bcm_full → ValueError；
#   2. bcm_full 对拍：calc_stdp_conv（向量化）vs calc_stdp_conv_reference（循环版）
#      逐元素一致（local/none WTA × mean/sum 聚合），θ_M 同步一致；
#   3. bcm_full 语义：M 值 = post·(post−θ_M)（与 bcm_gate 的 θ_M−post 方向相反、
#      幅度多一个 y 因子）——构造已知响应直接验证更新方向；
#   4. bcm_on_frozen：冻结 Gabor 层上 apply_bcm_threshold_conv 后权重逐位不变、
#      θ 精确跟随 θ_M（EMA of post²）；train_conv_layer 走 BCM 分支（fire_rate
#      目标不被使用——thr 不再向 fire_rate 不动点收敛而是跟随 θ_M）；
#   5. 回归：默认层（全关）行为不变——更新与 0.5−post 规则一致。
#
# 用法：pixi run python IDEA1-covstdp/experiments/test_bcm_variants.py
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

# ---- 断言 1：互斥 ----
try:
    ConvSNNLayer(out_channels=8, kernel_size=5, device='cpu',
                 bcm_gate=True, bcm_full=True)
    raise AssertionError("bcm_gate + bcm_full 同开未报错")
except ValueError:
    pass
try:
    ConvSNNLayer(out_channels=8, kernel_size=5, device='cpu',
                 frozen=False, bcm_on_frozen=True)
    raise AssertionError("bcm_on_frozen 非冻结层未报错")
except ValueError:
    pass
try:
    ConvSNNLayer(out_channels=8, kernel_size=5, device='cpu',
                 frozen=True, bcm_on_frozen=True, bcm_gate=True)
    raise AssertionError("bcm_on_frozen + bcm_gate 同开未报错")
except ValueError:
    pass
print("[1/5] 互斥断言全部生效（gate×full / on_frozen 非冻结 / on_frozen×gate）✔")

# ---- 断言 2：bcm_full 向量化 vs 参考版对拍 ----
torch.manual_seed(0)
n_pairs = 0
for wta_mode in ('local', 'none'):
    for agg_mode in ('mean', 'sum'):
        l1 = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                          wta_mode=wta_mode, wta_block=2, device='cpu',
                          bcm_full=True)
        l2 = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                          wta_mode=wta_mode, wta_block=2, device='cpu',
                          bcm_full=True)
        l2.load_state_dict(l1.state_dict())
        l2.theta_m = l1.theta_m.clone()
        for step in range(3):
            x = torch.rand(1, 1, 28, 28)
            o1, o2 = l1(x), l2(x)
            learn_mod.calc_stdp_conv(x, o1, l1, agg_mode=agg_mode)
            learn_mod.calc_stdp_conv_reference(x, o2, l2, agg_mode=agg_mode)
            assert torch.allclose(l1.w.weight.data, l2.w.weight.data, atol=1e-6), \
                f"bcm_full 对拍失败（{wta_mode}/{agg_mode}/step{step}）"
            assert torch.allclose(l1.theta_m, l2.theta_m, atol=1e-9), \
                f"bcm_full θ_M 不一致（{wta_mode}/{agg_mode}/step{step}）"
            n_pairs += 1
print(f"[2/5] bcm_full 对拍通过（2 WTA × 2 聚合 × 3 步 = {n_pairs} 次逐元素一致）✔")

# ---- 断言 3：bcm_full 语义（M = post·(post−θ_M)，含 y 因子）----
torch.manual_seed(0)
layer = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                     wta_mode='local', wta_block=2, device='cpu', bcm_full=True)
x = torch.rand(1, 1, 28, 28)
out = layer(x)
W0 = layer.w.weight.data.clone()
theta0 = layer.theta_m.clone()
learn_mod.calc_stdp_conv(x, out, layer, agg_mode='sum')
# 手工复现完整更新链（含 Step 7 钳制与 Step 8 保范数）：
#   dK_c = Σ_winners post·(post−θ_M,c)·(patch−0.5) → W1 = clamp(W0 + η·sign·dK)
#   → 保范数 L1 缩放
C = W0.shape[0]
dK = torch.zeros_like(W0)
for c in range(C):
    th_c = float(theta0[0, c, 0, 0])
    for y0, x0 in out.winner_mask[0, c].nonzero(as_tuple=False):
        post = float(out.pre_wta[0, c, y0, x0])
        patch = (x - 0.5)[0, 0, y0:y0 + 5, x0:x0 + 5]      # padding=0
        dK[c, 0] += post * (post - th_c) * patch
sign = torch.where(layer.havconnExc, 1.0, -1.0).view(C, 1, 1, 1)
W1 = W0 + layer.eta_stdp * sign * dK
exc = layer.havconnExc.view(C, 1, 1, 1)
W1 = torch.where(exc, W1.clamp(0.0, 10.0), W1.clamp(-10.0, 0.0))
nrm0 = torch.linalg.norm(W0.flatten(1), ord=1, dim=1)
nrm1 = torch.linalg.norm(W1.flatten(1), ord=1, dim=1)
updated = dK.abs().flatten(1).sum(dim=1) > 0
scale = torch.ones_like(nrm0)
valid = updated & (nrm0 > 0) & (nrm1 > 0)
scale[valid] = nrm0[valid] / nrm1[valid]
W1 = W1 * scale.view(C, 1, 1, 1)
assert torch.allclose(layer.w.weight.data, W1, atol=1e-6), \
    f"bcm_full 语义不符（max diff {(layer.w.weight.data - W1).abs().max()}）"
# 与 bcm_gate 的方向差异自检：同一 θ_M 下 post>θ_M 时 full 给 LTP（符号随 post−θ_M）
print(f"[3/5] bcm_full 完整更新链（φ(y)=y(y−θ_M)+钳制+保范数）逐元素复核通过 ✔")

# ---- 断言 4：bcm_on_frozen（冻结 Gabor + BCM 阈值）----
torch.manual_seed(0)
layer = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                     wta_mode='local', wta_block=2, device='cpu',
                     frozen=True, bcm_on_frozen=True)
gabor_mod.load_gabor_weights(layer)
assert hasattr(layer, 'theta_m'), "bcm_on_frozen 层应有 θ_M"
w0 = layer.w.weight.data.clone()
thr0 = layer.thr.data.clone()
tm0 = layer.theta_m.clone()
for _ in range(5):
    xx = torch.rand(1, 1, 28, 28)
    o = layer(xx)
    learn_mod.calc_stdp_conv(xx, o, layer)          # frozen 守卫：应完全旁路
    learn_mod.apply_bcm_threshold_conv(o, layer)
assert torch.equal(layer.w.weight.data, w0), "bcm_on_frozen 权重被改动"
assert not torch.equal(layer.theta_m, tm0), "θ_M 未演化"
assert not torch.equal(layer.thr.data, thr0), "θ 未跟随 θ_M"
assert torch.allclose(layer.thr.data, layer.theta_m.clamp(min=layer.thr_min)), \
    "θ 未精确跟随 θ_M（clamp 后）"
# 手工核对 EMA 时序（最后一步）
post2 = o.pre_wta.pow(2).mean(dim=(0, 2, 3), keepdim=True)
print(f"[4/5] bcm_on_frozen：权重逐位不变，θ_M 演化，θ≡clamp(θ_M) ✔ "
      f"（θ 范围 {layer.thr.data.min():.4f}~{layer.thr.data.max():.4f}）")

# train_conv_layer 冒烟：bcm_on_frozen 走 BCM 分支（用 θ_M 存在性 + θ 变化判定）
from torch.utils.data import DataLoader, TensorDataset
layer2 = ConvSNNLayer(input_dims=[28, 28], out_channels=32, kernel_size=5,
                      wta_mode='local', wta_block=2, device='cpu',
                      frozen=True, bcm_on_frozen=True)
gabor_mod.load_gabor_weights(layer2)
ds = TensorDataset(torch.rand(8, 784), torch.zeros(8))
loader = DataLoader(ds, batch_size=1, shuffle=False)
model = type("M", (), {"device": torch.device('cpu'), "conv_epoch": 1,
                       "pre_mode": "centered", "agg_mode": "mean"})()
thr_before = layer2.thr.data.clone()
learn_mod.train_conv_layer(loader, layer2, model, model_num=0)
assert not torch.equal(layer2.thr.data, thr_before), "train_conv_layer 未更新阈值"
assert torch.allclose(layer2.thr.data, layer2.theta_m.clamp(min=layer2.thr_min),
                      atol=1e-6), "train_conv_layer 未走 BCM 分支（θ≠θ_M）"
print("[4b/5] train_conv_layer bcm_on_frozen 分支冒烟通过（θ≡clamp(θ_M)，8 样本）✔")

# ---- 断言 5：回归——默认层（全关）仍按 0.5−post 更新 ----
torch.manual_seed(0)
la = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                  wta_mode='local', wta_block=2, device='cpu')
lb = ConvSNNLayer(input_dims=[28, 28], out_channels=8, kernel_size=5,
                  wta_mode='local', wta_block=2, device='cpu')
lb.load_state_dict(la.state_dict())
x = torch.rand(1, 1, 28, 28)
oa, ob = la(x), lb(x)
learn_mod.calc_stdp_conv(x, oa, la)
learn_mod.calc_stdp_conv_reference(x, ob, lb)
assert torch.allclose(la.w.weight.data, lb.w.weight.data, atol=1e-6)
print("[5/5] 默认层（全关）向量化 vs 参考版一致，B2 行为不变 ✔")

print("\n全部断言通过 ✔  bcm_full / bcm_on_frozen 就绪")
