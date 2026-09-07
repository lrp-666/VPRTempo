#!/usr/bin/env python
# ================================================================================
# test_feat_bcm.py — feat-BCM（feature_layer BCM 滑动阈值）验收断言
#
# 断言清单：
#   1. 回归：bcm_gate 关闭（默认 / 显式 False）时与原版逐比特一致
#      （5 步 calc_stdp 后权重、阈值 torch.equal）；
#   2. 语义：手工复现单步完整更新链（gate=旧 θ_M、兴奋/抑制分组、符号钳制、
#      Homeostasis），与实现逐元素一致；θ_M 时序 = 先旧阈值门控、后 EMA 更新；
#   3. θ_M 单调收敛无振荡：权重/阈值冻结（eta_stdp=eta_ip=0）+ 恒定输入下，
#      θ_M 是目标 post² 的纯 EMA——逐神经元单调且收敛到 post²；
#   4. θ_M 为普通属性，不进 state_dict；
#   5. 开启后特征层发放率分布变化：同初始化 bcm on/off 对跑 300 步随机输入，
#      每神经元平均发放率分布显著不同，且 θ_M 离开初值 0.25。
#
# 用法：pixi run python IDEA1-covstdp/experiments/test_feat_bcm.py
# ================================================================================
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"
sys.path.insert(0, str(REPO_ROOT))

import vprtempo.src.blitnet as bn


def make_layer(seed=0, dims=(16, 32), **kw):
    """同种子构造 => 权重/阈值/fire_rate/const_inp 全同（addWeights 用 np.random，
    初始化用 torch，双种子覆盖）"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return bn.SNNLayer(dims=list(dims), thr_range=[0, 0.5], fire_rate=[0.2, 0.9],
                       ip_rate=0.15, stdp_rate=0.005, p=[0.3, 0.5],
                       device='cpu', **kw)


def rand_step(layer, in_dim, seed):
    """生成一个随机样本并跑 calc_stdp 普通分支（spk_force=False）"""
    g = torch.Generator().manual_seed(seed)
    prespike = torch.rand(1, in_dim, generator=g)
    spikes = bn.clamp_spikes(layer.w(prespike), layer)
    noclp = layer.w(prespike).detach()
    bn.calc_stdp(prespike, spikes, noclp, layer, idx=torch.tensor([0]))


# ---- 断言 1：关闭 = 原版逐比特一致 ----
la = make_layer()                                  # 默认（bcm_gate 缺省）
lb = make_layer(bcm_gate=False, bcm_alpha=0.001)   # 显式关闭
assert not hasattr(la, 'theta_m') and not hasattr(lb, 'theta_m')
for s in range(5):
    rand_step(la, 16, 100 + s)
    rand_step(lb, 16, 100 + s)
assert torch.equal(la.w.weight.data, lb.w.weight.data), "关闭时权重不一致"
assert torch.equal(la.thr.data, lb.thr.data), "关闭时阈值不一致"
print("[1/5] 回归：bcm_gate 关闭（默认/显式）逐比特一致 ✔")

# ---- 断言 2：语义对拍（gate=旧 θ_M，先门控后 EMA）----
torch.manual_seed(1)
np.random.seed(1)
layer = make_layer(seed=1, bcm_gate=True, bcm_alpha=0.01)
layer.eta_ip = torch.tensor(0.0)                   # 关掉 ITP，聚焦 STDP 门控语义
g = torch.Generator().manual_seed(7)
prespike = torch.rand(1, 16, generator=g)
spikes = bn.clamp_spikes(layer.w(prespike), layer)
noclp = layer.w(prespike).detach()

W0 = layer.w.weight.data.clone()
thr0 = layer.thr.data.clone()
theta0 = layer.theta_m.clone()                     # 旧 θ_M
bn.calc_stdp(prespike, spikes, noclp, layer, idx=torch.tensor([0]))

# 手工复现：普通分支 + 符号钳制 + Homeostasis（ITP 已关）
shape = W0.shape                                   # [out, in]
pre = torch.tile(torch.reshape(prespike, (shape[1], 1)), (1, shape[0]))
post = torch.tile(spikes, (shape[1], 1))
gate = torch.tile(theta0, (shape[1], 1))           # 旧 θ_M 门控
W1 = W0 + (((gate - post) * (pre > 0) * (post > 0) * layer.havconnCombinedExc.T)
           * layer.eta_stdp).T
W1 = W1 + (((gate - post) * (pre > 0) * (post > 0) * layer.havconnCombinedInh.T)
           * (layer.eta_stdp * -1)).T
W1[layer.havconnCombinedExc] = W1[layer.havconnCombinedExc].clamp(min=1e-06, max=10)
W1[layer.havconnCombinedInh] = W1[layer.havconnCombinedInh].clamp(min=-10, max=-1e-06)
inhW = W1.T.clone()
inhW[inhW > 0] = 0
W1 = W1 + (torch.mul(noclp, inhW) * layer.eta_stdp * 50).T
assert torch.allclose(layer.w.weight.data, W1, atol=1e-7), \
    f"feat-BCM 更新语义不符（max diff {(layer.w.weight.data - W1).abs().max()}）"
theta1 = theta0 * (1.0 - 0.01) + 0.01 * spikes.pow(2)   # EMA 在门控之后
assert torch.allclose(layer.theta_m, theta1, atol=1e-9), "θ_M EMA 时序/公式不符"
assert torch.equal(layer.thr.data, thr0), "eta_ip=0 时阈值不应变化"
print("[2/5] 语义：gate=旧 θ_M、先门控后 EMA，手工复现逐元素一致 ✔")

# ---- 断言 3：θ_M 单调收敛无振荡（冻结权重/阈值 + 恒定输入 => 纯 EMA）----
layer = make_layer(seed=2, bcm_gate=True, bcm_alpha=0.01)
layer.eta_stdp = torch.tensor(0.0)
layer.eta_ip = torch.tensor(0.0)
spikes = torch.zeros(1, 32)
spikes[0, :8] = 0.9
spikes[0, 8:16] = 0.3                              # 16 个神经元静默、8 个高、8 个中
target = spikes.pow(2)
hist = []
prespike = torch.rand(1, 16, generator=torch.Generator().manual_seed(3))
for _ in range(2000):
    bn.calc_stdp(prespike, spikes, torch.zeros(1, 32), layer, idx=torch.tensor([0]))
    hist.append(layer.theta_m.clone())
H = torch.cat(hist)                                # [T, 32]
d = H[1:] - H[:-1]
for j in range(32):
    col = d[:, j]
    assert (col >= -1e-12).all() or (col <= 1e-12).all(), f"神经元 {j} θ_M 非单调（振荡）"
assert torch.allclose(H[-1], target[0], atol=1e-4), \
    f"θ_M 未收敛到 post²（max diff {(H[-1] - target[0]).abs().max()}）"
print(f"[3/5] θ_M 2000 步逐神经元单调收敛到 post²（终值误差 "
      f"{(H[-1] - target[0]).abs().max():.2e}），无振荡 ✔")

# ---- 断言 4：θ_M 不进 state_dict ----
sd = layer.state_dict()
assert not any('theta_m' in k for k in sd.keys()), f"theta_m 泄漏进 state_dict: {list(sd.keys())}"
print(f"[4/5] state_dict 键 = {list(sd.keys())}，无 theta_m ✔")

# ---- 断言 5：开启后发放率分布变化 ----
loff = make_layer(seed=5, dims=(64, 128))
lon = make_layer(seed=5, dims=(64, 128), bcm_gate=True, bcm_alpha=0.001)
theta_init = lon.theta_m.clone()
T = 300
fr_off = torch.zeros(128)
fr_on = torch.zeros(128)
for t in range(T):
    prespike = torch.rand(1, 64, generator=torch.Generator().manual_seed(1000 + t))
    for lyr, acc in ((loff, fr_off), (lon, fr_on)):
        spikes = bn.clamp_spikes(lyr.w(prespike), lyr)
        acc += (spikes > 0).float().squeeze(0)
        bn.calc_stdp(prespike, spikes, lyr.w(prespike).detach(), lyr,
                     idx=torch.tensor([0]))
fr_off /= T
fr_on /= T
assert not torch.allclose(fr_off, fr_on), "bcm 开启后发放率分布无变化"
assert not torch.equal(lon.theta_m, theta_init), "θ_M 未更新"
assert not torch.equal(loff.w.weight.data, lon.w.weight.data), "bcm 开启后权重轨迹无差异"
print(f"[5/5] 发放率分布变化：mean(off)={fr_off.mean():.4f} vs mean(on)={fr_on.mean():.4f}，"
      f"std(off)={fr_off.std():.4f} vs std(on)={fr_on.std():.4f}，"
      f"逐神经元 |Δfr| 均值={(fr_off - fr_on).abs().mean():.4f} ✔")

print("\n全部 5 项断言通过：feat-BCM 机制实现正确且默认行为不变。")
