# MIT License
#
# Copyright (c) 2023 Adam Hines, Peter Stratton, Michael Milford, Tobias Fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
================================================================================
模块说明：conv_frontend —— IDEA1 卷积前端与 vprtempo 主仓库之间的桥接层
================================================================================
存在理由（PLAN.md S2.5 / ADR-2）：
  ConvSNNLayer 的实现位于 IDEA1-covstdp/src/（目录名含连字符，无法常规 import）。
  本模块按路径懒加载它们，向 VPRTempoTrain / VPRTempo 暴露统一入口，
  使主仓库的接入改动最小、且 frontend='none'（B0 默认路径）时完全不触发加载。

暴露内容：
  ConvSNNLayer / ConvFrontendModule   —— 类（isinstance 分发与推理链适配用）
  train_conv_layer                    —— conv 层训练循环
  build_conv_layer(args, dims, device, inference) —— 按配置构造 conv 层
  conv_forward(layer, spikes_flat)    —— 平向量 → pooled_flat（prev_layers / 推理用）
  is_conv_layer(obj)                  —— 安全的 isinstance 封装
================================================================================
"""
import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOADED = {}


def _load(mod_name, rel_path):
    if mod_name not in _LOADED:
        spec = importlib.util.spec_from_file_location(mod_name, _REPO_ROOT / rel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LOADED[mod_name] = mod
    return _LOADED[mod_name]


def _layer_mod():
    return _load("idea1_conv_snn_layer", "IDEA1-covstdp/src/conv_snn_layer.py")


def _learn_mod():
    return _load("idea1_conv_learning", "IDEA1-covstdp/src/conv_learning.py")


def _gabor_mod():
    return _load("idea1_gabor_frontend", "IDEA1-covstdp/src/gabor_frontend.py")


# ---- 类与函数的惰性代理（首次访问时才真正加载 IDEA1 模块）----
def __getattr__(name):
    if name in ("ConvSNNLayer", "ConvFrontendModule"):
        return getattr(_layer_mod(), name)
    if name in ("train_conv_layer", "calc_stdp_conv", "apply_itp_conv"):
        return getattr(_learn_mod(), name)
    raise AttributeError(name)


def is_conv_layer(obj):
    """安全的 isinstance 封装（未加载 IDEA1 模块时返回 False，B0 路径零开销）"""
    if obj.__class__.__name__ != "ConvSNNLayer":
        return False
    return isinstance(obj, _layer_mod().ConvSNNLayer)


def build_conv_layer(model, dims, device, inference):
    """
    按配置构造 ConvSNNLayer。
    frontend='conv_stdp'    —— 正常训练；
    frontend='random_conv'  —— B1 对照：结构相同但 frozen=True（S2.6，train_new_model 分发时跳过训练）；
    frontend='gabor'        —— B5：载入手工 Gabor 组并 frozen=True（S2.9；frozen 同时旁路
                              STDP 的符号钳制与保范数归一化，守卫在 calc_stdp_conv 入口）。
    frozen 经构造参数传入（S2.9 起成为 ConvSNNLayer 的正式构造参数；对 random_conv
    与原先"构造后赋值属性"完全等价——不涉及随机数消耗，B1/B2 初始化可比性不变）。
    """
    frontend = getattr(model, 'frontend', 'none')
    ConvSNNLayer = _layer_mod().ConvSNNLayer
    layer = ConvSNNLayer(
        input_dims=dims,
        in_channels=1,
        out_channels=int(getattr(model, 'conv_channels', 32)),
        kernel_size=int(getattr(model, 'conv_kernel', 5)),
        thr_range=[0, 0.5],
        fire_rate=[0.2, 0.9],
        ip_rate=0.15,
        stdp_rate=0.005,
        wta_mode=getattr(model, 'wta_mode', 'local'),
        wta_block=int(getattr(model, 'wta_block', 4)),
        device=device,
        inference=inference,
        frozen=frontend in ('random_conv', 'gabor'),
    )
    if frontend == 'gabor':
        # 推理侧同样载入：值与 state_dict 中的保存值逐元素一致（Gabor 组全程确定性），
        # load_model 加载后与训练侧无差异
        _gabor_mod().load_gabor_weights(layer)
    return layer


def conv_forward(layer, spikes_flat):
    """平向量 [1, H*W] → conv 前向 → pooled_flat（prev_layers 冻结前向 / 推理特征点用）"""
    import torch
    spikes_flat = spikes_flat.to(layer.w.weight.device)  # 与层设备对齐（GPU/CPU 混用安全）
    with torch.no_grad():
        out = layer(layer.reshape_input(spikes_flat))
    return out.pooled_flat
