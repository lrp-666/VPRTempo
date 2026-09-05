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


def _parse_pair(value):
    """解析 "lo,hi" 字符串为 [lo, hi] 浮点对（与 main.py --dims "28,28" 风格一致）。
    已非字符串（如旧配置直接给列表）时原样透传。默认值与原硬编码完全一致，
    默认路径行为逐比特不变。"""
    if isinstance(value, str):
        return [float(x) for x in value.split(",")]
    return list(value)


def build_conv_layer(model, dims, device, inference):
    """
    按配置构造 ConvSNNLayer。
    frontend='conv_stdp'           —— 正常训练（B2）；
    frontend='random_conv'         —— B1 对照：结构相同但 frozen=True（S2.6，train_new_model 分发时跳过训练）；
    frontend='gabor'               —— B5：载入手工 Gabor 组并 frozen=True（S2.9；frozen 同时旁路
                                     STDP 的符号钳制与保范数归一化，守卫在 calc_stdp_conv 入口）。
    frontend='gabor_stdp'          —— B6a（S2.10）：16 个 σ=1.0 Gabor 模式 ON/OFF 分解载入
                                     （通道 i ON 装 G⁺、i+16 OFF 装 −G⁻，确定性配对），frozen=False
                                     ——STDP+ITP 正常跑，符号钳制不放松（ON−OFF ≡ conv(x,G)）。
    frontend='gabor_stdp_freesign' —— B6b（S2.10）：signed Gabor 原样载入，frozen=False +
                                     free_sign=True（前向跳过 ON/OFF 取负；Step 7 放开符号钳制、
                                     仅留 clamp(-10,10) 幅度安全钳）。
    frontend='conv_stdp_freesign'  —— S3.3-8 消融：带符号随机初始化 + free_sign=True，
                                     其余与 B2 完全相同（机制验证"符号约束⇒无条纹"）。
    frozen 经构造参数传入（S2.9 起成为 ConvSNNLayer 的正式构造参数；对 random_conv
    与原先"构造后赋值属性"完全等价——不涉及随机数消耗，B1/B2 初始化可比性不变）。

    S3.2a 调参窗：thr_range / fire_rate / ip_rate / stdp_rate 四个超参改为经
    model 属性配置（conv_thr_range / conv_fire_rate 为 "lo,hi" 字符串对，
    conv_ip_rate / conv_stdp_rate 为标量），缺省时取原硬编码默认值，默认行为不变。

    S2.11 规则锦标赛 Round 1：bcm_gate / rank_push / oja_decay / attractor 四个
    规则手术开关（及参数 bcm_alpha / rank_delta / rank_k）经 model 属性透传到
    ConvSNNLayer，缺省全关 = B2 主组合行为不变；仅训练路径（calc_stdp_conv）消费，
    推理前向不受任何影响。

    fork G（S3.2 主表前置）：
    - conv_thr_min：ITP 阈值地板（apply_itp_conv 的 clamp 下界），默认 0.0 行为
      不变；free-sign 变体允许负值解除 thr 触底卡死（s210_b6_preview.md §3）。
    - itp_on_frozen：model 属性为 True 时在冻结前端上保留 ITP 阈值自适应
      （B1+ITP 主表行：random_conv + ITP，与 B5 同待遇的 ITP 匹配对照，PLAN S3.2
      第 6 条）。B5 的 itp_on_frozen 仍由 load_gabor_weights 内部设置（S2.9 修订）。
    """
    frontend = getattr(model, 'frontend', 'none')
    ConvSNNLayer = _layer_mod().ConvSNNLayer
    free_sign = frontend in ('gabor_stdp_freesign', 'conv_stdp_freesign')
    layer = ConvSNNLayer(
        input_dims=dims,
        in_channels=1,
        out_channels=int(getattr(model, 'conv_channels', 32)),
        kernel_size=int(getattr(model, 'conv_kernel', 5)),
        thr_range=_parse_pair(getattr(model, 'conv_thr_range', '0,0.5')),
        thr_min=float(getattr(model, 'conv_thr_min', 0.0)),
        fire_rate=_parse_pair(getattr(model, 'conv_fire_rate', '0.2,0.9')),
        ip_rate=float(getattr(model, 'conv_ip_rate', 0.15)),
        stdp_rate=float(getattr(model, 'conv_stdp_rate', 0.005)),
        wta_mode=getattr(model, 'wta_mode', 'local'),
        wta_block=int(getattr(model, 'wta_block', 4)),
        device=device,
        inference=inference,
        frozen=frontend in ('random_conv', 'gabor'),
        free_sign=free_sign,
        # S2.11 Round 1 规则开关（默认全关 = B2 行为不变；推理侧不使用但保持属性一致）
        bcm_gate=bool(getattr(model, 'bcm_gate', False)),
        bcm_alpha=float(getattr(model, 'bcm_alpha', 0.001)),
        rank_push=bool(getattr(model, 'rank_push', False)),
        rank_delta=float(getattr(model, 'rank_delta', 0.4)),
        rank_k=int(getattr(model, 'rank_k', 2)),
        oja_decay=bool(getattr(model, 'oja_decay', False)),
        attractor=bool(getattr(model, 'attractor', False)),
    )
    if frontend == 'gabor':
        # 推理侧同样载入：值与 state_dict 中的保存值逐元素一致（Gabor 组全程确定性），
        # load_model 加载后与训练侧无差异
        _gabor_mod().load_gabor_weights(layer)
    elif frontend == 'gabor_stdp':
        # B6a 分解载入（训练+推理两侧同路径；推理侧随后被 state_dict 覆盖，无害）
        _gabor_mod().load_gabor_weights_decomposed(layer)
    elif frontend == 'gabor_stdp_freesign':
        _gabor_mod().load_gabor_weights_signed(layer)
    # fork G（S3.2 主表 B1+ITP 行）：冻结前端 + ITP 阈值自适应。Gabor 路径的
    # itp_on_frozen 由 load_gabor_weights 设置；此处覆盖 random_conv 等其余冻结
    # 前端（model.itp_on_frozen=True 时开启）。只影响训练分发（VPRTempoTrain
    # 对 frozen+itp_on_frozen 层仍跑 train_conv_layer），权重始终冻结。
    if bool(getattr(model, 'itp_on_frozen', False)):
        layer.itp_on_frozen = True
    return layer


def conv_forward(layer, spikes_flat):
    """平向量 [1, H*W] → conv 前向 → pooled_flat（prev_layers 冻结前向 / 推理特征点用）"""
    import torch
    spikes_flat = spikes_flat.to(layer.w.weight.device)  # 与层设备对齐（GPU/CPU 混用安全）
    with torch.no_grad():
        out = layer(layer.reshape_input(spikes_flat))
    return out.pooled_flat
