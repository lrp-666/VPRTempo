# MIT License
#
# Copyright (c) 2023 Adam Hines, Peter G Stratton, Michael Milford, Tobias Fischer
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
【模块级注释 Part 0】文件整体说明
================================================================================
文件名    : VPRTempoTrain.py
功能定位  : VPRTempo 的 fp32 训练模型定义与训练流程控制
核心对应  :
  - 论文公式 (3) 学习率退火      → _anneal_learning_rate()
  - 论文表 I 超参数              → __init__() 中 feature_layer / output_layer
  - 论文 III-A 网络架构          → feature_layer (LI→LF) + output_layer (LF→LO)
  - 论文 III-C 高效实现          → train_model() 逐层训练 + 多模块并行
  - 论文公式 (6) Spike Forcing   → train_model() 中 idx 的计算与传入
调用关系  : 由 main.py 调用 train_new_model() 启动完整训练流程
================================================================================
"""

# -----------------------------------------------------------------------------
# 【行级注释】标准库导入
# -----------------------------------------------------------------------------
import os       # 操作系统接口：路径拼接、文件存在性检查
import gc       # 垃圾回收：训练结束后手动释放显存
import torch    # PyTorch 核心库：张量运算、神经网络、CUDA 支持
import sys      # 系统接口：sys.exit() 用于退出程序
import random   # IDEA1 S1.3：worker_init_fn 中固定 worker 随机种子

import numpy as np                     # NumPy：数值计算，此处用于 np.mod 取模
import torch.nn as nn                  # PyTorch 神经网络模块：nn.Module 基类
import vprtempo.src.blitnet as bn      # 自定义 SNN 核心层：SNNLayer、STDP、钳制函数
import torchvision.transforms as transforms  # torchvision 图像变换：Compose 组合预处理

from tqdm import tqdm                    # 进度条库：训练循环可视化
from torch.utils.data import DataLoader  # PyTorch 数据加载器：批量、多进程、打乱
from vprtempo.src.loggers import model_logger         # 模型配置日志打印函数
from vprtempo.src.dataset import CustomImageDataset, ProcessImage  # 自定义数据集与图像预处理
from vprtempo.src import conv_frontend as cf  # IDEA1 S2.5：卷积前端桥接（frontend='none' 时零开销）


# ================================================================================
# 【模块级注释 Part 1】VPRTempoTrain 类 —— fp32 训练模型定义
# ================================================================================
# 类定位   : 继承自 nn.Module，是 VPRTempo 训练时的核心类
# 模块概念 : 一个 VPRTempoTrain 实例对应一个"专家模块"（Expert Module）
# 多模块   : 当 database_places > max_module 时，main.py 会创建多个实例
# 架构     : 两层 SNN —— feature_layer (稀疏连接) + output_layer (全连接)
# ================================================================================
class VPRTempoTrain(nn.Module):
    def __init__(self,  # self,用来访问实例属性和方法
                 args,  # argparse.Namespace，命令行参数对象，包含所有用户配置的超参数和选项
                 dims,  # 图像缩放尺寸 [H, W]，例如 [56, 56]，用于计算输入和特征层维度
                 logger, # 日志记录器对象，用于输出训练过程中的信息和调试日志
                 num_modules, # 总模块数（main.py 根据 database_places/max_module 计算），用于训练流程控制
                 out_dim,     # 每个标准模块的输出神经元数（地点数），例如 500，定义 output_layer 的输出维度
                 out_dim_remainder=None # 最后一个模块的输出神经元数（余数模块），当最后一个模块的地点数不足 max_module 时使用，例如 136
                 ):
        """
        ================================================================================
        【函数级注释】构造函数 __init__
        ================================================================================
        功能     : 初始化单个训练模块的网络结构、超参数、数据路径、设备
        参数说明 :
            args              — argparse.Namespace，命令行所有参数
            dims              — 图像缩放尺寸 [H, W]，默认 [56, 56]
            logger            — 日志记录器对象
            num_modules       — 总模块数（main.py 根据 database_places/max_module 计算）
            out_dim           — 每个标准模块的输出神经元数（地点数）
            out_dim_remainder — 最后一个模块的输出神经元数（余数模块）
        关键计算 :
            input   = H * W
            feature = input * 2
            T       = max_module * location_repeat * epoch  （总时间步，用于退火）
        ================================================================================
        """
        # -----------------------------------------------------------------------------
        # 【行级注释】调用父类 nn.Module 的构造函数，注册子模块、参数等
        # -----------------------------------------------------------------------------
        super(VPRTempoTrain, self).__init__()

        # ----------------------------------------
        # 【模块级子注释】解析并存储命令行参数
        # ----------------------------------------
        # 将所有 args 属性逐一复制到 self，方便后续通过 self.xxx 直接访问
        # 例如 self.epoch、self.filter、self.max_module 等
        # ----------------------------------------
        self.args = args                          # 保留原始 args 对象引用
        for arg in vars(args):                    # vars(args) 将 Namespace 转为字典
            setattr(self, arg, getattr(args, arg))  # self.arg = args.arg
        setattr(self, 'dims', dims)               # 将 dims 也作为实例属性

        # ----------------------------------------
        # 【模块级子注释】自动选择计算设备（GPU / MPS / CPU）
        # ----------------------------------------
        # cuda:0  — NVIDIA GPU，训练速度最快
        # mps     — Apple Silicon GPU（Metal Performance Shaders）
        # cpu     — fallback，无加速硬件时使用
        # ----------------------------------------
        if torch.cuda.is_available():
            self.device = "cuda:0"                # 优先使用 CUDA
        elif torch.backends.mps.is_available():
            self.device = "mps"                   # 次优先使用 Apple MPS
        else:
            self.device = "cpu"                   # 最后 fallback 到 CPU
        self.logger = logger                      # 绑定日志器
        self.num_modules = num_modules            # 存储总模块数

        # ----------------------------------------
        # 【模块级子注释】数据集 CSV 文件路径生成
        # ----------------------------------------
        # 训练时从 database_dirs 指定的文件夹中读取图像
        # 每个文件夹对应一个 CSV 标注文件，存放在 ./vprtempo/dataset/ 下
        # 文件名格式：<dataset>-<dir>.csv，例如 nordland-spring.csv
        # ----------------------------------------
        fields = self.database_dirs.split(',')    # 按逗号拆分多个文件夹名
        if len(fields) > 1:
            # 多文件夹情况：生成 CSV 路径列表
            self.dataset_file = []
            for field in fields:
                csv_path = os.path.join('./vprtempo/dataset', f'{self.dataset}-{field}' + '.csv')
                self.dataset_file.append(csv_path)
        else:
            # 单文件夹情况：生成单个 CSV 路径字符串
            self.dataset_file = os.path.join('./vprtempo/dataset', f'{self.dataset}-{self.database_dirs}' + '.csv')

        # ----------------------------------------
        # 【模块级子注释】层管理字典 —— 记录层的添加顺序
        # ----------------------------------------
        # layer_dict    : 映射 {层名称 -> 顺序索引}，用于按序逐层训练
        # layer_counter : 自增计数器，每 add_layer 一次 +1
        # ----------------------------------------
        self.layer_dict = {}
        self.layer_counter = 0

        # ----------------------------------------
        # 【模块级子注释】网络架构维度计算
        # ----------------------------------------
        # VPRTempo 默认架构（对应论文 III-A）：
        #   输入层 (LI)   = H * W
        #   特征层 (LF)   = 2 * LI
        #   输出层 (LO)   = 每个模块的地点数
        # 例：dims=[56,56] → input=3136, feature=6272
        # ----------------------------------------
        # -----------------------------------------------------------------------------
        # 【IDEA1 S2.5】conv 前端注册（frontend != 'none' 时）
        # -----------------------------------------------------------------------------
        # conv 层必须在 feature_layer 之前加入 layer_dict（顺序 0 → 逐层训练时先训 conv）。
        # 维度重算（ADR-1）：feature_layer 的输入不再是 H*W，而是 conv 池化后的 flat_dim
        # （28×28 / C=32 / k=5 / local WTA 4×4 → 32×6×6 = 1152）。
        # frontend='none'（B0 默认路径）时此分支完全不触发，行为与原来一致。
        # -----------------------------------------------------------------------------
        if getattr(self, 'frontend', 'none') != 'none':
            conv_layer = cf.build_conv_layer(self, dims, self.device, inference=False)
            setattr(self, 'conv_layer', conv_layer)
            self.layer_dict['conv_layer'] = self.layer_counter
            self.layer_counter += 1
            self.input = conv_layer.flat_dim      # 维度重算（ADR-1）
        else:
            self.input = int(dims[0] * dims[1])   # 图像像素数 = 输入神经元数
        self.feature = int(self.input * 2)        # 特征层神经元数 = 2 * 输入
        if not out_dim_remainder is None:
            self.output = out_dim_remainder       # 余数模块使用实际的余数输出数
        else:
            self.output = out_dim                 # 标准模块使用标准输出数

        # ----------------------------------------
        # 【模块级子注释】计算总训练时间步 T（学习率退火的分母）
        # ----------------------------------------
        # 公式对应：论文公式 (3) 中的 T
        # 计算方式：T = 每个模块最大地点数 × 训练遍历次数 × epoch 数
        # 例：max_module=500, location_repeat=2 (spring+fall), epoch=4
        #     → T = 500 * 2 * 4 = 4000 步
        # ----------------------------------------
        # 处理 database_dirs，去除多余空格，得到文件夹列表
        self.database_dirs = [dir.strip() for dir in self.database_dirs.split(',')]  
        # 训练时遍历的文件夹数（季节数）
        self.location_repeat = len(self.database_dirs)  
        # 计算总时间步 T，优先使用 out_dim_remainder 计算最后一个模块的 T，否则使用 max_module 计算标准模块的 T
        if not out_dim_remainder is None:
            self.T = int(out_dim_remainder * self.location_repeat * self.epoch)
        else:
            self.T = int(self.max_module * self.location_repeat * self.epoch)

        # ================================================================================
        # 【模块级子注释】定义可训练层（对应论文表 I 超参数）
        # ================================================================================
        # feature_layer : LI → LF，稀疏连接，启用 STDP + ITP
        #   - thr_range : 阈值初始化范围 [0, 0.5]    （表 I: θ_max = 0.5）
        #   - fire_rate : 目标发放率范围 [0.2, 0.9]  （表 I: f_min, f_max）
        #   - ip_rate   : ITP 初始学习率 0.15        （表 I: η_init_ITP）
        #   - stdp_rate : STDP 初始学习率 0.005       （表 I: η_init_STDP）
        #   - p         : 连接概率 [0.1, 0.5]         （表 I: P_exc, P_inh）
        # output_layer  : LF → LO，全连接，启用 Spike Forcing（监督读出）
        #   - p=[1.0,1.0]  : 全连接，每个特征神经元都连接到所有输出神经元
        #   - spk_force=True: 启用 Spike Forcing（论文公式 6）
        # ================================================================================
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature],
            thr_range=[0, 0.5],
            fire_rate=[0.2, 0.9],
            ip_rate=0.15,
            stdp_rate=0.005,
            p=[0.1, 0.5],
            device=self.device
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            ip_rate=0.15,
            stdp_rate=0.005,
            p=[1.0, 1.0],
            spk_force=True,
            device=self.device
        )
        
    def add_layer(self, name, **kwargs):
        """
        ================================================================================
        【函数级注释】动态添加 SNNLayer
        ================================================================================
        功能     : 通过 setattr 将 bn.SNNLayer 实例附加到当前对象，并在 layer_dict 中记录顺序
        参数说明 :
            name    — 层名称字符串，例如 'feature_layer'、'output_layer'
            **kwargs— 传递给 bn.SNNLayer 构造函数的键值参数
        异常处理 :
            若 name 已存在于 layer_dict，抛出 ValueError 防止重复添加
        调用位置 :
            仅在 __init__() 中被调用两次，分别添加 feature_layer 和 output_layer
        ================================================================================
        """
        # -----------------------------------------------------------------------------
        # 【行级注释】检查层名是否已存在，防止重复注册导致状态字典混乱
        # -----------------------------------------------------------------------------
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        
        # -----------------------------------------------------------------------------
        # 【行级注释】实例化 SNNLayer 并将其作为当前对象的属性
        # 例：执行后 self.feature_layer = bn.SNNLayer(...)
        # -----------------------------------------------------------------------------
        setattr(self, name, bn.SNNLayer(**kwargs))
        
        # -----------------------------------------------------------------------------
        # 【行级注释】记录层顺序，保证逐层训练时按正确顺序遍历
        # 例：self.layer_dict = {'feature_layer': 0, 'output_layer': 1}
        # -----------------------------------------------------------------------------
        self.layer_dict[name] = self.layer_counter
        self.layer_counter += 1
        
    def model_logger(self):
        """
        ================================================================================
        【函数级注释】输出模型配置日志
        ================================================================================
        功能     : 调用 vprtempo.src.loggers.model_logger() 打印网络结构和超参数
        调用时机 : 训练开始前，由 main.py 调用，便于用户确认配置是否正确
        ================================================================================
        """
        model_logger(self)

    # ================================================================================
    # 【模块级注释 Part 2】学习率退火 —— 论文公式 (3) 的实现
    # ================================================================================
    # 理论来源 : VPRTempo 公式 (3):
    #            η_STDP(t) = η_init_STDP * (1 - t/T)^2
    #            η_ITP(t)  = η_init_ITP  * (1 - t/T)^2
    # 物理意义 : 训练初期学习率较大，快速捕捉粗粒度特征；
    #           训练后期学习率衰减，精细调整权重，防止震荡。
    # 实现注意 : 代码中使用 (T-mod)^2 / T^2，与 (1-t/T)^2 数学等价。
    # 触发频率 : 每 100 个时间步更新一次，减少计算开销
    # ================================================================================
    def _anneal_learning_rate(self, 
                              layer, 
                              mod, 
                              itp, 
                              stdp
                              ):
        """
        ================================================================================
        【函数级注释】学习率多项式退火
        ================================================================================
        功能     : 根据当前时间步 mod 和总时间步 T，按公式 (3) 更新 layer 的学习率
        参数说明 :
            layer — 当前正在训练的 SNNLayer 实例（包含 eta_ip 和 eta_stdp 属性）
            mod   — 当前时间步计数器（从 0 开始，每个训练样本递增 1）
            itp   — ITP 初始学习率（退火前的基准值，已 detach 的副本）
            stdp  — STDP 初始学习率（退火前的基准值，已 detach 的副本）
        返回值   : 更新学习率后的 layer 对象
        更新频率 : 每 100 步更新一次；其他时间步直接返回原 layer，不修改
        ================================================================================
        """
        # -----------------------------------------------------------------------------
        # 【行级注释】每 100 个时间步执行一次退火，减少计算开销
        # np.mod(mod, 100) == 0 表示 mod 能被 100 整除（包括 mod=0 的初始状态）
        # -----------------------------------------------------------------------------
        if np.mod(mod, 100) == 0:
            # -------------------------------------------------------------------------
            # 【行级注释】计算退火比例因子 pt = ((T - mod) / T)^2 = (1 - mod/T)^2
            # mod=0  时 pt=1.0，学习率保持初始值（最大）
            # mod=T  时 pt=0.0，学习率衰减到 0
            # -------------------------------------------------------------------------
            pt = pow(float(self.T - mod) / self.T, 2)
            
            # -------------------------------------------------------------------------
            # 【行级注释】逐元素乘法更新当前层的学习率
            # layer.eta_ip   : ITP 学习率，控制阈值可塑性的步长
            # layer.eta_stdp : STDP 学习率，控制权重更新的步长
            # torch.mul 进行逐元素乘法，保持张量形状不变
            # -------------------------------------------------------------------------
            layer.eta_ip = torch.mul(itp, pt)     # ITP 退火后的新学习率
            layer.eta_stdp = torch.mul(stdp, pt)  # STDP 退火后的新学习率
            
        return layer

    # ================================================================================
    # 【模块级注释 Part 3】单模块单层训练循环 —— 核心训练逻辑
    # ================================================================================
    # 训练策略 : BLiTNet 的逐层训练（Layer-wise Training）
    #   1. 固定已训练层（prev_layers），只训练当前层（layer）
    #   2. 对每个训练样本：前向传播 → clamp → calc_stdp → 退火
    #   3. 输出层通过 idx 传入 Spike Forcing 的神经元索引（论文公式 6）
    # 内存优化 : 已训练层前向传播使用 torch.no_grad()，不保存计算图
    # ================================================================================
    def train_model(self, 
                    train_loader, 
                    layer, 
                    model, 
                    model_num, 
                    prev_layers=None
                    ):
        """
        ================================================================================
        【函数级注释】训练网络的某一层（单个模块）
        ================================================================================
        功能     : 对单个模块的某一层执行完整训练循环
        参数说明 :
            train_loader — DataLoader，每次迭代返回 (spikes, labels)
                           spikes: 脉冲编码后的图像张量 [1, input]
                           labels: 图像全局索引（用于计算 Spike Forcing 的 idx）
            layer        — 当前要训练的 SNNLayer（feature_layer 或 output_layer）
            model        — 当前 VPRTempoTrain 实例（用于获取 prev_layers 层对象）
            model_num    — 当前模块序号（0, 1, 2...），用于计算 idx_scale
            prev_layers  — 已经训练好的层名称列表，前向传播时固定参数不更新
        核心流程 :
            for epoch in range(self.epoch):
                for (spikes, labels) in train_loader:
                    1. 若 prev_layers 存在，前向传播通过已训练层（no_grad 模式）
                    2. 当前层前向传播 → clamp_spikes 钳制
                    3. calc_stdp(pre_spike, spikes, spikes_noclp, layer, idx)
                    4. _anneal_learning_rate 更新学习率
        ================================================================================
        """
        # -----------------------------------------------------------------------------
        # 【行级注释】初始化 tqdm 进度条
        # total=self.T    : 进度条总长度 = 总时间步
        # desc            : 进度条描述，显示当前是第几个模块
        # position=0      : 进度条位置，防止多模块时输出错乱
        # -----------------------------------------------------------------------------
        pbar = tqdm(total=self.T,
                    desc=f"Module {model_num+1}",
                    position=0)
        
        # ----------------------------------------
        # 【模块级子注释】保存初始学习率用于退火
        # ----------------------------------------
        # detach() 创建与计算图分离的独立副本，
        # 防止退火过程中修改原始值，保证每次退火都基于初始值计算
        # ----------------------------------------
        init_itp = layer.eta_ip.detach()      # ITP 初始学习率副本
        init_stdp = layer.eta_stdp.detach()   # STDP 初始学习率副本
        mod = 0                                # 时间步计数器，每个训练样本递增 1
        
        # ----------------------------------------
        # 【模块级子注释】计算模块偏移量 idx_scale
        # ----------------------------------------
        # 多模块场景下，每个模块负责不同的地点范围：
        #   module 0 : 地点 0           ~ max_module-1
        #   module 1 : 地点 max_module  ~ 2*max_module-1
        # idx_scale 用于将全局地点标签转换为模块内的局部神经元索引
        # 计算式：idx_scale = (max_module * filter) * model_num
        # ----------------------------------------
        idx_scale = (self.max_module * self.filter) * model_num

        # ----------------------------------------
        # 【模块级子注释】外层 epoch 循环
        # ----------------------------------------
        # self.epoch 默认为 4，即完整遍历数据集 4 次
        # 每次遍历都会重新从 DataLoader 中取数据（shuffle=True 打乱顺序）
        # ----------------------------------------
        for _ in range(self.epoch):
            
            # -------------------------------------------------------------------------
            # 【行级注释】内层数据遍历循环，每次取出一个样本 (batch_size=1)
            # spikes : 脉冲编码后的图像张量，形状 [1, input]
            # labels : 图像的全局索引张量，形状 [1]
            # -------------------------------------------------------------------------
            for spikes, labels in train_loader:
                # ---------------------------------------------------------------------
                # 【行级注释】将数据迁移到计算设备（GPU / MPS / CPU）
                # .to(self.device) 是异步操作，不会阻塞 CPU
                # ---------------------------------------------------------------------
                spikes, labels = spikes.to(self.device), labels.to(self.device)
                
                # -----------------------------------------------------------------
                # 【模块级子注释】计算 Spike Forcing 的神经元索引 idx
                # -----------------------------------------------------------------
                # 公式：idx = round((labels - idx_scale) / filter)
                # 物理意义：将全局图像标签映射到当前模块的输出神经元索引
                # 示例：全局标签=1508，module 1（idx_scale=1000），filter=8
                #       idx = (1508 - 1000) / 8 = 63.5 → round → 64
                # 结果：该图像应该强制激活输出层第 64 个神经元
                # -----------------------------------------------------------------
                idx = torch.round((labels - idx_scale) / self.filter)
                
                # -----------------------------------------------------------------
                # 【模块级子注释】前向传播通过已训练层（固定参数）
                # -----------------------------------------------------------------
                # BLiTNet 逐层训练核心：
                #   - 训练 feature_layer 时：prev_layers=None，直接输入原始脉冲
                #   - 训练 output_layer 时：prev_layers=['feature_layer']
                #     先通过 feature_layer 提取特征，再用特征训练输出层
                # torch.no_grad() 确保：
                #   1. 不计算梯度（节省显存和计算）
                #   2. 不更新已训练层的权重（固定参数）
                # -----------------------------------------------------------------
                if prev_layers:
                    with torch.no_grad():         # 禁用梯度计算上下文
                        for prev_layer_name in prev_layers:  # 遍历所有已训练层
                            prev_layer = getattr(model, prev_layer_name)  # 获取层对象
                            # 【IDEA1 S2.5】conv 层走 conv 前向路径（reshape→conv→WTA→池化→flatten）
                            if cf.is_conv_layer(prev_layer):
                                spikes = cf.conv_forward(prev_layer, spikes)
                            else:
                                spikes = self.forward(spikes, prev_layer)      # 前向传播：W*x
                                spikes = bn.clamp_spikes(spikes, prev_layer)   # 钳制到 [0, 0.9]
                else:
                    prev_layer = None             # 无已训练层时设为 None，供 calc_stdp 使用
                
                # -----------------------------------------------------------------
                # 【模块级子注释】当前层前向传播与 STDP 计算
                # -----------------------------------------------------------------
                pre_spike = spikes.detach()                # 保存前层脉冲，供 STDP 使用（前突触活动）
                spikes = self.forward(spikes, layer)        # 当前层前向传播：计算 W*x（尚未减阈值）
                spikes_noclp = spikes.detach()             # 保存未钳制的值，供 Homeostasis 使用
                #关于这个地方为什么要保留未钳制的值，
                #是因为在 calc_stdp 中会同时使用钳制后的 spikes 和 未钳制的 spikes_noclp 来分别计算 STDP 权重更新和 Homeostasis 稳态归一化。
                #钳制后的 spikes 用于权重更新，确保神经元发放率在目标范围内；
                # 而未钳制的 spikes_noclp 则用于 Homeostasis 计算，反映神经元的真实活动水平，帮助调整阈值以维持稳定的发放率。
                spikes = bn.clamp_spikes(spikes, layer)    # 钳制到 [0, fire_rate_max=0.9]
                
                # -----------------------------------------------------------------
                # 【行级注释】调用 blitnet.py 中的 calc_stdp，执行完整学习规则
                # 包含：
                #   1. STDP 权重更新（突触可塑性）
                #   2. ITP  阈值调整（内在阈值可塑性）
                #   3. Homeostasis 稳态归一化（防止某些神经元过活跃）
                #   4. Spike Forcing（如果 layer.spk_force=True，通过 idx 强制发放）
                # -----------------------------------------------------------------
                layer = bn.calc_stdp(pre_spike, spikes, spikes_noclp, layer, idx, prev_layer=prev_layer)
                
                # -----------------------------------------------------------------
                # 【行级注释】学习率退火，每 100 步更新一次
                # -----------------------------------------------------------------
                layer = self._anneal_learning_rate(layer, mod, init_itp, init_stdp)
                
                # -----------------------------------------------------------------
                # 【行级注释】更新计数器和进度条
                # -----------------------------------------------------------------
                mod += 1           # 时间步计数器 +1
                pbar.update(1)     # 进度条前进一格

        # -----------------------------------------------------------------------------
        # 【行级注释】关闭进度条，释放资源
        # -----------------------------------------------------------------------------
        pbar.close()

        # ----------------------------------------
        # 【模块级子注释】显存清理（仅 CUDA 设备）
        # ----------------------------------------
        # torch.cuda.empty_cache() : 释放 PyTorch CUDA 缓存分配器中的未使用显存
        # gc.collect()             : Python 垃圾回收，清理循环引用等
        # MPS 和 CPU 不需要此操作
        # ----------------------------------------
        if self.device == "cuda:0":
            torch.cuda.empty_cache()
            gc.collect()

    def forward(self, spikes, layer):
        """
        ================================================================================
        【函数级注释】单层前向传播
        ================================================================================
        功能     : 执行单层的矩阵乘法 y = x · W^T
        参数说明 :
            spikes — 输入脉冲张量，形状 [batch_size, input_dim]
            layer  — SNNLayer 实例，包含权重矩阵 W
        返回值   : 加权求和后的张量，形状 [batch_size, output_dim]
        理论对应 : 论文公式 (1) 中的 Σ_i x_i (W^+_ji - W^- ji) 部分
        注意     : 此处仅做矩阵乘法，尚未减去神经元阈值，阈值比较在 clamp_spikes 中完成
        ================================================================================
        """
        # -----------------------------------------------------------------------------
        # 【行级注释】layer.w 是 SNNLayer 中的 nn.Linear 实例
        # 调用 layer.w(spikes) 等价于执行 F.linear(spikes, layer.w.weight, layer.w.bias)
        # 即 y = x · W^T + b
        # -----------------------------------------------------------------------------
        spikes = layer.w(spikes)
        return spikes 
    
    def save_model(self, models, model_out):    
        """
        ================================================================================
        【函数级注释】保存多模块组合模型
        ================================================================================
        功能     : 将多个 module 的 state_dict 组合成一个字典保存
        参数说明 :
            models    — VPRTempoTrain 实例列表，每个实例对应一个模块
            model_out — 输出文件路径，例如 './vprtempo/models/nordland_VPRTempo.pth'
        保存格式 :
            state_dicts = {
                'model_0': model_0.state_dict(),
                'model_1': model_1.state_dict(),
                ...
            }
        加载方式 : 推理时通过 torch.load() 加载，键名为 model_0, model_1, ...
        ================================================================================
        """
        # -----------------------------------------------------------------------------
        # 【行级注释】遍历所有模块，收集各自的 state_dict
        # state_dict 是 PyTorch 模型参数的 OrderedDict，包含所有可学习参数和缓冲区
        # -----------------------------------------------------------------------------
        state_dicts = {}                # 用于存储所有模块的 state_dict，键名为 'model_0', 'model_1', ...
        # 遍历 models 列表，使用 enumerate 获取模块索引 i 和模块对象 model
        for i, model in enumerate(models):
            state_dicts[f'model_{i}'] = model.state_dict()
        
        # -----------------------------------------------------------------------------
        # 【行级注释】使用 torch.save 将组合字典序列化到磁盘
        # 默认使用 PyTorch 的 pickle 协议保存
        # -----------------------------------------------------------------------------
        torch.save(state_dicts, model_out)
            

# ================================================================================
# 【模块级注释 Part 4】独立工具函数
# ================================================================================
# 这些函数不属于 VPRTempoTrain 类，是模块级别的辅助函数
# ================================================================================

def check_pretrained_model(model_name):
    """
    ================================================================================
    【函数级注释】检查预训练模型是否存在
    ================================================================================
    功能     : 若指定路径已存在预训练模型文件，交互式询问用户是否重新训练
    参数说明 :
        model_name — 模型文件名，例如 'nordland_VPRTempo_IN3136_FN6272_DB500.pth'
    行为逻辑 :
        - 文件存在 → 提示输入 y/n
            - y → 返回 True，继续训练（覆盖旧模型）
            - n → 打印取消信息，调用 sys.exit() 退出程序
        - 文件不存在 → 不执行任何操作，隐式继续
    调用位置 : 由 main.py 在训练前调用，防止误覆盖已有模型
    ================================================================================
    """
    # -----------------------------------------------------------------------------
    # 【行级注释】拼接完整路径并检查文件存在性
    # -----------------------------------------------------------------------------
    if os.path.exists(os.path.join('./vprtempo/models', model_name)):
        # -------------------------------------------------------------------------
        # 【行级注释】文件已存在，向用户发起交互式确认
        # -------------------------------------------------------------------------
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        retrain = input(prompt).strip().lower()   # 读取用户输入并规范化
        if retrain == 'y':
            return True                            # 用户确认重新训练
        elif retrain == 'n':
            print('Training new model cancelled')  # 用户取消训练
            sys.exit()                             # 退出整个程序


# ================================================================================
# 【模块级注释 Part 5】train_new_model —— 完整训练流程总控（多模块 + 逐层）
# ================================================================================
# 函数定位 : 训练的总控入口，由 main.py 调用
# 核心逻辑 :
#   1. 为每个 module 划分独立的图像范围（user_input_ranges）
#   2. 逐层训练：先训练所有 module 的 feature_layer，再训练 output_layer
#   3. 每个 module 训练完成后移回 CPU 节省显存
#   4. 全部训练完成后保存组合模型
# ================================================================================
def train_new_model(models, model_name):
    """
    ================================================================================
    【函数级注释】训练新模型（支持多模块、逐层训练）
    ================================================================================
    功能     : 协调多个模块、多层网络的完整训练流程
    参数说明 :
        models     — VPRTempoTrain 实例列表，每个实例对应一个专家模块
        model_name — 保存模型的文件名（不含路径前缀）
    核心流程 :
        1. 准备图像预处理变换 ProcessImage
        2. 为每个 module 计算图像索引范围 user_input_ranges
        3. 逐层循环（feature_layer → output_layer）:
             a. 遍历所有模块
             b. 为当前模块创建 CustomImageDataset（限制图像范围）
             c. 创建 DataLoader（batch_size=1, shuffle=True）
             d. 调用 model.train_model() 训练当前层
             e. 训练完移回 CPU
        4. 所有层训练完成后切换 eval() 模式
        5. 调用 save_model() 保存组合模型
    ================================================================================
    """
    # -----------------------------------------------------------------------------
    # 【行级注释】以第一个模块为基准获取公共参数
    # 所有模块的 dims、patches、database_dirs 等参数相同，只需取 models[0]
    # -----------------------------------------------------------------------------
    model = models[0]
    
    # -----------------------------------------------------------------------------
    # 【行级注释】定义图像预处理流程
    # ProcessImage 内部执行：RGB→灰度 → Gamma校正 → 缩放 → 块归一化 → 脉冲编码
    # -----------------------------------------------------------------------------
    image_transform = transforms.Compose([
        ProcessImage(model.dims, model.patches, patch_norm=getattr(model, 'patch_norm', 'on') == 'on')
    ])
    
    # ----------------------------------------
    # 【模块级子注释】为每个 module 生成图像索引范围
    # ----------------------------------------
    # 目的：将完整数据库切分为互不重叠的子集，每个模块只处理自己的子集
    # 计算方式：
    #   range_temp = [start_idx, start_idx + (max_module - 1) * filter]
    # 示例：max_module=500, filter=8, num_modules=3
    #   module 0: [0,     0 + 499*8]     = [0,     3992]
    #   module 1: [4000,  4000 + 499*8]  = [4000,  7992]
    #   module 2: [8000,  8000 + 499*8]  = [8000,  11992]
    #    [
    #    [0, 3992],      # model_0 负责读取 CSV 里的第 0 到 3992 行
    #    [4000, 7992],   # model_1 负责读取 CSV 里的第 4000 到 7992 行
    #    [8000, 11992]   # model_2 负责读取 CSV 里的第 8000 到 11992 行
    #    ]
    # 注意：相邻模块的间隔 = filter，保证不重叠且连续
    # ----------------------------------------
    user_input_ranges = [] # 用于存储每个模块负责的图像索引范围
    start_idx = 0
    for _ in range(models[0].num_modules):
        # 计算当前模块的结束索引（注意是 (max_module-1)*filter，因为包含两端）
        range_temp = [start_idx, start_idx + ((models[0].max_module - 1) * models[0].filter)]
        # 将当前模块的图像索引范围添加到列表中
        user_input_ranges.append(range_temp)
        # 下一个模块的起始索引 = 当前结束索引 + filter
        start_idx = range_temp[1] + models[0].filter

    # ----------------------------------------
    # 【模块级子注释】逐层训练
    # ----------------------------------------
    # 策略：按 layer_dict 中的顺序（feature_layer → output_layer）逐层训练
    # trained_layers : 记录已经训练好的层名称
    # 训练 output_layer 时，feature_layer 已在 prev_layers 中，前向传播固定
    # ----------------------------------------
    trained_layers = []  # 用于记录已训练层的名称，供后续层训练时固定参数使用
    
    # -----------------------------------------------------------------------------
    # 【行级注释】按 layer_dict 中的顺序排序层名（feature_layer=0, output_layer=1）
    # sorted(..., key=lambda item: item[1]) 按索引值排序，保证 feature_layer 先训练
    # -----------------------------------------------------------------------------
    for layer_name, _ in sorted(models[0].layer_dict.items(), key=lambda item: item[1]):
        print(f"Training layer: {layer_name}")   # 打印当前正在训练的层名
        
        # -------------------------------------------------------------------------
        # 【行级注释】遍历所有模块，分别训练当前层
        # 注意：不同模块的 train_loader 数据范围不同，但层结构相同
        # -------------------------------------------------------------------------
        for i, model in enumerate(models):
            model.train()                         # 设置 PyTorch 训练模式（启用 dropout 等，此处主要用于标识）
            model.to(torch.device(model.device))  # 将模型参数迁移到计算设备
            layer = getattr(model, layer_name)    # 通过名称获取层对象（feature_layer 或 output_layer）
            
            # -----------------------------------------------------------------
            # 【行级注释】确定当前模块的最大样本数
            # 逻辑：
            #   - 若 database_places < max_module : 数据库地点数不足一个模块上限，用实际地点数
            #   - 若 output < max_module          : 最后一个余数模块，用余数输出数
            #   - 否则                            : 标准模块，用 max_module
            # -----------------------------------------------------------------
            if model.database_places < model.max_module:
                max_samples = model.database_places
            elif model.output < model.max_module:
                max_samples = model.output
            else:
                max_samples = model.max_module
            
            # -----------------------------------------------------------------
            # 【行级注释】为当前模块创建数据集
            # 关键参数：
            #   img_range   = user_input_ranges[i]  — 只加载当前模块负责的图像范围
            #   max_samples = max_samples           — 限制最大样本数
            #   test=False                         — 训练模式（合并多个 CSV，返回标签）
            # -----------------------------------------------------------------
            img_range = user_input_ranges[i]
            train_dataset = CustomImageDataset(
                annotations_file=models[0].dataset_file,   # CSV 标注文件路径（或路径列表）
                base_dir=models[0].data_dir,               # 图像基础目录
                img_dirs=models[0].database_dirs,          # 图像子目录列表
                transform=image_transform,                 # 图像预处理变换
                filter=models[0].filter,                   # 帧子采样步长
                skip=models[0].skip,                       # 起始处跳过的图像数
                test=False,                                 # False=训练模式（带标签）
                img_range=img_range,                        # 当前模块的图像索引范围
                max_samples=max_samples                     # 最大样本数限制
            )
            
            # -----------------------------------------------------------------
            # 【行级注释】创建 DataLoader
            # 关键配置：
            #   batch_size=1          — SNN 时序编码要求单样本逐个处理
            #   shuffle=True          — 打乱顺序，防止网络记住时序而非学习特征
            #   num_workers           — MPS 设备必须为 0（MPS 不支持多进程）
            #   persistent_workers    — 是否保持工作进程存活，此处设为 False
            # 【IDEA1 S1.3】model.seed 非 None 时附加 generator 与 worker_init_fn，
            #   使 shuffle 顺序与 worker 内随机序列可复现；为 None 时保持原行为。
            # -----------------------------------------------------------------
            if model.device == "mps":
                num_workers = 0
                persistent_workers = False
            else:
                num_workers = 4
                persistent_workers = False
            loader_kwargs = {}
            if getattr(model, 'seed', None) is not None:
                g = torch.Generator()
                g.manual_seed(model.seed)
                loader_kwargs["generator"] = g
                def _seed_worker(worker_id):
                    ws = (torch.initial_seed() + worker_id) % 2**32
                    np.random.seed(ws)
                    random.seed(ws)
                loader_kwargs["worker_init_fn"] = _seed_worker
            train_loader = DataLoader(
                train_dataset,
                batch_size=1,              # SNN 时序编码要求 batch_size=1
                shuffle=True,              # 打乱顺序，防止过拟合时序
                num_workers=num_workers,   # 数据加载的子进程数
                persistent_workers=persistent_workers,
                **loader_kwargs
            )
            
            # -----------------------------------------------------------------
            # 【IDEA1 S2.5】conv 层分发：训练方式与 SNNLayer 不同（无监督 STDP+ITP）
            #   - i == 0：正常训练（train_conv_layer）
            #   - i > 0 ：多模块共享前端——直接拷贝 module 0 训好的权重（S2.5 卡片：
            #             "通用视觉特征"，省算力且保持模块间特征空间一致）
            #   - frozen（B1 random_conv / B5 Gabor）：跳过训练，直接进已训练层列表
            # -----------------------------------------------------------------
            if cf.is_conv_layer(layer):
                if getattr(layer, 'frozen', False):
                    pass                                      # 冻结前端：不训练
                elif i == 0:
                    cf.train_conv_layer(train_loader, layer, model, model_num=i)
                else:
                    layer.load_state_dict(models[0].conv_layer.state_dict())
                model.to(torch.device("cpu"))
                continue
            
            # -----------------------------------------------------------------
            # 【行级注释】调用当前模块的 train_model 训练当前层
            # prev_layers=trained_layers 确保已训练层参数固定
            # -----------------------------------------------------------------
            model.train_model(train_loader, layer, model, i, prev_layers=trained_layers)
            
            # -----------------------------------------------------------------
            # 【行级注释】当前模块当前层训练完成后，移回 CPU 释放 GPU 显存
            # 这样多个模块可以共用同一块 GPU，避免显存溢出
            # -----------------------------------------------------------------
            model.to(torch.device("cpu"))
        
        # -----------------------------------------------------------------------------
        # 【行级注释】当前层在所有模块上训练完成后，加入已训练层列表
        # 下一层训练时，这些层将作为 prev_layers 被固定
        # -----------------------------------------------------------------------------
        trained_layers.append(layer_name)
    
    # -----------------------------------------------------------------------------
    # 【行级注释】所有层训练完成后，将所有模块切换到评估模式
    # eval() 会禁用 dropout、batchnorm 的统计更新等
    # -----------------------------------------------------------------------------
    for model in models:
        model.eval()
    
    # -----------------------------------------------------------------------------
    # 【行级注释】保存训练好的多模块组合模型到磁盘
    # 保存路径：./vprtempo/models/<model_name>
    # -----------------------------------------------------------------------------
    model.save_model(models, os.path.join('./vprtempo/models', model_name))
