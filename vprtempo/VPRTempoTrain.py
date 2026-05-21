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
VPRTempoTrain.py —— fp32 训练模型
================================================================================
理论对应：
  - 公式 (3) 学习率退火      → _anneal_learning_rate()
  - 表 I 超参数              → __init__() 中 feature_layer / output_layer 的定义
  - III-A 网络架构           → feature_layer (LI→LF) + output_layer (LF→LO)
  - III-C 高效实现           → train_model() 中逐层训练 + 多模块并行
  - 公式 (6) Spike Forcing   → train_model() 中 idx 的计算与传入
================================================================================
"""

import os
import gc
import torch
import sys

import numpy as np
import torch.nn as nn
import vprtempo.src.blitnet as bn
import torchvision.transforms as transforms

from tqdm import tqdm
from torch.utils.data import DataLoader
from vprtempo.src.loggers import model_logger
from vprtempo.src.dataset import CustomImageDataset, ProcessImage


# ================================================================================
# Part 1: VPRTempoTrain 类 —— fp32 训练模型定义
# ================================================================================
# 该类继承自 nn.Module，是 VPRTempo 训练时的核心类。
# 一个 VPRTempoTrain 实例对应一个"专家模块"（Expert Module），
# 当 database_places > max_module 时，main.py 会创建多个实例。
# ================================================================================
class VPRTempoTrain(nn.Module):
    def __init__(self, args, dims, logger, num_modules, out_dim, out_dim_remainder=None):
        """
        ================================================================================
        函数层说明：VPRTempoTrain 构造函数
        ================================================================================
        参数说明：
            args              : 命令行参数 Namespace，包含 database_dirs, query_dir 等
            dims              : 图像缩放尺寸 [H, W]，默认 [56, 56]
            logger            : 日志记录器
            num_modules       : 总模块数（main.py 计算得出）
            out_dim           : 每个模块的输出神经元数（地点数）
            out_dim_remainder : 最后一个模块的输出神经元数（如果不是整除的话）
        ================================================================================
        """
        super(VPRTempoTrain, self).__init__()

        # ----------------------------------------
        # 逐行说明：解析并存储命令行参数
        # ----------------------------------------
        self.args = args
        for arg in vars(args):
            # 将所有 args 属性复制到 self，方便后续通过 self.xxx 访问
            setattr(self, arg, getattr(args, arg))
        setattr(self, 'dims', dims)

        # ----------------------------------------
        # 逐行说明：设备选择
        # ----------------------------------------
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.logger = logger
        self.num_modules = num_modules

        # ----------------------------------------
        # 逐行说明：数据集 CSV 文件路径生成
        # ----------------------------------------
        # 训练时需要从 database_dirs 中读取图像，每个 dir 对应一个 CSV
        fields = self.database_dirs.split(',')
        if len(fields) > 1:
            self.dataset_file = []
            for field in fields:
                self.dataset_file.append(os.path.join('./vprtempo/dataset', f'{self.dataset}-{field}' + '.csv'))
        else:
            self.dataset_file = os.path.join('./vprtempo/dataset', f'{self.dataset}-{self.database_dirs}' + '.csv')

        # ----------------------------------------
        # 逐行说明：层管理字典
        # ----------------------------------------
        # layer_dict 记录层的名称到索引的映射，保证逐层训练时的顺序正确
        self.layer_dict = {}
        self.layer_counter = 0

        # ----------------------------------------
        # 逐行说明：网络架构维度计算
        # ----------------------------------------
        # VPRTempo 默认架构：输入层 = H*W，特征层 = 2*输入层
        # 例如 dims=[56,56] → input=3136, feature=6272
        self.input = int(dims[0]*dims[1])
        self.feature = int(self.input * 2)
        if not out_dim_remainder is None:
            self.output = out_dim_remainder
        else:
            self.output = out_dim

        # ----------------------------------------
        # 逐行说明：计算总训练时间步 T
        # ----------------------------------------
        # T = 每个模块的地点数 × 训练遍历次数(季节数) × epoch 数
        # 例如：max_module=500, location_repeat=2 (spring+fall), epoch=4
        #       → T = 500 * 2 * 4 = 4000 步
        # 这是学习率退火公式 (3) 中的分母 T
        self.database_dirs = [dir.strip() for dir in self.database_dirs.split(',')]
        self.location_repeat = len(self.database_dirs)
        if not out_dim_remainder is None:
            self.T = int(out_dim_remainder * self.location_repeat * self.epoch)
        else:
            self.T = int(self.max_module * self.location_repeat * self.epoch)

        # ================================================================================
        # 逐行说明：定义可训练层（对应表 I 超参数）
        # ================================================================================
        # feature_layer: LI → LF，稀疏连接，启用 STDP + ITP
        # output_layer : LF → LO，全连接，启用 Spike Forcing（监督读出）
        # ================================================================================
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature],
            thr_range=[0, 0.5],      # 表 I: θ_max = 0.5
            fire_rate=[0.2, 0.9],    # 表 I: f_min, f_max
            ip_rate=0.15,            # 表 I: η_init_ITP
            stdp_rate=0.005,         # 表 I: η_init_STDP
            p=[0.1, 0.5],            # 表 I: P_exc, P_inh
            device=self.device
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            ip_rate=0.15,
            stdp_rate=0.005,
            p=[1.0, 1.0],            # 全连接：概率为 1
            spk_force=True,          # 启用 Spike Forcing（公式 6）
            device=self.device
        )
        
    def add_layer(self, name, **kwargs):
        """
        ================================================================================
        函数层说明：动态添加 SNNLayer
        ================================================================================
        通过 setattr 将 bn.SNNLayer 实例附加到当前对象，
        并在 layer_dict 中记录顺序，供逐层训练使用。
        ================================================================================
        """
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        
        # 创建 SNNLayer 实例（位于 blitnet.py）
        setattr(self, name, bn.SNNLayer(**kwargs))
        
        # 记录层顺序
        self.layer_dict[name] = self.layer_counter
        self.layer_counter += 1                            
        
    def model_logger(self):
        """
        输出模型配置日志。
        """
        model_logger(self)

    # ================================================================================
    # 函数：_anneal_learning_rate —— 公式 (3) 实现
    # ================================================================================
    # 理论来源：VPRTempo 公式 (3):
    #          η_STDP(t) = η_init_STDP * (1 - t/T)^2
    #          η_ITP(t)  = η_init_ITP  * (1 - t/T)^2
    # 物理意义：训练初期学习率较大，快速捕捉粗粒度特征；
    #          训练后期学习率衰减，精细调整权重，防止震荡。
    # 注意：代码中实际实现为 (T-mod)^2 / T^2，与 (1-t/T)^2 数学等价。
    # ================================================================================
    def _anneal_learning_rate(self, layer, mod, itp, stdp):
        """
        逐层说明：
            每 100 个时间步更新一次学习率，实现多项式退火。
            输入：
                layer : 当前正在训练的 SNNLayer
                mod   : 当前时间步计数器（从 0 开始，每步递增）
                itp   : ITP 初始学习率（退火前的基准值）
                stdp  : STDP 初始学习率（退火前的基准值）
            输出：
                更新学习率后的 layer 对象
        """
        # 每 100 个时间步执行一次退火，减少计算开销
        if np.mod(mod, 100) == 0:
            # pt = ((T - mod) / T)^2，即 (1 - mod/T)^2
            # 当 mod=0 时 pt=1（满学习率）；当 mod=T 时 pt=0（学习率归零）
            pt = pow(float(self.T - mod) / self.T, 2)
            
            # 逐元素乘法更新学习率
            layer.eta_ip = torch.mul(itp, pt)    # ITP 退火
            layer.eta_stdp = torch.mul(stdp, pt) # STDP 退火
            
        return layer

    # ================================================================================
    # 函数：train_model —— 单层的训练循环
    # ================================================================================
    # 该函数实现了 BLiTNet 的逐层训练策略：
    #   1. 固定已训练层（prev_layers），只训练当前层（layer）
    #   2. 对每个训练样本，前向传播 → clamp → calc_stdp → 退火
    #   3. 输出层通过 idx 传入 Spike Forcing 的神经元索引
    # ================================================================================
    def train_model(self, train_loader, layer, model, model_num, prev_layers=None):
        """
        ================================================================================
        函数层说明：训练网络的某一层
        ================================================================================
        输入：
            train_loader : DataLoader，每次迭代返回 (spikes, labels)
            layer        : 当前要训练的 SNNLayer（feature_layer 或 output_layer）
            model        : 当前 VPRTempoTrain 实例（用于获取 prev_layers 的对象）
            model_num    : 当前模块的序号（0, 1, 2...），用于计算 idx_scale
            prev_layers  : 已经训练好的层名称列表，前向传播时固定其参数
        核心逻辑：
            for epoch in range(self.epoch):
                for (spikes, labels) in train_loader:
                    1. 若 prev_layers 存在，前向传播通过已训练层（no_grad）
                    2. 当前层前向传播 → clamp_spikes
                    3. calc_stdp(pre_spike, spikes, spikes_noclp, layer, idx)
                    4. _anneal_learning_rate 更新学习率
        ================================================================================
        """
        # 初始化 tqdm 进度条，总步数为 T
        pbar = tqdm(total=self.T,
                    desc=f"Module {model_num+1}",
                    position=0)
        
        # ----------------------------------------
        # 逐行说明：保存初始学习率用于退火
        # ----------------------------------------
        # detach() 创建独立的副本，防止退火过程中修改原始值
        init_itp = layer.eta_ip.detach()
        init_stdp = layer.eta_stdp.detach()
        mod = 0  # 时间步计数器，每个训练样本递增 1
        
        # ----------------------------------------
        # 逐行说明：计算模块偏移量 idx_scale
        # ----------------------------------------
        # 当使用多模块时，每个模块负责不同的地点范围。
        # 例如 module 0 负责地点 0~499，module 1 负责 500~999。
        # idx_scale 用于将全局地点标签转换为模块内的局部神经元索引。
        idx_scale = (self.max_module*self.filter)*model_num

        # ----------------------------------------
        # 逐行说明：外层 epoch 循环
        # ----------------------------------------
        for _ in range(self.epoch):
            # 内层数据遍历循环
            for spikes, labels in train_loader:
                # 将数据迁移到计算设备
                spikes, labels = spikes.to(self.device), labels.to(self.device)
                
                # ----------------------------------------
                # 逐行说明：计算 Spike Forcing 的神经元索引
                # ----------------------------------------
                # idx = (labels - idx_scale) / filter，四舍五入取整
                # labels 是图像的全局索引，通过此公式映射到当前模块的输出神经元
                # 例如：全局标签=1508，module 1（idx_scale=1000），filter=8
                #       idx = (1508 - 1000) / 8 = 63.5 → round → 64
                # 这意味着该图像应该激活输出层第 64 个神经元
                idx = torch.round((labels - idx_scale) / self.filter)
                
                # ----------------------------------------
                # 逐行说明：前向传播通过已训练层（固定参数）
                # ----------------------------------------
                # 这是 BLiTNet 逐层训练的核心：先训练 feature_layer，固定它，
                # 再用 feature_layer 的输出作为输入训练 output_layer。
                # torch.no_grad() 确保不计算梯度、不更新已训练层的权重。
                if prev_layers:
                    with torch.no_grad():
                        for prev_layer_name in prev_layers:
                            prev_layer = getattr(model, prev_layer_name)
                            spikes = self.forward(spikes, prev_layer)
                            spikes = bn.clamp_spikes(spikes, prev_layer)
                else:
                    prev_layer = None
                
                # ----------------------------------------
                # 逐行说明：当前层前向传播与 STDP 计算
                # ----------------------------------------
                pre_spike = spikes.detach()  # 保存前层脉冲，供 STDP 使用
                spikes = self.forward(spikes, layer)     # 当前层前向：W * x
                spikes_noclp = spikes.detach()           # 保存未钳制的值，供 Homeostasis 使用
                spikes = bn.clamp_spikes(spikes, layer)  # 钳制到 [0, 0.9]
                
                # 调用 calc_stdp（blitnet.py），执行完整的 STDP + ITP + Homeostasis
                layer = bn.calc_stdp(pre_spike, spikes, spikes_noclp, layer, idx, prev_layer=prev_layer)
                
                # 学习率退火
                layer = self._anneal_learning_rate(layer, mod, init_itp, init_stdp)
                
                # 更新计数器和进度条
                mod += 1
                pbar.update(1)

        pbar.close()

        # ----------------------------------------
        # 逐行说明：显存清理（仅 CUDA）
        # ----------------------------------------
        if self.device == "cuda:0":
            torch.cuda.empty_cache()
            gc.collect()

    def forward(self, spikes, layer):
        """
        ================================================================================
        函数层说明：单层前向传播
        ================================================================================
        直接调用 layer.w(spikes)，即执行矩阵乘法 y = x · W^T
        对应公式 (1) 中的 Σ_i x_i (W^+_ji - W^- ji) 部分（尚未减阈值）
        ================================================================================
        """
        spikes = layer.w(spikes)
        return spikes 
    
    def save_model(self, models, model_out):    
        """
        ================================================================================
        函数层说明：保存多模块组合模型
        ================================================================================
        将多个 module 的 state_dict 组合成一个字典保存，
        键名为 model_0, model_1, ...，便于推理时加载。
        ================================================================================
        """
        state_dicts = {}
        for i, model in enumerate(models):
            state_dicts[f'model_{i}'] = model.state_dict()
        torch.save(state_dicts, model_out)
            

def check_pretrained_model(model_name):
    """
    检查预训练模型是否存在，若存在则询问是否重新训练。
    """
    if os.path.exists(os.path.join('./vprtempo/models', model_name)):
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        retrain = input(prompt).strip().lower()
        if retrain == 'y':
            return True
        elif retrain == 'n':
            print('Training new model cancelled')
            sys.exit()

# ================================================================================
# 函数：train_new_model —— 完整训练流程（多模块 + 逐层）
# ================================================================================
# 该函数是训练的总控入口，由 main.py 调用。
# 核心逻辑：
#   1. 为每个 module 划分独立的图像范围（user_input_ranges）
#   2. 逐层训练：先训练所有 module 的 feature_layer，再训练 output_layer
#   3. 训练完成后保存组合模型
# ================================================================================
def train_new_model(models, model_name):
    """
    ================================================================================
    函数层说明：训练新模型（多模块版本）
    ================================================================================
    输入：
        models     : VPRTempoTrain 实例列表，每个实例对应一个模块
        model_name : 保存模型的文件名
    核心流程：
        for layer_name in ['feature_layer', 'output_layer']:
            for i, model in enumerate(models):
                加载 model i 的数据范围 → DataLoader → train_model()
            trained_layers.append(layer_name)
    ================================================================================
    """
    # 以第一个模块为基准获取公共参数
    model = models[0]
    
    # 图像预处理流程
    image_transform = transforms.Compose([
        ProcessImage(model.dims, model.patches)
    ])
    
    # ----------------------------------------
    # 逐行说明：为每个 module 生成图像索引范围
    # ----------------------------------------
    # 例如 max_module=500, filter=8, num_modules=3：
    #   module 0: [0,       (500-1)*8] = [0, 3992]
    #   module 1: [4000,    4000+3992] = [4000, 7992]
    #   module 2: [8000,    8000+3992] = [8000, 11992]
    # 这样每个 module 处理不重叠的图像子集，实现 III-B 的模块化。
    user_input_ranges = []
    start_idx = 0
    for _ in range(models[0].num_modules):
        range_temp = [start_idx, start_idx+((models[0].max_module-1)*models[0].filter)]
        user_input_ranges.append(range_temp)
        start_idx = range_temp[1] + models[0].filter

    # ----------------------------------------
    # 逐行说明：逐层训练
    # ----------------------------------------
    # trained_layers 记录已经训练好的层名称，后续层的前向传播会固定它们
    trained_layers = [] 
    for layer_name, _ in sorted(models[0].layer_dict.items(), key=lambda item: item[1]):
        print(f"Training layer: {layer_name}")
        
        # 遍历所有模块，分别训练当前层
        for i, model in enumerate(models):
            model.train()
            model.to(torch.device(model.device))
            layer = getattr(model, layer_name)
            
            # 确定当前 module 的最大样本数
            if model.database_places < model.max_module:
                max_samples = model.database_places
            elif model.output < model.max_module:
                max_samples = model.output
            else:
                max_samples = model.max_module
            
            # 为当前 module 创建数据集（只包含其负责的图像范围）
            img_range = user_input_ranges[i]
            train_dataset = CustomImageDataset(
                annotations_file=models[0].dataset_file, 
                base_dir=models[0].data_dir,
                img_dirs=models[0].database_dirs,
                transform=image_transform,
                filter=models[0].filter,
                skip=models[0].skip,
                test=False,          # test=False 表示训练模式（会合并多个 CSV）
                img_range=img_range, # 只加载当前模块的图像范围
                max_samples=max_samples
            )
            
            # DataLoader 配置
            if model.device == "mps":
                num_workers = 0
                persistent_workers = False
            else:
                num_workers = 4
                persistent_workers = False
            train_loader = DataLoader(
                train_dataset, 
                batch_size=1,       # SNN 时序编码要求 batch_size=1
                shuffle=True,       # 打乱顺序，防止网络记住时序
                num_workers=num_workers,
                persistent_workers=persistent_workers
            )
            
            # 训练当前层
            model.train_model(train_loader, layer, model, i, prev_layers=trained_layers)
            model.to(torch.device("cpu"))  # 训练完移回 CPU，节省显存
        
        # 当前层训练完成后，加入已训练层列表
        trained_layers.append(layer_name)
    
    # 所有层训练完成后，切换到评估模式
    for model in models:
        model.eval()
    
    # 保存模型
    model.save_model(models, os.path.join('./vprtempo/models', model_name))
