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
【模块级注释 —— 文件整体说明】
================================================================================
文件名称：VPRTempo.py
所属项目：VPRTempo —— 视觉场景识别（Visual Place Recognition, VPR）脉冲神经网络
核心功能：fp32 推理模型（Inference Model）

理论对应关系（与论文中的章节/公式对应）：
  - 公式 (9) 匹配规则        → evaluate() 中 argmax 取最高输出脉冲
  - III-C 并行张量           → forward() 中 torch.cat(outputs, dim=1)
  - IV-C 评估指标            → evaluate() 中 recallAtK / createPR
  - III-B 模块化推理         → evaluate() 中为每个 model 构建 nn.Sequential

与训练模型（VPRTempoTrain.py）的核心区别：
  1. 本文件仅用于推理（inference），不包含任何学习机制（STDP、ITP、Homeostasis）。
  2. 通过 nn.Sequential 直接做权重矩阵连乘，实现高速前向传播。
  3. 支持多模块（multi-module）拼接：当数据库图像数 > max_module 时，自动拆分为
     多个子模型分别推理，最后拼接输出。
================================================================================
"""

# ================================================================================
# 【行级注释 —— 导入部分】
# ================================================================================
# os      : 操作系统接口，用于路径拼接、文件存在性检查
# json    : 用于保存 PR（Precision-Recall）曲线数据为 JSON 格式
# torch   : PyTorch 核心库，提供张量运算与神经网络基础
# numpy   : 科学计算库，主要用于构建 Ground Truth 矩阵（GT）
# nn      : PyTorch 神经网络模块，本文件使用 nn.Sequential 构建推理链
# plt     : Matplotlib 绘图库，用于绘制 PR 曲线和相似度矩阵
# bn      : 自定义脉冲神经网络层（blitnet.py），提供 SNNLayer
# tqdm    : 进度条库，用于显示推理进度
# demo    : 自定义动画演示模块（demo.py），用于 run_demo 模式
# PrettyTable : 格式化表格输出，用于打印 Recall@N 结果
# DataLoader  : PyTorch 数据加载器，用于批量读取查询图像
# get_data_model : 自定义下载器，自动从 Dropbox 下载预训练模型和示例数据
# recallAtK, createPR : VPR 评估指标（metrics.py）
# CustomImageDataset, ProcessImage : 自定义数据集和图像预处理（dataset.py）
# ================================================================================
import os
import json
import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import vprtempo.src.blitnet as bn

from tqdm import tqdm
from vprtempo.src.demo import demo
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from vprtempo.src.download import get_data_model
from vprtempo.src.metrics import recallAtK, createPR
from vprtempo.src.gt import build_ground_truth
from vprtempo.src.dataset import CustomImageDataset, ProcessImage
from vprtempo.src import conv_frontend as cf  # IDEA1 S2.5：卷积前端桥接（frontend='none' 时零开销）


# ================================================================================
# 【模块级注释 —— Part 1: VPRTempo 类】
# ================================================================================
# VPRTempo 是 fp32 推理模型的核心类，继承自 torch.nn.Module。
# 
# 核心设计思想：
#   训练完成后，权重矩阵已经收敛，不再需要逐时间步模拟脉冲发放和 STDP 更新。
#   因此推理时可以直接将两层权重矩阵（feature_layer.w 和 output_layer.w）
#   串接成 nn.Sequential，做一次性矩阵乘法，极大提升推理速度。
#
# 网络结构（固定两层）：
#   输入层（input）   → 特征层（feature_layer） → 输出层（output_layer）
#   维度: input_dim  →  feature_dim = input_dim * 2  →  output_dim = 模块内场景数
# ================================================================================
class VPRTempo(nn.Module):
    def __init__(self, 
                 args, 
                 dims, 
                 logger, 
                 num_modules, 
                 output_folder, 
                 out_dim, 
                 out_dim_remainder=None
                 ):
        """
        ================================================================================
        【函数级注释 —— VPRTempo 构造函数】
        ================================================================================
        功能：初始化 VPRTempo 推理模型实例
        
        参数说明：
          args                : argparse.Namespace，包含所有命令行参数（如 dataset, 
                                query_dir, database_dirs, filter, skip 等）
          dims                : tuple/list，图像缩放后的尺寸，例如 (56, 56)
          logger              : logging.Logger 实例，用于输出日志信息
          num_modules         : int，当前模型在所有模块中的索引（从 0 开始）
          output_folder       : str，输出文件夹路径，用于保存 PR 曲线等结果
          out_dim             : int，输出层神经元数量（即每个模块对应的场景数）
          out_dim_remainder   : int/None，最后一个模块的场景数（当不能整除时使用）
        
        与训练模型 VPRTempoTrain.__init__ 的区别：
          1. inference=True：SNNLayer 只创建权重 w 和阈值 thr，不创建学习相关参数
             （如 eta_stdp、eta_ip、mu 等），节省显存/内存。
          2. 不需要传入 dataset_file（训练时才需要读取数据库 CSV），但需要 query_dir
             用于推理时加载查询图像。
          3. 需要 output_folder 以保存评估过程中生成的 PR 曲线等文件。
        ================================================================================
        """
        # ----------------------------------------
        # 【行级注释】调用父类 nn.Module 的构造函数，完成 PyTorch 模块初始化
        # ----------------------------------------
        super(VPRTempo, self).__init__()

        # ----------------------------------------
        # 【行级注释 —— 解析参数】
        # ----------------------------------------
        # 若传入的 args 不为 None，将所有命令行参数动态设置为类的属性。
        # 例如：args.dataset = 'nordland' → self.dataset = 'nordland'
        # 这样做的好处是后续代码中可以直接用 self.xxx 访问参数，而不必写 self.args.xxx。
        # ----------------------------------------
        if args is not None:
            self.args = args
            for arg in vars(args):
                setattr(self, arg, getattr(args, arg))
        # 将图像尺寸 dims 也设为类属性，供后续使用
        setattr(self, 'dims', dims)

        # ----------------------------------------
        # 【行级注释 —— 设备选择】
        # ----------------------------------------
        # 按照优先级依次检测可用的计算设备：
        #   1. CUDA（NVIDIA GPU）：速度最快，优先使用
        #   2. MPS（Apple Silicon GPU）：macOS 上的 Metal Performance Shaders
        #   3. CPU：兜底选项，所有平台都支持
        # ----------------------------------------
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # ----------------------------------------
        # 【行级注释 —— 输入参数存储】
        # ----------------------------------------
        # logger        : 日志记录器，用于在控制台/文件输出运行信息
        # num_modules   : 当前模块编号（当数据库较大时，会拆分为多个模块）
        # output_folder : 评估结果保存路径（如 PR 曲线图、JSON 数据）
        # ----------------------------------------
        self.logger = logger
        self.num_modules = num_modules
        self.output_folder = output_folder

        # ----------------------------------------
        # 【行级注释 —— 查询数据集文件路径】
        # ----------------------------------------
        # 推理阶段需要加载查询图像（query images），这些图像的列表存储在 CSV 文件中。
        # 路径格式：./vprtempo/dataset/{dataset}-{query_dir}.csv
        # 例如：./vprtempo/dataset/nordland-fall.csv
        # query_dir 可能是逗号分隔的多个文件夹，因此需要 split 并去除空格。
        # ----------------------------------------
        self.dataset_file = os.path.join('./vprtempo/dataset', f'{self.dataset}-{self.query_dir}' + '.csv')  
        self.query_dir = [dir.strip() for dir in self.query_dir.split(',')]

        # ----------------------------------------
        # 【行级注释 —— 层管理字典】
        # ----------------------------------------
        # layer_dict     : 记录每层名称到索引的映射，例如 {'feature_layer': 0, 'output_layer': 1}
        # layer_counter  : 层计数器，每次 add_layer 后自增
        # database_dirs  : 数据库图像文件夹列表，同样支持逗号分隔多个文件夹
        # ----------------------------------------
        self.layer_dict = {} # 用于存储层名称到索引的映射，方便后续访问
        self.layer_counter = 0
        self.database_dirs = [dir.strip() for dir in self.database_dirs.split(',')]

        # ----------------------------------------
        # 【行级注释 —— 网络架构维度】
        # ----------------------------------------
        # 网络维度必须与训练时完全一致，否则加载的权重矩阵形状不匹配。
        # input   : 输入神经元数 = 图像宽 × 高（例如 56×56 = 3136）
        # feature : 特征层神经元数 = input × 2（固定扩展一倍，例如 6272）
        # output  : 输出层神经元数 = 当前模块负责的场景数
        #           若 out_dim_remainder 不为 None（最后一个模块且不能整除），
        #           则使用 remainder，否则使用 out_dim。
        # ----------------------------------------
        # -----------------------------------------------------------------------------
        # 【IDEA1 S2.5】conv 前端注册（推理侧，inference=True，与训练侧维度严格一致）
        # frontend='none'（B0 默认路径）时此分支完全不触发。
        # -----------------------------------------------------------------------------
        if getattr(self, 'frontend', 'none') != 'none':
            conv_layer = cf.build_conv_layer(self, list(self.dims), self.device, inference=True)
            setattr(self, 'conv_layer', conv_layer)
            self.layer_dict['conv_layer'] = self.layer_counter
            self.layer_counter += 1
            self.input = conv_layer.flat_dim      # 维度重算（ADR-1，与训练侧一致）
        else:
            self.input = int(self.dims[0]*self.dims[1])
        self.feature = int(self.input * 2)
        if not out_dim_remainder is None:
            self.output = out_dim_remainder
        else:
            self.output = out_dim

        # ----------------------------------------
        # 【行级注释 —— 默认 demo 模型路径】
        # ----------------------------------------
        # 当用户运行演示模式（--run_demo）且未指定模型时，默认加载此路径的模型。
        # 文件名格式：{database_dirs}_VPRTempo_IN{input}_FN{feature}_DB{places}.pth
        # 例如：springfall_VPRTempo_IN3136_FN6272_DB500.pth
        # 若文件不存在，会自动调用 get_data_model() 从 Dropbox 下载。
        # ----------------------------------------
        self.demo = './vprtempo/models/springfall_VPRTempo_IN3136_FN6272_DB500.pth'

        # ================================================================================
        # 【行级注释 —— 定义推理层（inference=True）】
        # ================================================================================
        # 调用 self.add_layer() 动态添加两层：
        #   1. feature_layer : 输入 → 特征（权重形状 [input, feature]）
        #   2. output_layer  : 特征 → 输出（权重形状 [feature, output]）
        # 
        # 关键参数 inference=True：
        #   此时 SNNLayer 内部只创建 self.w（权重）和 self.thr（阈值），
        #   不创建训练所需的 eta_stdp（STDP 学习率）、eta_ip（内在可塑性学习率）、
        #   mu（目标发放率）等参数。这大幅减少了推理时的内存占用。
        # ================================================================================
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature],
            device=self.device,
            inference=True
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            device=self.device,
            inference=True
        )
        
    def add_layer(self, name, **kwargs):
        """
        ================================================================================
        【函数级注释 —— 动态添加推理层】
        ================================================================================
        功能：根据名称和参数动态添加一个 SNNLayer 到当前模型
        
        参数说明：
          name    : str，层的名称（如 'feature_layer', 'output_layer'）
          **kwargs: 可变关键字参数，传递给 bn.SNNLayer 的构造函数
        
        内部机制：
          1. 检查名称是否已存在（避免重复添加）。
          2. 使用 setattr 将层实例设置为类的属性（如 self.feature_layer）。
          3. 在 layer_dict 中记录名称到索引的映射。
          4. layer_counter 自增，为下一层预留索引。
        
        与 VPRTempoTrain.add_layer 的区别：
          训练模型可能传入额外的学习参数，而推理模型传入 inference=True。
        ================================================================================
        """
        # 若层名称已存在，抛出异常防止意外覆盖
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        # 动态创建属性：例如 self.feature_layer = bn.SNNLayer(...)
        setattr(self, name, bn.SNNLayer(**kwargs))
        # 记录该层在 layer_dict 中的索引位置
        self.layer_dict[name] = self.layer_counter
        # 层计数器加 1，供下一层使用
        self.layer_counter += 1                           

    # ================================================================================
    # 【模块级注释 —— evaluate 函数】
    # ================================================================================
    # 对应论文：公式 (9) + IV-C 评估指标
    # 
    # 核心流程（5 个阶段）：
    #   阶段 1：为每个 module 构建 nn.Sequential（权重连乘，跳过 STDP）
    #   阶段 2：遍历 test_loader，收集所有查询图像经网络前向传播后的输出脉冲
    #   阶段 3：拼接多模块输出，构建相似度矩阵 S ∈ R^(database_places × query_places)
    #   阶段 4：根据 skip、filter、GT_tolerance 构建 Ground Truth 矩阵 GT
    #   阶段 5：基于 S 和 GT 计算 Recall@N（N=1,5,10,15,20,25），可选 PR 曲线
    # ================================================================================
    def evaluate(self, models, test_loader):
        """
        ================================================================================
        【函数级注释 —— 运行推理模型并计算精度】
        ================================================================================
        功能：执行完整推理流程，计算并输出 VPR 评估指标
        
        参数说明：
          models      : list[VPRTempo]，VPRTempo 实例列表，每个元素对应一个模块。
                        例如数据库有 1500 张图、max_module=500，则 models 长度为 3。
          test_loader : DataLoader，查询图像的数据加载器，每次迭代返回
                        (spikes, label)，其中 spikes 是脉冲编码后的图像张量。
        
        输出：
          - 控制台打印 PrettyTable 表格，展示 Recall@1,5,10,15,20,25
          - 若启用 --PR_curve，保存 PR 曲线数据到 JSON 并绘制图像
          - 若启用 --sim_mat，绘制相似度矩阵和 Ground Truth 矩阵
          - 若启用 --run_demo，调用 demo() 运行动画演示
        ================================================================================
        """
        # ----------------------------------------
        # 【行级注释 —— 初始化 tqdm 进度条】
        # ----------------------------------------
        # total     : 进度条总长度 = 查询图像总数（query_places）
        # desc      : 进度条前方显示的描述文字
        # position  : 进度条在终端中的行位置（0 表示第一行）
        # ----------------------------------------
        pbar = tqdm(total=self.query_places,
                    desc="Running the test network",
                    position=0)
        
        # ================================================================================
        # 【行级注释 —— 阶段 1：为每个 module 构建 nn.Sequential 推理链】
        # ================================================================================
        # 原理：训练完成后，脉冲神经网络的动态行为可以等效为静态权重矩阵的连乘。
        #       对于两层网络：output = input × W_feature × W_output
        #       因此可以用 nn.Sequential 将两层权重串接，PyTorch 会自动连续执行矩阵乘法。
        # 
        # 为什么比训练快？
        #   训练时：每个时间步都要逐层计算脉冲发放、STDP 更新、ITP 更新，无法合并。
        #   推理时：直接矩阵乘法，一次性得到结果，充分利用 GPU 并行计算。
        # 
        # 注意：这里取的是 model.feature_layer.w 和 model.output_layer.w，
        #       它们是 nn.Parameter 或 nn.Linear 对象，可以作为 nn.Sequential 的层。
        # ================================================================================
        self.inferences = [] # 存储每个模块的 nn.Sequential 推理链，每个元素对应一个模块
        for model in models:
            # 【IDEA1 S2.5】conv 前端：推理链前加 ConvFrontendModule（平向量→conv→pooled_flat）
            if hasattr(model, 'conv_layer'):
                self.inferences.append(nn.Sequential(
                    cf.ConvFrontendModule(model.conv_layer),
                    model.feature_layer.w,
                    model.output_layer.w,
                ))
            else:
                self.inferences.append(nn.Sequential(
                    model.feature_layer.w,
                    model.output_layer.w,
                ))
            # 将构建好的 Sequential 移动到计算设备（GPU/CPU）上
            self.inferences[-1].to(torch.device(self.device))
        
        # ----------------------------------------
        # 【行级注释 —— 阶段 2：收集所有查询图像的输出脉冲】
        # ----------------------------------------
        # out     : list，存储每帧查询图像经过所有模块后的输出脉冲向量
        #           每个元素形状为 [1, total_output_neurons]
        # labels  : list，存储对应的图像标签（用于调试或验证，实际评估中主要靠 GT）
        # ----------------------------------------
        out = []
        labels = []
        for spikes, label in test_loader:
            # 将脉冲数据移动到当前计算设备
            spikes = spikes.to(self.device)
            # 将标签从张量转为 Python 标量，移回 CPU（避免占用 GPU 内存）
            labels.append(label.detach().cpu().item())
            
            # 前向传播：spikes 经过 forward()，在所有模块中并行/串行计算，输出拼接
            spikes = self.forward(spikes)
            # 将结果从 GPU 移回 CPU，并 detach（断开计算图，节省内存）
            out.append(spikes.detach().cpu())
            # 更新进度条（每处理 1 张查询图像，进度 +1）
            pbar.update(1)
        # 关闭进度条
        pbar.close()

        # ================================================================================
        # 【行级注释 —— 阶段 3：构建相似度矩阵 S】
        # ================================================================================
        # 此时 out 是一个长度为 query_places 的列表。
        # 每个元素形状：[1, total_output_neurons]
        # 
        # torch.stack(out, dim=2) 的操作：
        #   将列表中的张量沿新维度 dim=2 堆叠，结果形状：[1, total_output_neurons, query_places]
        # squeeze(0) 去掉 batch 维度（因为 batch_size=1），得到：
        #   [total_output_neurons, query_places]
        # 
        # 物理意义：
        #   S[i, j] 表示第 j 个查询图像与第 i 个数据库地点的相似度（脉冲幅度）。
        #   对应论文公式 (9) 中的 x_i，用于后续 argmax 或 top-K 匹配。
        # ================================================================================
        out = torch.stack(out, dim=2) #在dim=1(神经元数)上添加一个新维度——查询图像维度
        out = out.squeeze(0).numpy()  #去掉 batch 维度，转换为 NumPy 数组，形状 [total_output_neurons, query_places]
       
        # ================================================================================
        # 【行级注释 —— 阶段 4：构建 Ground Truth 矩阵 GT】
        # ================================================================================
        # IDEA1 S1.4：GT 构造已抽取为共用函数 vprtempo/src/gt.py::build_ground_truth，
        # 轨 A（本函数）与轨 B（eval_retrieval.py）共用同一实现，保证逐比特一致。
        # 逻辑不变：skip!=0 → GT[skip//filter + j, j] = 1（skip 先整除 filter 换算成
        # 降采样后偏移）；skip==0 → 单位矩阵；GT_tolerance>0 → 每列正例行 ±tolerance 膨胀。
        # ================================================================================
        GT = build_ground_truth(model.database_places, model.query_places,
                                skip=model.skip, filter=model.filter,
                                tolerance=self.GT_tolerance)
        
        # ================================================================================
        # 【行级注释 —— 阶段 5a：生成 Precision-Recall 曲线（可选）】
        # ================================================================================
        # 若命令行参数启用了 --PR_curve：
        #   1. 调用 metrics.py 中的 createPR()，对相似度矩阵 S 设定不同阈值，
        #      计算对应的 Precision（精确率）和 Recall（召回率）。
        #   2. 将 P、R 数据保存为 JSON 文件，方便后续分析或绘图。
        #   3. 若非 demo 模式，直接调用 Matplotlib 绘制 PR 曲线图。
        # 
        # createPR 参数说明：
        #   out       : 相似度矩阵 [database_places, query_places]
        #   GT        : Ground Truth 矩阵 [database_places, query_places]
        #   matching  : 'single' 表示每个查询只匹配一个最近邻
        #   n_thresh  : 阈值数量（默认 100），越多曲线越平滑
        # ================================================================================
        if model.PR_curve:
            # createPR 返回两个列表 P 和 R，分别对应不同阈值下的 Precision 和 Recall 值
            P, R = createPR(out, GT, matching='single', n_thresh=100)
            # 将 Precision 和 Recall 数据保存为 JSON 格式，文件名为 PR_curve_data.json
            PR_data = {"Precision": P, "Recall": R}
            full_path = f"{model.output_folder}/PR_curve_data.json"
            # 确保输出文件夹存在，如果不存在则创建
            with open(full_path, 'w') as file:
                json.dump(PR_data, file) 
            # 仅在非 demo 模式下显示图表（demo 模式有自己的动画展示）
            if not model.run_demo:
                plt.plot(R, P)    
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.show()
                plt.close()

        # ================================================================================
        # 【行级注释 —— 阶段 5b：绘制相似度矩阵和 GT（可选）】
        # ================================================================================
        # 若命令行参数启用了 --sim_mat：
        #   并排绘制两个热力图：
        #     左图：相似度矩阵（Similarity matrix）
        #            颜色越亮表示查询与数据库地点越相似。
        #     右图：Ground Truth 矩阵
        #            白色对角线（或带状区域）表示真实匹配。
        #   通过对比两图，可以直观判断模型的匹配效果。
        # ================================================================================
        if model.sim_mat and not model.run_demo:
            # 创建 1 行 2 列的子图，figsize 单位是英寸（宽 15，高 5）
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            # 绘制相似度矩阵，使用 viridis 颜色映射（蓝→黄）
            cax1 = axs[0].matshow(out, cmap='viridis')
            fig.colorbar(cax1, ax=axs[0], shrink=0.8)
            axs[0].set_title('Similarity matrix')
            # 绘制 GT 矩阵，使用 plasma 颜色映射（紫→黄）
            cax2 = axs[1].matshow(GT, cmap='plasma')
            fig.colorbar(cax2, ax=axs[1], shrink=0.8)
            axs[1].set_title('GT')
            plt.tight_layout()
            plt.show()
            plt.close()
        
        # ================================================================================
        # 【行级注释 —— 阶段 5c：计算 Recall@N（N = 1, 5, 10, 15, 20, 25）】
        # ================================================================================
        # 对应论文 IV-C "Evaluation Metrics"：
        #   Recall@1 : 强制每个查询只匹配一个最近邻，计算正确率。
        #              即"第 1 个候选就是正确答案"的比例。
        #   Recall@N : 真实匹配是否在前 N 个最相似候选中。
        #              即"前 N 个候选中包含正确答案"的比例。
        # 
        # 评估方式：
        #   对每列（每个查询），按相似度从高到低排序，看前 N 个是否包含 GT 为 1 的行。
        #   统计所有查询的正确率，取平均得到 Recall@N。
        # ================================================================================
        N = [1, 5, 10, 15, 20, 25]
        RN = []
        for n in N:
            # recallAtK 返回 0~1 之间的小数，round(..., 2) 保留两位小数
            RN.append(round(recallAtK(out, GT, K=n), 2))

        # ----------------------------------------
        # 【行级注释 —— demo 模式处理】
        # ----------------------------------------
        # 若启用 --run_demo，不打印表格，而是调用 demo() 运行动画演示。
        # demo() 会展示查询图像与匹配结果的动态对比。
        # ----------------------------------------
        if model.run_demo:
            demo(model.data_dir, model.query_dir[0], model.database_dirs[0], out, GT, N, RN, R, P)
            return

        # ----------------------------------------
        # 【行级注释 —— 打印结果表格】
        # ----------------------------------------
        # 使用 PrettyTable 构建格式化的 ASCII 表格：
        #   第一行表头：N 和对应的 K 值（1, 5, 10, 15, 20, 25）
        #   第二行数据：Recall 和对应的 Recall@K 值
        # 然后通过 logger 输出到控制台/日志文件。
        # ----------------------------------------
        table = PrettyTable()
        table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        table.add_row(["Recall", RN[0], RN[1], RN[2], RN[3], RN[4], RN[5]])
        self.logger.info(table)

    # ================================================================================
    # 【模块级注释 —— forward 函数】
    # ================================================================================
    # 理论来源：VPRTempo 论文 III-C "Efficient implementation"
    # 
    # 核心操作：torch.cat(outputs, dim=1)
    #   每个 module 独立计算输出向量（形状 [1, module_output_dim]），
    #   最终在维度 1（神经元维度）上拼接，形成完整数据库的相似度向量。
    # 
    # 时间复杂度优势：
    #   训练时：O(T × N) 其中 T 是时间步数，N 是神经元数。
    #   推理时：O(log n) 查询缩放（论文 Fig. 2C），因为矩阵乘法可并行。
    # ================================================================================
    def forward(self, spikes):
        """
        ================================================================================
        【函数级注释 —— 推理前向传播（多模块并行）】
        ================================================================================
        功能：将单张查询图像的脉冲编码输入网络，经过所有模块计算后拼接输出
        
        参数说明：
          spikes : Tensor，形状 [1, input_dim]
                   单张查询图像经 ProcessImage 预处理并编码为脉冲后的张量。
        
        返回：
          concatenated_output : Tensor，形状 [1, total_output_neurons]
                   所有模块输出拼接后的完整相似度向量。
                   例如：3 个模块各输出 500 维，拼接后为 [1, 1500]。
        
        执行流程：
          1. 复制输入脉冲（避免修改原始张量）。
          2. 遍历 self.inferences（每个模块的 nn.Sequential）。
          3. 对每个模块，输入相同 spikes，得到输出 out_spikes。
          4. 收集所有输出到列表 outputs。
          5. torch.cat(outputs, dim=1) 沿神经元维度拼接。
          6. 返回拼接结果。
        ================================================================================
        """
        # 使用 detach().clone() 创建输入脉冲的独立副本，避免后续计算影响原始数据
        in_spikes = spikes.detach().clone()
        # 初始化空列表，用于存储每个模块的输出
        outputs = []

        # 遍历所有 module 预先构建好的 nn.Sequential
        for inference in self.inferences:
            # 输入脉冲经过当前模块的两层权重连乘，得到输出脉冲
            out_spikes = inference(in_spikes)
            # 将当前模块的输出加入列表
            outputs.append(out_spikes)

        # ================================================================================
        # 【行级注释 —— 多模块输出拼接】
        # ================================================================================
        # torch.cat(outputs, dim=1) 的作用：
        #   dim=1 表示在第 1 个维度（列/特征维度）上拼接。
        #   假设有 3 个模块，每个输出形状为 [1, 500]：
        #     outputs = [ [1, 500], [1, 500], [1, 500] ]
        #   拼接后：concatenated_output 形状为 [1, 1500]
        # 
        # 物理意义：
        #   拼接后的 1500 个数值分别代表该查询图像与数据库中 1500 个地点的相似度。
        #   数值越高，表示匹配程度越高（脉冲发放越强）。
        # ================================================================================
        concatenated_output = torch.cat(outputs, dim=1)
    
        
        return concatenated_output
        
    def load_model(self, models, model_path):
        """
        ================================================================================
        【函数级注释 —— 加载预训练的多模块组合模型】
        ================================================================================
        功能：从磁盘加载训练好的模型权重，并分配到各个模块实例中
        
        参数说明：
          models    : list[VPRTempo]，待加载权重的模型实例列表
          model_path: str，权重文件路径（.pth 格式）
        
        加载方式：
          训练时保存的是"组合状态字典"（combined state dict），键名为 model_0, model_1, ...
          例如：
            combined_state_dict['model_0'] → models[0].load_state_dict()
            combined_state_dict['model_1'] → models[1].load_state_dict()
            ...
        
        特殊处理：
          若 model_path 不存在且恰好是 demo 默认路径，自动调用 get_data_model()
          从 Dropbox 下载预训练模型和示例数据（约 600 MB）。
        
        安全：
          torch.load 使用 weights_only=True，防止恶意 pickle 代码执行。
        ================================================================================
        """
        # 若模型文件不存在
        if not os.path.exists(model_path):
            # 若请求的是默认 demo 模型，自动下载
            if model_path == self.demo:
                get_data_model()  # 自动下载 demo 数据和模型
            else:
                # 其他路径不存在则报错
                raise ValueError(f"Model path {model_path} does not exist.")
            
        # 加载组合状态字典，map_location 确保在 CPU/GPU 间兼容加载
        # weights_only=True 是安全选项，只加载张量和参数，不执行任意代码
        combined_state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        # 遍历每个模块实例，加载对应的状态字典
        for i, model in enumerate(models):
            model.load_state_dict(combined_state_dict[f'model_{i}'])
            # 设置为评估模式，关闭 Dropout/BatchNorm 等训练专用层（虽然 VPRTempo 没有这些）
            model.eval()


# ================================================================================
# 【模块级注释 —— Part 2: run_inference 函数】
# ================================================================================
# 本函数是推理流程的总控入口，由 main.py 调用。
# 
# 职责：
#   1. 构建图像预处理流程（ProcessImage）。
#   2. 构建查询数据集（CustomImageDataset）和 DataLoader。
#   3. 调用 load_model() 加载预训练权重。
#   4. 在 torch.no_grad() 上下文中调用 evaluate()，关闭梯度计算以节省显存。
#
# 注意：
#   推理时不涉及数据库图像的加载（权重已包含训练好的记忆），
#   只需要加载查询图像（query images）即可。
# ================================================================================
def run_inference(models, model_name):
    """
    ================================================================================
    【函数级注释 —— 运行推理流程】
    ================================================================================
    功能：作为推理总控入口，完成数据加载 → 模型加载 → 评估的完整流程
    
    参数说明：
      models     : list[VPRTempo]，已初始化的 VPRTempo 实例列表（每个对应一个模块）
      model_name : str，预训练模型文件名（保存在 ./vprtempo/models/ 目录下）
    
    流程详解：
      1. 以 models[0] 为参数基准（所有模块的参数如 dims、device 等一致）。
      2. 创建图像预处理实例 image_transform = ProcessImage(dims, patches)。
      3. 创建查询数据集 test_dataset = CustomImageDataset(...)。
      4. 创建 DataLoader（batch_size=1，因为推理逐帧进行）。
      5. 加载模型权重：model.load_model(models, model_path)。
      6. 在 torch.no_grad() 环境下执行 model.evaluate()。
    ================================================================================
    """
    # 以第一个模块为参数基准（所有模块的 dims、device、filter 等参数相同）
    model = models[0]
    
    # 创建图像预处理流水线：
    #   ProcessImage 内部执行：RGB→灰度→Gamma校正→缩放→块归一化→脉冲编码
    image_transform = ProcessImage(model.dims, model.patches, patch_norm=getattr(model, 'patch_norm', 'on') == 'on')
    
    # 构建查询数据集：
    #   annotations_file : 查询图像的 CSV 标注文件路径
    #   base_dir         : 数据集根目录（如 ./vprtempo/dataset/）
    #   img_dirs         : 查询图像文件夹列表
    #   transform        : 图像预处理函数
    #   max_samples      : 最多加载的查询图像数（由 query_places 参数指定）
    #   filter           : 降采样步长（每 filter 帧取 1 帧）
    #   skip             : 开头跳过的帧数
    test_dataset = CustomImageDataset(
        annotations_file=model.dataset_file,  # CSV 文件路径，例如 ./vprtempo/dataset/nordland-fall.csv
        base_dir=model.data_dir, # 数据集根目录，例如 ./vprtempo/dataset/
        img_dirs=model.query_dir, # 查询图像文件夹列表，例如 ['nordland/fall/query/']
        transform=image_transform, # 图像预处理函数实例
        max_samples=model.query_places, # 最多加载的查询图像数，例如 750
        filter=model.filter, # 降采样步长，例如 8（每 8 帧取 1 帧）
        skip=model.skip
    )

    # ================================================================================
    # 【行级注释 —— DataLoader 配置】
    # ================================================================================
    # batch_size = 1 ：
    #   推理时通常逐帧处理，因为每张查询图像需要独立计算与所有数据库地点的相似度。
    # 
    # num_workers 配置：
    #   默认使用 4 个子进程加速数据加载（CPU 预读取图像）。
    #   但若设备为 mps（Apple Silicon），必须设为 0，因为 MPS 后端不支持多进程 DataLoader。
    # 
    # persistent_workers = False ：
    #   每次 epoch 结束后关闭 worker 进程。推理只有一个 epoch，无需保持。
    # ================================================================================
    if model.device == "mps": 
        num_workers = 0
        persistent_workers = False 
    else:
        num_workers = 4
        persistent_workers = False
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )

    # 加载模型权重，路径：./vprtempo/models/{model_name}
    model.load_model(models, os.path.join('./vprtempo/models', model_name))

    # ================================================================================
    # 【行级注释 —— 推理（关闭梯度计算）】
    # ================================================================================
    # torch.no_grad() 上下文管理器：
    #   在此环境下，PyTorch 不会记录任何计算操作到计算图中，
    #   因此不会分配用于反向传播的中间变量内存，显著降低显存占用。
    #   这对于纯推理任务（不需要梯度）是必须的优化。
    # ================================================================================
    with torch.no_grad():  # 关闭梯度计算，节省显存
        model.evaluate(models, test_loader)
