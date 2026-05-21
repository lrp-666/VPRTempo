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
VPRTempo.py —— fp32 推理模型
================================================================================
理论对应：
  - 公式 (9) 匹配规则        → evaluate() 中 argmax 取最高输出脉冲
  - III-C 并行张量           → forward() 中 torch.cat(outputs, dim=1)
  - IV-C 评估指标            → evaluate() 中 recallAtK / createPR
  - III-B 模块化推理         → evaluate() 中为每个 model 构建 nn.Sequential
================================================================================
"""

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
from vprtempo.src.dataset import CustomImageDataset, ProcessImage


# ================================================================================
# Part 1: VPRTempo 类 —— fp32 推理模型定义
# ================================================================================
# 推理模型剥离了所有学习机制（STDP/ITP/Homeostasis），
# 仅保留权重矩阵和阈值，通过 nn.Sequential 高效前向传播。
# ================================================================================
class VPRTempo(nn.Module):
    def __init__(self, args, dims, logger, num_modules, output_folder, out_dim, out_dim_remainder=None):
        """
        ================================================================================
        函数层说明：VPRTempo 推理模型构造函数
        ================================================================================
        与 VPRTempoTrain 的区别：
          1. inference=True：SNNLayer 只创建权重和阈值，不创建学习相关参数
          2. 不需要 dataset_file（训练时才需要），但需要 query_dir 用于推理
          3. 需要 output_folder 保存 PR 曲线等评估结果
        ================================================================================
        """
        super(VPRTempo, self).__init__()

        # ----------------------------------------
        # 逐行说明：解析参数
        # ----------------------------------------
        if args is not None:
            self.args = args
            for arg in vars(args):
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

        # ----------------------------------------
        # 逐行说明：输入参数存储
        # ----------------------------------------
        self.logger = logger
        self.num_modules = num_modules
        self.output_folder = output_folder

        # ----------------------------------------
        # 逐行说明：查询数据集文件路径
        # ----------------------------------------
        # 推理时需要加载 query_dir 的 CSV 文件
        self.dataset_file = os.path.join('./vprtempo/dataset', f'{self.dataset}-{self.query_dir}' + '.csv')  
        self.query_dir = [dir.strip() for dir in self.query_dir.split(',')]

        # ----------------------------------------
        # 逐行说明：层管理字典
        # ----------------------------------------
        self.layer_dict = {}
        self.layer_counter = 0
        self.database_dirs = [dir.strip() for dir in self.database_dirs.split(',')]

        # ----------------------------------------
        # 逐行说明：网络架构维度（与训练时完全一致）
        # ----------------------------------------
        self.input = int(self.dims[0]*self.dims[1])
        self.feature = int(self.input * 2)
        if not out_dim_remainder is None:
            self.output = out_dim_remainder
        else:
            self.output = out_dim

        # ----------------------------------------
        # 逐行说明：默认 demo 模型路径
        # ----------------------------------------
        self.demo = './vprtempo/models/springfall_VPRTempo_IN3136_FN6272_DB500.pth'

        # ================================================================================
        # 逐行说明：定义推理层（inference=True，无学习参数）
        # ================================================================================
        # 推理时 SNNLayer 只保留 self.w 和 self.thr，不创建 eta_ip/eta_stdp 等
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
        函数层说明：动态添加推理层
        ================================================================================
        与 VPRTempoTrain 相同，但传入 inference=True。
        ================================================================================
        """
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        setattr(self, name, bn.SNNLayer(**kwargs))
        self.layer_dict[name] = self.layer_counter
        self.layer_counter += 1                           

    # ================================================================================
    # 函数：evaluate —— 推理与评估（对应公式 9 + IV-C 评估指标）
    # ================================================================================
    # 核心流程：
    #   1. 为每个 module 构建 nn.Sequential（权重连乘，跳过 STDP）
    #   2. 遍历 test_loader，收集所有查询图像的输出脉冲
    #   3. 拼接多模块输出，构建相似度矩阵 S ∈ R^(database_places × query_places)
    #   4. 根据 S 和 GT 计算 Recall@N、PR 曲线
    # ================================================================================
    def evaluate(self, models, test_loader):
        """
        ================================================================================
        函数层说明：运行推理模型并计算精度
        ================================================================================
        输入：
            models     : VPRTempo 实例列表（每个对应一个模块）
            test_loader: 查询图像的 DataLoader
        输出：
            控制台打印 Recall@1,5,10,15,20,25 表格
            可选：PR 曲线、相似度矩阵图
        ================================================================================
        """
        # 初始化 tqdm 进度条
        pbar = tqdm(total=self.query_places,
                    desc="Running the test network",
                    position=0)
        
        # ================================================================================
        # 逐行说明：为每个 module 构建 nn.Sequential 推理链
        # ================================================================================
        # 推理时不需要逐层 clamp，因为训练完成后权重已经收敛。
        # 直接连续矩阵乘法：input → feature_layer.w → output_layer.w → output
        # 这比训练时逐层 clamp 快得多，是 VPRTempo 高速推理的关键。
        # ================================================================================
        self.inferences = []
        for model in models:
            self.inferences.append(nn.Sequential(
                model.feature_layer.w,
                model.output_layer.w,
            ))
            self.inferences[-1].to(torch.device(self.device))
        
        # ----------------------------------------
        # 逐行说明：收集所有查询图像的输出脉冲
        # ----------------------------------------
        out = []      # 存储每帧查询图像的输出脉冲向量
        labels = []   # 存储对应的标签（用于调试/验证）
        for spikes, label in test_loader:
            spikes = spikes.to(self.device)
            labels.append(label.detach().cpu().item())
            
            # 前向传播：输入脉冲经过所有模块，拼接输出
            spikes = self.forward(spikes)
            out.append(spikes.detach().cpu())
            pbar.update(1)
        pbar.close()

        # ================================================================================
        # 逐行说明：构建相似度矩阵 S
        # ================================================================================
        # out 是长度为 query_places 的列表，每个元素形状 [1, total_output_neurons]
        # stack 后形状：[1, total_output_neurons, query_places]
        # squeeze(0) 后：[total_output_neurons, query_places]
        # 即 S[i, j] 表示第 j 个查询图像与第 i 个数据库地点的相似度（脉冲幅度）
        # 对应公式 (9) 中的 x_i（用于 argmax 匹配）
        # ================================================================================
        out = torch.stack(out, dim=2)
        out = out.squeeze(0).numpy()
       
        # ================================================================================
        # 逐行说明：构建 Ground Truth 矩阵 GT
        # ================================================================================
        # GT 是二值矩阵，GT[i, j] = 1 表示查询 j 的真实匹配地点是数据库 i。
        # 对于"on-the-rails"数据集（Nordland/Oxford），匹配是顺序对应的。
        # skip != 0 时：查询图像从数据库第 skip 帧开始对应。
        # ================================================================================
        if self.skip != 0:
            GT = np.zeros((model.database_places, model.query_places))
            skip = model.skip // model.filter
            query_indices = np.arange(model.query_places)
            GT[skip + query_indices, query_indices] = 1
        else:
            GT = np.eye(model.database_places, model.query_places)

        # ================================================================================
        # 逐行说明：应用 GT 容差（GT_tolerance）
        # ================================================================================
        # VPR 任务中，相邻几帧可能对应同一地点（100米范围内）。
        # GT_tolerance 允许在对角线附近的一定窗口内都视为正确匹配。
        # 例如 GT_tolerance=5：真实匹配点 ±5 帧都视为正例。
        # ================================================================================
        if self.GT_tolerance > 0:
            num_rows, num_cols = GT.shape
            for col in range(num_cols):
                ones_indices = np.where(GT[:, col] == 1)[0]
                for row in ones_indices:
                    start_row = max(row - self.GT_tolerance, 0)
                    end_row = min(row + self.GT_tolerance + 1, num_rows)
                    GT[start_row:end_row, col] = 1
        
        # ================================================================================
        # 逐行说明：生成 Precision-Recall 曲线（可选）
        # ================================================================================
        # 调用 metrics.py 中的 createPR，对相似度矩阵 S 设定不同阈值，
        # 计算 Precision/Recall 并保存为 JSON。
        # ================================================================================
        if model.PR_curve:
            P, R = createPR(out, GT, matching='single', n_thresh=100)
            PR_data = {"Precision": P, "Recall": R}
            full_path = f"{model.output_folder}/PR_curve_data.json"
            with open(full_path, 'w') as file:
                json.dump(PR_data, file) 
            if not model.run_demo:
                plt.plot(R, P)    
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.show()
                plt.close()

        # ================================================================================
        # 逐行说明：绘制相似度矩阵和 GT（可选）
        # ================================================================================
        if model.sim_mat and not model.run_demo:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            cax1 = axs[0].matshow(out, cmap='viridis')
            fig.colorbar(cax1, ax=axs[0], shrink=0.8)
            axs[0].set_title('Similarity matrix')
            cax2 = axs[1].matshow(GT, cmap='plasma')
            fig.colorbar(cax2, ax=axs[1], shrink=0.8)
            axs[1].set_title('GT')
            plt.tight_layout()
            plt.show()
            plt.close()
        
        # ================================================================================
        # 逐行说明：计算 Recall@N（N = 1, 5, 10, 15, 20, 25）
        # ================================================================================
        # 对应 VPRTempo 论文 IV-C：
        #   Recall@1：强制每个查询只匹配一个最近邻，正确率多少
        #   Recall@N：真实匹配是否在前 N 个最相似候选中
        # ================================================================================
        N = [1, 5, 10, 15, 20, 25]
        RN = []
        for n in N:
            RN.append(round(recallAtK(out, GT, K=n), 2))

        # 若启用 demo 模式，运行动画演示
        if model.run_demo:
            demo(model.data_dir, model.query_dir[0], model.database_dirs[0], out, GT, N, RN, R, P)
            return

        # 打印结果表格
        table = PrettyTable()
        table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        table.add_row(["Recall", RN[0], RN[1], RN[2], RN[3], RN[4], RN[5]])
        self.logger.info(table)

    # ================================================================================
    # 函数：forward —— III-C 并行张量推理
    # ================================================================================
    # 理论来源：VPRTempo 论文 III-C "Efficient implementation"
    # 核心操作：torch.cat(outputs, dim=1)
    #   每个 module 独立计算输出向量，最终在维度 1（神经元维度）上拼接，
    #   形成完整数据库的相似度向量。
    # 时间复杂度：O(log n) 查询缩放（论文 Fig. 2C）
    # ================================================================================
    def forward(self, spikes):
        """
        ================================================================================
        函数层说明：推理前向传播（多模块并行）
        ================================================================================
        输入：spikes [1, input_dim] —— 单张查询图像的脉冲编码
        输出：concatenated_output [1, total_output_neurons] —— 所有模块输出拼接
        ================================================================================
        """
        in_spikes = spikes.detach().clone()
        outputs = []

        # 遍历所有 module 的 nn.Sequential，独立计算输出
        for inference in self.inferences:
            out_spikes = inference(in_spikes)
            outputs.append(out_spikes)

        # 在维度 1（输出神经元维度）上拼接
        # 例如 module 0 输出 [1, 500], module 1 输出 [1, 500]
        # 拼接后 [1, 1000]，表示完整数据库中 1000 个地点的相似度
        concatenated_output = torch.cat(outputs, dim=1)
        
        return concatenated_output
        
    def load_model(self, models, model_path):
        """
        ================================================================================
        函数层说明：加载预训练的多模块组合模型
        ================================================================================
        加载方式：
          combined_state_dict['model_0'] → models[0].load_state_dict()
          combined_state_dict['model_1'] → models[1].load_state_dict()
          ...
        ================================================================================
        """
        if not os.path.exists(model_path):
            if model_path == self.demo:
                get_data_model()  # 自动下载 demo 数据和模型
            else:
                raise ValueError(f"Model path {model_path} does not exist.")
            
        combined_state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        for i, model in enumerate(models):
            model.load_state_dict(combined_state_dict[f'model_{i}'])
            model.eval()


# ================================================================================
# 函数：run_inference —— 推理总控入口
# ================================================================================
# 由 main.py 调用，完成：DataLoader 构建 → 模型加载 → evaluate() 调用
# ================================================================================
def run_inference(models, model_name):
    """
    ================================================================================
    函数层说明：运行推理流程
    ================================================================================
    输入：
        models     : VPRTempo 实例列表（已初始化）
        model_name : 预训练模型文件名
    ================================================================================
    """
    # 以第一个模块为参数基准
    model = models[0]
    
    # 图像预处理
    image_transform = ProcessImage(model.dims, model.patches)
    
    # 构建查询数据集
    test_dataset = CustomImageDataset(
        annotations_file=model.dataset_file, 
        base_dir=model.data_dir,
        img_dirs=model.query_dir,
        transform=image_transform,
        max_samples=model.query_places,
        filter=model.filter,
        skip=model.skip
    )

    # DataLoader（推理时不需要 shuffle）
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

    # 加载模型权重
    model.load_model(models, os.path.join('./vprtempo/models', model_name))

    # 推理（关闭梯度计算）
    with torch.no_grad():
        model.evaluate(models, test_loader)
