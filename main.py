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
main.py —— CLI 入口与训练/推理编排
================================================================================
理论对应：
  - III-B 模块化          → initialize_and_run_model() 第 87-102 行
  - IV-E 超参数搜索        → parse_network() 中所有 argparse 参数定义
  - 表 I 超参数默认值      → parse_network() 中 --filter, --epoch, --patches, --dims 等
================================================================================
"""

import os
import sys
import torch
import argparse

import torch.quantization as quantization

from tqdm import tqdm
from vprtempo.VPRTempo import VPRTempo, run_inference
from vprtempo.VPRTempoTrain import VPRTempoTrain, train_new_model
from vprtempo.src.loggers import model_logger, model_logger_quant
from vprtempo.VPRTempoQuant import VPRTempoQuant, run_inference_quant
from vprtempo.VPRTempoQuantTrain import VPRTempoQuantTrain, train_new_model_quant


def generate_model_name(model, quant=False, custom_name=None):
    """
    ================================================================================
    函数层说明：自动生成模型文件名
    ================================================================================
    命名规则：<database_dirs>_VPRTempo[_Quant]_IN<input>_FN<feature>_DB<places>.pth
    示例：springfall_VPRTempo_IN3136_FN6272_DB500.pth
    ================================================================================
    """
    if custom_name:
        if not custom_name.endswith('.pth'):
            custom_name += '.pth'
        return custom_name
    
    if quant:
        model_name = (''.join(model.database_dirs)+"_"+
                "VPRTempoQuant_" +
                "IN"+str(model.input)+"_" +
                "FN"+str(model.feature)+"_" + 
                "DB"+str(model.database_places) +
                ".pth")
    else:
        model_name = (''.join(model.database_dirs)+"_"+
                "VPRTempo_" +
                "IN"+str(model.input)+"_" +
                "FN"+str(model.feature)+"_" + 
                "DB"+str(model.database_places) +
                ".pth")
    return model_name

def check_pretrained_model(model_name):
    """
    检查预训练模型是否存在，若存在则提示用户是否重新训练。
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
# 函数：initialize_and_run_model —— 训练/推理总控（III-B 模块化核心）
# ================================================================================
# 该函数是 main.py 的核心，负责：
#   1. 根据 database_places 和 max_module 计算模块数量
#   2. 计算每个模块的输出维度（处理非整除情况）
#   3. 根据 args.train_new_model 和 args.quantize 分发到 4 条路径：
#      - fp32 训练、fp32 推理、int8 QAT 训练、int8 QAT 推理
# ================================================================================
def initialize_and_run_model(args, dims):
    """
    ================================================================================
    函数层说明：VPRTempo/VPRTempoQuant 的训练或推理总控入口
    ================================================================================
    输入：
        args : argparse.Namespace，包含所有命令行参数
        dims : 图像缩放尺寸 [H, W]
    核心逻辑：
        1. 计算 num_modules（模块数）和 out_dim（每模块输出神经元数）
        2. if train_new_model:
              if quantize: 创建 VPRTempoQuantTrain 列表 → train_new_model_quant()
              else:        创建 VPRTempoTrain 列表      → train_new_model()
           else:
              if quantize: 创建 VPRTempoQuant 列表      → run_inference_quant()
              else:        创建 VPRTempo 列表           → run_inference()
    ================================================================================
    """
    # ================================================================================
    # 逐行说明：计算模块数量（对应 III-B 模块化）
    # ================================================================================
    # 当 database_places > max_module 时，将数据库拆分为多个不重叠的模块。
    # 例如：database_places=3300, max_module=1100 → num_modules=3
    # 每个模块独立训练，推理时输出拼接。
    # 这是 VPRTempo 实现大规模地点识别的核心机制。
    # ================================================================================
    places = args.database_places  # 数据库总地点数
    num_modules = 1
    while places > args.max_module:
        places -= args.max_module
        num_modules += 1

    # ================================================================================
    # 逐行说明：计算每个模块的输出维度
    # ================================================================================
    # 处理非整除情况：若 3300 / 3 = 1100（整除），则每模块输出 1100 神经元。
    # 若 500 / 3 = 166 余 2，则前 2 个模块输出 166，最后一个输出 168。
    # remainder 用于调整最后一个模块的输出神经元数。
    # ================================================================================
    remainder = args.database_places % args.max_module
    if remainder != 0:
        # 有余数：前 (num_modules-1) 个模块均分，最后一个模块负责 remainder
        out_dim = int((args.database_places - remainder) / (num_modules - 1))
        final_out_dim = remainder
    else:
        out_dim = int(args.database_places / num_modules)
        final_out_dim = out_dim

    # ================================================================================
    # 分支一：训练新模型
    # ================================================================================
    if args.train_new_model:
        
        # ----------------------------------------
        # 子分支 A：量化感知训练（QAT）
        # ----------------------------------------
        if args.quantize:
            models = []
            logger = model_logger_quant()
            # fbgemm 是 PyTorch 推荐的 x86 CPU 量化后端
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            
            final_out = None
            for mod in tqdm(range(num_modules), desc="Initializing modules"):
                # 创建量化训练模型实例
                model = VPRTempoQuantTrain(args, dims, logger, num_modules, out_dim, out_dim_remainder=final_out)
                model.train()
                model.qconfig = qconfig
                # prepare_qat：插入 FakeQuantize 观察器到权重和激活中
                quantization.prepare_qat(model, inplace=True)
                models.append(model)
                # 倒数第二个模块时，设置 final_out 为 remainder
                if mod == num_modules - 2:
                    final_out = final_out_dim
            
            model_name = generate_model_name(model, args.quantize, args.model_name)
            check_pretrained_model(model_name)
            train_new_model_quant(models, model_name)

        # ----------------------------------------
        # 子分支 B：基础 fp32 训练
        # ----------------------------------------
        else:
            models = []
            logger = model_logger()
            
            final_out = None
            for mod in tqdm(range(num_modules), desc="Initializing modules"):
                # 创建 fp32 训练模型实例
                model = VPRTempoTrain(args, dims, logger, num_modules, out_dim, out_dim_remainder=final_out)
                model.to(torch.device('cpu'))  # 初始化时先放 CPU，训练时再移设备
                models.append(model)
                if mod == num_modules - 2:
                    final_out = final_out_dim

            model_name = generate_model_name(model, custom_name=args.model_name)
            print(f"Model name: {model_name}")
            check_pretrained_model(model_name)
            train_new_model(models, model_name)

    # ================================================================================
    # 分支二：推理（Inference）
    # ================================================================================
    else:
        
        # ----------------------------------------
        # 子分支 C：量化模型推理
        # ----------------------------------------
        if args.quantize:
            models = []
            logger, output_folder = model_logger_quant()
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            final_out = None
            for _ in tqdm(range(num_modules), desc="Initializing modules"):
                model = VPRTempoQuant(args, dims, logger, num_modules, output_folder, out_dim, out_dim_remainder=final_out)
                model.eval()
                model.qconfig = qconfig
                quantization.prepare(model, inplace=True)
                quantization.convert(model, inplace=True)  # 转换为 int8 表示
                models.append(model)
            model_name = generate_model_name(model, args.quantize, args.model_name)
            run_inference_quant(models, model_name)
            
        # ----------------------------------------
        # 子分支 D：基础 fp32 推理
        # ----------------------------------------
        else:
            models = []
            logger, output_folder = model_logger()
            
            final_out = None
            for mod in tqdm(range(num_modules), desc="Initializing modules"):
                model = VPRTempo(args, dims, logger, num_modules, output_folder, out_dim, out_dim_remainder=final_out)
                model.eval()
                model.to(torch.device('cpu'))
                models.append(model)
                if mod == num_modules - 2:
                    final_out = final_out_dim
            
            model_name = generate_model_name(model, custom_name=args.model_name)
            print(f"Model name: {model_name}")
            run_inference(models, model_name)


def parse_network():
    '''
    ================================================================================
    函数层说明：命令行参数解析器（对应表 I 超参数默认值）
    ================================================================================
    定义所有用户可配置参数及其默认值。
    关键参数与论文对应：
      --database_places : 训练地点数（论文 Nordland 3300，Oxford 450）
      --max_module      : 每模块最大地点数（论文 1100）
      --filter          : 子采样步长（论文每 8 秒取一帧，即 filter=8）
      --epoch           : 训练轮数（论文表 I 默认 4）
      --patches         : Patch Normalization 窗口大小（论文 7×7）
      --dims            : 图像缩放尺寸（论文 28×28，代码默认 56×56）
    ================================================================================
    '''
    parser = argparse.ArgumentParser(description="Args for base configuration file")

    # 数据集参数
    parser.add_argument('--dataset', type=str, default='nordland',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='./vprtempo/dataset/',
                            help="Directory where dataset files are stored")
    parser.add_argument('--database_places', type=int, default=500,
                            help="Number of places to use for training")
    parser.add_argument('--query_places', type=int, default=500,
                            help="Number of places to use for inferencing")
    parser.add_argument('--max_module', type=int, default=500,
                            help="Maximum number of images per module")
    parser.add_argument('--database_dirs', type=str, default='spring,fall',
                            help="Directories to use for training")
    parser.add_argument('--query_dir', type=str, default='summer',
                            help="Directories to use for testing")
    parser.add_argument('--GT_tolerance', type=int, default=0,
                            help="Ground truth tolerance for matching")
    parser.add_argument('--skip', type=int, default=0,
                            help="Images to skip for training and/or inferencing")

    # 训练参数
    parser.add_argument('--filter', type=int, default=8,
                            help="Images to skip for training and/or inferencing")
    parser.add_argument('--epoch', type=int, default=4,
                            help="Number of epochs to train the model")

    # 图像变换参数
    parser.add_argument('--patches', type=int, default=15,
                            help="Number of patches to generate for patch normalization image into")
    parser.add_argument('--dims', type=str, default="56,56",
                        help="Dimensions to resize the image to")

    # 网络功能开关
    parser.add_argument('--train_new_model', action='store_true',
                            help="Flag to run the training or inferencing model")
    parser.add_argument('--quantize', action='store_true',
                            help="Enable/disable quantization for the model")
    parser.add_argument('--model_name', type=str, default=None,
                            help="Custom model name (optional). If provided, overrides auto-generated name.")
    
    # 评估功能开关
    parser.add_argument('--PR_curve', action='store_true',
                            help="Flag to generate a Precision-Recall curve")
    parser.add_argument('--sim_mat', action='store_true',
                            help="Flag to plot the similarity matrix, GT, and GTsoft")
    
    # 演示模式
    parser.add_argument('--run_demo', action='store_true',
                            help="Flag to run the demo script")
    
    # 解析参数
    args = parser.parse_args()
    dims = [int(x) for x in args.dims.split(",")]

    # 运行网络
    initialize_and_run_model(args, dims)

if __name__ == "__main__":
    parse_network()
