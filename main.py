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
【模块级注释】main.py —— VPRTempo 项目的 CLI 入口与训练/推理编排中心
================================================================================
【1. 文件定位】
    本文件是整个 VPRTempo 仓库的唯一命令行入口（CLI Entrypoint）。用户通过
    `python main.py [参数]` 触发所有训练、推理、评估和演示流程。

【2. 核心职责】
    a) 参数解析：通过 parse_network() 汇总所有 argparse 参数（数据集路径、
       图像尺寸、训练轮数、量化开关等）。
    b) 模块拆分：通过 initialize_and_run_model() 依据 database_places 与
       max_module 的比值，将大规模数据库动态拆分为多个子模型（Module），
       这是论文 III-B 节“模块化”机制的实现核心。
    c) 四路分发：根据 --train_new_model 和 --quantize 两个布尔开关，将
       执行流精确分发到以下四条路径：
           ├─ fp32  训练  → VPRTempoTrain      + train_new_model()
           ├─ fp32  推理  → VPRTempo           + run_inference()
           ├─ int8  训练  → VPRTempoQuantTrain + train_new_model_quant()
           └─ int8  推理  → VPRTempoQuant      + run_inference_quant()
    d) 模型命名：自动生成符合约定 <dataset>_VPRTempo[_Quant]_IN<>_FN<>_DB<>.pth
       的权重文件名，并检测本地是否已存在同名模型，防止误覆盖。

【3. 数据流概览】
    命令行参数(args, dims)
           ↓
    initialize_and_run_model()
           ↓
    ┌─────────────────┬─────────────────┐
    ↓                 ↓                 ↓
  计算模块数      计算每模块输出维度    四路分支选择
  num_modules     out_dim / final_out   (train/quant)
    ↓                 ↓                 ↓
  for 循环创建    模型列表(models)      训练 or 推理函数
  模型实例                              (保存/加载权重)
    ↓                 ↓                 ↓
  调用底层        vprtempo/*.py         输出结果到
  训练/推理        中的 SNN 层逻辑       ./vprtempo/output/

【4. 论文对应关系】
    - III-B 模块化              → initialize_and_run_model() 第 127–147 行
    - IV-E 超参数搜索           → parse_network() 中全部 argparse 定义
    - 表 I 超参数默认值          → --filter, --epoch, --patches, --dims 等
================================================================================
"""

# ================================================================================
# 【模块级】标准库与第三方库导入
# ================================================================================
# os/sys   : 用于路径拼接、文件存在性检查、程序退出控制
# torch    : PyTorch 深度学习框架，提供设备管理和量化 API
# argparse : Python 标准命令行参数解析库
# ================================================================================
import os
import sys
import torch
import argparse

import torch.quantization as quantization  # PyTorch 量化工具包（QAT / PTQ）

from tqdm import tqdm  # 进度条库，用于模块初始化的可视化

# ================================================================================
# 【模块级】项目内部模块导入
# ================================================================================
# VPRTempo / VPRTempoTrain         : 基础 fp32 模型的推理与训练类及入口函数
# VPRTempoQuant / VPRTempoQuantTrain: 量化 int8 模型的推理与训练类及入口函数
# model_logger / model_logger_quant : 分别创建普通/量化模式的日志记录器与输出目录
# ================================================================================
from vprtempo.VPRTempo import VPRTempo, run_inference
from vprtempo.VPRTempoTrain import VPRTempoTrain, train_new_model
from vprtempo.src.loggers import model_logger, model_logger_quant
from vprtempo.VPRTempoQuant import VPRTempoQuant, run_inference_quant
from vprtempo.VPRTempoQuantTrain import VPRTempoQuantTrain, train_new_model_quant


# ================================================================================
# 【函数级】generate_model_name —— 模型权重文件自动命名
# ================================================================================
# 【输入】
#   model      : 已初始化的模型实例（VPRTempo* 或其训练变体），用于读取结构参数
#   quant      : bool，是否为量化模型，决定是否插入 "Quant" 后缀
#   custom_name: str|None，用户通过 --model_name 指定的自定义文件名
# 【输出】
#   str : 最终使用的模型文件名（必须以 .pth 结尾）
# 【命名规则】
#   <database_dirs拼接>_VPRTempo[_Quant]_IN<input>_FN<feature>_DB<places>.pth
# 【示例】
#   database_dirs=['spring','fall'], input=3136, feature=6272, places=500
#   → springfall_VPRTempo_IN3136_FN6272_DB500.pth
# ================================================================================
def generate_model_name(model, quant=False, custom_name=None):
    """
    ================================================================================
    函数层说明：自动生成模型文件名
    ================================================================================
    命名规则：<database_dirs>_VPRTempo[_Quant]_IN<input>_FN<feature>_DB<places>.pth
    示例：springfall_VPRTempo_IN3136_FN6272_DB500.pth
    ================================================================================
    """
    # ------------------------------------------------------------------------
    # 【行级】若用户提供了自定义名称，直接返回并强制补全 .pth 后缀
    # ------------------------------------------------------------------------
    if custom_name:
        if not custom_name.endswith('.pth'):  #endwith: 确保用户输入的自定义名称以 .pth 结尾
            custom_name += '.pth'
        return custom_name
    
    # ------------------------------------------------------------------------
    # 【行级】依据 quant 标志拼接字符串，提取模型内部维度信息
    # model.database_dirs : 列表，如 ['spring', 'fall']
    # model.input         : 输入层神经元数（= H×W，如 56×56=3136）
    # model.feature       : 特征层神经元数（通常为 input×2）
    # model.database_places: 数据库总地点数
    # ------------------------------------------------------------------------
    if quant:
        model_name = (''.join(model.database_dirs)+"_"+
                "VPRTempoQuant_" +
                "IN"+str(model.input)+"_" +
                "FN"+str(model.feature)+"_" + 
                "DB"+str(model.database_places) +
                ".pth")
    else:
        # 【IDEA1 S2.5】conv 前端标记，防止覆盖 B0 模型（S2.5 卡片）
        frontend_tag = ""
        if getattr(model, 'frontend', 'none') != 'none':
            frontend_tag = f"CONVC{getattr(model, 'conv_channels', 32)}K{getattr(model, 'conv_kernel', 5)}_"
        model_name = (''.join(model.database_dirs)+"_"+
                "VPRTempo_" + frontend_tag +
                "IN"+str(model.input)+"_" +
                "FN"+str(model.feature)+"_" + 
                "DB"+str(model.database_places) +
                ".pth")
    return model_name


# ================================================================================
# 【函数级】check_pretrained_model —— 预训练模型冲突检测
# ================================================================================
# 【输入】
#   model_name : str，待检查的文件名
# 【副作用】
#   若 ./vprtempo/models/<model_name> 已存在，则在终端交互式询问用户是否覆盖。
#   用户输入 'n' 时直接调用 sys.exit() 终止程序，防止误删已有权重。
# 【输出】
#   bool : True 表示用户同意重新训练/覆盖；若文件不存在则默认返回 None（隐式）
# ================================================================================
def check_pretrained_model(model_name):
    """
    检查预训练模型是否存在，若存在则提示用户是否重新训练。
    """
    # ------------------------------------------------------------------------
    # 【行级】构造完整路径并判断文件系统是否存在；仅在训练路径中被调用
    # ------------------------------------------------------------------------
    if os.path.exists(os.path.join('./vprtempo/models', model_name)):
        prompt = "A network with these parameters exists, re-train network? (y/n):\n"
        retrain = input(prompt).strip().lower()
        if retrain == 'y':
            return True
        elif retrain == 'n':
            print('Training new model cancelled')
            sys.exit()


# ================================================================================
# 【函数级】initialize_and_run_model —— 训练/推理总控核心（III-B 模块化实现）
# ================================================================================
# 【输入】
#   args : argparse.Namespace，包含全部命令行超参数
#   dims : list[int]，图像缩放后的高和宽，如 [56, 56]
# 【核心职责】
#   1. 根据 database_places 与 max_module 计算需要拆分成多少个子模型(num_modules)。
#   2. 处理不能整除的情况，得到每模块输出神经元数 out_dim 和最后一个模块的
#      特殊输出维度 final_out_dim。
#   3. 用 for 循环实例化模型列表（每个模块一个模型），并依据 train/quant 标志
#      分发到对应的训练或推理函数。
# 【论文映射】
#   这里的循环拆分直接对应论文 III-B “Modular Approach”：当数据库规模超过单模块
#   容量上限时，将地点集划分为互不重叠的子集，分别训练后拼接输出。
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
    # 【IDEA1 S1.3】可选种子固定（默认 args.seed=None 时完全跳过，行为与原来一致）
    # 必须在创建任何模型/DataLoader 之前执行：blitnet.addWeights 的稀疏化用
    # np.random（blitnet.py:308），权重初始化用 torch.normal_（blitnet.py:290），
    # 两者都在这里覆盖。DataLoader 的 shuffle 顺序由 train_new_model 内的
    # generator/worker_init_fn 负责（使用 model.seed）。
    # ================================================================================
    if getattr(args, 'seed', None) is not None:
        import random as _random
        import numpy as _np
        torch.manual_seed(args.seed)
        _np.random.seed(args.seed)
        _random.seed(args.seed)

    # ================================================================================
    # 【行级/块级】计算模块数量（对应 III-B 模块化）
    # ================================================================================
    # 当 database_places > max_module 时，将数据库拆分为多个不重叠的模块。
    # 例如：database_places=3300, max_module=1100 → num_modules=3
    # 每个模块独立训练，推理时输出拼接。
    # 这是 VPRTempo 实现大规模地点识别的核心机制。
    # ================================================================================
    places = args.database_places  # 数据库总地点数（如 Nordland 3300）
    num_modules = 1                # 初始假设仅需 1 个模块
    while places > args.max_module:
        places -= args.max_module  # 逐次减去一个模块容量
        num_modules += 1           # 每减一次，模块计数 +1

    # ================================================================================
    # 【行级/块级】计算每个模块的输出维度（处理非整除余数）
    # ================================================================================
    # 处理非整除情况：若 3300 / 3 = 1100（整除），则每模块输出 1100 神经元。
    # 若 500 / 3 = 166 余 2，则前 2 个模块各输出 167 神经元？
    # 注意代码逻辑：remainder ≠ 0 时，前 (num_modules-1) 个模块输出 out_dim，
    # 最后一个模块输出 final_out_dim = remainder。
    # 例如 500 % 500 = 0 → out_dim=500, final_out_dim=500（整除退化情况）。
    # 若 database_places=700, max_module=500 → 循环后 num_modules=2, remainder=200
    #   → out_dim=(700-200)/(2-1)=500, final_out_dim=200
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
    # 【分支一】训练新模型（train_new_model=True）
    # ================================================================================
    if args.train_new_model:
        
        # ----------------------------------------
        # 【子分支 A】量化感知训练（QAT, int8）
        # ----------------------------------------
        if args.quantize:
            models = []                       # 存放各模块模型的列表
            logger = model_logger_quant()     # 初始化量化模式日志器
            # fbgemm 是 PyTorch 推荐的 x86 CPU 量化后端，支持高效的 int8 内核
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            
            final_out = None                  # 前几个模块默认使用标准 out_dim；在倒数第二个模块时改为 final_out_dim
            # tqdm 进度条：可视化模块初始化过程，desc 为进度条前缀描述
            for mod in tqdm(range(num_modules), desc="Initializing modules"):
                # 实例化量化训练模型；out_dim_remainder 仅在最后一个模块生效
                model = VPRTempoQuantTrain(args, dims, logger, num_modules, out_dim, out_dim_remainder=final_out)
                model.train()                 # 设为训练模式（启用 Dropout/BN 学习等）
                model.qconfig = qconfig       # 将全局 qconfig 绑定到模型
                # prepare_qat：在模型中插入 FakeQuantize 观察器，模拟 int8 前向/反向
                quantization.prepare_qat(model, inplace=True)
                models.append(model)
                # 当循环到倒数第二个模块时，将 final_out 设为余数维度，
                # 使得下一次（即最后一个模块）实例化时使用 final_out_dim
                if mod == num_modules - 2:
                    final_out = final_out_dim
            
            # 生成模型文件名，并检查是否已存在同名权重
            model_name = generate_model_name(model, args.quantize, args.model_name)
            check_pretrained_model(model_name)
            # 调用量化训练入口函数，执行完整的 STDP + ITP 学习流程
            train_new_model_quant(models, model_name)

        # ----------------------------------------
        # 【子分支 B】基础 fp32 训练
        # ----------------------------------------
        else:
            models = []
            logger = model_logger()           # 初始化普通模式日志器
            
            final_out = None
            for mod in tqdm(range(num_modules), desc="Initializing modules"):
                # 创建 fp32 训练模型实例
                model = VPRTempoTrain(args, dims, logger, num_modules, out_dim, out_dim_remainder=final_out)
                model.to(torch.device('cpu')) # 初始化时先放 CPU，训练函数内部再移目标设备
                models.append(model)
                if mod == num_modules - 2:
                    final_out = final_out_dim

            # 生成文件名、打印、检查冲突，然后启动 fp32 训练
            model_name = generate_model_name(model, custom_name=args.model_name)
            print(f"Model name: {model_name}")
            check_pretrained_model(model_name)
            train_new_model(models, model_name)

    # ================================================================================
    # 【分支二】推理/评估（train_new_model=False）
    # ================================================================================
    else:
        
        # ----------------------------------------
        # 【子分支 C】量化模型推理（int8）
        # ----------------------------------------
        if args.quantize:
            models = []
            logger, output_folder = model_logger_quant()  # 量化日志器同时返回输出目录
            qconfig = quantization.get_default_qat_qconfig('fbgemm')
            final_out = None
            for _ in tqdm(range(num_modules), desc="Initializing modules"):
                model = VPRTempoQuant(args, dims, logger, num_modules, output_folder, out_dim, out_dim_remainder=final_out)
                model.eval()                  # 设为评估模式（关闭梯度、固定 BN/DP）
                model.qconfig = qconfig
                quantization.prepare(model, inplace=True)   # 准备量化图
                quantization.convert(model, inplace=True)   # 将模型转换为真正的 int8 表示
                models.append(model)
            # 生成模型名并启动量化推理
            model_name = generate_model_name(model, args.quantize, args.model_name)
            run_inference_quant(models, model_name)
            
        # ----------------------------------------
        # 【子分支 D】基础 fp32 推理
        # ----------------------------------------
        else:
            models = []
            logger, output_folder = model_logger()  # 普通日志器同时返回输出目录
            
            final_out = None
            for mod in tqdm(range(num_modules), desc="Initializing modules"):
                model = VPRTempo(args, dims, logger, num_modules, output_folder, out_dim, out_dim_remainder=final_out)
                model.eval()
                model.to(torch.device('cpu'))
                models.append(model)
                if mod == num_modules - 2:  #在倒数第二个模块时，将 final_out 设为余数维度，使得最后一个模块使用 final_out_dim
                    final_out = final_out_dim
            
            # 生成模型名、打印，并启动 fp32 推理
            model_name = generate_model_name(model, custom_name=args.model_name)
            print(f"Model name: {model_name}")
            run_inference(models, model_name)


# ================================================================================
# 【函数级】parse_network —— 命令行参数解析与程序触发
# ================================================================================
# 【职责】
#   1. 定义并解析所有 argparse 参数（数据集、训练、图像变换、功能开关）。
#   2. 将 --dims 字符串（如 "56,56"）解析为整数列表 [56, 56]。
#   3. 调用 initialize_and_run_model() 进入核心逻辑。
# 【论文映射】
#   以下参数默认值与论文表 I 及实验章节直接对应：
#     --database_places : Nordland 数据集训练地点数（论文 3300）
#     --max_module      : 每模块最大地点数（论文 1100）
#     --filter          : 帧子采样步长（论文每 8 秒取一帧 → filter=8）
#     --epoch           : 训练轮数（论文默认 4）
#     --patches         : Patch Normalization 窗口大小（论文 7×7，代码默认 15）
#     --dims            : 图像缩放尺寸（论文 28×28，代码默认 56×56）
# ================================================================================
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
    # ------------------------------------------------------------------------
    # 【行级】创建 ArgumentParser 实例，description 会在用户输入 -h 时显示
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Args for base configuration file")

    # ------------------------------------------------------------------------
    # 【行级】数据集与路径相关参数
    # ------------------------------------------------------------------------
    # --dataset        : 数据集名称，影响 CSV 标注文件查找逻辑
    # --data_dir       : 数据集根目录，默认在项目内 ./vprtempo/dataset/
    # --database_places: 训练/建库时使用的地点（图像）数量
    # --query_places   : 查询/测试时使用的地点数量
    # --max_module     : 单个模块最多容纳的地点数；超过则触发模块化拆分
    # --database_dirs  : 逗号分隔的训练图像文件夹名（如 spring,fall）
    # --query_dir      : 查询图像文件夹名（如 summer）
    # --GT_tolerance   : 地面真值对角线容差像素值；>0 时允许一定位置偏移匹配
    # --skip           : 在数据集开头跳过的图像帧数
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # 【行级】训练超参数
    # ------------------------------------------------------------------------
    # --filter: 帧子采样间隔；每隔 filter 张图像取一帧，降低数据冗余
    # --epoch : 训练轮数；脉冲神经网络使用 STDP，通常 4 轮即可收敛
    # ------------------------------------------------------------------------
    parser.add_argument('--filter', type=int, default=8,
                            help="Images to skip for training and/or inferencing")
    parser.add_argument('--epoch', type=int, default=4,
                            help="Number of epochs to train the model")

    # ------------------------------------------------------------------------
    # 【行级】图像预处理参数
    # ------------------------------------------------------------------------
    # --patches: Patch Normalization 的滑动窗口尺寸（边长像素数）
    # --dims   : 图像缩放目标尺寸，格式 "高,宽"；将输入图像统一缩放到该分辨率
    # ------------------------------------------------------------------------
    parser.add_argument('--patches', type=int, default=15,
                            help="Number of patches to generate for patch normalization image into")
    parser.add_argument('--dims', type=str, default="56,56",
                        help="Dimensions to resize the image to")
    # --patch_norm: IDEA1 S1.2 新增。PatchNorm 开关，默认 on（不改变现有行为）。
    # off 时 ProcessImage 跳过局部块归一化，用于 PatchNorm × 前端交互消融（Table 3）。
    parser.add_argument('--patch_norm', type=str, default='on', choices=['on', 'off'],
                        help="Enable/disable patch normalization in preprocessing (IDEA1)")
    # --seed: IDEA1 S1.3 新增。默认 None = 不做任何种子固定（保持原有非确定性行为）。
    # 提供时固定 torch/numpy/random 三件套 + DataLoader shuffle 生成器。
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility (IDEA1); default None = unseeded (original behavior)")
    # ---- IDEA1 S2.5：卷积前端相关参数（默认 frontend='none' = B0 原行为）----
    parser.add_argument('--frontend', type=str, default='none',
                        choices=['none', 'conv_stdp', 'random_conv', 'gabor',
                                 'gabor_stdp', 'gabor_stdp_freesign',
                                 'conv_stdp_freesign'],
                        help="Conv frontend type (IDEA1); default none = original VPRTempo")
    parser.add_argument('--wta_mode', type=str, default='local', choices=['global', 'local', 'none'],
                        help="WTA competition mode for conv frontend (IDEA1)")
    parser.add_argument('--wta_block', type=int, default=4,
                        help="Local WTA block size (IDEA1)")
    parser.add_argument('--agg_mode', type=str, default='mean', choices=['mean', 'sum'],
                        help="Winner aggregation mode for conv STDP (IDEA1)")
    parser.add_argument('--pre_mode', type=str, default='centered', choices=['centered', 'amp', 'heaviside'],
                        help="Pre-term mode for conv STDP (IDEA1)")
    parser.add_argument('--conv_channels', type=int, default=32,
                        help="Number of conv frontend channels (IDEA1)")
    parser.add_argument('--conv_kernel', type=int, default=5,
                        help="Conv frontend kernel size (IDEA1)")
    parser.add_argument('--conv_epoch', type=int, default=2,
                        help="Conv frontend training epochs (IDEA1)")
    # ---- IDEA1 S3.2a 调参窗：conv 前端四个原硬编码超参（默认值与硬编码完全一致，默认行为不变）----
    parser.add_argument('--conv_thr_range', type=str, default="0,0.5",
                        help="Conv frontend initial threshold range \"lo,hi\" (IDEA1 tuning)")
    parser.add_argument('--conv_fire_rate', type=str, default="0.2,0.9",
                        help="Conv frontend target firing rate range \"lo,hi\" (IDEA1 tuning)")
    parser.add_argument('--conv_ip_rate', type=float, default=0.15,
                        help="Conv frontend ITP learning rate (IDEA1 tuning)")
    parser.add_argument('--conv_stdp_rate', type=float, default=0.005,
                        help="Conv frontend STDP learning rate (IDEA1 tuning)")
    # ---- IDEA1 S2.11 规则锦标赛 Round 1：四个独立可组合的规则手术开关（默认全关 = B2 不变）----
    parser.add_argument('--bcm_gate', action='store_true',
                        help="R1: BCM sliding threshold theta_M (EMA of post^2) replaces fixed 0.5 in conv STDP (IDEA1 S2.11)")
    parser.add_argument('--bcm_alpha', type=float, default=0.001,
                        help="R1: EMA rate for theta_M (10-50x slower than weight learning, prevents ITP oscillation)")
    parser.add_argument('--rank_push', action='store_true',
                        help="R2: rank-k channel within each WTA block gets -delta times update (IDEA1 S2.11)")
    parser.add_argument('--rank_delta', type=float, default=0.4,
                        help="R2: negative update strength delta for the rank-k channel")
    parser.add_argument('--rank_k', type=int, default=2,
                        help="R2: which cross-channel rank receives the negative update")
    parser.add_argument('--oja_decay', action='store_true',
                        help="R3: replace norm-preserving renorm with Oja decay term -post^2*w (IDEA1 S2.11)")
    parser.add_argument('--attractor', action='store_true',
                        help="R4: pre-term (pre-0.5) -> (patch - kernel) reconstruction attractor (IDEA1 S2.11)")

    # ------------------------------------------------------------------------
    # 【行级】网络功能开关（布尔标志）
    # ------------------------------------------------------------------------
    # --train_new_model: action='store_true' 表示命令行中出现该标志时值为 True，
    #                    否则为 False。控制进入训练分支还是推理分支。
    # --quantize       : 启用量化感知训练（QAT）或量化推理。
    # --model_name     : 允许用户覆盖自动生成的模型文件名。
    # ------------------------------------------------------------------------
    parser.add_argument('--train_new_model', action='store_true',
                            help="Flag to run the training or inferencing model")
    parser.add_argument('--quantize', action='store_true',
                            help="Enable/disable quantization for the model")
    parser.add_argument('--model_name', type=str, default=None,
                            help="Custom model name (optional). If provided, overrides auto-generated name.")
    
    # ------------------------------------------------------------------------
    # 【行级】评估与可视化功能开关
    # ------------------------------------------------------------------------
    # --PR_curve: 在推理结束后生成精确率-召回率（Precision-Recall）曲线
    # --sim_mat : 绘制相似度矩阵、硬地面真值（GT）和软地面真值（GTsoft）
    # ------------------------------------------------------------------------
    parser.add_argument('--PR_curve', action='store_true',
                            help="Flag to generate a Precision-Recall curve")
    parser.add_argument('--sim_mat', action='store_true',
                            help="Flag to plot the similarity matrix, GT, and GTsoft")
    
    # ------------------------------------------------------------------------
    # 【行级】演示模式开关
    # ------------------------------------------------------------------------
    # --run_demo: 启动 Matplotlib 动画演示（由 vprtempo/src/demo.py 实现）
    # ------------------------------------------------------------------------
    parser.add_argument('--run_demo', action='store_true',
                            help="Flag to run the demo script")
    
    # ------------------------------------------------------------------------
    # 【行级】参数解析与后处理
    # ------------------------------------------------------------------------
    args = parser.parse_args()                    # 解析用户输入的命令行参数
    dims = [int(x) for x in args.dims.split(",")] # 将 "56,56" → [56, 56]

    # ------------------------------------------------------------------------
    # 【行级】进入核心编排函数，正式启动训练或推理流程
    # ------------------------------------------------------------------------
    initialize_and_run_model(args, dims)


# ================================================================================
# 【模块级】主程序入口保护
# ================================================================================
# 当本文件被直接运行时（python main.py），__name__ == "__main__" 成立，
# 调用 parse_network() 开始执行；当被作为模块导入时，不自动执行，
# 符合 Python 标准实践，便于测试和复用。
# ================================================================================
if __name__ == "__main__":
    parse_network()
