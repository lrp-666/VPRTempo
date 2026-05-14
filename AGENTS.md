# VPRTempo — 智能体开发指南

> VPRTempo：一种用于视觉场景识别（VPR）的快速时序编码脉冲神经网络
> 昆士兰科技大学，机器人研究中心
> 版本：1.1.11 (setup.py) / 1.1.10 (pixi.toml)

---

## 项目概述

VPRTempo 是一个基于 PyTorch 的脉冲神经网络（SNN），用于视觉场景识别（VPR）。它使用时序编码的脉冲和自定义学习规则（STDP + 内在阈值可塑性）将查询图像与参考场景数据库进行匹配。该架构源自 [BLiTNet](https://arxiv.org/pdf/2208.01204.pdf)，并适配于 [VPRSNN](https://github.com/QVPR/VPRSNN) 框架。

本仓库提供两种变体：
- **VPRTempo** — 基础 fp32 模型。
- **VPRTempoQuant** — 支持量化感知训练（QAT）的 int8 模型。

开箱即用的支持数据集：Nordland（春/秋/夏/冬）和 Oxford RobotCar（晴/雨/黄昏）。自定义数据集可通过提供图像文件夹和 CSV 标注文件来使用。

---

## 技术栈

- **语言：** Python >=3.6, <3.13
- **深度学习：** PyTorch >=2.4.0, torchvision >=0.19.0
- **科学计算：** numpy >=1.26.2 (<2), pandas >=2.2.2
- **工具库：** tqdm, prettytable, matplotlib, requests, imageio
- **依赖管理：** [pixi](https://pixi.sh)（主要）；conda/micromamba 和 pip 作为替代方案
- **许可证：** MIT（注意：`vprtempo/src/metrics.py` 为 GPL v3，由 Stefan Schubert 导入）

---

## 仓库结构

```
├── main.py                      # CLI 入口；参数解析与编排
├── pixi.toml                    # Pixi 清单（依赖、任务、环境）
├── setup.py                     # PyPI 分发的 setuptools 配置
├── requirements.txt             # 纯 pip 依赖（不推荐）
├── vprtempo/
│   ├── __init__.py              # 包初始化；暴露 __version__
│   ├── VPRTempo.py              # 推理模型（fp32）
│   ├── VPRTempoTrain.py         # 训练模型（fp32）
│   ├── VPRTempoQuant.py         # 推理模型（int8 QAT）
│   ├── VPRTempoQuantTrain.py    # 训练模型（int8 QAT）
│   ├── src/
│   │   ├── blitnet.py           # 核心 SNNLayer（权重、STDP、阈值可塑性）
│   │   ├── dataset.py           # CustomImageDataset, ProcessImage, 块归一化
│   │   ├── loggers.py           # model_logger() 和 model_logger_quant()
│   │   ├── metrics.py           # VPR 指标：recallAtK, createPR, recallAt100precision
│   │   ├── demo.py              # Matplotlib 动画演示
│   │   ├── download.py          # 预训练模型/数据集 Dropbox 下载器
│   │   ├── nordland.py          # Nordland 数据集解压/重命名助手
│   │   ├── create_data_csv.py   # 从图像文件夹生成 CSV 标注
│   │   └── process_orc.py       # Oxford RobotCar 预处理助手
│   ├── dataset/                 # CSV 标注文件（nordland-*.csv, orc-*.csv）
│   └── models/                  # 预训练模型存储（.gitkeep + README.txt）
├── tutorials/                   # Jupyter 笔记本（中/英文）：基础演示、训练、模块
├── .github/workflows/main.yml   # GitHub 发布时自动发布到 PyPI
└── orc_list.txt                 # Oxford RobotCar 运行标识符
```

---

## 构建、运行与开发命令

### 主要方式：pixi

如果未安装 pixi：`curl -fsSL https://pixi.sh/install.sh | bash`

```bash
# 运行预训练演示（下载约 600 MB 的模型 + 图像）
pixi run demo

# 训练/评估基础 fp32 模型
pixi run train
pixi run eval

# 训练/评估量化 int8 模型
pixi run train_quant
pixi run eval_quant

# 复现会议结果（Nordland）
pixi run nordland_train
pixi run nordland_eval

# 复现会议结果（Oxford RobotCar）
pixi run oxford_train
pixi run oxford_eval
```

Pixi 环境：
- 默认：仅 CPU 的 PyTorch
- `cuda`：`pixi run --environment cuda <task>` 启用 CUDA 12 支持（仅限 Linux x86_64）

### 替代方案：conda/micromamba

```bash
micromamba create -n vprtempo -c conda-forge vprtempo
micromamba activate vprtempo
```

### 直接 Python 执行

```bash
python main.py --train_new_model   # 训练
python main.py                     # 评估
python main.py --quantize          # 量化评估
```

---

## CLI 参数（main.py）

通过 `argparse` 暴露的关键参数：
- `--dataset` — 数据集名称（`nordland`、`orc` 或自定义）
- `--data_dir` — 基础数据集目录（默认：`./vprtempo/dataset/`）
- `--database_places` / `--query_places` — 参考/查询图像数量
- `--max_module` — 每个模块的最大图像数（默认 500）；更大的数据库会被拆分为多个模块
- `--database_dirs` / `--query_dir` — 逗号分隔的文件夹名称
- `--dims` — 图像缩放尺寸，例如 `56,56`
- `--patches` — 块归一化的块大小（默认 15）
- `--filter` — 帧子采样步长（默认 8）
- `--epoch` — 训练轮数（默认 4）
- `--GT_tolerance` — 地面真值对角线容差（像素）
- `--skip` — 起始处跳过的图像数
- `--train_new_model` — 训练而非推理
- `--quantize` — 使用 QAT 变体
- `--PR_curve` — 输出精确率-召回率曲线
- `--sim_mat` — 绘制相似度矩阵和地面真值
- `--run_demo` — 运行动画演示

---

## 代码组织与模块划分

### 核心模型架构

每个模型（`VPRTempo`、`VPRTempoTrain`、`VPRTempoQuant`、`VPRTempoQuantTrain`）都是一个 `torch.nn.Module`，包含两个动态添加的层：

1. **feature_layer** — `input -> feature`（feature = input * 2）
2. **output_layer** — `feature -> output`（output = 每个模块的场景数）

层通过 `add_layer(name, **kwargs)` 添加，该方法从 `blitnet.py` 实例化 `bn.SNNLayer`。

### 训练与推理

- 训练类（`VPRTempoTrain`、`VPRTempoQuantTrain`）包含：
  - STDP 权重更新（`blitnet.py` 中的 `calc_stdp`）
  - 内在阈值可塑性（ITP）
  - 学习率退火（`_anneal_learning_rate`）
  - 在输出层强制发放脉冲，将图像与输出神经元关联
- 推理类（`VPRTempo`、`VPRTempoQuant`）剥离学习机制，运行简单的 `nn.Sequential` 前向传递。

### 多模块支持

当 `database_places > max_module` 时，代码库将数据库拆分为多个模型实例（模块）。每个模块独立训练，推理时输出拼接。模型以组合状态字典保存/加载，键为 `model_0`、`model_1` 等。

### 图像流水线（`dataset.py`）

`ProcessImage` 执行以下步骤：
1. RGB -> 灰度
2. Gamma 校正
3. 缩放到 `dims`
4. 块归一化（`PatchNormalisePad`）— 滑动窗口上的局部均值/标准差归一化
5. 缩放到 uint8 并编码为脉冲（`SetImageAsSpikes`）

对于量化训练，`SetImageAsSpikes` 包含一个 `FakeQuantize` 观察器。

### 日志（`loggers.py`）

日志器写入 `./vprtempo/output/<DDMMYY-HH-MM-SS>/logfile.log` 并回显到控制台。启动时打印 ASCII 横幅和设备信息。

---

## 代码风格指南

- **许可证头：** 每个 Python 文件以 MIT 许可证说明开头，后跟简短的 `Imports` 注释块。
- **文档字符串：** 公共方法使用 Google 风格或 reST 风格的参数/类型文档。
- **进度条：** 所有长时间运行的循环使用 `tqdm`。
- **结果表格：** 评估时打印 `PrettyTable`，包含 Recall@1,5,10,15,20,25。
- **设备处理：** 显式检查 `cuda:0`、`mps` 和 `cpu`。注意：MPS 强制 DataLoader 中 `num_workers=0`。
- **模型命名约定：** `<database_dirs>_VPRTempo[_Quant]_IN<input>_FN<feature>_DB<places>.pth`
- **Git 忽略：** `__pycache__/`, 数据集图像文件夹, `*.pth`, `vprtempo/output/`, `.vscode/`, `.DS_Store`

---

## 测试与评估

本项目**没有正式的单元测试套件**（没有 pytest、unittest 或 CI 测试任务）。正确性通过以下方式评估：

- **Recall@K**（`metrics.py` 中的 `recallAtK`），K = 1, 5, 10, 15, 20, 25
- **精确率-召回率曲线**（`metrics.py` 中的 `createPR`）
- 地面真值矩阵是单位矩阵，带有可选的对角线容差（`GT_tolerance`）和跳过偏移

添加新功能时，通过运行以下命令验证：
```bash
pixi run eval --dataset <dataset> --database_dirs <dirs> --query_dir <dir>
```
并确认 Recall@K 值是否合理。

---

## 部署与分发

- **PyPI：** 由 `.github/workflows/main.yml` 在 GitHub 创建 Release 时自动发布，使用 `setup.py sdist bdist_wheel` + `twine upload`。
- **Conda：** 可在 `conda-forge` 上获取。
- **Pixi：** 提交 Lockfile（`pixi.lock`）以确保可复现的依赖解析。

---

## 安全注意事项

- 演示会自动从 Dropbox URL 下载预训练模型和图像，这些 URL 硬编码在 `vprtempo/src/download.py` 中。如果修改这些 URL，请确保它们可信。
- 在推理路径中，`torch.load` 以 `weights_only=True` 调用。
- 仓库中不存储任何密钥、API 密钥或凭证。

---

## 自定义数据集快速参考

1. 将图像放入 `./vprtempo/dataset/<name>/` 下的文件夹。
2. 使用 `vprtempo/src/create_data_csv.py` 生成 CSV（更新 `dataset_name` 变量）。
3. 训练：`pixi run train --dataset <name> --database_dirs <name>`
4. 评估：`pixi run eval --dataset <name> --database_dirs <name> --query_dir <name>`

---

## 有用链接

- 项目页面：https://vprtempo.github.io
- 论文（IEEE ICRA 2024）：https://ieeexplore.ieee.org/document/10610918
- 仓库：https://github.com/QVPR/VPRTempo
