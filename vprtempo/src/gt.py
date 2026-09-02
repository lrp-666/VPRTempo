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

# ================================================================================
# IDEA1 S1.4：Ground Truth 构造的单一事实来源（Single Source of Truth）
#
# 背景：轨 A（VPRTempo.evaluate）与轨 B（eval_retrieval.py）必须使用**逐比特一致**
# 的 GT。GT 错位一格会让 Recall 静默变得很难看且极难排查。
# 因此轨 A 原有的 GT 构造逻辑（VPRTempo.py 阶段 4/4b）抽取到这里，两边共用。
#
# 逻辑（与原 VPRTempo.py:402-437 一致）：
#   - skip != 0：GT[skip//filter + j, j] = 1（查询相对数据库有固定帧偏移）
#   - skip == 0：GT = 单位矩阵
#   - GT_tolerance > 0：每列的 1 向上下膨胀 ±tolerance 行
# ================================================================================
import numpy as np


def build_ground_truth(database_places, query_places, skip=0, filter=8, tolerance=0):
    """
    构建 VPR 地面真值矩阵。

    参数:
        database_places : 数据库地点数（GT 行数）
        query_places    : 查询地点数（GT 列数）
        skip            : 数据集开头跳过的帧数（会先整除 filter 换算成降采样后偏移）
        filter          : 帧子采样步长
        tolerance       : GT 对角线容差（行数），>0 时每列的正例行向 ±tolerance 膨胀

    返回:
        GT : np.ndarray [database_places, query_places]，0/1 矩阵
    """
    if skip != 0:
        GT = np.zeros((database_places, query_places))
        # 整除 filter 得到降采样后的帧偏移
        skip_frames = skip // filter
        query_indices = np.arange(query_places)
        GT[skip_frames + query_indices, query_indices] = 1
    else:
        GT = np.eye(database_places, query_places)

    if tolerance > 0:
        num_rows, num_cols = GT.shape
        for col in range(num_cols):
            ones_indices = np.where(GT[:, col] == 1)[0]
            for row in ones_indices:
                start_row = max(row - tolerance, 0)
                end_row = min(row + tolerance + 1, num_rows)
                GT[start_row:end_row, col] = 1

    return GT


def build_ground_truth_multiseason(database_places, query_places, n_seasons,
                                   skip=0, filter=8, tolerance=0):
    """
    轨 B 专用：多季节数据库（如 spring+fall 合并）的 GT。

    数据库特征按 CustomImageDataset(test=False) 的合并顺序排列：
    先季节 0 的全部地点，再季节 1 的全部地点……（每季节 database_places 行）。
    因此 GT 是单季节 GT 的纵向平铺：查询 j 匹配第 j 行（季节 0）、
    第 j + database_places 行（季节 1）……

    返回:
        GT : np.ndarray [n_seasons * database_places, query_places]
    """
    base = build_ground_truth(database_places, query_places, skip, filter, tolerance)
    return np.tile(base, (n_seasons, 1))
