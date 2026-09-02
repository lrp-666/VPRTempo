#!/usr/bin/env python
# ================================================================================
# seeded_ref_run.py — IDEA1 S1.3 / 第 -1b 步：种子化参照跑驱动
#
# 功能：在做任何事之前固定 torch/numpy/random 三件套种子，然后调用 main 的入口函数。
#       **不依赖任何新增参数或新代码**——因此：
#         - 在 main 分支上跑 → 得到"确定性化的 main 参照"（Gate 0 / B0 回归的真参照）；
#         - 在 feat/convstdp-base 分支上跑 → 两边必须逐比特一致（真·B0 回归）。
#
# 在 main 分支上运行的方法（文件只存在于 feat 分支时）：
#   git show feat/convstdp-base:IDEA1-covstdp/experiments/seeded_ref_run.py > /tmp/seeded_ref_run.py
#   pixi run python /tmp/seeded_ref_run.py --seed 0 --train
#
# 用法：pixi run python IDEA1-covstdp/experiments/seeded_ref_run.py --seed 0 [--train] [--eval]
# ================================================================================
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path.cwd()  # 要求从仓库根目录运行（相对路径依赖）
sys.path.insert(0, str(REPO_ROOT))


def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def build_namespace(seed: int, train: bool, model_name: str):
    """B0 参照配置（500 地，28×28/7×7，skip=0）——写死以保持独立、可跨分支复现"""
    return argparse.Namespace(
        dataset="nordland",
        data_dir="/mnt/d/Data/datasets/vpr/Nordland",
        database_places=500, query_places=500, max_module=500,
        database_dirs="spring,fall", query_dir="summer",
        GT_tolerance=0, skip=0, filter=8, epoch=4,
        patches=7, dims="28,28",
        train_new_model=train, quantize=False,
        model_name=model_name,
        PR_curve=True, sim_mat=False, run_demo=False,
    )


def model_md5(model_name: str) -> str:
    path = REPO_ROOT / "vprtempo" / "models" / model_name
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    cli = ap.parse_args()
    if not cli.train and not cli.eval:
        cli.eval = True

    model_name = f"b0ref__seed{cli.seed}.pth"

    # 先固定种子，再 import/调用任何会用到随机数的代码
    seed_all(cli.seed)
    from main import initialize_and_run_model

    if cli.train:
        # 覆盖旧模型（规避 check_pretrained_model 的交互式询问）
        mp = REPO_ROOT / "vprtempo" / "models" / model_name
        if mp.exists():
            mp.unlink()
        ns = build_namespace(cli.seed, train=True, model_name=model_name)
        t0 = time.time()
        initialize_and_run_model(ns, [28, 28])
        print(f"[seeded_ref] train done, wall={time.time()-t0:.1f}s, "
              f"model_md5={model_md5(model_name)}")

    if cli.eval:
        ns = build_namespace(cli.seed, train=False, model_name=model_name)
        t0 = time.time()
        initialize_and_run_model(ns, [28, 28])
        print(f"[seeded_ref] eval done, wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
