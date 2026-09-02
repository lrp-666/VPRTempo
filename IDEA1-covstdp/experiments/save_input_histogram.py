#!/usr/bin/env python
# ================================================================================
# save_input_histogram.py — IDEA1 S1.2 验收证据
#
# 功能：对同一批 Nordland 图像分别用 patch_norm=on/off 跑 ProcessImage，
#       绘制最终 spike 幅度（uint8/255 ∈ [0,1]）的像素值直方图，存到 results/。
# 预期：on 时平坦背景 ≈ 0.5（127.5/255），off 时保留暗背景（≈0 峰）。
#       这是 PLAN.md S2.3 直流塌缩分析（Table 3b）的前置证据。
#
# 用法：pixi run python IDEA1-covstdp/experiments/save_input_histogram.py
# ================================================================================
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import read_image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vprtempo.src.dataset import ProcessImage  # noqa: E402

DATA_DIR = REPO_ROOT / "IDEA1-covstdp" / "phase1" / "configs"
OUT = REPO_ROOT / "IDEA1-covstdp" / "results" / "input_histogram_patchnorm.png"
N_IMAGES = 50
STRIDE = 200  # 每隔 200 张取一张，覆盖不同路段


def main():
    import json
    with open(DATA_DIR / "local_override.json") as f:
        data_dir = Path(json.load(f)["data_dir"])

    img_dir = data_dir / "spring"
    files = sorted(img_dir.glob("*.png"))[::STRIDE][:N_IMAGES]
    assert files, f"在 {img_dir} 下没有找到图像"

    proc_on = ProcessImage([28, 28], 7, patch_norm=True)
    proc_off = ProcessImage([28, 28], 7, patch_norm=False)

    vals_on, vals_off = [], []
    for f in files:
        img = read_image(str(f))  # [3,H,W] uint8
        vals_on.append(proc_on(img.clone()).numpy().ravel())
        vals_off.append(proc_off(img.clone()).numpy().ravel())
    vals_on = np.concatenate(vals_on)
    vals_off = np.concatenate(vals_off)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(vals_off, bins=bins, alpha=0.6, density=True,
            label=f"patch_norm=off (mean={vals_off.mean():.3f})")
    ax.hist(vals_on, bins=bins, alpha=0.6, density=True,
            label=f"patch_norm=on (mean={vals_on.mean():.3f})")
    ax.axvline(0.5, color="red", ls="--", lw=1, label="0.5（on 的平坦背景）")
    ax.set_xlabel("spike amplitude (pixel/255)")
    ax.set_ylabel("density")
    ax.set_title(f"Input spike amplitude histogram, {len(files)} Nordland spring images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"已保存 {OUT}")
    print(f"on : mean={vals_on.mean():.3f}  P(x≈0.5)={np.mean(np.abs(vals_on-0.5)<0.02):.3f}")
    print(f"off: mean={vals_off.mean():.3f}  P(x<0.05)={np.mean(vals_off<0.05):.3f}")


if __name__ == "__main__":
    main()
