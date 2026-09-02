#!/usr/bin/env python
# ================================================================================
# eval_retrieval.py — IDEA1 S1.4：轨 B 评测（raw feature retrieval）
#
# 功能：绕开 output layer 的 spike-forcing 读出，直接用 encoder 特征做最近邻检索：
#   1. 加载训练好的模型（与轨 A 同一个 .pth）；
#   2. 提取数据库（多季节合并，test=False 顺序）与查询图像的 encoder 特征；
#   3. L2 归一化 → cosine 相似度矩阵 S；
#   4. GT 用与轨 A 同一实现（vprtempo/src/gt.py），多季节纵向平铺；
#   5. recallAtK + recall@100%precision，落盘 results/<exp_id>/seed_<n>/。
#
# 特征提取点（PLAN.md S1.4 钉死）：
#   - B0：feature_layer 输出（clamp 后）。注：轨 A 推理是纯矩阵连乘（不减阈值不 clamp），
#     轨 B 采用 clamp 后输出作为"encoder 的标准产出"，差异已在 PLAN 中声明。
#   - 后续 B1/B2/B3/B5：conv 前端池化后 flatten 的 1152 维（阶段 2 接入）。
#
# 用法：pixi run python IDEA1-covstdp/experiments/eval_retrieval.py <config.json> --seed 0
# ================================================================================
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_exp import load_config, RESULTS_DIR  # noqa: E402


def build_inference_models(cfg):
    """镜像 main.py 推理分支（main.py:335-351）构造 VPRTempo 模型列表"""
    from vprtempo.VPRTempo import VPRTempo
    from vprtempo.src.loggers import model_logger

    args = argparse.Namespace(**cfg)
    dims = [int(x) for x in cfg["dims"].split(",")]

    # 模块数与输出维度计算（与 main.py:225-249 一致；阶段 1 只跑单模块配置）
    places = args.database_places
    num_modules = 1
    while places > args.max_module:
        places -= args.max_module
        num_modules += 1
    remainder = args.database_places % args.max_module
    if remainder != 0:
        out_dim = int((args.database_places - remainder) / (num_modules - 1))
        final_out_dim = remainder
    else:
        out_dim = int(args.database_places / num_modules)
        final_out_dim = out_dim

    logger, output_folder = model_logger()
    models = []
    final_out = None
    for mod in range(num_modules):
        model = VPRTempo(args, dims, logger, num_modules, output_folder,
                         out_dim, out_dim_remainder=final_out)
        model.eval()
        model.to(torch.device('cpu'))
        models.append(model)
        if mod == num_modules - 2:
            final_out = final_out_dim

    model_name = cfg.get("model_name") or f"{cfg['exp_id']}__seed{cfg['seed']}.pth"
    models[0].load_model(models, os.path.join('./vprtempo/models', model_name))
    return models


def extract_features(models, dataset, device):
    """逐样本提取 encoder 特征：feature_layer 线性输出 → clamp_spikes（减 thr + clamp [0,0.9]）"""
    import vprtempo.src.blitnet as bn

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    feats = []
    with torch.no_grad():
        for spikes, _ in loader:
            spikes = spikes.to(device)
            # 阶段 1 为单模块；多模块时各模块特征拼接（与轨 A 输出拼接同构）
            per_module = []
            for model in models:
                x = model.feature_layer.w(spikes)
                x = bn.clamp_spikes(x, model.feature_layer)
                per_module.append(x)
            feats.append(torch.cat(per_module, dim=1).cpu())
    return torch.cat(feats, dim=0)  # [N, feature_dim]


def main():
    ap = argparse.ArgumentParser(description="IDEA1 轨 B：raw feature retrieval 评测")
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, default=0)
    cli = ap.parse_args()

    assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"
    os.environ["PYTHONHASHSEED"] = str(cli.seed)
    cfg = load_config(cli.config, cli.seed)

    from vprtempo.src.dataset import CustomImageDataset, ProcessImage
    from vprtempo.src.gt import build_ground_truth_multiseason
    from vprtempo.src.metrics import recallAtK, createPR

    models = build_inference_models(cfg)
    model = models[0]
    patch_norm = getattr(model, 'patch_norm', 'on') == 'on'
    transform = ProcessImage(model.dims, model.patches, patch_norm=patch_norm)

    # ---- 数据库数据集：多季节合并（test=False），顺序 = [季节0 全部地点, 季节1 ...] ----
    # skip_db：数据库侧的 skip（会议口径下训练 skip=0、查询 skip=4800，两者不同）
    skip_db = cfg.get("skip_db", cfg["skip"])
    db_csvs = [os.path.join('./vprtempo/dataset', f'{cfg["dataset"]}-{d}.csv')
               for d in cfg["database_dirs"].split(',')]
    db_dataset = CustomImageDataset(
        annotations_file=db_csvs, base_dir=cfg["data_dir"],
        img_dirs=cfg["database_dirs"].split(','), transform=transform,
        filter=cfg["filter"], skip=skip_db, test=False,
        max_samples=cfg["database_places"])
    n_seasons = len(db_csvs)

    # ---- 查询数据集（test=True，与轨 A run_inference 同构）----
    q_csv = os.path.join('./vprtempo/dataset', f'{cfg["dataset"]}-{cfg["query_dir"]}.csv')
    q_dataset = CustomImageDataset(
        annotations_file=q_csv, base_dir=cfg["data_dir"],
        img_dirs=[cfg["query_dir"]], transform=transform,
        filter=cfg["filter"], skip=cfg["skip"], test=True,
        max_samples=cfg["query_places"])

    t0 = time.time()
    F_db = extract_features(models, db_dataset, model.device)
    F_q = extract_features(models, q_dataset, model.device)
    wall = time.time() - t0

    # ---- cosine 相似度矩阵 ----
    F_db = torch.nn.functional.normalize(F_db, dim=1)
    F_q = torch.nn.functional.normalize(F_q, dim=1)
    S = (F_db @ F_q.T).numpy()  # [n_seasons*db_places, query_places]

    # ---- GT：与轨 A 同一实现，多季节纵向平铺 ----
    GT = build_ground_truth_multiseason(
        cfg["database_places"], cfg["query_places"], n_seasons,
        skip=cfg["skip"], filter=cfg["filter"], tolerance=cfg["GT_tolerance"])

    # ---- 指标 ----
    Ks = [1, 5, 10, 15, 20, 25]
    recalls = {k: round(recallAtK(S, GT, K=k), 4) for k in Ks}
    P, R = createPR(S, GT, matching='single', n_thresh=100)
    P_arr, R_arr = np.array(P), np.array(R)
    r100 = float(np.max(R_arr[P_arr == 1.0])) if np.any(P_arr == 1.0) else 0.0
    p100 = float(np.max(P_arr[R_arr >= 1.0 - 1e-9])) if np.any(R_arr >= 1.0 - 1e-9) else 0.0

    print(f"[eval_retrieval] Recall@K = {recalls}")
    print(f"[eval_retrieval] R@100%P = {r100:.4f} | P@100%R = {p100:.4f}")

    # ---- 落盘 ----
    out_dir = RESULTS_DIR / cfg["exp_id"] / f"seed_{cli.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "exp_id": cfg["exp_id"], "seed": cli.seed, "phase": "trackB",
        "feature_point": "feature_layer_clamped" if cfg.get("frontend", "none") == "none"
                         else "conv_frontend_pooled",
        "feature_dim": int(F_db.shape[1]),
        "n_db_rows": int(S.shape[0]), "n_query": int(S.shape[1]),
        "n_seasons": n_seasons,
        "wall_time_s": round(wall, 2), "device": model.device,
        "config": cfg,
        "recallAtK": recalls, "recallAt100precision": round(r100, 4),
        "precisionAt100recall": round(p100, 4),
    }
    out_file = out_dir / f"{cfg['exp_id']}__seed{cli.seed}__trackB.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.save(out_dir / f"{cfg['exp_id']}__seed{cli.seed}__S_trackB.npy", S)
    print(f"[eval_retrieval] 结果已写入 {out_file}")


if __name__ == "__main__":
    main()
