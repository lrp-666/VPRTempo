#!/usr/bin/env python3
# ================================================================================
# make_table1.py — IDEA1 S3.2 主表（Table 1）汇总脚本（fork G）
#
# 读取 results/t1_<variant>_<scale>/seed_<s>/ 下的正式主表结果：
#   - 轨 A：<exp>__seed<s>__eval.json              （经 output layer 的 Recall@K）
#   - 轨 B：<exp>__seed<s>__trackB_{conv,feature_layer}.json（raw feature retrieval）
# 汇总 3 seeds 的 mean±std，输出：
#   - results/table1_main.md   表 1a（500 地）/ 表 1b（3300 地）/ 表 1c（关键对比
#     差值，按 PLAN.md §6 Gate 判据排版）+ 训练墙钟附注 + 完整性/异常检查
#   - results/table1_main.json 机器可读全量（每格逐 seed 值 + 聚合 + 差值）
#
# 纪律：只做汇总，不做 Gate 判定解读（判定回母会话）。数字一律以 JSON 为准。
# 训练墙钟从 results/table1_logs/ 或 results/formal_logs/ 的 <exp>__seed<s>.log 的
# "[run_exp] train 完成，墙钟 Xs" 行解析（JSON 中无训练墙钟字段）。
#
# 用法：pixi run python IDEA1-covstdp/experiments/make_table1.py
# ================================================================================

import json
import math
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "IDEA1-covstdp" / "results"
LOG_DIRS = [RESULTS / "table1_logs", RESULTS / "formal_logs", RESULTS / "zoo_logs"]

VARIANTS = ["b0", "b1", "b1itp", "b2", "b2bcm", "b5", "b6a", "freesign",
            "b5bcm", "dct", "loggabor"]
VARIANT_LABEL = {
    "b0": "B0（基线，无前端）",
    "b1": "B1（随机 Conv 冻结）",
    "b1itp": "B1+ITP",
    "b2": "B2（Conv-STDP 主组合）",
    "b2bcm": "B2+BCM",
    "b5": "B5（手工 Gabor 冻结+ITP）",
    "b6a": "B6a（Gabor 初始化+STDP）",
    "freesign": "freesign（放开符号钳制）",
    "b5bcm": "B5+BCM（冻结 Gabor + BCM 阈值，探索批升档）",
    "dct": "DCT 基（频率无方向，探索批升档）",
    "loggabor": "Log-Gabor（探索批升档）",
}
SCALES = [500, 3300]
SEEDS = [0, 1, 2]

WALL_RE = re.compile(r"\[run_exp\] train 完成，墙钟 ([\d.]+)s")


def exp_id(variant: str, scale: int) -> str:
    return f"t1_{variant}_{scale}"


def trackb_suffix(variant: str) -> str:
    return "feature_layer" if variant == "b0" else "conv"


def load_cell(variant: str, scale: int, seed: int) -> dict:
    """读一格（variant × scale × seed）的双轨 JSON + 训练墙钟。"""
    exp = exp_id(variant, scale)
    d = RESULTS / exp / f"seed_{seed}"
    cell = {"exp_id": exp, "variant": variant, "scale": scale, "seed": seed,
            "missing": [], "anomalies": []}

    eval_f = d / f"{exp}__seed{seed}__eval.json"
    tb_f = d / f"{exp}__seed{seed}__trackB_{trackb_suffix(variant)}.json"
    npy_f = d / f"{exp}__seed{seed}__S_trackB_{trackb_suffix(variant)}.npy"
    log_f = d / "logfile_eval.log"
    for f in (eval_f, tb_f, npy_f, log_f):
        if not f.exists():
            cell["missing"].append(str(f.relative_to(RESULTS)))

    if eval_f.exists():
        e = json.loads(eval_f.read_text())
        cell["trackA"] = {
            "recallAtK": {int(k): v for k, v in e["recallAtK"].items()},
            "recallAt100precision": e["recallAt100precision"],
            "precisionAt100recall": e["precisionAt100recall"],
            "wall_time_s": e["wall_time_s"],
        }
    if tb_f.exists():
        t = json.loads(tb_f.read_text())
        cell["trackB"] = {
            "recallAtK": {int(k): v for k, v in t["recallAtK"].items()},
            "recallAt100precision": t["recallAt100precision"],
            "precisionAt100recall": t["precisionAt100recall"],
            "wall_time_s": t["wall_time_s"],
        }

    # 训练墙钟：跑批 cell 日志（JSON 无此字段）；table1 批与 formal 批两个日志目录
    cell["train_wall_s"] = None
    for log_dir in LOG_DIRS:
        cell_log = log_dir / f"{exp}__seed{seed}.log"
        if cell_log.exists():
            m = WALL_RE.search(cell_log.read_text(errors="replace"))
            if m:
                cell["train_wall_s"] = float(m.group(1))
            break

    # 异常扫描（预注册阈值：R@1 < 0.1 或 == 1.0 判可疑，标注待母会话复核）
    for track in ("trackA", "trackB"):
        if track in cell:
            r1 = cell[track]["recallAtK"][1]
            if r1 < 0.1 or r1 == 1.0:
                cell["anomalies"].append(
                    f"{track} R@1={r1} 触发可疑值阈值（<0.1 或 =1.0），待母会话复核")
    return cell


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    """样本标准差（ddof=1）；n<2 时返回 0。"""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def agg(cells, track, metric):
    """聚合 3 seeds 的 mean/std。metric: ('recallAtK', 1) 或单值字段名。"""
    xs = []
    for c in cells:
        if track not in c:
            return None
        if isinstance(metric, tuple):
            xs.append(c[track][metric[0]][metric[1]])
        else:
            xs.append(c[track][metric])
    return {"mean": mean(xs), "std": std(xs), "values": xs, "n": len(xs)}


def fmt(a, digits=3):
    if a is None:
        return "—"
    return f"{a['mean']:.{digits}f} ± {a['std']:.{digits}f}"


def diff(a_cells, b_cells, track, metric):
    """a − b 的差值（mean 差 ± 联合 std = sqrt(std_a²+std_b²)），并给配对逐 seed 差。"""
    xa = [c[track][metric[0]][metric[1]] if isinstance(metric, tuple)
          else c[track][metric] for c in a_cells]
    xb = [c[track][metric[0]][metric[1]] if isinstance(metric, tuple)
          else c[track][metric] for c in b_cells]
    paired = [x - y for x, y in zip(xa, xb)]
    return {
        "diff_mean": mean(xa) - mean(xb),
        "joint_std": math.sqrt(std(xa) ** 2 + std(xb) ** 2),
        "paired_diff_mean": mean(paired),
        "paired_diff_std": std(paired),
        "a_mean": mean(xa), "b_mean": mean(xb),
    }


def main():
    cells = {}  # (variant, scale) -> [cell seed0..2]
    problems = []
    for v in VARIANTS:
        for sc in SCALES:
            cs = [load_cell(v, sc, s) for s in SEEDS]
            cells[(v, sc)] = cs
            for c in cs:
                for m in c["missing"]:
                    problems.append(f"缺文件：{m}")
                for a in c["anomalies"]:
                    problems.append(f"异常：{c['exp_id']} seed{c['seed']}：{a}")

    md = []
    md.append("# Table 1 — IDEA1 Conv-STDP 主表（正式档，3 seeds × 双规模 × 双轨）\n")
    md.append("> 由 `experiments/make_table1.py` 自动生成；数字一律以各格 JSON 为准。\n"
              "> 轨 A = 完整系统（经 output layer，run_inference）；轨 B = raw feature retrieval\n"
              "> （前端输出 flatten + cosine 最近邻）。mean ± std 为 3 seeds（0/1/2）样本统计。\n"
              "> 后三行（B5+BCM / DCT 基 / Log-Gabor）为探索批升档变体，seed0 复用迭代/确认档\n"
              "> （exp_id 与配置逐键核对一致），seed1/2 由 run_zoo_formal.sh 补齐。\n"
              "> 本表只做汇总，Gate 判定解读回母会话（判据见 PLAN.md §6）。\n")

    # ---- 完整性 / 异常 ----
    md.append("\n## 完整性与异常检查\n")
    n_cells = len(VARIANTS) * len(SCALES) * len(SEEDS)
    if not problems:
        md.append(f"- {n_cells} 格（{len(VARIANTS)} 变体 × {len(SCALES)} 规模 × {len(SEEDS)} seeds）轨 A + 轨 B JSON、相似度矩阵 .npy、"
                  f"logfile 全部齐全（共扫描 {n_cells} 格）。")
        md.append("- 未发现 R@1 < 0.1 或 R@1 = 1.0 的可疑值。")
    else:
        for p in problems:
            md.append(f"- ⚠️ {p}")
    # 软异常提示：跨 seed 离散度大的格（std > 0.1）仅标注，不判读
    loose = []
    for (v, sc), cs in cells.items():
        for track, metric, name in (("trackA", ("recallAtK", 1), "轨A R@1"),
                                    ("trackB", ("recallAtK", 1), "轨B R@1"),
                                    ("trackB", "recallAt100precision", "轨B R@100%P")):
            a = agg(cs, track, metric)
            if a and a["std"] > 0.1:
                loose.append(f"{exp_id(v, sc)} {name} std={a['std']:.3f}"
                             f"（逐 seed：{['%.3f' % x for x in a['values']]}）")
    if loose:
        md.append("- 跨 seed 离散度较大（std > 0.1，非硬异常，列此备查）：")
        for l in loose:
            md.append(f"  - {l}")

    # ---- 表 1a / 1b ----
    for tag, scale in (("1a", 500), ("1b", 3300)):
        md.append(f"\n## 表 {tag}：{scale} 地（{'迭代规模' if scale == 500 else '会议规模，Gate 2 判定规模'}）\n")
        md.append("| 变体 | 轨A R@1 | 轨B R@1 | 轨B R@100%P | 轨B P@100%R |")
        md.append("|---|---|---|---|---|")
        for v in VARIANTS:
            cs = cells[(v, scale)]
            md.append("| {} | {} | {} | {} | {} |".format(
                VARIANT_LABEL[v],
                fmt(agg(cs, "trackA", ("recallAtK", 1))),
                fmt(agg(cs, "trackB", ("recallAtK", 1))),
                fmt(agg(cs, "trackB", "recallAt100precision")),
                fmt(agg(cs, "trackB", "precisionAt100recall")),
            ))

    # ---- 表 1c：关键对比差值（按 PLAN §6 Gate 排版）----
    md.append("\n## 表 1c：关键对比差值（mean 差 ± 联合 std = √(std_a²+std_b²)，n=3 seeds）\n")
    md.append("> 仅列差值与离散度，Gate 通过/不通过的判定回母会话。"
              "括号内为该对比对应的 PLAN §6 Gate 判据原文要点。\n")
    md.append("| 对比 | 轨 / 指标 | 对应 Gate | 500 地 | 3300 地 |")
    md.append("|---|---|---|---|---|")

    DIFFS = [
        # (label, a, b, track, gate_note)
        ("B2 − B1", "b2", "b1", "trackB",
         "Gate 1：轨B B2>B1 且差值 > 3-seed 联合 std"),
        ("B2+BCM − B2", "b2bcm", "b2", "trackB", "—（BCM 规则对照，非 Gate 判据）"),
        ("B5 − B2", "b5", "b2", "trackB",
         "Gate 1.5：轨B B2 ≥ B5 − 1×std（此处以 B5−B2 呈现，≤std 即满足）"),
        ("freesign − B2", "freesign", "b2", "trackB", "—（符号约束消融升主表行，非 Gate 判据）"),
        ("B6a − B5", "b6a", "b5", "trackB",
         "Gate 1.5（v5 主判据）：B6 vs B5 判定学习在好先验上有无增量"),
        ("B2 − B0", "b2", "b0", "trackA",
         "Gate 2：轨A B2>B0（判定在 3300 地会议规模）"),
        ("B5 − B0", "b5", "b0", "trackA", "—（参照 PLAN §0.4 B5 双轨超 B0 的迭代档发现）"),
        ("freesign − B0", "freesign", "b0", "trackA", "—"),
        ("B2+BCM − B0", "b2bcm", "b0", "trackA", "—"),
        ("B5+BCM − B5", "b5bcm", "b5", "trackB",
         "—（探索批升档：BCM 阈值替代 ITP 对冻结 Gabor 有无增量，判定回母会话）"),
        ("DCT 基 − B5", "dct", "b5", "trackB",
         "—（探索批升档：频率覆盖 vs 方向选择性，判定回母会话）"),
        ("Log-Gabor − B5", "loggabor", "b5", "trackB", "—（探索批升档）"),
    ]
    diff_records = []
    for label, a, b, track, gate in DIFFS:
        row = {"label": label, "a": a, "b": b, "track": track,
               "metric": "R@1", "gate": gate, "per_scale": {}}
        cols = []
        for sc in SCALES:
            d = diff(cells[(a, sc)], cells[(b, sc)], track, ("recallAtK", 1))
            row["per_scale"][str(sc)] = d
            cols.append(f"{d['diff_mean']:+.3f} ± {d['joint_std']:.3f}")
        diff_records.append(row)
        md.append(f"| {label} | {'轨A' if track == 'trackA' else '轨B'} R@1 | "
                  f"{gate} | {cols[0]} | {cols[1]} |")

    # ---- 附注：训练墙钟 ----
    md.append("\n## 附注：训练墙钟（每格 mean ± std，秒，3 seeds）\n")
    md.append("> 来源：`results/table1_logs/`、`results/formal_logs/`、`results/zoo_logs/` 下 `<exp>__seed<s>.log` 中 run_exp 的"
              "「train 完成，墙钟 Xs」行（轨 A 训练阶段，双 RTX 卡混跑，仅供量级参考）。"
              "轨 A 评估与轨 B 检索墙钟在各格 JSON 的 wall_time_s 字段，见 table1_main.json。\n")
    md.append("| 变体 | 500 地 train | 3300 地 train |")
    md.append("|---|---|---|")
    wall_records = {}
    for v in VARIANTS:
        cols = []
        for sc in SCALES:
            ws = [c["train_wall_s"] for c in cells[(v, sc)] if c["train_wall_s"] is not None]
            if ws:
                wall_records[f"{v}_{sc}"] = {"mean": mean(ws), "std": std(ws), "values": ws}
                cols.append(f"{mean(ws):.0f} ± {std(ws):.0f}")
            else:
                cols.append("n/a（日志缺失）")
        md.append(f"| {VARIANT_LABEL[v]} | {cols[0]} | {cols[1]} |")

    out_md = RESULTS / "table1_main.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- 机器可读全量 ----
    full = {
        "generated_by": "IDEA1-covstdp/experiments/make_table1.py",
        "protocol": "11 variants (8 main + 3 zoo-promoted) × 2 scales × 3 seeds × dual-track (A=output layer, B=raw feature retrieval)",
        "variants": VARIANT_LABEL,
        "cells": {f"{v}_{sc}": cells[(v, sc)] for v in VARIANTS for sc in SCALES},
        "aggregates": {
            f"{v}_{sc}": {
                "trackA_R@1": agg(cells[(v, sc)], "trackA", ("recallAtK", 1)),
                "trackB_R@1": agg(cells[(v, sc)], "trackB", ("recallAtK", 1)),
                "trackB_R@100P": agg(cells[(v, sc)], "trackB", "recallAt100precision"),
                "trackB_P@100R": agg(cells[(v, sc)], "trackB", "precisionAt100recall"),
                "train_wall_s": wall_records.get(f"{v}_{sc}"),
            } for v in VARIANTS for sc in SCALES
        },
        "diffs": diff_records,
        "problems": problems,
    }
    out_json = RESULTS / "table1_main.json"
    out_json.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[make_table1] 写出 {out_md.relative_to(REPO_ROOT)} 与 {out_json.relative_to(REPO_ROOT)}")
    if problems:
        print(f"[make_table1] ⚠️ {len(problems)} 条完整性/异常问题，见 md 报告")
        return 1
    print(f"[make_table1] 完整性检查通过：{n_cells}/{n_cells} 格双轨齐全，无硬异常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
