"""Aggregate track-A P@100%R (and R@100%P) from table1_main.json.

Purpose: LoCS-Net protocol comparison is in P@100%R of the FULL system
(their classifier output). Our headline 0.802 was track-B (descriptor-level
cosine retrieval). This script fills the track-A P@100%R column for all
main-table variants so the external comparison is apples-to-apples.

Input : IDEA1-covstdp/results/table1_main.json
Output: IDEA1-covstdp/results/table1_p100r_trackA.md
"""
import json
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
ORDER = ["b0", "b1", "b1itp", "b2", "b2bcm", "b5", "b5bcm", "b6a", "dct",
         "loggabor", "freesign"]
NAMES = {
    "b0": "B0", "b1": "B1", "b1itp": "B1+ITP", "b2": "B2",
    "b2bcm": "B2+BCM", "b5": "B5", "b5bcm": "B5+BCM", "b6a": "B6a",
    "dct": "DCT", "loggabor": "Log-Gabor", "freesign": "freesign",
}


def main():
    d = json.load(open(RESULTS / "table1_main.json"))
    cells = d["cells"]
    lines = []
    for scale in (500, 3300):
        lines.append(f"\n## {scale} 地（3 seeds，mean ± std）\n")
        lines.append("| 变体 | 轨A P@100%R | 轨A R@100%P | 轨B P@100%R | 轨B R@1 |")
        lines.append("|---|---|---|---|---|")
        for v in ORDER:
            key = f"{v}_{scale}"
            if key not in cells:
                continue
            rows = [c for c in cells[key] if not c.get("missing")]
            if not rows:
                continue
            def ms(track, field):
                x = [c[track][field] for c in rows]
                return f"{np.mean(x):.3f} ± {np.std(x):.3f}"
            r1b = [c["trackB"]["recallAtK"]["1"] for c in rows]
            lines.append(
                f"| {NAMES[v]} | {ms('trackA','precisionAt100recall')} | "
                f"{ms('trackA','recallAt100precision')} | "
                f"{ms('trackB','precisionAt100recall')} | "
                f"{np.mean(r1b):.3f} ± {np.std(r1b):.3f} |"
            )
    out = RESULTS / "table1_p100r_trackA.md"
    header = ("# 主表补算：轨 A 的 P@100%R（LoCS-Net 对比口径）\n\n"
              "> 来源：table1_main.json 离线聚合（无重跑）。\n"
              "> 外部锚点：LoCS-Net Nordland P@100%R = 78.6%；"
              "VPRTempo（LoCS-Net 表内）= 73.0%。\n")
    out.write_text(header + "\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
