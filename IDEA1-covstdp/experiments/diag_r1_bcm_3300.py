#!/usr/bin/env python
# ================================================================================
# diag_r1_bcm_3300.py — R1（bcm_gate）3300 地确认档的 θ_M 训练曲线诊断
#
# 复用 diag_s211_round1.replay_curves（Part B 同口径：仅 conv 前端重放，
# 模块 0 数据段，每 25 步快照），只换配置为 phase2/configs/r1_bcm_3300.json。
# 注意：重放的 RNG 消耗序列与真实训练不完全一致，曲线是同分布诊断轨迹，
# 不与存盘模型逐元素对应（与 diag_s211_round1 同一 caveats）。
#
# 用法（必须从仓库根目录运行）：
#   pixi run --environment cuda python IDEA1-covstdp/experiments/diag_r1_bcm_3300.py
# ================================================================================
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
assert Path.cwd() == REPO_ROOT, "必须从仓库根目录运行"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "diag_s211_round1", REPO_ROOT / "IDEA1-covstdp/experiments/diag_s211_round1.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

diag.CONFIGS = {"r1_bcm_3300": "IDEA1-covstdp/phase2/configs/r1_bcm_3300.json"}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
curve = diag.replay_curves("r1_bcm_3300", device)

out_dir = diag.RESULTS_DIR / "r1_bcm_3300" / "seed_0"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "r1_bcm_3300__seed0__diag_theta_m.json"
payload = {
    "experiment": "R1 bcm_gate 3300-place θ_M replay diagnostic "
                  "(module-0 data segment, conv-only replay; "
                  "replay caveat: 与存盘模型不逐元素对应)",
    "seed": diag.SEED, "snap_every": diag.SNAP_EVERY,
    "curve": curve,
}
with open(out_file, "w") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)

tm = curve["theta_m_mean"]
print(f"[diag] θ_M mean: {tm[0]:.4f} → {tm[-1]:.4f} "
      f"(min={min(tm):.4f}, max={max(tm):.4f}, n={len(tm)})")
print(f"[diag] 已写入 {out_file}")
