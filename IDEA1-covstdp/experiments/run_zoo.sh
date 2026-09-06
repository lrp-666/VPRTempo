#!/usr/bin/env bash
# ================================================================================
# run_zoo.sh — IDEA1 S33 下午批：手工特征动物园 + BCM 变体 + 消融补格
#
# 矩阵：10 格（dog loggabor dct fullbcm b5bcm b6abcm b2block3 b5block3 b5global
#       b5none）× 500 地迭代档 × 单 seed（0）；每格 = 轨 A（run_exp.py train+eval）
#       + 轨 B（eval_retrieval.py）。配置在 IDEA1-covstdp/phase3/configs/zoo/。
#       3300 地确认档不在本脚本排——迭代档出来后由子会话判断排哪些并标注理由。
#
# 用法（GPU 工作站，tmux 里跑）：
#   tmux new -s zoo
#   cd /home/ps/workspace/VPRTempo
#   export PATH="$HOME/.pixi/bin:$PATH"   # 工作站非交互 shell 不带 pixi，必须先加
#   bash IDEA1-covstdp/experiments/run_zoo.sh              # 默认双槽位都在 GPU1
#   GPU_IDS="0 1" bash IDEA1-covstdp/experiments/run_zoo.sh # 双卡各一槽
#   PIXI_ENV="" bash ...                                   # 默认 pixi 环境（本机 CPU）
#
# 特性（同 run_table1.sh）：槽位队列、格间 sleep 2 防 logger 秒级撞目录、
# 断点续跑（轨 A/轨 B JSON 齐则跳）、逐格日志、失败格自动重试。
# ================================================================================
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CFG_DIR="IDEA1-covstdp/phase3/configs/zoo"
RESULTS="IDEA1-covstdp/results"
PROGRESS_LOG="$RESULTS/zoo_progress.log"
CELL_LOG_DIR="$RESULTS/zoo_logs"
mkdir -p "$CELL_LOG_DIR"

VARIANTS=(dog loggabor dct fullbcm b5bcm b6abcm b2block3 b5block3 b5global b5none)
SEEDS=(0)

GPU_IDS=(${GPU_IDS:-"1 1"})    # 槽位→GPU 映射；默认两个槽位都绑 1 号卡（0 号有他人容器）
PIXI_ENV=${PIXI_ENV:-cuda}     # 工作站 cuda 环境；本机 CPU 调试设 PIXI_ENV=""
if [ -n "${PY_CMD:-}" ]; then
    PY=($PY_CMD)
elif [ -n "$PIXI_ENV" ]; then
    PY=(pixi run --environment "$PIXI_ENV" python)
else
    PY=(pixi run python)
fi

log_progress() {  # log_progress <status> <exp> <seed> <gpu> <wall_s>
    echo "$(date '+%F %T') | $1 | $2 | seed$3 | gpu$4 | wall=${5}s" >> "$PROGRESS_LOG"
}

tracka_done() {  # tracka_done <exp> <seed>
    [ -f "$RESULTS/$1/seed_$2/$1__seed$2__eval.json" ]
}

trackb_done() {  # trackb_done <exp> <seed>（本批全部有 conv 前端，特征点恒为 conv）
    [ -f "$RESULTS/$1/seed_$2/$1__seed$2__trackB_conv.json" ]
}

run_cell() {  # run_cell <variant> <seed> <gpu>
    local v=$1 seed=$2 gpu=$3
    local exp="t1_${v}_500"
    local cell_log="$CELL_LOG_DIR/${exp}__seed${seed}.log"
    local t0=$SECONDS
    {
        echo "===== $(date '+%F %T') cell ${exp} seed${seed} on GPU ${gpu} ====="
        export CUDA_VISIBLE_DEVICES=$gpu
        export MPLBACKEND=Agg
        if ! tracka_done "$exp" "$seed"; then
            "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                "$CFG_DIR/${v}_500.json" --train --eval --seed "$seed" || exit 1
        else
            echo "[run_zoo] 轨 A 已存在，跳过训练+评估"
        fi
        if ! trackb_done "$exp" "$seed"; then
            "${PY[@]}" IDEA1-covstdp/experiments/eval_retrieval.py \
                "$CFG_DIR/${v}_500.json" --seed "$seed" || exit 1
        else
            echo "[run_zoo] 轨 B 已存在，跳过"
        fi
    } > "$cell_log" 2>&1
    local rc=$?
    local wall=$((SECONDS - t0))
    if [ $rc -eq 0 ] && tracka_done "$exp" "$seed" && trackb_done "$exp" "$seed"; then
        log_progress DONE "$exp" "$seed" "$gpu" "$wall"
    else
        log_progress "FAIL(rc=$rc)" "$exp" "$seed" "$gpu" "$wall"
    fi
    return $rc
}

# ---------------- 槽位调度队列 ----------------
declare -A SLOT_PID=()   # slot -> pid
declare -A SLOT_CELL=()  # slot -> "exp seed"
declare -A SLOT_GPU=()   # slot -> gpu

pending=()
for v in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        exp="t1_${v}_500"
        if tracka_done "$exp" "$seed" && trackb_done "$exp" "$seed"; then
            echo "[run_zoo] SKIP（已完成）$exp seed$seed"
        else
            pending+=("$v $seed")
        fi
    done
done
total=${#pending[@]}
nslots=${#GPU_IDS[@]}
echo "[run_zoo] 待跑 $total / $((${#VARIANTS[@]} * ${#SEEDS[@]})) 格，槽位 ${GPU_IDS[*]}，进度日志 $PROGRESS_LOG"
log_progress "BATCH_START" "remaining=$total" "-" "-" 0

idx=0
while [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ]; do
    for slot in "${!SLOT_PID[@]}"; do
        if ! kill -0 "${SLOT_PID[$slot]}" 2>/dev/null; then
            wait "${SLOT_PID[$slot]}" 2>/dev/null
            echo "[run_zoo] 槽位 $slot（gpu${SLOT_GPU[$slot]}）完成：${SLOT_CELL[$slot]}"
            unset "SLOT_PID[$slot]" "SLOT_CELL[$slot]" "SLOT_GPU[$slot]"
        fi
    done
    for ((slot = 0; slot < nslots; slot++)); do
        [ $idx -ge $total ] && break
        [ -n "${SLOT_PID[$slot]:-}" ] && continue
        read -r v seed <<< "${pending[$idx]}"
        sleep 2   # 防 logger 秒级时间戳撞目录（每格启动间）
        run_cell "$v" "$seed" "${GPU_IDS[$slot]}" &
        SLOT_PID[$slot]=$!
        SLOT_CELL[$slot]="t1_${v}_500 seed$seed"
        SLOT_GPU[$slot]="${GPU_IDS[$slot]}"
        echo "[run_zoo] 启动 $((idx + 1))/$total：t1_${v}_500 seed$seed → gpu${GPU_IDS[$slot]}（槽位 $slot）"
        idx=$((idx + 1))
    done
    if [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ]; then
        sleep 10
    fi
done

done_n=$(grep -c "| DONE |" "$PROGRESS_LOG" 2>/dev/null || echo 0)
fail_n=$(grep -c "| FAIL" "$PROGRESS_LOG" 2>/dev/null || echo 0)
log_progress "BATCH_END" "done_cells=$done_n fail_cells=$fail_n" "-" "-" 0
echo "[run_zoo] 全部格子调度结束：累计 DONE=$done_n FAIL=$fail_n（见 $PROGRESS_LOG）"
