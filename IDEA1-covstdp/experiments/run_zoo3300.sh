#!/usr/bin/env bash
# ================================================================================
# run_zoo3300.sh — IDEA1 S33 下午批·确认档（3300 地会议规模）
#
# 排格依据（s33afternoon_batch.md）：500 地迭代档轨 B R@1 ≥ 0.90 的格子
# （dog 0.916 / loggabor 0.978 / dct 0.976 / b5bcm 0.984 / b6abcm 0.924 /
# b5block3 0.972）。fullbcm(0.856) / b2block3(0.854) 不达线；b5global 崩塌；
# b5none 与 b5 在冻结前端下逐比特等价（竞争即池化恒等），直接引用 t1_b5_3300。
#
# 用法同 run_zoo.sh（tmux 里）：bash IDEA1-covstdp/experiments/run_zoo3300.sh
# 3300 地为 train/eval 分离双文件（沿用 b5_3300 样板：train skip=0，
# eval skip=4800 / skip_db=0）。
# ================================================================================
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CFG_DIR="IDEA1-covstdp/phase3/configs/zoo"
RESULTS="IDEA1-covstdp/results"
PROGRESS_LOG="$RESULTS/zoo_progress.log"
CELL_LOG_DIR="$RESULTS/zoo_logs"
mkdir -p "$CELL_LOG_DIR"

VARIANTS=(dog loggabor dct b5bcm b6abcm b5block3)
SEEDS=(0)

GPU_IDS=(${GPU_IDS:-1 1})      # 槽位→GPU 映射；默认两个槽位都绑 1 号卡（0 号有他人容器）
PIXI_ENV=${PIXI_ENV:-cuda}
if [ -n "${PY_CMD:-}" ]; then
    PY=($PY_CMD)
elif [ -n "$PIXI_ENV" ]; then
    PY=(pixi run --environment "$PIXI_ENV" python)
else
    PY=(pixi run python)
fi

log_progress() {
    echo "$(date '+%F %T') | $1 | $2 | seed$3 | gpu$4 | wall=${5}s" >> "$PROGRESS_LOG"
}

tracka_done() { [ -f "$RESULTS/$1/seed_$2/$1__seed$2__eval.json" ]; }
trackb_done() { [ -f "$RESULTS/$1/seed_$2/$1__seed$2__trackB_conv.json" ]; }

run_cell() {  # run_cell <variant> <seed> <gpu>
    local v=$1 seed=$2 gpu=$3
    local exp="t1_${v}_3300"
    local cell_log="$CELL_LOG_DIR/${exp}__seed${seed}.log"
    local t0=$SECONDS
    {
        echo "===== $(date '+%F %T') cell ${exp} seed${seed} on GPU ${gpu} ====="
        export CUDA_VISIBLE_DEVICES=$gpu
        export MPLBACKEND=Agg
        if ! tracka_done "$exp" "$seed"; then
            "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                "$CFG_DIR/${v}_3300.json" --train --seed "$seed" || exit 1
            sleep 2   # train/eval 两进程间留 logger 时间戳间隔
            "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                "$CFG_DIR/${v}_3300_eval.json" --eval --seed "$seed" || exit 1
        else
            echo "[run_zoo3300] 轨 A 已存在，跳过训练+评估"
        fi
        if ! trackb_done "$exp" "$seed"; then
            "${PY[@]}" IDEA1-covstdp/experiments/eval_retrieval.py \
                "$CFG_DIR/${v}_3300_eval.json" --seed "$seed" || exit 1
        else
            echo "[run_zoo3300] 轨 B 已存在，跳过"
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
declare -A SLOT_PID=()
declare -A SLOT_CELL=()
declare -A SLOT_GPU=()

pending=()
for v in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        exp="t1_${v}_3300"
        if tracka_done "$exp" "$seed" && trackb_done "$exp" "$seed"; then
            echo "[run_zoo3300] SKIP（已完成）$exp seed$seed"
        else
            pending+=("$v $seed")
        fi
    done
done
total=${#pending[@]}
nslots=${#GPU_IDS[@]}
echo "[run_zoo3300] 待跑 $total / $((${#VARIANTS[@]} * ${#SEEDS[@]})) 格，槽位 ${GPU_IDS[*]}，进度日志 $PROGRESS_LOG"
log_progress "BATCH_START" "remaining=$total(3300)" "-" "-" 0

idx=0
while [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ]; do
    for slot in "${!SLOT_PID[@]}"; do
        if ! kill -0 "${SLOT_PID[$slot]}" 2>/dev/null; then
            wait "${SLOT_PID[$slot]}" 2>/dev/null
            echo "[run_zoo3300] 槽位 $slot（gpu${SLOT_GPU[$slot]}）完成：${SLOT_CELL[$slot]}"
            unset "SLOT_PID[$slot]" "SLOT_CELL[$slot]" "SLOT_GPU[$slot]"
        fi
    done
    for ((slot = 0; slot < nslots; slot++)); do
        [ $idx -ge $total ] && break
        [ -n "${SLOT_PID[$slot]:-}" ] && continue
        read -r v seed <<< "${pending[$idx]}"
        sleep 2
        run_cell "$v" "$seed" "${GPU_IDS[$slot]}" &
        SLOT_PID[$slot]=$!
        SLOT_CELL[$slot]="t1_${v}_3300 seed$seed"
        SLOT_GPU[$slot]="${GPU_IDS[$slot]}"
        echo "[run_zoo3300] 启动 $((idx + 1))/$total：t1_${v}_3300 seed$seed → gpu${GPU_IDS[$slot]}（槽位 $slot）"
        idx=$((idx + 1))
    done
    if [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ]; then
        sleep 10
    fi
done

log_progress "BATCH_END" "3300 done" "-" "-" 0
echo "[run_zoo3300] 全部格子调度结束（见 $PROGRESS_LOG）"
