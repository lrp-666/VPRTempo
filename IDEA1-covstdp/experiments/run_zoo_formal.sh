#!/usr/bin/env bash
# ================================================================================
# run_zoo_formal.sh — IDEA1 探索批三关键行正式档升级批
#
# 矩阵：3 变体（b5bcm / dct / loggabor）× seeds {0,1,2} × 双规模
#       （500 单文件 + 3300 train/eval 双文件）× 双轨（轨A run_exp train+eval
#       + 轨B eval_retrieval），共 18 格。
#       seed0 六格已有迭代档/确认档结果（exp_id 均为 t1_<variant>_<scale>，
#       与本轮一致，且结果 JSON 内嵌 config 与当前 zoo 配置逐键核对无差异），
#       断点续跑自动跳过，实际新跑 12 格。
#
# 用法（GPU 工作站，tmux 里跑）：
#   tmux new -s formal
#   cd /home/ps/workspace/VPRTempo
#   export PATH="$HOME/.pixi/bin:$PATH"   # 工作站非交互 shell 不带 pixi，必须先加
#   GPU_IDS="0 1" bash IDEA1-covstdp/experiments/run_zoo_formal.sh
#   PIXI_ENV="" bash ...                  # 本机 CPU 调试
#
# 特性（照抄 run_zoo3300.sh 结构）：槽位队列、格间 sleep 2 防 logger 秒级撞目录、
# 断点续跑（轨 A/轨 B JSON 齐则跳）、逐格日志、进度日志。
# ================================================================================
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CFG_DIR="IDEA1-covstdp/phase3/configs/zoo"
RESULTS="IDEA1-covstdp/results"
PROGRESS_LOG="$RESULTS/formal_progress.log"
CELL_LOG_DIR="$RESULTS/formal_logs"
mkdir -p "$CELL_LOG_DIR"

VARIANTS=(b5bcm dct loggabor)
SEEDS=(0 1 2)
SCALES=(500 3300)

GPU_IDS=(${GPU_IDS:-1 1})      # 槽位→GPU 映射；默认两个槽位都绑 1 号卡。
                               # 注意默认值不能加引号（"1 1" 会被当成一个元素，退化为单槽串行）
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

trackb_done() {  # trackb_done <exp> <seed>（本批三个变体全部有 conv 前端，特征点恒为 conv）
    [ -f "$RESULTS/$1/seed_$2/$1__seed$2__trackB_conv.json" ]
}

run_cell() {  # run_cell <variant> <scale> <seed> <gpu>
    local v=$1 scale=$2 seed=$3 gpu=$4
    local exp="t1_${v}_${scale}"
    local cell_log="$CELL_LOG_DIR/${exp}__seed${seed}.log"
    local t0=$SECONDS
    {
        echo "===== $(date '+%F %T') cell ${exp} seed${seed} on GPU ${gpu} ====="
        export CUDA_VISIBLE_DEVICES=$gpu
        export MPLBACKEND=Agg
        if ! tracka_done "$exp" "$seed"; then
            if [ "$scale" = "500" ]; then
                # 500 地：单文件 train+eval 一次完成
                "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                    "$CFG_DIR/${v}_500.json" --train --eval --seed "$seed" || exit 1
            else
                # 3300 地：train/eval 分离双文件（train skip=0，eval skip=4800）
                "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                    "$CFG_DIR/${v}_3300.json" --train --seed "$seed" || exit 1
                sleep 2   # train/eval 两进程间留 logger 时间戳间隔
                "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                    "$CFG_DIR/${v}_3300_eval.json" --eval --seed "$seed" || exit 1
            fi
        else
            echo "[run_zoo_formal] 轨 A 已存在，跳过训练+评估"
        fi
        if ! trackb_done "$exp" "$seed"; then
            if [ "$scale" = "500" ]; then
                "${PY[@]}" IDEA1-covstdp/experiments/eval_retrieval.py \
                    "$CFG_DIR/${v}_500.json" --seed "$seed" || exit 1
            else
                # 轨B 用 eval 配置（skip_db=0，与确认档口径一致）
                "${PY[@]}" IDEA1-covstdp/experiments/eval_retrieval.py \
                    "$CFG_DIR/${v}_3300_eval.json" --seed "$seed" || exit 1
            fi
        else
            echo "[run_zoo_formal] 轨 B 已存在，跳过"
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
for scale in "${SCALES[@]}"; do
    for v in "${VARIANTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp="t1_${v}_${scale}"
            if tracka_done "$exp" "$seed" && trackb_done "$exp" "$seed"; then
                echo "[run_zoo_formal] SKIP（已完成）$exp seed$seed"
            else
                pending+=("$v $scale $seed")
            fi
        done
    done
done
total=${#pending[@]}
nslots=${#GPU_IDS[@]}
ntotal_cells=$((${#VARIANTS[@]} * ${#SEEDS[@]} * ${#SCALES[@]}))
echo "[run_zoo_formal] 待跑 $total / $ntotal_cells 格，槽位 ${GPU_IDS[*]}，进度日志 $PROGRESS_LOG"
log_progress "BATCH_START" "remaining=$total(formal)" "-" "-" 0

idx=0
while [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ]; do
    for slot in "${!SLOT_PID[@]}"; do
        if ! kill -0 "${SLOT_PID[$slot]}" 2>/dev/null; then
            wait "${SLOT_PID[$slot]}" 2>/dev/null
            echo "[run_zoo_formal] 槽位 $slot（gpu${SLOT_GPU[$slot]}）完成：${SLOT_CELL[$slot]}"
            unset "SLOT_PID[$slot]" "SLOT_CELL[$slot]" "SLOT_GPU[$slot]"
        fi
    done
    for ((slot = 0; slot < nslots; slot++)); do
        [ $idx -ge $total ] && break
        [ -n "${SLOT_PID[$slot]:-}" ] && continue
        read -r v scale seed <<< "${pending[$idx]}"
        sleep 2   # 防 logger 秒级时间戳撞目录（每格启动间）
        run_cell "$v" "$scale" "$seed" "${GPU_IDS[$slot]}" &
        SLOT_PID[$slot]=$!
        SLOT_CELL[$slot]="t1_${v}_${scale} seed$seed"
        SLOT_GPU[$slot]="${GPU_IDS[$slot]}"
        echo "[run_zoo_formal] 启动 $((idx + 1))/$total：t1_${v}_${scale} seed$seed → gpu${GPU_IDS[$slot]}（槽位 $slot）"
        idx=$((idx + 1))
    done
    if [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ]; then
        sleep 10
    fi
done

done_n=$(grep -c "| DONE |" "$PROGRESS_LOG" 2>/dev/null || echo 0)
fail_n=$(grep -c "| FAIL" "$PROGRESS_LOG" 2>/dev/null || echo 0)
log_progress "BATCH_END" "done_cells=$done_n fail_cells=$fail_n" "-" "-" 0
echo "[run_zoo_formal] 全部格子调度结束：累计 DONE=$done_n FAIL=$fail_n（见 $PROGRESS_LOG）"
