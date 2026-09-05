#!/usr/bin/env bash
# ================================================================================
# run_table1.sh — IDEA1 S3.2 主表（Table 1）批量跑批脚本
#
# 矩阵：8 变体（b0 b1 b1itp b2 b2bcm b5 b6a freesign）× 2 规模（500 / 3300 地）
#       × 3 seeds（0/1/2）= 48 格；每格 = 轨 A（run_exp.py train+eval）+ 轨 B
#       （eval_retrieval.py）。配置在 IDEA1-covstdp/phase3/configs/table1/，
#       exp_id 一律 t1_<variant>_<scale>（与 phase2 迭代档同名旧结果目录隔离）。
#
# 用法（在 GPU 工作站上，tmux 里跑）：
#   tmux new -s table1
#   cd /home/ps/workspace/VPRTempo
#   export PATH="$HOME/.pixi/bin:$PATH"   # 工作站非交互 shell 不带 pixi，必须先加
#   bash IDEA1-covstdp/experiments/run_table1.sh            # 双卡（GPU 0/1）
#   NUM_GPUS=1 bash IDEA1-covstdp/experiments/run_table1.sh # 单卡调试
#   PIXI_ENV="" bash ...                                    # 默认 pixi 环境（本机 CPU）
#   # Ctrl+B D 挂起；重进 tmux attach -t table1
#
# 特性：
#   - 双卡队列：每张卡同时至多一格，格与格启动间 sleep 2 秒（防 logger 秒级
#     时间戳撞 output 目录）；
#   - 断点续跑：轨 A 结果 JSON 与轨 B 结果 JSON 都存在的格子跳过；只有轨 A
#     时只补轨 B（不重训）；重复执行本脚本即续跑；
#   - 每格完成/失败写一行到 IDEA1-covstdp/results/table1_progress.log；
#   - 每格完整 stdout/stderr 存 IDEA1-covstdp/results/table1_logs/<exp>__seed<n>.log；
#   - 失败的格子不标完成，下次运行自动重试（模型 .pth 由 run_exp 覆盖重训）。
#
# 纪律：本脚本只负责执行；何时点火由母会话审定。
# ================================================================================
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CFG_DIR="IDEA1-covstdp/phase3/configs/table1"
RESULTS="IDEA1-covstdp/results"
PROGRESS_LOG="$RESULTS/table1_progress.log"
CELL_LOG_DIR="$RESULTS/table1_logs"
mkdir -p "$CELL_LOG_DIR"

VARIANTS=(b0 b1 b1itp b2 b2bcm b5 b6a freesign)
SCALES=(500 3300)
SEEDS=(0 1 2)

NUM_GPUS=${NUM_GPUS:-2}        # 双卡队列；单卡设 NUM_GPUS=1
PIXI_ENV=${PIXI_ENV:-cuda}     # 工作站 cuda 环境；本机 CPU 调试设 PIXI_ENV=""
# PY_CMD：测试注入口（调度逻辑干跑验证），正常跑批不要设置
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

trackb_suffix() {  # B0 无 conv 前端 → 轨 B 特征点是 feature_layer
    if [ "$1" = "b0" ]; then echo "feature_layer"; else echo "conv"; fi
}

tracka_done() {  # tracka_done <exp> <seed>
    [ -f "$RESULTS/$1/seed_$2/$1__seed$2__eval.json" ]
}

trackb_done() {  # trackb_done <variant> <exp> <seed>
    [ -f "$RESULTS/$2/seed_$3/$2__seed$3__trackB_$(trackb_suffix "$1").json" ]
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
                "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                    "$CFG_DIR/${v}_500.json" --train --eval --seed "$seed" || exit 1
            else
                "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                    "$CFG_DIR/${v}_3300.json" --train --seed "$seed" || exit 1
                sleep 2   # train/eval 两进程间同样留 logger 时间戳间隔
                "${PY[@]}" IDEA1-covstdp/experiments/run_exp.py \
                    "$CFG_DIR/${v}_3300_eval.json" --eval --seed "$seed" || exit 1
            fi
        else
            echo "[run_table1] 轨 A 已存在，跳过训练+评估"
        fi
        if ! trackb_done "$v" "$exp" "$seed"; then
            local eval_cfg="$CFG_DIR/${v}_${scale}.json"
            [ "$scale" = "3300" ] && eval_cfg="$CFG_DIR/${v}_3300_eval.json"
            "${PY[@]}" IDEA1-covstdp/experiments/eval_retrieval.py \
                "$eval_cfg" --seed "$seed" || exit 1
        else
            echo "[run_table1] 轨 B 已存在，跳过"
        fi
    } > "$cell_log" 2>&1
    local rc=$?
    local wall=$((SECONDS - t0))
    if [ $rc -eq 0 ] && tracka_done "$exp" "$seed" && trackb_done "$v" "$exp" "$seed"; then
        log_progress DONE "$exp" "$seed" "$gpu" "$wall"
    else
        log_progress "FAIL(rc=$rc)" "$exp" "$seed" "$gpu" "$wall"
    fi
    return $rc
}

# ---------------- 双卡调度队列 ----------------
# 每格一个后台子进程绑定一张卡；轮询空位，启动前 sleep 2（防秒级 logger 撞目录）。
declare -A SLOT_PID=()   # gpu -> pid
declare -A SLOT_CELL=()  # gpu -> "exp seed"

pending=()
for v in "${VARIANTS[@]}"; do
    for scale in "${SCALES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            exp="t1_${v}_${scale}"
            if tracka_done "$exp" "$seed" && trackb_done "$v" "$exp" "$seed"; then
                echo "[run_table1] SKIP（已完成）$exp seed$seed"
            else
                pending+=("$v $scale $seed")
            fi
        done
    done
done
total=${#pending[@]}
echo "[run_table1] 待跑 $total / 48 格，GPU 数 $NUM_GPUS，进度日志 $PROGRESS_LOG"
log_progress "BATCH_START" "remaining=$total/48" "-" "-" 0

idx=0
while [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ]; do
    # 回收已完成槽位
    for gpu in "${!SLOT_PID[@]}"; do
        if ! kill -0 "${SLOT_PID[$gpu]}" 2>/dev/null; then
            wait "${SLOT_PID[$gpu]}" 2>/dev/null
            echo "[run_table1] 槽位 gpu$gpu 完成：${SLOT_CELL[$gpu]}"
            unset "SLOT_PID[$gpu]" "SLOT_CELL[$gpu]"
        fi
    done
    # 填空位
    for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
        [ $idx -ge $total ] && break
        [ -n "${SLOT_PID[$gpu]:-}" ] && continue
        read -r v scale seed <<< "${pending[$idx]}"
        sleep 2   # 防 logger 秒级时间戳撞目录（每格启动间）
        run_cell "$v" "$scale" "$seed" "$gpu" &
        SLOT_PID[$gpu]=$!
        SLOT_CELL[$gpu]="t1_${v}_${scale} seed$seed"
        echo "[run_table1] 启动 $((idx + 1))/$total：t1_${v}_${scale} seed$seed → gpu$gpu"
        idx=$((idx + 1))
    done
    # 还有待跑或还有在跑 → 稍等再轮询
    if [ $idx -lt $total ] || [ ${#SLOT_PID[@]} -gt 0 ]; then
        sleep 10
    fi
done

done_n=$(grep -c "| DONE |" "$PROGRESS_LOG" 2>/dev/null || echo 0)
fail_n=$(grep -c "| FAIL" "$PROGRESS_LOG" 2>/dev/null || echo 0)
log_progress "BATCH_END" "done_cells=$done_n fail_cells=$fail_n" "-" "-" 0
echo "[run_table1] 全部格子调度结束：累计 DONE=$done_n FAIL=$fail_n（见 $PROGRESS_LOG）"
