#!/usr/bin/env bash
# sweep_lr.sh — 用法: bash sweep_lr.sh
set -u    # 引用未定义变量时报错;不用 -e,单个 run 崩(发散)不应中断整个 sweep

# BASE=exp/smoke.json

for lr in 1e-4 3e-4 1e-3 3e-3 1e-2; do
    min_lr=$(uv run python3 -c "print(${lr} * 0.1)")
    echo "=== lr=${lr} ==="
    # uv run python3 -m cs336_basics.train --config "$BASE" \
    uv run python3 -m cs336_basics.train \
        --set schedule.max_learning_rate=$lr schedule.min_learning_rate=$min_lr \
        || echo "!!! lr=$lr run failed, continuing"
done

# 汇总 + 画图(复用 Python 侧的函数)
uv run python3 -c "
from pathlib import Path
from cs336_basics.sweep import summarize
from cs336_basics.data import plot
logs = sorted(str(p) for p in Path('logs').glob('lr*.jsonl'))
summarize(logs); plot(logs, out='lr_sweep.png')
"