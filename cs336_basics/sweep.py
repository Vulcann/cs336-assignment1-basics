"""LR sweep driver:逐个起 train.py 子进程,结束后汇总结果并画对比图。

用法:  python -m cs336_basics.sweep
扫别的超参:改 GRID 即可,run_name 由 train 侧自动派生(或在此显式指定)。
"""

import json
import subprocess
import sys
from pathlib import Path

from .data import plot  # 复用已有的画图函数

# ---- 按需修改的部分 --------------------------------------------------------
TRAIN_MODULE = "cs336_basics.train"  # python -m 的模块路径,按你的包名改
BASE_CONFIG = "exp/smoke.json"  # 公共基线(小实验配置)
GRID = [  # 每个元素 = 一个 run 的 --set 覆盖表
    {"schedule.max_learning_rate": lr, "schedule.min_learning_rate": lr * 0.1} for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
]
# ---------------------------------------------------------------------------


def run_one(overrides: dict) -> tuple[str, bool]:
    """起一个训练子进程。返回 (描述, 是否正常结束)。"""
    sets = [f"{k}={v}" for k, v in overrides.items()]
    cmd = [sys.executable, "-m", TRAIN_MODULE, "--config", BASE_CONFIG, "--set", *sets]
    print(f"\n=== {' '.join(sets)} ===")
    proc = subprocess.run(cmd)  # 阻塞直到该 run 结束;输出直通终端
    ok = proc.returncode == 0
    if not ok:  # 发散到 NaN 崩了也别中断整个 sweep
        print(f"!!! run failed (returncode {proc.returncode}), 继续下一个")
    return " ".join(sets), ok


def summarize(log_paths: list[str]) -> None:
    """读各 run 的 jsonl,打印最终/最优 loss 汇总表。"""
    print(f"\n{'run':<40} {'final':>8} {'best':>8} {'steps':>7}")
    rows = []
    for path in log_paths:
        recs = [json.loads(l) for l in Path(path).read_text().splitlines()]
        losses = [(r["step"], r["train_loss"]) for r in recs if r.get("type") == "metrics" and "train_loss" in r]
        if not losses:
            continue
        final_step, final_loss = losses[-1]
        best_loss = min(l for _, l in losses)
        rows.append((Path(path).stem, final_loss, best_loss, final_step))
    for name, final_loss, best_loss, steps in sorted(rows, key=lambda r: r[2]):
        print(f"{name:<40} {final_loss:>8.4f} {best_loss:>8.4f} {steps:>7}")


if __name__ == "__main__":
    results = [run_one(ov) for ov in GRID]

    done = sum(ok for _, ok in results)
    print(f"\nsweep 完成: {done}/{len(GRID)} 个 run 正常结束")

    logs = sorted(str(p) for p in Path("logs").glob("lr*.jsonl"))
    summarize(logs)
    plot(logs, out="lr_sweep.png")
    print("曲线图: lr_sweep.png")
