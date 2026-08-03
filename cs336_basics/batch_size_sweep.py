"""batch_size_experiment 完整驱动:找显存上限 → sqrt-scale lr → 跑网格 → 三横轴画图。

用法:
    python -m cs336_basics.batch_size_sweep probe      # 只探显存上限
    python -m cs336_basics.batch_size_sweep run        # 跑完整 sweep
    python -m cs336_basics.batch_size_sweep plot        # 只重画图
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

TRAIN = "cs336_basics.train"
BASE_CONFIG = "exp/base.json"  # 公共基线(含 max_iters/model/data 等)
ANCHOR_BS, ANCHOR_LR = 32, 1e-3  # 你 lr sweep 出的锚点:(bs=32, lr=1e-3)
BATCH_SIZES = [1, 8, 32, 64, 128, 131072]  # probe 出上限后把它追加进来
LOG_DIR = Path("logs")


def scaled_lr(bs: int) -> float:
    """平方根缩放:batch 越大梯度噪声越小,可用更大 lr。"""
    return ANCHOR_LR * (bs / ANCHOR_BS) ** 0.5


def run_train(bs: int, max_iters: int | None = None, extra: list[str] = ()) -> bool:
    lr = scaled_lr(bs)
    sets = [
        f"batch_size={bs}",
        f"schedule.max_learning_rate={lr}",
        f"schedule.min_learning_rate={lr * 0.1}",
        f"run_name=bs{bs}",  # 显式命名,画图时好按 bs 排序
    ]
    if max_iters is not None:
        sets.append(f"max_iters={max_iters}")
    # cmd = [sys.executable, "-m", TRAIN, "--config", BASE_CONFIG, "--set", *sets, *extra]
    cmd = [sys.executable, "-m", TRAIN, "--set", *sets, *extra]
    print(f"\n=== bs={bs}  lr={lr:.2e} ===")
    return subprocess.run(cmd).returncode == 0


def probe_memory_limit():
    """从 128 起翻倍,每档只跑 20 步,直到 OOM(returncode != 0)。"""
    bs = 128
    last_ok = None
    while True:
        print(f"\n--- probing bs={bs} (20 steps) ---")
        ok = run_train(bs, max_iters=20, extra=["--set", "run_name=probe", "eval_interval=999999"])
        if not ok:
            print(f"\nbs={bs} OOM(或报错)。最大可用 batch size ≈ {last_ok}")
            break
        last_ok = bs
        bs *= 2
    # 清掉 probe 的日志/ckpt,别污染正式结果
    for p in list(LOG_DIR.glob("probe.*")) + list(Path("ckpt").glob("probe.*")):
        p.unlink(missing_ok=True)
    return last_ok


def run_sweep():
    results = {bs: run_train(bs) for bs in BATCH_SIZES}
    ok = sum(results.values())
    print(f"\nsweep 完成: {ok}/{len(BATCH_SIZES)} 正常结束")
    for bs, good in results.items():
        if not good:
            print(f"  bs={bs} 失败(可能 OOM 或发散)")


def load_run(path: Path):
    """返回 (steps, minutes, tokens, val_loss, batch_size)。tokens 从 config 首行推。"""
    recs = [json.loads(l) for l in path.read_text().splitlines()]
    cfg = next((r for r in recs if r.get("type") == "config"), {})
    bs = cfg.get("batch_size", 1)
    ctx = cfg.get("context_length", 256)
    m = [r for r in recs if r.get("type") == "metrics" and "val_loss" in r]
    steps = [r["step"] for r in m]
    minutes = [r["wall_time"] / 60 for r in m]
    tokens = [r["step"] * bs * ctx for r in m]
    val = [r["val_loss"] for r in m]
    return steps, minutes, tokens, val, bs


def plot_all(out: str = "batch_size_curves.png"):
    paths = sorted(LOG_DIR.glob("bs*.jsonl"), key=lambda p: int(p.stem[2:]))  # 按 bs 数值排序
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for path in paths:
        steps, minutes, tokens, val, bs = load_run(path)
        label = f"bs={bs}"
        axes[0].plot(steps, val, label=label)
        axes[1].plot(minutes, val, label=label)
        axes[2].plot(tokens, val, label=label)
    titles = ["per-step efficiency", "wall-clock efficiency", "per-token efficiency"]
    xlabels = ["gradient steps", "wall-clock (min)", "tokens seen"]
    for ax, t, xl in zip(axes, titles, xlabels):
        ax.set_title(t)
        ax.set_xlabel(xl)
        ax.set_ylabel("val loss")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[2].set_xscale("log")  # token 轴跨度大,取 log
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"图已保存: {out}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "probe":
        probe_memory_limit()
    elif mode == "plot":
        plot_all()
    else:  # run
        run_sweep()
        plot_all()
