"""LR sweep driver：逐个起 train 子进程，结束后汇总结果并画对比图。

用法:  python -m cs336_basics.lr_sweep
       ./tmux_sweep.sh start lr        # 同上，挂到 tmux 后台跑

只动 lr，不动 schedule
---------------------
Config.finalize() 里 schedule 是**整个重建**的：

    self.schedule = {"max_learning_rate": self.lr,
                     "min_learning_rate": self.lr * self.lr_min_ratio, ...}

所以 schedule.max_learning_rate 是派生量，不是自由量。往它上面写覆盖会被 finalize
原样冲掉——不报错、不警告，只是没效果。lr 才是那个唯一的自由量，扫它就够了。

sweep 怎么跟 train 说话
----------------------
把这一 run 的覆盖项写成一个小 JSON，然后 train --config 它。用到的接口只有
「--config 读一个 JSON」这一条——不用拼 --set 字符串，也不用管点号路径怎么解析。
覆盖表在本进程里先往 Config 上叠一遍，字段名写错当场 KeyError，不会等烧掉一小时
GPU 才发现。

tag 由这里拼，不由 train 猜
--------------------------
每个 run 的 tag 是 "sweep_lr/<自动名>_s<seed>"，所以 sweep 事先就知道每个 run 的
产物会落在哪，汇总时直接按名单去读，不用 glob——glob 会把历史上所有旧 run 一起
捞进来，图上多出几条不知从哪来的线。
"""

import json
import math
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from .data import Config

# ---- 按需修改的部分 --------------------------------------------------------
TRAIN_MODULE = "cs336_basics.train"
GROUP = "sweep_lr"  # 这一族的目录：runs/sweep_lr/...
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]  # 要扫的取值
SEED = 0  # 扫 lr 时固定 seed，否则 lr 的效应和 seed 噪声混在一起
FORCE = False  # True = 允许覆盖上一轮同名 run 的结果（改了代码想重扫时用）
BASE: dict = {}  # 这一族共享的改动，如 {"max_iters": 2000, "batch_size": 64}
GROUPS = ("model", "optim", "schedule", "decode")  # 这几个是 dict，合并而不是整个替换

# 每个 run 的覆盖表写在这，不写进 run 目录本身。
# 原因很实际：train 的 prepare_run_dir() 看见目录非空就拒绝启动（防手打撞名），
# 而 sweep 要是先往里丢一个 json，就等于自己把自己挡在门外——三个 run 全部
# "已经有结果了"，一个都跑不起来。run 目录只放 run 产出的东西。
SWEEP_DIR = Path("runs/_sweep")
# ---------------------------------------------------------------------------


def plan(lr: float) -> tuple[Config, dict]:
    """把一个取值翻译成 (定好 tag 的 cfg, 覆盖表)。字段名写错这里就炸。

    这里叠覆盖的方式必须和 data.load_config 完全一致（标量 setattr、组字典 update），
    否则 sweep 算出来的 tag 会和 train 实际用的对不上，汇总时按名单去读就读空。
    """
    overrides = {**BASE, "lr": lr, "seed": SEED, "force": FORCE}
    cfg = Config()
    valid = {f.name for f in fields(cfg)}
    for key, val in overrides.items():
        if key not in valid:  # 字段名写错当场炸，不用等烧掉一小时 GPU 才发现
            raise KeyError(f"Config 里没有 {key!r} 这个字段；可用的: {sorted(valid)}")
        if key in GROUPS:
            getattr(cfg, key).update(val)
        else:
            setattr(cfg, key, val)
    overrides["tag"] = cfg.tag = f"{GROUP}/{cfg.auto_name()}_s{SEED}"
    return cfg.finalize(), overrides


def run_one(cfg: Config, overrides: dict) -> bool:
    """起一个训练子进程，返回是否正常结束。

    起子进程而不是 import train 直接跑：一个 run 发散/OOM 崩掉不影响后面的，
    显存也随进程退出彻底归还。
    """
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = SWEEP_DIR / f"{cfg.tag.replace('/', '-')}.json"
    cfg_path.write_text(json.dumps(overrides, indent=2) + "\n")

    cmd = [sys.executable, "-u", "-m", TRAIN_MODULE, "--config", str(cfg_path)]
    print(f"\n=== {cfg.tag}  (lr={cfg.lr:g}) ===", flush=True)
    # -u + DEVNULL 是给后台跑准备的：tmux 里 stdout 是管道不是终端，Python 默认块
    # 缓冲，不加 -u 训练输出要攒几 KB 才吐一次，看着像卡死；后台也没有终端可读。
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL)  # 阻塞到该 run 结束，输出直通
    ok = proc.returncode == 0
    if not ok:  # 发散到 NaN 崩了也别中断整个 sweep
        print(f"!!! {cfg.tag} failed (returncode {proc.returncode}), 继续下一个", flush=True)
    return ok


def summarize(runs: list[tuple[str, str]]) -> None:
    """读各 run 的 jsonl，打印最终/最优 loss 汇总表。runs = [(tag, log_path), ...]"""
    print(f"\n{'run':<40} {'final':>8} {'best':>8} {'steps':>7}")
    rows = []
    for tag, path in runs:
        if not Path(path).exists():  # 崩在第一个 log_interval 之前，什么都没写出来
            continue
        recs = [json.loads(l) for l in Path(path).read_text().splitlines()]
        losses = [(r["step"], r["train_loss"]) for r in recs if r.get("type") == "metrics" and "train_loss" in r]
        if not losses:
            continue
        final_step, final_loss = losses[-1]
        # 发散的 run 后半段全是 nan，得先滤掉：min() 遇到 nan 的结果取决于它排在第几个。
        finite = [l for _, l in losses if math.isfinite(l)]
        best_loss = min(finite) if finite else float("nan")
        rows.append((tag, final_loss, best_loss, final_step))
    # inf 兜底：全程 nan 的 run 排最后，而不是靠 nan 的比较行为随机插在中间
    for tag, final_loss, best_loss, steps in sorted(rows, key=lambda r: r[2] if math.isfinite(r[2]) else math.inf):
        print(f"{tag:<40} {final_loss:>8.4f} {best_loss:>8.4f} {steps:>7}")


if __name__ == "__main__":
    plans = [plan(lr) for lr in LRS]  # 先全部构造一遍：要炸在第一秒炸，别跑一半才炸
    print(f"sweep {len(plans)} 个 run:", flush=True)
    for cfg, _ in plans:
        print(f"  {cfg.run_dir}", flush=True)

    results = [(cfg, run_one(cfg, ov)) for cfg, ov in plans]

    done = sum(ok for _, ok in results)
    print(f"\nsweep 完成: {done}/{len(plans)} 个 run 正常结束")
    for cfg, ok in results:
        if not ok:
            print(f"  失败: {cfg.tag}")

    runs = [(cfg.tag, cfg.log_path) for cfg, _ in results]  # 按名单读，不 glob
    summarize(runs)

    # 画图的 import 放这里而不是文件顶上：matplotlib 装坏了之类的问题不该让整个
    # sweep 在第一秒就起不来——训练是几小时的事，画图是几秒的事。
    try:
        from .data import plot  # 复用已有的画图函数

        out = f"runs/{GROUP}/lr_sweep.png"
        plot([p for _, p in runs if Path(p).exists()], out=out)
        print(f"曲线图: {out}")
    except Exception as e:
        print(f"画图失败（不影响已跑完的结果）: {type(e).__name__}: {e}")
