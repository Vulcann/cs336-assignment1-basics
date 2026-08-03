"""扫描 logs/*.jsonl，生成所有实验的排行榜 + runs.md 索引表。

用法：
    python -m cs336_basics.runs                 # 打印表格
    python -m cs336_basics.runs --md runs.md    # 同时写 markdown（贴进实验日志）
    python -m cs336_basics.runs --filter lr     # 只看 run_name 含 "lr" 的一族

设计原则：日志文件是唯一事实来源（first line = config 快照），
这个脚本纯只读、可随时重跑——不需要手工维护任何表格。


cs336_basics/          代码
data/                  *.bin 语料 + tokenizer.json      ← .gitignore
exp/                   配置：smoke.json / base.json / lr_sweep.json   ← 进 git
logs/{run_name}.jsonl                                   ← .gitignore
ckpt/{run_name}.pt  {run_name}.config.json              ← .gitignore
samples/{run_name}__t0.7_p0.9_s0.txt                    ← 少量可进 git
figs/                  画出来的图，随时可重建           ← .gitignore
experiment_log.md      手写的实验日志                   ← 进 git
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_run(path: Path) -> dict | None:
    """读一个 JSONL 日志：第一行是 config 快照，其余是每步指标。"""
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        return None
    cfg = lines[0].get("config", lines[0])
    rows = [r for r in lines[1:] if "step" in r]
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    train = [(r["step"], r["train_loss"]) for r in rows if "train_loss" in r]
    if not val and not train:
        return None

    best_step, best_val = min(val, key=lambda t: t[1]) if val else (None, None)
    last_wall = rows[-1].get("wall_time", 0.0) if rows else 0.0

    return {
        "run_name": cfg.get("run_name", path.stem),
        "steps": rows[-1]["step"] if rows else 0,
        "lr": (cfg.get("schedule") or {}).get("max_learning_rate"),
        "bs": cfg.get("batch_size"),
        "ctx": cfg.get("context_length"),
        "d_model": (cfg.get("model") or {}).get("d_model"),
        "n_layer": (cfg.get("model") or {}).get("num_layers"),
        "final_train": train[-1][1] if train else None,
        "best_val": best_val,
        "best_step": best_step,
        "minutes": last_wall / 60.0,
        "path": str(path),
    }


COLS = [
    ("run_name", 28, "s"),
    ("lr", 9, ".1e"),
    ("bs", 4, "d"),
    ("ctx", 5, "d"),
    ("d_model", 8, "d"),
    ("n_layer", 8, "d"),
    ("steps", 7, "d"),
    ("final_train", 12, ".4f"),
    ("best_val", 9, ".4f"),
    ("best_step", 10, "d"),
    ("minutes", 8, ".1f"),
]


def fmt(rec: dict) -> str:
    out = []
    for name, w, spec in COLS:
        v = rec.get(name)
        out.append(f"{'-':>{w}}" if v is None else f"{v:>{w}{spec}}")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--filter", default="", help="run_name 子串过滤")
    ap.add_argument("--md", default="", help="额外写出 markdown 表格")
    args = ap.parse_args()

    runs = [
        r for p in sorted(Path(args.log_dir).glob("*.jsonl")) if (r := read_run(p)) and args.filter in r["run_name"]
    ]
    if not runs:
        print(f"{args.log_dir} 下没有可读的日志")
        return
    runs.sort(key=lambda r: (r["best_val"] is None, r["best_val"]))  # 好的排前面

    header = " ".join(f"{n:>{w}}" for n, w, _ in COLS)
    print(header)
    print("-" * len(header))
    for r in runs:
        print(fmt(r))

    if args.md:
        names = [n for n, _, _ in COLS]
        lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
        for r in runs:
            lines.append(
                "| "
                + " | ".join(
                    "-" if r[n] is None else (f"{r[n]:.4f}" if isinstance(r[n], float) else str(r[n])) for n in names
                )
                + " |"
            )
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
