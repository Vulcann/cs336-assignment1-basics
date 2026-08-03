import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np  # 移到顶部(原来藏在 get_batch 函数体里)
import torch


def get_batch(dataset, batch_size, context_length, device):
    starts = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))
    inputs = np.stack([dataset[i : i + context_length] for i in starts]).astype(np.int64)
    targets = np.stack([dataset[i + 1 : i + context_length + 1] for i in starts]).astype(np.int64)
    return torch.tensor(inputs, device=device), torch.tensor(targets, device=device)
    # 注:加了 .astype(np.int64)——memmap 是 uint16,不转的话 torch.tensor 得到 int16/uint16,
    # embedding 索引会报错;顺带删掉了注释掉的旧实现


def save_checkpoint(model, optimizer, iteration, out):
    if isinstance(out, (str, Path)):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"it": iteration, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, out)


def load_checkpoint(src, model, optimizer=None, map_location="cpu") -> int:
    obj = torch.load(src, map_location=map_location)
    model.load_state_dict(obj["model"])
    if optimizer is not None:
        optimizer.load_state_dict(obj["optimizer"])
    return obj["it"]


import subprocess


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "nogit"


# ---------------------------------------------------------------- config


@dataclass
class Config:
    # -- run identity:一切路径由 run_name 派生 --
    run_name: str = ""  # 空 = finalize() 时自动从超参拼名
    log_dir: str = "logs"
    ckpt_dir: str = "ckpt"
    # -- data / assets --
    train_path: str = "data/train.bin"
    val_path: str = "data/val.bin"
    tokenizer_path: str = "artifacts/tokenizer.json"
    # -- training --
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    context_length: int = 256
    max_iters: int = 10_000
    max_l2_norm: float = 1.0
    # -- intervals --
    log_interval: int = 50
    eval_interval: int = 500
    eval_batches: int = 20
    ckpt_interval: int = 1000
    resume: str = ""  # 非空则从该 checkpoint 路径恢复
    # -- grouped hyperparameters --
    model: dict = field(
        default_factory=lambda: {
            "vocab_size": 10_000,
            "num_layers": 4,
            "d_model": 512,
            "num_heads": 16,
            "d_ff": 1344,
            "max_seq_len": 256,
            "theta": 10_000.0,
        }
    )
    optim: dict = field(
        default_factory=lambda: {
            "lr": 3e-4,
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }
    )
    schedule: dict = field(
        default_factory=lambda: {
            "max_learning_rate": 3e-4,
            "min_learning_rate": 3e-5,
            "warmup_iters": 200,
            "cosine_cycle_iters": 10_000,
        }
    )
    decode: dict = field(
        default_factory=lambda: {
            "temperature": 0.8,
            "sampling_p": 0.9,
            "max_new_tokens": 256,
            # eot_id 已删除:运行时从 tokenizer 查(派生量不进 config)
        }
    )

    def finalize(self) -> "Config":
        """所有覆盖合并完成后调用:定名。"""
        if not self.run_name:
            self.run_name = (
                f"lr{self.schedule['max_learning_rate']:.0e}"
                f"_bs{self.batch_size}"
                f"_d{self.model['d_model']}"
                f"_L{self.model['num_layers']}"
            )
            self.git_sha = _git_sha()  # git commit hash
        return self

    # -- 派生路径:只读属性,不进 asdict 快照 --
    @property
    def ckpt_path(self) -> str:
        return f"{self.ckpt_dir}/{self.run_name}.pt"

    @property
    def log_path(self) -> str:
        return f"{self.log_dir}/{self.run_name}.jsonl"

    @property
    def config_snapshot_path(self) -> str:
        return f"{self.ckpt_dir}/{self.run_name}.config.json"


def load_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = Config()
    if args.config:
        for key, value in json.loads(Path(args.config).read_text()).items():
            if isinstance(getattr(cfg, key, None), dict):
                getattr(cfg, key).update(value)
            else:
                setattr(cfg, key, value)
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if "." in key:
            group, sub = key.split(".", 1)
            getattr(cfg, group)[sub] = value
        else:
            setattr(cfg, key, value)

    return cfg.finalize()  # ← 覆盖全部生效后才定名派生


def save_config(cfg: Config, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))


class ExperimentLogger:
    """每条记录一行 JSON:{"step": ..., "wall_time": ..., "train_loss": ..., ...}"""

    def __init__(self, cfg: Config):  # 直接收 cfg,路径不再拼两遍
        self.path = Path(cfg.log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self._write({"type": "config", "run_name": cfg.run_name, **asdict(cfg)})

    def log(self, step: int, **metrics):
        self._write({"type": "metrics", "step": step, "wall_time": time.time() - self.t0, **metrics})

    def _write(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def plot(paths: list[str], out: str = "loss_curves.png"):
    """
    画出训练曲线
    i.e.  paths = ["logs/lr3e-4_bs32.jsonl", "logs/lr1e-3_bs32.jsonl"]:
    """
    import matplotlib.pyplot as plt

    def load_run(path):
        recs = [json.loads(l) for l in open(path)]
        m = [r for r in recs if r.get("type") == "metrics" and "train_loss" in r]
        return ([r["step"] for r in m], [r["wall_time"] / 60 for r in m], [r["train_loss"] for r in m])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for path in paths:
        steps, minutes, loss = load_run(path)
        ax1.plot(steps, loss, label=Path(path).stem)
        ax2.plot(minutes, loss, label=Path(path).stem)
    ax1.set_xlabel("gradient steps")
    ax2.set_xlabel("wall-clock (min)")
    ax1.set_ylabel("train loss")
    ax1.legend()
    ax2.legend()
    plt.savefig(out, dpi=120)  # 输出名可指定,多组对比不互相覆盖
