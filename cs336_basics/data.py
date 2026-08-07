import argparse
import json
import random
import secrets
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np  # 移到顶部(原来藏在 get_batch 函数体里)
import torch

from . import optimizers


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


# ---------------------------------------------------------------- 版本戳

# 锚在源码文件上，不用进程 cwd:sweep 起的子进程、tmux runner、从别的目录手跑,
# cwd 是什么都有可能,而这个文件永远在仓库里。
_REPO = Path(__file__).resolve().parent


def _git(*args) -> tuple[int, str]:
    """跑一条 git。失败一律不抛——版本戳不该把训练搞崩。"""
    try:
        p = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True, timeout=5)
        return p.returncode, p.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):  # 机器上压根没装 git
        return 127, ""


def _git_sha() -> str:
    """短 sha;工作区有未提交改动时拼上 -dirty。"""
    rc, rev = _git("rev-parse", "--short", "HEAD")
    if rc != 0:  # 不在仓库里,或者仓库还没有任何 commit
        return "nogit"
    # --untracked-files=no 是关键:没有它,runs/ 下刚写出来的输出、__pycache__ 会让你
    # 从第一次训练之后永远 dirty,这个标记也就再不带信息了。代价是新建但没 add 的 .py
    # 不算脏——真要堵这个口子,先把 .gitignore 写干净,再换成 --untracked-files=all。
    rc, out = _git("status", "--porcelain", "--untracked-files=no")
    return f"{rev}-dirty" if (rc != 0 or out) else rev


def _save_patch(run_dir) -> str | None:
    """脏的时候把未提交的改动存成 patch,让 -dirty 从"警告"变成"可复现"。

    只有 sha 的话,-dirty 只是告诉你"这个结果复现不了";存了 patch 就是
    git checkout <sha> && git apply uncommitted.patch。
    """
    rc, diff = _git("diff", "HEAD")  # 已 add 未 commit 的也在里面
    if rc != 0 or not diff:
        return None
    out = Path(run_dir) / "uncommitted.patch"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(diff + "\n")
    return str(out)


# ---------------------------------------------------------------- 随机性


def seed_everything(seed: int) -> None:
    """给所有会影响训练的 RNG 播种。

    别指望过头:这样得到的是"同机同环境下大致可复现",不是逐比特相同。GPU 上
    cuDNN 的算法选择和 atomics 的累加顺序都是不确定的,真要逐比特得开
    torch.use_deterministic_algorithms(True) 并设 CUBLAS_WORKSPACE_CONFIG,代价是
    明显变慢。做 ablation 用不着——seed 的作用是让你能估出噪声底,不是让两次跑出
    一模一样的数。
    """
    random.seed(seed)
    np.random.seed(seed)  # get_batch 里的 np.random.randint 靠这一句
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- config

"""
Config dataclass 默认值     ← 代码，git 里，唯一事实来源
       ↓ 被覆盖
exp/*.json                  ← 具名实验，手写，只写"和默认值的差异"
       ↓ 被覆盖
--set k=v                   ← 一次性微调，不落文件
       ↓ 合成后
runs/<tag>/config.json      ← 机器写的全量快照，只读，用于复现
"""

SCRATCH = "_scratch"  # 随手跑的默认落脚点;它下面不做覆盖检查


@dataclass
class Config:
    # -- run identity:一切路径由 tag 派生 --
    #
    # tag 是人给的,不是从超参推导出来的。推导名字的老办法(lr..._bs..._d..._L...)
    # 有个致命毛病:公式只覆盖当时想到的那几个维度,而 ablation 恰恰是去改不在公式
    # 里的东西(norm 位置、激活函数),于是几个 arm 派生出同一个名字,静默互相覆盖。
    # 人给的 tag 没有这个问题——代价是可能手打重复,由 prepare_run_dir() 挡。
    #
    # tag 可以带 /,直接就是目录层级:"abl_norm/postnorm_s0" → runs/abl_norm/postnorm_s0/
    # 一层平铺撑不住 arm x seed:三个 arm 三个 seed 就是九个同级目录,下周再来一族就
    # 只能靠字符串前缀去 glob 了。
    tag: str = ""  # 空 → finalize() 落到 _scratch/<自动名>
    git_sha: str = ""
    runs_dir: str = "runs"
    force: bool = False  # 允许覆盖已有结果。这不是超参,是调用方式,别写进 exp/*.json
    # -- 随机性 --
    # None = 随机抽一个。随机默认之所以安全,不是因为随机,而是因为 finalize() 会把
    # 抽出来的值定死并落进 config.json——从那一刻起它和显式指定的 run 没有区别。
    seed: int | None = None
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
    # -- 学习率：唯一的自由量，schedule / optim.lr 全部由它派生 --
    lr: float = 1e-3
    lr_min_ratio: float = 0.1  # min_lr = lr * ratio
    warmup_iters: int = 200
    cosine_cycle_iters: int = 0  # 0 → finalize 里设成 max_iters
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
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }
    )
    schedule: dict = field(default_factory=dict)
    decode: dict = field(
        default_factory=lambda: {
            "temperature": 0.8,
            "sampling_p": 0.9,
            "max_new_tokens": 256,
            # eot_id 已删除:运行时从 tokenizer 查(派生量不进 config)
        }
    )

    def auto_name(self) -> str:
        """按超参拼一个描述性名字。只在没给 tag 时兜底,以及给 sweep 当叶子名用。"""
        return f"lr{self.lr:.0e}_bs{self.batch_size}_d{self.model['d_model']}_L{self.model['num_layers']}"

    def finalize(self) -> "Config":
        # seed 必须是 `is None`,不能写 `if not self.seed`:seed=0 是假值,用 not 会把
        # 显式指定的 0 当成"没设"再抽一次,于是你以为在跑 seed 0,实际每次都是新的。
        if self.seed is None:
            # 用系统熵而不是 random.randint:如果进程里有任何东西已经给全局 RNG 播过种,
            # random 抽出来的"随机 seed"其实是确定的,你会得到一堆每次重启都一样的 seed。
            # 上界取 2**31 是各处都安全的交集(np.random.seed 要求 < 2**32)。
            self.seed = secrets.randbelow(2**31)

        if self.cosine_cycle_iters == 0:
            self.cosine_cycle_iters = self.max_iters  # 默认让余弦周期和训练长度对齐

        self.schedule = {
            "max_learning_rate": self.lr,
            "min_learning_rate": self.lr * self.lr_min_ratio,
            "warmup_iters": self.warmup_iters,
            "cosine_cycle_iters": self.cosine_cycle_iters,
        }
        # 构造 optimizer 需要一个初值 —— 直接问 schedule "第 0 步该是多少"，天然自洽
        self.optim["lr"] = optimizers.cosine_annealing_lr_step(0, **self.schedule)

        if not self.tag:
            self.tag = f"{SCRATCH}/{self.auto_name()}"
        # git_sha 无条件记。原来它嵌在 `if not run_name` 里,显式给了名字的 run 就
        # 一个字都不记,而那恰恰是你最认真跑的那些。
        self.git_sha = _git_sha()
        return self

    # -- 派生路径:只读属性,不进 asdict 快照 --
    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.tag

    @property
    def ckpt_path(self) -> str:
        return str(self.run_dir / "ckpt.pt")

    @property
    def log_path(self) -> str:
        return str(self.run_dir / "metrics.jsonl")

    @property
    def config_snapshot_path(self) -> str:
        return str(self.run_dir / "config.json")


def _check_tag(tag: str) -> None:
    """tag 会拼进路径,而它来自命令行——挡掉能跑出 runs/ 之外的写法。"""
    p = Path(tag)
    if p.is_absolute() or ".." in p.parts or tag.strip() != tag or not tag:
        raise SystemExit(f"非法 tag {tag!r}:不能是绝对路径、不能含 '..'、不能有首尾空白。")


def prepare_run_dir(cfg: Config) -> Path:
    """建 run 目录,并挡住"手打了同一个 tag 两次"。

    人给 tag 解决了派生名字互撞的问题,但没解决人自己撞自己:崩了重跑、忘了这名字
    上周用过、改一行想再跑一次——都会覆盖上一次的结果,而且和以前一样是静默的。
    这个检查就是那道闸,没有它,整套方案的可靠性全押在你的记性上。
    """
    _check_tag(cfg.tag)
    run_dir = cfg.run_dir
    scratch = Path(cfg.tag).parts[0] == SCRATCH  # 随手跑的,语义就是"随便冲"
    if run_dir.exists() and any(run_dir.iterdir()) and not cfg.force and not scratch:
        raise SystemExit(
            f"{run_dir} 已经有结果了,不覆盖。\n"
            f"   换个 tag  : --set tag=...\n"
            f"   确实要覆盖: --set force=true\n"
            f"   (随手跑的不用起名,默认落在 {SCRATCH}/ 下,那里不做这个检查)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    # force 是"重跑",不是"接着跑"——旧的 jsonl 必须清掉。ExperimentLogger 用 "a" 打开
    # (resume 时接着写才对),不清的话两次 run 的记录会串在一个文件里:step 序列变成
    # 0..29 然后又跳回 0,画出来是锯齿,summarize 的 final/steps 也全错。
    # cfg.resume 非空时是真的要接着写,那就别动。
    old_log = Path(cfg.log_path)
    if cfg.force and not cfg.resume and old_log.exists():
        old_log.unlink()
        print(f"--force: 已清掉旧的 {old_log}")

    if cfg.git_sha.endswith("-dirty") and (patch := _save_patch(run_dir)):
        print(f"工作区有未提交改动,已存 {patch}(复现: git checkout {cfg.git_sha[:-6]} && git apply 它)")
    return run_dir


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
            value = json.loads(raw)  # --set seed=null 走这里还原成 None
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
        self._write({"type": "config", "tag": cfg.tag, **asdict(cfg)})

    def log(self, step: int, **metrics):
        self._write({"type": "metrics", "step": step, "wall_time": time.time() - self.t0, **metrics})

    def _write(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def _label_of(path: str) -> str:
    """图例名。文件名现在统一叫 metrics.jsonl,拿 stem 会得到一堆一模一样的图例,
    所以去 jsonl 头上那条 config 记录里读 tag——那是机器写的,不会因为手滑而错。"""
    try:
        with open(path) as f:
            head = json.loads(f.readline())
        if head.get("type") == "config" and head.get("tag"):
            return head["tag"]
    except Exception:
        pass
    return str(Path(path).parent.name or Path(path).stem)


def plot(paths: list[str], out: str = "loss_curves.png"):
    """
    画出训练曲线
    i.e.  paths = ["runs/abl_norm/prenorm_s0/metrics.jsonl", "runs/abl_norm/postnorm_s0/metrics.jsonl"]
    """
    import matplotlib.pyplot as plt

    def load_run(path):
        recs = [json.loads(l) for l in open(path)]
        m = [r for r in recs if r.get("type") == "metrics" and "train_loss" in r]
        return ([r["step"] for r in m], [r["wall_time"] / 60 for r in m], [r["train_loss"] for r in m])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for path in paths:
        steps, minutes, loss = load_run(path)
        label = _label_of(path)
        ax1.plot(steps, loss, label=label)
        ax2.plot(minutes, loss, label=label)
    ax1.set_xlabel("gradient steps")
    ax2.set_xlabel("wall-clock (min)")
    ax1.set_ylabel("train loss")
    ax1.legend()
    ax2.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)  # 输出名可指定,多组对比不互相覆盖
