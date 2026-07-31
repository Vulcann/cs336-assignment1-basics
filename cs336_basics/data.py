import torch
import numpy.typing as npt
import os
import argparse
import json

from pathlib import Path
from typing import IO, BinaryIO

from dataclasses import asdict, dataclass, field


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """

    import numpy as np

    x = dataset
    m = context_length
    starts = np.random.randint(0, len(x) - context_length, size=(batch_size,))
    inputs = np.stack([x[i : i + m] for i in starts])  # m = context_length
    targets = np.stack([x[i + 1 : i + m + 1] for i in starts])
    return torch.tensor(inputs, device=device), torch.tensor(targets, device=device)
    # import random
    # data_in = torch.empty(batch_size, context_length, dtype=int, device=device)
    # data_target = torch.empty(batch_size, context_length, dtype=int, device=device)
    # for k in range(batch_size):
    #     i = random.randint(0, len(dataset) - context_length - 1)
    #     data_in[k, :] = torch.from_numpy(dataset[i : i + context_length])
    #     data_target[k, :] = torch.from_numpy(dataset[i + 1 : i + 1 + context_length])
    # return (data_in, data_target)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    obj = {"it": iteration}
    obj["model"] = model.state_dict()
    obj["optimizer"] = optimizer.state_dict()
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    if optimizer is not None:
        optimizer.load_state_dict(obj["optimizer"])
    return obj["it"]


"""load_config / save_config / log_val_loss — 配合 train.py 的骨架使用。
 
设计：dataclass 定默认值（单一事实来源）→ 可选 JSON 文件覆盖 → 命令行 --set 精细覆盖。
优先级：命令行 > JSON > 默认值。cfg.model/optim/schedule 保持为 dict，
方便 TransformerLM(**cfg.model) 这样的解包调用。
"""


# ---------------------------------------------------------------- config
@dataclass
class Config:
    # -- data --
    train_path: str = "data/train.bin"
    val_path: str = "data/val.bin"
    # -- training --
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    context_length: int = 256
    max_iters: int = 10_000
    max_l2_norm: float = 1.0
    # -- intervals --
    log_interval: int = 50
    eval_interval: int = 500
    eval_batches: int = 20  # 每次验证平均多少个 batch
    ckpt_interval: int = 1000
    ckpt_path: str = "ckpt/model.pt"
    resume: str = ""  # 非空则从该 checkpoint 恢复
    # -- grouped hyperparameters（保持 dict 以便 ** 解包）--
    model: dict = field(
        default_factory=lambda: {
            "vocab_size": 10_000,
            "num_layers": 4,
            "d_model": 512,
            "num_heads": 16,
            "d_ff": 1344,  # ≈ 8/3 · 512，取 64 的倍数
            "max_seq_len": 256,
            "theta": 10_000.0,
        }
    )
    optim: dict = field(
        default_factory=lambda: {
            "lr": 3e-4,  # 会被 schedule 每步覆盖，仅作初值
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
            "eot_id": 0,  # 按你 tokenizer 里 <|endoftext|> 的实际 id 填
        }
    )


def load_config() -> Config:
    """默认值 → JSON 文件（--config）→ 命令行 --set 逐项覆盖。
 
    用法示例：
        python train.py
        python train.py --config exp/small.json
        python train.py --config exp/small.json \
            --set batch_size=64 model.d_model=768 schedule.max_learning_rate=1e-3
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="JSON 配置文件路径")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="点号路径覆盖，如 model.d_model=768")
    args = parser.parse_args()

    cfg = Config()

    # 1) JSON 覆盖（浅层字段直接替换；model/optim/schedule 做 dict 合并）
    if args.config:
        file_cfg = json.loads(Path(args.config).read_text())
        for key, value in file_cfg.items():
            if isinstance(getattr(cfg, key, None), dict):
                getattr(cfg, key).update(value)  # 组内合并，未提及的键保留默认
            else:
                setattr(cfg, key, value)

    # 2) 命令行 --set 覆盖（值先按 JSON 解析，失败则当字符串）
    for item in args.set:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if "." in key:  # 组内键：model.d_model=768
            group, sub = key.split(".", 1)
            getattr(cfg, group)[sub] = value
        else:  # 顶层键：batch_size=64
            setattr(cfg, key, value)

    return cfg


def save_config(cfg: Config, path: str) -> None:
    """把最终生效的配置快照成 JSON（训练开始时调用，存到 ckpt 目录旁）。

    价值：实验可复现——多次覆盖合成后的"实际配置"落盘，
    之后 --config 指回这个文件即可精确复跑。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
