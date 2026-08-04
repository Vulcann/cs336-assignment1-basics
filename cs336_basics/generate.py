"""CS336 Assignment 1 — Problem (generate)

从训练好的 checkpoint 采样文本，至少 256 个 token（或提前遇到 <|endoftext|> 停止），
把结果连同采样参数一起落盘到 samples/{run_name}__{标签}.txt，方便横向比较不同解码设置。

用法：
    # 用某次 run 的配置快照 + 该 run 的 checkpoint
    python -m cs336_basics.generate --config ckpt/lr1e-03_bs32_d512_L4.config.json

    # 覆盖解码参数（单次）
    python -m cs336_basics.generate --config ... \
        --set decode.temperature=0.8 decode.sampling_p=0.9 \
        --prompt "Once upon a time"

    # 一次跑完对比矩阵（作业里"讨论影响质量的因素"直接引用这些文件）
    python -m cs336_basics.generate --config ... --sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .data import Config, load_config, load_checkpoint
from .decoder import Decoder
from .tokenizer import Tokenizer
from .nn_modules import TransformerLM  # 按你的实际模块名调整

EOT = "<|endoftext|>"


# ------------------------------------------------------------------ 装配


def build(cfg: Config):
    """tokenizer → eot_id → 模型 → 载入权重 → eval 模式。返回 (lm, tokenizer, eot_id)。"""
    tokenizer = Tokenizer.load(cfg.tokenizer_path)

    # eot_id 必须从 tokenizer 查，不写死在 config 里：
    # id 0..255 是 256 个原始字节，special token 排在其后 → 这个词表里是 256。
    eot_id = tokenizer.token_to_id(EOT.encode("utf-8"))
    if eot_id is None:
        raise ValueError(f"{EOT} 不在词表里，检查 BPETrainer 的 special_tokens")

    lm = TransformerLM(**cfg.model).to(cfg.device)

    print(f"cfg: {cfg}")

    ckpt = Path(cfg.ckpt_path)
    if not ckpt.exists():
        sys.exit(f"找不到 checkpoint: {ckpt}")
    it = load_checkpoint(ckpt, lm, optimizer=None)  # 推理不需要优化器状态
    print(f"loaded {ckpt}  (iteration {it})  vocab={len(tokenizer._vocab)}  eot_id={eot_id}")

    lm.eval()  # 关 dropout / 切 norm 到推理行为
    return lm, tokenizer, eot_id


# ------------------------------------------------------------------ 单次采样


@torch.no_grad()
def sample(
    lm,
    tokenizer,
    eot_id,
    cfg: Config,
    prompt: str,
    temperature: float,
    sampling_p: float,
    max_new_tokens: int,
    seed: int | None = None,
) -> dict:
    """跑一次生成，返回一个自描述的 dict（正文 + 全部采样参数）。"""
    if seed is not None:
        torch.manual_seed(seed)  # 采样是随机的 → 想复现就固定种子

    decoder = Decoder(
        lm=lm,
        temperature=temperature,
        sampling_p=sampling_p,
        eot_id=eot_id,
        max_seq_len=cfg.model["max_seq_len"],
        device=cfg.device,
    )

    prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    out_ids = decoder.generate(prompt_ids, max_new_tokens=max_new_tokens)

    new_ids = out_ids[len(prompt_ids) :]
    hit_eot = bool(len(new_ids) and new_ids[-1].item() == eot_id)

    return {
        "run_name": cfg.run_name,
        "prompt": prompt,
        "temperature": temperature,
        "sampling_p": sampling_p,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "n_new_tokens": int(len(new_ids)),
        "hit_eot": hit_eot,
        "text": tokenizer.decode(out_ids.tolist()),  # 含 prompt 的完整文本
    }


def dump(rec: dict, out_dir: str = "samples") -> Path:
    """把一次采样写成带 YAML 头的 txt——文件本身自描述，不依赖记忆。"""
    tag = f"t{rec['temperature']}_p{rec['sampling_p']}_s{rec['seed']}"
    path = Path(out_dir) / f"{rec['run_name']}__{tag}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)

    header = "\n".join(
        f"# {k}: {rec[k]}"
        for k in (
            "run_name",
            "prompt",
            "temperature",
            "sampling_p",
            "max_new_tokens",
            "seed",
            "n_new_tokens",
            "hit_eot",
        )
    )
    path.write_text(header + "\n" + "-" * 60 + "\n" + rec["text"], encoding="utf-8")
    return path


# ------------------------------------------------------------------ 入口


def main():
    # 复用 data.load_config 的 --config / --set 机制，再补两个本脚本专属的参数。
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--prompt", type=str, default="Once upon a time")
    extra.add_argument("--seed", type=int, default=0)
    extra.add_argument("--sweep", action="store_true", help="跑温度/top-p 对比矩阵")
    args, rest = extra.parse_known_args()
    sys.argv = [sys.argv[0]] + rest  # 剩下的交给 load_config
    cfg = load_config()

    lm, tokenizer, eot_id = build(cfg)

    # 作业要求：至少 256 个 token，或第一个 <|endoftext|> 为止
    max_new = int(cfg.decode.get("max_new_tokens", 256))

    if args.sweep:
        settings = [
            (0.0, 1.0),  # greedy：看模型的"最自信"输出，通常会陷入重复
            (0.7, 1.0),  # 只调温度，不截断
            (1.0, 0.95),  # 只做 nucleus
            (0.7, 0.9),  # 常用组合
            (1.5, 1.0),  # 高温：看退化成什么样
        ]
    else:
        settings = [(float(cfg.decode["temperature"]), float(cfg.decode["sampling_p"]))]

    for temp, p in settings:
        rec = sample(
            lm,
            tokenizer,
            eot_id,
            cfg,
            prompt=args.prompt,
            temperature=temp,
            sampling_p=p,
            max_new_tokens=max_new,
            seed=args.seed,
        )
        path = dump(rec)
        print(f"\n=== T={temp} p={p} → {path} ({rec['n_new_tokens']} tokens, eot={rec['hit_eot']}) ===")
        print(rec["text"][:600])


if __name__ == "__main__":
    main()
