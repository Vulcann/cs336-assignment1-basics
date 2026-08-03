"""把 TinyStories 的 txt 语料编码成 train.bin / val.bin (uint16 raw binary)。

用法:
    python -m cs336_basics.encode_corpus
输出可直接被 np.memmap(path, dtype=np.uint16, mode="r") 读取。
"""

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from . import pretokenization_example
from .tokenizer import Tokenizer

EOT = "<|endoftext|>"

# ---- worker 进程各自持有一份 tokenizer(initializer 里加载一次,而不是每个任务 pickle 传一遍) ----
_tok: Tokenizer | None = None


def _init_worker(tokenizer_path: str):
    global _tok
    _tok = Tokenizer.load(tokenizer_path)


def _encode_span(args: tuple[str, int, int]) -> np.ndarray:
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.array(_tok.encode(text), dtype=np.uint16)


def encode_file(
    tokenizer_path: str, input_path: str, output_path: str, num_chunks: int = 128, num_workers: int | None = None
) -> int:
    # 按 EOT 对齐分块(复用训练 BPE 时的边界函数,保证不切断 pretoken/特殊 token)
    with open(input_path, "rb") as f:
        boundaries = pretokenization_example.find_chunk_boundaries(f, num_chunks, EOT.encode("utf-8"))
    jobs = [(input_path, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total, t0 = 0, time.time()
    with (
        mp.Pool(num_workers or mp.cpu_count(), initializer=_init_worker, initargs=(tokenizer_path,)) as pool,
        open(output_path, "wb") as fout,
    ):
        # imap:结果按提交顺序流式返回 → 边编码边顺序写盘,内存里最多只有几个块
        for i, arr in enumerate(pool.imap(_encode_span, jobs)):
            arr.tofile(fout)
            total += arr.size
            print(f"\r{output_path}: chunk {i + 1}/{len(jobs)}, {total:,} tokens, {time.time() - t0:.0f}s", end="")
    print()
    return total


if __name__ == "__main__":
    tokenizer_path = "artifacts/tokenizer.json"

    # uint16 前置检查:词表必须 < 65536,否则 id 溢出(静默回绕,产出错误数据)
    tok = Tokenizer.load(tokenizer_path)
    assert len(tok._vocab) < 2**16, f"vocab {len(tok._vocab)} 超出 uint16 范围"

    n_val = encode_file(tokenizer_path, "data/TinyStoriesV2-GPT4-valid.txt", "data/val.bin")
    n_train = encode_file(tokenizer_path, "data/TinyStoriesV2-GPT4-train.txt", "data/train.bin")
    print(f"train: {n_train:,} tokens, val: {n_val:,} tokens")

    # ---- 验证:memmap 读回,解码开头一段,肉眼确认是正常文本 ----
    data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")
    print("val 前 60 个 token 解码:")
    print(tok.decode(data[:60].astype(int).tolist()))
