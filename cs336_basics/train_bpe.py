from collections import Counter
from typing import BinaryIO

import regex

from . import tokenizer
from . import pretokenization_example


class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str] = None):
        self.vocab_size: int = vocab_size
        self.vocab: dict[int, bytes] = {}
        self.special_tokens: list[str] = special_tokens if special_tokens is not None else []
        self.merges: list[tuple[bytes, bytes]] = []

    def train(self, f: BinaryIO, desired_num_chunks: int, split_special_token: bytes):
        pre_tokens: dict[bytes, tuple[int, list[bytes], dict[tuple[bytes, bytes], int]]] = {}

        self.special_tokens = list(set(self.special_tokens + [split_special_token.decode("utf-8")]))

        # ---------- 预分词部分：与你原版完全相同 ----------
        boundaries = pretokenization_example.find_chunk_boundaries(f, desired_num_chunks, split_special_token)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            split_chunk = tokenizer.Tokenizer.split_on_special_tokens(chunk, self.special_tokens)
            for chunk in split_chunk:
                if chunk in self.special_tokens:
                    continue
                for pre_token in regex.findall(tokenizer.Tokenizer.GPT2_PAT, chunk):
                    pre_token_bytes = pre_token.encode("utf-8")
                    if pre_token_bytes not in pre_tokens:
                        bytes_list = [bytes([b]) for b in pre_token_bytes]
                        pairs_count_local = dict(Counter(zip(bytes_list, bytes_list[1:])))
                        pre_tokens[pre_token_bytes] = (1, bytes_list, pairs_count_local)
                    else:
                        count, bytes_list, pair_count = pre_tokens[pre_token_bytes]
                        pre_tokens[pre_token_bytes] = (count + 1, bytes_list, pair_count)

        self.vocab = {i: bytes([i]) for i in range(256)}
        for t in self.special_tokens:
            self.vocab[len(self.vocab)] = t.encode("utf-8")

        # ---------- 新增：循环前一次性构建全局计数 + 倒排索引 ----------
        # pairs_count: pair → 全局出现次数（之后只做增量,不再重算）
        # pair_index:  pair → 含有该 pair 的 pretoken 集合（用于定位受影响的词）
        pairs_count: dict[tuple[bytes, bytes], int] = {}
        pair_index: dict[tuple[bytes, bytes], set[bytes]] = {}
        for pre_token_bytes, (count, bytes_list, pairs_count_local) in pre_tokens.items():
            for pair, in_count in pairs_count_local.items():
                pairs_count[pair] = pairs_count.get(pair, 0) + in_count * count
                pair_index.setdefault(pair, set()).add(pre_token_bytes)

        # ---------- merge 循环：全量重算改为增量维护 ----------
        while self.vocab_size > len(self.vocab):
            if not pairs_count:
                break
            most_frequent_pair = max(pairs_count.items(), key=lambda x: (x[1], x[0]))[0]
            self.merges.append(most_frequent_pair)
            self.vocab[len(self.vocab)] = most_frequent_pair[0] + most_frequent_pair[1]

            a, b = most_frequent_pair
            merged = a + b
            # 只遍历真正含有这个 pair 的词（原版:遍历全部 pre_tokens）
            for pre_token_bytes in list(pair_index.get(most_frequent_pair, ())):
                count, bytes_list, pairs_count_local = pre_tokens[pre_token_bytes]

                # 1) 撤销:把这个词的旧贡献从全局计数和索引里减掉
                for pair, in_count in pairs_count_local.items():
                    pairs_count[pair] -= in_count * count
                    if pairs_count[pair] <= 0:
                        del pairs_count[pair]
                    pair_index[pair].discard(pre_token_bytes)

                # 2) 词内合并:与你原版相同
                new_bytes_list = []
                i = 0
                while i < len(bytes_list):
                    if i < len(bytes_list) - 1 and bytes_list[i] == a and bytes_list[i + 1] == b:
                        new_bytes_list.append(merged)
                        i += 2
                    else:
                        new_bytes_list.append(bytes_list[i])
                        i += 1

                # 3) 登记:把新贡献加回全局计数和索引
                new_local = dict(Counter(zip(new_bytes_list, new_bytes_list[1:])))
                for pair, in_count in new_local.items():
                    pairs_count[pair] = pairs_count.get(pair, 0) + in_count * count
                    pair_index.setdefault(pair, set()).add(pre_token_bytes)
                pre_tokens[pre_token_bytes] = (count, new_bytes_list, new_local)

        # # for _ in range(self.vocab_size - len(vocab_list)):
        # while self.vocab_size > len(self.vocab):
        #     pairs_count: dict[tuple[bytes, bytes], int] = {}
        #     for pre_token_bytes, (count, bytes_list, pairs_count_local) in pre_tokens.items():
        #         for pair, in_count in pairs_count_local.items():
        #             if pair not in pairs_count:
        #                 pairs_count[pair] = in_count * count
        #             else:
        #                 pairs_count[pair] += in_count * count

        #     most_frequent_pair = max(pairs_count.items(), key=lambda x: (x[1], x[0]))[0]
        #     self.merges.append(most_frequent_pair)
        #     self.vocab[len(self.vocab)] = most_frequent_pair[0] + most_frequent_pair[1]

        #     # 把 most_frequent_pair 合并进受影响的词
        #     a, b = most_frequent_pair
        #     merged = a + b
        #     for pre_token_bytes, (count, bytes_list, pairs_count_local) in list(pre_tokens.items()):
        #         if most_frequent_pair not in pairs_count_local:
        #             continue
        #         new_bytes_list = []
        #         i = 0
        #         while i < len(bytes_list):
        #             if i < len(bytes_list) - 1 and bytes_list[i] == a and bytes_list[i + 1] == b:
        #                 new_bytes_list.append(merged)
        #                 i += 2
        #             else:
        #                 new_bytes_list.append(bytes_list[i])
        #                 i += 1

        #         pairs_count_local: dict[tuple[bytes, bytes], int] = {}
        #         for i in range(len(new_bytes_list) - 1):
        #             pair = (new_bytes_list[i], new_bytes_list[i + 1])
        #             if pair not in pairs_count_local:
        #                 pairs_count_local[pair] = 1
        #             else:
        #                 pairs_count_local[pair] += 1
        #         pre_tokens[pre_token_bytes] = (count, new_bytes_list, pairs_count_local)


EOT = "<|endoftext|>"


def train_and_save_tokenizer(
    input_path: str,  # 训练语料,txt 文件
    vocab_size: int,  # 目标词表大小(如 10000)
    tokenizer_path: str,  # 序列化输出路径(如 artifacts/tokenizer.json)
    desired_num_chunks: int = 8,
) -> tuple[tokenizer.Tokenizer, int]:

    # 1) 读文件训练 —— 注意 "rb":BPE 在字节层工作,trainer 签名也是 BinaryIO
    trainer = BPETrainer(vocab_size=vocab_size, special_tokens=[EOT])
    with open(input_path, "rb") as f:
        trainer.train(
            f,
            desired_num_chunks=desired_num_chunks,
            split_special_token=EOT.encode("utf-8"),  # str → bytes
        )

    # 2) 用训练产物初始化 Tokenizer,并序列化
    #    注意类型衔接:trainer.special_tokens 是 list[str],Tokenizer 收 list[bytes]
    tok = tokenizer.Tokenizer(
        vocab=trainer.vocab,
        merges=trainer.merges,
        special_tokens=[s.encode("utf-8") for s in trainer.special_tokens],
    )
    tok.save(tokenizer_path)

    # 3) eot_id:反查词表里 EOT 这个字节串对应的 id
    eot_id = tok._vocab_inv[EOT.encode("utf-8")]

    print(f"vocab_size: {len(trainer.vocab)}")
    print(f"tokenizer saved to: {tokenizer_path}")
    print(f"eot_id: {eot_id}")
    return tok, eot_id


if __name__ == "__main__":
    tok, eot_id = train_and_save_tokenizer(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        tokenizer_path="artifacts/tokenizer.json",
    )

    # round-trip 自检:load 回来编码结果一致
    tok2 = tokenizer.Tokenizer.load("artifacts/tokenizer.json")
    sample = "Once upon a time, there was a little fox."
    assert tok.encode(sample) == tok2.encode(sample)
    print("round-trip OK:", tok2.decode(tok2.encode(sample)))
