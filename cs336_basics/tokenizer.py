import re
import regex
import base64
import json

from pathlib import Path
from collections.abc import Iterable, Iterator


class Tokenizer:
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[bytes]):
        self._vocab = vocab
        self._special_tokens = special_tokens or []
        self._merges = {p: i for i, p in enumerate(merges)}
        self._vocab_inv = {v: k for k, v in self._vocab.items()}

    # ---------------------------------------------------------- serialization

    def save(self, path: str | Path) -> None:
        """存成单个 JSON 文件。bytes 一律 base64 编码（JSON 不支持二进制）。"""
        payload = {
            "vocab": {str(idx): base64.b64encode(tok).decode("ascii") for idx, tok in self._vocab.items()},
            # _merges 是 pair→rank 的 dict，按 rank 排序还原成有序列表——顺序就是合并优先级，不能丢
            "merges": [
                [base64.b64encode(a).decode("ascii"), base64.b64encode(b).decode("ascii")]
                for (a, b), _ in sorted(self._merges.items(), key=lambda kv: kv[1])
            ],
            "special_tokens": [base64.b64encode(t).decode("ascii") for t in self._special_tokens],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """从 save() 的文件重建 Tokenizer。"""
        payload = json.loads(Path(path).read_text())
        vocab = {int(idx): base64.b64decode(tok) for idx, tok in payload["vocab"].items()}
        merges = [(base64.b64decode(a), base64.b64decode(b)) for a, b in payload["merges"]]
        special = [base64.b64decode(t) for t in payload["special_tokens"]]
        return cls(vocab=vocab, merges=merges, special_tokens=special)

    # ... encode / decode 等原有方法 ...

    @staticmethod
    def find_matches(bytes_list: list[bytes], special_tokens: list[bytes]) -> list[tuple[int, int]]:
        return [(i, j) for i, p in enumerate(bytes_list) for j, s in enumerate(special_tokens) if p == s]

    def decode(self, ids: list[int]) -> str:
        """Decode the input tokens into a string."""
        bytes_list = [self._vocab[id] for id in ids]

        matches = Tokenizer.find_matches(bytes_list, self._special_tokens)
        match_list_idx, match_special_idx = map(list, zip(*matches) if matches else ([], []))

        boundaries = {x for i in match_list_idx for x in (i, i + 1)}
        list_idx_2 = sorted(boundaries | {0, len(bytes_list)})
        intervals = zip(list_idx_2[:-1], list_idx_2[1:])
        str_list = []
        for start, end in intervals:
            if start in match_list_idx:
                str_list.append(
                    self._special_tokens[match_special_idx[match_list_idx.index(start)]].decode(
                        "utf-8", errors="replace"
                    )
                )
            else:
                str_list.append(b"".join(bytes_list[start:end]).decode("utf-8", errors="replace"))
        return "".join(str_list)
        # 特殊 token(如 <|endoftext|>)本身就是合法的 UTF-8 字节串,直接和普通 token 一起拼接、一次性解码,结果完全一样。整个函数可以简化为标准写法:
        # return b"".join(self._vocab[id] for id in ids).decode("utf-8", errors="replace")

    # staticmethod
    # def byte_pairs(data: bytes) -> list[tuple[bytes, bytes]]:
    #     from itertools import pairwise
    #     return list(pairwise(data[i : i + 1] for i in range(len(data))))

    @staticmethod
    def split_on_special_tokens(text: str, special_tokens: list[str] | None) -> list[str]:
        """按任意特殊 token 切分文本，保留分隔符本身。
        返回的列表中，特殊 token 作为独立元素出现，其余是普通文本片段。
        special_tokens 为空或 None 时直接原样返回（no-op）。
        """

        if not special_tokens:
            return [text]
        # 长者优先：防止 "<|eot|>" 抢先截断 "<|eot|><|eot|>" 这类重叠 token
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        # re.escape 处理 | < > 等元字符；外层括号形成捕获组 → split 保留分隔符
        pattern = "(" + "|".join(re.escape(t) for t in sorted_tokens) + ")"
        # 相邻分隔符之间、以及开头/结尾处 split 会产生空串 ''，过滤掉
        return [piece for piece in re.split(pattern, text) if piece]

    def encode(self, text: str) -> list[int]:
        """Encode the input text into a list of token IDs."""
        special_tokens_in_str = [token.decode("utf-8") for token in self._special_tokens]
        split_text = Tokenizer.split_on_special_tokens(text, special_tokens_in_str)

        ids = []
        for piece in split_text:
            if piece in special_tokens_in_str:
                bytes_piece = piece.encode("utf-8")
                bytes_id = self._vocab_inv.get(bytes_piece)
                if bytes_id is None:
                    raise ValueError(f"Special token {piece} not found in vocabulary.")
                ids.append(bytes_id)
            else:
                ids.extend(self._encode_piece(piece))
        return ids

    def _encode_piece(self, text: str) -> list[int]:
        """Encode a piece of text that does not contain any special tokens."""
        ids = []
        for pre_token in regex.findall(Tokenizer.GPT2_PAT, text):
            pre_token_bytes = pre_token.encode("utf-8")
            ids.extend(self._merge_word(pre_token_bytes))
        return ids

    def _merge_word(self, text: bytes) -> list[int]:
        def find_first_merge(base_bytes, merges):
            base_pairs = list(zip(range(len(base_bytes) - 1), range(1, len(base_bytes))))
            pairs_with_index = []
            for i, j in base_pairs:
                index = merges.get((base_bytes[i], base_bytes[j]), None)
                if index is not None:
                    pairs_with_index.append((index, i))

            if len(pairs_with_index) == 0:
                return None, None
            i = min(pairs_with_index, key=lambda x: x[0])[1]
            return (i, i + 1), (base_bytes[i], base_bytes[i + 1])
            # for pair in merges:
            #     for i, j in base_pairs:
            #         if base_bytes[i] == pair[0] and base_bytes[j] == pair[1]:
            #             return (i, j), pair
            # return None, None

        base_bytes = list(text[i : i + 1] for i in range(len(text)))
        # find the first occurrence of pair in self._merges in the base pairs
        while True:
            first_merge_indices, first_merge_pair = find_first_merge(base_bytes, self._merges)
            if first_merge_indices is None:
                break
            i, j = first_merge_indices
            merged_token = b"".join(first_merge_pair)
            base_bytes = base_bytes[:i] + [merged_token] + base_bytes[j + 1 :]
        return [self._vocab_inv[bs] for bs in base_bytes]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings into an iterator of token IDs."""
        for text in iterable:
            yield from self.encode(text)

    def token_to_id(self, token: bytes) -> int:
        """按字节串反查 token id;不存在则 KeyError。"""
        return self._vocab_inv[token]


if __name__ == "__main__":
    # ------------------------ 使用样例 ------------------------
    # 1) 假设训练 BPE 后得到了 vocab / merges（这里用微型示意数据）
    vocab = {0: b"<|endoftext|>", 1: b"a", 2: b"b", 3: b"ab"}
    merges = [(b"a", b"b")]
    tok = Tokenizer(vocab=vocab, merges=merges, special_tokens=[b"<|endoftext|>"])

    # 2) 保存
    tok.save("artifacts/tokenizer.json")

    # 3) 加载（另一个进程/训练脚本里）
    tok2 = Tokenizer.load("artifacts/tokenizer.json")

    # 4) 验证 round-trip 完整性：内部状态逐项一致
    assert tok2._vocab == tok._vocab
    assert tok2._merges == tok._merges
    assert tok2._special_tokens == tok._special_tokens
    # 更实际的验证：编码结果一致
    # assert tok.encode("ab") == tok2.encode("ab")
    print("save/load round-trip OK:", Path("artifacts/tokenizer.json").stat().st_size, "bytes")
