from collections.abc import Iterable, Iterator
import re
import regex


class Tokenizer:
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[bytes]):
        self._vocab = vocab
        self._special_tokens = special_tokens or []
        self._merges = {p: i for i, p in enumerate(merges)}
        self._vocab_inv = {v: k for k, v in self._vocab.items()}

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
