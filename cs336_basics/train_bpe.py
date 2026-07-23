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

        self.special_tokens = list(
            set(self.special_tokens + [split_special_token.decode("utf-8")])
        )  # Ensure uniqueness

        boundaries = pretokenization_example.find_chunk_boundaries(f, desired_num_chunks, split_special_token)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Split the chunk into pre-tokens based on special tokens
            split_chunk = tokenizer.Tokenizer.split_on_special_tokens(chunk, self.special_tokens)
            for chunk in split_chunk:
                if chunk in self.special_tokens:
                    continue  # Skip special tokens themselves

                for pre_token in regex.findall(tokenizer.Tokenizer.GPT2_PAT, chunk):
                    pre_token_bytes = pre_token.encode("utf-8")
                    if pre_token_bytes not in pre_tokens:
                        bytes_list = [bytes([b]) for b in pre_token_bytes]
                        pairs_count_local: dict[tuple[bytes, bytes], int] = {}
                        for i in range(len(bytes_list) - 1):
                            pair = (bytes_list[i], bytes_list[i + 1])
                            if pair not in pairs_count_local:
                                pairs_count_local[pair] = 1
                            else:
                                pairs_count_local[pair] += 1
                        pre_tokens[pre_token_bytes] = (1, bytes_list, pairs_count_local)
                    else:
                        count, bytes_list, pair_count = pre_tokens[pre_token_bytes]
                        pre_tokens[pre_token_bytes] = (count + 1, bytes_list, pair_count)

        # init vocab with individual characters
        self.vocab = {i: bytes([i]) for i in range(256)}
        for t in self.special_tokens:
            self.vocab[len(self.vocab)] = t.encode("utf-8")

        # for _ in range(self.vocab_size - len(vocab_list)):
        while self.vocab_size > len(self.vocab):
            pairs_count: dict[tuple[bytes, bytes], int] = {}
            for pre_token_bytes, (count, bytes_list, pairs_count_local) in pre_tokens.items():
                for pair, in_count in pairs_count_local.items():
                    if pair not in pairs_count:
                        pairs_count[pair] = in_count * count
                    else:
                        pairs_count[pair] += in_count * count

            most_frequent_pair = max(pairs_count.items(), key=lambda x: (x[1], x[0]))[0]
            self.merges.append(most_frequent_pair)
            self.vocab[len(self.vocab)] = most_frequent_pair[0] + most_frequent_pair[1]

            # 把 most_frequent_pair 合并进受影响的词
            a, b = most_frequent_pair
            merged = a + b
            for pre_token_bytes, (count, bytes_list, pairs_count_local) in list(pre_tokens.items()):
                if most_frequent_pair not in pairs_count_local:
                    continue
                new_bytes_list = []
                i = 0
                while i < len(bytes_list):
                    if i < len(bytes_list) - 1 and bytes_list[i] == a and bytes_list[i + 1] == b:
                        new_bytes_list.append(merged)
                        i += 2
                    else:
                        new_bytes_list.append(bytes_list[i])
                        i += 1

                pairs_count_local: dict[tuple[bytes, bytes], int] = {}
                for i in range(len(new_bytes_list) - 1):
                    pair = (new_bytes_list[i], new_bytes_list[i + 1])
                    if pair not in pairs_count_local:
                        pairs_count_local[pair] = 1
                    else:
                        pairs_count_local[pair] += 1
                pre_tokens[pre_token_bytes] = (count, new_bytes_list, pairs_count_local)
