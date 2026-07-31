import torch

from . import data, nn_modules


def top_p_filter(probs, p):
    # probs: [B, vocab]
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)  # 各 [B, vocab]
    cum = torch.cumsum(sorted_probs, dim=-1)  # [B, vocab] 累计概率

    # 剔除条件:在"我之前"累计已达 p 的词。减去自身保证核至少含 1 个词、
    # 且"恰好越过 p 的那个词"被保留(边界约定)
    remove = (cum - sorted_probs) >= p  # [B, vocab] bool

    sorted_probs[remove] = 0.0
    # 把排序域的结果撒回原 token id 位置:scatter 是 sort 的逆操作
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)


class Decoder:
    def __init__(
        self,
        lm: nn_modules.TransformerLM,
        temperature: float,
        sampling_p: float,
        eot_id: int,
        max_seq_len: int,
        device: str,
    ):
        self.lm = lm
        self.temperature = temperature
        self.sampling_p = sampling_p
        self.device = device
        self.eot_id = eot_id
        self.max_seq_len = max_seq_len
        self.softmax = nn_modules.Softmax()

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        prompt_ids = prompt_ids.to(self.device)
        x = prompt_ids.unsqueeze(0)  # [1, t]
        for _ in range(max_new_tokens):
            logits = self.lm(x[:, -self.max_seq_len :])
            logits = logits[:, -1, :]
            if self.temperature == 0: 
                next_id = logits.argmax(-1, keepdim=True)
            else:
                dist = self.softmax(logits / self.temperature, dim=-1)
                dist = top_p_filter(dist, self.sampling_p)
                next_id = torch.multinomial(dist, num_samples=1)

            x = torch.cat([x, next_id], dim=-1)
            if next_id.item() == self.eot_id:
                break
        return x.squeeze(0)


if __name__ == "__main__":
    cfg = data.load_config()
    # data.save_config(cfg, Path(cfg.ckpt_path).with_suffix(".config.json"))

    lm = nn_modules.TransformerLM(**cfg.model).to(cfg.device)
    # load model
    _ = data.load_checkpoint(src=cfg.ckpt_path, model=lm, optimizer=None)
    lm.eval()
    decoder = Decoder(
        lm=lm,
        temperature=cfg.decode["temperature"],
        sampling_p=cfg.decode["sampling_p"],
        eot_id=cfg.decode["eot_id"],
        max_seq_len=cfg.model["max_seq_len"],  # 引用 model 组,单一事实来源
        device=cfg.device,
    )

    # load data
    # train_data = np.memmap(cfg.train_path, dtype=np.uint16, mode="r")
    # val_data = np.memmap(cfg.val_path, dtype=np.uint16, mode="r")

    prompt_ids = torch.randint(0, cfg.model["vocab_size"], (16,), dtype=torch.long)
    out = decoder.generate(prompt_ids, max_new_tokens=cfg.decode["max_new_tokens"])
    print(f"prompt_ids: {prompt_ids}, out: {out}")

    # tokenizer = Tokenizer.from_files(cfg.vocab_path, cfg.merges_path, special_tokens=["<|endoftext|>"])
    # prompt_ids = torch.tensor(tokenizer.encode("Once upon a time"), dtype=torch.long)
    # out = decoder.generate(prompt_ids, max_new_tokens=cfg.decode["max_new_tokens"])
    # print(tokenizer.decode(out.tolist()))  # 打印文本,而不是 id 串
