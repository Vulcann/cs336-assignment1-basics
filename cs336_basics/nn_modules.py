import math

from einops import einsum, rearrange
import torch
from torch import nn


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=True
        )

        std = math.sqrt(2.0 / (out_features + in_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_v = x.view(-1, self.in_features)
        # m = self.weight.view(self.out_features, self.in_features)
        # return x @ m.T
        # return x @ self.weight.T
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), requires_grad=True
        )

        std = 1.0
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor):
        # (batch_size, sequence_length, embedding_dim)
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad=True)

    def forward(self, x: torch.Tensor):
        intype = x.dtype
        x = x.to(dtype=torch.float32)

        sq = x.square()
        ms = sq.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        result = x / rms * self.weight

        return result.to(dtype=intype)


class SiLU(torch.nn.Module):
    def __init__(self, device=None, dtype=None):
        super(SiLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.silu = SiLU()

        self.d_model = d_model
        self.d_ff = d_ff
        # self.d_ff = round(self.d_model * 8.0 / 3.0 / 64) * 64

        # self.weight_w1 = nn.Parameter(
        #     torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype), requires_grad=True
        # )
        # self.weight_w3 = nn.Parameter(
        #     torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype), requires_grad=True
        # )
        # self.weight_w2 = nn.Parameter(
        #     torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype), requires_grad=True
        # )
        # std = math.sqrt(2.0 / (self.d_ff + self.d_model))
        # nn.init.trunc_normal_(self.weight_w1, mean=0.0, std=std, a=-3 * std, b=3 * std)
        # nn.init.trunc_normal_(self.weight_w2, mean=0.0, std=std, a=-3 * std, b=3 * std)
        # nn.init.trunc_normal_(self.weight_w3, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w1x = einsum(self.weight_w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        # w3x = einsum(self.weight_w3, x, "d_ff d_model, ... d_model -> ... d_ff")

        # silu = w1x * torch.sigmoid(w1x)
        # silu_w3x = silu * w3x
        # return einsum(self.weight_w2, silu_w3x, "d_model d_ff, ... d_ff -> ... d_model")
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RoPE(torch.nn.Module):
    def __init__(self, d_model: int, max_seq_length: int, theta: float, device=None, dtype=None):
        super(RoPE, self).__init__()

        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.theta = theta

        k = torch.arange(0, d_model, 2, dtype=torch.float32, device=device) / d_model
        inv_freq = self.theta ** (-k)
        pos = torch.arange(0, self.max_seq_length, dtype=torch.float32, device=device)
        angles = torch.outer(pos, inv_freq)
        # # 写法1: reshape 显式指定形状(你要的方法)
        # angles = pos.reshape(-1, 1) * inv_freq.reshape(1, -1)     # (L,1) * (1,d/2) -> (L,d/2)
        # # 写法2: None 索引 —— 最常见,本质就是 reshape 加一个长度1的维
        # angles = pos[:, None] * inv_freq[None, :]
        # # 写法3: unsqueeze —— PyTorch 风格,语义最清楚
        # angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0)         # 在 dim1/dim0 各插一维
        # # 写法4：
        # angles = torch.einsum("i,j->ij", pos, inv_freq)

        # use registered_buffer to save max_seq_length * d_model/2 rotation angles (not learnable),
        # or two tables of length max_seq_length * d_model/2, one for sin and one for cos
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: "batch_size, seq_length, d_model"
        positions: "batch_size, seq_length"

        docstring写 x: (batch, seq, d_model),但这不是 RoPE 在注意力里实际收到的形状。
        RoPE 必须逐头作用在 query/key 上——在多头注意力里,q/k 早已 reshape 成 (batch, heads, seq, d_k),
        RoPE 收到的 x 是这个四维张量,于是 x1 = x[..., 0::2] 是 (batch, heads, seq, d/2)
        """

        # use indexing to get the block matrix (2x2) for each position i
        cos = self.cos[positions]
        sin = self.sin[positions]

        # construct a matrix "d_model d_model" from d_model/2 blocks
        x1 = x[..., 0::2]  # 偶数维 [..., seq_len, d/2]
        x2 = x[..., 1::2]  # 奇数维

        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class Softmax(nn.Module):
    def __init__(self, device=None, dtype=None):
        super(Softmax, self).__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """balance logits and create the normalized probability distribution

        Args:
            x (torch.Tensor): "..."
            int: the dimention which softmax applies on

        Returns:
            torch.Tensor: "..."
        """

        x_dim = x - torch.max(x, dim, keepdim=True).values
        x_dim_exp = torch.exp(x_dim)
        return x_dim_exp / torch.sum(x_dim_exp, dim, keepdim=True)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, device=None, dtype=None):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = Softmax()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... keys d_v"]): Values tensor
            mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        """

        d_k = Q.shape[-1]
        scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
        scores = scores + torch.where(mask, 0.0, float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = self.softmax(scores, dim=-1)
        return einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float | None = None):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
            k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
            v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
            o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        Returns:
        """

        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_k = self.d_model // self.num_heads
        self.attn = ScaledDotProductAttention()

        if theta is not None:
            self.rope = RoPE(d_model=self.d_k, max_seq_length=max_seq_len, theta=theta)

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        '''
        Args:
            in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
            implementation with the given QKV projection weights and input features.
        """
        '''
        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))

        # 投影后拆头：[..., seq, d_model] -> [..., heads, seq, d_k]
        Q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)

        if positions is not None:
            Q = self.rope(Q, positions)
            K = self.rope(K, positions)

        out = self.attn(Q=Q, K=K, V=V, mask=mask)  # 一次调用，heads 随 ... 广播
        out = rearrange(out, "... h s d -> ... s (h d)")  # 合头
        return self.output_proj(out)
        # Q = self.q_proj(x)
        # K = self.k_proj(x)
        # V = self.v_proj(x)
        # attn = torch.zeros_like(x)
        # for i in range(0, self.d_model, self.d_k):
        #     Q_i = Q[..., i : i + self.d_k]
        #     K_i = K[..., i : i + self.d_k]
        #     V_i = V[..., i : i + self.d_k]
        #     if positions is not None:
        #         Q_i = self.rope(Q_i, positions)
        #         K_i = self.rope(K_i, positions)
        #     attn[..., i : i + self.d_k] = self.attn(Q=Q_i, K=K_i, V=V_i, mask=mask)
        # return self.output_proj(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, d_ff: int, theta: float | None = None):
        super(TransformerBlock, self).__init__()

        self.attn = MultiheadSelfAttention(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        ln1 = self.ln1(x)
        attn = self.attn(ln1, positions)
        x = x + attn

        ln2 = self.ln2(x)
        result = self.ffn(ln2)
        return x + result


class TransformerLM(nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        d_ff: int,
        theta: int | None = None,
    ):
        super(TransformerLM, self).__init__()

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.num_layers = num_layers
        self.transformer_blocks = []
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, d_ff=d_ff, theta=theta)
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-1]
        positions = torch.arange(seq_len, dtype=torch.int, device=x.device)

        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, positions)

        x = self.ln_final(x)
        return self.lm_head(x)
