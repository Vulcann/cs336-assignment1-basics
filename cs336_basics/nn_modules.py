import math

from einops import einsum
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
