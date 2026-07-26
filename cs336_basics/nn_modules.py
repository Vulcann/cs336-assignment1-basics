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
