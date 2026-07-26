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

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_v = x.view(-1, self.in_features)
        # m = self.weight.view(self.out_features, self.in_features)
        # return x @ m.T
        # return x @ self.weight.T
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
