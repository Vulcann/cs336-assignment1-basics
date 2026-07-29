from collections.abc import Callable, Iterable
import math
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params: list[torch.nn.Parameter], lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params: list[torch.nn.Parameter], lr: float, weight_decay: float, betas: tuple[float, float], eps: float
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params=params, defaults=defaults)

        pass

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                grad = p.grad.data
                lr_t = math.sqrt(1 - math.pow(b2, t)) / (1 - math.pow(b1, t)) * lr
                p.data -= lr * weight_decay * p.data
                m = m * b1 + (1 - b1) * grad
                v = v * b2 + (1 - b2) * grad * grad
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def cosine_annealing_lr_step(
    it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    # warm-up
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """

    params = list(parameters)
    qs: float = 0
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad.data
        qs += (grad * grad).sum()
    l2_norm = math.sqrt(qs)
    if l2_norm > max_l2_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data *= max_l2_norm / (l2_norm + eps)
