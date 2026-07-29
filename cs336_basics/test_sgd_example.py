import torch

from cs336_basics import optimizers


def test_sgd_lr(lr: float, weights: torch.nn.Parameter, num_loops: int = 10):
    opt = optimizers.SGD(params=[weights], lr=lr)

    for t in range(num_loops):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"index: {t}, loss: {loss.cpu().item()}")
        loss.backward()
        opt.step()


if __name__ == "__main__":
    t = 5 * torch.randn(10, 10)

    lr = 1e1
    print(f"test with lr: {lr}")
    test_sgd_lr(lr, weights=torch.nn.Parameter(t.clone().detach()))

    lr = 1e2
    print(f"test with lr: {lr}")
    test_sgd_lr(lr, weights=torch.nn.Parameter(t.clone().detach()))

    lr = 1e3
    print(f"test with lr: {lr}")
    test_sgd_lr(lr, weights=torch.nn.Parameter(t.clone().detach()))
