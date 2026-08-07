import torch
import numpy as np
import math
from pathlib import Path

from . import nn_modules, data, optimizers

# ---------------------------------------------------------------- eval


@torch.no_grad()  # 整个函数不建计算图：不存 activation、省一倍内存
def log_val_loss(lm, ce, val_data, cfg, it: int) -> float:
    """在验证集上采 eval_batches 个随机 batch，返回平均 loss 并打印。"""
    losses = []
    for _ in range(cfg.eval_batches):
        x, targets = data.get_batch(
            dataset=val_data,
            batch_size=cfg.batch_size,
            context_length=cfg.context_length,
            device=cfg.device,
        )
        losses.append(ce(logits=lm(x), targets=targets).item())
    val_loss = sum(losses) / len(losses)
    print(f"it {it}  val_loss {val_loss:.4f}  val_ppl {math.exp(val_loss):.1f}", flush=True)
    return val_loss


def train(cfg):
    # init
    lm = nn_modules.TransformerLM(**cfg.model).to(cfg.device)
    ce = nn_modules.CrossEntropy()
    optim = optimizers.AdamW(params=lm.parameters(), **cfg.optim)
    schedule = optimizers.CosinAnnealingLRSchedule(**cfg.schedule)

    # load data
    train_data = np.memmap(cfg.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(cfg.val_path, dtype=np.uint16, mode="r")

    logger = data.ExperimentLogger(cfg=cfg)
    # start training
    start_it = data.load_checkpoint(src=cfg.resume, model=lm, optimizer=optim) if cfg.resume else 0
    for it in range(start_it, cfg.max_iters):
        # batch
        x, targets = data.get_batch(
            dataset=train_data, batch_size=cfg.batch_size, context_length=cfg.context_length, device=cfg.device
        )

        # 1) forward
        loss = ce(logits=lm(x), targets=targets)

        # 2) backward for grad
        optim.zero_grad()
        loss.backward()
        optimizers.gradient_clipping(lm.parameters(), max_l2_norm=cfg.max_l2_norm)

        # 3) update with optimizer
        lr_t = schedule(it=it)
        for group in optim.param_groups:
            group["lr"] = lr_t
        optim.step()

        # 4) eval
        if it % cfg.log_interval == 0:
            logger.log(it, train_loss=loss.item(), lr=lr_t)
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            logger.log(it, val_loss=log_val_loss(lm, ce, val_data, cfg, it))
        if it % cfg.ckpt_interval == 0:
            data.save_checkpoint(lm, optim, it, cfg.ckpt_path)

    data.save_checkpoint(lm, optim, cfg.max_iters, cfg.ckpt_path)  # 收尾:最后一步也留一份


if __name__ == "__main__":
    # 这四步的顺序是有讲究的:
    #   load_config    合并所有覆盖,定 tag、抽 seed —— 之后 cfg 不再变
    #   prepare_run_dir建目录,并在 tag 撞车时当场退出 —— 要死死在建模型之前,
    #                  而不是训练两小时后写 ckpt 时才发现覆盖了别人
    #   seed_everything必须在建模型之前:参数初始化本身就要吃随机数
    #   save_config    快照落在 run 目录里,和它描述的那次 run 待在一起
    cfg = data.load_config()
    run_dir = data.prepare_run_dir(cfg)
    data.seed_everything(cfg.seed)
    data.save_config(cfg, cfg.config_snapshot_path)
    print(f"run: {run_dir}  (seed={cfg.seed}, git={cfg.git_sha})", flush=True)
    train(cfg)
