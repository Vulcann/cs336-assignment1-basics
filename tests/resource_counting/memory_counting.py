"""CS336 Assignment 1 — Problem (adamw_accounting)
AdamW 训练的峰值内存记账（float32，4 bytes/元素）。

四个桶：
    parameters      : 模型参数
    gradients       : 与参数同形状，1 份
    optimizer state : AdamW 的 m、v，各 1 份 → 2 × parameters
    activations     : forward 产生、需保留到 backward 的中间张量（唯一随 B、L 增长的桶）

峰值时刻：forward 刚结束、backward 未开始。

记号：B = batch_size, L = context_length, d = d_model, h = num_heads,
      d_ff = 8/3·d, V = vocab_size, n_layer = num_layers
"""


def activation_elements(B: int, L: int, d: int, h: int, d_ff: int, V: int, n_layer: int) -> int:
    """按题目简化口径逐操作记账 activation 元素数。

    口径说明：
      - 「activation 记账」列 = 清单中每个操作的输出记一笔（每个输出恰是某下游
        backward 需要的输入，逐输出计数 ≈ 逐需求计数，不重不漏）。
      - 「导数需要」列是严格推导（"带谁存谁"）；标(已记)的张量已在上一行计过。

    == Transformer block 内（每层，×n_layer） ==========================================
    | 操作               | 输出形状      | 导数（概要）            | 导数需要             | 记账      |
    |--------------------|---------------|-------------------------|----------------------|-----------|
    | RMSNorm ×2         | [B,L,d]       | 含 x 与 RMS(x)          | 输入 x               | 2·BLd     |
    | Q/K/V 投影         | 3×[B,L,d]     | matmul: xᵀG / GWᵀ       | 输入(已记) + W       | 3·BLd     |
    | QKᵀ (scores)       | [B,h,L,L]     | matmul                  | Q、K(已记)           | B·h·L²    |
    | softmax            | [B,h,L,L]     | y_i(δ_ij − y_j)         | 自己的输出           | B·h·L²    |
    | S@V (weighted sum) | [B,L,d]       | matmul                  | softmax 输出、V(已记)| BLd       |
    | output 投影 W_O    | [B,L,d]       | matmul                  | 输入(已记) + W       | BLd       |
    | FFN: W₁x (gate)    | [B,L,d_ff]    | matmul                  | 输入(已记) + W₁      | BL·d_ff   |
    | FFN: W₃x (内容)    | [B,L,d_ff]    | matmul                  | 输入(已记) + W₃      | BL·d_ff   |
    | SiLU(W₁x)          | [B,L,d_ff]    | σ(x)+x·σ(x)(1−σ(x))     | 输入(已记)           | BL·d_ff   |
    | ⊙ 逐元素乘         | [B,L,d_ff]    | ∂/∂a=b, ∂/∂b=a          | 两个操作数(已记)     | BL·d_ff   |
    | FFN: W₂(·)         | [B,L,d]       | matmul                  | 输入(已记) + W₂      | BLd       |
    |--------------------|---------------|-------------------------|----------------------|-----------|
    | 每层小计: 8·BLd + 2·B·h·L² + 4·BL·d_ff                                              |

    == Block 外（一次） ================================================================
    | final RMSNorm      | [B,L,d]       | 同上                    | 输入                 | BLd       |
    | lm_head            | [B,L,V]       | matmul                  | 输入(已记) + W       | B·L·V     |
    | cross-entropy      | 标量          | exp 存输出/log 存输入   | exp 输出 [B,L,V]     | B·L·V     |
    |------------------------------------------------------------------------------------|
    | 小计: BLd + 2·B·L·V                                                                |

    细节：QKᵀ 的 scores 严格来说可不存（softmax backward 只要自己的输出），
    简化口径按约定记上（轻微高估）。B·L·V 两项是单笔最大（V ≈ 31×d）。
    """
    per_block = 8 * B * L * d + 2 * B * h * L * L + 4 * B * L * d_ff
    outside = B * L * d + 2 * B * L * V
    return n_layer * per_block + outside


def parameter_elements(d: int, d_ff: int, V: int, n_layer: int) -> int:
    """参数元素数（无 bias；embedding 与 lm_head 不共享；RoPE 无参数）。
    每层: 2 个 RMSNorm(2d) + QKV/O 投影(4d²) + SwiGLU(3·d·d_ff)
    全局: token embedding(V·d) + final RMSNorm(d) + lm_head(d·V)
    """
    per_block = 2 * d + 4 * d * d + 3 * d * d_ff
    return n_layer * per_block + 2 * V * d + d


def peak_memory_bytes(B: int, L: int, d: int, h: int, V: int, n_layer: int) -> dict:
    """四个桶的字节数与总峰值。d_ff 取 8/3·d 就近 64 的倍数。"""
    d_ff = round(8 / 3 * d / 64) * 64
    FP32 = 4
    params = parameter_elements(d, d_ff, V, n_layer) * FP32
    grads = params  # 与参数同形状
    opt_state = 2 * params  # AdamW 的 m、v
    acts = activation_elements(B, L, d, h, d_ff, V, n_layer) * FP32
    return {
        "parameters": params,
        "gradients": grads,
        "optimizer_state": opt_state,
        "activations": acts,
        "total": params + grads + opt_state + acts,
    }


def max_batch_size(budget_bytes: float, L: int, d: int, h: int, V: int, n_layer: int) -> int:
    """total = a·B + b（activations 对 B 严格线性，其余三桶是常数 b），解 B。"""
    m1 = peak_memory_bytes(1, L, d, h, V, n_layer)
    m2 = peak_memory_bytes(2, L, d, h, V, n_layer)
    a = m2["total"] - m1["total"]  # 每加 1 个 batch 的增量 = activations(B=1)
    b = m1["total"] - a  # 三个常数桶
    print(f"total = a·B + b,  a = {a:.4e} bytes/batch, b = {b:.4e} bytes")
    return int((budget_bytes - b) // a)


if __name__ == "__main__":
    # GPT-2 XL 配置
    cfg = dict(L=1024, d=1600, h=25, V=50257, n_layer=48)
    GiB = 1024**3

    for B in (1, 4, 16):
        m = peak_memory_bytes(B=B, **cfg)
        print(f"\nbatch_size = {B}")
        for k, v in m.items():
            print(f"  {k:<16} {v / GiB:8.2f} GiB  ({v / m['total']:6.1%})")

    B_max = max_batch_size(80 * GiB, **cfg)
    print(f"\nmax batch_size within 80 GiB: {B_max}")
