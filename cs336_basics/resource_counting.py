def tf_block_parameters_size(d_model: int, d_ff: int):
    count = 0
    # attn
    count += d_model
    count += 4 * d_model * d_model
    # swiglu
    count += d_model
    count += 3 * d_model * d_ff
    return count


def tf_lm_parameters_size(d_model: int, d_ff: int, vocab_size: int, num_layers: int):
    count = 0
    # embedding:
    count += d_model * vocab_size
    # blocks
    count += num_layers * tf_block_parameters_size(d_model=d_model, d_ff=d_ff)
    count += d_model
    # lm_head
    count += d_model * vocab_size
    return count


def tf_lm_FLOPs(
    d_model: int,
    d_ff: int,
    vocab_size: int,
    context_length: int,
    num_layers: int,
):
    # for matmal and its FLOPs
    num_matmul = 0

    # for each blocks:
    # 2 matmal and
    # 4 Linear(matmal)
    num_matmul += 6
    # for each swiGlu:
    # 3 Linear
    num_matmul += 3
    # {num_layers} blocks:
    num_matmul *= num_layers
    # total plus im_head
    num_matmul += 1

    # FLOPs for each matmal in attn
    # S = Q @ K^T:
    # S @ V
    # Q,K,V: [... context_length, d_model]
    # S: [... context_length context_length]
    FLOPs_attn_QK = 2 * context_length * context_length * d_model
    FLOPs_attn_SV = 2 * context_length * context_length * d_model
    # Linear
    FLOPs_attn_ln = 2 * context_length * d_model * d_model
    FLOPs_attn = 4 * FLOPs_attn_ln + FLOPs_attn_QK + FLOPs_attn_SV

    # FLOPs in swiGLU
    FLOPs_swiglu_w1 = 2 * context_length * d_model * d_ff
    FLOPs_swiglu_w3 = 2 * context_length * d_model * d_ff
    FLOPs_swiglu_w2 = 2 * context_length * d_ff * d_model
    FLOPs_swiglu = FLOPs_swiglu_w1 + FLOPs_swiglu_w2 + FLOPs_swiglu_w3

    FLOPs_block = FLOPs_attn + FLOPs_swiglu
    FLOPs = FLOPs_block * num_layers

    # FLOPs in lm_head
    FLOPs_lm_head = 2 * context_length * d_model * vocab_size
    FLOPs += FLOPs_lm_head
    return {
        "num_matmul": num_matmul,
        "FLOPs": FLOPs,
        "breakdown": {
            "attn_proj": num_layers * 4 * FLOPs_attn_ln,
            "attn_qk_sv": num_layers * (FLOPs_attn_QK + FLOPs_attn_SV),
            "swiglu": num_layers * FLOPs_swiglu,
            "lm_head": FLOPs_lm_head,
        },
    }


def sci(obj, digits=3):
    """递归地把数字转成科学计数法字符串"""
    if isinstance(obj, dict):
        return {k: sci(v, digits) for k, v in obj.items()}
    if isinstance(obj, (int, float)) and abs(obj) >= 1e6:
        return f"{obj:.{digits}e}"
    return obj


def print_flops(result):
    total = result["FLOPs"]
    print(f"num_matmul: {result['num_matmul']}")
    print(f"FLOPs total: {total:.4e}")
    for name, v in result["breakdown"].items():
        print(f"  {name:<12} {v:.4e}  {v / total:7.2%}")


if __name__ == "__main__":

    def print_config_gpt2_xl():
        vocab_size = 50257
        context_length = 1024
        num_layers = 48
        d_model = 1600
        num_heads = 25
        d_ff = round(d_model * 8.0 / 3.0 / 64) * 64

        print("GPT-2 XL:")
        print(f"d_ff: {d_ff}")
        parameters_size = tf_lm_parameters_size(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, num_layers=num_layers
        )
        print(f"parameters_size: {parameters_size}")
        result = tf_lm_FLOPs(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, context_length=context_length, num_layers=num_layers
        )
        print_flops(result)

    def print_config_gpt2_small():
        vocab_size = 50257
        context_length = 1024
        num_layers = 12
        d_model = 768
        num_heads = 12
        d_ff = round(d_model * 8.0 / 3.0 / 64) * 64

        print("GPT-2 small:")
        print(f"d_ff: {d_ff}")
        parameters_size = tf_lm_parameters_size(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, num_layers=num_layers
        )
        print(f"parameters_size: {parameters_size}")
        result = tf_lm_FLOPs(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, context_length=context_length, num_layers=num_layers
        )
        print_flops(result)

    def print_config_gpt2_medium():
        vocab_size = 50257
        context_length = 1024
        num_layers = 24
        d_model = 1024
        num_heads = 16
        d_ff = round(d_model * 8.0 / 3.0 / 64) * 64

        print("GPT-2 medium:")
        print(f"d_ff: {d_ff}")
        parameters_size = tf_lm_parameters_size(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, num_layers=num_layers
        )
        print(f"parameters_size: {parameters_size}")
        result = tf_lm_FLOPs(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, context_length=context_length, num_layers=num_layers
        )
        print_flops(result)

    def print_config_gpt2_large():
        vocab_size = 50257
        context_length = 1024
        num_layers = 36
        d_model = 1280
        num_heads = 20
        d_ff = round(d_model * 8.0 / 3.0 / 64) * 64

        print("GPT-2 large:")
        print(f"d_ff: {d_ff}")
        parameters_size = tf_lm_parameters_size(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, num_layers=num_layers
        )
        print(f"parameters_size: {parameters_size}")
        result = tf_lm_FLOPs(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, context_length=context_length, num_layers=num_layers
        )
        print_flops(result)

    def print_config_gpt2_xl_increase_context_length():
        vocab_size = 50257
        context_length = 16384
        num_layers = 48
        d_model = 1600
        num_heads = 25
        d_ff = round(d_model * 8.0 / 3.0 / 64) * 64

        print("GPT-2 XL increase_context_length(1024->16384):")
        print(f"d_ff: {d_ff}")
        parameters_size = tf_lm_parameters_size(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, num_layers=num_layers
        )
        print(f"parameters_size: {parameters_size}")
        result = tf_lm_FLOPs(
            d_model=d_model, d_ff=d_ff, vocab_size=vocab_size, context_length=context_length, num_layers=num_layers
        )
        print_flops(result)

    print_config_gpt2_xl()
    print_config_gpt2_small()
    print_config_gpt2_medium()
    print_config_gpt2_large()
    print_config_gpt2_xl_increase_context_length()
