import triton
import triton.language as tl


@triton.jit
def softmax(x, row_max, row_sum_exp):
    """
    x: unnormalized logits
    row_max: max value per row/block
    row_sum_exp: sum of exp(x - row_max) per row/block
    """
    x_fp32 = x.to(tl.float32)
    exp_x = tl.exp(x_fp32 - row_max)
    y = exp_x / row_sum_exp
    return y.to(x.dtype)


@triton.jit
def softmax_cross_entropy(logits, target_idx, row_max, row_sum_exp, is_target_mask):
    """
    logits: raw logit values
    target_idx: target class index
    row_max: max value per row/block
    row_sum_exp: sum of exp(logits - row_max) per row/block
    is_target_mask: boolean mask indicating if current element matches target index
    """
    logits_fp32 = logits.to(tl.float32)
    log_softmax = (logits_fp32 - row_max) - tl.log(row_sum_exp)
    loss = tl.where(is_target_mask, -log_softmax, 0.0)
    return loss
