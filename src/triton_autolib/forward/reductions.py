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

@triton.jit
def logsumexp_combine(m1, l1, m2, l2):
    """
    Merges (max, sum_exp) states into one, e.g. for
    accumulating a numerically-stable row max/sum-exp across blocks.
    l1, l2 are sums of exp(x - m) over their respective partitions.
    """
    m1_fp32 = m1.to(tl.float32)
    l1_fp32 = l1.to(tl.float32)
    m2_fp32 = m2.to(tl.float32)
    l2_fp32 = l2.to(tl.float32)

    m = tl.maximum(m1_fp32, m2_fp32)
    l = l1_fp32 * tl.exp(m1_fp32 - m) + l2_fp32 * tl.exp(m2_fp32 - m)
    return m.to(m1.dtype), l.to(l1.dtype)


@triton.jit
def welford_combine(count1, mean1, m2_1, count2, mean2, m2_2):
    """
    Merges two partial Welford (count, mean, M2) states into one.
    M2 is the running sum of squared deviations from the mean
    (i.e. variance = M2 / count, not yet divided).
    """
    count1_fp32 = count1.to(tl.float32)
    mean1_fp32 = mean1.to(tl.float32)
    m2_1_fp32 = m2_1.to(tl.float32)
    count2_fp32 = count2.to(tl.float32)
    mean2_fp32 = mean2.to(tl.float32)
    m2_2_fp32 = m2_2.to(tl.float32)

    count = count1_fp32 + count2_fp32
    delta = mean2_fp32 - mean1_fp32
    mean = mean1_fp32 + delta * count2_fp32 / count
    m2 = m2_1_fp32 + m2_2_fp32 + delta * delta * count1_fp32 * count2_fp32 / count

    return count.to(count1.dtype), mean.to(mean1.dtype), m2.to(m2_1.dtype)
