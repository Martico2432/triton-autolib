import triton
import triton.language as tl

@triton.jit
def d_softmax(dy, y, row_sum_dy_y):
    return y * (dy - row_sum_dy_y)

@triton.jit
def d_softmax_cross_entropy(logits_softmax, target_idx, is_target_mask):
    grad = logits_softmax - tl.where(is_target_mask, 1.0, 0.0)
    return grad

@triton.jit
def d_logsumexp_combine(dm, dl, m1, l1, m2, l2):
    """
    Backward for logsumexp_combine.
    Returns (dm1, dl1, dm2, dl2).
    """
    dm_fp32 = dm.to(tl.float32)
    dl_fp32 = dl.to(tl.float32)
    m1_fp32 = m1.to(tl.float32)
    l1_fp32 = l1.to(tl.float32)
    m2_fp32 = m2.to(tl.float32)
    l2_fp32 = l2.to(tl.float32)

    m = tl.maximum(m1_fp32, m2_fp32)
    w1 = tl.exp(m1_fp32 - m)
    w2 = tl.exp(m2_fp32 - m)
    is_m1 = tl.where(m1_fp32 >= m2_fp32, 1.0, 0.0)
    is_m2 = 1.0 - is_m1

    g = w1 * l1_fp32 * is_m2 - w2 * l2_fp32 * is_m1

    dm1 = dm_fp32 * is_m1 + dl_fp32 * g
    dm2 = dm_fp32 * is_m2 - dl_fp32 * g
    dl1 = dl_fp32 * w1
    dl2 = dl_fp32 * w2

    return dm1.to(m1.dtype), dl1.to(l1.dtype), dm2.to(m2.dtype), dl2.to(l2.dtype)


@triton.jit
def d_welford_combine(d_mean, d_m2, count1, mean1, count2, mean2):
    """
    Backward for welford_combine.
    Returns (d_mean1, d_m2_1, d_mean2, d_m2_2).
    """
    d_mean_fp32 = d_mean.to(tl.float32)
    d_m2_fp32 = d_m2.to(tl.float32)
    count1_fp32 = count1.to(tl.float32)
    mean1_fp32 = mean1.to(tl.float32)
    count2_fp32 = count2.to(tl.float32)
    mean2_fp32 = mean2.to(tl.float32)

    count = count1_fp32 + count2_fp32
    delta = mean2_fp32 - mean1_fp32
    cross = count1_fp32 * count2_fp32 / count

    d_mean1 = d_mean_fp32 * (count1_fp32 / count) - d_m2_fp32 * 2.0 * delta * cross
    d_mean2 = d_mean_fp32 * (count2_fp32 / count) + d_m2_fp32 * 2.0 * delta * cross

    return (
        d_mean1.to(mean1.dtype),
        d_m2_fp32.to(mean1.dtype),
        d_mean2.to(mean2.dtype),
        d_m2_fp32.to(mean2.dtype),
    )
