import triton
import triton.language as tl

@triton.jit
def d_softmax(dy, y, row_sum_dy_y):
    return y * (dy - row_sum_dy_y)

@triton.jit
def d_softmax_cross_entropy(logits_softmax, target_idx, is_target_mask):
    grad = logits_softmax - tl.where(is_target_mask, 1.0, 0.0)
    return grad
