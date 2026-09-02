import triton
import triton.language as tl
from triton_autolib.constants import GELU_COEFF, SQRT_2_OVER_PI


@triton.jit
def sigmoid(x):
    x_fp32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_fp32))
    return sig.to(x.dtype)


@triton.jit
def gelu_tanh(
    x,
    SQRT_2_OVER_PI: tl.constexpr = SQRT_2_OVER_PI,
    GELU_COEFF: tl.constexpr = GELU_COEFF,
):
    x_fp32 = x.to(tl.float32)
    u = SQRT_2_OVER_PI * (x_fp32 + GELU_COEFF * x_fp32 * x_fp32 * x_fp32)
    tanh_u = tl.extra.cuda.libdevice.tanh(u)
    res = 0.5 * x_fp32 * (1.0 + tanh_u)
    return res.to(x.dtype)


@triton.jit
def relu(x):
    return tl.maximum(0.0, x)


@triton.jit
def silu_fwd(x):
    x_fp32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x_fp32))
    res = x_fp32 * sig
    return res.to(x.dtype)


@triton.jit
def elu_fwd(x, alpha: tl.constexpr = 1.0):
    x_fp32 = x.to(tl.float32)
    exp_minus_1 = tl.exp(x_fp32) - 1.0
    res = tl.where(x_fp32 > 0.0, x_fp32, alpha * exp_minus_1)
    return res.to(x.dtype)
