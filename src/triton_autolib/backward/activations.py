import triton
import triton.language as tl
from triton_autolib.constants import GELU_COEFF, GELU_DERIV_COEFF, SQRT_2_OVER_PI


@triton.jit
def d_sigmoid(dy, saved_y):
    return dy * saved_y * (1.0 - saved_y)


@triton.jit
def d_gelu_tanh(dy, x,
    SQRT_2_OVER_PI: tl.constexpr = SQRT_2_OVER_PI,
    GELU_COEFF: tl.constexpr = GELU_COEFF,
    GELU_DERIV_COEFF: tl.constexpr = GELU_DERIV_COEFF,
):
    x_fp32 = x.to(tl.float32)
    dy_fp32 = dy.to(tl.float32)

    u = SQRT_2_OVER_PI * (x_fp32 + GELU_COEFF * x_fp32 * x_fp32 * x_fp32)
    tanh_u = tl.extra.cuda.libdevice.tanh(u)
    cdf = 0.5 * (1.0 + tanh_u)
    du_dx = SQRT_2_OVER_PI * (1.0 + GELU_DERIV_COEFF * x_fp32 * x_fp32)
    dcdf_dx = 0.5 * (1.0 - tanh_u * tanh_u) * du_dx
    gelu_prime = cdf + x_fp32 * dcdf_dx

    res = dy_fp32 * gelu_prime
    return res.to(x.dtype)


@triton.jit
def d_relu(dy, x):
    return tl.where(x > 0, dy, 0.0)


@triton.jit
def d_silu(dy, x):
    x_fp32 = x.to(tl.float32)
    dy_fp32 = dy.to(tl.float32)

    sig = 1.0 / (1.0 + tl.exp(-x_fp32))
    silu_prime = sig * (1.0 + x_fp32 * (1.0 - sig))
    res = dy_fp32 * silu_prime

    return res.to(x.dtype)


@triton.jit
def d_elu(dy, x, saved_y, alpha: tl.constexpr = 1.0):
    x_fp32 = x.to(tl.float32)
    dy_fp32 = dy.to(tl.float32)
    saved_y_fp32 = saved_y.to(tl.float32)
    dx_fp32 = tl.where(x_fp32 > 0, dy_fp32, dy_fp32 * (saved_y_fp32 + alpha))
    return dx_fp32.to(x.dtype)
