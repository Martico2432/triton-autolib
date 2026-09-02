import pytest
import torch
import triton
import triton.language as tl

from triton_autolib.backward.activations import (
    d_elu,
    d_gelu_tanh,
    d_relu,
    d_sigmoid,
    d_silu,
)


@triton.jit
def _bwd_test_kernel(
    dy_ptr,
    x_ptr,
    saved_y_ptr,
    dx_ptr,
    n_elements,
    op_type: tl.constexpr,
    alpha: tl.constexpr = 1.0,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dy = tl.load(dy_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    if saved_y_ptr is not None:
        saved_y = tl.load(saved_y_ptr + offsets, mask=mask)
    else:
        saved_y = x

    if op_type == 0:
        dx = d_sigmoid(dy, saved_y)
    elif op_type == 1:
        dx = d_gelu_tanh(dy, x)
    elif op_type == 2:
        dx = d_relu(dy, x)
    elif op_type == 3:
        dx = d_silu(dy, x)
    elif op_type == 4:
        dx = d_elu(dy, x, saved_y, alpha=alpha)

    tl.store(dx_ptr + offsets, dx, mask=mask)


def run_triton_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    op_type: int,
    saved_y: torch.Tensor = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    dx = torch.empty_like(x)
    _bwd_test_kernel[grid](
        dy, x, saved_y, dx, n_elements, op_type=op_type, alpha=alpha
    )
    return dx


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (32, 128), (17, 33)])
def test_backward_activations(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def get_torch_grads(fn, x, dy):
        x_grad = x.clone().detach().requires_grad_(True)
        y = fn(x_grad)
        y.backward(dy)
        return y.detach(), x_grad.grad

    x = torch.randn(shape, device="cuda", dtype=dtype)
    dy = torch.randn(shape, device="cuda", dtype=dtype)

    # d_sigmoid (uses saved_y)
    y_ref, dx_ref = get_torch_grads(torch.sigmoid, x, dy)
    dx_tri = run_triton_bwd(dy, x, op_type=0, saved_y=y_ref)
    torch.testing.assert_close(dx_tri, dx_ref)

    # d_gelu_tanh
    gelu_fn = lambda t: torch.nn.functional.gelu(t, approximate="tanh")
    _, dx_ref = get_torch_grads(gelu_fn, x, dy)
    dx_tri = run_triton_bwd(dy, x, op_type=1)
    torch.testing.assert_close(dx_tri, dx_ref)

    # d_relu
    _, dx_ref = get_torch_grads(torch.relu, x, dy)
    dx_tri = run_triton_bwd(dy, x, op_type=2)
    torch.testing.assert_close(dx_tri, dx_ref)

    # d_silu
    _, dx_ref = get_torch_grads(torch.nn.functional.silu, x, dy)
    dx_tri = run_triton_bwd(dy, x, op_type=3)
    torch.testing.assert_close(dx_tri, dx_ref)

    # d_elu (uses saved_y)
    elu_fn = lambda t: torch.nn.functional.elu(t, alpha=1.0)
    y_ref, dx_ref = get_torch_grads(elu_fn, x, dy)
    dx_tri = run_triton_bwd(dy, x, op_type=4, saved_y=y_ref, alpha=1.0)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(dx_tri, dx_ref, rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(dx_tri, dx_ref)
