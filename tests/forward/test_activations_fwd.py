import pytest
import torch
import triton
import triton.language as tl

from triton_autolib.forward.activations import (
    elu_fwd,
    gelu_tanh,
    relu,
    sigmoid,
    silu_fwd,
)


@triton.jit
def _fwd_test_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    op_type: tl.constexpr,
    alpha: tl.constexpr = 1.0,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if op_type == 0:
        y = sigmoid(x)
    elif op_type == 1:
        y = gelu_tanh(x)
    elif op_type == 2:
        y = relu(x)
    elif op_type == 3:
        y = silu_fwd(x)
    elif op_type == 4:
        y = elu_fwd(x, alpha=alpha)

    tl.store(y_ptr + offsets, y, mask=mask)


def run_triton_fwd(x: torch.Tensor, op_type: int, alpha: float = 1.0) -> torch.Tensor:
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    y = torch.empty_like(x)
    _fwd_test_kernel[grid](x, y, n_elements, op_type=op_type, alpha=alpha)
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (32, 128), (17, 33)])
def test_forward_activations(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    x = torch.randn(shape, device="cuda", dtype=dtype)

    # Sigmoid
    y_ref = torch.sigmoid(x)
    y_tri = run_triton_fwd(x, op_type=0)
    torch.testing.assert_close(y_tri, y_ref)

    # GELU Tanh Approximation
    y_ref = torch.nn.functional.gelu(x, approximate="tanh")
    y_tri = run_triton_fwd(x, op_type=1)
    torch.testing.assert_close(y_tri, y_ref)

    # ReLU
    y_ref = torch.relu(x)
    y_tri = run_triton_fwd(x, op_type=2)
    torch.testing.assert_close(y_tri, y_ref)

    # SiLU
    y_ref = torch.nn.functional.silu(x)
    y_tri = run_triton_fwd(x, op_type=3)
    torch.testing.assert_close(y_tri, y_ref)

    # ELU
    y_ref = torch.nn.functional.elu(x, alpha=1.0)
    y_tri = run_triton_fwd(x, op_type=4, alpha=1.0)
    torch.testing.assert_close(y_tri, y_ref)
