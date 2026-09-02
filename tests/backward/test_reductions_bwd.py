import pytest
import torch
import triton
import triton.language as tl

from triton_autolib.backward.reductions import d_softmax, d_softmax_cross_entropy


@triton.jit
def _softmax_bwd_kernel(
    Y_ptr, DY_ptr, DX_ptr,
    stride_row,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    row_offset = row_idx * stride_row
    y = tl.load(Y_ptr + row_offset + col_offsets, mask=mask, other=0.0)
    dy = tl.load(DY_ptr + row_offset + col_offsets, mask=mask, other=0.0)

    y_fp32 = y.to(tl.float32)
    dy_fp32 = dy.to(tl.float32)
    row_sum_dy_y = tl.sum(dy_fp32 * y_fp32, axis=0)

    dx = d_softmax(dy_fp32, y_fp32, row_sum_dy_y)
    tl.store(DX_ptr + row_offset + col_offsets, dx.to(y.dtype), mask=mask)


@triton.jit
def _cross_entropy_bwd_kernel(
    Y_ptr, Target_ptr, DX_ptr,
    stride_x_row,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    row_offset = row_idx * stride_x_row
    y = tl.load(Y_ptr + row_offset + col_offsets, mask=mask, other=0.0)
    target = tl.load(Target_ptr + row_idx)
    is_target_mask = col_offsets == target

    dx = d_softmax_cross_entropy(y.to(tl.float32), target, is_target_mask)
    tl.store(DX_ptr + row_offset + col_offsets, dx.to(y.dtype), mask=mask)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 128), (32, 512), (4, 1024)])
def test_softmax_bwd(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    n_rows, n_cols = shape
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(shape, device="cuda", dtype=dtype)

    y_ref = torch.softmax(x, dim=-1)
    y_ref.backward(dy)
    dx_ref = x.grad.clone()

    dx_tri = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_bwd_kernel[grid](
        y_ref.detach(), dy, dx_tri,
        x.stride(0),
        N_COLS=n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
    )

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(dx_tri, dx_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 128), (32, 512), (4, 1024)])
def test_softmax_cross_entropy_bwd(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    n_rows, n_cols = shape
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    target = torch.randint(0, n_cols, (n_rows,), device="cuda", dtype=torch.int64)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(x, target)
    loss.sum().backward()
    dx_ref = x.grad.clone()

    y_probs = torch.softmax(x, dim=-1).detach()
    dx_tri = torch.empty_like(x)

    grid = (n_rows,)
    _cross_entropy_bwd_kernel[grid](
        y_probs, target, dx_tri,
        x.stride(0),
        N_COLS=n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
    )

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(dx_tri, dx_ref, atol=atol, rtol=rtol)
