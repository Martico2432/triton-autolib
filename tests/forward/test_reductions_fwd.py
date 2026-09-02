import pytest
import torch
import triton
import triton.language as tl

from triton_autolib.forward.reductions import softmax, softmax_cross_entropy


@triton.jit
def _softmax_fwd_kernel(
    X_ptr, Y_ptr,
    stride_row,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    row_offset = row_idx * stride_row
    x = tl.load(X_ptr + row_offset + col_offsets, mask=mask, other=-float("inf"))

    x_fp32 = x.to(tl.float32)
    row_max = tl.max(x_fp32, axis=0)
    row_sum_exp = tl.sum(tl.exp(x_fp32 - row_max), axis=0)

    y = softmax(x, row_max, row_sum_exp)
    tl.store(Y_ptr + row_offset + col_offsets, y, mask=mask)


@triton.jit
def _cross_entropy_fwd_kernel(
    X_ptr, Target_ptr, Loss_ptr,
    stride_x_row,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    row_offset = row_idx * stride_x_row
    x = tl.load(X_ptr + row_offset + col_offsets, mask=mask, other=-float("inf"))
    target = tl.load(Target_ptr + row_idx)
    is_target_mask = col_offsets == target

    x_fp32 = x.to(tl.float32)
    row_max = tl.max(x_fp32, axis=0)
    row_sum_exp = tl.sum(tl.exp(x_fp32 - row_max), axis=0)

    loss = softmax_cross_entropy(x, target, row_max, row_sum_exp, is_target_mask)
    row_loss = tl.sum(loss, axis=0)

    tl.store(Loss_ptr + row_idx, row_loss)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 128), (32, 512), (4, 1024)])
def test_softmax_fwd(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    n_rows, n_cols = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    y_ref = torch.softmax(x, dim=-1)

    y_tri = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_fwd_kernel[grid](
        x, y_tri,
        x.stride(0),
        N_COLS=n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
    )

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(y_tri, y_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(16, 128), (32, 512), (4, 1024)])
def test_softmax_cross_entropy_fwd(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    n_rows, n_cols = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    target = torch.randint(0, n_cols, (n_rows,), device="cuda", dtype=torch.int64)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss_ref = loss_fn(x, target)

    loss_tri = torch.empty(n_rows, device="cuda", dtype=torch.float32)
    grid = (n_rows,)
    _cross_entropy_fwd_kernel[grid](
        x, target, loss_tri,
        x.stride(0),
        N_COLS=n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
    )

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(loss_tri.to(dtype), loss_ref, atol=atol, rtol=rtol)
