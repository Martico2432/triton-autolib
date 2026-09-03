import pytest
import torch
import triton
import triton.language as tl

from triton_autolib.forward.reductions import softmax, softmax_cross_entropy, logsumexp_combine, welford_combine


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

@triton.jit
def _logsumexp_combine_fwd_kernel(
    M1_ptr, L1_ptr, M2_ptr, L2_ptr, M_ptr, L_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    m1 = tl.load(M1_ptr + offsets, mask=mask)
    l1 = tl.load(L1_ptr + offsets, mask=mask)
    m2 = tl.load(M2_ptr + offsets, mask=mask)
    l2 = tl.load(L2_ptr + offsets, mask=mask)

    m, l = logsumexp_combine(m1, l1, m2, l2)

    tl.store(M_ptr + offsets, m, mask=mask)
    tl.store(L_ptr + offsets, l, mask=mask)


@triton.jit
def _welford_combine_fwd_kernel(
    C1_ptr, Mean1_ptr, M2_1_ptr, C2_ptr, Mean2_ptr, M2_2_ptr,
    C_ptr, Mean_ptr, M2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c1 = tl.load(C1_ptr + offsets, mask=mask)
    mean1 = tl.load(Mean1_ptr + offsets, mask=mask)
    m2_1 = tl.load(M2_1_ptr + offsets, mask=mask)
    c2 = tl.load(C2_ptr + offsets, mask=mask)
    mean2 = tl.load(Mean2_ptr + offsets, mask=mask)
    m2_2 = tl.load(M2_2_ptr + offsets, mask=mask)

    c, mean, m2 = welford_combine(c1, mean1, m2_1, c2, mean2, m2_2)

    tl.store(C_ptr + offsets, c, mask=mask)
    tl.store(Mean_ptr + offsets, mean, mask=mask)
    tl.store(M2_ptr + offsets, m2, mask=mask)


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

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (32, 128), (17, 33)])
def test_logsumexp_combine_fwd(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    m1 = torch.randn(shape, device="cuda", dtype=dtype)
    l1 = torch.rand(shape, device="cuda", dtype=dtype) + 0.1
    m2 = torch.randn(shape, device="cuda", dtype=dtype)
    l2 = torch.rand(shape, device="cuda", dtype=dtype) + 0.1

    stacked_m = torch.stack([m1, m2])
    stacked_l = torch.stack([l1, l2])
    m_ref = torch.logsumexp(stacked_m + torch.log(stacked_l), dim=0)
    # reference via direct definition (avoids relying on logsumexp's own internals)
    m_ref = torch.maximum(m1, m2).to(torch.float32)
    l_ref = (
        l1.to(torch.float32) * torch.exp(m1.to(torch.float32) - m_ref)
        + l2.to(torch.float32) * torch.exp(m2.to(torch.float32) - m_ref)
    ).to(dtype)
    m_ref = m_ref.to(dtype)

    m_tri = torch.empty_like(m1)
    l_tri = torch.empty_like(l1)
    n_elements = m1.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _logsumexp_combine_fwd_kernel[grid](m1, l1, m2, l2, m_tri, l_tri, n_elements)

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(m_tri, m_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(l_tri, l_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (32, 128), (17, 33)])
def test_welford_combine_fwd(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    c1 = torch.randint(1, 50, shape, device="cuda", dtype=torch.float32)
    c2 = torch.randint(1, 50, shape, device="cuda", dtype=torch.float32)
    mean1 = torch.randn(shape, device="cuda", dtype=dtype)
    mean2 = torch.randn(shape, device="cuda", dtype=dtype)
    m2_1 = torch.rand(shape, device="cuda", dtype=dtype) + 0.1
    m2_2 = torch.rand(shape, device="cuda", dtype=dtype) + 0.1

    c1f, c2f = c1.to(torch.float32), c2.to(torch.float32)
    mean1f, mean2f = mean1.to(torch.float32), mean2.to(torch.float32)
    m2_1f, m2_2f = m2_1.to(torch.float32), m2_2.to(torch.float32)

    c_ref = (c1f + c2f)
    delta = mean2f - mean1f
    mean_ref = (mean1f + delta * c2f / (c1f + c2f)).to(dtype)
    m2_ref = (m2_1f + m2_2f + delta * delta * c1f * c2f / (c1f + c2f)).to(dtype)

    c_tri = torch.empty_like(c1)
    mean_tri = torch.empty_like(mean1)
    m2_tri = torch.empty_like(m2_1)
    n_elements = c1.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _welford_combine_fwd_kernel[grid](
        c1, mean1, m2_1, c2, mean2, m2_2, c_tri, mean_tri, m2_tri, n_elements
    )

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(mean_tri, mean_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(m2_tri, m2_ref, atol=atol, rtol=rtol)
