import pytest
import torch
import triton
import triton.language as tl

from triton_autolib.backward.reductions import d_softmax, d_softmax_cross_entropy, d_logsumexp_combine, d_welford_combine


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

@triton.jit
def _logsumexp_combine_bwd_kernel(
    DM_ptr, DL_ptr, M1_ptr, L1_ptr, M2_ptr, L2_ptr,
    DM1_ptr, DL1_ptr, DM2_ptr, DL2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    dm = tl.load(DM_ptr + offsets, mask=mask)
    dl = tl.load(DL_ptr + offsets, mask=mask)
    m1 = tl.load(M1_ptr + offsets, mask=mask)
    l1 = tl.load(L1_ptr + offsets, mask=mask)
    m2 = tl.load(M2_ptr + offsets, mask=mask)
    l2 = tl.load(L2_ptr + offsets, mask=mask)

    dm1, dl1, dm2, dl2 = d_logsumexp_combine(dm, dl, m1, l1, m2, l2)

    tl.store(DM1_ptr + offsets, dm1, mask=mask)
    tl.store(DL1_ptr + offsets, dl1, mask=mask)
    tl.store(DM2_ptr + offsets, dm2, mask=mask)
    tl.store(DL2_ptr + offsets, dl2, mask=mask)


@triton.jit
def _welford_combine_bwd_kernel(
    DMean_ptr, DM2_ptr, C1_ptr, Mean1_ptr, C2_ptr, Mean2_ptr,
    DMean1_ptr, DM2_1_ptr, DMean2_ptr, DM2_2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    d_mean = tl.load(DMean_ptr + offsets, mask=mask)
    d_m2 = tl.load(DM2_ptr + offsets, mask=mask)
    c1 = tl.load(C1_ptr + offsets, mask=mask)
    mean1 = tl.load(Mean1_ptr + offsets, mask=mask)
    c2 = tl.load(C2_ptr + offsets, mask=mask)
    mean2 = tl.load(Mean2_ptr + offsets, mask=mask)

    d_mean1, d_m2_1, d_mean2, d_m2_2 = d_welford_combine(
        d_mean, d_m2, c1, mean1, c2, mean2
    )

    tl.store(DMean1_ptr + offsets, d_mean1, mask=mask)
    tl.store(DM2_1_ptr + offsets, d_m2_1, mask=mask)
    tl.store(DMean2_ptr + offsets, d_mean2, mask=mask)
    tl.store(DM2_2_ptr + offsets, d_m2_2, mask=mask)


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

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (32, 128), (17, 33)])
def test_logsumexp_combine_bwd(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def combine(m1, l1, m2, l2):
        m = torch.maximum(m1, m2)
        l = l1 * torch.exp(m1 - m) + l2 * torch.exp(m2 - m)
        return m, l

    m1 = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    l1 = (torch.rand(shape, device="cuda", dtype=torch.float32) + 0.1).requires_grad_(True)
    m2 = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    l2 = (torch.rand(shape, device="cuda", dtype=torch.float32) + 0.1).requires_grad_(True)
    dm = torch.randn(shape, device="cuda", dtype=dtype).to(torch.float32)
    dl = torch.randn(shape, device="cuda", dtype=dtype).to(torch.float32)

    m_ref, l_ref = combine(m1, l1, m2, l2)
    (m_ref * dm + l_ref * dl).sum().backward()
    dm1_ref, dl1_ref, dm2_ref, dl2_ref = m1.grad, l1.grad, m2.grad, l2.grad

    dm1_tri = torch.empty_like(m1, requires_grad=False).detach()
    dl1_tri = torch.empty_like(l1).detach()
    dm2_tri = torch.empty_like(m2).detach()
    dl2_tri = torch.empty_like(l2).detach()
    n_elements = m1.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _logsumexp_combine_bwd_kernel[grid](
        dm, dl,
        m1.detach(), l1.detach(), m2.detach(), l2.detach(),
        dm1_tri, dl1_tri, dm2_tri, dl2_tri,
        n_elements,
    )

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(dm1_tri, dm1_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dl1_tri, dl1_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dm2_tri, dm2_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dl2_tri, dl2_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1024,), (32, 128), (17, 33)])
def test_welford_combine_bwd(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def combine(c1, mean1, m2_1, c2, mean2, m2_2):
        c = c1 + c2
        delta = mean2 - mean1
        mean = mean1 + delta * c2 / c
        m2 = m2_1 + m2_2 + delta * delta * c1 * c2 / c
        return mean, m2

    c1 = torch.randint(1, 50, shape, device="cuda", dtype=torch.float32)
    c2 = torch.randint(1, 50, shape, device="cuda", dtype=torch.float32)
    mean1 = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    mean2 = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    m2_1 = (torch.rand(shape, device="cuda", dtype=torch.float32) + 0.1).requires_grad_(True)
    m2_2 = (torch.rand(shape, device="cuda", dtype=torch.float32) + 0.1).requires_grad_(True)
    d_mean = torch.randn(shape, device="cuda", dtype=dtype).to(torch.float32)
    d_m2 = torch.randn(shape, device="cuda", dtype=dtype).to(torch.float32)

    mean_ref, m2_ref = combine(c1, mean1, m2_1, c2, mean2, m2_2)
    (mean_ref * d_mean + m2_ref * d_m2).sum().backward()
    dmean1_ref, dm2_1_ref = mean1.grad, m2_1.grad
    dmean2_ref, dm2_2_ref = mean2.grad, m2_2.grad

    dmean1_tri = torch.empty_like(mean1).detach()
    dm2_1_tri = torch.empty_like(m2_1).detach()
    dmean2_tri = torch.empty_like(mean2).detach()
    dm2_2_tri = torch.empty_like(m2_2).detach()
    n_elements = mean1.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _welford_combine_bwd_kernel[grid](
        d_mean, d_m2,
        c1, mean1.detach(), c2, mean2.detach(),
        dmean1_tri, dm2_1_tri, dmean2_tri, dm2_2_tri,
        n_elements,
    )

    atol = 1e-2 if dtype == torch.bfloat16 else 1e-4
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(dmean1_tri, dmean1_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dm2_1_tri, dm2_1_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dmean2_tri, dmean2_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dm2_2_tri, dm2_2_ref, atol=atol, rtol=rtol)
