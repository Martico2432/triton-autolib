# AI Agent Guidelines for triton_autolib

If you are an automated agent, AI assistant, or code generator contributing to `triton_autolib`, you MUST strictly adhere to these instructions.

---

## 1. Triton JIT Syntax & Constraints

- **No `None` values inside JIT:** NEVER assign `None` inside a `@triton.jit` function. Triton variables must evaluate to valid Triton tensors or scalar primitives.
- **No Pythonic Inline Ternaries on Pointers:** Do NOT write `tl.load(...) if ptr else None`. Use explicit `if ptr is not None:` blocks instead.
- **`tl.constexpr` Usage:** Do NOT call `tl.constexpr(...)` as a runtime casting function on float variables. Use `tl.constexpr` exclusively as a type annotation in kernel signatures or default function arguments.

---

## 2. Floating-Point Precision & Numerical Stability

- **Intermediate FP32 Computation:** Always cast tensor inputs to `tl.float32` prior to performing transcendental math operations (`tl.exp`, `tl.log`, `tl.sigmoid`, division), then cast back to `x.dtype` before storing or returning.
- **`bfloat16` Tolerances:** When writing reference tests against PyTorch, account for lower mantissa precision in `bfloat16` by setting appropriate tolerances (`rtol=1e-2, atol=1e-2`).

---

## 3. Repository Organization

- **Forward Kernel Functions:** Place under `src/triton_autolib/forward/`.
- **Backward Kernel Functions:** Place under `src/triton_autolib/backward/`.
- **Unit Tests:** Place under `tests/forward/` or `tests/backward/`.

---

## 4. Verification & Quality Assurance

- **PyTorch Equivalence:** Every kernel implementation MUST have a corresponding test using PyTorch's native function or autograd engine as ground truth.
- **Local CUDA Execution:** Confirm that `pytest` has passed on local CUDA hardware before creating or updating a PR.
- **Human Oversight:** All AI-generated PRs require manual inspection and approval by a human maintainer.
