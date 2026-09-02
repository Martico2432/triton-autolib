# Contributing

Thanks for your interest in contributing! We welcome bug fixes, performance improvements, and new elementary activation or reduction kernels.

There are only a basic set of rules for contributing to this project.

---

## 1. Core Principles & Rules

To keep `triton_autolib` lightweight, clean, and maintainable, please follow these rules:

1. **No Fused Operations:** Fused combinations (e.g., Fused Multiply-Add, SiLU-Mul, Mul-SiLU-Mul) are **not allowed** unless explicitly requested or pre-approved by core maintainers. Keep operations modular and atomic.
2. **Low-Comment Style:** Avoid cluttering kernel code with line-by-line comments.
   - Prefer expressive, self-describing variable names.
   - Use concise docstrings only when explaining complex mathematical formulations or non-obvious algorithm behavior.
3. **Clean Code & Formatting:** Follow PEP 8 guidelines for Python code. Maintain consistent indentation and structure in both Python modules and `@triton.jit` functions.

---

## 2. Project Structure

When adding new activation functions, loss functions, or reductions, adhere strictly to the package layout.

---

## 3. Pull Request Guidelines

Before submitting a Pull Request (PR):

1. **Write Reference Tests:** Every new Triton kernel or modified kernel MUST include unit tests in `tests/` verifying accuracy against PyTorch reference implementations.
2. **Test Multiple Precision Types:** Tests must pass across target data types, specifically `torch.float32` and `torch.bfloat16`.
3. **Human Review Required:** Every PR must be reviewed and approved by at least one maintainer before merging.

---

## 4. AI & Automated Contributions

If you use AI coding tools (e.g., Cursor, GitHub Copilot, Claude Code, or automated agents) to generate code or PRs:

1. **Follow `AGENTS.md`:** Ensure the AI agent reads and complies with all strict JIT syntax and typing rules defined in [`AGENTS.md`](./AGENTS.md).
2. **Local GPU Validation Required:** Because standard CI runners do not have physical CUDA hardware, all AI-generated or modified kernels **must** be tested locally using `pytest` on NVIDIA GPUs before submitting a PR.
3. **Check the Checklist:** You must explicitly check off local CUDA test execution in the PR description template.
