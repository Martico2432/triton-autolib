## Description
Briefly describe the changes, new activation functions, or reduction helpers introduced in this PR.

## Verification Checklist
- [ ] Added unit tests covering both `float32` and `bfloat16`.
- [ ] Ran `pytest` locally on CUDA hardware and verified all tests pass.
- [ ] Verified `torch.testing.assert_close` matches PyTorch reference implementations.
