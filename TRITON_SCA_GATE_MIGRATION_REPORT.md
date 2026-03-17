# Triton SCA Gate Migration Report

## Scope
This report documents the work done to remove dependency on the custom C++/CUDA SCA gate extension during runtime and replace gate routing with a Triton implementation where feasible.

## What Was Implemented

### 1. Triton gate kernel added
- New file: `llama3_neuroplastic/triton_sca_gate.py`
- Added:
  - `triton_sca_gate_available()`
  - `triton_compute_active_blocks_topk(...)`
  - Triton kernel `_triton_spatial_topk_kernel` for per-row top-k block selection
- Behavior:
  - Computes score from query-to-center distance with sigma scaling
  - Supports refractory masking
  - Uses deterministic tie-break (`tie_break_left=True`) and index-bias strategy

### 2. Routing integration in sparse adapter
- Modified: `llama3_neuroplastic/sca_sparse_adapter.py`
- `compute_active_blocks(...)` now:
  1. Prefers Triton gate on CUDA (when inhibition lambda is 0, Triton available)
  2. Falls back to custom extension path if available
  3. Falls back to torch implementation otherwise

### 3. Gradient-safe fallback behavior
- Triton gate is used only when `torch.is_grad_enabled() == False`.
- Training path keeps torch routing to preserve gradient flow.
- This fixes runtime failure:
  - `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

### 4. Model routing gate adjustment
- Modified: `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`
- `use_cuda_kernel` is now based on:
  - `sca_config.use_cuda` and `query.is_cuda`
- It no longer requires the custom extension object to exist before enabling CUDA gate path.

## Extension Build Hardening (Environment Track)

### 5. Loader robustness updates
- Modified: `llama3_neuroplastic/cuda/sca_sparse_loader.py`
- Added:
  - Source staging to ASCII temp directory
  - Include path normalization with Windows short paths
  - `-allow-unsupported-compiler` in NVCC flags

### 6. CUDA source fix
- Modified: `llama3_neuroplastic/cuda/sca_sparse_kernels.cu`
- Replaced `CUDART_INF_F` usage with `-INFINITY` to resolve identifier errors under current toolchain.

## Tests Added/Updated

### 7. Triton gate parity test
- Modified: `tests/test_sca_sparse_adapter.py`
- Added CUDA test:
  - `test_triton_gate_matches_torch_topk_smoke`
- Validates Triton gate index/score shape and parity with torch top-k routing under supported conditions.

## Commands and Verification Results

### 8. Static + unit checks
- `py_compile` on changed modules: passed
- `pytest tests/test_sca_sparse_adapter.py`: passed (`10 passed`)
- Full adapter test suite status after changes: passed

### 9. Runner smoke (end-to-end)
- Command run:
  - `run_sca_recalibration_from_hybrid_baseline.py` with:
    - `sca_use_cuda=true`
    - `spmm_impl=cuda_spmm`
    - fast fallback disabled
    - quality gate enabled
- Result:
  - Run completed
  - Fallback rate on calibrated layers = `0.0`
  - Quality gate passed
  - Artifacts written under:
    - `results/sca_recalibration_triton_gate_smoke/`

## Findings

### 10. Custom extension status
- The custom extension still does not fully compile on this machine due to deeper compiler/header compatibility issues (beyond the initial missing `cl.exe` and unsupported-compiler gate).
- This no longer blocks runtime because Triton gate + torch fallback path is functional.

### 11. Practical outcome
- Routing no longer depends on successful custom extension build for functional operation.
- CUDA sparse MLP path (`cuda_spmm`) remains usable.
- Recalibration flow remains stable with enforced sparse settings.

## Current Readiness
- Ready to proceed with sparse-attention implementation work.
- Remaining extension issues are now an optimization/environment track, not a blocker for functional progress.

## Files Changed in This Phase
- `llama3_neuroplastic/triton_sca_gate.py` (new)
- `llama3_neuroplastic/sca_sparse_adapter.py`
- `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`
- `llama3_neuroplastic/cuda/sca_sparse_loader.py`
- `llama3_neuroplastic/cuda/sca_sparse_kernels.cu`
- `tests/test_sca_sparse_adapter.py`
