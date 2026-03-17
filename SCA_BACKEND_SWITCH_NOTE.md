# SCA Backend Switch Note

## What Was Removed
- Custom C++/CUDA extension runtime path from the hybrid model wrapper:
  - previous loader: `llama3_neuroplastic/cuda/sca_sparse_loader.py`
  - previous extension entrypoints: `spatial_gate_cuda`, `sparse_adapter_cuda`

## What Replaces It
- Routing backend:
  - primary: Triton gate (`triton_compute_active_blocks_topk`)
  - fallback: torch routing (`compute_active_blocks_torch`)
- Sparse MLP compute backend:
  - `sca_spmm_impl='cuda_spmm'` path already implemented via Triton/materialized sparse kernels in Python runtime.

## Why
- Extension build was a toolchain blocker on this machine (MSVC/CUDA compile failures).
- The extension-free path keeps strict sparse runtime functional and reproducible without C++ build dependency.
