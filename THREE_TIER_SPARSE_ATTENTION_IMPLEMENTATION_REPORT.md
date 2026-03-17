# Three-Tier Sparse Attention Implementation Report

## Scope Implemented
- Added a new paged sparse-attention runtime module: `llama3_neuroplastic/paged_sparse_attention.py`.
- Integrated sparse-attention mode into `NeuroplasticLlama` as a decode-time feature flag.
- Added strict fully-sparse invariants for runtime checks.
- Added inference CLI controls for sparse attention and diagnostics export.
- Added unit tests for page routing, archive behavior, strict checks, and helper mappings.

## Files Added
- `llama3_neuroplastic/paged_sparse_attention.py`
- `tests/test_paged_sparse_attention.py`
- `THREE_TIER_SPARSE_ATTENTION_IMPLEMENTATION_REPORT.md`

## Files Modified
- `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`
- `llama3_neuroplastic/bounded_context.py`
- `llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py`
- `tests/test_bounded_context.py`
- `tests/test_hybrid_gqa_mamba.py`

## Functional Changes

### 1) New sparse attention runtime primitives
- Added:
  - `SparseAttentionConfig`
  - `SparseAttentionStepStats`
  - `LongRangePageArchive`
  - `LongRangeSummaryTable`
  - `TwoStagePageRetriever`
  - `SparseAttentionRuntime`
- Implemented page helpers:
  - `page_for_token`
  - `page_token_span`
  - `gqa_query_head_to_kv_group`
- Archive supports `fp16` and `int4` storage modes.
- Two-stage routing:
  - Stage A: query-vs-summary score.
  - Stage B: fetch selected pages only.

### 2) `NeuroplasticLlama` sparse-attention integration
- Constructor now accepts:
  - `attention_sparse_mode`
  - `attention_local_window_tokens`
  - `attention_sink_tokens`
  - `attention_page_size_tokens`
  - `attention_retrieval_top_k_pages`
  - `attention_retrieval_head_group_ids`
  - `attention_retrieval_start_layer`
  - `attention_archive_cpu_dtype`
  - `attention_hot_archive_gpu_pages`
  - `attention_disable_ssd_fetch_in_decode`
  - `attention_force_single_model_runtime`
- Added public methods:
  - `set_sparse_attention_mode(...)`
  - `get_sparse_attention_diagnostics()`
  - `reset_sparse_attention_state()`
- Cache path wiring:
  - `_pack_past_key_values` now runs `SparseAttentionRuntime.update_and_compress_cache(...)` when sparse attention is enabled.
  - Legacy bounded global-window compression is skipped when sparse attention mode is enabled.
- `save_pretrained` / `from_pretrained` now persist and restore `sparse_attention` config.

### 3) Strict fully-sparse runtime checks
- Added `BoundedContextConfig.validate_sparse_attention_runtime(...)`.
- In generation path, strict mode now validates:
  - `sca_use_cuda=True`
  - `sca_spmm_impl='cuda_spmm'`
  - fast fallback disabled (`stability_dense_fallback_threshold <= 0`)
  - `disable_ssd_fetch_in_decode=True`

### 4) Inference CLI
- Added flags in `run_hybrid_gqa_mamba_inference.py`:
  - `--enable-sparse-attention`
  - `--local-window-tokens`
  - `--sink-tokens`
  - `--page-size-tokens`
  - `--retrieval-top-k-pages`
  - `--retrieval-head-groups`
  - `--retrieval-start-layer`
  - `--archive-cpu-dtype {int4,fp16}`
  - `--hot-archive-gpu-pages`
  - `--strict-fully-sparse`
  - `--dump-attention-diagnostics-json`

## Test Coverage Added

### `tests/test_paged_sparse_attention.py`
- Archive roundtrip for `int4` and `fp16`.
- Deterministic top-k summary routing smoke.
- Runtime compression/selection loop smoke.
- Strict-mode rejection when sparse fallback diagnostics indicate fallback usage.

### `tests/test_bounded_context.py`
- GQA head-group mapping helper test.
- Page index/span helper roundtrip test.
- Strict sparse-runtime validation test.

### `tests/test_hybrid_gqa_mamba.py`
- Sparse attention mode set/reset + diagnostics state smoke on a lightweight fake model object.

## Validation Run Results
- Compile check passed:
  - `python -m py_compile ...` for all changed modules/tests.
- Test run passed under project env:
  - `.\verification_env\Scripts\python.exe -m pytest -q tests/test_paged_sparse_attention.py tests/test_bounded_context.py tests/test_hybrid_gqa_mamba.py`
  - Result: `27 passed`.
- CLI parse check passed:
  - `.\verification_env\Scripts\python.exe -m llama3_neuroplastic.run_hybrid_gqa_mamba_inference --help`

## Important Caveats (Current State)
- This implementation adds a working three-tier sparse-attention runtime scaffold and integration path, but it does not yet replace the underlying transformer attention kernel with a fully custom per-head sparse kernel.
- Retrieval-head long-range pages are selected and compressed in cache management, with diagnostics and strict-mode guards, but per-head exclusion at raw attention-kernel level is still constrained by the base model’s shared cache tensor format.
- This is ready for the next phase: tighter attention-kernel behavior and quality-focused tuning on real long-context runs.

## Readiness for Next Step
- Code path, config, strict checks, and test scaffolding are in place.
- The repo is now ready to proceed to the attention-specific optimization/fidelity phase.
