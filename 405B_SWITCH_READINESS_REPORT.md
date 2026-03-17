# 405B Switch Readiness Report (2026-03-13)

## Verdict
- **Not ready to switch to 405B yet.**

## What Was Validated
- Strict sparse-attention runtime path executes on 8B with one model instance.
- Sparse-attention paging now activates on long-context probe.
- Strict-mode sparse invariants are enforced:
  - `sca_use_cuda=True`
  - fast fallback disabled (`stability_dense_fallback_threshold=0`)
  - sparse MLP fallback rate observed at `0.0` in strict quick runs

## Key Metrics Collected

### A) Short quality comparison (quick gate: 2 prompts, 12 tokens)
- Sparse baseline (SCA on, sparse-attention off):
  - `distinct1_mean`: `0.5424`
  - `max_run_mean`: `1.0`
  - `rep_frac_mean`: `0.0`
  - file: `results/readiness_sparse_baseline_quick.json`

- Strict sparse-attention (SCA on, sparse-attention on, strict):
  - `distinct1_mean`: `0.4576`
  - `max_run_mean`: `9.0`
  - `rep_frac_mean`: `0.5458`
  - file: `results/readiness_sparse_attention_strict_quick.json`

### B) Long-context probe (strict sparse-attention)
- Prompt length: `400` input tokens
- Decode: `1` token
- Latency: `27.97s` (`0.0358 tok/s`)
- Peak CUDA memory: `5.97 GiB`
- Sparse-attention diagnostics:
  - `mean_selected_pages_per_step`: `44.0`
  - `mean_bytes_cpu_to_gpu_per_step`: `630,784`
- file: `results/long_context_strict_sparse_attention_eval.json`

### C) Gate summary
- `strict_sparse_attention_active`: **PASS**
- `strict_sparse_mlp_no_fallback`: **PASS**
- `long_context_probe_completed`: **PASS**
- `strict_not_worse_than_sparse_baseline`: **FAIL**
- file: `results/405b_switch_readiness_gates.json`

## Fixes Implemented During This Readiness Pass

1. Enabled cache when sparse-attention mode is active
- Sparse-attention decode needs cache; cache was previously disabled by hybrid default.
- File: `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`

2. Fixed sparse-attention prefill performance path
- Removed token-by-token prefill dependency for sparse-attention mode.
- Added full-cache bootstrap ingest for archive pages.
- Files:
  - `llama3_neuroplastic/paged_sparse_attention.py`
  - `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`

3. Fixed archive bootstrap logic
- Bootstrap now initializes per layer/head instead of only once globally.
- File: `llama3_neuroplastic/paged_sparse_attention.py`

4. Prevented unnecessary retrieval when context is still local
- Retrieval is skipped unless `seq_len > local_window + sink_tokens`.
- File: `llama3_neuroplastic/paged_sparse_attention.py`

5. Hardened strict inference CLI settings
- Strict mode now sets:
  - `sca_spmm_impl='cuda_spmm'`
  - `stability_dense_fallback_threshold=0.0`
- File: `llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py`

6. Added tests and kept them green
- Updated/added tests for sparse-attention runtime and cache behavior.
- Current targeted test status remained passing.

## Remaining Blocking Problem
- **Strict mode quality is still materially worse than sparse baseline quality.**
- Failure mode on strict runs remains repetition/degenerate lexical loops.
- This is the blocker preventing a safe 405B switch.

## What Must Be Done Before 405B
1. Resolve strict-path decode quality regression so short prompt quality is not worse than sparse baseline.
2. Re-run short + long-context gates on 8B and pass all four:
   - strict sparse attention active
   - no fallback
   - long-context probe completed
   - strict quality not worse than sparse baseline
3. Only after that, run 405B pilot with same gate criteria.

---

## Update: Decoder Mirror Co-Warp Readiness Rerun (2026-03-13)

### Scope executed
- Added decoder mirror runtime + calibration path and reran readiness gates on 8B with strict sparse settings.
- Produced two mirror calibrations:
  - `results/decoder_mirror_calibration_full_sparse_gate_v1/decoder_mirror_calibration_metrics.json`
  - `results/decoder_mirror_calibration_full_sparse_gate_v2/decoder_mirror_calibration_metrics.json`
- Re-ran gate artifacts:
  - `results/readiness_sparse_baseline_quick_v2.json`
  - `results/readiness_sparse_attention_strict_quick_with_mirror_v3.json`
  - `results/long_context_strict_sparse_attention_eval_with_mirror_v2.json`
  - `results/405b_switch_readiness_gates_v2.json`

### Gate results (v2)
- `strict_sparse_attention_active`: **PASS**
- `strict_sparse_mlp_no_fallback`: **PASS**
- `long_context_probe_completed`: **PASS**
- `strict_not_worse_than_sparse_baseline`: **FAIL**

### Key v2 metrics
- Baseline quick (`sparse_attention=off`):
  - `distinct1_mean`: `0.86696`
  - `max_run_mean`: `1.0`
  - `rep_frac_mean`: `0.0`
- Strict quick + mirror (`sparse_attention=on`, strict):
  - `distinct1_mean`: `0.75`
  - `max_run_mean`: `4.0`
  - `rep_frac_mean`: `0.23333`
- Long-context strict + mirror:
  - `input_tokens`: `400`
  - `latency_s`: `24.47`
  - `tok_s`: `0.0409`
  - `mean_selected_pages_per_step`: `44.0`
  - `mean_bytes_cpu_to_gpu_per_step`: `630,784`

### Findings
- Decoder mirror did not close the strict quality gap enough to pass the gate.
- The strict path remains materially more repetitive than baseline.
- Long-context sparse-attention paging remains active and healthy in strict mode.
- No sparse MLP fallback was observed in strict mode during these checks.

### Current readiness verdict
- **Still not ready for 405B switch.**
- Blocking condition is unchanged: strict quality gate remains failing (`strict_not_worse_than_sparse_baseline=false`).

---

## Update: Speech-Layer Anchor Re-Probe (2026-03-13, corrected strict checks)

### What was rerun
- Re-validated speech-anchor recalibrated SCA checkpoint:
  - `results/sca_recalibration_speech_anchor_v1/sca_recalibrated_state.pt`
- Re-ran strict sparse inference probes and rebuilt gate artifact:
  - `results/405b_switch_readiness_gates_speech_anchor_v4.json`
- Re-ran long-context strict probe with cache explicitly enabled so sparse-attention paging is actually active:
  - `results/_tmp_diag_long_allow_cache.json`

### Important correction
- Earlier speech-anchor gate artifact (`results/405b_switch_readiness_gates_speech_anchor_v1.json`) was not a valid readiness signal:
  - `strict_sparse_attention_active=false` in that file
  - long-context diagnostics had `steps=0` and `mean_selected_pages_per_step=0.0`
  - quality metrics were over-optimistic for degenerate subword loops (for example `...udiudiudi...`, `...nescnesc...`), because token-level repetition checks can miss non-whitespace lexical collapse.

### Corrected gate results (v4)
- `strict_sparse_attention_active`: **PASS**
  - from explicit long-context strict run with `--allow-cache`, pages selected per step were non-zero (`mean_selected_pages_per_step=44.0`)
- `strict_sparse_mlp_no_fallback`: **PASS**
- `long_context_probe_completed`: **PASS**
- `strict_not_worse_than_sparse_baseline`: **FAIL**

### Quality findings on speech-anchor strict path
- Strict path still shows collapse/repetition patterns under short prompts (examples include repeated lexical fragments and repeated tokens).
- In the v4 short probe, strict candidate quality remained materially worse than sparse baseline:
  - baseline `max_run_mean=3.5`, strict `max_run_mean=11.0`
  - baseline `rep_frac_mean=0.1923`, strict `rep_frac_mean=0.5882`
  - baseline `degenerate_frac=0.5`, strict `degenerate_frac=1.0`

### Additional runtime finding
- Inference still logs:
  - `[sca-cuda] warning: Failed to build/load SCA CUDA extension`
- This does not stop execution, but it means strict sparse runtime validation should include explicit kernel/backend availability checks in addition to config-flag checks.

### Current readiness verdict after speech-anchor re-probe
- **Still not ready for 405B switch.**
- Blocking condition remains: strict sparse path quality is worse than sparse baseline, despite strict sparse-attention runtime now being verifiably active on long-context decode.
