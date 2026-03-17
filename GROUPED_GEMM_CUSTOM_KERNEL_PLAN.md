# Grouped GEMM Custom Kernel Status And Plan

## Purpose

This document records:

1. What is already implemented and measured.
2. What the current custom Triton sparse MLP kernel is doing.
3. Why the grouped-row prototype changed the performance picture.
4. The concrete implementation plan for the next proper benchmark and kernel pass.

This is the working design note for turning the current grouped-row proof of concept into a real custom-kernel path.

## Part 0: Implementation Update (2026-03-10)

Status of the planned next work in this document:

- [x] `llama3_neuroplastic/bench_grouped_row_gemm.py` added
- [x] `llama3_neuroplastic/check_grouped_correctness.py` added
- [x] `llama3_neuroplastic/bench_grouped_micro.py` added
- [x] short A/B validation run on a real checkpoint completed (`max_samples=64`, `max_eval_batches=8`)
- [x] larger repeated benchmark run completed (`max_samples=256`, `max_eval_batches=32`)
- [ ] grouped-pattern policy decision (exact vs partial pattern, pending measured data)
- [ ] dedicated grouped custom kernel (not started in this pass)

### What Was Implemented

`bench_grouped_row_gemm.py` now benchmarks exactly these modes:

- `baseline_cuda_spmm`
- `grouped_row_gemm_dense`
- `grouped_row_gemm_4bit_dequant`

It includes required dataset/checkpoint controls, warmup/timed batch handling, and JSON output with:

- tokens per second
- avg step ms
- avg LM loss
- perplexity
- peak VRAM
- timed batches/tokens/seconds

`check_grouped_correctness.py` now checks:

- grouped dense vs dense masked reference
- grouped 4-bit dequant vs bitsandbytes/dense reference

It runs:

- small synthetic dense cases
- one real-shape dense case (`rows=128, in_features=4096, out_features=14336, top_k=3, block_size=32`)
- small + real-shape 4-bit cases on a real 4-bit model layer

And reports:

- max abs error
- mean abs error
- pass/fail against configurable tolerances

`bench_grouped_micro.py` now benchmarks grouped math with `triton.testing.do_bench` for:

- current 4-bit Triton sparse path
- grouped dense GEMM path (materialized active columns)
- grouped 4-bit dequant-to-dense path

Across the requested row sweep:

- `rows in {1, 2, 4, 8, 32, 128}`

With fixed shape:

- `in_features=4096`
- `out_features=14336`
- `top_k=3`
- `block_size=32`

And reports:

- median ms
- p20 ms
- p80 ms
- estimated speedup vs baseline

### Findings From This Implementation Pass

1. Block size is effectively fixed at model-wrapper construction time.
   Changing `block_size` after model load is unsafe because sparse wrappers cache block geometry (`block_size`, `num_blocks`) at init. Scripts now hard-fail on mismatch instead of silently running inconsistent geometry.

2. `grouped_row_gemm_dense` can be invalid on pure 4-bit checkpoints.
   If weights are 4-bit and dequant fallback is disabled, grouped dense mode can raise at runtime. Benchmark script now keeps this mode in the output and records an error row rather than aborting the full run.

3. Shared JSON contract is mostly aligned, with one practical caveat for microbench.
   Microbench is a row sweep, so top-level `shape.rows` is set to `-1`, and each result row includes its explicit `rows` value.

4. Full numerical and throughput conclusions were pending at implementation time, then measured in the runs below.

### Measured Results (2026-03-10)

All results below are from:

- checkpoint: `experiments\llama_sca_objective_v1\model_step_0001000`
- shape regime: `in_features=4096`, `out_features=14336`, `top_k=3`, `block_size=32`

#### 1) End-to-End Grouped Benchmark (`bench_grouped_row_gemm.py`)

Short run (`max_samples=64`, `max_eval_batches=8`, `warmup_batches=2`):

- `baseline_cuda_spmm`: `68.17 tok/s`, `929.09 ms/step`, `peak_vram=5.90 GB`
- `grouped_row_gemm_4bit_dequant`: `150.77 tok/s`, `420.06 ms/step`, `peak_vram=6.10 GB`
- Throughput speedup: about `2.21x`
- Step-time reduction: about `54.8%`
- Loss quality remained effectively flat:
  - baseline `avg_lm_loss=14.0023`, grouped `13.9998`
  - baseline `ppl=1,205,409`, grouped `1,202,347`

Large run (`max_samples=256`, `max_eval_batches=32`, `warmup_batches=4`):

- `baseline_cuda_spmm`: `65.79 tok/s`, `933.13 ms/step`, `peak_vram=5.90 GB`
- `grouped_row_gemm_4bit_dequant`: `145.37 tok/s`, `422.32 ms/step`, `peak_vram=6.10 GB`
- Throughput speedup: about `2.21x`
- Step-time reduction: about `54.7%`
- Loss quality remained effectively flat:
  - baseline `avg_lm_loss=13.9572`, grouped `13.9554`
  - baseline `ppl=1,152,203.94`, grouped `1,150,202.19`

Dense grouped mode behavior:

- `grouped_row_gemm_dense` failed in both runs with:
  - `RuntimeError: grouped_row_gemm requires dense weights or grouped_row_allow_4bit_dequant=true`
- This confirms the current checkpoint is 4-bit and dense grouped mode is not directly runnable without dense-materialized weights.

#### 2) Correctness Validation (`check_grouped_correctness.py`)

Summary:

- total cases: `6`
- passed: `6`
- failed: `0`

Dense grouped vs dense masked reference:

- `small_a`: max abs error `3.0518e-05`, mean abs error `1.5460e-07`
- `small_b`: max abs error `3.0518e-05`, mean abs error `1.1537e-07`
- `real_shape (128,4096,14336)`: max abs error `7.6294e-06`, mean abs error `1.4829e-08`

Grouped 4-bit dequant vs bitsandbytes/dense reference:

- `rows=2`: max abs error `1.8311e-04`, mean abs error `1.0927e-06`
- `rows=8`: max abs error `2.4414e-04`, mean abs error `1.2388e-06`
- `rows=128`: max abs error `4.8828e-04`, mean abs error `1.2487e-06`

Interpretation:

- Numerical agreement is comfortably within the configured tolerances for all checked shapes.

Note:

- The correctness script currently reports an incorrect `out_features` metadata field for 4-bit cases (shows `29360128` instead of model intermediate size). This is a reporting bug in the script metadata only, not a failed numerical comparison.

#### 3) Microbench (`bench_grouped_micro.py`)

Key trend from median latency:

- `rows=1`: grouped dense `1.14x` faster, grouped 4-bit dequant `0.31x` (slower) vs baseline
- `rows=2`: grouped dense `0.85x` (slower), grouped 4-bit dequant `0.18x` (slower)
- `rows=4`: grouped dense `1.04x` faster, grouped 4-bit dequant `0.24x` (slower)
- `rows=8`: grouped dense `1.87x` faster, grouped 4-bit dequant `0.41x` (slower)
- `rows=32`: grouped dense `6.02x` faster, grouped 4-bit dequant `1.36x` faster
- `rows=128`: grouped dense `25.42x` faster, grouped 4-bit dequant `5.54x` faster

Interpretation:

- Grouped execution has a clear crossover behavior: benefits grow strongly with larger row buckets.
- For very small buckets, grouped 4-bit dequant overhead dominates; for moderate/large buckets, grouped paths win decisively.
- This supports exact-pattern grouped scheduling as a valid front-end for a future dedicated grouped 4-bit kernel, especially when bucket sizes are non-trivial.

## Part A: What Is Already Done

This section is status, not future work.

### Current Kernel Status

The current custom kernel path lives in `llama3_neuroplastic/triton_sparse_mlp.py`.

It implements:

- Dense sparse-input Triton path.
- Dense sparse-output Triton path.
- 4-bit sparse-input Triton path.
- 4-bit sparse-output Triton path.

The dense Triton kernels were not the main problem in measurement.

The main problem was the 4-bit Triton path:

- `triton_sparse_input_linear_4bit` was about `8.54 ms`.
- bitsandbytes reference input was about `2.45 ms`.
- `triton_sparse_output_linear_4bit` was about `9.49 ms`.
- bitsandbytes reference output was about `2.69 ms`.

That means the custom 4-bit Triton path was slower by more than `3x` in the relevant microbench regime.

### What The Custom Kernel Was Likely Losing On

Based on measurement and code structure, the main issues were:

- The 4-bit kernels perform packed weight loads, nibble unpacking, codebook lookup, absmax lookup, and scaling inside the inner loop.
- This creates a memory-latency-heavy path with low arithmetic intensity.
- Mechanical tuning alone did not close the gap.
- A first pass of low-risk tuning was tested and reverted because it made end-to-end throughput worse.

What this means:

- The dense Triton path is not the main blocker.
- The 4-bit per-row GEMV-style structure is the blocker.
- A better result likely requires changing the algorithmic structure, not just micro-tuning the old kernel.

### What Has Already Been Implemented For Grouped GEMM

The grouped-row prototype is already implemented in:

- `llama3_neuroplastic/sca_sparse_mlp.py`
- `llama3_neuroplastic/sca_sparse_config.py`
- `llama3_neuroplastic/neuroplastic_llama.py`
- `llama3_neuroplastic/train_llama_sca_objective.py`

The current grouped prototype already supports:

- a config flag for grouped-row execution
- a minimum bucket-size threshold
- optional 4-bit dequant-to-dense grouped execution
- correctness-tested integration into the current wrapper path

### Why Grouped Row GEMM Helped

The grouped-row prototype is implemented in `llama3_neuroplastic/sca_sparse_mlp.py`.

The grouped idea is:

- Group rows that share the same `active_idx` pattern.
- Build a dense local feature matrix from the active block columns only.
- Run larger GEMM-like matmuls on the grouped bucket instead of doing many separate per-row sparse GEMV operations.

This improves performance because:

- Weight slices are reused across multiple rows in a bucket.
- The math becomes more GEMM-like and less GEMV-like.
- Arithmetic intensity increases.
- The GPU gets larger, denser matrix operations instead of many tiny sparse dot products.

### What The Prototype Actually Proved

Small A/B benchmark result on the same checkpoint and settings, only toggling grouped mode:

- Grouped off: `67.73 tok/s`, `935.03 ms/step`
- Grouped on: `157.41 tok/s`, `402.34 ms/step`
- Delta: about `+132.4% tok/s`

This was only a small-sample benchmark and should be treated as a direction check, not a final production claim.

Important caveat:

- The current grouped-row prototype can use dense materialized weights when `grouped_row_allow_4bit_dequant=true`.
- That is acceptable for algorithm validation.
- It is not the final deployable design for the real custom-kernel path.

What the result proves:

- The grouped algorithm is in the right ballpark.
- The bottleneck is structural, not just a bad launch config.
- The next correct step is to build a proper grouped custom-kernel or grouped dense-kernel benchmark path.

### Why The Current Grouped Code Is Still A Rough Prototype

The current grouped implementation is a valid prototype, but it is still rough because:

- It groups only exact `active_idx` matches.
- It can dequantize 4-bit weights into dense weights for grouped execution.
- It does not yet use a dedicated custom grouped Triton kernel.
- It uses a simple bucket fallback path for very small groups.
- It is enough to validate the algorithmic direction, but not enough to call finished kernel engineering.

So the next step is not "more random optimization".

The next step is a disciplined benchmark and implementation pass.

## Part B: Planned Next Work

This section is forward-looking work that still needs to be implemented.

### Complete Script Plan

The next pass should create three scripts and one result format.

### 1. Benchmark Script

File to create:

- `llama3_neuroplastic/bench_grouped_row_gemm.py`

Purpose:

- Compare the current `cuda_spmm` path against grouped-row execution under controlled settings.

Required modes:

- `baseline_cuda_spmm`
- `grouped_row_gemm_dense`
- `grouped_row_gemm_4bit_dequant`

Required inputs:

- checkpoint dir
- model name
- dataset name/config/split
- max samples
- max sequence length
- batch size
- max eval batches
- warmup batches
- top_k
- block_size
- grouped min bucket
- seed

Required outputs:

- tokens per second
- average step ms
- perplexity
- average LM loss
- peak VRAM
- timed batches
- timed tokens
- timed seconds

Required file output:

- JSON file with one row per mode

### 2. Microbench Script To Build

File to create:

- `llama3_neuroplastic/bench_grouped_micro.py`

Purpose:

- Isolate the grouped-row math itself from the full model.

This script should benchmark:

- current 4-bit Triton sparse input/output path
- grouped dense GEMM path with materialized active columns
- grouped 4-bit dequant-to-dense path

Controlled shape set:

- `rows in {1, 2, 4, 8, 32, 128}`
- `in_features=4096`
- `out_features=14336`
- `top_k=3`
- `block_size=32`

Required output:

- median ms
- p20 ms
- p80 ms
- estimated speedup versus baseline mode

This script must use `triton.testing.do_bench` where appropriate.

### 3. Correctness Script To Build

File to create:

- `llama3_neuroplastic/check_grouped_correctness.py`

Purpose:

- Confirm that grouped-row execution matches the existing masked reference closely enough.

Required comparisons:

- grouped dense vs dense masked reference
- grouped 4-bit dequant vs bitsandbytes/dense reference

Required test shapes:

- small synthetic shapes
- one real-shape case: `rows=128, in_features=4096, out_features=14336, top_k=3, block_size=32`

Required outputs:

- max abs error
- mean abs error
- pass/fail against tolerance

### JSON Output Contract

Use one JSON schema for all benchmark scripts:

```json
{
  "run_name": "string",
  "checkpoint_dir": "string",
  "seed": 42,
  "shape": {
    "rows": 128,
    "in_features": 4096,
    "out_features": 14336,
    "top_k": 3,
    "block_size": 32
  },
  "results": [
    {
      "mode": "baseline_cuda_spmm",
      "tokens_per_second": 0.0,
      "avg_step_ms": 0.0,
      "avg_lm_loss": 0.0,
      "perplexity": 0.0,
      "peak_vram_gb": 0.0,
      "timed_batches": 0,
      "timed_tokens": 0,
      "timed_seconds": 0.0
    }
  ]
}
```

### Implementation Order

The implementation should happen in this order:

1. Write `bench_grouped_row_gemm.py`
2. Write `check_grouped_correctness.py`
3. Run a short A/B validation on the real checkpoint
4. Write `bench_grouped_micro.py`
5. Decide whether exact-pattern grouping is enough or whether partial-pattern grouping is needed
6. Only after that, build a dedicated grouped custom kernel

This order matters because the next kernel should be driven by measured grouped behavior, not by guesswork.

### Kernel Design Plan After Benchmarking

If grouped benchmarking continues to show the same kind of win, the real custom-kernel plan should be:

1. Keep row grouping as the front-end scheduling step.
2. Replace dequant-to-dense grouped execution with a grouped custom kernel.
3. Load one active block slice of packed 4-bit weights once per bucket.
4. Perform GEMM-like accumulation across the whole row bucket.
5. Write back only active output columns.

That future grouped custom kernel should target:

- larger row buckets
- better weight reuse
- less repeated 4-bit unpack overhead per row
- more GEMM-like execution

### Acceptance Criteria

The next implementation pass is successful only if:

- correctness checks pass against reference
- grouped mode remains faster in repeated small-sample A/B
- grouped mode does not explode VRAM unreasonably
- the performance gain survives a larger sample test

Minimum practical target:

- grouped mode stays clearly above the current `cuda_spmm` baseline on repeated runs

Preferred target:

- grouped mode remains at least `1.5x` faster than current `cuda_spmm` in the small benchmark regime

### Commands To Use After The Next Scripts Exist

Short benchmark:

```powershell
verification_env\Scripts\python.exe llama3_neuroplastic\bench_grouped_row_gemm.py --checkpoint-dir experiments\llama_sca_objective_v1\model_step_0001000 --max-samples 64 --max-eval-batches 8 --warmup-batches 2 --output-json results\grouped_row_bench_small.json
```

Larger benchmark:

```powershell
verification_env\Scripts\python.exe llama3_neuroplastic\bench_grouped_row_gemm.py --checkpoint-dir experiments\llama_sca_objective_v1\model_step_0001000 --max-samples 256 --max-eval-batches 32 --warmup-batches 4 --output-json results\grouped_row_bench_large.json
```

Correctness check:

```powershell
verification_env\Scripts\python.exe llama3_neuroplastic\check_grouped_correctness.py
```

Microbench:

```powershell
verification_env\Scripts\python.exe llama3_neuroplastic\bench_grouped_micro.py
```

## Bottom Line

The current custom 4-bit Triton kernel path was structurally losing.

The grouped-row prototype changed the structure and immediately moved performance into the right range.

That means the next serious work should be:

- benchmark grouped mode properly
- validate correctness properly
- then build the real grouped custom kernel around that algorithm

Not:

- more blind micro-tuning of the old per-row 4-bit sparse GEMV path
