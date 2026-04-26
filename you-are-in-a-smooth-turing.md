# Plan: Reach 3.3 tok/s on RTX 2080 8GB for Llama 3.1 405B

## Goal

Deliver a coherent decode path that reaches **3.3 tokens/sec** on an RTX 2080 8GB.

That target is:

- **303 ms/token**
- **2.4 ms/layer** across 126 decoder layers
- GPU-first execution with no CPU fallback on the hot path

The current runtime proves the model can stream and decode, but it is still dominated by dispatch overhead, CPU/GPU transfer overhead, and incomplete sparse-attention correctness.

---

## Current Evidence

### Baseline Runs

| File | Configuration | Decode tok/s | Notes |
|---|---:|---:|---|
| `results/chat_topk208_dense_attn_triton_ram16_tok4.json` | MLP top_k=208, dense attention, Triton sparse MLP path, CPU LM head | 0.0033 | Produces the coherent prefix `The capital of` |
| `results/qa_topk208_dense_attn_notriton_tok8.json` | MLP top_k=208, dense attention, non-Triton sparse MLP | 0.0069 | Produces a coherent QA prefix |
| `results/live_coherent_topk208.json` | MLP top_k=208, dense attention, profiler enabled, no hot cache | 0.0066 | Output is not acceptable, but the decode profile is still useful |
| `results/benchmark.json` | MLP top_k about 51, hot cache, sparse attention/KV enabled | 0.0336 mean | Incoherent regime, but token 5 reaches 2.32s and exposes the warm-cache floor |

### Profiled Bottleneck

From `results/live_coherent_topk208.json`:

| Component | Measured ms/token | Measured ms/layer | Problem |
|---|---:|---:|---|
| Attention load | 42,975 | 341 | Dense Q/O loading dominates attention |
| MLP | 116,319 | 923 | Sparse MLP is dispatch and gather bound |
| Other | 10 | 0.08 | Not currently relevant |
| Total | 159,304 | 1,264 | About 525x over target |

From `results/benchmark.json`:

- Decode token times: `57.46s, 36.12s, 26.53s, 26.52s, 2.32s`
- Decode traffic: **15.77 MB/layer**
- The fifth token shows that a mostly warm path can reach **0.43 tok/s**, but that is still **7.7x below 3.3 tok/s**

The 2.32s warm-token floor is the critical signal. At 15.77 MB/layer, PCIe transfer is not enough to explain the latency. The remaining time is kernel launch, PyTorch dispatch, dequantization, sparse-gather overhead, and non-fused decode work.

---

## Non-Negotiable Direction

The MLP fix is a **true single-kernel decode kernel**, not a two-kernel or three-kernel staging design.

The existing function named `triton_fused_sparse_mlp_decode_4bit` is not fused enough for the target. It currently composes:

1. sparse gate projection kernel
2. sparse up projection kernel
3. PyTorch `silu` and multiply
4. sparse down projection kernel

That path removes some dense work, but it still launches multiple kernels and materializes intermediate tensors. The target path must launch exactly one Triton kernel for the sparse MLP decode math of one layer.

---

## Target Budget

| Component | Current best signal | Required target | Required change |
|---|---:|---:|---|
| Sparse MLP | 923 ms/layer profiled without hot cache | 0.8 to 1.2 ms/layer | Single Triton kernel, hot-cache resident active blocks, no intermediate materialization |
| Sparse attention | 341 ms/layer dense Q/O path | 0.7 to 1.0 ms/layer | Compact sparse attention with coherent head selection |
| LM head | CPU dense, 3.9 GB if fp16 GPU | under 20 ms/token | NF4 GPU LM head only |
| Runtime overhead | visible in warm 2.32s token | under 40 ms/token total | Remove Python dispatch from per-block/per-projection decode |

The plan succeeds only if the final runtime status shows GPU-resident sparse execution:

- `decode_backend = "single_kernel_sparse_decode_sm75"`
- `lm_head_mode = "gpu_nf4"`
- `lm_head_on_gpu = true`
- `attn_backend_decode = "compact_sparse_v1"`
- `compact_sparse_attention_steps > 0`

---

## Change 1: Single-Kernel Sparse MLP Decode

**Files**

- `llama3_neuroplastic/triton_sparse_mlp.py`
- `llama3_neuroplastic/experiments/streaming_llama_runtime.py`
- `llama3_neuroplastic/experiments/runtime/session.py` if scratch state reset is needed

**Why**

This is the highest-leverage change. At top_k=208 and block_size=32, each layer activates 6,656 intermediate neurons. The current path still pays for separate projection kernels, intermediate tensors, Python wrapper dispatch, and per-projection scheduling.

The new path must fuse:

1. NF4 dequant for gate active blocks
2. NF4 dequant for up active blocks
3. gate dot product
4. up dot product
5. SiLU and multiply
6. NF4 dequant for down columns/rows
7. down accumulation into hidden output
8. output initialization and finalization

All of that happens inside one Triton launch per layer.

### New API

```python
triton_sparse_mlp_decode_4bit_single_kernel_sm75(
    x_flat,                 # [1, hidden_size], fp16 CUDA
    active_blocks,          # [top_k], int32 CUDA
    gate_packed,            # packed NF4 gate blocks for the active or cached block bank
    gate_absmax,            # gate absmax for the same bank
    gate_code,              # gate NF4 code table
    up_packed,              # packed NF4 up blocks
    up_absmax,              # up absmax
    up_code,                # up NF4 code table
    down_packed,            # packed NF4 down block columns in active-block order
    down_absmax,            # down absmax in active-block order
    down_code,              # down NF4 code table
    out,                    # [1, hidden_size], fp16 CUDA
    out_accum,              # [hidden_size], fp32 CUDA scratch
    tile_state,             # [ceil(hidden_size / BLOCK_OUT)], int32 CUDA scratch
    tile_done,              # [ceil(hidden_size / BLOCK_OUT)], int32 CUDA scratch
    epoch,                  # monotonically increasing int32
    hidden_size,            # 16384
    block_size,             # 32
    quant_block_size,       # usually 64
    top_k,                  # 208 target
) -> torch.Tensor
```

The wrapper returns `out.view_as(hidden)`.

### Kernel Design

Use one Triton program per active MLP block:

```text
grid = (top_k,)
pid = active block ordinal
block_id = active_blocks[pid]
```

Each program computes its block's 32 intermediate activations once:

1. Initialize `gate_acc[32]` and `up_acc[32]` in fp32.
2. Loop over hidden input tiles, using `BLOCK_IN = 64` on SM75.
3. Load packed NF4 gate and up nibbles for `[32, BLOCK_IN]`.
4. Decode through the per-projection NF4 code table and absmax.
5. Accumulate `gate_acc += gate_tile @ x_tile`.
6. Accumulate `up_acc += up_tile @ x_tile`.
7. Compute `intermediate = silu(gate_acc) * up_acc`.

Then the same program loops over output tiles:

1. Initialize the output tile for the current epoch using an in-kernel tile-state protocol.
2. Load and dequant the down weights for `[BLOCK_OUT, 32]`.
3. Compute `contrib[BLOCK_OUT] = down_tile @ intermediate`.
4. `tl.atomic_add(out_accum[tile_offsets], contrib)`.
5. Increment `tile_done[tile_id]`.
6. The program that observes `tile_done[tile_id] == top_k` converts `out_accum[tile]` to fp16 and stores `out[tile]`.

### In-Kernel Tile-State Protocol

The single-kernel path must not rely on an external memset or a second finalization kernel. Use per-output-tile state:

```text
state value for tile:
  epoch * 3 + 0 = not initialized for this epoch
  epoch * 3 + 1 = initialization in progress
  epoch * 3 + 2 = ready for atomic accumulation
```

For each output tile:

1. A program uses `atomic_cas` to claim initialization.
2. The winner zeros `out_accum[tile]` and sets `tile_done[tile] = 0`.
3. The winner publishes ready state with release semantics.
4. Other programs spin only until the tile is ready.
5. Programs atomically add their contribution.
6. The final contributor stores the fp16 output tile.

This gives one MLP kernel launch with no global barrier and no second reduction pass. It also avoids racing atomics against a host-side zero operation.

### SM75 Constraints

Use RTX 2080 constraints as the design center:

- CUDA capability: 7.5
- dtype: fp16 input/output
- accumulation: fp32 for gate/up and `out_accum`
- no bf16 assumptions
- `BLOCK_IN = 64`
- `BLOCK_OUT = 64`
- `block_size = 32`
- `num_warps = 4`
- `num_stages = 1`

If register pressure is too high, compile a second single-kernel variant with `BLOCK_OUT = 32`. That is still one kernel launch.

### Runtime Wiring

In `_sparse_mlp_forward_fast_triton`:

1. Detect decode batch size 1.
2. Require CUDA SM75.
3. Require `block_size == 32`.
4. Require active down blocks are available in GPU packed form.
5. Allocate persistent scratch buffers once:
   - `_single_kernel_mlp_out_accum`
   - `_single_kernel_mlp_tile_state`
   - `_single_kernel_mlp_tile_done`
   - `_single_kernel_mlp_epoch`
6. Call `triton_sparse_mlp_decode_4bit_single_kernel_sm75`.
7. Set runtime status `decode_backend = "single_kernel_sparse_decode_sm75"`.

Keep the existing composite Triton path as a correctness fallback, but it must not satisfy the performance gate.

### Correctness Gate

Run MLP-only verification first:

```powershell
$env:STREAMING_GPU_LM_HEAD = "1"
python -m llama3_neuroplastic.experiments.verify_sparse_mlp_generation_pair `
  --model-name unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit `
  --sparse-basis-path results/mlp_basis_intermediate_full126.pt `
  --sparse-top-k 208 `
  --sparse-basis-top-k 64 `
  --sparse-mlp-execution exact_blockwise_sparse `
  --sparse-mlp-prefill-mode dense `
  --dtype float16 `
  --ram-cache `
  --max-new-tokens 4 `
  --json-out results/single_kernel_mlp_pair_topk208.json
```

Pass condition:

- `completion_identical = true`
- no fallback from `single_kernel_sparse_decode_sm75`
- no NaNs in hidden state or logits

### Performance Gate

```powershell
$env:STREAMING_GPU_LM_HEAD = "1"
$env:STREAMING_GPU_LM_HEAD_RESERVE_GB = "0.25"
$env:STREAMING_BACKGROUND_PREFETCH = "1"
$env:STREAMING_WINDOWS_BATCH_PRELOAD = "1"
python -m llama3_neuroplastic.experiments.benchmark `
  --model-name unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit `
  --sparse-basis-path results/mlp_basis_intermediate_full126.pt `
  --sparse-top-k 208 `
  --sparse-basis-top-k 64 `
  --sparse-mlp-execution exact_blockwise_sparse `
  --sparse-mlp-prefill-mode hot_cache `
  --vram-hot-cache-gb 5.25 `
  --pre-warm `
  --calibrate-hot-cache `
  --hot-cache-calibration-tokens 64 `
  --profile-decode `
  --profile-max-steps 8 `
  --max-new-tokens 8 `
  --warmup-tokens 2 `
  --output-json results/benchmark_single_kernel_mlp_topk208.json
```

Pass condition:

- `decode_profile_summary.mean_mlp_ms <= 150 ms/token` as the first milestone
- then tune to `<= 100 ms/token`
- no CPU LM-head fallback
- hot-cache hit counters increase during decode

---

## Change 2: Fix GPU NF4 LM Head Preference

**Files**

- `llama3_neuroplastic/experiments/streaming_llama_runtime.py`
- `llama3_neuroplastic/experiments/runtime/lm_head.py`
- `llama3_neuroplastic/experiments/runtime/_helpers.py`

**Why**

The runtime status currently shows:

```text
lm_head_mode = cpu_dense
lm_head_on_gpu = false
lm_head_last_failure = gpu_lm_head_not_preferred
```

That is wrong for the target path. A dense fp16 GPU LM head costs about **3.9 GB**, but an NF4 GPU LM head is about **2 GB** and avoids CPU logits work. The plan needs the NF4 path, not a dense GPU path and not CPU fallback.

### Current Problem

`StreamingLlamaRuntime.__init__` disables GPU LM head preference on Windows pre-Ampere unless `STREAMING_GPU_LM_HEAD` is explicitly set:

```python
if _is_windows_pre_ampere_cuda(self.device) and not self._explicit_gpu_lm_head:
    self._prefer_gpu_lm_head = False
```

That blocks `_materialize_lm_head_nf4_on_gpu()` before it can prove the quantized path fits.

### Required Fix

Change the preference logic so Windows SM75 blocks only the dense fp16 GPU LM head by default, not the NF4 GPU LM head.

Required behavior:

1. If CUDA is available, attempt NF4 LM-head materialization first.
2. If NF4 succeeds, use `lm_head_mode = gpu_nf4`.
3. If NF4 fails due to missing quant metadata, record the failure.
4. Do not load dense fp16 LM head on Windows SM75 unless explicitly requested.
5. Keep `STREAMING_GPU_LM_HEAD=0` as an explicit opt-out.

Expected status after fix:

```text
lm_head_mode = gpu_nf4
lm_head_on_gpu = true
lm_head_gpu_preferred = true
lm_head_last_failure = null
```

### Verification

Run a one-token smoke benchmark and inspect `runtime_status`:

```powershell
$env:STREAMING_GPU_LM_HEAD = "1"
$env:STREAMING_GPU_LM_HEAD_RESERVE_GB = "0.25"
python -m llama3_neuroplastic.experiments.benchmark `
  --model-name unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit `
  --sparse-basis-path results/mlp_basis_intermediate_full126.pt `
  --sparse-top-k 208 `
  --sparse-basis-top-k 64 `
  --sparse-mlp-execution exact_blockwise_sparse `
  --sparse-mlp-prefill-mode dense `
  --vram-hot-cache-gb 0 `
  --max-new-tokens 1 `
  --warmup-tokens 0 `
  --output-json results/lm_head_gpu_nf4_smoke.json
```

Pass condition:

- `runtime_status.lm_head_mode = "gpu_nf4"`
- `runtime_status.lm_head_on_gpu = true`
- no dense fp16 LM-head allocation on RTX 2080 unless forced

---

## Change 3: Sparse Attention Correctness

**Files**

- `llama3_neuroplastic/experiments/streaming_llama_runtime.py`
- `llama3_neuroplastic/token_posting_archive.py`
- `llama3_neuroplastic/experiments/sweep_sparse_mlp_coherence.py`

**Why**

Dense Q and O projections dominate attention transfer:

| Projection | Shape | NF4 size |
|---|---:|---:|
| Q | 16384 x 16384 | 134 MB |
| K | 16384 x 1024 | 8.4 MB |
| V | 16384 x 1024 | 8.4 MB |
| O | 16384 x 16384 | 134 MB |
| Total | | 285 MB/layer |

With 5 active query heads, Q and O drop to about 5.2 MB each. Attention transfer becomes about **27 MB/layer** before hot-cache effects. This is mandatory for 3.3 tok/s.

### Known Correctness Risks

1. Token archive dedup stamps must derive from `position_index`, not a mutable archive step that can drift across `reset_caches()`.
2. Q/O skeleton buffers must never retain stale rows or columns when hot-head cache partially satisfies a request.
3. Compact sparse attention must update the same logical KV cache as dense attention for decode comparisons.
4. Head-importance artifacts must be treated as candidate artifacts, not assumed coherent at 5 heads.

### Required Fixes

Token archive:

- Keep `archive.fetch_shortlist_kv(layer_idx, group_idx, q_rep_gpu, position_index * 200 + layer_idx, M=self._retrieval_candidates)`.
- Keep `archive.append_token(layer_idx, position_index, k_new_cpu, v_new_cpu)`.
- Keep `self._token_archive.reset()` unconditional in `reset_caches()`.
- Remove any remaining use of `archive.step` from runtime logic. The field can remain inside `TokenPostingArchive` if it is unused.

Q/O sparse buffers:

- Call `_clear_sparse_attn_qo_buffers(q_skel=q_skel, o_skel=o_skel, force_full=force_full)` before hot-head lookup.
- Preserve loaded-row and loaded-column tracking on every hot-cache and cold-load path.
- If tracking is missing while state is sparse, promote to a full clear for that call.
- Add a debug assertion mode that checks inactive Q rows and inactive O columns are zero after `_load_sparse_attn_heads`.

Compact decode:

- Verify `_forward_compact_sparse_attn` writes and reads cache entries by `layer_idx`.
- Verify active heads are unique, sorted, and mapped to the correct GQA KV group.
- Verify `o_proj` receives compact heads in the same order used to gather O columns.

### Attention Verification

Use a dense-attention baseline and compare sparse-attention runs with identical MLP settings:

```powershell
$env:STREAMING_GPU_LM_HEAD = "1"
python -m llama3_neuroplastic.experiments.sweep_sparse_mlp_coherence `
  --model-name unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit `
  --sparse-basis-path results/mlp_basis_intermediate_full126.pt `
  --top-k-list 208 `
  --sparse-basis-top-k 64 `
  --sparse-mlp-execution exact_blockwise_sparse `
  --sparse-mlp-prefill-mode dense `
  --use-sparse-attention `
  --attn-head-importance-path results/attn_head_importance_405b_repaired.pt `
  --attn-active-heads 5 `
  --attn-min-active-heads 5 `
  --attn-max-active-heads 5 `
  --sparse-attn-prefill-mode sparse `
  --kv-basis-path results/kv_basis_r32.pt `
  --sparse-kv-prefill-mode sparse `
  --max-new-tokens 4 `
  --output-json results/sparse_attn_heads5_mlp208_verify.json
```

Pass condition:

- `completion_identical = true` against the dense-attention baseline for 4 tokens
- `compact_sparse_attention_steps > 0`
- decode traffic shows sparse Q/O tags instead of dense Q/O layer loads

### Head Sweep

Do not assume 5 heads is coherent until it passes. Sweep:

```text
active_heads = 5, 8, 12, 16, 24, 32, 48, 62
top_k = 208
max_new_tokens = 4 first, then 8
```

The target remains 5 active heads, but the engineering gate is the lowest head count that is coherent. If the lowest coherent setting is above 5, repair or regenerate the head-importance artifact rather than accepting incoherent output.

---

## Change 4: Coherence-Verified MLP top_k and Hot-Cache Sweep

**File**

- `llama3_neuroplastic/experiments/sweep_sparse_mlp_coherence.py`

**Why**

The hot-cache benchmark uses top_k about 51, which is not a valid coherence setting for the 405B path. The coherent dense-attention evidence is closer to top_k=208 or higher. We need the lowest coherent top_k after the single-kernel MLP and sparse-attention fixes are in place.

### Required Sweep

Run after Changes 1 to 3:

```text
top_k: 128, 160, 192, 208, 256
hot_cache_gb: 3.0, 4.0, 5.25
active_heads: lowest coherent value from Change 3
max_new_tokens: 8
```

Record for each run:

- completion token match against dense baseline
- decode tok/s
- mean MLP ms/token
- mean attention-load ms/token
- hot MLP blocks hit
- cold MLP blocks streamed
- down hot blocks hit
- down cold blocks streamed
- VRAM peak

### Acceptance Rule

Pick the fastest setting that satisfies:

- exact completion match for 8 deterministic tokens, or edit distance <= 1 token with manually coherent text
- no CPU LM-head fallback
- no composite Triton MLP fallback
- sparse attention actually used during decode

---

## Priority Order

| Priority | Change | Expected impact | Gate |
|---:|---|---:|---|
| 1 | Single-kernel sparse MLP decode | Largest latency reduction | `mean_mlp_ms <= 150 ms/token` first milestone |
| 2 | GPU NF4 LM head preference | Removes CPU logits and frees hot-cache budget versus fp16 GPU head | `lm_head_mode = gpu_nf4` |
| 3 | Sparse attention correctness | Removes dense Q/O load | sparse-attn completion match |
| 4 | top_k, head-count, hot-cache sweep | Finds coherent fast regime | fastest coherent 8-token decode |

---

## Final Throughput Gate

Run the final target probe:

```powershell
$env:STREAMING_GPU_LM_HEAD = "1"
$env:STREAMING_GPU_LM_HEAD_RESERVE_GB = "0.25"
$env:STREAMING_BACKGROUND_PREFETCH = "1"
$env:STREAMING_WINDOWS_BATCH_PRELOAD = "1"
$env:STREAMING_SHOW_PROGRESS = "1"
python -m llama3_neuroplastic.experiments.benchmark `
  --model-name unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit `
  --local-files-only `
  --sparse-basis-path results/mlp_basis_intermediate_full126.pt `
  --sparse-top-k 208 `
  --sparse-basis-top-k 64 `
  --sparse-mlp-execution exact_blockwise_sparse `
  --sparse-mlp-prefill-mode hot_cache `
  --vram-hot-cache-gb 5.25 `
  --pre-warm `
  --calibrate-hot-cache `
  --hot-cache-calibration-tokens 64 `
  --attn-head-importance-path results/attn_head_importance_405b_repaired.pt `
  --attn-active-heads 5 `
  --attn-min-active-heads 5 `
  --attn-max-active-heads 5 `
  --sparse-attn-prefill-mode sparse `
  --kv-basis-path results/kv_basis_r32.pt `
  --sparse-kv-prefill-mode sparse `
  --profile-decode `
  --profile-max-steps 16 `
  --max-new-tokens 16 `
  --warmup-tokens 4 `
  --output-json results/benchmark_3_3_final_probe.json
```

Pass conditions:

- `decode_tokens_per_second >= 3.3`
- output is coherent on the prompt
- `runtime_status.decode_backend = "single_kernel_sparse_decode_sm75"`
- `runtime_status.lm_head_mode = "gpu_nf4"`
- `runtime_status.compact_sparse_attention_steps > 0`
- `decode_profile_summary.mean_mlp_ms` is below the single-kernel milestone
- `decode_profile_summary.mean_load_attn_ms` reflects sparse Q/O loading, not dense Q/O loading

If 5 attention heads fails coherence, rerun the final probe with the lowest coherent head count from the head sweep, then repair the head-importance artifact until 5 heads passes.
