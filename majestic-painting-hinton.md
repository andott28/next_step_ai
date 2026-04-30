# 5-Layer Token Time: Root Cause Analysis & Fix Plan

## Context

The 5-layer token-time benchmark (`tests/run_5layer_token_time_with_runtime.py`) measures decode latency for layers 0-4 after seeding 256 tokens into KV cache. Results: **250 ms actual vs 12 ms target** (20x slower). The run also crashed the PC, indicating a memory leak. Target is 3.3 tok/s → 303 ms for all 126 layers → 2.4 ms/layer.

---

## Root Causes (ordered by severity)

### 1. CRITICAL — Cache cleared every layer, then immediately re-prefetched
**File:** `streaming_llama_runtime.py:7725-7737`

```python
if str(self._traffic_current_phase or "idle") == "decode":
    self.clear_sparse_transfer_caches()          # wipes everything
    if self._enable_decode_lookahead_prefetch:    # then immediately re-prefetches
        self._prefetch_mlp_blocks_for_layer(...)
```

`clear_sparse_transfer_caches()` (`session.py:160-169`) destroys `_sparse_block_transfer_cache`, `_downproj_transfer_cache`, AND `_mlp_prefetch_active_blocks` on **every layer of every token**. This means:
- H2D-transferred GPU tensors from prior layers are discarded — no reuse across layers
- The lookahead prefetch scheduled on line 7736-7737 is immediately wiped by the *next* layer's `clear_sparse_transfer_caches()` call at line 7726
- Every layer re-loads weights from CPU/SSD instead of hitting the transfer cache

**Estimated impact:** ~40-45 ms/layer wasted on redundant H2D transfers (accounts for bulk of the 50 ms/layer actual)

### 2. CRITICAL — Synchronous `torch.cuda.synchronize()` on every token
**File:** `streaming_llama_runtime.py:7772-7773`

```python
if _ev_lm_head_end is not None:
    _ev_lm_head_end.record()
    with contextlib.suppress(Exception):
        torch.cuda.synchronize(self.device)   # blocks CPU every token
```

This fires unconditionally whenever `_track_token_overhead` is True (line 7532: always True on CUDA). It:
- Forces CPU to wait for ALL GPU work to finish after every token
- Prevents any CPU-side prep (routing, cache lookups) from overlapping with GPU compute
- Adds ~2-5 ms overhead per token from the sync + Python timing code

### 3. HIGH — GPU event creation overhead in hot loop
**File:** `streaming_llama_runtime.py:7538-7544, 7563-7568`

5 top-level events created per token + 4 per-layer events = **25 events for 5 layers**. `torch.cuda.Event(enable_timing=True)` allocates CUDA resources each call. Per-layer events (lines 7563-7568) are only used when `profile_step is not None`, but the top-level events are always created.

### 4. HIGH — Memory leak: `_down_proj_col_cache` grows unbounded
**File:** `streaming_llama_runtime.py:4420-4449`

`_down_proj_col_cache[layer][block_id] = (col_slice, abs_slice)` stores ~320 KB per block. With 17 sparse layers x up to 1664 blocks = **~8.9 GB max**. No eviction policy, never cleared in `reset_caches()`. During the 256-token seeding loop this cache accumulates rapidly, which explains the PC crash.

### 5. MEDIUM — CPU→GPU sync in routing/KV path
**File:** `streaming_llama_runtime.py:3351-3353, 3390`

```python
blocks_cpu = active_col_blocks[0].cpu()          # GPU→CPU sync
blocks_cpu = active_col_blocks.reshape(-1).cpu()  # another sync
```

Multiple `.cpu()` calls on GPU tensors in the per-layer hot path cause implicit `cudaStreamSynchronize`.

### 6. LOW — `_materialize_sparse_4bit_param_raw_views` loads full 436 MB weight
**File:** `streaming_llama_runtime.py:4381-4394`

When `_down_proj_col_cache` has misses, `_load_raw_for_param()` loads the entire weight tensor (~436 MB) just to slice a few blocks. This is mitigated by the RAM cache after the first load, but the initial materialization + reshape is expensive. The col_cache (issue #4) was designed to fix this, but its unbounded growth causes the crash.

---

## Fix Plan

### Fix 1: Move cache clearing to token boundary (not per-layer)
**File:** `streaming_llama_runtime.py`

- Remove `self.clear_sparse_transfer_caches()` from line 7726 (inside the layer loop)
- Add it **once before** the layer loop starts (around line 7556), so each token starts fresh but caches persist across layers within a single token
- This allows lookahead prefetch from layer N to be consumed by layer N+1

### Fix 2: Gate the synchronize + profiling behind an opt-in flag
**File:** `streaming_llama_runtime.py:7532, 7770-7797`

- Change `_track_token_overhead` from always-on to gated by a flag (e.g., `self._profile_decode_overhead`)
- Default the flag to `False`
- When False: skip event creation (lines 7539-7543), skip sync + timing code (lines 7772-7797)
- The per-layer profiling events (7563-7568) are already gated on `profile_step` — no change needed

### Fix 3: Cap `_down_proj_col_cache` size (memory leak fix)
**File:** `streaming_llama_runtime.py` (near `_load_down_proj_cold_cols`)

- Add an LRU eviction policy or a max-entries-per-layer cap to `_down_proj_col_cache`
- Reasonable cap: ~64 blocks per layer (64 x 320 KB x 17 layers = ~348 MB total) — covers the hot working set
- Add `_down_proj_col_cache.clear()` to `reset_caches()` in `session.py:171` so it's freed between sessions

### Fix 4: Keep routing block indices on CPU from the start
**File:** `streaming_llama_runtime.py:3350-3354`

- The routing result `active_col_blocks` should be produced on CPU or cached as CPU tensor, avoiding the `.cpu()` sync in the KV sparse path
- If the routing kernel outputs on GPU, do a single `.cpu()` call at the routing site and pass the CPU tensor downstream

---

## Files to Modify

| File | Changes |
|------|---------|
| `streaming_llama_runtime.py:7725-7726` | Move `clear_sparse_transfer_caches()` out of layer loop |
| `streaming_llama_runtime.py:7532-7544, 7770-7797` | Gate profiling behind opt-in flag |
| `streaming_llama_runtime.py:4420-4449` | Add LRU cap to `_down_proj_col_cache` |
| `session.py:171-200` | Add `_down_proj_col_cache.clear()` to `reset_caches()` |
| `streaming_llama_runtime.py:3350-3354` | Eliminate redundant GPU→CPU transfers |

---

## Verification

1. Run the 5-layer benchmark:
   ```
   $env:PYTHONUNBUFFERED='1'; $env:PYTHONPATH='.'; .\verification_env\Scripts\python.exe tests/run_5layer_token_time_with_runtime.py
   ```
2. Check mean 5-layer time < 12 ms, per-layer < 2.4 ms, projected 126-layer < 310 ms
3. Monitor VRAM/RAM during seeding (256 tokens) — should stay stable, not grow unboundedly
4. Verify no PC crash / OOM during the full run

## Expected Impact

| Fix | Estimated savings per layer |
|-----|-----------------------------|
| Cache clearing → token boundary | ~40-45 ms (eliminates redundant H2D) |
| Profiling gate | ~2-5 ms per token total |
| Col cache cap | Prevents OOM crash |
| CPU routing fix | ~0.5-1 ms per layer |
| **Total** | **~45-50 ms/layer → target 2.4 ms/layer** |

Fix 1 alone should account for the majority of the 20x slowdown. The others are important for stability (Fix 3 prevents crash) and reaching the 2.4 ms/layer target.
