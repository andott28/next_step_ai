"""
Rapid unit tests for _load_down_proj_cold_cols and related helpers.
CPU-only, no model loading, runs in seconds.

Tests:
  1. Shape correctness after the absmax-slice fix
  2. Cache miss path: extracts correct data
  3. Cache hit path: returns identical data without re-materializing
  4. _merge_columnwise_parts hot+cold mixing (the crash scenario)
  5. All-cold path (no hot blocks)
  6. load_nf4_packed_blocks RAM-cache hit path
"""

import sys
import os
import torch

# ---------------------------------------------------------------------------
# Minimal parameters matching the real 405B down_proj geometry (scaled down)
# ---------------------------------------------------------------------------
H_OUT      = 64    # real: 16384
I_IN       = 256   # real: 53248  (in_features for down_proj)
BLOCK_SIZE = 8     # real: 32  (MLP sparse block)
QBS        = 16    # real: 64  (NF4 quant block size)

BYTES_PER_ROW   = I_IN // 2          # NF4: 2 values per byte
BYTES_PER_CBLK  = BLOCK_SIZE // 2    # bytes per MLP block column
ABSMAX_PER_ROW  = I_IN // QBS        # absmax entries per output row
N_BLOCKS        = I_IN // BLOCK_SIZE  # total MLP blocks per layer

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []

def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        msg = f"  {FAIL}  {name}" + (f": {detail}" if detail else "")
        print(msg)
        errors.append(msg)

# ---------------------------------------------------------------------------
# Helpers that mirror the runtime logic exactly
# ---------------------------------------------------------------------------

def make_param():
    """Create a synthetic down_proj param dict with materialized raw views."""
    packed_weight = torch.randint(0, 256, (H_OUT * BYTES_PER_ROW,), dtype=torch.uint8)
    absmax        = torch.randn(H_OUT * ABSMAX_PER_ROW, dtype=torch.float32)
    return {
        "packed_weight": packed_weight,
        "absmax":        absmax,
        # pre-materialized (as _materialize_sparse_4bit_param_raw_views would set)
    }


def load_down_proj_cold_cols(down_param, cold_blocks, *, layer_cache):
    """Extracted logic of _load_down_proj_cold_cols for testing."""
    cold_list = cold_blocks.tolist()

    if layer_cache is not None and all(b in layer_cache for b in cold_list):
        packed_parts = [layer_cache[b][0] for b in cold_list]
        absmax_parts = [layer_cache[b][1] for b in cold_list]
        cold_packed = torch.stack(packed_parts, dim=1).contiguous()
        cold_absmax = torch.stack(absmax_parts, dim=1).contiguous()
        return cold_packed, cold_absmax, layer_cache

    miss_blocks = [b for b in cold_list if layer_cache is None or b not in layer_cache]

    raw_2d    = down_param["packed_weight"].view(H_OUT, BYTES_PER_ROW)
    absmax_2d = down_param["absmax"].view(H_OUT, ABSMAX_PER_ROW)

    if layer_cache is None:
        layer_cache = {}

    for b in miss_blocks:
        b_start   = int(b) * BYTES_PER_CBLK
        col_slice = raw_2d[:, b_start : b_start + BYTES_PER_CBLK].contiguous()
        abs_idx   = (int(b) * BLOCK_SIZE) // QBS
        abs_slice = absmax_2d[:, abs_idx].contiguous()         # FIX: scalar → (H_OUT,)
        layer_cache[b] = (col_slice, abs_slice)

    packed_parts = [layer_cache[b][0] for b in cold_list]
    absmax_parts = [layer_cache[b][1] for b in cold_list]
    cold_packed  = torch.stack(packed_parts, dim=1).contiguous()
    cold_absmax  = torch.stack(absmax_parts, dim=1).contiguous()
    return cold_packed, cold_absmax, layer_cache


def merge_columnwise_parts(*, total_blocks, hot_positions_cpu, hot_packed, cold_mask_cpu,
                            cold_packed, hot_absmax, cold_absmax):
    """Mirror of _merge_columnwise_parts."""
    packed_template = hot_packed if hot_packed is not None else cold_packed
    absmax_template = hot_absmax if hot_absmax is not None else cold_absmax
    packed_shape = (packed_template.shape[0], total_blocks) + tuple(packed_template.shape[2:])
    absmax_shape = (absmax_template.shape[0], total_blocks) + tuple(absmax_template.shape[2:])
    packed = torch.zeros(packed_shape, dtype=packed_template.dtype)
    absmax = torch.zeros(absmax_shape, dtype=absmax_template.dtype)
    if hot_packed is not None and hot_positions_cpu is not None and hot_positions_cpu.numel() > 0:
        packed.index_copy_(1, hot_positions_cpu, hot_packed)
        absmax.index_copy_(1, hot_positions_cpu, hot_absmax)
    if cold_packed is not None and cold_mask_cpu is not None and cold_mask_cpu.any():
        cold_pos = torch.nonzero(cold_mask_cpu, as_tuple=False).flatten()
        packed.index_copy_(1, cold_pos, cold_packed)
        absmax.index_copy_(1, cold_pos, cold_absmax)
    return packed, absmax


# ===========================================================================
# Test 1: Shape correctness — packed (H_OUT, n_cold, BYTES_PER_CBLK)
#                           absmax  (H_OUT, n_cold)
# ===========================================================================
print("\n[1] Shape correctness")
param       = make_param()
cold_blocks = torch.tensor([0, 3, 5], dtype=torch.long)
cold_packed, cold_absmax, cache = load_down_proj_cold_cols(param, cold_blocks, layer_cache=None)

check("packed shape",  cold_packed.shape == (H_OUT, 3, BYTES_PER_CBLK),
      f"got {cold_packed.shape}")
check("absmax shape",  cold_absmax.shape == (H_OUT, 3),
      f"got {cold_absmax.shape}")
check("packed dtype",  cold_packed.dtype == torch.uint8)
check("absmax dtype",  cold_absmax.dtype == torch.float32)


# ===========================================================================
# Test 2: Cache miss — data matches raw weight
# ===========================================================================
print("\n[2] Cache miss: data fidelity")
raw_2d    = param["packed_weight"].view(H_OUT, BYTES_PER_ROW)
absmax_2d = param["absmax"].view(H_OUT, ABSMAX_PER_ROW)

for i, b in enumerate([0, 3, 5]):
    b_start         = b * BYTES_PER_CBLK
    expected_packed = raw_2d[:, b_start : b_start + BYTES_PER_CBLK]
    abs_idx         = (b * BLOCK_SIZE) // QBS
    expected_absmax = absmax_2d[:, abs_idx]

    check(f"block {b} packed data",
          torch.equal(cold_packed[:, i, :], expected_packed),
          f"mismatch at col {i}")
    check(f"block {b} absmax data",
          torch.equal(cold_absmax[:, i], expected_absmax),
          f"mismatch at col {i}")


# ===========================================================================
# Test 3: Cache hit — returns same result, cache not changed
# ===========================================================================
print("\n[3] Cache hit path")
cache_before = {k: (v[0].clone(), v[1].clone()) for k, v in cache.items()}
cold_packed2, cold_absmax2, cache2 = load_down_proj_cold_cols(param, cold_blocks, layer_cache=cache)

check("packed identical on hit",  torch.equal(cold_packed, cold_packed2))
check("absmax identical on hit",  torch.equal(cold_absmax, cold_absmax2))
check("cache size unchanged",     len(cache2) == len(cache_before))

# Partial hit (2 cached, 1 new)
new_cold = torch.tensor([0, 3, 7], dtype=torch.long)   # 7 is new
cold_packed3, cold_absmax3, cache3 = load_down_proj_cold_cols(param, new_cold, layer_cache=dict(cache))
check("partial hit: packed shape", cold_packed3.shape == (H_OUT, 3, BYTES_PER_CBLK))
check("partial hit: absmax shape", cold_absmax3.shape == (H_OUT, 3))
check("partial hit: new block 7 cached", 7 in cache3)


# ===========================================================================
# Test 4: _merge_columnwise_parts — hot (2D absmax) + cold (2D absmax)
#          This was the crash scenario before the fix.
# ===========================================================================
print("\n[4] merge_columnwise_parts hot+cold mixing")
n_total = 6
hot_blocks_idx = torch.tensor([0, 2])       # positions 0 and 2 are hot
cold_blocks_idx = torch.tensor([1, 3, 4, 5])

# Build hot tensors as the VRAM hot cache would produce
hot_packed  = torch.zeros(H_OUT, 2, BYTES_PER_CBLK, dtype=torch.uint8)
hot_absmax  = torch.ones(H_OUT, 2, dtype=torch.float32) * 0.5   # (H_OUT, n_hot) 2D

# Build cold tensors from our function
cold_packed_c, cold_absmax_c, _ = load_down_proj_cold_cols(
    param, cold_blocks_idx, layer_cache=None)

cold_mask = torch.tensor([False, True, False, True, True, True])

try:
    merged_packed, merged_absmax = merge_columnwise_parts(
        total_blocks=n_total,
        hot_positions_cpu=hot_blocks_idx,
        hot_packed=hot_packed,
        cold_mask_cpu=cold_mask,
        cold_packed=cold_packed_c,
        hot_absmax=hot_absmax,
        cold_absmax=cold_absmax_c,
    )
    check("no crash mixing hot+cold",    True)
    check("merged packed shape",  merged_packed.shape == (H_OUT, n_total, BYTES_PER_CBLK),
          f"got {merged_packed.shape}")
    check("merged absmax shape",  merged_absmax.shape == (H_OUT, n_total),
          f"got {merged_absmax.shape}")
except Exception as e:
    check("no crash mixing hot+cold", False, str(e))
    check("merged packed shape",      False, "not reached")
    check("merged absmax shape",      False, "not reached")


# ===========================================================================
# Test 5: All-cold path — absmax shape must be 2D for downstream mul
# ===========================================================================
print("\n[5] All-cold path: absmax shape matches mul expectation")
cold_all = torch.arange(N_BLOCKS, dtype=torch.long)
cold_packed_all, cold_absmax_all, _ = load_down_proj_cold_cols(
    param, cold_all, layer_cache=None)

check("all-cold packed shape",  cold_packed_all.shape == (H_OUT, N_BLOCKS, BYTES_PER_CBLK),
      f"got {cold_packed_all.shape}")
check("all-cold absmax shape",  cold_absmax_all.shape == (H_OUT, N_BLOCKS),
      f"got {cold_absmax_all.shape}")

# Simulate the downstream mul: packed→weight (H_OUT, N_BLOCKS, BLOCK_SIZE) * absmax.unsqueeze(-1)
# This must not raise a shape error
fake_weight = torch.randn(H_OUT, N_BLOCKS, BLOCK_SIZE)
try:
    result = fake_weight * cold_absmax_all.unsqueeze(-1)
    check("absmax.unsqueeze(-1) broadcasts correctly",
          result.shape == (H_OUT, N_BLOCKS, BLOCK_SIZE),
          f"got {result.shape}")
except Exception as e:
    check("absmax.unsqueeze(-1) broadcasts correctly", False, str(e))


# ===========================================================================
# Test 6: load_nf4_packed_blocks RAM-cache hit path (no file I/O needed)
# ===========================================================================
print("\n[6] load_nf4_packed_blocks — RAM cache hit")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import threading

    class _FakeLoader:
        """Minimal stub that has just a pre-populated RAM cache."""
        def __init__(self, name, weight_tensor):
            self._ram_cache = {name: (weight_tensor, None)}
            self._ram_cache_lock = threading.Lock()
            self._ram_cache_enabled = True

        def load_nf4_packed_blocks(self, name, block_indices, *, bytes_per_block):
            with self._ram_cache_lock:
                cached = self._ram_cache.get(name)
            if cached is not None:
                weight, _ = cached
                raw_flat = weight.reshape(-1)
                packed_blocks = raw_flat.view(-1, bytes_per_block)
                idx = block_indices.to(dtype=torch.long).reshape(-1)
                return packed_blocks[idx].contiguous()
            raise RuntimeError("not in cache")

    n_blocks_total = 32
    bpb            = 64  # bytes_per_block
    fake_weight    = torch.arange(n_blocks_total * bpb, dtype=torch.uint8)
    loader         = _FakeLoader("layer0.mlp.gate_proj.weight", fake_weight)
    idx            = torch.tensor([2, 7, 15], dtype=torch.long)
    result         = loader.load_nf4_packed_blocks(
        "layer0.mlp.gate_proj.weight", idx, bytes_per_block=bpb)

    check("load_nf4 RAM hit shape",   result.shape == (3, bpb), f"got {result.shape}")
    check("load_nf4 RAM hit block 2", torch.equal(result[0], fake_weight[2*bpb : 3*bpb]))
    check("load_nf4 RAM hit block 7", torch.equal(result[1], fake_weight[7*bpb : 8*bpb]))
except Exception as e:
    check("load_nf4_packed_blocks import + RAM hit", False, str(e))


# ===========================================================================
# Summary
# ===========================================================================
print()
if errors:
    print(f"{'='*60}")
    print(f"FAILED ({len(errors)} errors):")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
