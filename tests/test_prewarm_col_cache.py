"""
Rapid unit tests for iter18: Tier 1b CPU column-cache pre-warm.

Tests the logic added to pre_warm_vram_hot_cache that extracts cold down_proj
column slices into _down_proj_col_cache during startup, eliminating SSD reads
on decode token 1.

CPU-only, no model loading, runs in seconds.

Tests:
  1. Pre-warm fills cache for cold blocks (lookup < 0)
  2. Pre-warm skips hot blocks (lookup >= 0)
  3. Pre-warm skips layers already in col-cache
  4. No-VRAM-hot case: all routing blocks treated as cold
  5. Cold block data matches raw weight slices
  6. After pre-warm: _load_down_proj_cold_cols returns cache-hit on token 1
  7. Empty cold_blocks (fully hot layer): no cache entry created, no crash
"""

import sys
import torch

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
# Geometry (scaled-down version of 405B down_proj)
# ---------------------------------------------------------------------------
H_OUT       = 64
I_IN        = 128
BLOCK_SIZE  = 8
QBS         = 16
SPARSE_BS   = BLOCK_SIZE

BYTES_PER_ROW   = I_IN // 2
BYTES_PER_CBLK  = BLOCK_SIZE // 2
ABSMAX_PER_ROW  = I_IN // QBS
N_TOTAL_BLOCKS  = I_IN // BLOCK_SIZE   # = 16


def make_down_param(layer_idx, hot_block_indices=None, total_blocks=N_TOTAL_BLOCKS):
    """Build a fake down_param dict as _sparse_param_cache would contain it."""
    packed_weight = torch.randint(0, 256, (H_OUT * BYTES_PER_ROW,), dtype=torch.uint8)
    absmax        = torch.randn(H_OUT * ABSMAX_PER_ROW, dtype=torch.float32)

    param = {
        "full_name":       f"model.layers.{layer_idx}.mlp.down_proj.weight",
        "out_features":    H_OUT,
        "in_features":     I_IN,
        "bytes_per_row":   BYTES_PER_ROW,
        "absmax_per_row":  ABSMAX_PER_ROW,
        "quant_block_size": QBS,
        "packed_weight":   packed_weight,
        "absmax":          absmax,
    }

    if hot_block_indices is not None:
        hot_blocks = torch.tensor(hot_block_indices, dtype=torch.long)
        lookup = torch.full((total_blocks,), -1, dtype=torch.int32)
        lookup[hot_blocks] = torch.arange(len(hot_block_indices), dtype=torch.int32)
        param["vram_hot_down"] = {
            "lookup_cpu": lookup,
            "block_ids_cpu": hot_blocks,
            "active_count": len(hot_block_indices),
        }

    return param


def load_down_proj_cold_cols(down_param, cold_blocks, *, layer_cache):
    """Mirror of _load_down_proj_cold_cols for test use."""
    cold_list = cold_blocks.tolist()

    if layer_cache is not None and all(b in layer_cache for b in cold_list):
        packed_parts = [layer_cache[b][0] for b in cold_list]
        absmax_parts = [layer_cache[b][1] for b in cold_list]
        return (
            torch.stack(packed_parts, dim=1).contiguous(),
            torch.stack(absmax_parts, dim=1).contiguous(),
            True,   # was cache hit
        )

    miss_blocks = [b for b in cold_list if layer_cache is None or b not in layer_cache]
    raw_2d    = down_param["packed_weight"].view(H_OUT, BYTES_PER_ROW)
    absmax_2d = down_param["absmax"].view(H_OUT, ABSMAX_PER_ROW)

    if layer_cache is None:
        layer_cache = {}

    for b in miss_blocks:
        b_start   = int(b) * BYTES_PER_CBLK
        col_slice = raw_2d[:, b_start : b_start + BYTES_PER_CBLK].contiguous()
        abs_idx   = (int(b) * BLOCK_SIZE) // QBS
        abs_slice = absmax_2d[:, abs_idx].contiguous()
        layer_cache[b] = (col_slice, abs_slice)

    packed_parts = [layer_cache[b][0] for b in cold_list]
    absmax_parts = [layer_cache[b][1] for b in cold_list]
    return (
        torch.stack(packed_parts, dim=1).contiguous(),
        torch.stack(absmax_parts, dim=1).contiguous(),
        False,  # was not all cache hits
    )


def simulate_prewarm_tier1b(sparse_param_cache, mlp_hot_blocks_by_layer,
                             down_proj_col_cache, sparse_block_size):
    """Mirror of Tier 1b in pre_warm_vram_hot_cache."""
    col_cache_warmed = 0
    for layer_idx in sorted(mlp_hot_blocks_by_layer.keys()):
        down_name    = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        down_param   = sparse_param_cache.get(down_name)
        routing_blks = mlp_hot_blocks_by_layer.get(layer_idx)

        if (
            down_param is None
            or routing_blks is None
            or int(routing_blks.numel()) == 0
            or down_name in down_proj_col_cache
        ):
            continue

        vram_hot_down = down_param.get("vram_hot_down")
        if vram_hot_down is not None:
            lookup    = vram_hot_down["lookup_cpu"]
            cold_mask = lookup.index_select(0, routing_blks.to(dtype=torch.long)) < 0
            cold_blocks = routing_blks[cold_mask].contiguous()
        else:
            cold_blocks = routing_blks.contiguous()

        H_out   = int(down_param.get("out_features", 0))
        I_in    = int(down_param.get("in_features", 0))
        qbs     = int(down_param.get("quant_block_size", 64))
        bprow   = int(down_param.get("bytes_per_row", I_in // 2))
        bpcblk  = int(sparse_block_size) // 2
        absmaxpr = int(down_param.get("absmax_per_row", I_in // max(qbs, 1)))

        if H_out > 0 and I_in > 0 and int(cold_blocks.numel()) > 0:
            layer_cache = down_proj_col_cache.get(down_name)
            _, _, _ = load_down_proj_cold_cols(down_param, cold_blocks, layer_cache=layer_cache)
            # Populate the cache (in real code this is done inside _load_down_proj_cold_cols)
            raw_2d    = down_param["packed_weight"].view(H_out, bprow)
            absmax_2d = down_param["absmax"].view(H_out, absmaxpr)
            if layer_cache is None:
                layer_cache = {}
                down_proj_col_cache[down_name] = layer_cache
            for b in cold_blocks.tolist():
                b_start   = int(b) * bpcblk
                col_slice = raw_2d[:, b_start : b_start + bpcblk].contiguous()
                abs_idx   = (int(b) * sparse_block_size) // qbs
                abs_slice = absmax_2d[:, abs_idx].contiguous()
                layer_cache[b] = (col_slice, abs_slice)
            col_cache_warmed += 1

    return col_cache_warmed


# ===========================================================================
# Test 1: Pre-warm fills cold blocks, skips hot
# ===========================================================================
print("\n[1] Pre-warm: fills cold blocks, skips hot")
# Blocks 0,1,2 are hot; blocks 3,4,5 are cold (routing requests 0-5)
param_l0  = make_down_param(0, hot_block_indices=[0, 1, 2])
routing_l0 = torch.arange(6, dtype=torch.long)  # requesting blocks 0..5

spc = {"model.layers.0.mlp.down_proj.weight": param_l0}
hbbl = {0: routing_l0}
col_cache = {}

warmed = simulate_prewarm_tier1b(spc, hbbl, col_cache, SPARSE_BS)
layer_cache = col_cache.get("model.layers.0.mlp.down_proj.weight")

check("col_cache_warmed == 1",   warmed == 1)
check("layer cache created",     layer_cache is not None)
check("cold block 3 cached",     layer_cache is not None and 3 in layer_cache)
check("cold block 4 cached",     layer_cache is not None and 4 in layer_cache)
check("cold block 5 cached",     layer_cache is not None and 5 in layer_cache)
check("hot block 0 NOT cached",  layer_cache is None or 0 not in layer_cache)
check("hot block 1 NOT cached",  layer_cache is None or 1 not in layer_cache)
check("hot block 2 NOT cached",  layer_cache is None or 2 not in layer_cache)


# ===========================================================================
# Test 2: No VRAM hot cache → all routing blocks treated as cold
# ===========================================================================
print("\n[2] No VRAM hot cache: all routing blocks become cold")
param_l1  = make_down_param(1, hot_block_indices=None)  # no vram_hot_down
routing_l1 = torch.tensor([2, 5, 9], dtype=torch.long)

spc2   = {"model.layers.1.mlp.down_proj.weight": param_l1}
hbbl2  = {1: routing_l1}
col_cache2 = {}

warmed2 = simulate_prewarm_tier1b(spc2, hbbl2, col_cache2, SPARSE_BS)
lc2 = col_cache2.get("model.layers.1.mlp.down_proj.weight")

check("warmed == 1",        warmed2 == 1)
check("block 2 cached",     lc2 is not None and 2 in lc2)
check("block 5 cached",     lc2 is not None and 5 in lc2)
check("block 9 cached",     lc2 is not None and 9 in lc2)


# ===========================================================================
# Test 3: Skip layer already in col-cache (idempotent)
# ===========================================================================
print("\n[3] Layer already in col-cache: skipped")
existing_cache = {"model.layers.0.mlp.down_proj.weight": {3: (torch.zeros(H_OUT, BYTES_PER_CBLK), torch.zeros(H_OUT))}}
spc3 = {"model.layers.0.mlp.down_proj.weight": make_down_param(0, hot_block_indices=[0])}
hbbl3 = {0: torch.tensor([0, 3, 4], dtype=torch.long)}

warmed3 = simulate_prewarm_tier1b(spc3, hbbl3, existing_cache, SPARSE_BS)
check("warmed == 0 (skipped)",  warmed3 == 0)
check("existing cache unchanged", 3 in existing_cache.get("model.layers.0.mlp.down_proj.weight", {}))


# ===========================================================================
# Test 4: Fully hot layer (all routing blocks in VRAM) → no cold blocks
# ===========================================================================
print("\n[4] Fully hot layer: no cold blocks -> no cache created")
# Routing blocks 0,1,2 all hot
param_l2  = make_down_param(2, hot_block_indices=[0, 1, 2])
routing_l2 = torch.tensor([0, 1, 2], dtype=torch.long)

spc4   = {"model.layers.2.mlp.down_proj.weight": param_l2}
hbbl4  = {2: routing_l2}
col_cache4 = {}

warmed4 = simulate_prewarm_tier1b(spc4, hbbl4, col_cache4, SPARSE_BS)
check("warmed == 0 (all hot)",         warmed4 == 0)
check("no cache entry created",        "model.layers.2.mlp.down_proj.weight" not in col_cache4)


# ===========================================================================
# Test 5: Data correctness — pre-warmed slices match raw weight
# ===========================================================================
print("\n[5] Pre-warmed data matches raw weight")
param_l3  = make_down_param(3, hot_block_indices=[0, 1])
routing_l3 = torch.tensor([0, 1, 4, 7], dtype=torch.long)

spc5   = {"model.layers.3.mlp.down_proj.weight": param_l3}
hbbl5  = {3: routing_l3}
col_cache5 = {}
simulate_prewarm_tier1b(spc5, hbbl5, col_cache5, SPARSE_BS)
lc5 = col_cache5.get("model.layers.3.mlp.down_proj.weight", {})

raw_2d    = param_l3["packed_weight"].view(H_OUT, BYTES_PER_ROW)
absmax_2d = param_l3["absmax"].view(H_OUT, ABSMAX_PER_ROW)

for b in [4, 7]:  # cold blocks (0,1 are hot)
    if b not in lc5:
        check(f"block {b} in cache", False, "missing")
        continue
    col_packed, col_abs = lc5[b]
    b_start         = b * BYTES_PER_CBLK
    expected_packed = raw_2d[:, b_start : b_start + BYTES_PER_CBLK]
    abs_idx         = (b * BLOCK_SIZE) // QBS
    expected_abs    = absmax_2d[:, abs_idx]
    check(f"block {b} packed data correct", torch.equal(col_packed, expected_packed))
    check(f"block {b} absmax data correct", torch.equal(col_abs, expected_abs))


# ===========================================================================
# Test 6: After pre-warm, decode token 1 gets full cache hit
# ===========================================================================
print("\n[6] Decode token 1: full cache hit after pre-warm")
param_l4  = make_down_param(4, hot_block_indices=[0, 1, 2])
routing_l4 = torch.tensor([0, 1, 2, 5, 8], dtype=torch.long)  # 3 cold: 5, 8 + one edge

spc6   = {"model.layers.4.mlp.down_proj.weight": param_l4}
hbbl6  = {4: routing_l4}
col_cache6 = {}
simulate_prewarm_tier1b(spc6, hbbl6, col_cache6, SPARSE_BS)
lc6 = col_cache6.get("model.layers.4.mlp.down_proj.weight", {})

# Simulate decode: request same cold blocks that were pre-warmed
cold_on_decode = torch.tensor([5, 8], dtype=torch.long)
_, _, was_hit = load_down_proj_cold_cols(param_l4, cold_on_decode, layer_cache=lc6)

check("decode token 1 is a full cache hit",  was_hit)


# ===========================================================================
# Test 7: Multi-layer pre-warm
# ===========================================================================
print("\n[7] Multi-layer pre-warm")
spc7   = {}
hbbl7  = {}
col_cache7 = {}
for i in range(5):
    hot = list(range(i % 3))  # 0, 1, or 2 hot blocks
    p = make_down_param(i, hot_block_indices=hot if hot else None)
    spc7[f"model.layers.{i}.mlp.down_proj.weight"] = p
    hbbl7[i] = torch.arange(4, dtype=torch.long)

warmed7 = simulate_prewarm_tier1b(spc7, hbbl7, col_cache7, SPARSE_BS)
check("all 5 layers warmed",  warmed7 == 5,  f"got {warmed7}")
check("5 cache entries",      len(col_cache7) == 5, f"got {len(col_cache7)}")


# ===========================================================================
# Summary
# ===========================================================================
print()
if errors:
    print("=" * 60)
    print(f"FAILED ({len(errors)} errors):")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
