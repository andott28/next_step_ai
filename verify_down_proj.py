"""Compare down_proj dequant and check if MLP output std=288 is correct."""
import sys, torch
sys.path.insert(0, ".")
from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime
import bitsandbytes.functional as bnb_functional
from bitsandbytes.backends.cuda.ops import _dequantize_4bit_impl as _bnb_dequant_impl

MODEL = "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"
runtime = StreamingLlamaRuntime(
    model_name_or_path=MODEL,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
    local_files_only=True,
    ram_cache=False,
    enable_cuda_h2d_overlap=False,
)

BLOCK_SIZE = 32
CHUNK_BLOCKS = 64
active_dim = CHUNK_BLOCKS * BLOCK_SIZE  # 2048

# --- Verify down_proj chunk 0 ---
name_down = "model.layers.0.mlp.down_proj.weight"
raw_down, quant_aux_down = runtime.loader._load_raw_for_param(name_down)
qs_down = bnb_functional.QuantState.from_dict(quant_aux_down, device=torch.device("cpu"))
print(f"down_proj quant_state: blocksize={qs_down.blocksize} nested={qs_down.nested} shape={qs_down.shape}")

# BNB standard full dequant
w_down_full = bnb_functional.dequantize_4bit(raw_down.reshape(-1), qs_down).reshape(qs_down.shape)
# Chunk 0: columns 0-2047 of down_proj (= intermediate neurons 0-2047)
w_down_ref = w_down_full[:, :active_dim].float()
print(f"BNB down_proj chunk0 cols 0-2047: std={float(w_down_ref.std()):.6f} max={float(w_down_ref.abs().max()):.6f}")

# Runtime chunked dequant for down_proj chunk 0
param_down = runtime._get_sparse_4bit_param(name_down)
ordered_blocks = torch.arange(CHUNK_BLOCKS, dtype=torch.long)
quant_block_size = int(param_down["quant_block_size"])
in_features_down = int(param_down["in_features"])   # 53248 (intermediate dim)
out_features_down = int(param_down["out_features"])  # 16384 (hidden dim)
bytes_per_row_down = in_features_down // 2
bytes_per_cblk = BLOCK_SIZE // 2  # 16
col_b_range = torch.arange(bytes_per_cblk, dtype=torch.long)
down_raw_2d = param_down["packed_weight"].view(out_features_down, bytes_per_row_down)
absmax_per_row = in_features_down // quant_block_size  # 832
down_absmax_2d = param_down["absmax"].view(out_features_down, absmax_per_row)

down_col_starts = ordered_blocks * bytes_per_cblk
down_col_idx = (down_col_starts.unsqueeze(-1) + col_b_range.unsqueeze(0)).reshape(-1)
gathered_down_packed_cpu = (
    down_raw_2d[:, down_col_idx]
    .reshape(out_features_down, int(ordered_blocks.numel()), bytes_per_cblk)
    .contiguous()
)
# Correct absmax (deduped)
down_abs_idx_raw = (ordered_blocks * BLOCK_SIZE) // quant_block_size
down_abs_idx = torch.unique_consecutive(down_abs_idx_raw)
print(f"down_abs_idx unique count={down_abs_idx.numel()} (expected {active_dim // quant_block_size})")
gathered_down_absmax_cpu = down_absmax_2d[:, down_abs_idx].contiguous()

down_packed_gpu = gathered_down_packed_cpu.to(device="cuda", dtype=torch.uint8)
down_absmax_gpu = gathered_down_absmax_cpu.to(device="cuda", dtype=torch.float32)
down_weight_chunk = torch.empty(out_features_down, active_dim, dtype=torch.float32, device="cuda")
_bnb_dequant_impl(
    down_packed_gpu.reshape(-1),
    down_absmax_gpu.reshape(-1),
    quant_block_size,
    str(param_down["quant_type"]),
    down_weight_chunk.dtype,
    out=down_weight_chunk,
)
w_down_runtime = down_weight_chunk.cpu()
print(f"Runtime down_proj chunk0: std={float(w_down_runtime.std()):.6f} max={float(w_down_runtime.abs().max()):.6f}")
print(f"First 5 values of col 0 row 0 (runtime): {w_down_runtime[0, :5].tolist()}")
print(f"First 5 values of col 0 row 0 (bnb ref):  {w_down_ref[0, :5].tolist()}")
diff_down = (w_down_runtime - w_down_ref).abs()
print(f"down_proj diff: max={float(diff_down.max()):.6f} mean={float(diff_down.mean()):.6f}")

# --- Compute what the MLP output SHOULD BE using BNB standard weights ---
# Load and dequant all three projections fully
name_gate = "model.layers.0.mlp.gate_proj.weight"
name_up = "model.layers.0.mlp.up_proj.weight"
raw_gate, qs_gate_aux = runtime.loader._load_raw_for_param(name_gate)
raw_up, qs_up_aux = runtime.loader._load_raw_for_param(name_up)
qs_gate = bnb_functional.QuantState.from_dict(qs_gate_aux, device="cpu")
qs_up = bnb_functional.QuantState.from_dict(qs_up_aux, device="cpu")
w_gate = bnb_functional.dequantize_4bit(raw_gate.reshape(-1), qs_gate).reshape(qs_gate.shape).float()
w_up = bnb_functional.dequantize_4bit(raw_up.reshape(-1), qs_up).reshape(qs_up.shape).float()
print(f"\nFull gate_proj: std={float(w_gate.std()):.6f} shape={list(w_gate.shape)}")
print(f"Full up_proj: std={float(w_up.std()):.6f} shape={list(w_up.shape)}")

# Use actual flat_hidden from the prefill debug: std=1.01, shape=[6, 16384]
# Simulate with synthetic input of same shape and std
torch.manual_seed(42)
flat_hidden = torch.randn(6, 16384, dtype=torch.float32) * 1.01
gate_full = torch.nn.functional.linear(flat_hidden, w_gate)
up_full = torch.nn.functional.linear(flat_hidden, w_up)
activated_full = torch.nn.functional.silu(gate_full) * up_full
print(f"\nWith synthetic input std=1.01:")
print(f"  gate_full std={float(gate_full.std()):.4f}")
print(f"  activated_full std={float(activated_full.std()):.4f}")
mlp_out_full = torch.nn.functional.linear(activated_full, w_down_full.float())
print(f"  mlp_out_full std={float(mlp_out_full.std()):.4f}")
print(f"  => Expected MLP output std with BNB standard weights: {float(mlp_out_full.std()):.4f}")
