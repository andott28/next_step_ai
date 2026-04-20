"""Compare runtime MLP dequant vs. bitsandbytes standard dequant for layer 0 gate_proj."""
import sys

import torch
from bitsandbytes.backends.cuda.ops import _dequantize_4bit_impl as _bnb_dequant_impl

sys.path.insert(0, ".")
import bitsandbytes.functional as bnb_functional

from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime

MODEL = "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"
runtime = StreamingLlamaRuntime(
    model_name_or_path=MODEL,
    device=torch.device("cuda"),
    dtype=torch.bfloat16,
    local_files_only=True,
    ram_cache=False,
    enable_cuda_h2d_overlap=False,
)


name = "model.layers.0.mlp.gate_proj.weight"
raw_weight, quant_aux = runtime.loader._load_raw_for_param(name)
print(f"raw_weight shape={raw_weight.shape} dtype={raw_weight.dtype}")


quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
print(f"quant_state: blocksize={quant_state.blocksize} dtype={quant_state.dtype} nested={quant_state.nested}")
print(f"absmax (first level) shape={quant_state.absmax.shape} dtype={quant_state.absmax.dtype}")
if quant_state.nested:
    print(f"state2.absmax shape={quant_state.state2.absmax.shape}")
    print(f"quant_state.offset={quant_state.offset}")
    absmax_deq = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
    absmax_deq = absmax_deq + quant_state.offset
    print(f"dequantized absmax stats: mean={float(absmax_deq.mean()):.6f} max={float(absmax_deq.max()):.6f} min={float(absmax_deq.min()):.6f}")
else:
    absmax_deq = quant_state.absmax.float()
    print(f"absmax (direct) stats: mean={float(absmax_deq.mean()):.6f} max={float(absmax_deq.max()):.6f}")


w_ref_full = bnb_functional.dequantize_4bit(
    raw_weight.reshape(-1),
    quant_state,
).reshape(quant_state.shape)
w_ref = w_ref_full[:2048, :].float()

print(f"\nBNB standard dequant chunk0 rows 0-2047: std={float(w_ref.std()):.6f} max={float(w_ref.abs().max()):.6f}")
print(f"First 5 values of row 0 (bnb): {w_ref[0, :5].tolist()}")


in_features = int(quant_state.shape[1])
absmax_per_row = in_features // quant_state.blocksize
absmax_chunk0 = absmax_deq[:2048 * absmax_per_row]
print(f"BNB absmax chunk0: mean={float(absmax_chunk0.mean()):.6f} max={float(absmax_chunk0.max()):.6f}")


BLOCK_SIZE = 32
CHUNK_BLOCKS = 64
active_dim = CHUNK_BLOCKS * BLOCK_SIZE


param = runtime._get_sparse_4bit_param(name)
ordered_blocks = torch.arange(CHUNK_BLOCKS, dtype=torch.long)


packed_blocks = param["packed_blocks"][ordered_blocks].contiguous()
absmax_blocks = param["absmax_blocks"][ordered_blocks].contiguous().to(dtype=torch.float32)


packed_flat = packed_blocks.reshape(-1)
absmax_flat = absmax_blocks.reshape(-1)

print(f"\nRuntime chunk0 packed_flat shape={packed_flat.shape}")
print(f"Runtime chunk0 absmax_flat shape={absmax_flat.shape} mean={float(absmax_flat.mean()):.6f}")


gate_weight_runtime = torch.empty((active_dim, int(param["in_features"])), dtype=torch.float32)


if torch.cuda.is_available():
    packed_gpu = packed_flat.to(device="cuda")
    absmax_gpu = absmax_flat.to(device="cuda")
    gate_weight_gpu = gate_weight_runtime.to(device="cuda")
    _bnb_dequant_impl(
        packed_gpu, absmax_gpu,
        int(param["quant_block_size"]),
        str(param["quant_type"]),
        gate_weight_gpu.dtype,
        out=gate_weight_gpu,
    )
    gate_weight_runtime = gate_weight_gpu.cpu()

print(f"\nRuntime dequant chunk0: std={float(gate_weight_runtime.std()):.6f} max={float(gate_weight_runtime.abs().max()):.6f}")
print(f"First 5 values of row 0 (runtime): {gate_weight_runtime[0, :5].tolist()}")
print(f"First 5 values of row 0 (bnb ref):  {w_ref[0, :5].tolist()}")


diff = (gate_weight_runtime - w_ref[:active_dim, :]).abs()
print(f"\nDifference stats: max={float(diff.max()):.6f} mean={float(diff.mean()):.6f}")
print(f"Max relative diff: {float(diff.max() / (w_ref[:active_dim, :].abs().max() + 1e-8)):.6f}")
