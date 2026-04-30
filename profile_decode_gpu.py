"""Profile single decode step with per-layer GPU event timing.

Reports per-layer breakdown of:
  - load+attn GPU time
  - mlp GPU time
  - cold H2D bytes and transfer time (measured on h2d_stream)
  - hot cache bytes (VRAM gather, zero PCIe cost)
  - Python-loop overhead (cpu_wall - GPU event total)
"""
from __future__ import annotations
import os
import time, types
import torch
from transformers import AutoTokenizer
from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime

MODEL = "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"
BASIS = "results/mlp_basis_intermediate_full126.pt"
ATTN  = "results/attn_head_importance_405b.pt"
PROMPT = "In the beginning, there was a vast and silent void. Then, slowly, the first light emerged from the darkness."

tokenizer = AutoTokenizer.from_pretrained(MODEL)
input_ids = tokenizer.encode(PROMPT, return_tensors="pt")
print(f"Prompt: {input_ids.shape[1]} tokens")

runtime = StreamingLlamaRuntime(
    model_name_or_path=MODEL,
    sparse_basis_path=BASIS,
    attn_head_importance_path=ATTN,
    vram_hot_cache_gb=4.0,
    taylor_layers=[],
)
device = runtime.device
_max_runtime_layers = int((os.getenv("PROFILE_MAX_RUNTIME_LAYERS", "") or "0").strip() or "0")
if _max_runtime_layers > 0:
    runtime.num_layers = min(int(runtime.num_layers), int(_max_runtime_layers))
_single_layer_idx = int((os.getenv("PROFILE_SINGLE_LAYER_IDX", "-1") or "-1").strip() or "-1")
if _single_layer_idx >= 0:
    _single_layer_idx = min(int(_single_layer_idx), int(runtime.num_layers) - 1)
_skip_calibration = os.getenv("PROFILE_SKIP_CALIBRATION", "1").strip().lower() in {"1", "true", "yes", "on"}
_skip_warmup_decode = os.getenv("PROFILE_SKIP_WARMUP_DECODE", "1").strip().lower() in {"1", "true", "yes", "on"}

print("Warmup + calibrate...")
runtime.reset_caches()
runtime.generate(input_ids.to(device), max_new_tokens=2, do_sample=False)
if not _skip_calibration:
    calib_ids = input_ids.detach().to(device=torch.device("cpu"), dtype=torch.long)
    runtime.calibrate_vram_hot_cache(
        calib_ids,
        max_tokens=int(calib_ids.shape[1]),
        rebuild_cache=True,
        generate_decode_tokens=50,
    )
else:
    print("Skipping calibration (PROFILE_SKIP_CALIBRATION=1)", flush=True)

print("Running 4 warm-up decode tokens...")
runtime.reset_caches()
runtime.begin_traffic_phase("prefill")
logits = runtime.prefill_logits(input_ids.to(device))
runtime.begin_traffic_phase("decode")
runtime.materialize_lm_head()
generated = input_ids.to(device).clone()
if not _skip_warmup_decode:
    for step in range(4):
        next_token = runtime.sample_next_token(logits[:, -1, :], do_sample=False, temperature=1.0, top_k=None, top_p=1.0).view(1, 1).to(device)
        generated = torch.cat([generated, next_token], dim=-1)
        logits = runtime.decode_token_logits(next_token, position_index=int(generated.shape[1]) - 1)
        print(f"  Warmup token {step+1}/4 done", flush=True)
else:
    print("Skipping decode warmup (PROFILE_SKIP_WARMUP_DECODE=1)", flush=True)


# ── Instrumentation state ──────────────────────────────────────────────────────
layer_gpu_ms: dict[int, dict] = {}
layer_cpu_ms: dict[int, float] = {}
# Per-layer cold H2D bytes accumulated by the patched _copy_cpu_to_gpu
_h2d_bytes_accum: dict[int, int] = {}
_h2d_hot_bytes_accum: dict[int, int] = {}
_current_layer_idx: list[int] = [-1]  # mutable box so nested closure can write
_h2d_evt_start: dict[int, torch.cuda.Event] = {}
_h2d_evt_end: dict[int, torch.cuda.Event] = {}
_h2d_evt_down_start: dict[int, torch.cuda.Event] = {}
_h2d_evt_down_end: dict[int, torch.cuda.Event] = {}


def _patched_copy_cpu_to_gpu(_self_rt, tensor, *, dtype, layer_idx=None, tag="h2d", h2d_stream=None):
    """Accumulate H2D byte counts per layer."""
    _li = _current_layer_idx[0]
    result = _orig_copy_cpu_to_gpu(
        tensor,
        dtype=dtype,
        layer_idx=layer_idx,
        tag=tag,
        h2d_stream=h2d_stream,
    )
    nbytes = int(result.numel() * result.element_size())
    if _li >= 0:
        _h2d_bytes_accum[_li] = _h2d_bytes_accum.get(_li, 0) + nbytes
        if _self_rt.device.type == "cuda":
            is_down = bool(h2d_stream is _self_rt._h2d_stream_down)
            target_stream = h2d_stream if h2d_stream is not None else _self_rt._h2d_stream
            if target_stream is not None:
                if is_down:
                    if _li not in _h2d_evt_down_start:
                        _h2d_evt_down_start[_li] = torch.cuda.Event(enable_timing=True)
                        with torch.cuda.stream(target_stream):
                            _h2d_evt_down_start[_li].record()
                    _h2d_evt_down_end[_li] = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(target_stream):
                        _h2d_evt_down_end[_li].record()
                else:
                    if _li not in _h2d_evt_start:
                        _h2d_evt_start[_li] = torch.cuda.Event(enable_timing=True)
                        with torch.cuda.stream(target_stream):
                            _h2d_evt_start[_li].record()
                    _h2d_evt_end[_li] = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(target_stream):
                        _h2d_evt_end[_li].record()
    return result

# Bind the patched method; preserve original as a closure
_orig_copy_cpu_to_gpu = runtime._copy_cpu_to_gpu.__func__  # unbound from instance
def _bound_patched(tensor, *, dtype, layer_idx=None, tag="h2d", h2d_stream=None):
    return _patched_copy_cpu_to_gpu(
        runtime,
        tensor,
        dtype=dtype,
        layer_idx=layer_idx,
        tag=tag,
        h2d_stream=h2d_stream,
    )

# Also patch the VRAM hot gather tracker — count hot bytes before index_select calls
_orig_index_select = torch.Tensor.index_select
def _tracking_forward_token(self_rt, token_ids, *, position_index, _capture_layers=None, use_attention_cache=True):
    hidden = self_rt._embed_tokens_cpu(token_ids).to(device=self_rt.device, dtype=self_rt.dtype)
    position_ids = torch.tensor([[int(position_index)]], device=self_rt.device, dtype=torch.long)
    rope = self_rt.rotary_emb(hidden, position_ids)

    ev_layer_starts, ev_attn_ends, ev_mlp_ends, ev_layer_ends = [], [], [], []
    ev_h2d_ends: list = []  # event recorded on h2d_stream after waiting, to measure true overlap

    layer_indices = (
        [int(_single_layer_idx)]
        if int(_single_layer_idx) >= 0
        else list(range(self_rt.num_layers))
    )
    for layer_idx in layer_indices:
        _current_layer_idx[0] = layer_idx
        _h2d_bytes_accum[layer_idx] = 0
        _h2d_evt_start.pop(layer_idx, None)
        _h2d_evt_end.pop(layer_idx, None)
        _h2d_evt_down_start.pop(layer_idx, None)
        _h2d_evt_down_end.pop(layer_idx, None)
        t_cpu_start = time.perf_counter()

        ev_start      = torch.cuda.Event(enable_timing=True)
        ev_attn_end   = torch.cuda.Event(enable_timing=True)
        ev_mlp_end    = torch.cuda.Event(enable_timing=True)
        ev_end        = torch.cuda.Event(enable_timing=True)
        ev_h2d_done   = torch.cuda.Event(enable_timing=True)  # recorded after h2d wait

        ev_start.record()

        layer = self_rt._load_layer(layer_idx)
        residual = hidden
        hidden_norm = layer.input_layernorm(hidden)
        _use_shared_attn = self_rt._should_use_attn_share_for_layer(layer_idx)
        _active_attn_heads = None if _use_shared_attn else self_rt._get_attn_active_heads(layer_idx)

        _active_kv_blocks = None
        if layer_idx not in self_rt._retrieval_layers:
            _active_kv_blocks = self_rt._route_kv_blocks(hidden_norm, layer_idx)
        if _active_kv_blocks is not None:
            self_rt._populate_sparse_kv_skeleton(layer_idx, _active_kv_blocks, layer)

        taylor_attn = None
        if layer_idx in self_rt.taylor_layer_set and use_attention_cache:
            taylor_attn = self_rt._shared_taylor_attn
            taylor_attn.layer_idx = layer_idx
            attn_out, _attn_weights, present = taylor_attn(
                hidden_states=hidden_norm, position_ids=position_ids,
                position_embeddings=rope, past_key_value=self_rt._taylor_caches[layer_idx], use_cache=True,
            )
            self_rt._taylor_caches[layer_idx] = present
        elif (layer_idx in self_rt._retrieval_layers and self_rt._token_archive is not None and use_attention_cache):
            attn_out = self_rt._forward_retrieval_attn(
                layer_idx=layer_idx, layer=layer, hidden_norm=hidden_norm, rope=rope,
                position_index=int(position_ids.view(-1)[0].item()), active_heads=_active_attn_heads,
            )
        else:
            attn_out, _attn_weights = layer.self_attn(
                hidden_states=hidden_norm, position_embeddings=rope, attention_mask=None,
                past_key_values=self_rt._dense_cache if use_attention_cache else None,
                cache_position=position_ids.view(-1),
            )

        ev_attn_end.record()
        hidden = residual + attn_out
        residual = hidden
        mlp_input = layer.post_attention_layernorm(hidden)
        mlp_out = self_rt._mlp_forward_dispatch(layer_idx, layer, mlp_input)
        ev_mlp_end.record()

        hidden = residual + mlp_out
        if _active_kv_blocks is not None:
            self_rt._update_kv_block_banking(layer_idx, _active_kv_blocks)
        self_rt._release_modules(layer, *([taylor_attn] if taylor_attn is not None else []))

        # Wait for h2d stream and record an event so we can detect overlap failures
        if self_rt._h2d_stream is not None:
            self_rt._wait_for_h2d_stream()
            ev_h2d_done.record()  # recorded on default stream after h2d wait — shows serialization point
        ev_end.record()

        ev_layer_starts.append(ev_start)
        ev_attn_ends.append(ev_attn_end)
        ev_mlp_ends.append(ev_mlp_end)
        ev_layer_ends.append(ev_end)
        ev_h2d_ends.append(ev_h2d_done if self_rt._h2d_stream is not None else None)
        layer_cpu_ms[layer_idx] = (time.perf_counter() - t_cpu_start) * 1000

    _current_layer_idx[0] = -1
    if self_rt._kv_routing:
        self_rt._kv_bank_step += 1

    hidden = self_rt.norm(hidden)
    if not self_rt._materialize_lm_head:
        lm_logits = torch.zeros((hidden.shape[0], hidden.shape[1], 1), dtype=torch.float32)
    else:
        lm_logits = self_rt._lm_head_forward(hidden).float()

    torch.cuda.synchronize()
    for i in layer_indices:
        load_attn_ms = float(ev_layer_starts[i].elapsed_time(ev_attn_ends[i]))
        mlp_ms       = float(ev_attn_ends[i].elapsed_time(ev_mlp_ends[i]))
        other_ms     = float(ev_mlp_ends[i].elapsed_time(ev_layer_ends[i]))
        total_ms     = float(ev_layer_starts[i].elapsed_time(ev_layer_ends[i]))
        cpu_ms       = float(layer_cpu_ms.get(i, 0))
        cold_bytes   = int(_h2d_bytes_accum.get(i, 0))
        h2d_main_ms = 0.0
        h2d_down_ms = 0.0
        if i in _h2d_evt_start and i in _h2d_evt_end:
            h2d_main_ms = float(_h2d_evt_start[i].elapsed_time(_h2d_evt_end[i]))
        if i in _h2d_evt_down_start and i in _h2d_evt_down_end:
            h2d_down_ms = float(_h2d_evt_down_start[i].elapsed_time(_h2d_evt_down_end[i]))
        h2d_total_ms = max(h2d_main_ms, h2d_down_ms)
        h2d_spill_ms = max(0.0, h2d_total_ms - load_attn_ms)
        python_overhead_ms = max(0.0, cpu_ms - total_ms)
        layer_gpu_ms[i] = {
            "load_attn_ms":        load_attn_ms,
            "mlp_ms":              mlp_ms,
            "other_ms":            other_ms,
            "total_ms":            total_ms,
            "cpu_ms":              cpu_ms,
            "cold_h2d_bytes":      cold_bytes,
            "cold_h2d_mb":         cold_bytes / (1024**2),
            "h2d_main_ms":         h2d_main_ms,
            "h2d_down_ms":         h2d_down_ms,
            "h2d_total_ms":        h2d_total_ms,
            "h2d_spill_ms":        h2d_spill_ms,
            "python_overhead_ms":  python_overhead_ms,
        }
    return lm_logits, {}

runtime.forward_token = types.MethodType(_tracking_forward_token, runtime)
# Patch _copy_cpu_to_gpu to accumulate per-layer bytes
runtime._copy_cpu_to_gpu = types.MethodType(
    lambda _self, tensor, *, dtype, layer_idx=None, tag="h2d", h2d_stream=None: (
        _patched_copy_cpu_to_gpu(
            _self,
            tensor,
            dtype=dtype,
            layer_idx=layer_idx,
            tag=tag,
            h2d_stream=h2d_stream,
        )
    ),
    runtime,
)


# ── Profile one token ─────────────────────────────────────────────────────────
print("\nProfiling GPU time for token 5...")
next_token = runtime.sample_next_token(logits[:, -1, :], do_sample=False, temperature=1.0, top_k=None, top_p=1.0).view(1, 1).to(device)
generated = torch.cat([generated, next_token], dim=-1)

ev_total_start = torch.cuda.Event(enable_timing=True)
ev_total_end   = torch.cuda.Event(enable_timing=True)
t_wall_start   = time.perf_counter()
ev_total_start.record()
logits = runtime.decode_token_logits(next_token, position_index=int(generated.shape[1]) - 1)
ev_total_end.record()
torch.cuda.synchronize()
t_wall          = (time.perf_counter() - t_wall_start) * 1000
total_gpu_ms    = float(ev_total_start.elapsed_time(ev_total_end))

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\nTotal wall: {t_wall:.1f} ms  |  Total GPU: {total_gpu_ms:.1f} ms  |  Implied tok/s: {1000/t_wall:.3f}")
print(f"\n{'Layer':>5}  {'attn':>7}  {'mlp':>7}  {'h2d_ms':>7}  {'spill':>7}  {'total':>7}  {'cold_mb':>8}")
print("-" * 90)

total_cold_mb = 0.0
layer_indices = (
    [int(_single_layer_idx)]
    if int(_single_layer_idx) >= 0
    else list(range(runtime.num_layers))
)
for i in layer_indices:
    g = layer_gpu_ms.get(i, {})
    total_ms = g.get("total_ms", 0.0)
    cold_mb  = g.get("cold_h2d_mb", 0.0)
    total_cold_mb += cold_mb
    print(
        f"  {i:3d}  {g.get('load_attn_ms',0):7.3f}  {g.get('mlp_ms',0):7.3f}  "
        f"{g.get('h2d_total_ms',0):7.3f}  {g.get('h2d_spill_ms',0):7.3f}  "
        f"{total_ms:7.3f}  {cold_mb:8.3f}"
    )

print("-" * 90)
gpu_totals = {k: sum(layer_gpu_ms.get(i, {}).get(k, 0) for i in layer_indices)
              for k in ["load_attn_ms", "mlp_ms", "other_ms", "total_ms", "cpu_ms", "python_overhead_ms"]}
gpu_totals["h2d_total_ms"] = sum(layer_gpu_ms.get(i, {}).get("h2d_total_ms", 0) for i in layer_indices)
gpu_totals["h2d_spill_ms"] = sum(layer_gpu_ms.get(i, {}).get("h2d_spill_ms", 0) for i in layer_indices)

print(f"\n=== Aggregates over {len(layer_indices)} layers ===")
print(f"  load+attn GPU  : {gpu_totals['load_attn_ms']:8.1f} ms  avg {gpu_totals['load_attn_ms']/max(len(layer_indices),1):.3f} ms/layer")
print(f"  mlp GPU        : {gpu_totals['mlp_ms']:8.1f} ms  avg {gpu_totals['mlp_ms']/max(len(layer_indices),1):.3f} ms/layer")
print(f"  h2d wall       : {gpu_totals['h2d_total_ms']:8.1f} ms  avg {gpu_totals['h2d_total_ms']/max(len(layer_indices),1):.3f} ms/layer")
print(f"  h2d spill      : {gpu_totals['h2d_spill_ms']:8.1f} ms  avg {gpu_totals['h2d_spill_ms']/max(len(layer_indices),1):.3f} ms/layer")
print(f"  other GPU      : {gpu_totals['other_ms']:8.1f} ms  avg {gpu_totals['other_ms']/max(len(layer_indices),1):.3f} ms/layer")
print(f"  sum GPU (evts) : {gpu_totals['total_ms']:8.1f} ms")
print(f"  CPU wall sum   : {gpu_totals['cpu_ms']:8.1f} ms")
print(f"  Python overhead: {gpu_totals['python_overhead_ms']:8.1f} ms  avg {gpu_totals['python_overhead_ms']/max(len(layer_indices),1):.3f} ms/layer")
print(f"  Cold H2D total : {total_cold_mb:8.1f} MB  ({total_cold_mb/1024:.3f} GB/token)")
hot_hits = getattr(runtime, "_decode_mlp_hot_blocks_hit", 0)
cold_hits = getattr(runtime, "_decode_mlp_cold_blocks_streamed", 0)
total_blocks = hot_hits + cold_hits
hot_pct = 100 * hot_hits / max(total_blocks, 1)
print(f"  Hot cache hits : {hot_hits} / {total_blocks} blocks  ({hot_pct:.1f}% hit rate)")
print(f"  Cold H2D at 14GB/s would take: {total_cold_mb/1024/14*1000:.1f} ms/token (PCIe lower bound)")
print(f"  Effective tok/s ceiling (PCIe only): {14*1024/max(total_cold_mb,0.001):.2f}")
