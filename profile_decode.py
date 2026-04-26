"""Profile a single decode step with per-layer and per-phase timing."""
from __future__ import annotations
import time
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

# Warmup + calibrate
print("Warmup pass...")
runtime.reset_caches()
runtime.generate(input_ids.to(device), max_new_tokens=2, do_sample=False)

# Calibrate from actual prompt
calib_ids = input_ids.detach().to(device=torch.device("cpu"), dtype=torch.long)
runtime.calibrate_vram_hot_cache(calib_ids, max_tokens=int(calib_ids.shape[1]), rebuild_cache=True)

print("Timed pass: warming 4 tokens before profiling token 5...")
runtime.reset_caches()
runtime._reset_traffic_stats()
runtime.begin_traffic_phase("prefill")
logits = runtime.prefill_logits(input_ids.to(device))
runtime.begin_traffic_phase("decode")
runtime.materialize_lm_head()

generated = input_ids.to(device).clone()
for step in range(4):
    next_token = runtime.sample_next_token(logits[:, -1, :], do_sample=False, temperature=1.0, top_k=None, top_p=1.0).view(1,1).to(device)
    generated = torch.cat([generated, next_token], dim=-1)
    logits = runtime.decode_token_logits(next_token, position_index=int(generated.shape[1]) - 1)
    print(f"  Warmup token {step+1}/4 done", flush=True)

print("\nProfiling token 5...")

# Monkey-patch forward_token to add per-phase timing
import types

original_forward_token = runtime.forward_token.__func__

layer_times = {}

def patched_forward_token(self, token_ids, *, position_index, capture_layers=None, use_attention_cache=True):
    from llama3_neuroplastic.experiments.streaming_llama_runtime import GQATaylorSSDSelfAttention
    import torch.nn as nn

    capture_set = {int(idx) for idx in capture_layers or []}
    captures = {}

    t_total_start = time.perf_counter()
    hidden = self._embed_tokens_cpu(token_ids).to(device=self.device, dtype=self.dtype)
    position_ids = torch.tensor([[int(position_index)]], device=self.device, dtype=torch.long)
    rope = self.rotary_emb(hidden, position_ids)

    for layer_idx in range(self.num_layers):
        t0 = time.perf_counter()
        layer = self._load_layer(layer_idx)
        t_load = time.perf_counter()

        residual = hidden
        hidden_norm = layer.input_layernorm(hidden)
        _use_shared_attn = self._should_use_attn_share_for_layer(layer_idx)
        _active_attn_heads = None if _use_shared_attn else self._get_attn_active_heads(layer_idx)

        _active_kv_blocks = None
        if layer_idx not in self._retrieval_layers:
            _active_kv_blocks = self._route_kv_blocks(hidden_norm, layer_idx)
        if _active_kv_blocks is not None:
            self._populate_sparse_kv_skeleton(layer_idx, _active_kv_blocks, layer)

        t_route = time.perf_counter()

        taylor_attn = None
        if layer_idx in self.taylor_layer_set and use_attention_cache:
            taylor_attn = self._shared_taylor_attn
            taylor_attn.layer_idx = layer_idx
            attn_out, _attn_weights, present = taylor_attn(
                hidden_states=hidden_norm, position_ids=position_ids,
                position_embeddings=rope, past_key_value=self._taylor_caches[layer_idx], use_cache=True,
            )
            self._taylor_caches[layer_idx] = present
        elif (layer_idx in self._retrieval_layers and self._token_archive is not None and use_attention_cache):
            attn_out = self._forward_retrieval_attn(
                layer_idx=layer_idx, layer=layer, hidden_norm=hidden_norm, rope=rope,
                position_index=int(position_ids.view(-1)[0].item()), active_heads=_active_attn_heads,
            )
        else:
            attn_out, _attn_weights = layer.self_attn(
                hidden_states=hidden_norm, position_embeddings=rope,
                attention_mask=None,
                past_key_values=self._dense_cache if use_attention_cache else None,
                cache_position=position_ids.view(-1),
            )

        t_attn = time.perf_counter()

        hidden = residual + attn_out
        residual = hidden
        mlp_input = layer.post_attention_layernorm(hidden)
        mlp_out = self._mlp_forward_dispatch(layer_idx, layer, mlp_input)

        t_mlp = time.perf_counter()

        hidden = residual + mlp_out
        if _active_kv_blocks is not None:
            self._update_kv_block_banking(layer_idx, _active_kv_blocks)
        self._release_modules(layer, *([taylor_attn] if taylor_attn is not None else []))

        t_end = time.perf_counter()

        layer_times[layer_idx] = {
            "load_ms": (t_load - t0) * 1000,
            "route_ms": (t_route - t_load) * 1000,
            "attn_ms": (t_attn - t_route) * 1000,
            "mlp_ms": (t_mlp - t_attn) * 1000,
            "other_ms": (t_end - t_mlp) * 1000,
            "total_ms": (t_end - t0) * 1000,
        }

    if self._kv_routing:
        self._kv_bank_step += 1

    hidden = self.norm(hidden)
    if not self._materialize_lm_head:
        logits = torch.zeros((hidden.shape[0], hidden.shape[1], 1), dtype=torch.float32)
    else:
        logits = self._lm_head_forward(hidden).float()
    return logits, captures

runtime.forward_token = types.MethodType(patched_forward_token, runtime)

next_token = runtime.sample_next_token(logits[:, -1, :], do_sample=False, temperature=1.0, top_k=None, top_p=1.0).view(1,1).to(device)
generated = torch.cat([generated, next_token], dim=-1)
t_start = time.perf_counter()
logits = runtime.decode_token_logits(next_token, position_index=int(generated.shape[1]) - 1)
t_total = time.perf_counter() - t_start
print(f"Token 5 total: {t_total*1000:.1f} ms")

# Aggregate stats
loads = [layer_times[i]["load_ms"] for i in range(126)]
routes = [layer_times[i]["route_ms"] for i in range(126)]
attns = [layer_times[i]["attn_ms"] for i in range(126)]
mlps = [layer_times[i]["mlp_ms"] for i in range(126)]
others = [layer_times[i]["other_ms"] for i in range(126)]
totals = [layer_times[i]["total_ms"] for i in range(126)]

print(f"\n=== Per-phase totals over 126 layers (ms) ===")
print(f"  _load_layer:     {sum(loads):.1f} ms  (avg {sum(loads)/126:.2f} ms/layer)")
print(f"  route_kv:        {sum(routes):.1f} ms  (avg {sum(routes)/126:.2f} ms/layer)")
print(f"  attn_forward:    {sum(attns):.1f} ms  (avg {sum(attns)/126:.2f} ms/layer)")
print(f"  mlp_forward:     {sum(mlps):.1f} ms  (avg {sum(mlps)/126:.2f} ms/layer)")
print(f"  other:           {sum(others):.1f} ms  (avg {sum(others)/126:.2f} ms/layer)")
print(f"  total in loop:   {sum(totals):.1f} ms")

# Show top 5 slowest layers
top_layers = sorted(range(126), key=lambda i: totals[i], reverse=True)[:5]
print(f"\nTop 5 slowest layers:")
for i in top_layers:
    d = layer_times[i]
    print(f"  Layer {i:3d}: load={d['load_ms']:.1f} route={d['route_ms']:.1f} attn={d['attn_ms']:.1f} mlp={d['mlp_ms']:.1f} total={d['total_ms']:.1f} ms")
