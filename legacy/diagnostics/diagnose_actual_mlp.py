from __future__ import annotations

import gc
import sys

import bitsandbytes.functional as bnb_functional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, ".")

from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime

MODEL = "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"
PROMPT = "The capital of France is"
PARIS_ID = 12366
BLOCK_SIZE = 32
CHUNK_BLOCKS = 64
ACTIVE_DIM = BLOCK_SIZE * CHUNK_BLOCKS


def _causal_mask(dtype: torch.dtype, device: torch.device, seq_len: int) -> torch.Tensor | None:
    if seq_len <= 1:
        return None
    min_val = torch.finfo(dtype).min
    q_pos = torch.arange(seq_len, device=device)
    k_pos = torch.arange(seq_len, device=device)
    future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    return future.to(dtype=dtype).mul(min_val).unsqueeze(0).unsqueeze(0)


def _load_dequant(runtime: StreamingLlamaRuntime, name: str) -> torch.Tensor:
    raw_weight, quant_aux = runtime.loader._load_raw_for_param(name)
    quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
    return bnb_functional.dequantize_4bit(raw_weight.reshape(-1), quant_state).reshape(quant_state.shape).float()


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"]
    print(f"prompt_ids={input_ids[0].tolist()}")
    print(f"prompt_tokens={[tokenizer.decode([int(t)]) for t in input_ids[0].tolist()]}")
    print(f"paris_id={PARIS_ID} token={tokenizer.decode([PARIS_ID])!r}")

    runtime = StreamingLlamaRuntime(
        model_name_or_path=MODEL,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        taylor_layers=[],
        local_files_only=True,
        ram_cache=False,
        enable_cuda_h2d_overlap=False,
        materialize_lm_head=False,
    )
    runtime.reset_caches()
    runtime._set_traffic_phase("prefill")

    token_ids = input_ids.to(device=runtime.device)
    seq_len = int(token_ids.shape[1])
    all_hidden = runtime._embed_tokens_cpu(token_ids).to(device=runtime.device, dtype=runtime.dtype)
    position_ids = torch.arange(seq_len, device=runtime.device, dtype=torch.long)
    position_ids_batch = position_ids.unsqueeze(0)
    mask = _causal_mask(runtime.dtype, runtime.device, seq_len)

    layer = runtime._load_layer(0)
    h_norm = layer.input_layernorm(all_hidden)
    rope = runtime.rotary_emb(all_hidden, position_ids_batch)
    attn_out, _ = layer.self_attn(
        hidden_states=h_norm,
        position_embeddings=rope,
        attention_mask=mask,
        past_key_values=runtime._dense_cache,
        cache_position=position_ids,
    )
    post_attn = all_hidden + attn_out
    mlp_input = layer.post_attention_layernorm(post_attn)
    flat_hidden_gpu = mlp_input.view(-1, mlp_input.shape[-1]).contiguous()
    flat_hidden = flat_hidden_gpu.detach().cpu().float()

    print(f"embed_last_std={float(all_hidden[0, -1, :].float().std()):.6f}")
    print(f"attn_out_last_std={float(attn_out[0, -1, :].float().std()):.6f}")
    print(f"post_attn_last_std={float(post_attn[0, -1, :].float().std()):.6f}")
    print(f"mlp_input_std={float(flat_hidden.std()):.6f} mlp_input_mean={float(flat_hidden.mean()):.6f}")

    mlp_runtime = runtime._dense_guard_mlp_forward_exact_chunked_4bit(0, layer.mlp, mlp_input)
    mlp_runtime_cpu = mlp_runtime.detach().cpu().float().view(int(flat_hidden.shape[0]), int(flat_hidden.shape[1]))
    print(f"runtime_mlp_std={float(mlp_runtime_cpu.std()):.6f}")
    print(f"runtime_mlp_last_std={float(mlp_runtime_cpu[-1, :].std()):.6f}")

    prefix = "model.layers.0.mlp."
    gate_weight = _load_dequant(runtime, f"{prefix}gate_proj.weight")
    gate_chunk = gate_weight[:ACTIVE_DIM, :].contiguous()
    real_gate = F.linear(flat_hidden, gate_chunk)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    random_same_std = torch.randn(flat_hidden.shape, generator=generator, dtype=flat_hidden.dtype)
    random_same_std = random_same_std * float(flat_hidden.std()) + float(flat_hidden.mean())
    random_gate = F.linear(random_same_std, gate_chunk)

    feature_perm = torch.randperm(int(flat_hidden.shape[-1]), generator=generator)
    shuffled_features = flat_hidden.index_select(1, feature_perm)
    shuffled_gate = F.linear(shuffled_features, gate_chunk)

    iid_formula = (float(flat_hidden.shape[-1]) ** 0.5) * float(flat_hidden.std()) * float(gate_chunk.std())
    print(f"gate_chunk_weight_std={float(gate_chunk.std()):.6f}")
    print(f"gate_chunk_real_std={float(real_gate.std()):.6f}")
    print(f"gate_chunk_iid_formula={iid_formula:.6f}")
    print(f"gate_chunk_random_same_std={float(random_gate.std()):.6f}")
    print(f"gate_chunk_shuffled_features={float(shuffled_gate.std()):.6f}")

    _, singular_values, vh = torch.linalg.svd(flat_hidden, full_matrices=False)
    projected = F.linear(vh, gate_chunk)
    projection_stds = projected.std(dim=1)
    reconstructed_gate = (torch.diag(singular_values) @ projected).reshape(-1)
    print(f"svd_singular_values={[round(float(v), 6) for v in singular_values.tolist()]}")
    print(f"svd_gate_projection_stds={[round(float(v), 8) for v in projection_stds.tolist()]}")
    print(f"svd_reconstructed_gate_std={float(reconstructed_gate.std()):.6f}")

    del real_gate, random_gate, shuffled_gate, projected, reconstructed_gate, gate_chunk
    gate_full = F.linear(flat_hidden, gate_weight)
    print(f"full_gate_actual_std={float(gate_full.std()):.6f}")
    del gate_weight
    gc.collect()

    up_weight = _load_dequant(runtime, f"{prefix}up_proj.weight")
    up_full = F.linear(flat_hidden, up_weight)
    print(f"full_up_actual_std={float(up_full.std()):.6f}")
    del up_weight
    gc.collect()

    activated = F.silu(gate_full) * up_full
    print(f"full_activated_actual_std={float(activated.std()):.6f}")
    del gate_full, up_full
    gc.collect()

    down_weight = _load_dequant(runtime, f"{prefix}down_proj.weight")
    mlp_reference = F.linear(activated, down_weight)
    print(f"reference_full_mlp_std={float(mlp_reference.std()):.6f}")
    print(f"reference_full_mlp_last_std={float(mlp_reference[-1, :].std()):.6f}")

    diff = (mlp_reference - mlp_runtime_cpu).abs()
    denom = mlp_reference.abs().clamp_min(1e-6)
    print(f"runtime_vs_reference_max_abs={float(diff.max()):.6f}")
    print(f"runtime_vs_reference_mean_abs={float(diff.mean()):.6f}")
    print(f"runtime_vs_reference_mean_rel={float((diff / denom).mean()):.6f}")
    print(f"runtime_vs_reference_std_delta={float(mlp_runtime_cpu.std() - mlp_reference.std()):.6f}")


if __name__ == "__main__":
    main()
