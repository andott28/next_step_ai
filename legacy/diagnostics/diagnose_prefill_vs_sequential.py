from __future__ import annotations

import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")

from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime

MODEL = "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"
PROMPT = "The capital of France is"


def _causal_mask(dtype: torch.dtype, device: torch.device, seq_len: int) -> torch.Tensor | None:
    if seq_len <= 1:
        return None
    min_val = torch.finfo(dtype).min
    q_pos = torch.arange(seq_len, device=device)
    k_pos = torch.arange(seq_len, device=device)
    future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    return future.to(dtype=dtype).mul(min_val).unsqueeze(0).unsqueeze(0)


def _run_layer_first_prefill(runtime: StreamingLlamaRuntime, input_ids: torch.LongTensor, layers: int) -> torch.Tensor:
    runtime.reset_caches()
    runtime._set_traffic_phase("prefill")
    token_ids = input_ids.to(device=runtime.device)
    seq_len = int(token_ids.shape[1])
    all_hidden = runtime._embed_tokens_cpu(token_ids).to(device=runtime.device, dtype=runtime.dtype)
    position_ids = torch.arange(seq_len, device=runtime.device, dtype=torch.long)
    position_ids_batch = position_ids.unsqueeze(0)
    mask = _causal_mask(runtime.dtype, runtime.device, seq_len)

    for layer_idx in range(int(layers)):
        layer = runtime._load_layer(layer_idx)
        h_norm = layer.input_layernorm(all_hidden)
        rope = runtime.rotary_emb(all_hidden, position_ids_batch)
        attn_out, _ = layer.self_attn(
            hidden_states=h_norm,
            position_embeddings=rope,
            attention_mask=mask,
            past_key_values=runtime._dense_cache,
            cache_position=position_ids,
        )
        all_hidden = all_hidden + attn_out
        residual = all_hidden
        mlp_input = layer.post_attention_layernorm(all_hidden)
        mlp_out = runtime._mlp_forward_dispatch(layer_idx, layer, mlp_input)
        all_hidden = residual + mlp_out

    return all_hidden.detach().cpu().float()


def _run_token_sequential(runtime: StreamingLlamaRuntime, input_ids: torch.LongTensor, layers: int) -> torch.Tensor:
    runtime.reset_caches()
    runtime._set_traffic_phase("decode")
    token_ids = input_ids.to(device=runtime.device)
    last_hidden = None

    for pos in range(int(token_ids.shape[1])):
        hidden = runtime._embed_tokens_cpu(token_ids[:, pos : pos + 1]).to(device=runtime.device, dtype=runtime.dtype)
        position_ids = torch.tensor([pos], device=runtime.device, dtype=torch.long)
        position_ids_batch = position_ids.view(1, 1)
        rope = runtime.rotary_emb(hidden, position_ids_batch)

        for layer_idx in range(int(layers)):
            layer = runtime._load_layer(layer_idx)
            residual = hidden
            hidden_norm = layer.input_layernorm(hidden)
            attn_out, _ = layer.self_attn(
                hidden_states=hidden_norm,
                position_embeddings=rope,
                attention_mask=None,
                past_key_values=runtime._dense_cache,
                cache_position=position_ids,
            )
            hidden = residual + attn_out
            residual = hidden
            mlp_input = layer.post_attention_layernorm(hidden)
            mlp_out = runtime._mlp_forward_dispatch(layer_idx, layer, mlp_input)
            hidden = residual + mlp_out

        last_hidden = hidden.detach().cpu().float()

    if last_hidden is None:
        raise RuntimeError("No tokens were processed")
    return last_hidden[:, -1, :]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    print(f"prompt_ids={input_ids[0].tolist()}")

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

    for layers in (1, 2, 4):
        prefill_hidden = _run_layer_first_prefill(runtime, input_ids, layers)[:, -1, :]
        sequential_hidden = _run_token_sequential(runtime, input_ids, layers)
        diff = (prefill_hidden - sequential_hidden).abs()
        denom = sequential_hidden.abs().clamp_min(1e-6)
        cosine = torch.nn.functional.cosine_similarity(prefill_hidden.flatten(), sequential_hidden.flatten(), dim=0)
        print(
            f"layers={layers} "
            f"prefill_std={float(prefill_hidden.std()):.6f} "
            f"sequential_std={float(sequential_hidden.std()):.6f} "
            f"max_abs={float(diff.max()):.6f} "
            f"mean_abs={float(diff.mean()):.6f} "
            f"mean_rel={float((diff / denom).mean()):.6f} "
            f"cosine={float(cosine):.8f}"
        )


if __name__ == "__main__":
    main()
