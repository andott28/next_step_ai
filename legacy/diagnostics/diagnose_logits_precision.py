from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, ".")

from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime

MODEL = "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"


def _causal_mask(dtype: torch.dtype, device: torch.device, seq_len: int) -> torch.Tensor | None:
    if seq_len <= 1:
        return None
    min_val = torch.finfo(dtype).min
    q_pos = torch.arange(seq_len, device=device)
    k_pos = torch.arange(seq_len, device=device)
    future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    return future.to(dtype=dtype).mul(min_val).unsqueeze(0).unsqueeze(0)


def _scan_lm_head_fp32(hidden_last_cpu: torch.Tensor, lm_head_cpu: torch.Tensor, chunk_rows: int = 2048) -> torch.Tensor:
    pieces = []
    hidden_f32 = hidden_last_cpu.float().view(1, -1)
    for start in range(0, int(lm_head_cpu.shape[0]), int(chunk_rows)):
        stop = min(int(lm_head_cpu.shape[0]), start + int(chunk_rows))
        logits_chunk = F.linear(hidden_f32, lm_head_cpu[start:stop].float())
        pieces.append(logits_chunk.reshape(-1))
    return torch.cat(pieces, dim=0)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not torch.is_tensor(ids):
        ids = ids["input_ids"]
    print(f"prompt_len={int(ids.shape[1])}", flush=True)

    runtime = StreamingLlamaRuntime(
        model_name_or_path=MODEL,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        taylor_layers=[],
        local_files_only=True,
        ram_cache=False,
        enable_cuda_h2d_overlap=False,
        materialize_lm_head=True,
    )
    runtime.reset_caches()
    runtime._set_traffic_phase("prefill")

    token_ids = ids.to(device=runtime.device)
    seq_len = int(token_ids.shape[1])
    hidden = runtime._embed_tokens_cpu(token_ids).to(device=runtime.device, dtype=runtime.dtype)
    position_ids = torch.arange(seq_len, device=runtime.device, dtype=torch.long)
    position_ids_batch = position_ids.unsqueeze(0)
    mask = _causal_mask(runtime.dtype, runtime.device, seq_len)

    with torch.no_grad():
        for layer_idx in range(runtime.num_layers):
            layer = runtime._load_layer(layer_idx)
            h_norm = layer.input_layernorm(hidden)
            rope = runtime.rotary_emb(hidden, position_ids_batch)
            attn_out, _ = layer.self_attn(
                hidden_states=h_norm,
                position_embeddings=rope,
                attention_mask=mask,
                past_key_values=runtime._dense_cache,
                cache_position=position_ids,
            )
            hidden = hidden + attn_out
            residual = hidden
            mlp_input = layer.post_attention_layernorm(hidden)
            mlp_out = runtime._mlp_forward_dispatch(layer_idx, layer, mlp_input)
            hidden = residual + mlp_out
            if layer_idx in {0, 1, 2, 10, 62, 125}:
                print(
                    f"layer={layer_idx} hidden_last_std={float(hidden[0, -1, :].float().std()):.6f}",
                    flush=True,
                )

        prenorm_std = float(hidden[0, -1, :].float().std())
        hidden_normed = runtime.norm(hidden)
        hidden_last_cpu = hidden_normed[0, -1, :].detach().cpu()
        print(
            f"prenorm_last_std={prenorm_std:.6f} postnorm_last_std={float(hidden_last_cpu.float().std()):.6f}",
            flush=True,
        )

    lm_head = runtime._ensure_lm_head_weight_cpu()
    logits_bf16 = F.linear(hidden_last_cpu.view(1, -1).to(dtype=lm_head.dtype), lm_head).float().reshape(-1)
    logits_fp32 = _scan_lm_head_fp32(hidden_last_cpu, lm_head, chunk_rows=2048)

    for name, logits in [("cpu_bf16", logits_bf16), ("chunk_fp32", logits_fp32)]:
        top = torch.topk(logits, 10)
        print(f"{name}_top_ids={top.indices.tolist()}", flush=True)
        print(f"{name}_top_vals={[round(float(v), 4) for v in top.values.tolist()]}", flush=True)
        print(
            f"{name}_top_text={[repr(tokenizer.decode([int(i)], skip_special_tokens=False, clean_up_tokenization_spaces=False)) for i in top.indices.tolist()]}",
            flush=True,
        )
        for token_id in [12366, 60704, 6569, 70235, 69655, 13, 320]:
            print(
                f"{name}_logit {token_id} "
                f"{tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)!r} "
                f"{float(logits[token_id]):.4f}",
                flush=True,
            )

    diff = (logits_bf16 - logits_fp32).abs()
    print(f"logit_diff_max={float(diff.max()):.6f}", flush=True)
    print(f"logit_diff_mean={float(diff.mean()):.6f}", flush=True)


if __name__ == "__main__":
    main()
