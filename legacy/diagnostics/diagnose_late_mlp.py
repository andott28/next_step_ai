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
TARGET_LAYER = 125


def _load_dequant_float(runtime: StreamingLlamaRuntime, name: str) -> torch.Tensor:
    raw_weight, quant_aux = runtime.loader._load_raw_for_param(name)
    quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
    return bnb_functional.dequantize_4bit(raw_weight.reshape(-1), quant_state).reshape(quant_state.shape).float()


def _causal_mask(dtype: torch.dtype, device: torch.device, seq_len: int) -> torch.Tensor | None:
    if seq_len <= 1:
        return None
    min_val = torch.finfo(dtype).min
    q_pos = torch.arange(seq_len, device=device)
    k_pos = torch.arange(seq_len, device=device)
    future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
    return future.to(dtype=dtype).mul(min_val).unsqueeze(0).unsqueeze(0)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "What is the capital of France?"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    print(f"prompt_len={int(input_ids.shape[1])} ids={input_ids[0].tolist()}", flush=True)

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
    captured_input = None
    captured_runtime_out = None

    with torch.no_grad():
        for layer_idx in range(runtime.num_layers):
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
            if layer_idx == TARGET_LAYER:
                captured_input = mlp_input.detach().cpu().float().view(seq_len, -1).contiguous()
                captured_runtime_out = mlp_out.detach().cpu().float().view(seq_len, -1).contiguous()
            all_hidden = residual + mlp_out
            if layer_idx in {0, 1, 2, 10, 62, 125}:
                print(
                    f"layer={layer_idx} hidden_last_std={float(all_hidden[0, -1, :].float().std()):.6f} "
                    f"mlp_last_std={float(mlp_out[0, -1, :].float().std()):.6f}",
                    flush=True,
                )

    if captured_input is None or captured_runtime_out is None:
        raise RuntimeError("Target layer was not captured")

    prefix = f"model.layers.{TARGET_LAYER}.mlp."
    print(f"captured_input_std={float(captured_input.std()):.6f}", flush=True)
    print(f"runtime_out_std={float(captured_runtime_out.std()):.6f}", flush=True)
    print(f"runtime_out_last_std={float(captured_runtime_out[-1].std()):.6f}", flush=True)

    gate_weight = _load_dequant_float(runtime, f"{prefix}gate_proj.weight")
    gate = F.linear(captured_input, gate_weight)
    del gate_weight
    gc.collect()

    up_weight = _load_dequant_float(runtime, f"{prefix}up_proj.weight")
    up = F.linear(captured_input, up_weight)
    del up_weight
    gc.collect()

    activated = F.silu(gate) * up
    del gate, up
    gc.collect()

    down_weight = _load_dequant_float(runtime, f"{prefix}down_proj.weight")
    reference_out = F.linear(activated, down_weight)
    del down_weight, activated
    gc.collect()

    diff = (captured_runtime_out - reference_out).abs()
    denom = reference_out.abs().clamp_min(1e-6)
    print(f"reference_out_std={float(reference_out.std()):.6f}", flush=True)
    print(f"reference_out_last_std={float(reference_out[-1].std()):.6f}", flush=True)
    print(f"diff_max_abs={float(diff.max()):.6f}", flush=True)
    print(f"diff_mean_abs={float(diff.mean()):.6f}", flush=True)
    print(f"diff_mean_rel={float((diff / denom).mean()):.6f}", flush=True)


if __name__ == "__main__":
    main()
