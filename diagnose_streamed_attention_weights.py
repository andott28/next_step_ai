from __future__ import annotations

import gc
import sys

import torch

sys.path.insert(0, ".")

from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime


MODEL = "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"


def _compare(name: str, streamed: torch.Tensor, reference: torch.Tensor) -> None:
    streamed_cpu = streamed.detach().cpu().float()
    reference_cpu = reference.detach().cpu().float()
    diff = (streamed_cpu - reference_cpu).abs()
    denom = reference_cpu.abs().clamp_min(1e-6)
    print(
        f"{name} "
        f"streamed_std={float(streamed_cpu.std()):.6f} "
        f"reference_std={float(reference_cpu.std()):.6f} "
        f"max_abs={float(diff.max()):.6f} "
        f"mean_abs={float(diff.mean()):.6f} "
        f"mean_rel={float((diff / denom).mean()):.6f}"
    )


def main() -> None:
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
    runtime._set_traffic_phase("prefill")
    for layer_idx in (0, 125):
        layer = runtime._load_layer(layer_idx)
        module_by_suffix = {
            "self_attn.q_proj.weight": layer.self_attn.q_proj.weight,
            "self_attn.k_proj.weight": layer.self_attn.k_proj.weight,
            "self_attn.v_proj.weight": layer.self_attn.v_proj.weight,
            "self_attn.o_proj.weight": layer.self_attn.o_proj.weight,
            "input_layernorm.weight": layer.input_layernorm.weight,
            "post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        }

        for suffix, streamed_weight in module_by_suffix.items():
            full_name = f"model.layers.{layer_idx}.{suffix}"
            reference = runtime.loader.load_parameter(full_name).to(dtype=runtime.dtype)
            if suffix.endswith("layernorm.weight"):
                reference = reference + torch.ones_like(reference)
            _compare(full_name, streamed_weight, reference)
            del reference
            gc.collect()

    final_norm = runtime.loader.load_parameter("model.norm.weight").to(device=runtime.device, dtype=runtime.dtype)
    _compare("model.norm.weight", runtime.norm.weight, final_norm)

    lm_head = runtime._ensure_lm_head_weight_cpu()
    print(
        f"lm_head_name={runtime._lm_head_weight_name} "
        f"shape={list(lm_head.shape)} "
        f"dtype={lm_head.dtype} "
        f"std={float(lm_head.float().std()):.6f} "
        f"paris_row_std={float(lm_head[12366].float().std()):.6f} "
        f"igen_row_std={float(lm_head[6569].float().std()):.6f}"
    )


if __name__ == "__main__":
    main()
