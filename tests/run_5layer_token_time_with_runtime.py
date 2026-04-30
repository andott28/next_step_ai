from __future__ import annotations

import os
import torch

from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime
import tests.test_5layer_token_time as bench

MODEL = os.getenv("MODEL", "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit")
BASIS = os.getenv("BASIS", "results/mlp_basis_intermediate_full126.pt")
ATTN = os.getenv("ATTN", "results/attn_head_importance_405b.pt")


def build_runtime() -> StreamingLlamaRuntime:
    rt = StreamingLlamaRuntime(
        model_name_or_path=MODEL,
        sparse_basis_path=BASIS,
        attn_head_importance_path=ATTN,
        vram_hot_cache_gb=float(os.getenv("VRAM_HOT_CACHE_GB", "4.0")),
        taylor_layers=[],
        materialize_lm_head=False,
    )
    return rt


def prewarm_and_seed(rt: StreamingLlamaRuntime) -> None:
    rt.reset_caches()
    orig_num_layers = rt.num_layers
    rt.num_layers = 5
    rt._traffic_current_phase = "decode"
    try:
        rt.pre_warm_vram_hot_cache()

        # Seed KV cache to 256 tokens for layers 0..4 only.
        dummy_token = torch.zeros(1, 1, dtype=torch.long, device=rt.device)
        with torch.no_grad():
            for pos in range(256):
                rt.forward_token(dummy_token, position_index=pos, use_attention_cache=True)
        if rt.device.type == "cuda":
            torch.cuda.synchronize(rt.device)
    finally:
        rt.num_layers = orig_num_layers
        # leave _traffic_current_phase = "decode" so bench.run() also uses compact sparse attn


def main() -> None:
    print("Building runtime...")
    rt = build_runtime()
    print("Pre-warming + seeding KV cache for layers 0-4...")
    prewarm_and_seed(rt)
    print("Running 5-layer token-time benchmark...")
    bench.run(rt)


if __name__ == "__main__":
    main()
