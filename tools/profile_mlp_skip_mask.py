"""profile_mlp_skip_mask.py — record per-layer MLP contribution norms and save a static skip mask.

Patches _mlp_forward_dispatch to record ||MLP_out|| / ||MLP_in|| per layer during a normal
sparse prefill.  Layers below the threshold are marked for permanent skipping.

Usage:
    python tools/profile_mlp_skip_mask.py \
        --model-name unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit \
        --sparse-basis-path results/mlp_basis_intermediate_full126.pt \
        --output results/mlp_skip_mask.pt \
        --threshold 0.005
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime

SAMPLE_PROMPTS = [
    "The theory of relativity states that",
    "In machine learning, gradient descent is",
    "The capital of France is",
    "To implement a binary search tree in Python,",
    "The human brain contains approximately",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit")
    parser.add_argument("--sparse-basis-path", default="results/mlp_basis_intermediate_full126.pt")
    parser.add_argument("--output", default="results/mlp_skip_mask.pt")
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--vram-hot-cache-gb", type=float, default=4.0)
    args = parser.parse_args()

    print("[profile] Loading runtime ...", flush=True)
    runtime = StreamingLlamaRuntime(
        model_name_or_path=args.model_name,
        sparse_basis_path=args.sparse_basis_path,
        vram_hot_cache_gb=args.vram_hot_cache_gb,
        taylor_layers=[],
    )

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"[profile] tokenizer failed: {e}", flush=True)
        sys.exit(1)

    num_layers = runtime.num_layers
    score_sum = torch.zeros(num_layers, dtype=torch.float64)
    count     = torch.zeros(num_layers, dtype=torch.int64)


    _orig_dispatch = runtime._mlp_forward_dispatch.__func__

    def _patched_dispatch(self, layer_idx, layer, mlp_input):
        out = _orig_dispatch(self, layer_idx, layer, mlp_input)
        lidx = int(layer_idx)
        in_norm  = float(mlp_input.float().norm().item())
        out_norm = float(out.float().norm().item())
        score_sum[lidx] += out_norm / max(in_norm, 1e-8)
        count[lidx]     += 1
        return out

    import types
    runtime._mlp_forward_dispatch = types.MethodType(_patched_dispatch, runtime)

    for prompt in SAMPLE_PROMPTS:
        print(f"[profile] prompt: {prompt[:50]!r} ...", flush=True)
        ids = tokenizer.encode(prompt, return_tensors="pt")
        runtime.reset_caches()
        runtime._set_traffic_phase("prefill")
        runtime._sparse_mlp_prefill_mode = "sparse"
        with torch.no_grad():
            runtime._forward_prefill(ids.to(runtime.device), compute_logits=False)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize()

    mean_scores = (score_sum / count.float().clamp(min=1)).float()
    skip_mask   = mean_scores < args.threshold
    n_skip = int(skip_mask.sum().item())

    print(f"\n[profile] Results (threshold={args.threshold}):", flush=True)
    for i, (s, skip) in enumerate(zip(mean_scores.tolist(), skip_mask.tolist(), strict=False)):
        flag = " ← SKIP" if skip else ""
        print(f"  layer {i:3d}: score={s:.6f}{flag}")

    print(f"\n[profile] {n_skip}/{num_layers} layers will be statically skipped.", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"skip_mask": skip_mask, "scores": mean_scores}, str(out_path))
    print(f"[profile] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
