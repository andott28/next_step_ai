from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import LlamaForCausalLM

from gqa_mamba_rank_collapse import (
    EightGBExecutionBudget,
    GQALayout,
    TokenLatencyEstimate,
    collapse_llama_gqa_attention_layer,
)


def _layer_report(collapsed_groups: List, layer_idx: int) -> Dict[str, float]:
    ranks = [int(g.rank) for g in collapsed_groups]
    variances = [float(g.explained_variance) for g in collapsed_groups]
    return {
        "layer_idx": float(layer_idx),
        "mean_rank": float(sum(ranks) / max(len(ranks), 1)),
        "max_rank": float(max(ranks) if ranks else 0.0),
        "min_rank": float(min(ranks) if ranks else 0.0),
        "mean_explained_variance": float(sum(variances) / max(len(variances), 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-1/3 GQA-Mamba rank-collapse pipeline report.")
    p.add_argument("--model-name", type=str, required=True, help="HF checkpoint with Llama attention layers")
    p.add_argument("--target-rank", type=int, default=None, help="Fixed truncated rank per GQA group")
    p.add_argument("--variance-threshold", type=float, default=0.90, help="Energy threshold when target-rank is unset")
    p.add_argument("--max-layers", type=int, default=None, help="Optional layer cap for quick dry runs")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--output", type=str, default="", help="Optional JSON report path")
    args = p.parse_args()

    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    model = LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map="cpu")
    cfg = model.config
    layout = GQALayout(
        num_attention_heads=int(cfg.num_attention_heads),
        num_key_value_heads=int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        head_dim=int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)),
    )

    layers = model.model.layers
    if args.max_layers is not None:
        layers = layers[: max(0, int(args.max_layers))]

    layer_reports = []
    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        collapsed = collapse_llama_gqa_attention_layer(
            w_q=attn.q_proj.weight.detach().float().cpu(),
            w_k=attn.k_proj.weight.detach().float().cpu(),
            w_v=attn.v_proj.weight.detach().float().cpu(),
            w_o=attn.o_proj.weight.detach().float().cpu(),
            layout=layout,
            target_rank=args.target_rank,
            variance_threshold=args.variance_threshold,
        )
        layer_reports.append(_layer_report(collapsed, idx))

    budget = EightGBExecutionBudget()
    latency = TokenLatencyEstimate()
    report = {
        "model_name": args.model_name,
        "num_layers_profiled": len(layer_reports),
        "variance_threshold": float(args.variance_threshold),
        "target_rank": None if args.target_rank is None else int(args.target_rank),
        "layer_reports": layer_reports,
        "phase3_execution_math": {
            "vram_budget_gb": 8.0,
            "total_hot_vram_gb": budget.total_vram_gb(),
            "fits": budget.fits(8.0),
            "pcie_transfer_seconds": latency.transfer_seconds(),
            "token_seconds_estimate": latency.total_seconds(),
            "tokens_per_second_estimate": latency.tokens_per_second(),
        },
    }

    text = json.dumps(report, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
