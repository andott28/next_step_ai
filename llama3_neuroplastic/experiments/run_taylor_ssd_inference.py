#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

try:
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from .strict_decode_metrics import evaluate_decode_prefixes
except ImportError:  # pragma: no cover
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from strict_decode_metrics import evaluate_decode_prefixes


SMOKE_PROMPTS = [
    "Write one sentence about Oslo.",
    "Explain why recurrent attention can lower cache pressure.",
    "Describe block-bank sparse MLP in one short paragraph.",
    "Give two concise facts about grouped-query attention.",
    "Write one practical tip for debugging decode instability.",
]


def _safe_console_text(text: str) -> str:
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _parse_layer_selection(spec: str) -> Optional[List[int]]:
    if spec is None or str(spec).strip() == "":
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {token}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return sorted(out)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Taylor-SSD attention inference and strict decode diagnostics.")
    p.add_argument("--model-name", type=str, default="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    p.add_argument("--prompt", action="append", default=[], help="Prompt to run; may be repeated")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--allow-cache", action="store_true")
    p.add_argument("--enable-sparse-mlp", action="store_true")
    p.add_argument("--sca-block-size", type=int, default=32)
    p.add_argument("--sca-top-k", type=int, default=2)
    p.add_argument("--sca-bottom-buffer-layers", type=int, default=2)
    p.add_argument("--sca-top-buffer-layers", type=int, default=None)
    p.add_argument("--sca-decode-guard-layers", type=int, default=12)
    p.add_argument("--sparse-mlp-manifest", type=str, default="")
    p.add_argument("--router-state-checkpoint", type=str, default="")
    p.add_argument(
        "--sca-sparse-placement",
        type=str,
        choices=["input_mask", "output_sparse", "intermediate_group", "learned_basis"],
        default="input_mask",
    )
    p.add_argument(
        "--sca-routing-mode",
        type=str,
        choices=["spatial_grid", "semantic_latent"],
        default="spatial_grid",
    )
    p.add_argument("--sca-dense-anchor-stride", type=int, default=0,
                   help="Interleave a dense anchor layer every N layers to reset the manifold (0=none)")
    p.add_argument("--taylor-layers", type=str, default="", help="Layer selection, for example '8-23' or '8,9,10'")
    p.add_argument("--taylor-order", type=int, default=2)
    p.add_argument("--taylor-symmetric-quadratic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--taylor-eps", type=float, default=1e-6)
    p.add_argument("--taylor-force-disable-optimized-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--taylor-feature-map",
        type=str,
        default="hybrid_performer",
        choices=["hybrid_performer", "elu", "taylor"],
        help="Tail feature map for hybrid Taylor attention. 'hybrid_performer' is the new default.",
    )
    p.add_argument("--taylor-state-decay", type=float, default=1.0,
                   help="Per-step recurrent state decay; 1.0=no decay, <1.0 gives recency bias")
    p.add_argument("--taylor-local-window", type=int, default=64,
                   help="Exact causal softmax window retained before evicting tokens into the recurrent tail")
    p.add_argument("--taylor-feature-dim", type=int, default=64,
                   help="Random feature width for the Performer-style recurrent tail")
    p.add_argument("--taylor-bottom-buffer", type=int, default=4,
                   help="Number of bottom layers to keep as dense softmax attention")
    p.add_argument("--taylor-top-buffer", type=int, default=4,
                   help="Number of top layers to keep as dense softmax attention")
    p.add_argument("--strict-metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-rollout-horizon", type=int, default=8)
    p.add_argument("--dump-json", type=str, default="")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    prompts = list(args.prompt) if args.prompt else list(SMOKE_PROMPTS)
    taylor_layers = _parse_layer_selection(args.taylor_layers)

    model = NeuroplasticLlama(
        model_name=str(args.model_name),
        neuroplasticity_enabled=bool(args.enable_sparse_mlp),
        sca_block_size=int(args.sca_block_size),
        sca_sparse_placement=str(args.sca_sparse_placement),
        sca_routing_mode=str(args.sca_routing_mode),
        sca_top_k=int(args.sca_top_k),
        sca_dense_anchor_stride=int(args.sca_dense_anchor_stride),
        sca_bottom_buffer_layers=int(args.sca_bottom_buffer_layers),
        sca_decode_guard_layers=int(args.sca_decode_guard_layers),
        attention_hybrid_enabled=False,
        attention_gqa_mamba_enabled=False,
        attention_taylor_ssd_enabled=True,
        attention_taylor_layers=taylor_layers,
        attention_taylor_order=int(args.taylor_order),
        attention_taylor_symmetric_quadratic=bool(args.taylor_symmetric_quadratic),
        attention_taylor_eps=float(args.taylor_eps),
        attention_taylor_force_disable_optimized_cache=bool(args.taylor_force_disable_optimized_cache),
        attention_taylor_feature_map=str(args.taylor_feature_map),
        attention_taylor_state_decay=float(args.taylor_state_decay),
        attention_taylor_local_window=int(args.taylor_local_window),
        attention_taylor_feature_dim=int(args.taylor_feature_dim),
        attention_taylor_bottom_buffer=int(args.taylor_bottom_buffer),
        attention_taylor_top_buffer=int(args.taylor_top_buffer),
    )
    if args.sca_top_buffer_layers is not None:
        model.buffer_layers = max(int(args.sca_top_buffer_layers), 0)
    model.eval()

    if args.sparse_mlp_manifest:
        load_info = model.load_sparse_mlp_bank_manifest(str(args.sparse_mlp_manifest), strict=False)
        print(f"sparse_mlp_manifest_load_info={load_info}")
    if args.router_state_checkpoint:
        router_info = model.load_sparse_router_state(str(args.router_state_checkpoint), strict=False)
        print(f"router_state_load_info={router_info}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                top_p=float(args.top_p),
                use_cache=bool(args.allow_cache),
                task_id=0,
            )
        elapsed = time.perf_counter() - t0
        total_new = max(int(generated.shape[-1] - input_ids.shape[-1]), 1)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        row = {
            "prompt_idx": int(idx),
            "latency_s": float(elapsed),
            "tok_s": float(total_new / max(elapsed, 1e-9)),
            "text": text,
        }
        rows.append(row)
        print(f"[prompt {idx}] latency_s={row['latency_s']:.3f} tok_s={row['tok_s']:.3f}")
        print(f"[prompt {idx}] text={_safe_console_text(text)}")

    strict_metrics: Dict[str, Any] = {}
    if bool(args.strict_metrics):
        prefixes: List[Dict[str, torch.Tensor]] = []
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            prefixes.append(
                {
                    "input_ids": encoded["input_ids"].to(model.device),
                    "attention_mask": encoded["attention_mask"].to(model.device),
                }
            )
        strict_metrics = evaluate_decode_prefixes(
            model=model,
            tokenizer=tokenizer,
            prefixes=prefixes,
            rollout_horizon=int(args.strict_rollout_horizon),
            use_cache=bool(args.allow_cache),
            task_id=0,
        )
        print(
            "strict_metrics="
            + json.dumps(
                {
                    "dense_top1_rank_median": strict_metrics.get("dense_top1_rank_median", 0.0),
                    "rollout_kl_mean": strict_metrics.get("rollout_kl_mean", 0.0),
                    "final_hidden_cosine_mean": strict_metrics.get("final_hidden_cosine_mean", 0.0),
                    "degenerate_frac": strict_metrics.get("quality", {}).get("degenerate_frac", 0.0),
                }
            )
        )

    payload = {
        "model_name": str(args.model_name),
        "taylor_layers": model.get_effective_taylor_layers(),
        "runtime_flags": dict(getattr(model, "_runtime_flags", {})),
        "rows": rows,
        "strict_metrics": strict_metrics,
        "sparse_mlp_diagnostics": model.get_sparse_mlp_diagnostics(),
    }
    if args.dump_json:
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote diagnostics: {out_path}")


if __name__ == "__main__":
    main()
