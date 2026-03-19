from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from .strict_decode_metrics import evaluate_decode_prefixes
except ImportError:  # pragma: no cover
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from strict_decode_metrics import evaluate_decode_prefixes


SMOKE_PROMPTS = [
    "Write one sentence about Norway.",
    "Explain why recurrent models can reduce memory traffic.",
    "Describe a sparse MLP routing system.",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile decode sensitivity of individual sparse MLP layers.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sca-recalibrated-checkpoint", type=str, default="")
    p.add_argument("--output-path", type=str, required=True)
    p.add_argument("--layers", type=str, default="")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--validation-prefix-count", type=int, default=128)
    p.add_argument("--validation-prefix-length", type=int, default=64)
    p.add_argument("--sca-basis-rank", type=int, default=96)
    p.add_argument("--sca-bottom-buffer-layers", type=int, default=2)
    p.add_argument("--sca-decode-guard-layers", type=int, default=12)
    return p.parse_args()


def _parse_layer_selection(spec: str | None) -> List[int] | None:
    if spec is None or str(spec).strip() == "":
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid layer range: {token}")
            for value in range(start, end + 1):
                out.add(int(value))
        else:
            out.add(int(token))
    return sorted(out)


def _build_validation_prefixes(tokenizer: Any, *, count: int, max_prefix_len: int) -> List[Dict[str, torch.Tensor]]:
    if load_dataset is None:
        return []
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    dataset = dataset.filter(lambda x: isinstance(x["text"], str) and len(x["text"].strip()) > 0)
    prefixes: List[Dict[str, torch.Tensor]] = []
    for row in dataset:
        encoded = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=int(max_prefix_len) + 8,
        )
        input_ids = encoded["input_ids"]
        if input_ids.shape[-1] < 9:
            continue
        prefix_len = int(min(max_prefix_len, input_ids.shape[-1] - 8))
        prefixes.append(
            {
                "input_ids": input_ids[:, :prefix_len].cpu(),
                "attention_mask": encoded["attention_mask"][:, :prefix_len].cpu(),
            }
        )
        if len(prefixes) >= int(count):
            break
    return prefixes


def _set_active_layer(model: NeuroplasticLlama, target_layer: int) -> None:
    for layer_idx, wrapper in enumerate(model.sca_sparse_mlps):
        wrapper.enabled_fn = lambda _idx, layer_idx=layer_idx, target_layer=target_layer: bool(
            model.neuroplasticity_enabled and layer_idx == target_layer
        )


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return [0.0 for _ in values]
    return [float((value - min_v) / (max_v - min_v)) for value in values]


def _bucket_by_rank(sorted_layers: List[int]) -> Dict[int, str]:
    total = len(sorted_layers)
    if total <= 0:
        return {}
    high_cut = max(1, total // 3)
    medium_cut = max(high_cut + 1, (2 * total) // 3)
    out: Dict[int, str] = {}
    for rank, layer_idx in enumerate(sorted_layers):
        if rank < high_cut:
            out[int(layer_idx)] = "high"
        elif rank < medium_cut:
            out[int(layer_idx)] = "medium"
        else:
            out[int(layer_idx)] = "low"
    return out


def _budget_for_bucket(bucket: str) -> Dict[str, int]:
    if bucket == "high":
        return {"basis_rank": 96, "basis_top_k": 12, "top_k": 6}
    if bucket == "medium":
        return {"basis_rank": 64, "basis_top_k": 8, "top_k": 4}
    return {"basis_rank": 48, "basis_top_k": 6, "top_k": 3}


def main() -> None:
    args = _parse_args()
    artifact = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(artifact, dict):
        raise RuntimeError("Hybrid checkpoint must be a dict artifact")
    model_name = str(artifact.get("model_name", "unsloth/Meta-Llama-3.1-8B-bnb-4bit"))
    model = NeuroplasticLlama(
        model_name=model_name,
        neuroplasticity_enabled=True,
        sca_use_cuda=True,
        sca_spmm_impl="cuda_spmm",
        sca_sparse_placement="learned_basis",
        sca_routing_mode="semantic_latent",
        sca_bottom_buffer_layers=int(args.sca_bottom_buffer_layers),
        sca_decode_guard_layers=int(args.sca_decode_guard_layers),
        sca_basis_rank=int(args.sca_basis_rank),
        sca_basis_top_k=12,
        sca_top_k=6,
        sca_adaptive_top_k=False,
        sca_stability_dense_fallback_threshold=0.0,
        attention_hybrid_enabled=True,
        attention_hybrid_layers=artifact.get("layer_selection"),
        attention_hybrid_target_rank=artifact.get("target_rank", 16),
        attention_hybrid_variance_threshold=float(artifact.get("variance_threshold", 0.90)),
        attention_hybrid_state_dim=artifact.get("state_dim"),
        attention_hybrid_force_no_cache=False,
        attention_sparse_mode=True,
        strict_decode_enable_repetition_penalty=False,
        strict_decode_upper_layer_cap_enabled=False,
    )
    model.load_hybrid_attention_state(args.checkpoint, strict=True)
    if str(args.sca_recalibrated_checkpoint).strip():
        model.load_sca_recalibration_state(str(args.sca_recalibrated_checkpoint).strip(), strict=True)
    model.set_sparse_attention_mode(strict_fully_sparse=True)
    model.disable_task_bias_injection = True
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    validation_prefixes = _build_validation_prefixes(
        tokenizer,
        count=int(args.validation_prefix_count),
        max_prefix_len=int(args.validation_prefix_length),
    )

    layer_metrics: List[Dict[str, Any]] = []
    auto_selected_layers = [
        idx for idx in range(len(model.sca_sparse_mlps)) if bool(model._sparse_layer_enabled(int(idx)))
    ]
    requested_layers = _parse_layer_selection(args.layers)
    if requested_layers is not None:
        selected_layers = [idx for idx in requested_layers if idx in auto_selected_layers]
    else:
        selected_layers = auto_selected_layers
    if not selected_layers:
        raise RuntimeError("No layers selected for profiling")
    print(f"[profile] selected layers: {selected_layers}")

    for pos, layer_idx in enumerate(selected_layers):
        print(f"[profile] running layer {layer_idx} ({pos + 1}/{len(selected_layers)})")
        _set_active_layer(model, target_layer=int(layer_idx))
        smoke = evaluate_decode_prefixes(
            model,
            tokenizer,
            SMOKE_PROMPTS,
            task_id=0,
            rollout_horizon=int(args.max_new_tokens),
            use_cache=True,
        )
        validation = evaluate_decode_prefixes(
            model,
            tokenizer,
            validation_prefixes,
            task_id=0,
            rollout_horizon=int(args.max_new_tokens),
            use_cache=True,
        )
        layer_metrics.append(
            {
                "layer_idx": int(layer_idx),
                "dense_top1_rank_median": float(validation.get("dense_top1_rank_median", 0.0)),
                "dense_top1_rank_p95": float(validation.get("dense_top1_rank_p95", 0.0)),
                "rollout_kl_mean": float(validation.get("rollout_kl_mean", 0.0)),
                "final_hidden_cosine_mean": float(validation.get("final_hidden_cosine_mean", 0.0)),
                "rep_frac_mean": float(smoke.get("quality", {}).get("rep_frac_mean", 0.0)),
                "degenerate_frac": float(smoke.get("quality", {}).get("degenerate_frac", 0.0)),
            }
        )

    rollout_norm = _normalize([row["rollout_kl_mean"] for row in layer_metrics])
    rank_norm = _normalize([max(row["dense_top1_rank_p95"] - 1.0, 0.0) for row in layer_metrics])
    cosine_norm = _normalize([max(1.0 - row["final_hidden_cosine_mean"], 0.0) for row in layer_metrics])
    deg_norm = _normalize([row["degenerate_frac"] for row in layer_metrics])
    for idx, row in enumerate(layer_metrics):
        row["composite_score"] = float(
            (0.40 * rollout_norm[idx])
            + (0.30 * rank_norm[idx])
            + (0.20 * cosine_norm[idx])
            + (0.10 * deg_norm[idx])
        )

    ranked_layers = [row["layer_idx"] for row in sorted(layer_metrics, key=lambda item: item["composite_score"], reverse=True)]
    buckets = _bucket_by_rank(ranked_layers)
    recommended_overrides = {
        "basis_rank_by_layer": {},
        "basis_top_k_by_layer": {},
        "top_k_by_layer": {},
    }
    for row in layer_metrics:
        bucket = buckets[int(row["layer_idx"])]
        row["impact_bucket"] = bucket
        budget = _budget_for_bucket(bucket)
        recommended_overrides["basis_rank_by_layer"][str(row["layer_idx"])] = int(budget["basis_rank"])
        recommended_overrides["basis_top_k_by_layer"][str(row["layer_idx"])] = int(budget["basis_top_k"])
        recommended_overrides["top_k_by_layer"][str(row["layer_idx"])] = int(budget["top_k"])

    report = {
        "recommended_overrides": recommended_overrides,
        "layers": sorted(layer_metrics, key=lambda item: item["composite_score"], reverse=True),
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
