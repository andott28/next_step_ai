from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
    "Summarize grouped-query attention in two sentences.",
    "Describe a sparse MLP routing system.",
    "Give one short fact about Mamba-2.",
]


@dataclass
class SettingResult:
    top_k: int
    basis_top_k: int
    score: float
    smoke: Dict[str, Any]
    validation: Dict[str, Any]
    diagnostics: Dict[str, float]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-command SCA diagnostic wipe")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sca-recalibrated-checkpoint", type=str, default="")
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=16)
    p.add_argument("--sweep-top-k", type=str, default="3,4")
    p.add_argument("--sweep-basis-top-k", type=str, default="8,12")
    p.add_argument("--sca-basis-rank", type=int, default=96)
    p.add_argument("--sca-bottom-buffer-layers", type=int, default=2)
    p.add_argument("--sca-decode-guard-layers", type=int, default=12)
    p.add_argument(
        "--sca-routing-mode",
        type=str,
        choices=["spatial_grid", "semantic_latent"],
        default="semantic_latent",
    )
    p.add_argument("--validation-prefix-count", type=int, default=128)
    p.add_argument("--validation-prefix-length", type=int, default=64)
    return p.parse_args()


def _build_validation_prefixes(
    tokenizer: Any,
    *,
    count: int,
    max_prefix_len: int,
) -> List[Dict[str, torch.Tensor]]:
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


def _apply_setting(model: NeuroplasticLlama, *, top_k: int, basis_top_k: int, routing_mode: str) -> None:
    model.sca_config.routing_mode = str(routing_mode)
    model.sca_config.adaptive_top_k = False
    model.sca_config.top_k = int(top_k)
    model.sca_config.basis_top_k = int(basis_top_k)


def _score(smoke_quality: Dict[str, float], validation: Dict[str, Any], fallback_rate: float) -> float:
    return (
        3.0 * float(smoke_quality.get("degenerate_frac", 0.0))
        + 2.0 * float(smoke_quality.get("rep_frac_mean", 0.0))
        + 0.4 * float(validation.get("rollout_kl_mean", 0.0))
        + 0.3 * max(float(validation.get("dense_top1_rank_p95", 1.0)) - 1.0, 0.0)
        + 0.2 * max(1.0 - float(validation.get("final_hidden_cosine_mean", 1.0)), 0.0)
        + 0.5 * float(fallback_rate)
    )


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
        sca_routing_mode=str(args.sca_routing_mode),
        sca_bottom_buffer_layers=int(args.sca_bottom_buffer_layers),
        sca_decode_guard_layers=int(args.sca_decode_guard_layers),
        sca_basis_rank=int(args.sca_basis_rank),
        sca_basis_top_k=12,
        sca_top_k=4,
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
    model.strict_decode_enable_repetition_penalty = False
    model.strict_decode_upper_layer_cap_enabled = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    validation_prefixes = _build_validation_prefixes(
        tokenizer,
        count=int(args.validation_prefix_count),
        max_prefix_len=int(args.validation_prefix_length),
    )

    topk_vals = [int(v.strip()) for v in str(args.sweep_top_k).split(",") if v.strip()]
    basis_vals = [int(v.strip()) for v in str(args.sweep_basis_top_k).split(",") if v.strip()]
    setting_results: List[SettingResult] = []
    for top_k in topk_vals:
        for basis_top_k in basis_vals:
            _apply_setting(model, top_k=top_k, basis_top_k=basis_top_k, routing_mode=str(args.sca_routing_mode))
            smoke = evaluate_decode_prefixes(
                model,
                tokenizer,
                SMOKE_PROMPTS[:3],
                task_id=0,
                rollout_horizon=int(args.max_new_tokens),
                use_cache=True,
            )
            validation = evaluate_decode_prefixes(
                model,
                tokenizer,
                validation_prefixes,
                task_id=0,
                rollout_horizon=int(args.rollout_steps),
                use_cache=True,
            )
            diag = model.get_sparse_mlp_diagnostics()
            agg_diag = {
                "mean_dense_fallback_rate": float(diag.get("mean_dense_fallback_rate", 0.0)),
                "mean_touched_weight_fraction": float(diag.get("mean_touched_weight_fraction", 0.0)),
                "mean_latent_support_overlap": float(diag.get("mean_latent_support_overlap", 0.0)),
                "mean_latent_support_unique_fraction": float(diag.get("mean_latent_support_unique_fraction", 0.0)),
            }
            score = _score(smoke["quality"], validation, agg_diag["mean_dense_fallback_rate"])
            setting_results.append(
                SettingResult(
                    top_k=top_k,
                    basis_top_k=basis_top_k,
                    score=float(score),
                    smoke=smoke,
                    validation=validation,
                    diagnostics=agg_diag,
                )
            )

    best = min(setting_results, key=lambda r: r.score)
    balance = best.validation.get("balance", {})
    issues: List[str] = []
    if best.validation.get("dense_top1_rank_p95", 0.0) > 5.0:
        issues.append("dense_top1_rank_p95_high")
    if best.validation.get("rollout_kl_mean", 0.0) > 1.0:
        issues.append("rollout_kl_high")
    if best.validation.get("final_hidden_cosine_mean", 0.0) < 0.95:
        issues.append("final_hidden_cosine_low")
    if best.smoke.get("quality", {}).get("degenerate_frac", 0.0) > 0.2:
        issues.append("strict_decode_degeneracy_high")
    if best.diagnostics.get("mean_dense_fallback_rate", 0.0) > 0.0:
        issues.append("not_honest_sparse_runtime")
    if balance.get("max_latent_importance_fraction", 0.0) > 0.35:
        issues.append("latent_importance_collapsed")
    if balance.get("max_block_importance_fraction", 0.0) > 0.20:
        issues.append("block_importance_collapsed")

    report = {
        "best_setting": {
            "top_k": int(best.top_k),
            "basis_top_k": int(best.basis_top_k),
            "score": float(best.score),
        },
        "issues": issues,
        "setting_results": [
            {
                "top_k": int(r.top_k),
                "basis_top_k": int(r.basis_top_k),
                "score": float(r.score),
                "smoke": r.smoke,
                "validation": r.validation,
                "diagnostics": r.diagnostics,
            }
            for r in setting_results
        ],
        "recommended_runtime": {
            "strict_fully_sparse": True,
            "sca_sparse_placement": "learned_basis",
            "sca_routing_mode": str(args.sca_routing_mode),
            "sca_top_k": int(best.top_k),
            "sca_basis_top_k": int(best.basis_top_k),
            "sca_adaptive_top_k": False,
            "sca_bottom_buffer_layers": int(args.sca_bottom_buffer_layers),
            "sca_decode_guard_layers": int(args.sca_decode_guard_layers),
            "strict_decode_enable_repetition_penalty": False,
            "strict_decode_upper_layer_cap_enabled": False,
            "disable_task_bias_injection": True,
            "sca_fallback_threshold": 0.0,
        },
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
