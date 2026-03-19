from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

try:
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from .strict_decode_metrics import text_quality_metrics
except ImportError:  # pragma: no cover
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from strict_decode_metrics import text_quality_metrics


DEFAULT_PROMPTS = [
    "Write two clear factual sentences about Oslo.",
    "Explain photosynthesis in one clear sentence.",
]


def _parse_int_list(spec: str) -> List[int]:
    values: List[int] = []
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast sparse-depth probe (single model load, many decode-guard sweeps).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to hybrid_attention_state.pt")
    p.add_argument("--sca-recalibrated-checkpoint", type=str, required=True, help="Path to sca_recalibrated_state.pt")
    p.add_argument("--decode-guards", type=str, default="26,23,20,17,12")
    p.add_argument("--sca-bottom-buffer-layers", type=int, default=2)
    p.add_argument("--sca-basis-rank", type=int, default=96)
    p.add_argument("--sca-basis-top-k", type=int, default=8)
    p.add_argument("--sca-top-k", type=int, default=3)
    p.add_argument("--semantic-block-score-normalized", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max-new-tokens", type=int, default=10)
    p.add_argument("--allow-cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--output-json", type=str, default="")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    guard_values = _parse_int_list(args.decode_guards)
    prompts = list(args.prompt) if args.prompt else list(DEFAULT_PROMPTS)

    artifact = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(artifact, dict):
        raise RuntimeError("Hybrid checkpoint must be a dict artifact")
    model_name = str(artifact.get("model_name", "unsloth/Meta-Llama-3.1-8B-bnb-4bit"))

    model = NeuroplasticLlama(
        model_name=model_name,
        neuroplasticity_enabled=True,
        sca_use_cuda=True,
        sca_top_k=int(args.sca_top_k),
        sca_basis_rank=int(args.sca_basis_rank),
        sca_basis_top_k=int(args.sca_basis_top_k),
        sca_routing_mode="semantic_latent",
        sca_semantic_block_score_normalized=bool(args.semantic_block_score_normalized),
        sca_spmm_impl="dense",
        sca_sparse_placement="learned_basis",
        sca_stability_dense_fallback_threshold=0.0,
        sca_adaptive_top_k=False,
        sca_bottom_buffer_layers=int(args.sca_bottom_buffer_layers),
        attention_hybrid_enabled=True,
        attention_hybrid_layers=artifact.get("layer_selection"),
        attention_hybrid_target_rank=artifact.get("target_rank", 16),
        attention_hybrid_variance_threshold=float(artifact.get("variance_threshold", 0.90)),
        attention_hybrid_state_dim=artifact.get("state_dim"),
        attention_hybrid_force_no_cache=not bool(args.allow_cache),
        attention_sparse_mode=False,
        strict_decode_enable_repetition_penalty=False,
        strict_decode_upper_layer_cap_enabled=False,
    )
    model.load_hybrid_attention_state(str(args.checkpoint), strict=True)
    model.load_sca_recalibration_state(str(args.sca_recalibrated_checkpoint), strict=True)
    model.disable_task_bias_injection = True
    model.strict_decode_enable_repetition_penalty = False
    model.strict_decode_upper_layer_cap_enabled = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows: List[Dict[str, Any]] = []
    for guard in guard_values:
        model.decode_guard_layers = int(max(guard, 0))
        prompt_rows: List[Dict[str, Any]] = []
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    use_cache=bool(args.allow_cache),
                    task_id=0,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            quality = text_quality_metrics(text)
            prompt_rows.append(
                {
                    "prompt": prompt,
                    "text": text,
                    "quality": quality,
                }
            )
        diag = model.get_sparse_mlp_diagnostics()
        mean_deg = float(sum(r["quality"]["degenerate"] for r in prompt_rows) / max(len(prompt_rows), 1))
        mean_rep = float(sum(r["quality"]["rep_frac"] for r in prompt_rows) / max(len(prompt_rows), 1))
        rows.append(
            {
                "decode_guard_layers": int(guard),
                "mean_dense_fallback_rate": float(diag.get("mean_dense_fallback_rate", 0.0)),
                "mean_touched_weight_fraction": float(diag.get("mean_touched_weight_fraction", 0.0)),
                "mean_deg": mean_deg,
                "mean_rep": mean_rep,
                "prompts": prompt_rows,
            }
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "sca_recalibrated_checkpoint": str(args.sca_recalibrated_checkpoint),
        "decode_guards": guard_values,
        "rows": rows,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if str(args.output_json).strip():
        out_path = Path(str(args.output_json).strip())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
