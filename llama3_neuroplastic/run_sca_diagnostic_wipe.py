from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
except ImportError:  # pragma: no cover
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama


SMOKE_PROMPTS = [
    "Write one sentence about Norway.",
    "Explain why recurrent models can reduce memory traffic.",
    "Summarize grouped-query attention in two sentences.",
    "Describe a sparse MLP routing system.",
    "Give one short fact about Mamba-2.",
]


def _char_repeat4_fraction(text: str) -> float:
    s = text.lower()
    if len(s) < 8:
        return 0.0
    total = max(len(s) - 3, 1)
    rep = 0
    for idx in range(len(s) - 7):
        if s[idx : idx + 4] == s[idx + 4 : idx + 8]:
            rep += 1
    return float(rep / total)


def _text_quality_metrics(text: str) -> Dict[str, float]:
    tokens = [tok for tok in text.replace("\n", " ").split(" ") if tok]
    if not tokens:
        return {"distinct1": 0.0, "max_run": 0.0, "rep_frac": 0.0, "char_repeat4_frac": 0.0, "degenerate": 0.0}
    distinct1 = float(len(set(tokens)) / len(tokens))
    run = 1
    max_run = 1
    rep = 0
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            run += 1
            rep += 1
            max_run = max(max_run, run)
        else:
            run = 1
    rep_frac = float(rep / max(len(tokens) - 1, 1))
    repeat4 = _char_repeat4_fraction(text)
    deg = float(1.0 if (max_run >= 4 or rep_frac > 0.2 or repeat4 > 0.2) else 0.0)
    return {
        "distinct1": distinct1,
        "max_run": float(max_run),
        "rep_frac": rep_frac,
        "char_repeat4_frac": repeat4,
        "degenerate": deg,
    }


def _agg(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "distinct1_mean": float(sum(r["distinct1"] for r in rows) / max(len(rows), 1)),
        "max_run_mean": float(sum(r["max_run"] for r in rows) / max(len(rows), 1)),
        "rep_frac_mean": float(sum(r["rep_frac"] for r in rows) / max(len(rows), 1)),
        "char_repeat4_frac_mean": float(sum(r["char_repeat4_frac"] for r in rows) / max(len(rows), 1)),
        "degenerate_frac": float(sum(r["degenerate"] for r in rows) / max(len(rows), 1)),
    }


def _run_strict_quality(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        mask = enc["attention_mask"].to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
                task_id=0,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        rows.append({"prompt": prompt, "text": text, **_text_quality_metrics(text)})
    return {"rows": rows, "agg": _agg(rows)}


def _dense_sparse_drift(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompt: str,
    rollout_steps: int,
) -> Dict[str, float]:
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    kl_values: List[float] = []
    entropy_values: List[float] = []
    margin_values: List[float] = []
    mismatch = 0
    for _ in range(int(rollout_steps)):
        model.neuroplasticity_enabled = False
        with torch.no_grad():
            dense = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, task_id=0)
        model.neuroplasticity_enabled = True
        with torch.no_grad():
            sparse = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, task_id=0)
        dlog = dense.logits[:, -1, :].float()
        slog = sparse.logits[:, -1, :].float()
        dprob = torch.softmax(dlog, dim=-1)
        sprob = torch.softmax(slog, dim=-1)
        kl = F.kl_div(torch.log_softmax(slog, dim=-1), dprob, reduction="batchmean")
        entropy = (-(sprob * sprob.clamp_min(1e-8).log()).sum()).item()
        top2_vals, top2_idx = torch.topk(sprob[0], 2)
        margin = float((top2_vals[0] - top2_vals[1]).item())
        dnext = int(torch.argmax(dlog, dim=-1).item())
        snext = int(torch.argmax(slog, dim=-1).item())
        mismatch += int(dnext != snext)
        kl_values.append(float(kl.detach().cpu().item()))
        entropy_values.append(float(entropy))
        margin_values.append(margin)
        ids = torch.cat([ids, torch.tensor([[snext]], device=ids.device, dtype=ids.dtype)], dim=-1)
        mask = torch.cat([mask, torch.ones((1, 1), device=mask.device, dtype=mask.dtype)], dim=-1)
    return {
        "mean_kl": float(sum(kl_values) / max(len(kl_values), 1)),
        "max_kl": float(max(kl_values) if kl_values else 0.0),
        "mean_sparse_entropy": float(sum(entropy_values) / max(len(entropy_values), 1)),
        "mean_top1_margin": float(sum(margin_values) / max(len(margin_values), 1)),
        "token_mismatch_rate": float(mismatch / max(int(rollout_steps), 1)),
    }


def _apply_setting(model: NeuroplasticLlama, *, top_k: int, basis_top_k: int, adaptive: bool) -> None:
    model.sca_config.adaptive_top_k = bool(adaptive)
    model.sca_config.top_k = int(top_k)
    model.sca_config.basis_top_k = int(basis_top_k)


@dataclass
class SettingResult:
    top_k: int
    basis_top_k: int
    score: float
    quality: Dict[str, float]
    drift: Dict[str, float]
    diagnostics: Dict[str, float]


def _score(quality: Dict[str, float], drift: Dict[str, float], fallback_rate: float) -> float:
    return (
        3.0 * quality["degenerate_frac"]
        + 2.0 * quality["rep_frac_mean"]
        + 0.2 * quality["max_run_mean"]
        + 0.02 * drift["mean_kl"]
        + 0.5 * fallback_rate
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-command SCA diagnostic wipe")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sca-recalibrated-checkpoint", type=str, default="")
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=12)
    p.add_argument("--sweep-top-k", type=str, default="3,4")
    p.add_argument("--sweep-basis-top-k", type=str, default="8,16")
    p.add_argument("--sca-basis-rank", type=int, default=64)
    return p.parse_args()


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
        sca_basis_rank=int(args.sca_basis_rank),
        sca_basis_top_k=16,
        sca_top_k=3,
        sca_adaptive_top_k=False,
        sca_stability_dense_fallback_threshold=0.0,
        attention_hybrid_enabled=True,
        attention_hybrid_layers=artifact.get("layer_selection"),
        attention_hybrid_target_rank=artifact.get("target_rank", 16),
        attention_hybrid_variance_threshold=float(artifact.get("variance_threshold", 0.90)),
        attention_hybrid_state_dim=artifact.get("state_dim"),
        attention_hybrid_force_no_cache=False,
        attention_sparse_mode=True,
    )
    model.load_hybrid_attention_state(args.checkpoint, strict=True)
    if str(args.sca_recalibrated_checkpoint).strip():
        model.load_sca_recalibration_state(str(args.sca_recalibrated_checkpoint).strip(), strict=True)
    model.set_sparse_attention_mode(strict_fully_sparse=True)
    model.disable_task_bias_injection = True
    model.strict_decode_enable_repetition_penalty = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    topk_vals = [int(v.strip()) for v in str(args.sweep_top_k).split(",") if v.strip()]
    basis_vals = [int(v.strip()) for v in str(args.sweep_basis_top_k).split(",") if v.strip()]
    setting_results: List[SettingResult] = []
    for top_k in topk_vals:
        for basis_top_k in basis_vals:
            _apply_setting(model, top_k=top_k, basis_top_k=basis_top_k, adaptive=False)
            quality = _run_strict_quality(model, tokenizer, SMOKE_PROMPTS[:3], max_new_tokens=int(args.max_new_tokens))
            drift = _dense_sparse_drift(model, tokenizer, SMOKE_PROMPTS[3], rollout_steps=int(args.rollout_steps))
            diag = model.get_sparse_mlp_diagnostics()
            agg_diag = {
                "mean_dense_fallback_rate": float(diag.get("mean_dense_fallback_rate", 0.0)),
                "mean_touched_weight_fraction": float(diag.get("mean_touched_weight_fraction", 0.0)),
                "mean_latent_support_overlap": float(diag.get("mean_latent_support_overlap", 0.0)),
                "mean_latent_support_unique_fraction": float(diag.get("mean_latent_support_unique_fraction", 0.0)),
            }
            score = _score(quality["agg"], drift, agg_diag["mean_dense_fallback_rate"])
            setting_results.append(
                SettingResult(
                    top_k=top_k,
                    basis_top_k=basis_top_k,
                    score=float(score),
                    quality=quality["agg"],
                    drift=drift,
                    diagnostics=agg_diag,
                )
            )

    best = min(setting_results, key=lambda r: r.score)
    issues: List[str] = []
    if best.quality["degenerate_frac"] > 0.2:
        issues.append("strict_decode_degeneracy_high")
    if best.quality["rep_frac_mean"] > 0.15:
        issues.append("strict_repetition_high")
    if best.drift["mean_kl"] > 3.0:
        issues.append("dense_sparse_logit_drift_high")
    if best.drift["mean_sparse_entropy"] > 8.5 and best.drift["mean_top1_margin"] < 0.02:
        issues.append("flat_off_manifold_sparse_logits")
    if best.diagnostics["mean_dense_fallback_rate"] > 0.0:
        issues.append("not_honest_sparse_runtime")

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
                "quality": r.quality,
                "drift": r.drift,
                "diagnostics": r.diagnostics,
            }
            for r in setting_results
        ],
        "recommended_runtime": {
            "strict_fully_sparse": True,
            "sca_sparse_placement": "learned_basis",
            "sca_top_k": int(best.top_k),
            "sca_basis_top_k": int(best.basis_top_k),
            "sca_adaptive_top_k": False,
            "strict_decode_enable_repetition_penalty": False,
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
