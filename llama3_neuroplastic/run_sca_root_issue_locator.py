from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
except ImportError:  # pragma: no cover
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama


PROMPTS = [
    "Write one sentence about Norway.",
    "Describe a sparse MLP routing system.",
]


def _text_quality_metrics(text: str) -> Dict[str, float]:
    toks = [t for t in text.replace("\n", " ").split(" ") if t]
    if not toks:
        return {"distinct1": 0.0, "max_run": 0.0, "rep_frac": 0.0, "degenerate": 0.0}
    distinct1 = float(len(set(toks)) / len(toks))
    run = 1
    max_run = 1
    reps = 0
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            run += 1
            reps += 1
            max_run = max(max_run, run)
        else:
            run = 1
    rep_frac = float(reps / max(len(toks) - 1, 1))
    deg = float(1.0 if max_run >= 4 or rep_frac > 0.2 else 0.0)
    return {
        "distinct1": distinct1,
        "max_run": float(max_run),
        "rep_frac": rep_frac,
        "degenerate": deg,
    }


def _agg(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "distinct1_mean": float(mean([r["distinct1"] for r in rows])) if rows else 0.0,
        "max_run_mean": float(mean([r["max_run"] for r in rows])) if rows else 0.0,
        "rep_frac_mean": float(mean([r["rep_frac"] for r in rows])) if rows else 0.0,
        "degenerate_frac": float(mean([r["degenerate"] for r in rows])) if rows else 0.0,
    }


def _gen_quality(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompts: List[str],
    *,
    max_new_tokens: int,
    do_sample: bool,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=0.8 if do_sample else 1.0,
                top_k=40,
                top_p=0.9,
                use_cache=True,
                task_id=0,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        rows.append({"prompt": prompt, "text": text, **_text_quality_metrics(text)})
    return {"rows": rows, "agg": _agg(rows)}


def _dense_sparse_hidden_geometry(model: NeuroplasticLlama, tokenizer: Any, prompts: List[str]) -> Dict[str, float]:
    cos_vals: List[float] = []
    mse_vals: List[float] = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        mask = enc["attention_mask"].to(model.device)
        model.neuroplasticity_enabled = False
        with torch.no_grad():
            dense = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, output_hidden_states=True, task_id=0)
        model.neuroplasticity_enabled = True
        with torch.no_grad():
            sparse = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, output_hidden_states=True, task_id=0)
        dh = dense.hidden_states[-1].float()
        sh = sparse.hidden_states[-1].float()
        cos = F.cosine_similarity(dh.reshape(1, -1), sh.reshape(1, -1), dim=-1).mean().item()
        mse = F.mse_loss(sh, dh).item()
        cos_vals.append(float(cos))
        mse_vals.append(float(mse))
    return {"final_hidden_cosine_mean": float(mean(cos_vals)), "final_hidden_mse_mean": float(mean(mse_vals))}


def _rollout_drift(model: NeuroplasticLlama, tokenizer: Any, prompt: str, steps: int) -> Dict[str, float]:
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    kls: List[float] = []
    ents: List[float] = []
    margins: List[float] = []
    mismatch = 0
    top1_tokens: List[str] = []
    for _ in range(int(steps)):
        model.neuroplasticity_enabled = False
        with torch.no_grad():
            d = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, task_id=0)
        model.neuroplasticity_enabled = True
        with torch.no_grad():
            s = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, task_id=0)
        dlog = d.logits[:, -1, :].float()
        slog = s.logits[:, -1, :].float()
        dprob = torch.softmax(dlog, dim=-1)
        sprob = torch.softmax(slog, dim=-1)
        kls.append(float(F.kl_div(torch.log_softmax(slog, dim=-1), dprob, reduction="batchmean").item()))
        ents.append(float((-(sprob * sprob.clamp_min(1e-8).log()).sum()).item()))
        vals, idx = torch.topk(sprob[0], 2)
        margins.append(float((vals[0] - vals[1]).item()))
        dnext = int(torch.argmax(dlog, dim=-1).item())
        snext = int(torch.argmax(slog, dim=-1).item())
        mismatch += int(dnext != snext)
        top1_tokens.append(str(snext))
        ids = torch.cat([ids, torch.tensor([[snext]], dtype=ids.dtype, device=ids.device)], dim=-1)
        mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=-1)
    max_token_run = 1
    run = 1
    for i in range(1, len(top1_tokens)):
        if top1_tokens[i] == top1_tokens[i - 1]:
            run += 1
            max_token_run = max(max_token_run, run)
        else:
            run = 1
    return {
        "mean_kl": float(mean(kls)),
        "max_kl": float(max(kls)),
        "mean_sparse_entropy": float(mean(ents)),
        "mean_top1_margin": float(mean(margins)),
        "token_mismatch_rate": float(mismatch / max(int(steps), 1)),
        "top1_token_max_run": float(max_token_run),
    }


def _routing_energy_coverage(model: NeuroplasticLlama, tokenizer: Any, prompt: str) -> Dict[str, float]:
    model.set_mlp_alignment_capture(True)
    for wrapper in model.sca_sparse_mlps:
        wrapper.set_route_capture(True)
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device)
    model.neuroplasticity_enabled = True
    with torch.no_grad():
        _ = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, task_id=0)
    ratios: List[float] = []
    for wrapper in model.sca_sparse_mlps:
        align = wrapper.get_last_alignment()
        if align is None:
            continue
        dense_out = align["dense_mlp_out"].float().reshape(-1, wrapper.num_blocks, wrapper.block_size)
        active_idx = align["active_idx"].to(dtype=torch.long)
        row_energy = dense_out.pow(2).sum(dim=-1)
        total = row_energy.sum(dim=-1).clamp_min(1e-8)
        covered = torch.zeros_like(total)
        for slot in range(active_idx.shape[1]):
            idx = active_idx[:, slot]
            valid = idx >= 0
            if not torch.any(valid):
                continue
            covered[valid] += row_energy[valid, idx[valid]]
        ratios.append(float((covered / total).mean().item()))
    return {"dense_energy_covered_by_routed_blocks_mean": float(mean(ratios)) if ratios else 0.0}


def _evaluate_setting(
    model: NeuroplasticLlama,
    tokenizer: Any,
    *,
    top_k: int,
    basis_top_k: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    model.sca_config.adaptive_top_k = False
    model.sca_config.top_k = int(top_k)
    model.sca_config.basis_top_k = int(basis_top_k)
    q = _gen_quality(model, tokenizer, PROMPTS, max_new_tokens=max_new_tokens, do_sample=False)["agg"]
    d = model.get_sparse_mlp_diagnostics()
    return {
        "top_k": int(top_k),
        "basis_top_k": int(basis_top_k),
        "quality": q,
        "diagnostics": {
            "mean_dense_fallback_rate": float(d.get("mean_dense_fallback_rate", 0.0)),
            "mean_touched_weight_fraction": float(d.get("mean_touched_weight_fraction", 0.0)),
            "mean_latent_support_overlap": float(d.get("mean_latent_support_overlap", 0.0)),
            "mean_latent_support_unique_fraction": float(d.get("mean_latent_support_unique_fraction", 0.0)),
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Locate strict repetition/junk root causes across 15 checks.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sca-recalibrated-checkpoint", type=str, required=True)
    p.add_argument("--learned-basis-init-checkpoint", type=str, default="")
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--rollout-steps", type=int, default=10)
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
    model.load_sca_recalibration_state(args.sca_recalibrated_checkpoint, strict=True)
    if str(args.learned_basis_init_checkpoint).strip():
        model.load_learned_basis_init(str(args.learned_basis_init_checkpoint).strip(), strict=True)
    model.set_sparse_attention_mode(strict_fully_sparse=True)
    model.disable_task_bias_injection = True
    model.strict_decode_enable_repetition_penalty = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    checks: Dict[str, Dict[str, Any]] = {}

    # 1
    geom = _dense_sparse_hidden_geometry(model, tokenizer, PROMPTS)
    checks["01_final_hidden_geometry_mismatch"] = {
        "status": "primary" if geom["final_hidden_cosine_mean"] < 0.95 else "contributing",
        "metrics": geom,
    }

    # 2
    model.set_mlp_alignment_capture(True)
    enc = tokenizer(PROMPTS[0], return_tensors="pt")
    with torch.no_grad():
        _ = model(input_ids=enc["input_ids"].to(model.device), attention_mask=enc["attention_mask"].to(model.device), use_cache=False, return_dict=True, task_id=0)
    loc_loss, loc_metrics = model.compute_sca_local_recalibration_loss(loss_mode="mse_plus_norm")
    checks["02_learned_basis_local_calibration_quality"] = {
        "status": "contributing" if float(loc_loss.item()) > 0.1 else "unlikely",
        "metrics": {"local_loss": float(loc_loss.item()), **{k: v for k, v in loc_metrics.items() if k in {"loss_mse", "loss_norm"}}},
    }

    # 3 + 15 sweep
    setting_results = []
    for tk in [3, 4]:
        for bk in [12, 16]:
            setting_results.append(_evaluate_setting(model, tokenizer, top_k=tk, basis_top_k=bk, max_new_tokens=int(args.max_new_tokens)))
    rep_vals = [r["quality"]["rep_frac_mean"] for r in setting_results]
    overlap_vals = [r["diagnostics"]["mean_latent_support_overlap"] for r in setting_results]
    if len(rep_vals) >= 2:
        rep_centered = [v - mean(rep_vals) for v in rep_vals]
        ov_centered = [v - mean(overlap_vals) for v in overlap_vals]
        denom = (sum(a * a for a in rep_centered) * sum(b * b for b in ov_centered)) ** 0.5
        corr = float(sum(a * b for a, b in zip(rep_centered, ov_centered)) / denom) if denom > 1e-8 else 0.0
    else:
        corr = 0.0
    checks["03_latent_budget_correctness"] = {
        "status": "primary" if min(rep_vals) > 0.1 else "contributing",
        "metrics": {"setting_results": setting_results},
    }

    # 4
    route_cov = _routing_energy_coverage(model, tokenizer, PROMPTS[1])
    checks["04_block_routing_compatibility"] = {
        "status": "contributing" if route_cov["dense_energy_covered_by_routed_blocks_mean"] < 0.35 else "unlikely",
        "metrics": route_cov,
    }

    # 5
    decode_start = max(int(len(model.sca_sparse_mlps) * 0.75), 0)
    with torch.no_grad():
        orig_scales = model.sca_layer_output_scale.clone()
    scale_quality = {}
    for sc in [0.4, 0.55, 0.75]:
        with torch.no_grad():
            model.sca_layer_output_scale[decode_start:] = float(sc)
        scale_quality[str(sc)] = _gen_quality(model, tokenizer, [PROMPTS[0]], max_new_tokens=8, do_sample=False)["agg"]
    with torch.no_grad():
        model.sca_layer_output_scale.copy_(orig_scales)
    checks["05_layer_output_scale_dynamics"] = {
        "status": "contributing" if min(v["degenerate_frac"] for v in scale_quality.values()) > 0.0 else "unlikely",
        "metrics": {"decode_scale_mean": float(orig_scales[decode_start:].mean().item()), "scale_quality": scale_quality},
    }

    # 6
    drift_short = _rollout_drift(model, tokenizer, PROMPTS[1], steps=2)
    drift_long = _rollout_drift(model, tokenizer, PROMPTS[1], steps=int(args.rollout_steps))
    checks["06_objective_horizon_mismatch"] = {
        "status": "primary" if drift_long["mean_kl"] > drift_short["mean_kl"] * 1.5 else "contributing",
        "metrics": {"short": drift_short, "long": drift_long},
    }

    # 7
    ranks: List[int] = []
    for prompt in PROMPTS:
        enc = tokenizer(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(model.device)
        mask = enc["attention_mask"].to(model.device)
        model.neuroplasticity_enabled = False
        with torch.no_grad():
            d = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, task_id=0)
        model.neuroplasticity_enabled = True
        with torch.no_grad():
            s = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True, task_id=0)
        dtop = int(torch.argmax(d.logits[:, -1, :], dim=-1).item())
        rank = int((torch.argsort(s.logits[:, -1, :], dim=-1, descending=True) == dtop).nonzero()[0, 1].item()) + 1
        ranks.append(rank)
    checks["07_logit_alignment_shape"] = {
        "status": "primary" if mean(ranks) > 50 else "contributing",
        "metrics": {"dense_top1_rank_in_sparse_mean": float(mean(ranks)), "dense_top1_rank_in_sparse": ranks},
    }

    # 8
    q_greedy = _gen_quality(model, tokenizer, PROMPTS, max_new_tokens=8, do_sample=False)["agg"]
    q_sample = _gen_quality(model, tokenizer, PROMPTS, max_new_tokens=8, do_sample=True)["agg"]
    checks["08_decode_policy_sensitivity"] = {
        "status": "contributing" if q_greedy["degenerate_frac"] > 0.0 and q_sample["degenerate_frac"] > 0.0 else "unlikely",
        "metrics": {"greedy": q_greedy, "sample": q_sample},
    }

    # 9
    parity_ok = (
        bool(model.disable_task_bias_injection)
        and not bool(model.strict_decode_enable_repetition_penalty)
        and float(model.sca_config.stability_dense_fallback_threshold) == 0.0
        and str(model.sca_config.sparse_placement) == "learned_basis"
    )
    checks["09_runtime_parity_gaps"] = {
        "status": "unlikely" if parity_ok else "contributing",
        "metrics": {
            "disable_task_bias_injection": bool(model.disable_task_bias_injection),
            "strict_decode_enable_repetition_penalty": bool(model.strict_decode_enable_repetition_penalty),
            "fallback_threshold": float(model.sca_config.stability_dense_fallback_threshold),
            "sparse_placement": str(model.sca_config.sparse_placement),
        },
    }

    # 10
    model.set_sparse_attention_mode(enabled=False, strict_fully_sparse=False)
    q_attn_off = _gen_quality(model, tokenizer, PROMPTS, max_new_tokens=8, do_sample=False)["agg"]
    model.set_sparse_attention_mode(strict_fully_sparse=True)
    q_attn_on = _gen_quality(model, tokenizer, PROMPTS, max_new_tokens=8, do_sample=False)["agg"]
    checks["10_attention_interaction_effect"] = {
        "status": "unlikely" if abs(q_attn_on["rep_frac_mean"] - q_attn_off["rep_frac_mean"]) < 0.05 else "contributing",
        "metrics": {"attention_on": q_attn_on, "attention_off": q_attn_off},
    }

    # 11
    q_dense_only = _gen_quality(model, tokenizer, PROMPTS, max_new_tokens=8, do_sample=False)["agg"]
    model.neuroplasticity_enabled = False
    q_teacher = _gen_quality(model, tokenizer, PROMPTS, max_new_tokens=8, do_sample=False)["agg"]
    model.neuroplasticity_enabled = True
    checks["11_quantization_sensitivity_proxy"] = {
        "status": "unlikely" if q_teacher["degenerate_frac"] == 0.0 and q_dense_only["degenerate_frac"] > 0.0 else "inconclusive",
        "metrics": {"sparse_path": q_dense_only, "dense_teacher_same_quantized_model": q_teacher},
    }

    # 12
    blame = {}
    original_guard = int(model.decode_guard_layers)
    for guard in [0, 6, 12, 18]:
        model.decode_guard_layers = int(guard)
        blame[str(guard)] = _gen_quality(model, tokenizer, [PROMPTS[1]], max_new_tokens=8, do_sample=False)["agg"]
    model.decode_guard_layers = original_guard
    checks["12_per_layer_blame_localization"] = {
        "status": "contributing",
        "metrics": {"decode_guard_sweep": blame},
    }

    # 13
    drift = _rollout_drift(model, tokenizer, PROMPTS[1], steps=int(args.rollout_steps))
    checks["13_token_family_attractor_diagnostics"] = {
        "status": "primary" if drift["top1_token_max_run"] >= 4 else "contributing",
        "metrics": drift,
    }

    # 14
    init_stats = {}
    if str(args.learned_basis_init_checkpoint).strip():
        blob = torch.load(str(args.learned_basis_init_checkpoint).strip(), map_location="cpu")
        if isinstance(blob, dict) and isinstance(blob.get("stats"), dict):
            init_stats = blob["stats"]
    explained_vals = [float(v.get("explained_variance_ratio", 0.0)) for v in init_stats.values()] if init_stats else []
    checks["14_basis_init_quality_by_layer"] = {
        "status": "contributing" if explained_vals and min(explained_vals) < 0.35 else "inconclusive",
        "metrics": {
            "explained_variance_min": float(min(explained_vals)) if explained_vals else None,
            "explained_variance_mean": float(mean(explained_vals)) if explained_vals else None,
            "layers_below_0_35": [k for k, v in init_stats.items() if float(v.get("explained_variance_ratio", 0.0)) < 0.35],
        },
    }

    # 15
    checks["15_support_concentration_vs_semantics"] = {
        "status": "contributing" if abs(corr) > 0.3 else "inconclusive",
        "metrics": {"corr_support_overlap_vs_rep_frac": float(corr), "setting_results": setting_results},
    }

    primary = [k for k, v in checks.items() if v["status"] == "primary"]
    contributing = [k for k, v in checks.items() if v["status"] == "contributing"]
    unlikely = [k for k, v in checks.items() if v["status"] == "unlikely"]
    inconclusive = [k for k, v in checks.items() if v["status"] == "inconclusive"]

    report = {
        "summary": {
            "primary_causes": primary,
            "contributing_causes": contributing,
            "unlikely_causes": unlikely,
            "inconclusive": inconclusive,
        },
        "checks": checks,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
