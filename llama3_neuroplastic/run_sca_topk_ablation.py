from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate strict sparse MLP top-k using honest autoregressive metrics.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sca-recalibrated-checkpoint", type=str, default="")
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--top-k-values", type=str, default="3,4,6,8,12,16")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=4)
    p.add_argument("--rollout-prompts", type=int, default=3)
    p.add_argument("--alignment-prompts", type=int, default=2)
    return p.parse_args()


def _load_artifact(path: str) -> Dict[str, Any]:
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise RuntimeError("Checkpoint artifact must be a dict")
    return blob


def _parse_int_list(spec: str) -> List[int]:
    out: List[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    return out


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
    tokens = re.findall(r"\w+|[^\w\s]", text)
    if not tokens:
        return {
            "distinct1": 0.0,
            "max_run": 0.0,
            "rep_frac": 0.0,
            "punct_frac": 0.0,
            "char_repeat4_frac": 0.0,
            "degenerate": 1.0,
        }
    distinct1 = float(len(set(tokens)) / len(tokens))
    max_run = 1
    run = 1
    rep = 0
    punct = 0
    for tok in tokens:
        if re.fullmatch(r"[^\w\s]+", tok):
            punct += 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            run += 1
            rep += 1
            max_run = max(max_run, run)
        else:
            run = 1
    char_repeat = _char_repeat4_fraction(text)
    return {
        "distinct1": distinct1,
        "max_run": float(max_run),
        "rep_frac": float(rep / max(len(tokens) - 1, 1)),
        "punct_frac": float(punct / len(tokens)),
        "char_repeat4_frac": float(char_repeat),
        "degenerate": float(1.0 if (max_run >= 4 or (rep / max(len(tokens) - 1, 1)) > 0.2 or char_repeat > 0.2) else 0.0),
    }


def _aggregate_quality(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "distinct1_mean": float(sum(r["distinct1"] for r in rows) / max(len(rows), 1)),
        "max_run_mean": float(sum(r["max_run"] for r in rows) / max(len(rows), 1)),
        "rep_frac_mean": float(sum(r["rep_frac"] for r in rows) / max(len(rows), 1)),
        "punct_frac_mean": float(sum(r["punct_frac"] for r in rows) / max(len(rows), 1)),
        "char_repeat4_frac_mean": float(sum(r["char_repeat4_frac"] for r in rows) / max(len(rows), 1)),
        "degenerate_frac": float(sum(r["degenerate"] for r in rows) / max(len(rows), 1)),
    }


def _quality_gate_not_worse(candidate: Dict[str, float], baseline: Dict[str, float]) -> tuple[bool, Dict[str, float]]:
    checks = {
        "distinct1_delta": float(candidate["distinct1_mean"] - baseline["distinct1_mean"]),
        "max_run_delta": float(candidate["max_run_mean"] - baseline["max_run_mean"]),
        "rep_frac_delta": float(candidate["rep_frac_mean"] - baseline["rep_frac_mean"]),
        "punct_frac_delta": float(candidate["punct_frac_mean"] - baseline["punct_frac_mean"]),
        "char_repeat4_frac_delta": float(candidate["char_repeat4_frac_mean"] - baseline["char_repeat4_frac_mean"]),
        "degenerate_frac_delta": float(candidate["degenerate_frac"] - baseline["degenerate_frac"]),
    }
    passed = (
        checks["distinct1_delta"] >= -0.05
        and checks["max_run_delta"] <= 1.0
        and checks["rep_frac_delta"] <= 0.05
        and checks["punct_frac_delta"] <= 0.05
        and checks["char_repeat4_frac_delta"] <= 0.05
        and checks["degenerate_frac_delta"] <= 0.05
    )
    return bool(passed), checks


def _build_model(
    artifact: Dict[str, Any],
    *,
    model_name: str,
    top_k: int,
    sca_checkpoint: str,
    enable_sparse_mlp: bool,
    sca_spmm_impl: str,
) -> NeuroplasticLlama:
    model = NeuroplasticLlama(
        model_name=model_name,
        neuroplasticity_enabled=bool(enable_sparse_mlp),
        sca_use_cuda=True,
        sca_spmm_impl=str(sca_spmm_impl),
        sca_top_k=int(top_k),
        sca_adaptive_top_k=False,
        sca_stability_dense_fallback_threshold=0.0,
        sca_decode_guard_layers=12,
        attention_hybrid_enabled=True,
        attention_hybrid_layers=artifact.get("layer_selection"),
        attention_hybrid_target_rank=artifact.get("target_rank", 16),
        attention_hybrid_variance_threshold=float(artifact.get("variance_threshold", 0.90)),
        attention_hybrid_state_dim=artifact.get("state_dim"),
        attention_hybrid_force_no_cache=True,
        attention_sparse_mode=False,
        strict_decode_enable_repetition_penalty=False,
    )
    model.load_hybrid_attention_state(args.checkpoint, strict=False)
    if sca_checkpoint:
        model.load_sca_recalibration_state(sca_checkpoint, strict=True)
    model.disable_task_bias_injection = True if enable_sparse_mlp else bool(model.disable_task_bias_injection)
    model.eval()
    return model


def _generate_texts(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    last_diag: Dict[str, Any] = {}
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_k=20,
                top_p=0.9,
                use_cache=False,
                task_id=0,
            )
        elapsed = float(time.perf_counter() - t0)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        quality = _text_quality_metrics(text)
        rows.append({"prompt": prompt, "text": text, "latency_s": elapsed, **quality})
        last_diag = model.get_sparse_mlp_diagnostics()
    return rows, last_diag


def _measure_rollout_kl(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompts: List[str],
    rollout_steps: int,
) -> Dict[str, Any]:
    kl_values: List[float] = []
    token_mismatch = 0
    total_steps = 0
    prompt_rows: List[Dict[str, Any]] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        cur_ids = encoded["input_ids"]
        cur_mask = encoded["attention_mask"]
        per_prompt: Dict[str, Any] = {"prompt": prompt, "steps": []}
        for _ in range(int(rollout_steps)):
            prev_enabled = bool(model.neuroplasticity_enabled)
            with torch.no_grad():
                model.neuroplasticity_enabled = False
                dense_out = model(
                    input_ids=cur_ids.to(model.device),
                    attention_mask=cur_mask.to(model.device),
                    use_cache=False,
                    return_dict=True,
                    task_id=0,
                )
                model.neuroplasticity_enabled = True
                sparse_out = model(
                    input_ids=cur_ids.to(model.device),
                    attention_mask=cur_mask.to(model.device),
                    use_cache=False,
                    return_dict=True,
                    task_id=0,
                )
                model.neuroplasticity_enabled = prev_enabled
            dense_logits = dense_out.logits[:, -1, :].float().cpu()
            sparse_logits = sparse_out.logits[:, -1, :].float().cpu()
            dense_p = F.softmax(dense_logits, dim=-1)
            sparse_logp = F.log_softmax(sparse_logits, dim=-1)
            kl = F.kl_div(sparse_logp, dense_p, reduction="batchmean").item()
            dense_next = torch.argmax(dense_logits, dim=-1)
            sparse_next = torch.argmax(sparse_logits, dim=-1)
            token_mismatch += int((dense_next != sparse_next).sum().item())
            total_steps += int(dense_next.numel())
            kl_values.append(float(kl))
            per_prompt["steps"].append(
                {
                    "kl": float(kl),
                    "dense_next_text": tokenizer.decode(dense_next),
                    "sparse_next_text": tokenizer.decode(sparse_next),
                }
            )
            sparse_next_cpu = sparse_next.detach().cpu()
            cur_ids = torch.cat([cur_ids, sparse_next_cpu.unsqueeze(-1)], dim=-1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones((cur_mask.shape[0], 1), device=cur_mask.device, dtype=cur_mask.dtype)],
                dim=-1,
            )
        prompt_rows.append(per_prompt)
    return {
        "mean_rollout_kl": float(sum(kl_values) / max(len(kl_values), 1)),
        "max_rollout_kl": float(max(kl_values) if kl_values else 0.0),
        "token_mismatch_rate": float(token_mismatch / max(total_steps, 1)),
        "rows": prompt_rows,
    }


def _measure_mlp_alignment(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompts: List[str],
) -> Dict[str, Any]:
    selected = [idx for idx in range(len(model.model.model.layers)) if bool(model._sparse_layer_enabled(idx))]
    model.prepare_sca_local_recalibration(
        include_task_embedding=False,
        include_spatial_proj=False,
        include_adapter_scale=False,
        include_layer_output_scale=False,
        layer_indices=selected,
        recalibration_mode="local_mlp_geometry",
        hybrid_checkpoint_path="",
    )
    norm_ratios: List[float] = []
    delta_ratios: List[float] = []
    per_layer: Dict[str, List[float]] = {}
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        prev_enabled = bool(model.neuroplasticity_enabled)
        model.neuroplasticity_enabled = True
        with torch.no_grad():
            model(
                input_ids=encoded["input_ids"].to(model.device),
                attention_mask=encoded["attention_mask"].to(model.device),
                use_cache=False,
                return_dict=True,
                task_id=0,
            )
        model.neuroplasticity_enabled = prev_enabled
        for idx, wrapper in enumerate(model.sca_sparse_mlps):
            if idx not in selected:
                continue
            alignment = wrapper.get_last_alignment()
            if alignment is None:
                continue
            dense = alignment["dense_mlp_out"].float()
            sparse = alignment["sparse_mlp_out"].float()
            dense_norm = dense.norm(dim=-1).mean().clamp_min(1e-6)
            sparse_norm = sparse.norm(dim=-1).mean()
            norm_ratio = float((sparse_norm / dense_norm).item())
            delta_ratio = float(((sparse - dense).pow(2).mean() / dense.pow(2).mean().clamp_min(1e-6)).item())
            norm_ratios.append(norm_ratio)
            delta_ratios.append(delta_ratio)
            per_layer.setdefault(str(idx), []).append(norm_ratio)
    return {
        "mean_norm_ratio": float(sum(norm_ratios) / max(len(norm_ratios), 1)),
        "mean_delta_norm_ratio": float(sum(delta_ratios) / max(len(delta_ratios), 1)),
        "per_layer_norm_ratio": {k: float(sum(v) / max(len(v), 1)) for k, v in per_layer.items()},
    }


def main() -> None:
    global args
    args = _parse_args()
    artifact = _load_artifact(args.checkpoint)
    model_name = str(artifact.get("model_name", "unsloth/Meta-Llama-3.1-8B-bnb-4bit"))
    top_k_values = _parse_int_list(args.top_k_values)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sweep_rows: List[Dict[str, Any]] = []
    for top_k in top_k_values:
        model = _build_model(
            artifact,
            model_name=model_name,
            top_k=int(top_k),
            sca_checkpoint=str(args.sca_recalibrated_checkpoint or ""),
            enable_sparse_mlp=True,
            sca_spmm_impl="dense",
        )
        prev_enabled = bool(model.neuroplasticity_enabled)
        model.neuroplasticity_enabled = False
        dense_rows, _ = _generate_texts(model, tokenizer, SMOKE_PROMPTS, int(args.max_new_tokens))
        dense_quality = _aggregate_quality(dense_rows)
        model.neuroplasticity_enabled = True
        alignment = _measure_mlp_alignment(model, tokenizer, SMOKE_PROMPTS[: int(args.alignment_prompts)])
        rollout = _measure_rollout_kl(
            model=model,
            tokenizer=tokenizer,
            prompts=SMOKE_PROMPTS[: int(args.rollout_prompts)],
            rollout_steps=int(args.rollout_steps),
        )
        quality_rows, sparse_diag = _generate_texts(model, tokenizer, SMOKE_PROMPTS, int(args.max_new_tokens))
        model.neuroplasticity_enabled = prev_enabled
        quality = _aggregate_quality(quality_rows)
        gate_ok, quality_delta = _quality_gate_not_worse(quality, dense_quality)
        fallback_rate = float(sparse_diag.get("mean_dense_fallback_rate", 0.0))
        pass_all = bool(
            fallback_rate == 0.0
            and gate_ok
            and alignment["mean_norm_ratio"] >= 0.1
            and rollout["mean_rollout_kl"] <= 20.0
        )
        sweep_rows.append(
            {
                "top_k": int(top_k),
                "fallback_rate": fallback_rate,
                "mean_touched_weight_fraction": float(sparse_diag.get("mean_touched_weight_fraction", 0.0)),
                "estimated_bytes_fetched_per_token": float(sparse_diag.get("estimated_bytes_fetched_per_token", 0.0)),
                "alignment": alignment,
                "rollout": {
                    "mean_rollout_kl": rollout["mean_rollout_kl"],
                    "max_rollout_kl": rollout["max_rollout_kl"],
                    "token_mismatch_rate": rollout["token_mismatch_rate"],
                },
                "quality": quality,
                "quality_gate_passed": bool(gate_ok),
                "quality_delta": quality_delta,
                "passes_all_gates": pass_all,
                "sample_texts": [{"prompt": row["prompt"], "text": row["text"]} for row in quality_rows],
            }
        )

    passing = [row for row in sweep_rows if bool(row["passes_all_gates"])]
    result = {
        "checkpoint": args.checkpoint,
        "sca_recalibrated_checkpoint": str(args.sca_recalibrated_checkpoint or ""),
        "ablation_spmm_impl": "dense",
        "top_k_values": top_k_values,
        "rows": sweep_rows,
        "smallest_passing_top_k": int(min(row["top_k"] for row in passing)) if passing else None,
        "passing_top_ks": [int(row["top_k"]) for row in passing],
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
