from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class BalanceAccumulator:
    latent_importance: Dict[int, torch.Tensor]
    latent_load: Dict[int, torch.Tensor]
    block_importance: Dict[int, torch.Tensor]
    block_load: Dict[int, torch.Tensor]


def char_repeat4_fraction(text: str) -> float:
    s = text.lower()
    if len(s) < 8:
        return 0.0
    total = max(len(s) - 3, 1)
    rep = 0
    for idx in range(len(s) - 7):
        if s[idx : idx + 4] == s[idx + 4 : idx + 8]:
            rep += 1
    return float(rep / total)


def text_quality_metrics(text: str) -> Dict[str, float]:
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
    repeat4 = char_repeat4_fraction(text)
    deg = float(1.0 if (max_run >= 4 or rep_frac > 0.2 or repeat4 > 0.2) else 0.0)
    return {
        "distinct1": distinct1,
        "max_run": float(max_run),
        "rep_frac": rep_frac,
        "char_repeat4_frac": repeat4,
        "degenerate": deg,
    }


def aggregate_quality(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "distinct1_mean": float(sum(r["distinct1"] for r in rows) / max(len(rows), 1)),
        "max_run_mean": float(sum(r["max_run"] for r in rows) / max(len(rows), 1)),
        "rep_frac_mean": float(sum(r["rep_frac"] for r in rows) / max(len(rows), 1)),
        "char_repeat4_frac_mean": float(sum(r["char_repeat4_frac"] for r in rows) / max(len(rows), 1)),
        "degenerate_frac": float(sum(r["degenerate"] for r in rows) / max(len(rows), 1)),
    }


def dense_top1_ranks(sparse_logits: torch.Tensor, dense_logits: torch.Tensor) -> torch.Tensor:
    if sparse_logits.shape != dense_logits.shape:
        raise ValueError("sparse_logits and dense_logits must have the same shape")
    dense_top = dense_logits.argmax(dim=-1, keepdim=True)
    sparse_ref = sparse_logits.gather(dim=-1, index=dense_top)
    return (sparse_logits > sparse_ref).sum(dim=-1) + 1


def summarize_ranks(ranks: Iterable[float]) -> Dict[str, float]:
    values = [float(v) for v in ranks]
    if not values:
        return {"median": 0.0, "p95": 0.0, "mean": 0.0}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "median": float(torch.quantile(tensor, 0.5).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "mean": float(tensor.mean().item()),
    }


def final_hidden_cosine(
    dense_hidden: torch.Tensor,
    sparse_hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if dense_hidden.shape != sparse_hidden.shape:
        raise ValueError("dense_hidden and sparse_hidden must have the same shape")
    if attention_mask is None:
        dense_last = dense_hidden[:, -1, :]
        sparse_last = sparse_hidden[:, -1, :]
    else:
        last_idx = attention_mask.sum(dim=-1).clamp_min(1).to(dtype=torch.long) - 1
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, dense_hidden.shape[-1])
        dense_last = dense_hidden.gather(1, gather_idx).squeeze(1)
        sparse_last = sparse_hidden.gather(1, gather_idx).squeeze(1)
    return F.cosine_similarity(dense_last.float(), sparse_last.float(), dim=-1, eps=1e-6)


def init_balance_accumulator() -> BalanceAccumulator:
    return BalanceAccumulator(latent_importance={}, latent_load={}, block_importance={}, block_load={})


def _accumulate_tensor_map(target: Dict[int, torch.Tensor], layer_idx: int, value: Optional[torch.Tensor]) -> None:
    if not torch.is_tensor(value) or value.numel() == 0:
        return
    tensor = value.detach().cpu().float()
    current = target.get(int(layer_idx))
    if current is None:
        target[int(layer_idx)] = tensor.clone()
        return
    if tuple(current.shape) != tuple(tensor.shape):
        raise RuntimeError(f"Balance tensor shape mismatch for layer {layer_idx}: {tuple(current.shape)} vs {tuple(tensor.shape)}")
    target[int(layer_idx)] = current + tensor


def accumulate_balance_from_model(model: Any, acc: BalanceAccumulator) -> None:
    for layer_idx, wrapper in enumerate(getattr(model, "sca_sparse_mlps", [])):
        if not hasattr(wrapper, "get_last_learned_basis_regularizer_tensors"):
            continue
        reg = wrapper.get_last_learned_basis_regularizer_tensors()
        if not isinstance(reg, dict):
            continue
        _accumulate_tensor_map(acc.latent_importance, layer_idx, reg.get("_latent_importance_probs"))
        _accumulate_tensor_map(acc.latent_load, layer_idx, reg.get("_latent_load_probs"))
        _accumulate_tensor_map(acc.block_importance, layer_idx, reg.get("_block_importance_probs"))
        _accumulate_tensor_map(acc.block_load, layer_idx, reg.get("_block_load_probs"))


def summarize_balance(acc: BalanceAccumulator) -> Dict[str, Any]:
    def _normalize_map(raw: Dict[int, torch.Tensor]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for layer_idx, tensor in raw.items():
            norm = tensor / tensor.sum().clamp_min(1e-6)
            out[str(layer_idx)] = norm.tolist()
        return out

    def _max_fraction(raw: Dict[int, torch.Tensor]) -> float:
        if not raw:
            return 0.0
        return float(max((tensor / tensor.sum().clamp_min(1e-6)).max().item() for tensor in raw.values()))

    return {
        "max_latent_importance_fraction": _max_fraction(acc.latent_importance),
        "max_latent_load_fraction": _max_fraction(acc.latent_load),
        "max_block_importance_fraction": _max_fraction(acc.block_importance),
        "max_block_load_fraction": _max_fraction(acc.block_load),
        "latent_importance_by_layer": _normalize_map(acc.latent_importance),
        "latent_load_by_layer": _normalize_map(acc.latent_load),
        "block_importance_by_layer": _normalize_map(acc.block_importance),
        "block_load_by_layer": _normalize_map(acc.block_load),
    }


def _encode_prefix(prefix: Any, tokenizer: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    if isinstance(prefix, dict):
        input_ids = prefix["input_ids"].to(device)
        attention_mask = prefix.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        else:
            attention_mask = attention_mask.to(device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    encoded = tokenizer(prefix, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def evaluate_decode_prefixes(
    model: Any,
    tokenizer: Any,
    prefixes: List[Any],
    *,
    task_id: int = 0,
    rollout_horizon: int = 16,
    use_cache: bool = True,
) -> Dict[str, Any]:
    rank_values: List[float] = []
    kl_values: List[float] = []
    cosine_values: List[float] = []
    quality_rows: List[Dict[str, float]] = []
    rows: List[Dict[str, Any]] = []
    balance = init_balance_accumulator()
    prev_enabled = bool(getattr(model, "neuroplasticity_enabled", True))

    for prefix in prefixes:
        encoded = _encode_prefix(prefix, tokenizer=tokenizer, device=model.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        with torch.no_grad():
            model.neuroplasticity_enabled = True
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(rollout_horizon),
                do_sample=False,
                use_cache=bool(use_cache),
                task_id=int(task_id),
            )
        full_mask = torch.ones_like(generated, device=model.device)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        quality = text_quality_metrics(text)
        quality_rows.append(quality)

        model.neuroplasticity_enabled = False
        with torch.no_grad():
            dense_out = model(
                input_ids=generated,
                attention_mask=full_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
                task_id=int(task_id),
            )
        model.neuroplasticity_enabled = True
        with torch.no_grad():
            sparse_out = model(
                input_ids=generated,
                attention_mask=full_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
                task_id=int(task_id),
            )
        accumulate_balance_from_model(model, balance)

        dense_shift = dense_out.logits[:, :-1, :].float()
        sparse_shift = sparse_out.logits[:, :-1, :].float()
        valid = full_mask[:, 1:].to(dtype=torch.bool)
        rank_tensor = dense_top1_ranks(sparse_shift, dense_shift)
        rank_values.extend(rank_tensor[valid].detach().cpu().view(-1).tolist())

        kl_tensor = F.kl_div(
            F.log_softmax(sparse_shift, dim=-1),
            F.softmax(dense_shift, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        if torch.any(valid):
            kl_values.append(float(kl_tensor[valid].mean().detach().cpu().item()))

        dense_hidden = dense_out.hidden_states[-1]
        sparse_hidden = sparse_out.hidden_states[-1]
        cosine = final_hidden_cosine(dense_hidden, sparse_hidden, attention_mask=full_mask)
        cosine_values.extend(cosine.detach().cpu().view(-1).tolist())

        rows.append(
            {
                "text": text,
                "quality": quality,
                "rollout_kl": float(kl_values[-1] if kl_values else 0.0),
                "final_hidden_cosine": float(cosine.mean().detach().cpu().item()),
            }
        )

    model.neuroplasticity_enabled = prev_enabled
    rank_summary = summarize_ranks(rank_values)
    return {
        "rows": rows,
        "quality": aggregate_quality(quality_rows),
        "dense_top1_rank_median": float(rank_summary["median"]),
        "dense_top1_rank_p95": float(rank_summary["p95"]),
        "dense_top1_rank_mean": float(rank_summary["mean"]),
        "rollout_kl_mean": float(sum(kl_values) / max(len(kl_values), 1)),
        "final_hidden_cosine_mean": float(sum(cosine_values) / max(len(cosine_values), 1)),
        "balance": summarize_balance(balance),
    }
