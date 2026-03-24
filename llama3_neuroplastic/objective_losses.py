from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _resolve_token_mask(
    *,
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    label_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    mask = None
    if attention_mask is not None:
        mask = attention_mask.to(device=logits.device, dtype=torch.bool)
    if label_mask is not None:
        current = label_mask.to(device=logits.device, dtype=torch.bool)
        mask = current if mask is None else (mask & current)
    return mask


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    if values.shape != mask.shape:
        raise ValueError("mask shape must match values shape")
    valid = mask.to(device=values.device, dtype=torch.bool)
    if not torch.any(valid):
        return values.new_zeros(())
    return values.masked_select(valid).mean()


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, seq, vocab]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [batch, seq]")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1).to(device=logits.device),
        ignore_index=int(ignore_index),
    )


def compute_kl_loss(
    adapted_logits: torch.Tensor,
    base_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
    label_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if adapted_logits.shape != base_logits.shape:
        raise ValueError("adapted_logits and base_logits must have the same shape")
    temp = max(float(temperature), 1e-5)
    adapted = adapted_logits.float() / temp
    base = base_logits.float() / temp
    token_kl = F.kl_div(
        F.log_softmax(adapted, dim=-1),
        F.softmax(base, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    mask = _resolve_token_mask(logits=adapted_logits, attention_mask=attention_mask, label_mask=label_mask)
    return _masked_mean(token_kl, mask) * (temp * temp)


def compute_token_entropy(
    logits: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    mask = _resolve_token_mask(logits=logits, attention_mask=attention_mask)
    return _masked_mean(entropy, mask)


def compute_entropy_loss(
    logits: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return -compute_token_entropy(logits, attention_mask=attention_mask)


def compute_ppl_drift(
    adapted_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    adapted_ce = compute_lm_loss(adapted_logits.float(), labels, ignore_index=ignore_index).float()
    base_ce = compute_lm_loss(base_logits.float(), labels, ignore_index=ignore_index).float()
    ce_delta = (adapted_ce - base_ce).clamp(min=-20.0, max=20.0)
    return torch.expm1(ce_delta).abs()


def compute_biological_loss(stats: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
    weights = dict(targets.get("weights", {}))
    total = torch.tensor(0.0, dtype=torch.float32)
    used = 0
    for key, target in targets.items():
        if key == "weights" or key not in stats:
            continue
        value = torch.tensor(float(stats[key]), dtype=torch.float32)
        reference = torch.tensor(float(target), dtype=torch.float32)
        weight = float(weights.get(key, 1.0))
        total = total + (weight * (value - reference).pow(2))
        used += 1
    if used == 0:
        return total
    return total / float(used)
