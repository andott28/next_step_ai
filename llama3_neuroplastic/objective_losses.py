from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _infer_device_dtype(
    reference: Optional[torch.Tensor],
    stats: Optional[Dict[str, Any]] = None,
) -> tuple[torch.device, torch.dtype]:
    if reference is not None:
        return reference.device, reference.dtype
    if stats is not None:
        for value in stats.values():
            if isinstance(value, torch.Tensor):
                return value.device, value.dtype
    return torch.device("cpu"), torch.float32


def _to_tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if isinstance(value, bool):
        return torch.tensor(float(value), device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype)


def compute_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits [B,T,V], got shape {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"Expected labels [B,T], got shape {tuple(labels.shape)}")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def compute_kl_loss(
    adapted_logits: torch.Tensor,
    base_logits: torch.Tensor,
    temperature: float = 1.0,
    labels: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    if adapted_logits.shape != base_logits.shape:
        raise ValueError(
            f"adapted_logits and base_logits must match, got {tuple(adapted_logits.shape)} vs {tuple(base_logits.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    t = float(temperature)
    adapted_logp = F.log_softmax(adapted_logits / t, dim=-1)
    base_p = F.softmax(base_logits / t, dim=-1)

    # Per-token KL, then mean over valid tokens to keep scale stable.
    kl_per_token = F.kl_div(adapted_logp, base_p, reduction="none").sum(dim=-1)

    if labels is not None:
        valid = (labels != ignore_index).to(dtype=kl_per_token.dtype)
        denom = valid.sum().clamp_min(1.0)
        kl = (kl_per_token * valid).sum() / denom
    else:
        kl = kl_per_token.mean()

    return kl * (t * t)


def compute_entropy_loss(adapted_logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(adapted_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return -entropy


def estimate_token_entropy(adapted_logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(adapted_logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).mean()


def compute_ppl_drift(
    adapted_lm_loss: torch.Tensor,
    base_lm_loss: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    # Compute drift in FP32 to avoid FP16 overflow in exp().
    adapted = adapted_lm_loss.float()
    base = base_lm_loss.float()
    adapted_ppl = torch.exp(torch.clamp(adapted, min=-20.0, max=20.0))
    base_ppl = torch.exp(torch.clamp(base, min=-20.0, max=20.0))
    return torch.abs(adapted_ppl - base_ppl) / torch.clamp(base_ppl, min=eps)


def compute_biological_loss(
    stats: Dict[str, Any],
    targets: Dict[str, Any],
    reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device, dtype = _infer_device_dtype(reference, stats)
    weights = targets.get("weights", {}) if isinstance(targets, dict) else {}

    def _target(name: str, default: float) -> float:
        value = targets.get(name, default)
        if value is None:
            return default
        return float(value)

    active_target = _target("active_block_ratio", 0.0)
    refractory_target = _target("refractory_violation_rate", 0.0)
    instability_target = _target("pattern_instability", 0.0)
    bio_active_target = _target("biological_active_ratio", 0.0)

    active_weight = float(weights.get("active_block_ratio", 1.0))
    refractory_weight = float(weights.get("refractory_violation_rate", 1.0))
    instability_weight = float(weights.get("pattern_instability", 1.0))
    bio_active_weight = float(weights.get("biological_active_ratio", 0.5))

    active_value = _to_tensor(stats.get("active_block_ratio", 0.0), device, dtype)
    refractory_value = _to_tensor(stats.get("refractory_violation_rate", 0.0), device, dtype)
    instability_value = _to_tensor(stats.get("pattern_instability", 0.0), device, dtype)
    bio_active_value = _to_tensor(stats.get("biological_active_ratio", 0.0), device, dtype)

    active_term = torch.abs(active_value - active_target) * active_weight
    refractory_term = F.relu(refractory_value - refractory_target) * refractory_weight
    instability_term = torch.abs(instability_value - instability_target) * instability_weight
    bio_active_term = torch.abs(bio_active_value - bio_active_target) * bio_active_weight

    bio_loss = active_term + refractory_term + instability_term + bio_active_term
    if not torch.isfinite(bio_loss):
        return torch.tensor(float(math.nan), device=device, dtype=dtype)
    return bio_loss
