from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class KVSparseConfig:
    hidden_size: int = 16384
    kv_hidden: int = 1024
    block_size: int = 32
    top_k: int = 51
    basis_rank: int = 32
    basis_top_k: int = 8
    adaptive_top_k: bool = False
    adaptive_top_k_min: int = 20
    adaptive_top_k_max: int = 80
    adaptive_top_k_min_score_ratio: float = 0.10
    use_block_banking: bool = True
    block_banking_ema_decay: float = 0.95
    block_banking_low_usage_threshold: float = 0.001
    block_banking_vote_threshold: int = 3
    block_banking_cooldown_steps: int = 64
    separate_k_v_routing: bool = False

    @property
    def num_col_blocks(self) -> int:
        return self.hidden_size // self.block_size


def route_kv_blocks(
    hidden: torch.Tensor,
    routing: dict,
    *,
    banked_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route K/V column blocks for the given hidden states.

    Args:
        hidden: [N, hidden_size] post-layernorm hidden states
        routing: dict with keys:
            enc_w  [basis_rank, hidden_size]
            enc_b  [basis_rank]
            dec_norm_t  [basis_rank, num_col_blocks]
            top_k  int
        banked_mask: optional bool [num_col_blocks] — banked blocks to exclude

    Returns:
        [N, top_k] long tensor of active column-block indices
    """
    enc_w = routing["enc_w"]
    enc_b = routing["enc_b"]
    dec_norm_t = routing["dec_norm_t"]
    top_k = int(routing["top_k"])


    N = int(hidden.shape[0]) if hidden.ndim == 2 else int(hidden.shape[0] * hidden.shape[1])
    h = hidden.reshape(N, -1).to(device=enc_w.device, dtype=enc_w.dtype)


    latent = F.silu(F.linear(h, enc_w, enc_b))

    scores = torch.matmul(latent.abs(), dec_norm_t)

    if banked_mask is not None:

        scores[:, banked_mask.to(device=scores.device)] = float("-inf")

    top_k = max(1, min(top_k, int(scores.shape[-1])))
    return scores.topk(top_k, dim=-1).indices


def update_kv_block_banking(
    active_blocks: torch.Tensor,
    usage_ema: torch.Tensor,
    low_usage_votes: torch.Tensor,
    banked_until: torch.Tensor,
    step: int,
    *,
    ema_decay: float,
    low_threshold: float,
    vote_threshold: int,
    cooldown_steps: int,
) -> None:
    """Update block banking state in-place.

    Args:
        active_blocks: [N, top_k] long tensor of active block indices
        usage_ema: [num_col_blocks] float32, EMA of per-block usage — updated in-place
        low_usage_votes: [num_col_blocks] int32, vote counter — updated in-place
        banked_until: [num_col_blocks] int32, step until which block is banked — updated in-place
        step: current global step counter
        ema_decay: EMA decay factor (e.g. 0.95)
        low_threshold: EMA below this → low-usage vote
        vote_threshold: consecutive votes needed to bank a block
        cooldown_steps: how many steps to bank a low-usage block
    """
    num_blocks = int(usage_ema.numel())


    used = active_blocks.reshape(-1).cpu()
    used = used[(used >= 0) & (used < num_blocks)]

    current_usage = torch.zeros(num_blocks, dtype=torch.float32)
    if int(used.numel()) > 0:
        current_usage.scatter_add_(
            0,
            used.long(),
            torch.ones(int(used.numel()), dtype=torch.float32),
        )
        current_usage = (current_usage > 0).float()


    usage_ema.mul_(ema_decay)
    usage_ema.add_(current_usage * (1.0 - ema_decay))


    low_mask = usage_ema < low_threshold
    in_cooldown = banked_until > step
    vote_mask = low_mask & ~in_cooldown

    low_usage_votes[vote_mask] += 1
    low_usage_votes[~vote_mask] = 0

    can_bank = low_usage_votes >= vote_threshold
    if can_bank.any():
        banked_until[can_bank] = step + cooldown_steps
        low_usage_votes[can_bank] = 0


def get_kv_banked_mask(banked_until: torch.Tensor, step: int) -> torch.Tensor:
    """Return bool [num_col_blocks]: True where block is currently banked."""
    return banked_until > step
