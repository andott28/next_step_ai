from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .sca_sparse_config import SCASparseConfig
    from .triton_sca_gate import triton_compute_active_blocks_topk, triton_sca_gate_available
except ImportError:
    from sca_sparse_config import SCASparseConfig
    from triton_sca_gate import triton_compute_active_blocks_topk, triton_sca_gate_available


def compute_active_blocks_torch(
    query: torch.Tensor,
    block_centers: torch.Tensor,
    config: SCASparseConfig,
    refractory_mask: torch.Tensor | None = None,
    inhibition_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    query: [N, 3]
    block_centers: [num_blocks, 3]
    returns active_idx [N, K], active_score [N, K]
    """
    route_k = int(config.route_top_k)
    if query.numel() == 0:
        empty_idx = torch.empty((0, route_k), device=query.device, dtype=torch.long)
        empty_score = torch.empty((0, route_k), device=query.device, dtype=torch.float32)
        return empty_idx, empty_score

    q = query.float()
    centers = block_centers.float()

    d2 = torch.sum((q[:, None, :] - centers[None, :, :]) ** 2, dim=-1)
    scores = torch.exp(-d2 / (2.0 * config.sigma * config.sigma))

    if refractory_mask is not None:
        scores = scores.masked_fill(refractory_mask.unsqueeze(0), float("-inf"))

    if config.inhibition_lambda > 0.0 and inhibition_matrix is not None:
        inhibition = torch.matmul(scores, inhibition_matrix.to(dtype=scores.dtype))
        scores = scores - (config.inhibition_lambda * inhibition)


    index_bias = (
        torch.arange(scores.shape[1], device=scores.device, dtype=scores.dtype).unsqueeze(0) * 1e-6
    )
    scores = scores - index_bias

    active_score, active_idx = torch.topk(scores, k=route_k, dim=-1, largest=True, sorted=True)

    invalid = torch.isinf(active_score) & (active_score < 0)
    active_idx = active_idx.masked_fill(invalid, -1)
    active_score = active_score.masked_fill(invalid, 0.0)
    active_idx, active_score = _apply_adaptive_top_k(active_idx=active_idx, active_score=active_score, config=config)
    return active_idx.long(), active_score.float()


def _apply_adaptive_top_k(
    active_idx: torch.Tensor,
    active_score: torch.Tensor,
    config: SCASparseConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(config.adaptive_top_k):
        return active_idx, active_score
    if active_idx.numel() == 0:
        return active_idx, active_score

    valid = active_idx >= 0
    if not torch.any(valid):
        return active_idx, active_score

    keep = valid.clone()
    best = active_score[:, :1].clamp_min(1e-8)
    relative = active_score / best
    keep = keep & (relative >= float(config.adaptive_top_k_min_score_ratio))

    min_k = int(min(config.adaptive_top_k_min, active_idx.shape[1]))
    if min_k > 0:
        keep[:, :min_k] = valid[:, :min_k]

    pruned_idx = active_idx.masked_fill(~keep, -1)
    pruned_score = active_score.masked_fill(~keep, 0.0)
    return pruned_idx, pruned_score


class SCABlockSparseAdapter(nn.Module):
    def __init__(self, config: SCASparseConfig) -> None:
        super().__init__()
        self.config = config
        self.num_blocks = config.num_blocks
        self.block_size = config.block_size
        self.block_rank = config.block_rank
        self.hidden_size = config.hidden_size

        self.down_w = nn.Parameter(
            torch.empty(self.num_blocks, self.block_size, self.block_rank)
        )
        self.down_b = nn.Parameter(torch.zeros(self.num_blocks, self.block_rank))
        self.up_w = nn.Parameter(
            torch.empty(self.num_blocks, self.block_rank, self.block_size)
        )
        self.up_b = nn.Parameter(torch.zeros(self.num_blocks, self.block_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down_w)
        nn.init.zeros_(self.down_b)
        nn.init.xavier_uniform_(self.up_w, gain=0.1)
        nn.init.zeros_(self.up_b)

    def forward_sparse(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        active_score: torch.Tensor | None = None,
        score_weights: torch.Tensor | None = None,
        use_cuda_kernel: bool = False,
        cuda_kernels=None,
    ) -> torch.Tensor:
        if use_cuda_kernel and cuda_kernels is not None and hidden_states.is_cuda:
            return self._forward_sparse_cuda(
                hidden_states,
                active_idx,
                active_score,
                score_weights,
                cuda_kernels,
            )
        return self._forward_sparse_torch(hidden_states, active_idx, active_score, score_weights)

    def _forward_sparse_cuda(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        active_score: torch.Tensor | None,
        score_weights: torch.Tensor | None,
        cuda_kernels,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, self.hidden_size).contiguous()


        hidden_fp16 = flat_hidden.to(dtype=torch.float16)
        delta = cuda_kernels.sparse_adapter(
            hidden_fp16,
            active_idx.contiguous(),
            self.down_w.contiguous(),
            self.down_b.contiguous(),
            self.up_w.contiguous(),
            self.up_b.contiguous(),
        )
        if score_weights is None and active_score is not None:



            valid_mask = active_idx >= 0
            raw = active_score.to(device=delta.device, dtype=delta.dtype)
            neg_inf = torch.full_like(raw, float("-inf"))
            masked = torch.where(valid_mask, raw, neg_inf)
            score_weights = torch.softmax(masked, dim=-1)
            score_weights = torch.where(valid_mask, score_weights, torch.zeros_like(score_weights))
        if score_weights is not None:
            n_rows = flat_hidden.shape[0]
            delta_blocks = delta.view(n_rows, self.num_blocks, self.block_size)
            row_idx = torch.arange(n_rows, device=delta.device)
            scaled = torch.zeros_like(delta_blocks)
            for slot in range(self.config.top_k):
                block_idx = active_idx[:, slot]
                valid = block_idx >= 0
                if not torch.any(valid):
                    continue
                rows = row_idx[valid]
                blocks = block_idx[valid]
                sc = score_weights[valid, slot].to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
                scaled[rows, blocks] += delta_blocks[rows, blocks] * sc
            delta = scaled.view_as(delta)
        delta = delta.to(dtype=hidden_states.dtype)
        return delta.view_as(hidden_states)

    def _forward_sparse_torch(
        self,
        hidden_states: torch.Tensor,
        active_idx: torch.Tensor,
        active_score: torch.Tensor | None = None,
        score_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, hidden = hidden_states.shape
        n_rows = batch * seq_len

        flat_hidden = hidden_states.reshape(n_rows, hidden)
        hidden_blocks = flat_hidden.view(n_rows, self.num_blocks, self.block_size)

        delta_blocks = torch.zeros_like(hidden_blocks)
        row_idx = torch.arange(n_rows, device=hidden_states.device)

        if score_weights is None and active_score is not None:
            valid_mask = active_idx >= 0
            neg_inf = torch.full_like(active_score, float("-inf"))
            masked = torch.where(valid_mask, active_score, neg_inf)
            score_weights = torch.softmax(masked, dim=-1)
            score_weights = torch.where(valid_mask, score_weights, torch.zeros_like(score_weights))

        for slot in range(self.config.top_k):
            block_idx = active_idx[:, slot]
            valid = block_idx >= 0
            if not torch.any(valid):
                continue

            valid_rows = row_idx[valid]
            valid_blocks = block_idx[valid]

            x = hidden_blocks[valid_rows, valid_blocks]
            down_w = self.down_w[valid_blocks]
            down_b = self.down_b[valid_blocks]
            up_w = self.up_w[valid_blocks]
            up_b = self.up_b[valid_blocks]

            rank = torch.bmm(x.unsqueeze(1), down_w).squeeze(1) + down_b
            rank = F.silu(rank)
            out = torch.bmm(rank.unsqueeze(1), up_w).squeeze(1) + up_b

            if score_weights is not None:
                score = score_weights[valid_rows, slot].to(dtype=out.dtype).unsqueeze(-1)
                out = out * score

            delta_blocks[valid_rows, valid_blocks] += out

        delta = delta_blocks.view(batch, seq_len, hidden)
        return delta

    def dense_adapter_flops_per_token(self) -> float:

        return float(4 * self.hidden_size * self.block_rank)

    def sparse_adapter_flops_per_token(self, mean_active_blocks: float) -> float:
        active_ratio = mean_active_blocks / float(self.num_blocks)
        return self.dense_adapter_flops_per_token() * active_ratio


def compute_active_blocks(
    query: torch.Tensor,
    block_centers: torch.Tensor,
    config: SCASparseConfig,
    refractory_until: torch.Tensor | None,
    step: int,
    decode_mode: bool,
    inhibition_matrix: torch.Tensor | None,
    use_cuda_kernel: bool,
    cuda_kernels,
) -> tuple[torch.Tensor, torch.Tensor]:
    refractory_mask = None
    if decode_mode and refractory_until is not None:
        refractory_mask = refractory_until > step


    if (
        use_cuda_kernel
        and query.is_cuda
        and triton_sca_gate_available()
        and float(config.inhibition_lambda) <= 0.0
        and (not torch.is_grad_enabled())
    ):
        try:
            idx, score = triton_compute_active_blocks_topk(
                query=query,
                block_centers=block_centers,
                sigma=float(config.sigma),
                top_k=int(config.route_top_k),
                refractory_until=(refractory_until if decode_mode and refractory_until is not None else None),
                step=(int(step) if decode_mode else -1),
            )
            idx, score = _apply_adaptive_top_k(active_idx=idx.long(), active_score=score.float(), config=config)
            return idx.long(), score.float()
        except Exception:

            pass

    if use_cuda_kernel and cuda_kernels is not None and query.is_cuda:
        refractory_vec = refractory_until if decode_mode and refractory_until is not None else torch.zeros(
            block_centers.shape[0],
            dtype=torch.int32,
            device=query.device,
        )
        idx, score = cuda_kernels.spatial_gate(
            query.float().contiguous(),
            block_centers.float().contiguous(),
            refractory_vec.int().contiguous(),
            int(step if decode_mode else -1),
            float(config.sigma),
            int(config.route_top_k),
        )
        idx, score = _apply_adaptive_top_k(active_idx=idx.long(), active_score=score.float(), config=config)
        return idx.long(), score.float()

    return compute_active_blocks_torch(
        query=query,
        block_centers=block_centers,
        config=config,
        refractory_mask=refractory_mask,
        inhibition_matrix=inhibition_matrix,
    )
