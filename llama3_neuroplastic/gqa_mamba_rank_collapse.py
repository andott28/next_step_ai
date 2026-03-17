from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GQALayout:
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def hidden_size(self) -> int:
        return int(self.num_attention_heads * self.head_dim)

    @property
    def query_heads_per_group(self) -> int:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"(got {self.num_attention_heads} and {self.num_key_value_heads})"
            )
        return int(self.num_attention_heads // self.num_key_value_heads)

    def validate(self) -> None:
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0 or self.head_dim <= 0:
            raise ValueError("All GQA layout dimensions must be positive")
        _ = self.query_heads_per_group


@dataclass
class SVDGroupCollapseResult:
    group_idx: int
    rank: int
    explained_variance: float
    basis_vh: torch.Tensor
    q_factor: torch.Tensor
    k_factor: torch.Tensor
    v_factor: torch.Tensor
    o_factor_t: torch.Tensor

    def reconstruct_group_weights(self) -> Dict[str, torch.Tensor]:
        w_q = self.q_factor @ self.basis_vh
        w_k = self.k_factor @ self.basis_vh
        w_v = self.v_factor @ self.basis_vh
        w_o = (self.o_factor_t @ self.basis_vh).t()
        return {"w_q": w_q, "w_k": w_k, "w_v": w_v, "w_o": w_o}


def _choose_rank(singular_values: torch.Tensor, target_rank: Optional[int], variance_threshold: float) -> int:
    if singular_values.numel() == 0:
        return 0
    if target_rank is not None:
        return max(1, min(int(target_rank), int(singular_values.numel())))
    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1]")
    energy = singular_values.square()
    ratio = torch.cumsum(energy, dim=0) / torch.clamp(energy.sum(), min=1e-12)
    rank = int(torch.searchsorted(ratio, torch.tensor(float(variance_threshold), device=ratio.device)).item() + 1)
    return max(1, min(rank, int(singular_values.numel())))


def collapse_gqa_group_with_svd(
    w_q_group: torch.Tensor,
    w_k_group: torch.Tensor,
    w_v_group: torch.Tensor,
    w_o_group: torch.Tensor,
    group_idx: int,
    target_rank: Optional[int] = None,
    variance_threshold: float = 0.90,
) -> SVDGroupCollapseResult:
    if w_q_group.ndim != 2 or w_k_group.ndim != 2 or w_v_group.ndim != 2 or w_o_group.ndim != 2:
        raise ValueError("Expected 2D matrices for all grouped projections")
    if w_q_group.shape[1] != w_k_group.shape[1] or w_q_group.shape[1] != w_v_group.shape[1]:
        raise ValueError("Grouped q/k/v matrices must share the same input dimension")
    if w_o_group.shape[0] != w_q_group.shape[1]:
        raise ValueError("w_o_group must have hidden_size rows matching q/k/v input dimension")
    if w_o_group.shape[1] != w_q_group.shape[0]:
        raise ValueError("w_o_group columns must match grouped q output rows")

    block = torch.cat([w_q_group, w_k_group, w_v_group, w_o_group.t()], dim=0)
    u, s, vh = torch.linalg.svd(block, full_matrices=False)
    rank = _choose_rank(s, target_rank=target_rank, variance_threshold=variance_threshold)
    if rank == 0:
        raise ValueError("SVD produced zero rank; expected non-empty grouped weights")

    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]

    left = u_r * s_r.unsqueeze(0)
    q_rows = w_q_group.shape[0]
    k_rows = w_k_group.shape[0]
    v_rows = w_v_group.shape[0]
    o_rows_t = w_o_group.shape[1]
    q_factor, k_factor, v_factor, o_factor_t = torch.split(left, [q_rows, k_rows, v_rows, o_rows_t], dim=0)

    explained = float(s_r.square().sum().item() / max(float(s.square().sum().item()), 1e-12))
    return SVDGroupCollapseResult(
        group_idx=int(group_idx),
        rank=int(rank),
        explained_variance=explained,
        basis_vh=vh_r,
        q_factor=q_factor,
        k_factor=k_factor,
        v_factor=v_factor,
        o_factor_t=o_factor_t,
    )


def _group_slices(layout: GQALayout, group_idx: int) -> Tuple[slice, slice]:
    layout.validate()
    if group_idx < 0 or group_idx >= layout.num_key_value_heads:
        raise IndexError(f"group_idx out of range: {group_idx}")
    q_group_width = layout.query_heads_per_group * layout.head_dim
    q_start = group_idx * q_group_width
    q_stop = q_start + q_group_width
    kv_start = group_idx * layout.head_dim
    kv_stop = kv_start + layout.head_dim
    return slice(q_start, q_stop), slice(kv_start, kv_stop)


def collapse_llama_gqa_attention_layer(
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    layout: GQALayout,
    target_rank: Optional[int] = None,
    variance_threshold: float = 0.90,
) -> List[SVDGroupCollapseResult]:
    layout.validate()
    if w_q.shape != (layout.hidden_size, layout.hidden_size):
        raise ValueError(f"Unexpected w_q shape {tuple(w_q.shape)} for hidden_size={layout.hidden_size}")
    if w_o.shape != (layout.hidden_size, layout.hidden_size):
        raise ValueError(f"Unexpected w_o shape {tuple(w_o.shape)} for hidden_size={layout.hidden_size}")
    expected_kv_shape = (layout.num_key_value_heads * layout.head_dim, layout.hidden_size)
    if w_k.shape != expected_kv_shape or w_v.shape != expected_kv_shape:
        raise ValueError(
            f"Unexpected w_k/w_v shape {tuple(w_k.shape)} / {tuple(w_v.shape)}, expected {expected_kv_shape}"
        )

    out: List[SVDGroupCollapseResult] = []
    for g in range(layout.num_key_value_heads):
        q_slice, kv_slice = _group_slices(layout, g)
        out.append(
            collapse_gqa_group_with_svd(
                w_q_group=w_q[q_slice, :],
                w_k_group=w_k[kv_slice, :],
                w_v_group=w_v[kv_slice, :],
                w_o_group=w_o[:, q_slice],
                group_idx=g,
                target_rank=target_rank,
                variance_threshold=variance_threshold,
            )
        )
    return out


@dataclass
class GroupMambaProjection:
    in_basis: torch.Tensor
    q_select: torch.Tensor
    k_select: torch.Tensor
    v_input: torch.Tensor
    out_proj: torch.Tensor

    @property
    def rank(self) -> int:
        return int(self.in_basis.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.in_basis.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.q_select.shape[0])


def _match_rows(matrix: torch.Tensor, rows: int) -> torch.Tensor:
    if matrix.shape[0] == rows:
        return matrix
    if matrix.shape[0] > rows:
        return matrix[:rows, :]
    pad = torch.zeros((rows - matrix.shape[0], matrix.shape[1]), dtype=matrix.dtype, device=matrix.device)
    return torch.cat([matrix, pad], dim=0)


def transplant_svd_group_to_mamba_projection(
    result: SVDGroupCollapseResult,
    query_heads_per_group: int,
    head_dim: int,
    state_dim: Optional[int] = None,
) -> GroupMambaProjection:
    rank = int(result.rank)
    q = result.q_factor.reshape(query_heads_per_group, head_dim, rank).mean(dim=0)
    k = result.k_factor
    v = result.v_factor
    o = result.o_factor_t.reshape(query_heads_per_group, head_dim, rank).mean(dim=0)

    d_state = int(state_dim) if state_dim is not None else int(head_dim)
    q_s = _match_rows(q, d_state)
    k_s = _match_rows(k, d_state)
    v_s = _match_rows(v, d_state)
    o_s = _match_rows(o, d_state)

    out_proj = result.basis_vh.t() @ o_s.t()
    return GroupMambaProjection(
        in_basis=result.basis_vh,
        q_select=q_s,
        k_select=k_s,
        v_input=v_s,
        out_proj=out_proj,
    )


class _MambaState(nn.Module):
    def __init__(self, state_dim: int, neutral_a: float = 0.999, neutral_delta: float = 1.0) -> None:
        super().__init__()
        if not 0.0 < neutral_a < 1.0:
            raise ValueError("neutral_a must be in (0, 1)")
        if neutral_delta <= 0:
            raise ValueError("neutral_delta must be > 0")
        a_init = torch.logit(torch.tensor(neutral_a, dtype=torch.float32))
        d_init = torch.log(torch.exp(torch.tensor(neutral_delta, dtype=torch.float32)) - 1.0)
        self.a_logit = nn.Parameter(torch.full((state_dim,), a_init))
        self.delta_log = nn.Parameter(torch.full((state_dim,), d_init))

    def step(
        self,
        prev: torch.Tensor,
        b_t: torch.Tensor,
        v_t: torch.Tensor,
        c_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.sigmoid(self.a_logit).view(1, -1)
        delta = F.softplus(self.delta_log).view(1, -1)
        new_state = a * prev + delta * (b_t * v_t)
        y = c_t * new_state
        return new_state, y


class _GroupMambaRuntime(nn.Module):
    def __init__(self, projection: GroupMambaProjection) -> None:
        super().__init__()
        self.in_basis = nn.Parameter(projection.in_basis.clone(), requires_grad=True)
        self.q_select = nn.Parameter(projection.q_select.clone(), requires_grad=True)
        self.k_select = nn.Parameter(projection.k_select.clone(), requires_grad=True)
        self.v_input = nn.Parameter(projection.v_input.clone(), requires_grad=True)
        self.out_proj = nn.Parameter(projection.out_proj.clone(), requires_grad=True)
        self.dynamics = _MambaState(state_dim=projection.state_dim)

    @property
    def state_dim(self) -> int:
        return int(self.q_select.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.in_basis.shape[1])

    def forward_step(self, x_t: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.linear(x_t, self.in_basis)
        b_t = F.linear(z, self.k_select)
        c_t = F.linear(z, self.q_select)
        v_t = F.linear(z, self.v_input)
        new_state, y = self.dynamics.step(state, b_t=b_t, v_t=v_t, c_t=c_t)
        out = F.linear(y, self.out_proj)
        return out, new_state


class GQAMambaRankCollapseBlock(nn.Module):
    def __init__(self, projections: Sequence[GroupMambaProjection]) -> None:
        super().__init__()
        if not projections:
            raise ValueError("Expected at least one group projection")
        self.groups = nn.ModuleList([_GroupMambaRuntime(p) for p in projections])
        hidden = self.groups[0].hidden_size
        for g in self.groups[1:]:
            if g.hidden_size != hidden:
                raise ValueError("All group projections must share hidden_size")
        self.hidden_size = hidden

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected [B,T,H], got {tuple(hidden_states.shape)}")
        bsz, seq_len, hidden = hidden_states.shape
        if hidden != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden}")

        states = [hidden_states.new_zeros((bsz, g.state_dim)) for g in self.groups]
        outputs = []
        for t in range(seq_len):
            x_t = hidden_states[:, t, :]
            acc = hidden_states.new_zeros((bsz, hidden))
            for i, group in enumerate(self.groups):
                y_t, states[i] = group.forward_step(x_t, states[i])
                acc = acc + y_t
            outputs.append(acc)
        return torch.stack(outputs, dim=1)

    def set_sync_trainability(self, train_projections: bool = True, train_dynamics: bool = True) -> None:
        for group in self.groups:
            group.in_basis.requires_grad = bool(train_projections)
            group.q_select.requires_grad = bool(train_projections)
            group.k_select.requires_grad = bool(train_projections)
            group.v_input.requires_grad = bool(train_projections)
            group.out_proj.requires_grad = bool(train_projections)
            group.dynamics.a_logit.requires_grad = bool(train_dynamics)
            group.dynamics.delta_log.requires_grad = bool(train_dynamics)


def freeze_modules_matching(model: nn.Module, name_fragments: Iterable[str]) -> None:
    fragments = tuple(str(x).lower() for x in name_fragments)
    for name, module in model.named_modules():
        low_name = name.lower()
        if any(fragment in low_name for fragment in fragments):
            for p in module.parameters(recurse=False):
                p.requires_grad = False


def sync_alignment_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mode: str = "mse",
) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student and teacher logits must match, got {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if mode == "mse":
        return F.mse_loss(student_logits, teacher_logits)
    if mode == "cosine":
        s = student_logits.reshape(-1, student_logits.shape[-1])
        t = teacher_logits.reshape(-1, teacher_logits.shape[-1])
        return 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
    raise ValueError(f"Unsupported sync mode: {mode}")


def run_geometry_sync(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int = 200,
    loss_mode: str = "mse",
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    teacher_model.eval()
    student_model.train()
    running_loss = 0.0
    steps = 0

    for batch in dataloader:
        if steps >= max_steps:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        loss = sync_alignment_loss(student_logits, teacher_logits, mode=loss_mode)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        running_loss += float(loss.item())
        steps += 1

    mean_loss = running_loss / max(steps, 1)
    return {"steps": float(steps), "mean_sync_loss": mean_loss}


@dataclass(frozen=True)
class EightGBExecutionBudget:
    compressed_mamba_gb: float = 4.7
    mamba_states_gb: float = 0.5
    streamed_sparse_mlp_gb: float = 1.9
    scratch_overhead_gb: float = 0.9

    def total_vram_gb(self) -> float:
        return float(
            self.compressed_mamba_gb + self.mamba_states_gb + self.streamed_sparse_mlp_gb + self.scratch_overhead_gb
        )

    def fits(self, vram_gb: float = 8.0) -> bool:
        return self.total_vram_gb() <= float(vram_gb) + 1e-9


@dataclass(frozen=True)
class TokenLatencyEstimate:
    pcie_bandwidth_gbps: float = 15.0
    streamed_sparse_mlp_gb: float = 1.9
    on_gpu_compute_seconds: float = 0.03

    def transfer_seconds(self) -> float:
        return float(self.streamed_sparse_mlp_gb / max(self.pcie_bandwidth_gbps, 1e-9))

    def total_seconds(self) -> float:
        return float(self.transfer_seconds() + self.on_gpu_compute_seconds)

    def tokens_per_second(self) -> float:
        return float(1.0 / max(self.total_seconds(), 1e-9))

