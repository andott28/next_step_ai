from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from llama3_neuroplastic.layer_selection import parse_layer_selection

try:
    from .streaming_llama_runtime import ShardedSafetensorLoader, _resolve_snapshot_dir
except ImportError:
    from streaming_llama_runtime import ShardedSafetensorLoader, _resolve_snapshot_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a no-training cross-layer q/o attention sharing artifact for streamed inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", type=str, required=True, help="HF repo id or local snapshot directory.")
    p.add_argument("--output-path", type=str, required=True, help="Output .pt path for the attention-sharing artifact.")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--layers", type=str, default="all", help='Comma-separated layers or "all".')
    p.add_argument("--group-size", type=int, default=2, help="Adjacent shared-layer group size.")
    p.add_argument("--exact-lower-layers", type=int, default=8, help="Keep the first N layers exact.")
    p.add_argument("--exact-upper-layers", type=int, default=6, help="Keep the last N layers exact.")
    p.add_argument("--base-rank", type=int, default=16, help="Low-rank size for the shared group q/o base.")
    p.add_argument("--residual-rank", type=int, default=4, help="Low-rank size for each layer-specific q/o residual.")
    p.add_argument("--match-sample-cols", type=int, default=256, help="Sampled q/o width used for greedy head matching.")
    p.add_argument("--match-sample-rows", type=int, default=256, help="Sampled q/o height used for greedy head matching.")
    p.add_argument(
        "--max-layer-rel-error",
        type=float,
        default=0.0,
        help="If > 0, reject a shared group when any layer's max(q_rel_error, o_rel_error) exceeds this threshold.",
    )
    p.add_argument(
        "--bridge-mode",
        type=str,
        default="ortho_centroid",
        choices=["none", "ortho_centroid"],
        help="Cross-layer bridge used before shared-base fitting.",
    )
    p.add_argument("--factor-device", type=str, default="", help='Device for low-rank factorization, e.g. "cuda" or "cpu".')
    return p.parse_args()


def _parse_layer_selection(spec: str, total_layers: int) -> list[int]:
    return parse_layer_selection(spec, total_layers=int(total_layers)) or []


def _build_layer_groups(
    selected_layers: Sequence[int],
    *,
    total_layers: int,
    group_size: int,
    exact_lower_layers: int,
    exact_upper_layers: int,
) -> tuple[list[list[int]], list[int]]:
    selected = sorted({int(v) for v in selected_layers if 0 <= int(v) < int(total_layers)})
    exact: list[int] = []
    shareable: list[int] = []
    lower_cut = max(int(exact_lower_layers), 0)
    upper_cut = int(total_layers) - max(int(exact_upper_layers), 0)
    for layer_idx in selected:
        if layer_idx < lower_cut or layer_idx >= upper_cut:
            exact.append(layer_idx)
        else:
            shareable.append(layer_idx)
    groups: list[list[int]] = []
    cur: list[int] = []
    prev: int | None = None
    for layer_idx in shareable:
        if prev is None or (layer_idx == prev + 1 and len(cur) < int(group_size)):
            cur.append(layer_idx)
        else:
            if cur:
                groups.append(cur)
            cur = [layer_idx]
        prev = layer_idx
        if len(cur) >= int(group_size):
            groups.append(cur)
            cur = []
            prev = None
    if cur:
        if len(cur) == 1:
            exact.append(cur[0])
        else:
            groups.append(cur)
    return groups, sorted(set(exact))


def _reshape_q_heads(weight: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    return weight.view(int(num_heads), int(head_dim), int(weight.shape[1]))


def _reshape_kv_heads(weight: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    return weight.view(int(num_heads), int(head_dim), int(weight.shape[1]))


def _reshape_o_heads(weight: torch.Tensor, *, num_heads: int, head_dim: int) -> torch.Tensor:
    return weight.view(int(weight.shape[0]), int(num_heads), int(head_dim)).permute(1, 0, 2).contiguous()


def _permute_q_rows(weight: torch.Tensor, head_perm: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    q_heads = _reshape_q_heads(weight, num_heads=int(head_perm.numel()), head_dim=int(head_dim))
    return q_heads.index_select(0, head_perm.to(device=q_heads.device, dtype=torch.long)).reshape_as(weight)


def _permute_o_cols(weight: torch.Tensor, head_perm: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    o_heads = _reshape_o_heads(weight, num_heads=int(head_perm.numel()), head_dim=int(head_dim))
    aligned = o_heads.index_select(0, head_perm.to(device=o_heads.device, dtype=torch.long))
    return aligned.permute(1, 0, 2).reshape_as(weight)


def _permute_kv_rows(weight: torch.Tensor, head_perm: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    kv_heads = _reshape_kv_heads(weight, num_heads=int(head_perm.numel()), head_dim=int(head_dim))
    return kv_heads.index_select(0, head_perm.to(device=kv_heads.device, dtype=torch.long)).reshape_as(weight)


def _evenly_spaced_indices(size: int, count: int) -> torch.Tensor:
    if count <= 0 or size <= count:
        return torch.arange(size, dtype=torch.long)
    return torch.linspace(0, size - 1, steps=count, dtype=torch.float32).round().long().unique(sorted=True)


def _build_head_signature(
    q_weight: torch.Tensor,
    o_weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    sample_cols: int,
    sample_rows: int,
) -> torch.Tensor:
    q_heads = _reshape_q_heads(q_weight, num_heads=num_heads, head_dim=head_dim)
    o_heads = _reshape_o_heads(o_weight, num_heads=num_heads, head_dim=head_dim)
    q_cols = _evenly_spaced_indices(int(q_heads.shape[-1]), int(sample_cols))
    o_rows = _evenly_spaced_indices(int(o_heads.shape[1]), int(sample_rows))
    q_sig = q_heads.index_select(2, q_cols).reshape(int(num_heads), -1).float()
    o_sig = o_heads.index_select(1, o_rows).reshape(int(num_heads), -1).float()
    sig = torch.cat([q_sig, o_sig], dim=-1)
    return F.normalize(sig, dim=-1, eps=1e-6)


def _build_kv_head_signature(
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    sample_cols: int,
) -> torch.Tensor:
    k_heads = _reshape_kv_heads(k_weight, num_heads=num_heads, head_dim=head_dim)
    v_heads = _reshape_kv_heads(v_weight, num_heads=num_heads, head_dim=head_dim)
    cols = _evenly_spaced_indices(int(k_heads.shape[-1]), int(sample_cols))
    k_sig = k_heads.index_select(2, cols).reshape(int(num_heads), -1).float()
    v_sig = v_heads.index_select(2, cols).reshape(int(num_heads), -1).float()
    sig = torch.cat([k_sig, v_sig], dim=-1)
    return F.normalize(sig, dim=-1, eps=1e-6)


def _greedy_head_permutation(ref_sig: torch.Tensor, cur_sig: torch.Tensor) -> torch.Tensor:
    if ref_sig.shape != cur_sig.shape:
        raise RuntimeError(f"Head-signature mismatch: ref={tuple(ref_sig.shape)} cur={tuple(cur_sig.shape)}")
    num_heads = int(ref_sig.shape[0])
    sim = torch.matmul(ref_sig, cur_sig.transpose(0, 1))
    perm = torch.empty(num_heads, dtype=torch.long)
    used = torch.zeros(num_heads, dtype=torch.bool)
    for ref_idx in range(num_heads):
        row = sim[ref_idx].clone()
        row[used] = float("-inf")
        cur_idx = int(torch.argmax(row).item())
        if used[cur_idx]:
            raise RuntimeError("Greedy head matching failed to find an unused head.")
        perm[ref_idx] = cur_idx
        used[cur_idx] = True
    return perm


def _factor_matrix_low_rank(matrix: torch.Tensor, rank: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = matrix.to(device=device, dtype=torch.float32)
    max_rank = max(1, min(int(rank), int(min(matrix.shape))))
    if max_rank >= min(int(matrix.shape[0]), int(matrix.shape[1])):
        return matrix.cpu().to(dtype=torch.float16), torch.eye(int(matrix.shape[1]), dtype=torch.float16)
    q = max_rank + min(8, max(1, min(matrix.shape) - max_rank))
    u, s, v = torch.svd_lowrank(matrix, q=int(q), niter=2)
    u = u[:, :max_rank]
    s = s[:max_rank]
    v = v[:, :max_rank]
    left = (u * s.unsqueeze(0)).to(device="cpu", dtype=torch.float16).contiguous()
    right = v.transpose(0, 1).to(device="cpu", dtype=torch.float16).contiguous()
    return left, right


def _reconstruct_from_factors(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.matmul(left.float(), right.float())


def _factor_head_stack_low_rank(head_stack: torch.Tensor, rank: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if head_stack.ndim != 3:
        raise RuntimeError(f"Expected [num_heads, rows, cols] head stack, got {tuple(head_stack.shape)}")
    num_heads = int(head_stack.shape[0])
    left_parts: list[torch.Tensor] = []
    right_parts: list[torch.Tensor] = []
    for head_idx in range(num_heads):
        left, right = _factor_matrix_low_rank(head_stack[head_idx], rank, device=device)
        left_parts.append(left)
        right_parts.append(right)
    return torch.stack(left_parts, dim=0).contiguous(), torch.stack(right_parts, dim=0).contiguous()


def _reconstruct_head_stack_from_factors(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.ndim != 3 or right.ndim != 3:
        raise RuntimeError(f"Expected rank-3 headwise factors, got left={tuple(left.shape)} right={tuple(right.shape)}")
    return torch.bmm(left.float(), right.float())


def _q_heads_to_matrix(heads: torch.Tensor) -> torch.Tensor:
    return heads.reshape(int(heads.shape[0]) * int(heads.shape[1]), int(heads.shape[2]))


def _kv_heads_to_matrix(heads: torch.Tensor) -> torch.Tensor:
    return heads.reshape(int(heads.shape[0]) * int(heads.shape[1]), int(heads.shape[2]))


def _o_heads_to_matrix(heads: torch.Tensor) -> torch.Tensor:
    return heads.permute(1, 0, 2).reshape(int(heads.shape[1]), int(heads.shape[0]) * int(heads.shape[2]))


def _relative_reconstruction_error(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    ref = reference.float()
    approx = approximation.float()
    denom = float(torch.linalg.norm(ref).item())
    if denom <= 1e-12:
        return 0.0
    numer = float(torch.linalg.norm(ref - approx).item())
    return numer / denom


def _bridge_head_matrix_ortho_centroid(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Approximate target from source using output-space orthogonal rotation + centroid correction.

    For head matrices [rows, cols] with rows << cols, we solve:
      centered_target ~ T_out * centered_source * D_in
    where T_out is orthogonal (Procrustes) and D_in is a cheap signed diagonal
    alignment from column correlations. Target centroids are restored at the end.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise RuntimeError(f"Expected rank-2 matrices, got source={tuple(source.shape)} target={tuple(target.shape)}")
    if source.shape != target.shape:
        raise RuntimeError(f"Bridge source/target shape mismatch: {tuple(source.shape)} vs {tuple(target.shape)}")

    src = source.float()
    tgt = target.float()

    src_row_mean = src.mean(dim=1, keepdim=True)
    src_col_mean = src.mean(dim=0, keepdim=True)
    src_global = src.mean()
    tgt_row_mean = tgt.mean(dim=1, keepdim=True)
    tgt_col_mean = tgt.mean(dim=0, keepdim=True)
    tgt_global = tgt.mean()

    src_center = src - src_row_mean - src_col_mean + src_global
    tgt_center = tgt - tgt_row_mean - tgt_col_mean + tgt_global

    cross = torch.matmul(tgt_center, src_center.transpose(0, 1))
    u, _s, vh = torch.linalg.svd(cross, full_matrices=False)
    t_out = torch.matmul(u, vh)
    rotated = torch.matmul(t_out, src_center)


    col_corr = (rotated * tgt_center).sum(dim=0)
    sign = torch.where(col_corr >= 0.0, torch.ones_like(col_corr), -torch.ones_like(col_corr))
    rotated = rotated * sign.unsqueeze(0)

    return rotated + tgt_row_mean + tgt_col_mean - tgt_global


def _bridge_head_stack_ortho_centroid(source_heads: torch.Tensor, target_heads: torch.Tensor) -> torch.Tensor:
    if source_heads.ndim != 3 or target_heads.ndim != 3:
        raise RuntimeError(
            f"Expected rank-3 head stacks, got source={tuple(source_heads.shape)} target={tuple(target_heads.shape)}"
        )
    if source_heads.shape != target_heads.shape:
        raise RuntimeError(
            f"Head stack shape mismatch: source={tuple(source_heads.shape)} target={tuple(target_heads.shape)}"
        )
    out = torch.empty_like(target_heads, dtype=torch.float32)
    for head_idx in range(int(source_heads.shape[0])):
        out[head_idx] = _bridge_head_matrix_ortho_centroid(source_heads[head_idx], target_heads[head_idx])
    return out


def _fit_group_shared_qo(
    layers: Sequence[int],
    layer_q: dict[int, torch.Tensor],
    layer_o: dict[int, torch.Tensor],
    *,
    num_heads: int,
    head_dim: int,
    base_rank: int,
    residual_rank: int,
    sample_cols: int,
    sample_rows: int,
    factor_device: torch.device,
    bridge_mode: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    ordered_layers = [int(v) for v in layers]
    ref_layer = int(ordered_layers[0])
    ref_sig = _build_head_signature(
        layer_q[ref_layer],
        layer_o[ref_layer],
        num_heads=num_heads,
        head_dim=head_dim,
        sample_cols=sample_cols,
        sample_rows=sample_rows,
    )
    aligned_q: dict[int, torch.Tensor] = {}
    aligned_o: dict[int, torch.Tensor] = {}
    share_q: dict[int, torch.Tensor] = {}
    share_o: dict[int, torch.Tensor] = {}
    head_perm_by_layer: dict[int, torch.Tensor] = {}

    q_sum: torch.Tensor | None = None
    o_sum: torch.Tensor | None = None
    for layer_idx in ordered_layers:
        if layer_idx == ref_layer:
            head_perm = torch.arange(num_heads, dtype=torch.long)
        else:
            cur_sig = _build_head_signature(
                layer_q[layer_idx],
                layer_o[layer_idx],
                num_heads=num_heads,
                head_dim=head_dim,
                sample_cols=sample_cols,
                sample_rows=sample_rows,
            )
            head_perm = _greedy_head_permutation(ref_sig, cur_sig)
        q_aligned = _permute_q_rows(layer_q[layer_idx], head_perm, head_dim=head_dim).float()
        o_aligned = _permute_o_cols(layer_o[layer_idx], head_perm, head_dim=head_dim).float()
        aligned_q[layer_idx] = q_aligned
        aligned_o[layer_idx] = o_aligned
        head_perm_by_layer[layer_idx] = head_perm

    ref_q_heads = _reshape_q_heads(aligned_q[ref_layer], num_heads=num_heads, head_dim=head_dim).float()
    ref_o_heads = _reshape_o_heads(aligned_o[ref_layer], num_heads=num_heads, head_dim=head_dim).float()
    for layer_idx in ordered_layers:
        q_heads = _reshape_q_heads(aligned_q[layer_idx], num_heads=num_heads, head_dim=head_dim).float()
        o_heads = _reshape_o_heads(aligned_o[layer_idx], num_heads=num_heads, head_dim=head_dim).float()
        if layer_idx == ref_layer or str(bridge_mode) == "none":
            q_heads_share = q_heads
            o_heads_share = o_heads
        else:
            q_heads_share = _bridge_head_stack_ortho_centroid(ref_q_heads, q_heads)

            o_heads_share_t = _bridge_head_stack_ortho_centroid(
                ref_o_heads.transpose(1, 2).contiguous(),
                o_heads.transpose(1, 2).contiguous(),
            )
            o_heads_share = o_heads_share_t.transpose(1, 2).contiguous()
        share_q[layer_idx] = _q_heads_to_matrix(q_heads_share)
        share_o[layer_idx] = _o_heads_to_matrix(o_heads_share)
        q_sum = share_q[layer_idx] if q_sum is None else q_sum + share_q[layer_idx]
        o_sum = share_o[layer_idx] if o_sum is None else o_sum + share_o[layer_idx]

    if q_sum is None or o_sum is None:
        raise RuntimeError("No q/o weights were accumulated for the shared attention group.")
    q_mean = q_sum / float(len(ordered_layers))
    o_mean = o_sum / float(len(ordered_layers))
    q_mean_heads = _reshape_q_heads(q_mean, num_heads=num_heads, head_dim=head_dim)
    o_mean_heads = _reshape_o_heads(o_mean, num_heads=num_heads, head_dim=head_dim)
    q_base_u_heads, q_base_v_heads = _factor_head_stack_low_rank(q_mean_heads, int(base_rank), device=factor_device)
    o_base_u_heads, o_base_v_heads = _factor_head_stack_low_rank(o_mean_heads, int(base_rank), device=factor_device)
    q_base_recon = _q_heads_to_matrix(_reconstruct_head_stack_from_factors(q_base_u_heads, q_base_v_heads))
    o_base_recon = _o_heads_to_matrix(_reconstruct_head_stack_from_factors(o_base_u_heads, o_base_v_heads))

    group_state = {
        "sharing_format": "headwise_v1",
        "bridge_mode": str(bridge_mode),
        "q_base_u_heads": q_base_u_heads,
        "q_base_v_heads": q_base_v_heads,
        "o_base_u_heads": o_base_u_heads,
        "o_base_v_heads": o_base_v_heads,
        "recon_error_by_layer": {},
    }
    layer_states: dict[int, dict[str, Any]] = {}
    for layer_idx in ordered_layers:
        q_resid = aligned_q[layer_idx] - q_base_recon
        o_resid = aligned_o[layer_idx] - o_base_recon
        q_resid_heads = _reshape_q_heads(q_resid, num_heads=num_heads, head_dim=head_dim)
        o_resid_heads = _reshape_o_heads(o_resid, num_heads=num_heads, head_dim=head_dim)
        q_resid_u_heads, q_resid_v_heads = _factor_head_stack_low_rank(q_resid_heads, int(residual_rank), device=factor_device)
        o_resid_u_heads, o_resid_v_heads = _factor_head_stack_low_rank(o_resid_heads, int(residual_rank), device=factor_device)
        q_full_recon = q_base_recon + _q_heads_to_matrix(_reconstruct_head_stack_from_factors(q_resid_u_heads, q_resid_v_heads))
        o_full_recon = o_base_recon + _o_heads_to_matrix(_reconstruct_head_stack_from_factors(o_resid_u_heads, o_resid_v_heads))
        q_rel_error = _relative_reconstruction_error(aligned_q[layer_idx], q_full_recon)
        o_rel_error = _relative_reconstruction_error(aligned_o[layer_idx], o_full_recon)
        group_state["recon_error_by_layer"][int(layer_idx)] = {
            "q_rel_l2": float(q_rel_error),
            "o_rel_l2": float(o_rel_error),
            "max_rel_l2": float(max(q_rel_error, o_rel_error)),
        }
        layer_states[int(layer_idx)] = {
            "head_perm": head_perm_by_layer[layer_idx],
            "q_resid_u_heads": q_resid_u_heads,
            "q_resid_v_heads": q_resid_v_heads,
            "o_resid_u_heads": o_resid_u_heads,
            "o_resid_v_heads": o_resid_v_heads,
        }
    return group_state, layer_states


def _load_qo_weights(
    loader: ShardedSafetensorLoader,
    *,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = f"model.layers.{int(layer_idx)}.self_attn."
    q_weight = loader.load_parameter(f"{prefix}q_proj.weight").contiguous()
    o_weight = loader.load_parameter(f"{prefix}o_proj.weight").contiguous()
    return q_weight, o_weight


def _load_kv_weights(
    loader: ShardedSafetensorLoader,
    *,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = f"model.layers.{int(layer_idx)}.self_attn."
    k_weight = loader.load_parameter(f"{prefix}k_proj.weight").contiguous()
    v_weight = loader.load_parameter(f"{prefix}v_proj.weight").contiguous()
    return k_weight, v_weight


def _fit_group_shared_kv(
    layers: Sequence[int],
    layer_k: dict[int, torch.Tensor],
    layer_v: dict[int, torch.Tensor],
    *,
    num_kv_heads: int,
    head_dim: int,
    base_rank: int,
    residual_rank: int,
    sample_cols: int,
    factor_device: torch.device,
    bridge_mode: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    ordered_layers = [int(v) for v in layers]
    ref_layer = int(ordered_layers[0])
    ref_sig = _build_kv_head_signature(
        layer_k[ref_layer],
        layer_v[ref_layer],
        num_heads=num_kv_heads,
        head_dim=head_dim,
        sample_cols=sample_cols,
    )
    aligned_k: dict[int, torch.Tensor] = {}
    aligned_v: dict[int, torch.Tensor] = {}
    share_k: dict[int, torch.Tensor] = {}
    share_v: dict[int, torch.Tensor] = {}
    head_perm_by_layer: dict[int, torch.Tensor] = {}

    k_sum: torch.Tensor | None = None
    v_sum: torch.Tensor | None = None
    for layer_idx in ordered_layers:
        if layer_idx == ref_layer:
            head_perm = torch.arange(num_kv_heads, dtype=torch.long)
        else:
            cur_sig = _build_kv_head_signature(
                layer_k[layer_idx],
                layer_v[layer_idx],
                num_heads=num_kv_heads,
                head_dim=head_dim,
                sample_cols=sample_cols,
            )
            head_perm = _greedy_head_permutation(ref_sig, cur_sig)
        k_aligned = _permute_kv_rows(layer_k[layer_idx], head_perm, head_dim=head_dim).float()
        v_aligned = _permute_kv_rows(layer_v[layer_idx], head_perm, head_dim=head_dim).float()
        aligned_k[layer_idx] = k_aligned
        aligned_v[layer_idx] = v_aligned
        head_perm_by_layer[layer_idx] = head_perm

    ref_k_heads = _reshape_kv_heads(aligned_k[ref_layer], num_heads=num_kv_heads, head_dim=head_dim).float()
    ref_v_heads = _reshape_kv_heads(aligned_v[ref_layer], num_heads=num_kv_heads, head_dim=head_dim).float()
    for layer_idx in ordered_layers:
        k_heads = _reshape_kv_heads(aligned_k[layer_idx], num_heads=num_kv_heads, head_dim=head_dim).float()
        v_heads = _reshape_kv_heads(aligned_v[layer_idx], num_heads=num_kv_heads, head_dim=head_dim).float()
        if layer_idx == ref_layer or str(bridge_mode) == "none":
            k_heads_share = k_heads
            v_heads_share = v_heads
        else:
            k_heads_share = _bridge_head_stack_ortho_centroid(ref_k_heads, k_heads)
            v_heads_share = _bridge_head_stack_ortho_centroid(ref_v_heads, v_heads)
        share_k[layer_idx] = _kv_heads_to_matrix(k_heads_share)
        share_v[layer_idx] = _kv_heads_to_matrix(v_heads_share)
        k_sum = share_k[layer_idx] if k_sum is None else k_sum + share_k[layer_idx]
        v_sum = share_v[layer_idx] if v_sum is None else v_sum + share_v[layer_idx]

    if k_sum is None or v_sum is None:
        raise RuntimeError("No k/v weights were accumulated for the shared attention group.")
    k_mean = k_sum / float(len(ordered_layers))
    v_mean = v_sum / float(len(ordered_layers))
    k_mean_heads = _reshape_kv_heads(k_mean, num_heads=num_kv_heads, head_dim=head_dim)
    v_mean_heads = _reshape_kv_heads(v_mean, num_heads=num_kv_heads, head_dim=head_dim)
    k_base_u_heads, k_base_v_heads = _factor_head_stack_low_rank(k_mean_heads, int(base_rank), device=factor_device)
    v_base_u_heads, v_base_v_heads = _factor_head_stack_low_rank(v_mean_heads, int(base_rank), device=factor_device)
    k_base_recon = _kv_heads_to_matrix(_reconstruct_head_stack_from_factors(k_base_u_heads, k_base_v_heads))
    v_base_recon = _kv_heads_to_matrix(_reconstruct_head_stack_from_factors(v_base_u_heads, v_base_v_heads))

    group_state = {
        "bridge_mode": str(bridge_mode),
        "k_base_u_heads": k_base_u_heads,
        "k_base_v_heads": k_base_v_heads,
        "v_base_u_heads": v_base_u_heads,
        "v_base_v_heads": v_base_v_heads,
        "kv_recon_error_by_layer": {},
    }
    layer_states: dict[int, dict[str, Any]] = {}
    for layer_idx in ordered_layers:
        k_resid = aligned_k[layer_idx] - k_base_recon
        v_resid = aligned_v[layer_idx] - v_base_recon
        k_resid_heads = _reshape_kv_heads(k_resid, num_heads=num_kv_heads, head_dim=head_dim)
        v_resid_heads = _reshape_kv_heads(v_resid, num_heads=num_kv_heads, head_dim=head_dim)
        k_resid_u_heads, k_resid_v_heads = _factor_head_stack_low_rank(k_resid_heads, int(residual_rank), device=factor_device)
        v_resid_u_heads, v_resid_v_heads = _factor_head_stack_low_rank(v_resid_heads, int(residual_rank), device=factor_device)
        k_full_recon = k_base_recon + _kv_heads_to_matrix(_reconstruct_head_stack_from_factors(k_resid_u_heads, k_resid_v_heads))
        v_full_recon = v_base_recon + _kv_heads_to_matrix(_reconstruct_head_stack_from_factors(v_resid_u_heads, v_resid_v_heads))
        k_rel_error = _relative_reconstruction_error(aligned_k[layer_idx], k_full_recon)
        v_rel_error = _relative_reconstruction_error(aligned_v[layer_idx], v_full_recon)
        group_state["kv_recon_error_by_layer"][int(layer_idx)] = {
            "k_rel_l2": float(k_rel_error),
            "v_rel_l2": float(v_rel_error),
            "max_rel_l2": float(max(k_rel_error, v_rel_error)),
        }
        layer_states[int(layer_idx)] = {
            "kv_head_perm": head_perm_by_layer[layer_idx],
            "k_resid_u_heads": k_resid_u_heads,
            "k_resid_v_heads": k_resid_v_heads,
            "v_resid_u_heads": v_resid_u_heads,
            "v_resid_v_heads": v_resid_v_heads,
        }
    return group_state, layer_states


def main() -> None:
    args = _parse_args()
    snapshot_dir = _resolve_snapshot_dir(str(args.model_name), local_files_only=bool(args.local_files_only))
    loader = ShardedSafetensorLoader(snapshot_dir, pin_ram_cache=False)
    from transformers import AutoConfig

    model_cfg = AutoConfig.from_pretrained(str(snapshot_dir), local_files_only=bool(args.local_files_only))
    num_layers = int(getattr(model_cfg, "num_hidden_layers", 0))
    num_heads = int(getattr(model_cfg, "num_attention_heads", 0))
    num_kv_heads = int(getattr(model_cfg, "num_key_value_heads", 0) or num_heads)
    head_dim = int(getattr(model_cfg, "head_dim", 0))
    hidden_size = int(getattr(model_cfg, "hidden_size", 0))
    if num_layers <= 0 or num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0 or hidden_size <= 0:
        raise RuntimeError("Invalid model config for attention-sharing artifact build.")

    selected_layers = _parse_layer_selection(str(args.layers), num_layers)
    groups, exact_layers = _build_layer_groups(
        selected_layers,
        total_layers=num_layers,
        group_size=int(args.group_size),
        exact_lower_layers=int(args.exact_lower_layers),
        exact_upper_layers=int(args.exact_upper_layers),
    )
    factor_device = torch.device(str(args.factor_device).strip() or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload: dict[str, Any] = {
        "config": {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "group_size": int(args.group_size),
            "exact_lower_layers": int(args.exact_lower_layers),
            "exact_upper_layers": int(args.exact_upper_layers),
            "base_rank": int(args.base_rank),
            "residual_rank": int(args.residual_rank),
            "match_sample_cols": int(args.match_sample_cols),
            "match_sample_rows": int(args.match_sample_rows),
            "max_layer_rel_error": float(args.max_layer_rel_error),
            "bridge_mode": str(args.bridge_mode),
        },
        "group_states": {},
        "layer_states": {},
        "exact_layers": list(sorted(set(int(v) for v in exact_layers))),
    }

    for group_idx, layers in enumerate(groups):
        print(f"[attn_share] fitting group {group_idx + 1}/{len(groups)}: layers {layers}", flush=True)
        layer_q: dict[int, torch.Tensor] = {}
        layer_o: dict[int, torch.Tensor] = {}
        layer_k: dict[int, torch.Tensor] = {}
        layer_v: dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            q_weight, o_weight = _load_qo_weights(loader, layer_idx=int(layer_idx))
            k_weight, v_weight = _load_kv_weights(loader, layer_idx=int(layer_idx))
            layer_q[int(layer_idx)] = q_weight
            layer_o[int(layer_idx)] = o_weight
            layer_k[int(layer_idx)] = k_weight
            layer_v[int(layer_idx)] = v_weight
        group_state, layer_states = _fit_group_shared_qo(
            layers,
            layer_q,
            layer_o,
            num_heads=num_heads,
            head_dim=head_dim,
            base_rank=int(args.base_rank),
            residual_rank=int(args.residual_rank),
            sample_cols=int(args.match_sample_cols),
            sample_rows=int(args.match_sample_rows),
            factor_device=factor_device,
            bridge_mode=str(args.bridge_mode),
        )
        kv_group_state, kv_layer_states = _fit_group_shared_kv(
            layers,
            layer_k,
            layer_v,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            base_rank=int(args.base_rank),
            residual_rank=int(args.residual_rank),
            sample_cols=int(args.match_sample_cols),
            factor_device=factor_device,
            bridge_mode=str(args.bridge_mode),
        )
        group_state.update(kv_group_state)
        group_max_rel_error = (
            max(
                float(metrics.get("max_rel_l2", 0.0))
                for metrics in (group_state.get("recon_error_by_layer", {}) or {}).values()
            )
            if group_state.get("recon_error_by_layer")
            else 0.0
        )
        group_max_rel_error = max(
            group_max_rel_error,
            (
                max(
                    float(metrics.get("max_rel_l2", 0.0))
                    for metrics in (group_state.get("kv_recon_error_by_layer", {}) or {}).values()
                )
                if group_state.get("kv_recon_error_by_layer")
                else 0.0
            ),
        )
        if float(args.max_layer_rel_error) > 0.0 and group_max_rel_error > float(args.max_layer_rel_error):
            payload["exact_layers"].extend(int(v) for v in layers)
            print(
                f"[attn_share] rejected group {group_idx + 1}/{len(groups)}: layers {layers} "
                f"| max_rel_l2={group_max_rel_error:.4f} > {float(args.max_layer_rel_error):.4f}",
                flush=True,
            )
            del layer_q, layer_o, layer_k, layer_v
            if factor_device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        gid = str(group_idx)
        group_state["layers"] = list(int(v) for v in layers)
        payload["group_states"][gid] = group_state
        for layer_idx, layer_state in layer_states.items():
            kv_layer_state = kv_layer_states[int(layer_idx)]
            payload["layer_states"][str(int(layer_idx))] = {
                "group_id": gid,
                **layer_state,
                **kv_layer_state,
            }
        del layer_q, layer_o, layer_k, layer_v
        if factor_device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = Path(str(args.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload["exact_layers"] = list(sorted(set(int(v) for v in payload["exact_layers"])))
    torch.save(payload, output_path)
    print(
        f"[attn_share] wrote {output_path} | groups={len(payload['group_states'])} "
        f"| shared_layers={len(payload['layer_states'])} | exact_layers={len(payload['exact_layers'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
