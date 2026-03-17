import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from sca_sparse_config import SCASparseConfig
from sca_sparse_mlp import SparseLlamaMLP, SparseRouteSelection


class _ToyMLP(nn.Module):
    def __init__(self, hidden_size: int = 64, intermediate_size: int = 80) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(self.act_fn(gate) * up)


def _config(**kwargs) -> SCASparseConfig:
    base = dict(
        hidden_size=64,
        block_size=8,
        block_rank=2,
        top_k=2,
        sigma=1.0,
        refractory_steps=0,
        inhibition_lambda=0.0,
        use_cuda=False,
        grid_size=4,
        spmm_impl="dense",
        soft_mask=False,
    )
    base.update(kwargs)
    return SCASparseConfig(**base)


def _fixed_route(hidden_states: torch.Tensor, _layer_idx: int) -> SparseRouteSelection:
    rows = hidden_states.shape[0] * hidden_states.shape[1]
    idx = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [1, 3],
            [0, 7],
        ],
        device=hidden_states.device,
        dtype=torch.long,
    )
    active_idx = idx[:rows].contiguous()
    score_weights = torch.full(
        active_idx.shape,
        0.5,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    return SparseRouteSelection(active_idx=active_idx, score_weights=score_weights)


def _build_mask(
    cfg: SCASparseConfig,
    hidden_states: torch.Tensor,
    active_idx: torch.Tensor,
    score_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    rows = hidden_states.shape[0] * hidden_states.shape[1]
    mask_blocks = torch.zeros((rows, cfg.num_blocks), device=hidden_states.device, dtype=hidden_states.dtype)
    row_idx = torch.arange(rows, device=hidden_states.device)
    for slot in range(active_idx.shape[1]):
        block_idx = active_idx[:, slot]
        valid = block_idx >= 0
        values = (
            score_weights[valid, slot].to(dtype=hidden_states.dtype)
            if score_weights is not None
            else torch.ones((int(valid.sum().item()),), device=hidden_states.device, dtype=hidden_states.dtype)
        )
        mask_blocks[row_idx[valid], block_idx[valid]] += values
    if score_weights is None:
        mask_blocks.clamp_(max=1.0)
    return (
        mask_blocks.unsqueeze(-1)
        .expand(rows, cfg.num_blocks, cfg.block_size)
        .reshape(hidden_states.shape[0], hidden_states.shape[1], cfg.hidden_size)
    )


def test_sparse_llama_mlp_dense_mode_matches_manual_masked_mlp():
    torch.manual_seed(0)
    cfg = _config(spmm_impl="dense", soft_mask=False)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    route = _fixed_route(hidden, 0)
    mask = _build_mask(cfg, hidden, route.active_idx, None)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(hidden * mask)) * mlp.up_proj(hidden * mask)) * mask
    got = wrapper(hidden)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_sparse_llama_mlp_outputs_only_active_blocks():
    torch.manual_seed(1)
    cfg = _config(spmm_impl="dense", soft_mask=False)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    route = _fixed_route(hidden, 0)
    out = wrapper(hidden).reshape(-1, cfg.num_blocks, cfg.block_size)
    for row in range(out.shape[0]):
        active = set(int(v.item()) for v in route.active_idx[row] if int(v.item()) >= 0)
        for block in range(cfg.num_blocks):
            if block not in active:
                assert torch.allclose(out[row, block], torch.zeros_like(out[row, block]), atol=1e-7)


def test_sparse_llama_mlp_output_sparse_mode_matches_manual_formula():
    torch.manual_seed(101)
    cfg = _config(spmm_impl="dense", soft_mask=False, sparse_placement="output_sparse")
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    route = _fixed_route(hidden, 0)
    activated = F.silu(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
    expected = wrapper._project_output_blocks(activated.reshape(-1, activated.shape[-1]), mlp.down_proj, route.active_idx).view_as(hidden)
    got = wrapper(hidden)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_sparse_llama_mlp_intermediate_group_mode_matches_manual_formula():
    torch.manual_seed(102)
    cfg = _config(spmm_impl="dense", soft_mask=False, sparse_placement="intermediate_group")
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    route = _fixed_route(hidden, 0)
    gate = mlp.gate_proj(hidden).reshape(-1, mlp.gate_proj.out_features)
    up = mlp.up_proj(hidden).reshape(-1, mlp.up_proj.out_features)
    activated = F.silu(gate) * up
    inter_mask = wrapper._build_intermediate_group_mask(
        active_idx=route.active_idx,
        rows=activated.shape[0],
        intermediate_size=activated.shape[-1],
        device=activated.device,
        dtype=activated.dtype,
        score_weights=None,
    )
    expected = mlp.down_proj((activated * inter_mask).view_as(gate)).view_as(hidden)
    got = wrapper(hidden)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_sparse_llama_mlp_learned_basis_only_writes_active_blocks():
    torch.manual_seed(103)
    cfg = _config(spmm_impl="dense", soft_mask=False, sparse_placement="learned_basis", basis_rank=4, basis_top_k=2)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    route = _fixed_route(hidden, 0)
    out = wrapper(hidden).reshape(-1, cfg.num_blocks, cfg.block_size)
    for row in range(out.shape[0]):
        active = set(int(v.item()) for v in route.active_idx[row] if int(v.item()) >= 0)
        for block in range(cfg.num_blocks):
            if block not in active:
                assert torch.allclose(out[row, block], torch.zeros_like(out[row, block]), atol=1e-7)


def test_sparse_llama_mlp_learned_basis_reports_sparse_latent_stats():
    torch.manual_seed(104)
    cfg = _config(spmm_impl="dense", soft_mask=False, sparse_placement="learned_basis", basis_rank=6, basis_top_k=2)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    _ = wrapper(hidden)
    stats = wrapper.get_last_learned_basis_stats()
    assert stats is not None
    assert stats["active_latent_fraction"] == pytest.approx(float(cfg.basis_top_k / cfg.basis_rank), rel=1e-6)
    assert 0.0 <= stats["support_overlap_mean"] <= 1.0
    reg_tensors = wrapper.get_last_learned_basis_regularizer_tensors()
    assert reg_tensors is not None
    assert reg_tensors["_coord_probs"].shape[0] == cfg.basis_rank


def test_sparse_llama_mlp_initialize_sparse_basis_from_dense_init():
    torch.manual_seed(105)
    cfg = _config(spmm_impl="dense", soft_mask=False, sparse_placement="learned_basis", basis_rank=6, basis_top_k=2)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=1,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )
    payload = {
        "encoder_weight": torch.full_like(wrapper.sparse_basis_encoder.weight, 0.25),
        "encoder_bias": torch.full_like(wrapper.sparse_basis_encoder.bias, 0.5),
        "decoder_blocks": torch.full_like(wrapper.sparse_basis_decoder, 0.75),
        "decoder_bias": torch.full_like(wrapper.sparse_basis_bias, 1.25),
        "scale": torch.tensor(1.5),
    }
    info = wrapper.initialize_sparse_basis_from_dense_init(payload, strict=True)
    assert info["loaded_items"] >= 4
    assert torch.allclose(wrapper.sparse_basis_encoder.weight, payload["encoder_weight"])
    assert torch.allclose(wrapper.sparse_basis_encoder.bias, payload["encoder_bias"])
    assert torch.allclose(wrapper.sparse_basis_decoder, payload["decoder_blocks"])
    assert torch.allclose(wrapper.sparse_basis_bias, payload["decoder_bias"])


def test_sparse_llama_mlp_torch_block_sparse_matches_dense_masked():
    torch.manual_seed(2)
    cfg = _config(spmm_impl="torch_block_sparse", soft_mask=False)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    route = _fixed_route(hidden, 0)
    mask = _build_mask(cfg, hidden, route.active_idx, None)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(hidden * mask)) * mlp.up_proj(hidden * mask)) * mask
    got = wrapper(hidden)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_sparse_llama_mlp_disabled_path_matches_base_mlp():
    torch.manual_seed(3)
    cfg = _config(spmm_impl="dense", soft_mask=True)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: False,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    expected = mlp(hidden)
    got = wrapper(hidden)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_sparse_llama_mlp_alignment_capture_records_dense_when_disabled():
    torch.manual_seed(106)
    cfg = _config(spmm_impl="dense", soft_mask=False)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: False,
    )
    wrapper.set_alignment_capture(True)
    hidden = torch.randn(2, 3, cfg.hidden_size)
    out = wrapper(hidden)
    align = wrapper.get_last_alignment()
    assert align is not None
    assert "mlp_input" in align
    assert torch.allclose(align["mlp_input"], hidden, atol=1e-6, rtol=1e-5)
    assert torch.allclose(align["dense_mlp_out"], out, atol=1e-6, rtol=1e-5)
    assert torch.allclose(align["sparse_mlp_out"], out, atol=1e-6, rtol=1e-5)


def test_sparse_llama_mlp_cuda_spmm_cpu_raises():
    torch.manual_seed(4)
    cfg = _config(spmm_impl="cuda_spmm", soft_mask=False)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    with pytest.raises(RuntimeError):
        wrapper(hidden)


def test_sparse_llama_mlp_grouped_row_gemm_matches_dense_masked():
    torch.manual_seed(5)
    cfg = _config(spmm_impl="dense", soft_mask=False, grouped_row_gemm=True, grouped_row_min_bucket=1)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    route = _fixed_route(hidden, 0)
    mask = _build_mask(cfg, hidden, route.active_idx, None)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(hidden * mask)) * mlp.up_proj(hidden * mask)) * mask
    got = wrapper(hidden)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_sparse_mlp_diagnostics_report_nonzero_activity():
    torch.manual_seed(6)
    cfg = _config(spmm_impl="dense", soft_mask=False)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=3,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    _ = wrapper(hidden)
    diagnostics = wrapper.get_last_diagnostics()
    assert diagnostics is not None
    assert diagnostics.layer_idx == 3
    assert diagnostics.mean_active_blocks > 0.0
    assert diagnostics.unique_active_blocks > 0
    assert diagnostics.touched_weight_fraction > 0.0
    assert diagnostics.estimated_bytes_fetched_per_token > 0.0


def test_sparse_mlp_block_bank_matches_dense_masked():
    torch.manual_seed(7)
    cfg = _config(spmm_impl="dense", soft_mask=False)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    gate_weight = mlp.gate_proj.weight.detach().float().cpu()
    up_weight = mlp.up_proj.weight.detach().float().cpu()
    down_weight = mlp.down_proj.weight.detach().float().cpu()
    payload = {
        "layer_idx": 0,
        "num_blocks": cfg.num_blocks,
        "block_size": cfg.block_size,
        "gate_proj_blocks": gate_weight.reshape(gate_weight.shape[0], cfg.num_blocks, cfg.block_size).permute(1, 0, 2).contiguous(),
        "up_proj_blocks": up_weight.reshape(up_weight.shape[0], cfg.num_blocks, cfg.block_size).permute(1, 0, 2).contiguous(),
        "down_proj_blocks": down_weight.reshape(cfg.num_blocks, cfg.block_size, down_weight.shape[1]).contiguous(),
        "gate_bias": None,
        "up_bias": None,
        "down_bias": None,
    }
    wrapper.load_block_bank(payload, strict=True)
    assert wrapper.has_block_bank() is True

    hidden = torch.randn(2, 3, cfg.hidden_size)
    route = _fixed_route(hidden, 0)
    mask = _build_mask(cfg, hidden, route.active_idx, None)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(hidden * mask)) * mlp.up_proj(hidden * mask)) * mask
    got = wrapper(hidden)
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_sparse_mlp_stability_fallback_uses_dense_output():
    torch.manual_seed(8)
    cfg = _config(spmm_impl="dense", soft_mask=False)
    cfg.stability_dense_fallback_threshold = 0.5
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    got = wrapper(hidden)
    expected = mlp(hidden)
    diagnostics = wrapper.get_last_diagnostics()
    assert diagnostics is not None
    assert diagnostics.touched_weight_fraction > 0.0
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_sparse_mlp_fast_fallback_skips_routing():
    torch.manual_seed(9)
    cfg = _config(spmm_impl="dense", soft_mask=False, top_k=1)
    cfg.stability_dense_fallback_threshold = 0.5
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)

    def _route_never_called(_hidden_states: torch.Tensor, _layer_idx: int):
        raise RuntimeError("route_fn should not be called in fast fallback")

    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_route_never_called,
        enabled_fn=lambda _idx: True,
    )

    hidden = torch.randn(2, 3, cfg.hidden_size)
    got = wrapper(hidden)
    expected = mlp(hidden)
    diagnostics = wrapper.get_last_diagnostics()
    assert diagnostics is not None
    assert diagnostics.mean_active_blocks > 0.0
    assert torch.allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_sparse_mlp_alignment_capture_records_dense_and_sparse_outputs():
    torch.manual_seed(10)
    cfg = _config(spmm_impl="dense", soft_mask=False)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=0,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )
    wrapper.set_alignment_capture(True)

    hidden = torch.randn(2, 3, cfg.hidden_size)
    _ = wrapper(hidden)
    alignment = wrapper.get_last_alignment()
    assert alignment is not None
    assert "dense_mlp_out" in alignment
    assert "sparse_mlp_out" in alignment
    assert "feature_mask" in alignment
    assert "active_idx" in alignment
    assert alignment["dense_mlp_out"].shape == hidden.shape
    assert alignment["sparse_mlp_out"].shape == hidden.shape


def test_route_snapshot_capture_records_active_idx_and_score_weights():
    torch.manual_seed(11)
    cfg = _config(spmm_impl="dense", soft_mask=False)
    mlp = _ToyMLP(hidden_size=cfg.hidden_size)
    wrapper = SparseLlamaMLP(
        base_mlp=mlp,
        config=cfg,
        layer_idx=2,
        route_fn=_fixed_route,
        enabled_fn=lambda _idx: True,
    )
    wrapper.set_route_capture(True)

    hidden = torch.randn(2, 3, cfg.hidden_size)
    _ = wrapper(hidden)
    snapshot = wrapper.get_last_route_snapshot()
    assert snapshot is not None
    assert snapshot["active_idx"].shape == (hidden.shape[0] * hidden.shape[1], cfg.top_k)
    assert snapshot["score_weights"].shape == (hidden.shape[0] * hidden.shape[1], cfg.top_k)
    assert int(snapshot["rows"].item()) == hidden.shape[0] * hidden.shape[1]
    assert int(snapshot["batch_size"].item()) == hidden.shape[0]
    assert int(snapshot["seq_len"].item()) == hidden.shape[1]
    assert int(snapshot["layer_idx"].item()) == 2
