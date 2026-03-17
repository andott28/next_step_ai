import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from sca_decoder_mirror import DecoderMirrorConfig, SparseDecoderMirrorSCA


def _mirror_config(**kwargs) -> DecoderMirrorConfig:
    base = dict(
        hidden_size=8,
        block_size=2,
        num_blocks=4,
        top_k=1,
        rank=2,
        grid_size=2,
        sigma=1.0,
        route_prior_scale_init=0.25,
        residual_scale_init=0.0,
        source_layer_indices=[0],
        enabled=True,
        route_conditioned=True,
    )
    base.update(kwargs)
    return DecoderMirrorConfig(**base)


def test_decoder_mirror_is_identity_at_init():
    torch.manual_seed(0)
    mirror = SparseDecoderMirrorSCA(_mirror_config(residual_scale_init=0.0))
    hidden = torch.randn(2, 3, 8)
    route_prior = torch.rand(2, 3, 4)
    warped, _diag = mirror(hidden, route_prior=route_prior, source_layers_used=[0])
    assert torch.allclose(warped, hidden, atol=1e-6, rtol=1e-6)


def test_decoder_mirror_route_prior_changes_block_selection():
    mirror = SparseDecoderMirrorSCA(_mirror_config())
    with torch.no_grad():
        mirror.spatial_proj.weight.zero_()
        mirror.spatial_proj.bias.zero_()
        mirror.route_prior_scale.fill_(10.0)

    hidden = torch.zeros(1, 1, 8)
    prior_a = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    prior_b = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
    _ = mirror(hidden, route_prior=prior_a, source_layers_used=[0])
    active_a = mirror.get_last_active_idx().clone()
    _ = mirror(hidden, route_prior=prior_b, source_layers_used=[0])
    active_b = mirror.get_last_active_idx().clone()
    assert int(active_a[0, 0].item()) == 0
    assert int(active_b[0, 0].item()) == 3


def test_decoder_mirror_sparse_residual_only_touches_active_blocks():
    torch.manual_seed(1)
    mirror = SparseDecoderMirrorSCA(_mirror_config(residual_scale_init=1.0))
    with torch.no_grad():
        mirror.spatial_proj.weight.zero_()
        mirror.spatial_proj.bias.zero_()
        mirror.route_prior_scale.fill_(10.0)
    hidden = torch.randn(1, 2, 8)
    route_prior = torch.tensor([[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
    warped, _ = mirror(hidden, route_prior=route_prior, source_layers_used=[0])
    delta = (warped - hidden).reshape(-1, 4, 2)
    for row in range(delta.shape[0]):
        for block in range(delta.shape[1]):
            if block == 2:
                assert not torch.allclose(delta[row, block], torch.zeros_like(delta[row, block]))
            else:
                assert torch.allclose(delta[row, block], torch.zeros_like(delta[row, block]), atol=1e-7)


def test_decoder_mirror_export_load_round_trip(tmp_path):
    torch.manual_seed(2)
    mirror_a = SparseDecoderMirrorSCA(_mirror_config(residual_scale_init=0.5))
    with torch.no_grad():
        mirror_a.route_prior_scale.fill_(1.5)
    payload = {
        "mirror_config": mirror_a.config.to_dict(),
        "decoder_mirror_state_dict": mirror_a.state_dict(),
    }
    ckpt = tmp_path / "decoder_mirror.pt"
    torch.save(payload, ckpt)

    mirror_b = SparseDecoderMirrorSCA(DecoderMirrorConfig(**payload["mirror_config"]))
    mirror_b.load_state_dict(payload["decoder_mirror_state_dict"], strict=True)

    hidden = torch.randn(1, 2, 8)
    route_prior = torch.rand(1, 2, 4)
    out_a, _ = mirror_a(hidden, route_prior=route_prior, source_layers_used=[0])
    out_b, _ = mirror_b(hidden, route_prior=route_prior, source_layers_used=[0])
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_decoder_mirror_no_route_prior_falls_back_to_hidden_only_behavior():
    mirror = SparseDecoderMirrorSCA(_mirror_config())
    with torch.no_grad():
        mirror.spatial_proj.weight.zero_()
        mirror.spatial_proj.bias.zero_()
    hidden = torch.zeros(1, 1, 8)
    _ = mirror(hidden, route_prior=None, source_layers_used=[])
    diag_none = mirror.get_last_diagnostics()
    active_none = mirror.get_last_active_idx().clone()
    _ = mirror(hidden, route_prior=torch.zeros(1, 1, 4), source_layers_used=[0])
    diag_zero = mirror.get_last_diagnostics()
    active_zero = mirror.get_last_active_idx().clone()
    assert diag_none["route_prior_missing"] is True
    assert diag_zero["route_prior_missing"] is False
    assert torch.equal(active_none, active_zero)
