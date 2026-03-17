import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from sca_sparse_adapter import SCABlockSparseAdapter, compute_active_blocks, compute_active_blocks_torch
from sca_sparse_config import SCASparseConfig, build_block_centers, build_inhibition_matrix
from triton_sca_gate import triton_sca_gate_available


def _config(**kwargs):
    base = dict(
        hidden_size=4096,
        block_size=32,
        block_rank=4,
        top_k=3,
        sigma=1.0,
        refractory_steps=100,
        inhibition_lambda=0.0,
        use_cuda=False,
        grid_size=16,
    )
    base.update(kwargs)
    return SCASparseConfig(**base)


def test_block_partition_shape_and_count():
    cfg = _config()
    centers = build_block_centers(cfg)
    assert cfg.num_blocks == 128
    assert centers.shape == (128, 3)


def test_coordinate_generation_deterministic():
    cfg = _config()
    c1 = build_block_centers(cfg)
    c2 = build_block_centers(cfg)
    assert torch.allclose(c1, c2)


def test_topk_returns_nearest_block():
    cfg = _config(top_k=3)
    centers = build_block_centers(cfg)
    q = centers[0].unsqueeze(0)
    idx, _ = compute_active_blocks_torch(q, centers, cfg)
    assert idx.shape == (1, 3)
    assert int(idx[0, 0].item()) == 0


def test_refractory_mask_blocks_recent_block():
    cfg = _config(top_k=1)
    centers = build_block_centers(cfg)
    q = centers[0].unsqueeze(0)

    refractory_mask = torch.zeros(cfg.num_blocks, dtype=torch.bool)
    refractory_mask[0] = True

    idx, _ = compute_active_blocks_torch(
        q,
        centers,
        cfg,
        refractory_mask=refractory_mask,
    )
    assert int(idx[0, 0].item()) != 0


def test_sparse_adapter_output_shape_and_finite():
    torch.manual_seed(42)
    cfg = _config()
    adapter = SCABlockSparseAdapter(cfg)

    hidden = torch.randn(2, 3, cfg.hidden_size)
    active_idx = torch.randint(0, cfg.num_blocks, (hidden.shape[0] * hidden.shape[1], cfg.top_k))

    delta = adapter.forward_sparse(hidden, active_idx, use_cuda_kernel=False)
    assert delta.shape == hidden.shape
    assert torch.isfinite(delta).all()


def test_topk_sparsity_invariant():
    torch.manual_seed(0)
    cfg = _config(top_k=3)
    centers = build_block_centers(cfg)
    q = torch.randn(17, 3)

    idx, _ = compute_active_blocks_torch(q, centers, cfg)
    valid_per_row = (idx >= 0).sum(dim=-1)
    assert torch.all(valid_per_row <= 3)
    assert torch.all(valid_per_row >= 1)


def test_inhibition_matrix_shape():
    cfg = _config()
    centers = build_block_centers(cfg)
    mat = build_inhibition_matrix(centers, radius=1.5)
    assert mat.shape == (cfg.num_blocks, cfg.num_blocks)


def test_flops_estimation_methods():
    cfg = _config(block_rank=4)
    adapter = SCABlockSparseAdapter(cfg)
    dense = adapter.dense_adapter_flops_per_token()
    sparse = adapter.sparse_adapter_flops_per_token(mean_active_blocks=3.0)
    assert dense > 0
    assert sparse > 0
    assert sparse < dense


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_path_parity_smoke():
    # Parity smoke against torch path only when CUDA exists.
    cfg = _config(use_cuda=False)
    adapter = SCABlockSparseAdapter(cfg).cuda().half()

    hidden = torch.randn(1, 2, cfg.hidden_size, device="cuda", dtype=torch.float16)
    centers = build_block_centers(cfg).to(device="cuda", dtype=torch.float32)
    q = torch.randn(hidden.shape[0] * hidden.shape[1], 3, device="cuda", dtype=torch.float32)
    idx, _ = compute_active_blocks_torch(q, centers, cfg)

    delta_torch = adapter.forward_sparse(hidden, idx, use_cuda_kernel=False)
    assert delta_torch.shape == hidden.shape
    assert torch.isfinite(delta_torch).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_gate_matches_torch_topk_smoke():
    if not triton_sca_gate_available():
        pytest.skip("Triton not available")

    torch.manual_seed(0)
    cfg = _config(top_k=3, inhibition_lambda=0.0, use_cuda=True)
    centers = build_block_centers(cfg).to(device="cuda", dtype=torch.float32)
    query = torch.randn(32, 3, device="cuda", dtype=torch.float32)

    idx_torch, score_torch = compute_active_blocks_torch(query, centers, cfg)
    idx_cuda, score_cuda = compute_active_blocks(
        query=query,
        block_centers=centers,
        config=cfg,
        refractory_until=None,
        step=-1,
        decode_mode=False,
        inhibition_matrix=None,
        use_cuda_kernel=True,
        cuda_kernels=None,
    )
    assert idx_cuda.shape == idx_torch.shape
    assert score_cuda.shape == score_torch.shape
    assert torch.equal(idx_cuda.cpu(), idx_torch.cpu())
    assert torch.isfinite(score_cuda).all()
