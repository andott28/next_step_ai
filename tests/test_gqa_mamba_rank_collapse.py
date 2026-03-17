import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from gqa_mamba_rank_collapse import (  # noqa: E402
    EightGBExecutionBudget,
    GQALayout,
    GQAMambaRankCollapseBlock,
    TokenLatencyEstimate,
    collapse_llama_gqa_attention_layer,
    sync_alignment_loss,
    transplant_svd_group_to_mamba_projection,
)


def _build_correlated_attention_weights(layout: GQALayout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(5)
    hidden = layout.hidden_size
    group_heads = layout.query_heads_per_group
    gwidth = group_heads * layout.head_dim

    w_q = torch.empty(hidden, hidden)
    w_k = torch.empty(layout.num_key_value_heads * layout.head_dim, hidden)
    w_v = torch.empty_like(w_k)
    w_o = torch.empty(hidden, hidden)

    for g in range(layout.num_key_value_heads):
        q_base = torch.randn(layout.head_dim, hidden)
        q_heads = [q_base + 0.03 * torch.randn_like(q_base) for _ in range(group_heads)]
        q_group = torch.cat(q_heads, dim=0)

        k_group = q_base + 0.05 * torch.randn_like(q_base)
        v_group = q_base + 0.05 * torch.randn_like(q_base)
        o_base = torch.randn(hidden, layout.head_dim)
        o_heads = [o_base + 0.03 * torch.randn_like(o_base) for _ in range(group_heads)]
        o_group = torch.cat(o_heads, dim=1)

        q_start = g * gwidth
        q_end = q_start + gwidth
        kv_start = g * layout.head_dim
        kv_end = kv_start + layout.head_dim
        w_q[q_start:q_end, :] = q_group
        w_k[kv_start:kv_end, :] = k_group
        w_v[kv_start:kv_end, :] = v_group
        w_o[:, q_start:q_end] = o_group

    return w_q, w_k, w_v, w_o


def test_rank_collapse_reaches_target_variance():
    layout = GQALayout(num_attention_heads=8, num_key_value_heads=2, head_dim=8)
    w_q, w_k, w_v, w_o = _build_correlated_attention_weights(layout)
    collapsed = collapse_llama_gqa_attention_layer(
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        layout=layout,
        variance_threshold=0.90,
    )
    assert len(collapsed) == layout.num_key_value_heads
    for group in collapsed:
        assert group.explained_variance >= 0.90
        reconstructed = group.reconstruct_group_weights()
        assert reconstructed["w_q"].shape[1] == layout.hidden_size
        assert reconstructed["w_o"].shape[0] == layout.hidden_size


def test_transplant_and_runtime_block_shape():
    layout = GQALayout(num_attention_heads=8, num_key_value_heads=2, head_dim=8)
    w_q, w_k, w_v, w_o = _build_correlated_attention_weights(layout)
    collapsed = collapse_llama_gqa_attention_layer(w_q, w_k, w_v, w_o, layout=layout, target_rank=6)
    projections = [
        transplant_svd_group_to_mamba_projection(
            result=g,
            query_heads_per_group=layout.query_heads_per_group,
            head_dim=layout.head_dim,
            state_dim=layout.head_dim,
        )
        for g in collapsed
    ]

    block = GQAMambaRankCollapseBlock(projections)
    x = torch.randn(3, 5, layout.hidden_size)
    y = block(x)
    assert y.shape == x.shape


def test_sync_alignment_loss_modes():
    torch.manual_seed(3)
    student = torch.randn(2, 4, 16)
    teacher = student + 0.01 * torch.randn_like(student)
    mse = sync_alignment_loss(student, teacher, mode="mse")
    cos = sync_alignment_loss(student, teacher, mode="cosine")
    assert torch.isfinite(mse)
    assert torch.isfinite(cos)
    assert float(mse.item()) >= 0.0
    assert float(cos.item()) >= 0.0


def test_8gb_execution_math_matches_target_band():
    budget = EightGBExecutionBudget()
    assert budget.fits(vram_gb=8.0)
    assert abs(budget.total_vram_gb() - 8.0) < 1e-6

    latency = TokenLatencyEstimate(pcie_bandwidth_gbps=15.0, streamed_sparse_mlp_gb=1.9, on_gpu_compute_seconds=0.03)
    tps = latency.tokens_per_second()
    assert 6.0 <= tps <= 8.0
