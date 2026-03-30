import os
import sys
import types

import pytest
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from gqa_taylor_ssd import (  # noqa: E402
    GQATaylorSSDSelfAttention,
    TaylorSSDLayerCache,
)
from neuroplastic_llama_gqa_mamba import NeuroplasticLlama  # noqa: E402


class _ToyAttention(nn.Module):
    def __init__(self, num_heads: int = 2, num_key_value_heads: int = 1, head_dim: int = 4) -> None:
        super().__init__()
        hidden_size = int(num_heads * head_dim)
        kv_size = int(num_key_value_heads * head_dim)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.num_heads = int(num_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)


class _Layer(nn.Module):
    def __init__(self, self_attn: nn.Module) -> None:
        super().__init__()
        self.self_attn = self_attn


class _ModelWrapper(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _RootModel(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.model = _ModelWrapper(layers)


def test_taylor_feature_map_is_deterministic():
    torch.manual_seed(0)
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=4)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source,
        layer_idx=0,
        order=2,
        symmetric_quadratic=True,
        eps=1e-6,
        feature_map="taylor",
    )
    hidden = torch.randn(2, 5, 8)
    out_a, _ = attn(hidden_states=hidden, use_cache=False)
    out_b, _ = attn(hidden_states=hidden, use_cache=False)
    assert torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6)
    assert int(attn.feature_dim) == 1 + 4 + (4 * 4)


def test_hybrid_performer_is_default_and_uses_configured_feature_dim():
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=4)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(source_attn=source, layer_idx=0)
    assert attn.config.feature_map == "hybrid_performer"
    assert int(attn.config.local_window) == 64
    assert int(attn.feature_dim) == 64


def test_elu_feature_map_has_smaller_feature_dim():
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=4)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source, layer_idx=0, order=2, eps=1e-6, feature_map="elu"
    )
    assert int(attn.feature_dim) == 4  # head_dim only, not 1+4+16


def test_elu_feature_map_output_is_not_constant_across_positions():
    torch.manual_seed(42)
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=4)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source, layer_idx=0, order=2, eps=1e-6, feature_map="elu"
    )
    # 64 distinct hidden states — output must not collapse to identical vectors
    hidden = torch.randn(1, 64, 8)
    out, _ = attn(hidden_states=hidden, use_cache=False)
    assert not torch.allclose(out[:, 0, :], out[:, 32, :], atol=1e-3), \
        "ELU attention output collapsed to constant mean — mean-value pooling still active"


def test_hybrid_performer_matches_exact_causal_softmax_within_local_window():
    torch.manual_seed(123)
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=4)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source,
        layer_idx=0,
        feature_map="hybrid_performer",
        local_window=8,
        feature_dim=16,
    )
    hidden = torch.randn(1, 5, 8)
    out, _ = attn(hidden_states=hidden, use_cache=False)

    q = attn.q_proj(hidden).view(1, 5, attn.layout.num_attention_heads, attn.layout.head_dim)
    k = attn.k_proj(hidden).view(1, 5, attn.layout.num_key_value_heads, attn.layout.head_dim)
    v = attn.v_proj(hidden).view(1, 5, attn.layout.num_key_value_heads, attn.layout.head_dim)
    manual_steps = []
    scale = float(attn.layout.head_dim) ** 0.5
    for step_idx in range(hidden.shape[1]):
        head_contexts = []
        for head_idx in range(attn.layout.num_attention_heads):
            kv_idx = int(attn.query_to_kv[head_idx])
            scores = torch.einsum(
                "bd,btd->bt",
                q[:, step_idx, head_idx, :].float(),
                k[:, : step_idx + 1, kv_idx, :].float(),
            ) / scale
            weights = torch.softmax(scores, dim=-1)
            ctx = torch.einsum("bt,btd->bd", weights, v[:, : step_idx + 1, kv_idx, :].float())
            head_contexts.append(ctx)
        manual_steps.append(torch.stack(head_contexts, dim=1).to(dtype=hidden.dtype))
    manual_context = torch.stack(manual_steps, dim=1)
    manual_out = attn.o_proj(manual_context.reshape(1, hidden.shape[1], -1))
    assert torch.allclose(out, manual_out, atol=1e-5, rtol=1e-5)


def test_taylor_recurrent_decode_matches_full_prefix_recompute():
    torch.manual_seed(1)
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=4)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source,
        layer_idx=0,
        order=2,
        symmetric_quadratic=False,
        eps=1e-6,
    )
    hidden = torch.randn(1, 4, 8)
    prefix = hidden[:, :3, :]
    step = hidden[:, 3:, :]

    seed_cache = TaylorSSDLayerCache(
        state_S=torch.zeros((1, 1, attn.feature_dim, 4), dtype=torch.float32),
        state_z=torch.zeros((1, 1, attn.feature_dim), dtype=torch.float32),
        local_k=torch.zeros((1, 1, attn.config.local_window, 4), dtype=torch.float32),
        local_v=torch.zeros((1, 1, attn.config.local_window, 4), dtype=torch.float32),
        local_valid=torch.zeros((1, 1, attn.config.local_window), dtype=torch.bool),
        cache_pos=0,
        seen_tokens=0,
    )
    _prefill_out, _attn_weights, cache = attn(
        hidden_states=prefix,
        use_cache=True,
        past_key_value=seed_cache,
    )
    step_out, _attn_weights, cache_next = attn(
        hidden_states=step,
        use_cache=True,
        past_key_value=cache,
    )
    full_out, _ = attn(hidden_states=hidden, use_cache=False)
    assert cache_next.seen_tokens == 4
    assert torch.isfinite(cache_next.state_S).all()
    assert torch.isfinite(cache_next.state_z).all()
    assert torch.allclose(step_out[:, -1, :], full_out[:, -1, :], atol=1e-4, rtol=1e-4)


def test_taylor_forward_returns_present_cache_when_use_cache_enabled():
    torch.manual_seed(7)
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=4)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source,
        layer_idx=0,
        order=2,
        symmetric_quadratic=True,
        eps=1e-6,
    )
    prefix = torch.randn(1, 3, 8)
    step = torch.randn(1, 1, 8)

    _prefix_out, _attn_weights, present = attn(hidden_states=prefix, use_cache=True, past_key_value=None)
    assert isinstance(present, TaylorSSDLayerCache)
    assert int(present.seen_tokens) == 3

    step_out, _attn_weights, present_next = attn(hidden_states=step, use_cache=True, past_key_value=present)
    assert isinstance(present_next, TaylorSSDLayerCache)
    assert int(present_next.seen_tokens) == 4
    assert tuple(present_next.local_k.shape) == (1, 1, attn.config.local_window, 4)
    assert tuple(present_next.local_v.shape) == (1, 1, attn.config.local_window, 4)
    assert tuple(present_next.local_valid.shape) == (1, 1, attn.config.local_window)
    assert torch.isfinite(step_out).all()


def test_taylor_model_managed_cache_signal_uses_runtime_cache_without_present_tuple():
    torch.manual_seed(8)
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=4)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(source_attn=source, layer_idx=0, local_window=8)
    hidden = torch.randn(1, 4, 8)
    prefix = hidden[:, :3, :]
    step = hidden[:, 3:, :]
    foreign_cache = object()

    prefix_out = attn(hidden_states=prefix, use_cache=False, past_key_value=foreign_cache)
    step_out = attn(hidden_states=step, use_cache=False, past_key_value=foreign_cache)
    full_out, _ = attn(hidden_states=hidden, use_cache=False)

    assert isinstance(prefix_out, tuple) and len(prefix_out) == 2
    assert isinstance(step_out, tuple) and len(step_out) == 2
    assert torch.allclose(step_out[0][:, -1, :], full_out[:, -1, :], atol=1e-4, rtol=1e-4)


def test_hybrid_performer_cache_evicts_into_recurrent_tail_after_window():
    torch.manual_seed(9)
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=4)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source,
        layer_idx=0,
        feature_map="hybrid_performer",
        local_window=2,
        feature_dim=8,
    )
    hidden = torch.randn(1, 4, 8)
    _out, _attn_weights, present = attn(hidden_states=hidden, use_cache=True)
    assert isinstance(present, TaylorSSDLayerCache)
    assert int(present.seen_tokens) == 4
    assert int(present.cache_pos) == 0
    assert present.local_valid.all()
    assert torch.count_nonzero(present.state_S).item() > 0
    assert torch.count_nonzero(present.state_z).item() > 0


def test_taylor_feature_map_matches_order2_kernel_symmetric():
    torch.manual_seed(11)
    head_dim = 4
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=head_dim)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source,
        layer_idx=0,
        order=2,
        symmetric_quadratic=True,
        eps=1e-6,
        feature_map="taylor",
    )
    q = torch.randn(3, 2, head_dim)
    k = torch.randn(3, 2, head_dim)
    phi_q = attn._phi_q(q)
    phi_k = attn._phi_k(k)
    approx = (phi_q * phi_k).sum(dim=-1)
    dot = (q * k).sum(dim=-1)
    target = 1.0 + (dot / (head_dim ** 0.5)) + ((dot * dot) / (2.0 * head_dim))
    assert torch.allclose(approx, target, atol=1e-5, rtol=1e-5)


def test_taylor_feature_map_matches_order2_kernel_asymmetric():
    torch.manual_seed(12)
    head_dim = 4
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=head_dim)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source,
        layer_idx=0,
        order=2,
        symmetric_quadratic=False,
        eps=1e-6,
        feature_map="taylor",
    )
    q = torch.randn(3, 2, head_dim)
    k = torch.randn(3, 2, head_dim)
    phi_q = attn._phi_q(q)
    phi_k = attn._phi_k(k)
    approx = (phi_q * phi_k).sum(dim=-1)
    dot = (q * k).sum(dim=-1)
    target = 1.0 + (dot / (head_dim ** 0.5)) + ((dot * dot) / (2.0 * head_dim))
    assert torch.allclose(approx, target, atol=1e-5, rtol=1e-5)


def test_taylor_feature_map_with_temperature():
    torch.manual_seed(13)
    head_dim = 4
    temp = 10.0
    source = _ToyAttention(num_heads=2, num_key_value_heads=1, head_dim=head_dim)
    attn = GQATaylorSSDSelfAttention.from_llama_attention(
        source_attn=source,
        layer_idx=0,
        order=2,
        symmetric_quadratic=True,
        taylor_temperature=temp,
        eps=1e-6,
        feature_map="taylor",
    )
    q = torch.randn(3, 2, head_dim)
    k = torch.randn(3, 2, head_dim)
    phi_q = attn._phi_q(q)
    phi_k = attn._phi_k(k)
    approx = (phi_q * phi_k).sum(dim=-1)
    dot = (q * k).sum(dim=-1)
    
    # Target: 1 + x/T + (x/T)^2 / 2
    x = dot / (head_dim ** 0.5)
    target = 1.0 + (x / temp) + ((x / temp) ** 2 / 2.0)
    assert torch.allclose(approx, target, atol=1e-5, rtol=1e-5)


def test_taylor_backend_subset_replacement_is_explicit():
    torch.manual_seed(2)
    fake = object.__new__(NeuroplasticLlama)
    nn.Module.__init__(fake)
    layers = [_Layer(_ToyAttention()), _Layer(_ToyAttention()), _Layer(_ToyAttention())]
    fake.model = _RootModel(layers)
    fake.hidden_size = 8
    fake.config = types.SimpleNamespace(num_attention_heads=2, num_key_value_heads=1, head_dim=4)

    NeuroplasticLlama._replace_attention_with_taylor_ssd(
        fake,
        layer_indices=[1],
        order=2,
        symmetric_quadratic=True,
        eps=1e-6,
        verbose=False,
    )

    assert isinstance(fake.model.model.layers[1].self_attn, GQATaylorSSDSelfAttention)
    assert not isinstance(fake.model.model.layers[0].self_attn, GQATaylorSSDSelfAttention)
    assert not isinstance(fake.model.model.layers[2].self_attn, GQATaylorSSDSelfAttention)


def test_attention_backend_mutual_exclusion_validation_rejects_multi_enable():
    NeuroplasticLlama._validate_attention_backend_selection(
        attention_hybrid_enabled=False,
        attention_gqa_mamba_enabled=False,
        attention_taylor_ssd_enabled=False,
    )
    with pytest.raises(ValueError):
        NeuroplasticLlama._validate_attention_backend_selection(
            attention_hybrid_enabled=True,
            attention_gqa_mamba_enabled=False,
            attention_taylor_ssd_enabled=True,
        )


def test_pack_restore_bypasses_taylor_cache_kv_only_logic():
    fake = object.__new__(NeuroplasticLlama)
    nn.Module.__init__(fake)
    fake.model = nn.Linear(1, 1, bias=False)
    fake.sparse_attention_config = types.SimpleNamespace(enabled=False)
    fake._sparse_attention_runtime = None
    fake._tier_cache_compressor = None
    fake.kv_int4_quantization = False
    fake.kv_cpu_offload = True
    fake._runtime_flags = {"kv_optimized_generation": True, "bounded_context_enabled": False}

    taylor_cache = TaylorSSDLayerCache(
        state_S=torch.randn(1, 1, 3, 2, dtype=torch.float32),
        state_z=torch.randn(1, 1, 3, dtype=torch.float32),
        local_k=torch.randn(1, 1, 4, 2, dtype=torch.float32),
        local_v=torch.randn(1, 1, 4, 2, dtype=torch.float32),
        local_valid=torch.ones(1, 1, 4, dtype=torch.bool),
        cache_pos=1,
        seen_tokens=3,
    )
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    packed = NeuroplasticLlama._pack_past_key_values(fake, (taylor_cache, (k, v)))
    assert isinstance(packed[0], TaylorSSDLayerCache)
    restored = NeuroplasticLlama._restore_past_key_values(fake, packed)
    assert isinstance(restored[0], TaylorSSDLayerCache)
    assert torch.allclose(restored[1][0], k, atol=1e-6, rtol=1e-6)
    assert torch.allclose(restored[1][1], v, atol=1e-6, rtol=1e-6)


def test_taylor_cache_policy_disables_optimized_features_in_v1():
    fake = object.__new__(NeuroplasticLlama)
    nn.Module.__init__(fake)
    fake.kv_int4_quantization = True
    fake.kv_cpu_offload = True
    fake.bounded_context_config = types.SimpleNamespace(enabled=True)
    fake.sparse_attention_config = types.SimpleNamespace(enabled=True, strict_fully_sparse=False)
    fake._tier_policy = None
    fake._tier_cache_compressor = None
    fake._tier_mask_bundle = None
    fake.attention_hybrid_enabled = False
    fake.attention_hybrid_force_no_cache = True
    fake.attention_gqa_mamba_enabled = False
    fake.attention_taylor_ssd_enabled = True
    fake.attention_taylor_force_disable_optimized_cache = True

    NeuroplasticLlama._refresh_runtime_flags(fake)
    assert fake.kv_int4_quantization is False
    assert fake.kv_cpu_offload is False
    assert fake.bounded_context_config.enabled is False
    assert fake.sparse_attention_config.enabled is False
    assert NeuroplasticLlama._should_disable_generation_cache(fake, True) is False
