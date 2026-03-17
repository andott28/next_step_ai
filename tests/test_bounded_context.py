import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from bounded_context import (
    BoundedContextConfig,
    TieredAttentionMaskBundle,
    TieredCacheCompressor,
    TieredContextPolicy,
    CodeRepoMemoryIndex,
    gqa_query_head_to_kv_group,
    token_to_page_index,
    page_index_to_token_span,
)


def test_memory_estimate_matches_plan_math():
    cfg = BoundedContextConfig(
        enabled=True,
        sink_tokens=8,
        local_window_tokens=10_000,
        global_window_tokens=128_000,
        global_start_layer=84,
        global_group_id=0,
        kv_cache_bits=4,
        vram_budget_gib=3.5,
    )
    estimate = cfg.estimate_memory(num_hidden_layers=126, num_key_value_heads=16, head_dim=128)

    assert estimate.bytes_per_token_all_layers == 258_048
    assert 2.3 <= (estimate.local_cache_bytes / (1024.0**3)) <= 2.5
    assert 0.6 <= (estimate.global_cache_bytes / (1024.0**3)) <= 0.7
    assert estimate.total_gib <= 3.5


def test_tier_policy_selection_local_vs_global():
    cfg = BoundedContextConfig(
        enabled=True,
        sink_tokens=4,
        local_window_tokens=16,
        global_window_tokens=64,
        global_start_layer=1,
    )
    policy = TieredContextPolicy(
        config=cfg,
        num_hidden_layers=2,
        num_key_value_heads=4,
        num_attention_heads=8,
    )

    layer0 = policy.select_layer_positions(layer_idx=0, seq_len=100, device=torch.device("cpu"))
    layer1 = policy.select_layer_positions(layer_idx=1, seq_len=100, device=torch.device("cpu"))

    assert not layer0.uses_global
    assert layer0.retained_positions.numel() == 20  # sink(4) + local(16)
    assert layer1.uses_global
    assert layer1.retained_positions.numel() == 68  # sink(4) + global(64)


def test_tier_cache_compressor_truncates_per_layer():
    cfg = BoundedContextConfig(
        enabled=True,
        sink_tokens=4,
        local_window_tokens=16,
        global_window_tokens=64,
        global_start_layer=1,
    )
    policy = TieredContextPolicy(
        config=cfg,
        num_hidden_layers=2,
        num_key_value_heads=4,
        num_attention_heads=8,
    )
    compressor = TieredCacheCompressor(policy)

    k0 = torch.randn(1, 4, 100, 8)
    v0 = torch.randn(1, 4, 100, 8)
    k1 = torch.randn(1, 4, 100, 8)
    v1 = torch.randn(1, 4, 100, 8)
    compressed, meta = compressor.compress_legacy_cache(((k0, v0), (k1, v1)))

    assert len(compressed) == 2
    assert compressed[0][0].shape[-2] == 20
    assert compressed[1][0].shape[-2] == 68
    assert meta.layers[0].retained_positions.numel() == 20
    assert meta.layers[1].retained_positions.numel() == 68


def test_tier_attention_mask_masks_non_global_heads():
    cfg = BoundedContextConfig(
        enabled=True,
        sink_tokens=2,
        local_window_tokens=5,
        global_window_tokens=20,
        global_start_layer=1,
        global_group_id=0,
    )
    policy = TieredContextPolicy(
        config=cfg,
        num_hidden_layers=2,
        num_key_value_heads=4,
        num_attention_heads=8,
    )
    bundle = TieredAttentionMaskBundle(policy)

    base_mask = torch.zeros((1, 1, 1, 20), dtype=torch.float32)
    layer_mask = bundle.build_layer_mask(
        layer_idx=1,
        base_mask=base_mask,
        kv_length=20,
        num_attention_heads=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert layer_mask is not None
    assert layer_mask.shape == (1, 8, 1, 20)

    # Heads mapped to KV-group 0 keep global visibility (no -inf terms).
    assert torch.isfinite(layer_mask[0, 0, 0]).all()
    assert torch.isfinite(layer_mask[0, 1, 0]).all()

    # Heads in other KV groups should have disallowed middle tokens.
    blocked = layer_mask[0, 2, 0, 2:15]
    assert torch.all(blocked <= (torch.finfo(blocked.dtype).min / 2.0))


def test_repo_memory_index_build_and_retrieve(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text(
        "import json\n\n"
        "def parse_config(path):\n"
        "    with open(path, 'r', encoding='utf-8') as f:\n"
        "        return json.load(f)\n",
        encoding="utf-8",
    )
    (src / "README.md").write_text("This module parses config files.", encoding="utf-8")

    index = CodeRepoMemoryIndex(str(src), chunk_min_tokens=10, chunk_max_tokens=40)
    built = index.build()
    assert built > 0

    hits = index.retrieve("parse config json", top_k=3)
    assert len(hits) > 0
    assert any("a.py" in hit.path for hit in hits)


def test_gqa_head_group_mapping():
    assert gqa_query_head_to_kv_group(0, num_attention_heads=8, num_key_value_heads=2) == 0
    assert gqa_query_head_to_kv_group(3, num_attention_heads=8, num_key_value_heads=2) == 0
    assert gqa_query_head_to_kv_group(4, num_attention_heads=8, num_key_value_heads=2) == 1
    assert gqa_query_head_to_kv_group(7, num_attention_heads=8, num_key_value_heads=2) == 1


def test_page_index_roundtrip():
    idx = token_to_page_index(token_index=513, page_size_tokens=256)
    assert idx == 2
    start, end = page_index_to_token_span(page_index=idx, page_size_tokens=256)
    assert start == 512
    assert end == 768


def test_strict_sparse_runtime_validation():
    cfg = BoundedContextConfig(enabled=True)
    cfg.validate_sparse_attention_runtime(
        strict_fully_sparse=True,
        sca_use_cuda=True,
        sca_spmm_impl="cuda_spmm",
        fast_fallback_threshold=0.0,
        disable_ssd_fetch_in_decode=True,
    )
    try:
        cfg.validate_sparse_attention_runtime(
            strict_fully_sparse=True,
            sca_use_cuda=False,
            sca_spmm_impl="cuda_spmm",
            fast_fallback_threshold=0.0,
            disable_ssd_fetch_in_decode=True,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised is True
