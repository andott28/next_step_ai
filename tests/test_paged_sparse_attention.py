import os
import sys

import torch
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from paged_sparse_attention import (
    LongRangePageArchive,
    LongRangeSummaryTable,
    SparseAttentionConfig,
    SparseAttentionRuntime,
    TwoStagePageRetriever,
)


def test_page_archive_roundtrip_int4_and_fp16():
    for dtype_name in ("int4", "fp16"):
        cfg = SparseAttentionConfig(enabled=True, page_size_tokens=2, archive_cpu_dtype=dtype_name)
        archive = LongRangePageArchive(config=cfg, head_dim=4)
        for i in range(4):
            k = torch.tensor([i, i + 1, i + 2, i + 3], dtype=torch.float32)
            v = torch.tensor([i + 0.5, i + 1.5, i + 2.5, i + 3.5], dtype=torch.float32)
            archive.append_token(layer_idx=0, head_idx=0, key_token=k, value_token=v)
        archive.flush_partial_page(layer_idx=0, head_idx=0)
        assert archive.num_pages(layer_idx=0, head_idx=0) == 2
        keys, values, _ = archive.fetch_pages(
            layer_idx=0,
            head_idx=0,
            page_indices=[0, 1],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert keys.shape == (4, 4)
        assert values.shape == (4, 4)


def test_summary_routing_is_deterministic_topk():
    cfg = SparseAttentionConfig(enabled=True, page_size_tokens=1, retrieval_top_k_pages=2)
    archive = LongRangePageArchive(config=cfg, head_dim=2)
    table = LongRangeSummaryTable()
    retriever = TwoStagePageRetriever(config=cfg, archive=archive, table=table)
    for vec in ([1.0, 0.0], [0.1, 0.9], [2.0, 0.0], [0.0, 1.0]):
        t = torch.tensor(vec, dtype=torch.float32)
        archive.append_token(layer_idx=0, head_idx=0, key_token=t, value_token=t)
        archive.flush_partial_page(layer_idx=0, head_idx=0)
    summaries = archive.get_summaries(layer_idx=0, head_idx=0, device=torch.device("cpu"))
    table.update(layer_idx=0, head_idx=0, summaries=summaries)
    pages, _, _, _ = retriever.retrieve(
        layer_idx=0,
        head_idx=0,
        query_vec=torch.tensor([1.0, 0.0], dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert pages[0] in (0, 2)
    assert len(pages) == 2


def test_sparse_attention_runtime_fetches_selected_pages_only():
    cfg = SparseAttentionConfig(
        enabled=True,
        local_window_tokens=2,
        sink_tokens=1,
        page_size_tokens=2,
        retrieval_top_k_pages=1,
        retrieval_head_group_ids=(0,),
        retrieval_start_layer=0,
        archive_cpu_dtype="fp16",
    )
    runtime = SparseAttentionRuntime(config=cfg, num_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=4)

    for step in range(6):
        seq = step + 1
        k = torch.randn(1, 2, seq, 4)
        v = torch.randn(1, 2, seq, 4)
        compressed, stats = runtime.update_and_compress_cache(legacy_cache=((k, v),), sparse_mlp_diagnostics={"layers": []})
        assert len(compressed) == 1
        assert compressed[0][0].shape[-2] <= seq
        assert isinstance(stats.bytes_cpu_to_gpu, int)
    diag = runtime.diagnostics()
    assert diag["steps"] == 6
    assert "mean_selected_pages_per_step" in diag


def test_strict_sparse_rejects_fast_fallback():
    cfg = SparseAttentionConfig(enabled=True, strict_fully_sparse=True)
    runtime = SparseAttentionRuntime(config=cfg, num_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=4)
    k = torch.randn(1, 2, 2, 4)
    v = torch.randn(1, 2, 2, 4)
    with pytest.raises(RuntimeError, match="fallback"):
        runtime.update_and_compress_cache(
            legacy_cache=((k, v),),
            sparse_mlp_diagnostics={"layers": [{"dense_fallback_rate": 0.2}]},
        )
