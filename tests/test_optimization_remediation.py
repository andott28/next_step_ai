from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from llama3_neuroplastic import basis_fitting
from llama3_neuroplastic.experiments import benchmark as benchmark_mod
from llama3_neuroplastic.experiments import run_streaming_inference as cli_mod
from llama3_neuroplastic.experiments.eval_perplexity import _run_forward_only
from llama3_neuroplastic.experiments.init_learned_basis_from_dense_mlp import (
    _parse_args,
    _save_basis_resume,
    group_tokenized_texts,
)
from llama3_neuroplastic.token_posting_archive import TokenPostingArchive


def _archive() -> TokenPostingArchive:
    archive = TokenPostingArchive(
        retrieval_layers=[0],
        num_kv_groups=1,
        head_dim=1,
        basis_rank=1,
        ring_size=1,
        num_sinks=0,
        archive_capacity=3,
        token_topk=1,
        r_query=1,
        candidates=3,
        device=torch.device("cpu"),
    )
    archive.load_basis(
        0,
        0,
        basis=np.asarray([[1.0]], dtype=np.float32),
        idf=np.asarray([1.0], dtype=np.float32),
        key_mean=np.asarray([0.0], dtype=np.float32),
    )
    return archive


def test_token_posting_archive_wraparound_returns_storage_slots() -> None:
    archive = _archive()
    for value in range(6):
        kv = torch.tensor([[float(value + 1)]], dtype=torch.float16)
        archive.append_token(0, value, kv, kv)

    assert archive.archive_count[0] == 5
    assert archive.archive_generation[0].tolist() == [3, 4, 2]

    candidates = archive.select_candidates(
        0,
        0,
        np.asarray([1.0], dtype=np.float32),
        step=1,
        M=3,
    )

    assert candidates.size > 0
    assert set(candidates.tolist()).issubset({0, 1, 2})
    assert archive.archive_k_cpu[0][candidates, 0, :].shape[0] == candidates.size


def test_token_posting_archive_filters_stale_posting_generation() -> None:
    archive = _archive()
    kv = torch.tensor([[1.0]], dtype=torch.float16)
    archive.append_token(0, 0, kv, kv)
    archive.append_token(0, 1, kv, kv)

    head = archive._post_head[0][0, 0]
    archive._post_tok[0][0, 0, head] = 0
    archive._post_gen[0][0, 0, head] = -1
    archive._post_coeff[0][0, 0, head] = 127
    archive._post_scale[0][0, 0, head] = 1_000_000.0
    archive._post_head[0][0, 0] = (head + 1) % archive.archive_capacity
    archive._post_count[0][0, 0] += 1

    touched = archive._probe(
        0,
        0,
        np.asarray([1.0], dtype=np.float32),
        step=2,
    )

    assert touched == [0]
    assert float(archive._score_buf[0][0][0]) < 10.0


def test_contract_reports_share_canonical_decode_tok_s_schema() -> None:
    payload = {
        "new_tokens": 2,
        "decode_tok_s": 3.31,
        "decode_tokens_per_second": 3.31,
        "mean_decode_ms_per_token": 302.0,
        "decode_avg_mb_per_layer": 22.5,
        "runtime_status": {
            "num_layers": 126,
            "lm_head_on_gpu": True,
            "lm_head_mode": "gpu_nf4",
            "decode_backend": "single_kernel_sparse_decode_sm75",
            "attn_backend_decode": "compact_sparse_v1",
            "compact_sparse_attention_steps": 8,
            "vram_hot_cache_live_calibrated": True,
            "decode_mlp_cold_blocks_streamed": 0,
            "decode_down_cold_blocks_streamed": 0,
            "sparse_attention_layers": 126,
            "sparse_kv_enabled_for_decode": True,
        },
        "traffic": {"decode": {"layer_visits": 252}},
        "traffic_report": {"decode": {"layer_visits": 252}},
    }

    cli_report = cli_mod._build_throughput_contract_report(dict(payload))
    benchmark_report = benchmark_mod._build_throughput_contract_report(dict(payload))

    assert cli_report["checks"][0]["name"] == "decode_tok_s"
    assert benchmark_report["checks"][0]["name"] == "decode_tok_s"
    assert cli_report["passed"] is True
    assert benchmark_report["passed"] is True


def test_incremental_pca_merges_final_short_batch(monkeypatch) -> None:
    seen_shapes: list[tuple[int, int]] = []

    class FakeIncrementalPCA:
        def __init__(self, n_components: int, batch_size: int) -> None:
            self.n_components = int(n_components)
            self.batch_size = int(batch_size)
            self.components_ = np.eye(2, dtype=np.float32)
            self.explained_variance_ratio_ = np.asarray([0.75, 0.20], dtype=np.float32)

        def partial_fit(self, chunk: np.ndarray) -> None:
            seen_shapes.append(tuple(chunk.shape))

    monkeypatch.setattr(basis_fitting, "IncrementalPCA", FakeIncrementalPCA)
    y = torch.arange(18, dtype=torch.float32).view(9, 2)

    _basis, coeff, explained = basis_fitting._fit_incremental_pca(y, rank_eff=2, batch_rows=4)

    assert seen_shapes == [(4, 2), (5, 2)]
    assert tuple(coeff.shape) == (9, 2)
    assert explained > 0.9


def test_group_tokenized_texts_matches_flattening_behavior() -> None:
    grouped = group_tokenized_texts({"input_ids": [[1, 2], [3], [4, 5, 6]]}, seq_len=3)

    assert grouped == {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 1]],
    }


def test_resume_buffers_are_opt_in(tmp_path: Path) -> None:
    resume_path = tmp_path / "basis.resume.pt"
    layer_x = {0: [torch.ones((1, 2))]}
    layer_y = {0: [torch.ones((1, 2))]}

    _save_basis_resume(
        resume_path=resume_path,
        layer_states={"0": {"score_weight": torch.ones((2, 2))}},
        stats={"0": {"explained_variance_ratio": 0.9}},
        layer_x=layer_x,
        layer_y=layer_y,
        selected_layers=[0],
        include_buffers=False,
        layer_kv_x={0: [torch.ones((1, 2))]},
        layer_kv_rows={0: 1},
    )

    payload = torch.load(resume_path, map_location="cpu")
    assert "layer_x" not in payload
    assert "layer_y" not in payload
    assert "layer_kv_x" not in payload
    assert payload["layer_kv_rows"] == {0: 1}


def test_resume_save_buffers_cli_default_is_false(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--model-name", "dummy/model", "--output-path", "results/out.pt"],
    )

    args = _parse_args()

    assert args.resume_save_buffers is False


def test_run_forward_only_uses_runtime_scoring_api() -> None:
    class DummyRuntime:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.called = False

        def score_window_logits(self, input_ids: torch.LongTensor) -> torch.Tensor:
            self.called = True
            batch, seq = input_ids.shape
            logits = torch.zeros((batch, seq, 5), dtype=torch.float32)
            logits[:, :, 3] = 1.0
            return logits

    runtime = DummyRuntime()
    out = _run_forward_only(runtime, torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert runtime.called is True
    assert out is not None
    assert tuple(out.shape) == (1, 3, 5)
