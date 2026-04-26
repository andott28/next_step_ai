from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from llama3_neuroplastic.experiments import benchmark as benchmark_mod
from llama3_neuroplastic.experiments import run_streaming_inference as cli_mod
from llama3_neuroplastic.experiments import streaming_llama_runtime as runtime_mod
from llama3_neuroplastic.experiments.runtime import lm_head as lm_head_mod
from llama3_neuroplastic.experiments.runtime.lm_head import RuntimeLmHeadMixin
from llama3_neuroplastic.experiments.runtime.session import RuntimeSessionMixin
from llama3_neuroplastic import triton_sparse_mlp as triton_sparse_mlp_mod


class _DummyLmHead(RuntimeLmHeadMixin):
    pass


class _DummySession(RuntimeSessionMixin):
    pass


def test_cli_parser_accepts_throughput_contract_and_profile_flags() -> None:
    parser = cli_mod._build_arg_parser()

    args = parser.parse_args(
        [
            "--model-name",
            "dummy/model",
            "--prompt",
            "hello",
            "--throughput-contract",
            "strict",
            "--profile-decode",
            "--profile-max-steps",
            "2",
        ]
    )

    assert args.throughput_contract == "strict"
    assert args.profile_decode is True
    assert args.profile_max_steps == 2


def test_benchmark_contract_report_passes_when_all_targets_met() -> None:
    report = benchmark_mod._build_throughput_contract_report(
        {
            "new_tokens": 2,
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
            "traffic_report": {
                "decode": {
                    "layer_visits": 252,
                }
            },
        }
    )

    assert report["contract"] == "strict"
    assert report["passed"] is True
    assert report["failed_checks"] == []


def test_benchmark_contract_report_flags_gpu_and_cold_block_failures() -> None:
    report = benchmark_mod._build_throughput_contract_report(
        {
            "new_tokens": 1,
            "decode_tokens_per_second": 0.5,
            "mean_decode_ms_per_token": 2000.0,
            "decode_avg_mb_per_layer": 50.0,
            "runtime_status": {
                "num_layers": 126,
                "lm_head_on_gpu": False,
                "lm_head_mode": "cpu_dense",
                "decode_backend": "fused_sparse_decode_v1",
                "attn_backend_decode": "dense",
                "compact_sparse_attention_steps": 0,
                "vram_hot_cache_live_calibrated": False,
                "decode_mlp_cold_blocks_streamed": 4,
                "decode_down_cold_blocks_streamed": 9,
                "sparse_attention_layers": 120,
                "sparse_kv_enabled_for_decode": False,
            },
            "traffic_report": {
                "decode": {
                    "layer_visits": 100,
                }
            },
        }
    )

    failed = set(report["failed_checks"])
    assert report["passed"] is False
    assert "lm_head_on_gpu" in failed
    assert "decode_mlp_cold_blocks_streamed" in failed
    assert "decode_down_cold_blocks_streamed" in failed
    assert "decode_layer_visits" in failed


def test_lm_head_status_reports_gpu_dense_mode() -> None:
    runtime = _DummyLmHead()
    runtime._materialize_lm_head = True
    runtime._lm_head_weight_name = "lm_head.weight"
    runtime._lm_head_nf4_meta_gpu = None
    runtime._lm_head_weight_gpu = torch.empty((2, 3), dtype=torch.float16)
    runtime._lm_head_weight_cpu = None
    runtime._lm_head_gpu_attempted = True
    runtime._prefer_gpu_lm_head = True
    runtime._lm_head_gpu_last_failure = None

    status = runtime.get_lm_head_status()

    assert status["mode"] == "gpu_dense"
    assert status["on_gpu"] is True
    assert status["weight_name"] == "lm_head.weight"


def test_lm_head_status_reports_gpu_nf4_mode() -> None:
    runtime = _DummyLmHead()
    runtime._materialize_lm_head = True
    runtime._lm_head_weight_name = "lm_head.weight"
    runtime._lm_head_nf4_meta_gpu = {
        "packed_weight": torch.empty((8,), dtype=torch.uint8),
        "absmax": torch.empty((4,), dtype=torch.float32),
        "code": torch.empty((16,), dtype=torch.float32),
    }
    runtime._lm_head_weight_gpu = None
    runtime._lm_head_weight_cpu = None
    runtime._lm_head_gpu_attempted = True
    runtime._prefer_gpu_lm_head = True
    runtime.dtype = torch.float16
    runtime._lm_head_gpu_last_failure = None

    status = runtime.get_lm_head_status()

    assert status["mode"] == "gpu_nf4"
    assert status["on_gpu"] is True
    assert status["weight_name"] == "lm_head.weight"


def test_lm_head_forward_uses_nf4_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _DummyLmHead()
    runtime._materialize_lm_head = True
    runtime._lm_head_nf4_meta_gpu = {
        "packed_weight": torch.empty((8,), dtype=torch.uint8),
        "absmax": torch.empty((4,), dtype=torch.float32),
        "code": torch.empty((16,), dtype=torch.float32),
        "out_features": 5,
        "in_features": 4,
        "quant_block_size": 64,
        "block_size": 2,
        "active_idx": torch.tensor([[0, 1]], dtype=torch.int32),
    }
    runtime._lm_head_weight_gpu = None
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float16

    def _fake_sparse_input(x_flat: torch.Tensor, active_idx: torch.Tensor, **_: object) -> torch.Tensor:
        assert tuple(x_flat.shape) == (1, 4)
        assert tuple(active_idx.shape) == (1, 2)
        return torch.arange(5, dtype=torch.float16).view(1, 5)

    monkeypatch.setattr(lm_head_mod, "triton_sparse_input_linear_4bit", _fake_sparse_input)
    hidden = torch.ones((1, 1, 4), dtype=torch.float16)

    logits = runtime._lm_head_forward(hidden)

    assert tuple(logits.shape) == (1, 1, 5)
    assert logits[0, 0, 4].item() == 4


def test_materialize_lm_head_blocks_dense_gpu_when_not_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _DummyLmHead()
    runtime._lm_head_gpu_attempted = False
    runtime._materialize_lm_head = True
    runtime._prefer_gpu_lm_head = True
    runtime._prefer_gpu_quant_lm_head = True
    runtime._allow_dense_gpu_lm_head = False
    runtime._explicit_gpu_lm_head = False
    runtime._lm_head_gpu_last_failure = ""
    runtime._lm_head_nf4_meta_gpu = None
    runtime._lm_head_weight_gpu = None
    runtime.device = torch.device("cuda")
    runtime.dtype = torch.float16

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_args, **_kwargs: (7, 5))

    def _nf4_fail() -> bool:
        runtime._lm_head_gpu_last_failure = "lm_head_not_nf4"
        return False

    runtime._materialize_lm_head_nf4_on_gpu = _nf4_fail  # type: ignore[method-assign]
    runtime._ensure_lm_head_weight_cpu = lambda: torch.empty((2, 2), dtype=torch.float16)  # type: ignore[method-assign]

    runtime._materialize_lm_head_on_gpu()

    assert runtime._lm_head_gpu_attempted is True
    assert runtime._lm_head_weight_gpu is None
    assert runtime._lm_head_gpu_last_failure == "lm_head_not_nf4"


def test_quantize_dense_lm_head_nf4_cpu_shapes() -> None:
    runtime = _DummyLmHead()
    dense = torch.randn((2, 64), dtype=torch.float16)

    packed, absmax, code, out_features, in_features, quant_block_size = runtime._quantize_dense_lm_head_nf4_cpu(dense)

    assert out_features == 2
    assert in_features == 64
    assert quant_block_size == 64
    assert int(packed.numel()) == 64
    assert int(absmax.numel()) == 2
    assert int(code.numel()) == 16


def test_decode_profiler_records_cpu_only_layers() -> None:
    runtime = _DummySession()
    runtime.device = torch.device("cpu")
    runtime._traffic_current_phase = "decode"
    runtime._decode_profile_enabled = False
    runtime._decode_profile_max_steps = 0
    runtime._decode_profile_steps = []
    runtime._last_decode_profile_report = None

    runtime.enable_decode_profiler(True, max_steps=1)
    step = runtime._begin_decode_profile_step(position_index=17)
    runtime._record_decode_profile_layer(
        step,
        layer_idx=3,
        cpu_ms=12.5,
        events=None,
    )
    finalized = runtime._finalize_decode_profile_step(step)
    report = runtime.get_decode_profile_report()

    assert finalized is not None
    assert finalized["summary"]["layers"] == 1
    assert finalized["layers"][0]["layer_idx"] == 3
    assert finalized["layers"][0]["total_ms"] == 12.5
    assert report is not None
    assert report["steps_recorded"] == 1
    assert report["summary"]["mean_total_ms"] == 12.5


def test_hot_cache_calibration_uses_recency_weighting() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._hot_cache_calibration_active = True
    runtime._hot_cache_calibration_hits = {}
    runtime._hot_cache_calibration_recency_power = 2.0
    runtime._sparse_num_blocks = 8
    runtime._sparse_routing = {
        0: {
            "num_blocks": 8,
        }
    }

    runtime._record_hot_cache_calibration_blocks(
        0,
        torch.tensor(
            [
                [1, 1],
                [2, 2],
                [3, 3],
            ],
            dtype=torch.long,
        ),
    )

    counts = runtime._hot_cache_calibration_hits[0]
    assert float(counts[3].item()) > float(counts[2].item())
    assert float(counts[2].item()) > float(counts[1].item())


def test_hot_cache_target_blocks_not_clamped_to_fractional_topk() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._vram_hot_cache_limit_bytes = 5 * (1024 ** 3)
    runtime._vram_hot_cache_used_bytes = 0
    runtime._sparse_routing = {layer_idx: {} for layer_idx in range(126)}
    runtime.num_layers = 126
    runtime._estimate_mlp_hot_cache_bytes_per_block = lambda layer_idx: 256 * 1024

    target = runtime._target_hot_blocks_for_layer(
        layer_idx=0,
        runtime_top_k=51,
        count_slots=256,
        previous_hot_count=0,
    )

    assert target >= 51


def test_fixed_capacity_bank_update_uses_free_slots_then_replaces_lowest_score() -> None:
    slots, block_ids, scores, active_count = runtime_mod.StreamingLlamaRuntime._plan_fixed_capacity_bank_update(
        existing_block_ids_cpu=torch.tensor([10, 11, -1], dtype=torch.long),
        existing_scores_cpu=torch.tensor([5.0, 1.0, 0.0], dtype=torch.float32),
        active_count=2,
        capacity=3,
        novel_blocks_cpu=torch.tensor([20, 21], dtype=torch.long),
    )

    assert active_count == 3
    assert slots.tolist() == [2, 1]
    assert block_ids.tolist() == [10, 21, 20]
    assert float(scores[1].item()) > 5.0
    assert float(scores[2].item()) > 5.0


def test_crop_attention_caches_trims_compact_cache() -> None:
    runtime = _DummySession()
    runtime.taylor_layer_set = set()
    runtime._dense_cache = None
    runtime._compact_attn_cache = {
        3: {
            "k": torch.randn(1, 2, 5, 4),
            "v": torch.randn(1, 2, 5, 4),
        }
    }

    cropped = runtime._crop_attention_caches(2)

    assert cropped is False
    assert tuple(runtime._compact_attn_cache[3]["k"].shape) == (1, 2, 2, 4)
    assert tuple(runtime._compact_attn_cache[3]["v"].shape) == (1, 2, 2, 4)


def test_should_use_compact_sparse_attention_only_on_decode_cuda() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._compact_sparse_attn_decode = True
    runtime.device = torch.device("cuda")
    runtime._traffic_current_phase = "decode"
    runtime._retrieval_layers = set()
    runtime.taylor_layer_set = set()

    assert runtime._should_use_compact_sparse_attention(
        layer_idx=0,
        active_heads=torch.tensor([0, 1], dtype=torch.long),
        use_attention_cache=True,
        use_shared_attn=False,
    ) is True
    assert runtime._should_use_compact_sparse_attention(
        layer_idx=0,
        active_heads=torch.tensor([0, 1], dtype=torch.long),
        use_attention_cache=False,
        use_shared_attn=False,
    ) is False


def test_triton_fused_sparse_mlp_decode_4bit_composes_sparse_linear_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_output(x_flat: torch.Tensor, active_idx: torch.Tensor, flat_mask: torch.Tensor, **_: object) -> torch.Tensor:
        calls.append("output")
        return torch.ones_like(flat_mask, dtype=x_flat.dtype)

    def _fake_input(x_flat: torch.Tensor, active_idx: torch.Tensor, **_: object) -> torch.Tensor:
        calls.append("input")
        assert tuple(x_flat.shape) == (1, 4)
        assert tuple(active_idx.shape) == (1, 2)
        return torch.full((1, 3), 7.0, dtype=x_flat.dtype)

    monkeypatch.setattr(triton_sparse_mlp_mod, "triton_sparse_output_linear_4bit", _fake_output)
    monkeypatch.setattr(triton_sparse_mlp_mod, "triton_sparse_input_linear_4bit", _fake_input)

    out = triton_sparse_mlp_mod.triton_fused_sparse_mlp_decode_4bit(
        torch.ones((1, 4), dtype=torch.float16),
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.ones((1, 4), dtype=torch.float16),
        gate_packed_weight=torch.empty((8,), dtype=torch.uint8),
        gate_absmax=torch.empty((4,), dtype=torch.float32),
        gate_code=torch.empty((16,), dtype=torch.float32),
        gate_input_dim=4,
        gate_quant_block_size=64,
        gate_bias=None,
        up_packed_weight=torch.empty((8,), dtype=torch.uint8),
        up_absmax=torch.empty((4,), dtype=torch.float32),
        up_code=torch.empty((16,), dtype=torch.float32),
        up_input_dim=4,
        up_quant_block_size=64,
        up_bias=None,
        down_packed_weight=torch.empty((8,), dtype=torch.uint8),
        down_absmax=torch.empty((4,), dtype=torch.float32),
        down_code=torch.empty((16,), dtype=torch.float32),
        down_out_features=3,
        down_in_features=4,
        down_quant_block_size=64,
        down_bias=None,
        block_size=2,
    )

    assert calls == ["output", "output", "input"]
    assert tuple(out.shape) == (1, 3)
