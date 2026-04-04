import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from llama3_neuroplastic.experiments import run_streaming_inference as cli_mod
from llama3_neuroplastic.experiments import streaming_llama_runtime as runtime_mod


def _write_index(snapshot_dir: Path) -> None:
    payload = {
        "weight_map": {
            "model.layers.0.mlp.up_proj.weight": "model-00001-of-00044.safetensors",
            "model.layers.0.mlp.up_proj.weight.absmax": "model-00001-of-00044.safetensors",
            "model.layers.1.mlp.up_proj.weight": "model-00002-of-00044.safetensors",
            "model.layers.1.mlp.up_proj.weight.absmax": "model-00002-of-00044.safetensors",
        }
    }
    (snapshot_dir / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")


def test_loader_reopens_shards_when_cache_disabled(tmp_path, monkeypatch) -> None:
    _write_index(tmp_path)
    opened_paths = []

    def _fake_load_safetensors_direct(path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
        opened_paths.append(str(path))
        return {key: torch.tensor([len(opened_paths)], dtype=torch.float32) for key in keys}

    monkeypatch.setattr(runtime_mod, "_load_safetensors_direct", _fake_load_safetensors_direct)
    loader = runtime_mod.ShardedSafetensorLoader(tmp_path, cache_shard_handles=False)

    first = loader._load_exact_tensors(["model.layers.0.mlp.up_proj.weight"])
    second = loader._load_exact_tensors(["model.layers.0.mlp.up_proj.weight"])

    assert len(opened_paths) == 2
    assert loader._shard_handles == {}
    assert first["model.layers.0.mlp.up_proj.weight"].item() == 1.0
    assert second["model.layers.0.mlp.up_proj.weight"].item() == 2.0


def test_loader_reuses_shards_when_cache_enabled(tmp_path, monkeypatch) -> None:
    _write_index(tmp_path)
    opened_paths = []

    class _FakeHandle:
        def __init__(self, path: str) -> None:
            self.path = path

        def __enter__(self):
            raise AssertionError("cached path should not use context-manager mode")

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get_tensor(self, key: str) -> torch.Tensor:
            return torch.tensor([1.0], dtype=torch.float32)

    def _fake_safe_open(path: str, framework: str, device: str):
        opened_paths.append(path)
        return _FakeHandle(path)

    monkeypatch.setattr(runtime_mod, "safe_open", _fake_safe_open)
    loader = runtime_mod.ShardedSafetensorLoader(tmp_path, cache_shard_handles=True)

    loader._load_exact_tensors(["model.layers.0.mlp.up_proj.weight"])
    loader._load_exact_tensors(["model.layers.0.mlp.up_proj.weight"])

    assert len(opened_paths) == 1
    assert set(loader._shard_handles.keys()) == {"model-00001-of-00044.safetensors"}


def test_parse_taylor_layer_selection_preserves_runtime_default_and_explicit_disable() -> None:
    assert cli_mod._parse_layer_selection(None) is None
    assert cli_mod._parse_layer_selection("") is None
    assert cli_mod._parse_layer_selection("all") is None
    assert cli_mod._parse_layer_selection("none") == []
    assert cli_mod._parse_layer_selection("0-2,5") == [0, 1, 2, 5]


def _make_sparse_runtime_stub() -> runtime_mod.StreamingLlamaRuntime:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.config = SimpleNamespace(hidden_size=4)
    runtime._sparse_routing = {}
    runtime._sparse_top_k_by_layer = {}
    runtime._sparse_basis_top_k_by_layer = {}
    runtime._sparse_basis_bias_mode = "selected"
    runtime._sparse_top_k = 1
    runtime._sparse_runtime_top_k = 1
    runtime._sparse_block_size = 2
    runtime._sparse_checkpoint_basis_rank = 2
    runtime._sparse_semantic_block_score_normalized = False
    runtime._mlp_hot_blocks_by_layer = {}
    runtime._session_sparse_route_layers = set()
    runtime._upper_decode_guard_layers = set()
    return runtime


def test_compute_sparse_basis_latent_applies_silu_and_topk_support() -> None:
    runtime = _make_sparse_runtime_stub()
    runtime._sparse_routing[0] = {
        "enc_w": torch.eye(2, dtype=torch.float32),
        "enc_b": torch.zeros(2, dtype=torch.float32),
        "basis_top_k": 1,
    }

    flat_hidden = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    latent = runtime._compute_sparse_basis_latent(flat_hidden, 0)

    expected = torch.tensor([[F.silu(torch.tensor(1.0)), 0.0]], dtype=torch.float32)
    assert torch.allclose(latent, expected)


def test_route_sparse_mlp_uses_masked_latent_support() -> None:
    runtime = _make_sparse_runtime_stub()
    runtime._sparse_routing[0] = {
        "enc_w": torch.eye(2, dtype=torch.float32),
        "enc_b": torch.zeros(2, dtype=torch.float32),
        "dec_norm_t": torch.tensor([[2.0, 0.0], [0.0, 10.0]], dtype=torch.float32),
        "top_k": 1,
        "basis_top_k": 1,
    }

    hidden = torch.tensor([[[10.0, 9.0]]], dtype=torch.float32)
    active_blocks = runtime._route_sparse_mlp(hidden, 0)

    assert active_blocks.shape == (1, 1)
    assert int(active_blocks.item()) == 0


def test_maybe_fit_local_decode_guard_basis_registers_session_sparse_route(monkeypatch) -> None:
    runtime = _make_sparse_runtime_stub()
    runtime._upper_decode_guard_layers = {5}

    monkeypatch.setattr(
        runtime_mod,
        "fit_layer_basis",
        lambda **kwargs: {
            "encoder_weight": torch.eye(2, 4, dtype=torch.float32),
            "encoder_bias": torch.zeros(2, dtype=torch.float32),
            "decoder_blocks": torch.ones((2, 2, 2), dtype=torch.float32),
            "decoder_bias": torch.zeros((2, 2), dtype=torch.float32),
            "scale": torch.tensor(1.0, dtype=torch.float32),
        },
    )

    mlp_input = torch.ones((1, 2, 4), dtype=torch.float32)
    mlp_out = torch.ones((1, 2, 4), dtype=torch.float32)
    runtime._maybe_fit_local_decode_guard_basis(5, mlp_input, mlp_out)

    assert 5 in runtime._sparse_routing
    assert runtime._sparse_top_k_by_layer[5] == 1
    assert runtime._sparse_basis_top_k_by_layer[5] == 2
    assert 5 in runtime._session_sparse_route_layers


def test_forward_learned_basis_mlp_applies_bias_only_to_selected_blocks() -> None:
    runtime = _make_sparse_runtime_stub()
    runtime._sparse_routing[0] = {
        "enc_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "enc_b": torch.zeros(1, dtype=torch.float32),
        "dec": torch.tensor(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ],
            dtype=torch.float32,
        ),
        "dec_bias": torch.tensor(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ],
            dtype=torch.float32,
        ),
        "scale": 1.0,
        "basis_top_k": 1,
    }

    hidden = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    active_blocks = torch.tensor([[1]], dtype=torch.long)
    out = runtime._forward_learned_basis_mlp(0, hidden, active_blocks)

    latent = F.silu(torch.tensor(2.0))
    expected = torch.tensor(
        [[[0.0, 0.0, latent * 3.0 + 30.0, latent * 4.0 + 40.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(out, expected)


def test_mlp_forward_dispatch_uses_sparse_basis_for_covered_layers_and_guard_for_others(monkeypatch) -> None:
    runtime = _make_sparse_runtime_stub()
    runtime._sparse_routing = {2: {"dummy": torch.tensor(1.0)}}
    layer = SimpleNamespace(mlp=object())
    mlp_input = torch.zeros((1, 1, 4), dtype=torch.float32)
    calls = []

    monkeypatch.setattr(
        runtime,
        "_route_sparse_mlp",
        lambda hidden, layer_idx: torch.tensor([[7]], dtype=torch.long) if int(layer_idx) == 2 else None,
    )

    def _fake_sparse(layer_idx: int, mlp, hidden: torch.Tensor, active_blocks: torch.Tensor) -> torch.Tensor:
        calls.append(("sparse", int(layer_idx), active_blocks.clone()))
        return torch.full_like(hidden, 3.0)

    def _fake_guard(layer_idx: int, mlp, hidden: torch.Tensor) -> torch.Tensor:
        calls.append(("guard", int(layer_idx)))
        return torch.full_like(hidden, 5.0)

    monkeypatch.setattr(runtime, "_sparse_mlp_forward_fast", _fake_sparse)
    monkeypatch.setattr(runtime, "_dense_guard_mlp_forward_exact_chunked_4bit", _fake_guard)

    sparse_out = runtime._mlp_forward_dispatch(2, layer, mlp_input)
    guard_out = runtime._mlp_forward_dispatch(1, layer, mlp_input)

    assert torch.all(sparse_out == 3.0)
    assert torch.all(guard_out == 5.0)
    assert calls[0][0] == "sparse"
    assert calls[0][1] == 2
    assert torch.equal(calls[0][2], torch.tensor([[7]], dtype=torch.long))
    assert calls[1] == ("guard", 1)


def test_get_attn_active_heads_preserves_importance_ranking_when_trimming() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._attn_sparse_disabled_reason = None
    runtime._attn_active_head_indices = {0: torch.tensor([50, 1, 3], dtype=torch.long)}
    runtime._attn_head_importance = {0: torch.ones(64, dtype=torch.float32)}
    runtime._taylor_caches = [None]
    runtime._attn_active_heads = 2
    runtime._attn_min_active_heads = 1
    runtime._attn_max_active_heads = 3
    runtime._attn_dynamic_threshold = 0.10
    runtime._attn_runtime_head_counts = {}

    active = runtime._get_attn_active_heads(0)

    assert torch.equal(active, torch.tensor([1, 50], dtype=torch.long))


def test_route_kv_blocks_uses_dense_prefill_guard() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._traffic_current_phase = "prefill"
    runtime._sparse_kv_prefill_mode = "dense"
    runtime._kv_routing = {
        0: {
            "enc_w": torch.eye(2, dtype=torch.float32),
            "enc_b": torch.zeros(2, dtype=torch.float32),
            "dec_norm_t": torch.eye(2, dtype=torch.float32),
            "top_k": 1,
        }
    }

    active = runtime._route_kv_blocks(torch.ones((1, 1, 2), dtype=torch.float32), 0)

    assert active is None


def test_generate_skips_extra_forward_token_on_final_step() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._materialize_lm_head = True
    runtime.device = torch.device("cpu")
    runtime._session_token_ids_cpu = None
    runtime._session_last_logits_cpu = None
    runtime.taylor_layer_set = set()
    runtime._reset_traffic_stats = lambda: None
    runtime._set_traffic_phase = lambda phase: None
    runtime._finalize_traffic_report = lambda: None
    runtime.reset_caches = lambda: None
    runtime._set_session_state = lambda token_ids_cpu, logits: None
    runtime._sample_next_token = lambda logits, **kwargs: torch.tensor([1], dtype=torch.long)
    runtime._forward_prefill = lambda generated, position_offset=0: torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)
    runtime._longest_common_prefix_len = lambda a, b: 0
    runtime._crop_attention_caches = lambda n: False

    def _unexpected_forward_token(*args, **kwargs):
        raise AssertionError("forward_token should not run after the final sampled token")

    runtime.forward_token = _unexpected_forward_token

    out = runtime.generate(
        input_ids=torch.tensor([[10, 11]], dtype=torch.long),
        max_new_tokens=1,
        eos_token_id=None,
        do_sample=False,
    )

    assert torch.equal(out, torch.tensor([[10, 11, 1]], dtype=torch.long))
