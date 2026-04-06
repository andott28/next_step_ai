import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from llama3_neuroplastic import token_posting_archive as posting_mod
from llama3_neuroplastic.experiments import init_attn_share as attn_share_mod
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


def test_cli_accepts_attention_share_checkpoint_path() -> None:
    parser = cli_mod._build_arg_parser()

    args = parser.parse_args(
        [
            "--model-name",
            "dummy/model",
            "--prompt",
            "hello",
            "--attn-share-path",
            "results/attn_share_qo.pt",
        ]
    )

    assert args.attn_share_path == "results/attn_share_qo.pt"


def test_cli_accepts_prompt_format_flag() -> None:
    parser = cli_mod._build_arg_parser()

    args = parser.parse_args(
        [
            "--model-name",
            "dummy/model",
            "--prompt",
            "hello",
            "--prompt-format",
            "chat",
        ]
    )

    assert args.prompt_format == "chat"


def test_run_single_prompt_reports_completion_text_only(monkeypatch) -> None:
    class _DummyTokenizer:
        eos_token_id = None

        def __call__(self, prompt, return_tensors="pt"):
            return {"input_ids": torch.tensor([[10, 11]], dtype=torch.long)}

        def decode(self, tokens, **kwargs):
            flat = [int(x) for x in torch.as_tensor(tokens).view(-1).tolist()]
            return " ".join(str(v) for v in flat)

    class _DummyRuntime:
        def generate(self, **kwargs):
            return torch.tensor([[10, 11, 42, 43]], dtype=torch.long)

        def get_last_traffic_report(self):
            return None

    args = SimpleNamespace(
        no_stream_output=True,
        max_new_tokens=2,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        prompt_format="raw",
    )

    row = cli_mod._run_single_prompt(
        "x",
        runtime=_DummyRuntime(),
        tokenizer=_DummyTokenizer(),
        args=args,
        prompt_idx=0,
    )

    assert row["text"] == "10 11 42 43"
    assert row["completion_text"] == "42 43"


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
    runtime._sparse_basis_execution = "full_output_latent"
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


def test_forward_learned_basis_mlp_full_output_reconstructs_all_blocks_from_sparse_latent() -> None:
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
    out = runtime._forward_learned_basis_mlp_full_output(0, hidden)

    latent = F.silu(torch.tensor(2.0))
    expected = torch.tensor(
        [[[latent * 1.0 + 10.0, latent * 2.0 + 20.0, latent * 3.0 + 30.0, latent * 4.0 + 40.0]]],
        dtype=torch.float32,
    )
    assert torch.allclose(out, expected)


def test_mlp_forward_dispatch_uses_full_output_basis_for_covered_layers_and_guard_for_others(monkeypatch) -> None:
    runtime = _make_sparse_runtime_stub()
    runtime._sparse_routing = {2: {"dummy": torch.tensor(1.0)}}
    layer = SimpleNamespace(mlp=object())
    mlp_input = torch.zeros((1, 1, 4), dtype=torch.float32)
    calls = []

    def _fake_full_output(layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        calls.append(("full_output", int(layer_idx)))
        return torch.full_like(hidden, 3.0)

    def _fake_guard(layer_idx: int, mlp, hidden: torch.Tensor) -> torch.Tensor:
        calls.append(("guard", int(layer_idx)))
        return torch.full_like(hidden, 5.0)

    monkeypatch.setattr(runtime, "_forward_learned_basis_mlp_full_output", _fake_full_output)
    monkeypatch.setattr(runtime, "_dense_guard_mlp_forward_exact_chunked_4bit", _fake_guard)

    sparse_out = runtime._mlp_forward_dispatch(2, layer, mlp_input)
    guard_out = runtime._mlp_forward_dispatch(1, layer, mlp_input)

    assert torch.all(sparse_out == 3.0)
    assert torch.all(guard_out == 5.0)
    assert calls[0][0] == "full_output"
    assert calls[0][1] == 2
    assert calls[1] == ("guard", 1)


def test_mlp_forward_dispatch_routed_block_mode_keeps_block_router(monkeypatch) -> None:
    runtime = _make_sparse_runtime_stub()
    runtime._sparse_basis_execution = "routed_blocks"
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
        calls.append(("routed_blocks", int(layer_idx), active_blocks.clone()))
        return torch.full_like(hidden, 3.0)

    monkeypatch.setattr(runtime, "_sparse_mlp_forward_fast", _fake_sparse)

    sparse_out = runtime._mlp_forward_dispatch(2, layer, mlp_input)

    assert torch.all(sparse_out == 3.0)
    assert calls[0][0] == "routed_blocks"
    assert calls[0][1] == 2
    assert torch.equal(calls[0][2], torch.tensor([[7]], dtype=torch.long))


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


def test_unpermute_attention_share_helpers_restore_original_head_order() -> None:
    head_perm = torch.tensor([1, 0], dtype=torch.long)
    q_factor = torch.tensor(
        [
            [10.0, 11.0],
            [12.0, 13.0],
            [20.0, 21.0],
            [22.0, 23.0],
        ],
        dtype=torch.float32,
    )
    o_factor = torch.tensor(
        [
            [100.0, 101.0, 200.0, 201.0],
            [110.0, 111.0, 210.0, 211.0],
        ],
        dtype=torch.float32,
    )

    q_out = runtime_mod._unpermute_q_factor_rows(q_factor, head_perm, head_dim=2)
    o_out = runtime_mod._unpermute_o_factor_cols(o_factor, head_perm, head_dim=2)

    assert torch.equal(
        q_out,
        torch.tensor(
            [
                [20.0, 21.0],
                [22.0, 23.0],
                [10.0, 11.0],
                [12.0, 13.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        o_out,
        torch.tensor(
            [
                [200.0, 201.0, 100.0, 101.0],
                [210.0, 211.0, 110.0, 111.0],
            ],
            dtype=torch.float32,
        ),
    )


def test_build_layer_groups_keeps_edges_exact_and_lonely_tail_exact() -> None:
    groups, exact = attn_share_mod._build_layer_groups(
        selected_layers=list(range(12)),
        total_layers=12,
        group_size=2,
        exact_lower_layers=2,
        exact_upper_layers=2,
    )

    assert groups == [[2, 3], [4, 5], [6, 7], [8, 9]]
    assert exact == [0, 1, 10, 11]

    groups, exact = attn_share_mod._build_layer_groups(
        selected_layers=[2, 3, 4],
        total_layers=12,
        group_size=2,
        exact_lower_layers=0,
        exact_upper_layers=0,
    )

    assert groups == [[2, 3]]
    assert exact == [4]


def test_bridge_head_matrix_ortho_centroid_recovers_rotated_shifted_target() -> None:
    source = torch.tensor(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.3, -0.7, 1.2, 2.0],
        ],
        dtype=torch.float32,
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    sign = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
    target = torch.matmul(rotation, source) * sign.unsqueeze(0)
    target = target + torch.tensor([[0.5], [-1.0]], dtype=torch.float32) + torch.tensor([[0.2, -0.3, 0.1, 0.4]], dtype=torch.float32)

    bridged = attn_share_mod._bridge_head_matrix_ortho_centroid(source, target)
    baseline = torch.linalg.norm(source - target) / torch.linalg.norm(target)

    rel = torch.linalg.norm(bridged - target) / torch.linalg.norm(target)
    assert float(rel) < float(baseline)


def test_bridge_head_stack_ortho_centroid_runs_per_head() -> None:
    source = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-1.0, 0.5], [2.0, -3.0]],
        ],
        dtype=torch.float32,
    )
    target = source.clone()
    target[0] = target[0] + 0.25
    target[1] = -target[1] + 0.5

    bridged = attn_share_mod._bridge_head_stack_ortho_centroid(source, target)
    rel = torch.linalg.norm(bridged - target) / torch.linalg.norm(target)
    assert float(rel) < 1e-4


def test_fit_group_shared_qo_records_reconstruction_error_metrics() -> None:
    q0 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    o0 = torch.tensor(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        dtype=torch.float32,
    )

    group_state, layer_states = attn_share_mod._fit_group_shared_qo(
        [0, 1],
        {0: q0, 1: q0.clone()},
        {0: o0, 1: o0.clone()},
        num_heads=2,
        head_dim=2,
        base_rank=4,
        residual_rank=2,
        sample_cols=4,
        sample_rows=4,
        factor_device=torch.device("cpu"),
        bridge_mode="ortho_centroid",
    )

    assert set(layer_states.keys()) == {0, 1}
    assert set(group_state["recon_error_by_layer"].keys()) == {0, 1}
    assert group_state["recon_error_by_layer"][0]["max_rel_l2"] < 1e-5
    assert group_state["recon_error_by_layer"][1]["max_rel_l2"] < 1e-5
    assert group_state["sharing_format"] == "headwise_v1"
    assert tuple(group_state["q_base_u_heads"].shape) == (2, 2, 4)
    assert tuple(group_state["o_base_v_heads"].shape) == (2, 2, 2)


def test_fit_group_shared_kv_records_reconstruction_error_metrics() -> None:
    k0 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    v0 = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 0.0],
            [0.0, 0.0, 0.0, 6.0],
        ],
        dtype=torch.float32,
    )

    group_state, layer_states = attn_share_mod._fit_group_shared_kv(
        [0, 1],
        {0: k0, 1: k0.clone()},
        {0: v0, 1: v0.clone()},
        num_kv_heads=2,
        head_dim=2,
        base_rank=4,
        residual_rank=2,
        sample_cols=4,
        factor_device=torch.device("cpu"),
        bridge_mode="ortho_centroid",
    )

    assert set(layer_states.keys()) == {0, 1}
    assert set(group_state["kv_recon_error_by_layer"].keys()) == {0, 1}
    assert group_state["kv_recon_error_by_layer"][0]["max_rel_l2"] < 1e-5
    assert group_state["kv_recon_error_by_layer"][1]["max_rel_l2"] < 1e-5
    assert tuple(group_state["k_base_u_heads"].shape) == (2, 2, 4)
    assert tuple(group_state["v_base_v_heads"].shape) == (2, 4, 4)


def test_should_use_attn_share_respects_dense_prefill_policy() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._attn_share_layer_state = {3: {"group_id": "0"}}
    runtime._traffic_current_phase = "prefill"
    runtime._attn_share_prefill_mode = "dense"

    assert runtime._should_use_attn_share_for_layer(3) is False

    runtime._attn_share_prefill_mode = "shared"
    assert runtime._should_use_attn_share_for_layer(3) is True

    runtime._traffic_current_phase = "decode"
    runtime._attn_share_prefill_mode = "dense"
    assert runtime._should_use_attn_share_for_layer(3) is True
    assert runtime._should_use_attn_share_for_layer(2) is False


def test_runtime_tracks_explicit_attn_share_exact_layers() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._attn_share_exact_layers = {1, 2, 5}

    assert runtime._attn_share_exact_layers == {1, 2, 5}


def test_load_shared_attn_qo_reconstructs_permuted_base_plus_residual() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.config = SimpleNamespace(head_dim=2)
    runtime._copy_cpu_to_gpu = lambda tensor, **kwargs: tensor.to(device=runtime.device, dtype=runtime.dtype).clone()
    runtime._wait_for_h2d_stream = lambda: None
    runtime._layer_skeleton = SimpleNamespace(
        self_attn=SimpleNamespace(
            q_proj=SimpleNamespace(weight=torch.empty((4, 4), dtype=torch.float32)),
            o_proj=SimpleNamespace(weight=torch.empty((4, 4), dtype=torch.float32)),
        )
    )
    runtime._attn_loaded_q_rows = torch.tensor([0], dtype=torch.long)
    runtime._attn_loaded_o_cols = torch.tensor([1], dtype=torch.long)
    runtime._attn_qo_state = "unknown"

    head_perm = torch.tensor([1, 0], dtype=torch.long)
    q_base = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=torch.float32,
    )
    q_resid = torch.tensor(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    o_base = torch.tensor(
        [
            [20.0, 21.0, 22.0, 23.0],
            [24.0, 25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0, 31.0],
            [32.0, 33.0, 34.0, 35.0],
        ],
        dtype=torch.float32,
    )
    o_resid = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    runtime._attn_share_groups = {
        "0": {
            "q_base_u": q_base,
            "q_base_v": torch.eye(4, dtype=torch.float32),
            "o_base_u": o_base,
            "o_base_v": torch.eye(4, dtype=torch.float32),
        }
    }
    runtime._attn_share_layer_state = {
        3: {
            "group_id": "0",
            "head_perm": head_perm,
            "q_resid_u": q_resid,
            "q_resid_v": torch.eye(4, dtype=torch.float32),
            "o_resid_u": o_resid,
            "o_resid_v": torch.eye(4, dtype=torch.float32),
        }
    }

    runtime._load_shared_attn_qo(3)

    expected_q = runtime_mod._unpermute_q_factor_rows(q_base + q_resid, head_perm, head_dim=2)
    expected_o = runtime_mod._unpermute_o_factor_cols(o_base + o_resid, head_perm, head_dim=2)

    assert torch.allclose(runtime._layer_skeleton.self_attn.q_proj.weight, expected_q)
    assert torch.allclose(runtime._layer_skeleton.self_attn.o_proj.weight, expected_o)
    assert runtime._attn_loaded_q_rows is None
    assert runtime._attn_loaded_o_cols is None
    assert runtime._attn_qo_state == "shared"


def test_load_shared_attn_qo_supports_headwise_artifact_format() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.config = SimpleNamespace(head_dim=2)
    runtime._copy_cpu_to_gpu = lambda tensor, **kwargs: tensor.to(device=runtime.device, dtype=runtime.dtype).clone()
    runtime._wait_for_h2d_stream = lambda: None
    runtime._layer_skeleton = SimpleNamespace(
        self_attn=SimpleNamespace(
            q_proj=SimpleNamespace(weight=torch.empty((4, 4), dtype=torch.float32)),
            o_proj=SimpleNamespace(weight=torch.empty((4, 4), dtype=torch.float32)),
        )
    )
    runtime._attn_loaded_q_rows = torch.tensor([0], dtype=torch.long)
    runtime._attn_loaded_o_cols = torch.tensor([1], dtype=torch.long)
    runtime._attn_qo_state = "unknown"

    head_perm = torch.tensor([1, 0], dtype=torch.long)
    q_heads = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ],
        dtype=torch.float32,
    )
    o_heads = torch.tensor(
        [
            [[20.0, 21.0], [22.0, 23.0], [24.0, 25.0], [26.0, 27.0]],
            [[28.0, 29.0], [30.0, 31.0], [32.0, 33.0], [34.0, 35.0]],
        ],
        dtype=torch.float32,
    )

    runtime._attn_share_groups = {
        "0": {
            "q_base_u_heads": q_heads,
            "q_base_v_heads": torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1),
            "o_base_u_heads": o_heads,
            "o_base_v_heads": torch.eye(2, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1),
        }
    }
    runtime._attn_share_layer_state = {
        3: {
            "group_id": "0",
            "head_perm": head_perm,
            "q_resid_u_heads": None,
            "q_resid_v_heads": None,
            "o_resid_u_heads": None,
            "o_resid_v_heads": None,
        }
    }

    runtime._load_shared_attn_qo(3)

    expected_q = runtime_mod._unpermute_headwise_tensor(q_heads, head_perm).reshape(4, 4)
    expected_o = runtime_mod._unpermute_headwise_tensor(o_heads, head_perm).permute(1, 0, 2).reshape(4, 4)

    assert torch.allclose(runtime._layer_skeleton.self_attn.q_proj.weight, expected_q)
    assert torch.allclose(runtime._layer_skeleton.self_attn.o_proj.weight, expected_o)
    assert runtime._attn_qo_state == "shared"


def test_load_shared_attn_kv_reconstructs_permuted_base_plus_residual() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.config = SimpleNamespace(head_dim=2)
    runtime._copy_cpu_to_gpu = lambda tensor, **kwargs: tensor.to(device=runtime.device, dtype=runtime.dtype).clone()
    runtime._wait_for_h2d_stream = lambda: None
    runtime._layer_skeleton = SimpleNamespace(
        self_attn=SimpleNamespace(
            k_proj=SimpleNamespace(weight=torch.empty((4, 4), dtype=torch.float32)),
            v_proj=SimpleNamespace(weight=torch.empty((4, 4), dtype=torch.float32)),
        )
    )
    runtime._kv_loaded_cols = torch.tensor([0], dtype=torch.long)

    kv_head_perm = torch.tensor([1, 0], dtype=torch.long)
    k_base = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=torch.float32,
    )
    k_resid = torch.tensor(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    v_base = torch.tensor(
        [
            [20.0, 21.0, 22.0, 23.0],
            [24.0, 25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0, 31.0],
            [32.0, 33.0, 34.0, 35.0],
        ],
        dtype=torch.float32,
    )
    v_resid = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    runtime._attn_share_groups = {
        "0": {
            "k_base_u": k_base,
            "k_base_v": torch.eye(4, dtype=torch.float32),
            "v_base_u": v_base,
            "v_base_v": torch.eye(4, dtype=torch.float32),
        }
    }
    runtime._attn_share_layer_state = {
        3: {
            "group_id": "0",
            "kv_head_perm": kv_head_perm,
            "k_resid_u": k_resid,
            "k_resid_v": torch.eye(4, dtype=torch.float32),
            "v_resid_u": v_resid,
            "v_resid_v": torch.eye(4, dtype=torch.float32),
        }
    }

    runtime._load_shared_attn_kv(3)

    expected_k = runtime_mod._unpermute_q_factor_rows(k_base + k_resid, kv_head_perm, head_dim=2)
    expected_v = runtime_mod._unpermute_q_factor_rows(v_base + v_resid, kv_head_perm, head_dim=2)

    assert torch.allclose(runtime._layer_skeleton.self_attn.k_proj.weight, expected_k)
    assert torch.allclose(runtime._layer_skeleton.self_attn.v_proj.weight, expected_v)
    assert runtime._kv_loaded_cols is None


def test_load_shared_attn_kv_supports_headwise_artifact_format() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.config = SimpleNamespace(head_dim=2)
    runtime._copy_cpu_to_gpu = lambda tensor, **kwargs: tensor.to(device=runtime.device, dtype=runtime.dtype).clone()
    runtime._wait_for_h2d_stream = lambda: None
    runtime._layer_skeleton = SimpleNamespace(
        self_attn=SimpleNamespace(
            k_proj=SimpleNamespace(weight=torch.empty((4, 4), dtype=torch.float32)),
            v_proj=SimpleNamespace(weight=torch.empty((4, 4), dtype=torch.float32)),
        )
    )
    runtime._kv_loaded_cols = torch.tensor([0], dtype=torch.long)

    kv_head_perm = torch.tensor([1, 0], dtype=torch.long)
    k_heads = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ],
        dtype=torch.float32,
    )
    v_heads = torch.tensor(
        [
            [[20.0, 21.0, 22.0, 23.0], [24.0, 25.0, 26.0, 27.0]],
            [[28.0, 29.0, 30.0, 31.0], [32.0, 33.0, 34.0, 35.0]],
        ],
        dtype=torch.float32,
    )
    runtime._attn_share_groups = {
        "0": {
            "k_base_u_heads": k_heads,
            "k_base_v_heads": torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1),
            "v_base_u_heads": v_heads,
            "v_base_v_heads": torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1),
        }
    }
    runtime._attn_share_layer_state = {
        3: {
            "group_id": "0",
            "kv_head_perm": kv_head_perm,
            "k_resid_u_heads": None,
            "k_resid_v_heads": None,
            "v_resid_u_heads": None,
            "v_resid_v_heads": None,
        }
    }

    runtime._load_shared_attn_kv(3)

    expected_k = runtime_mod._unpermute_headwise_tensor(k_heads, kv_head_perm).reshape(4, 4)
    expected_v = runtime_mod._unpermute_headwise_tensor(v_heads, kv_head_perm).reshape(4, 4)

    assert torch.allclose(runtime._layer_skeleton.self_attn.k_proj.weight, expected_k)
    assert torch.allclose(runtime._layer_skeleton.self_attn.v_proj.weight, expected_v)
    assert runtime._kv_loaded_cols is None


def test_route_kv_blocks_disables_old_sparse_kv_when_attn_share_is_active() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._traffic_current_phase = "decode"
    runtime._sparse_kv_prefill_mode = "dense"
    runtime._attn_share_layer_state = {0: {"group_id": "0"}}
    runtime._attn_share_prefill_mode = "dense"
    runtime._retrieval_layers = set()
    runtime._token_archive = None
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


def test_route_kv_blocks_uses_dense_prefill_guard() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._traffic_current_phase = "prefill"
    runtime._sparse_kv_prefill_mode = "dense"
    runtime._retrieval_layers = set()
    runtime._token_archive = None
    runtime._attn_share_layer_state = {}
    runtime._attn_share_prefill_mode = "dense"
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


def test_route_kv_blocks_disables_sparse_kv_on_retrieval_layers() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._traffic_current_phase = "decode"
    runtime._sparse_kv_prefill_mode = "sparse"
    runtime._attn_share_layer_state = {}
    runtime._attn_share_prefill_mode = "dense"
    runtime._retrieval_layers = {0}
    runtime._token_archive = object()
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


def test_release_dense_cache_for_retrieval_layers_sets_layer_entries_to_none() -> None:
    runtime = runtime_mod.StreamingLlamaRuntime.__new__(runtime_mod.StreamingLlamaRuntime)
    runtime._retrieval_layers = {1}
    runtime._dense_cache = SimpleNamespace(
        key_cache=[torch.ones(1), torch.ones(1), torch.ones(1)],
        value_cache=[torch.ones(1), torch.ones(1), torch.ones(1)],
    )

    runtime._release_dense_cache_for_retrieval_layers()

    assert runtime._dense_cache.key_cache[0] is not None
    assert runtime._dense_cache.value_cache[0] is not None
    assert runtime._dense_cache.key_cache[1] is None
    assert runtime._dense_cache.value_cache[1] is None


def test_token_posting_select_candidates_preserves_token_scale_signal() -> None:
    archive = posting_mod.TokenPostingArchive(
        retrieval_layers=[0],
        num_kv_groups=1,
        head_dim=4,
        basis_rank=2,
        ring_size=1,
        num_sinks=0,
        archive_capacity=8,
        token_topk=2,
        r_query=2,
        candidates=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    archive.load_basis(
        layer_idx=0,
        group_idx=0,
        basis=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        idf=np.asarray([1.0, 1.0], dtype=np.float32),
        key_mean=np.zeros((4,), dtype=np.float32),
    )

    k1 = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
    k2 = torch.tensor([[4.0, 4.0, 0.0, 0.0]], dtype=torch.float32)
    k3 = torch.tensor([[8.0, 8.0, 0.0, 0.0]], dtype=torch.float32)
    v = torch.zeros_like(k1)
    archive.append_token(0, 0, k1, v)
    archive.append_token(0, 1, k2, v)
    archive.append_token(0, 2, k3, v)

    # Both archived tokens have the same latent direction; larger scale should rank first.
    cand = archive.select_candidates(
        layer_idx=0,
        group_idx=0,
        q_cpu=np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        step=0,
        M=2,
    )
    assert cand.size == 2
    assert int(cand[0]) == 1
    assert int(cand[1]) == 0


def test_token_posting_fetch_shortlist_unions_multi_head_queries() -> None:
    archive = posting_mod.TokenPostingArchive(
        retrieval_layers=[0],
        num_kv_groups=1,
        head_dim=4,
        basis_rank=2,
        ring_size=1,
        num_sinks=0,
        archive_capacity=8,
        token_topk=1,
        r_query=1,
        candidates=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    archive.load_basis(
        layer_idx=0,
        group_idx=0,
        basis=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        idf=np.asarray([1.0, 1.0], dtype=np.float32),
        key_mean=np.zeros((4,), dtype=np.float32),
    )

    v = torch.zeros((1, 4), dtype=torch.float32)
    archive.append_token(0, 0, torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32), v)
    archive.append_token(0, 1, torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32), v)
    archive.append_token(0, 2, torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float32), v)

    q_multi = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    k_all, v_all = archive.fetch_shortlist_kv(layer_idx=0, group_idx=0, q_rep_gpu=q_multi, step=1, M=2)

    # 2 archive candidates (union of both queries) + 1 token in ring.
    assert int(k_all.shape[0]) == 3
    assert int(v_all.shape[0]) == 3


def test_token_posting_warmup_supports_legacy_cache_conversion() -> None:
    archive = posting_mod.TokenPostingArchive(
        retrieval_layers=[0],
        num_kv_groups=1,
        head_dim=4,
        basis_rank=2,
        ring_size=2,
        num_sinks=1,
        archive_capacity=8,
        token_topk=1,
        r_query=1,
        candidates=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    archive.load_basis(
        layer_idx=0,
        group_idx=0,
        basis=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        idf=np.asarray([1.0, 1.0], dtype=np.float32),
        key_mean=np.zeros((4,), dtype=np.float32),
    )

    # [B=1, G=1, T=3, D=4]
    k = torch.tensor(
        [[[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [[[[0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )

    class _LegacyOnlyCache:
        def to_legacy_cache(self):
            return ((k, v),)

    archive.warm_up_from_dense_cache(_LegacyOnlyCache(), seq_len=3)

    assert int(archive.sink_count[0]) == 1
    assert int(archive.ring_count[0]) == 2


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
