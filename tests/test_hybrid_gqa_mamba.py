import json
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from neuroplastic_llama_gqa_mamba import (  # noqa: E402
    HybridCollapsedMambaSelfAttention,
    NeuroplasticLlama,
)
from paged_sparse_attention import SparseAttentionConfig, SparseAttentionRuntime
from run_hybrid_gqa_mamba_inference import _build_arg_parser
from run_sca_recalibration_from_hybrid_baseline import _compute_latent_support_regularizer
from sca_sparse_config import SCASparseConfig


class _FakeAttention(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(delta, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        del attention_mask, position_ids, cache_position, position_embeddings, kwargs
        out = hidden_states + self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if use_cache:
            return out, None, past_key_value
        if output_attentions:
            return out, None
        return out, None


class _FakeDynamics(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a_logit = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.delta_log = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))


class _FakeGroup(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dynamics = _FakeDynamics()


class _FakeMambaBlock(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(delta, dtype=torch.float32))
        self.groups = nn.ModuleList([_FakeGroup()])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)


class _Layer(nn.Module):
    def __init__(self, self_attn: nn.Module) -> None:
        super().__init__()
        self.self_attn = self_attn


class _ModelWrapper(nn.Module):
    def __init__(self, layers: list[_Layer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _RootModel(nn.Module):
    def __init__(self, layers: list[_Layer]) -> None:
        super().__init__()
        self.model = _ModelWrapper(layers)


def _hybrid(delta_original: float, delta_mamba: float, mix_init: float = 0.5) -> HybridCollapsedMambaSelfAttention:
    return HybridCollapsedMambaSelfAttention(
        original_attn=_FakeAttention(delta_original),
        mamba_block=_FakeMambaBlock(delta_mamba),
        layer_idx=0,
        mix_init=mix_init,
        dtype=torch.float32,
    )


def _build_fake_neuroplastic_model(layers: list[_Layer]) -> NeuroplasticLlama:
    fake = object.__new__(NeuroplasticLlama)
    nn.Module.__init__(fake)
    fake.model = _RootModel(layers)
    fake.model_name = "toy-model"
    fake.attention_hybrid_target_rank = 16
    fake.attention_hybrid_variance_threshold = 0.90
    fake.attention_hybrid_state_dim = None
    fake.attention_hybrid_enabled = True
    fake.attention_hybrid_force_no_cache = True
    fake.attention_gqa_mamba_enabled = False
    fake.sca_sparse_mlps = []
    fake.strict_decode_upper_layer_cap_enabled = True
    return fake


def _sca_config(**kwargs) -> SCASparseConfig:
    base = dict(
        hidden_size=8,
        block_size=4,
        block_rank=2,
        basis_rank=4,
        basis_top_k=2,
        top_k=2,
        adaptive_top_k_min=1,
        adaptive_top_k_max=2,
        sigma=1.0,
        refractory_steps=0,
        inhibition_lambda=0.0,
        use_cuda=False,
        grid_size=2,
        spmm_impl="dense",
        soft_mask=False,
    )
    base.update(kwargs)
    return SCASparseConfig(**base)


def test_hybrid_attention_mix_zero_matches_original():
    attn = _hybrid(delta_original=1.0, delta_mamba=-1.0)
    attn.set_mix_value(0.0)
    hidden = torch.randn(2, 4, 6)
    out, _ = attn(hidden, use_cache=False)
    expected = hidden + 1.0
    assert torch.allclose(out, expected, atol=1e-4, rtol=1e-4)


def test_hybrid_attention_mix_one_matches_mamba():
    attn = _hybrid(delta_original=1.0, delta_mamba=-1.0)
    attn.set_mix_value(1.0)
    hidden = torch.randn(2, 4, 6)
    out, _ = attn(hidden, use_cache=False)
    expected = hidden - 1.0
    assert torch.allclose(out, expected, atol=1e-4, rtol=1e-4)


def test_hybrid_attention_state_round_trip(tmp_path):
    model_a = _build_fake_neuroplastic_model([_Layer(_hybrid(1.0, -1.0, mix_init=0.2))])
    exported = NeuroplasticLlama.export_hybrid_attention_state(model_a)
    ckpt = tmp_path / "hybrid_attention_state.pt"
    torch.save(
        {
            "model_name": "toy-model",
            "layer_selection": exported["layer_selection"],
            "hybrid_attention_state_dict": exported["hybrid_attention_state_dict"],
            "mix_values_by_layer": exported["mix_values_by_layer"],
        },
        ckpt,
    )

    model_b = _build_fake_neuroplastic_model([_Layer(_hybrid(2.0, 3.0, mix_init=0.8))])
    info = NeuroplasticLlama.load_hybrid_attention_state(model_b, str(ckpt), strict=True)
    assert info == {"loaded_layers": 1, "missing_layers": 0}
    layer_attn = model_b.model.model.layers[0].self_attn
    assert abs(float(layer_attn.mix_value.item()) - float(model_a.model.model.layers[0].self_attn.mix_value.item())) < 1e-5


def test_hybrid_layer_subset_loading(tmp_path):
    model_a = _build_fake_neuroplastic_model([_Layer(_hybrid(1.0, -1.0)), _Layer(_hybrid(2.0, -2.0))])
    exported = NeuroplasticLlama.export_hybrid_attention_state(model_a)
    exported["hybrid_attention_state_dict"].pop("layers.1.self_attn")
    ckpt = tmp_path / "hybrid_subset.pt"
    torch.save(exported, ckpt)

    model_b = _build_fake_neuroplastic_model([_Layer(_hybrid(0.0, 0.0)), _Layer(_hybrid(0.0, 0.0))])
    info = NeuroplasticLlama.load_hybrid_attention_state(model_b, str(ckpt), strict=False)
    assert info == {"loaded_layers": 1, "missing_layers": 1}


def test_local_geometry_calibration_unfreezes_only_hybrid_fp_params():
    hybrid_layer = _Layer(_hybrid(1.0, -1.0))
    plain_layer = _Layer(_FakeAttention(0.5))
    model = _build_fake_neuroplastic_model([hybrid_layer, plain_layer])
    trainable = NeuroplasticLlama.prepare_local_geometry_calibration(model, include_output_bias=False)

    assert trainable
    hybrid_attn = hybrid_layer.self_attn
    assert hybrid_attn.mix_logit.requires_grad
    assert hybrid_attn.output_gain.requires_grad
    assert not hybrid_attn.output_bias.requires_grad
    assert not hybrid_attn.original_attn.weight.requires_grad
    assert hybrid_attn.mamba_block.groups[0].dynamics.a_logit.requires_grad
    assert hybrid_attn.mamba_block.groups[0].dynamics.delta_log.requires_grad
    assert not plain_layer.self_attn.weight.requires_grad


def test_generation_disables_cache_in_hybrid_mode():
    model = _build_fake_neuroplastic_model([])
    assert NeuroplasticLlama._should_disable_generation_cache(model, True) is True


def test_sparse_attention_mode_keeps_cache_enabled_even_in_hybrid_mode():
    model = _build_fake_neuroplastic_model([])
    model.sparse_attention_config = types.SimpleNamespace(enabled=True)
    assert NeuroplasticLlama._should_disable_generation_cache(model, True) is False


def test_sparse_layer_enabled_respects_bottom_buffer_and_top_guard():
    model = _build_fake_neuroplastic_model([_Layer(_FakeAttention(0.0)) for _ in range(6)])
    model.neuroplasticity_enabled = True
    model.bottom_buffer_layers = 2
    model.buffer_layers = 1
    model.decode_guard_layers = 1
    expected_sparse = {2, 3}
    for layer_idx in range(6):
        is_sparse = NeuroplasticLlama._sparse_layer_enabled(model, layer_idx)
        assert is_sparse is (layer_idx in expected_sparse)


class _DummySparseWrapper:
    def __init__(self) -> None:
        self.loaded = False

    def clear_block_bank(self) -> None:
        self.loaded = False

    def load_block_bank(self, payload, strict: bool = True) -> None:
        del strict
        if not isinstance(payload, dict):
            raise RuntimeError("payload must be a dict")
        self.loaded = True

    def has_block_bank(self) -> bool:
        return bool(self.loaded)


class _DummySparseAlignmentWrapper:
    def __init__(self, alignment=None, route_snapshot=None) -> None:
        self._enabled = False
        self._alignment = alignment
        self._route_enabled = False
        self._route_snapshot = route_snapshot
        self._last_fallback_triggered = False

    def set_alignment_capture(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def get_last_alignment(self):
        return self._alignment if self._enabled else None

    def set_route_capture(self, enabled: bool) -> None:
        self._route_enabled = bool(enabled)

    def get_last_route_snapshot(self):
        return self._route_snapshot if self._route_enabled else None


class _DummyBasisWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self._enabled = False

    def set_alignment_capture(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def get_last_alignment(self):
        return None

    def set_route_capture(self, enabled: bool) -> None:
        del enabled

    def get_last_route_snapshot(self):
        return None

    def iter_sparse_basis_parameters(self):
        return [("sparse_basis_decoder", self.weight), ("sparse_basis_bias", self.bias)]

    def export_sparse_recalibration_state(self):
        return {
            "sparse_basis_decoder": self.weight.detach().cpu(),
            "sparse_basis_bias": self.bias.detach().cpu(),
        }

    def load_sparse_recalibration_state(self, payload, strict: bool = True):
        loaded = 0
        missing = 0
        for key, param in (("sparse_basis_decoder", self.weight), ("sparse_basis_bias", self.bias)):
            tensor = payload.get(key)
            if tensor is None:
                if strict:
                    raise RuntimeError(f"missing {key}")
                missing += 1
                continue
            param.data = tensor.to(device=param.device, dtype=param.dtype)
            loaded += 1
        return {"loaded_items": loaded, "missing_items": missing}

    def initialize_sparse_basis_from_dense_init(self, payload, strict: bool = True):
        loaded = 0
        missing = 0
        for key, param in (("decoder_blocks", self.weight), ("decoder_bias", self.bias)):
            tensor = payload.get(key)
            if tensor is None:
                if strict:
                    raise RuntimeError(f"missing {key}")
                missing += 1
                continue
            param.data = tensor.to(device=param.device, dtype=param.dtype)
            loaded += 1
        return {"loaded_items": loaded, "missing_items": missing}


class _DummyRegularizerWrapper:
    def __init__(self, *, latent_importance, latent_load, block_importance, block_load) -> None:
        self._stats = {
            "mean_latent_norm": 1.0,
            "active_latent_fraction": 0.5,
            "active_latent_coords_used": float(len(latent_importance)),
            "latent_usage_entropy": 0.0,
            "support_overlap_mean": 0.0,
            "support_unique_fraction": 1.0,
            "max_latent_importance_fraction": float(max(latent_importance)),
            "max_latent_load_fraction": float(max(latent_load)),
            "max_block_importance_fraction": float(max(block_importance)),
            "max_block_load_fraction": float(max(block_load)),
        }
        self._reg = {
            "_latent_importance_probs": torch.tensor(latent_importance, dtype=torch.float32),
            "_latent_load_probs": torch.tensor(latent_load, dtype=torch.float32),
            "_block_importance_probs": torch.tensor(block_importance, dtype=torch.float32),
            "_block_load_probs": torch.tensor(block_load, dtype=torch.float32),
        }

    def get_last_learned_basis_stats(self):
        return self._stats

    def get_last_learned_basis_regularizer_tensors(self):
        return self._reg


class _ToyInnerModel(nn.Module):
    def __init__(self, hidden_size: int = 8, vocab_size: int = 8) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleList([])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        del attention_mask, position_ids, use_cache, cache_position, output_attentions, kwargs
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        hidden = self.norm(hidden)
        hidden_states = (hidden,) if output_hidden_states else None
        out = types.SimpleNamespace(
            last_hidden_state=hidden,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None,
        )
        if return_dict:
            return out
        return (hidden, past_key_values, hidden_states, None)


class _ToyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 8, vocab_size: int = 8) -> None:
        super().__init__()
        self.model = _ToyInnerModel(hidden_size=hidden_size, vocab_size=vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = types.SimpleNamespace(vocab_size=vocab_size, use_return_dict=True)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def loss_function(self, logits, labels, vocab_size: int, **kwargs):
        del kwargs
        return F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=-100)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        result = types.SimpleNamespace(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        effective_return_dict = self.config.use_return_dict if return_dict is None else bool(return_dict)
        if effective_return_dict:
            return result
        values = (loss, logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
        return tuple(v for v in values if v is not None)


def test_sparse_mlp_manifest_loading_round_trip(tmp_path):
    model = object.__new__(NeuroplasticLlama)
    nn.Module.__init__(model)
    model.sca_sparse_mlps = [_DummySparseWrapper(), _DummySparseWrapper()]
    model._sparse_mlp_bank_manifest_path = None

    for layer_idx in range(2):
        layer_payload = {
            "layer_idx": layer_idx,
            "num_blocks": 1,
            "block_size": 1,
            "gate_proj_blocks": torch.zeros(1, 1, 1),
            "up_proj_blocks": torch.zeros(1, 1, 1),
            "down_proj_blocks": torch.zeros(1, 1, 1),
            "gate_bias": None,
            "up_bias": None,
            "down_bias": None,
        }
        torch.save(layer_payload, tmp_path / f"layer_{layer_idx:03d}_mlp_blocks.pt")

    manifest = {
        "model_name": "toy-model",
        "sca_config": {"block_size": 1},
        "layers": [
            {"layer_idx": 0, "file": "layer_000_mlp_blocks.pt"},
            {"layer_idx": 1, "file": "layer_001_mlp_blocks.pt"},
        ],
    }
    manifest_path = tmp_path / "mlp_bank_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    info = NeuroplasticLlama.load_sparse_mlp_bank_manifest(model, str(manifest_path), strict=True)
    status = NeuroplasticLlama.get_sparse_mlp_bank_status(model)
    assert info == {"loaded_layers": 2, "missing_layers": 0}
    assert status["loaded_layers"] == 2


def _build_fake_sca_model_for_recalibration() -> NeuroplasticLlama:
    model = object.__new__(NeuroplasticLlama)
    nn.Module.__init__(model)
    layers = [_Layer(_FakeAttention(0.0)), _Layer(_FakeAttention(0.0))]
    model.model = _RootModel(layers)
    model.model_name = "toy-model"
    model.attention_hybrid_enabled = False
    model.attention_hybrid_target_rank = 16
    model.attention_hybrid_variance_threshold = 0.9
    model.attention_hybrid_state_dim = None
    model.attention_hybrid_force_no_cache = True
    model.sca_sparse_mlps = [_DummySparseAlignmentWrapper(), _DummySparseAlignmentWrapper()]
    model.sca_config = _sca_config()
    model._sparse_mlp_bank_manifest_path = None
    model.disable_task_bias_injection = False
    model.spatial_proj = nn.Linear(8, 3, bias=True)
    model.task_embedding = nn.Embedding(4, 8)
    model.adapter_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    model.lm_head = nn.Linear(8, 8, bias=False)
    model.strict_decode_upper_layer_cap_enabled = True
    model._sca_recalibration_layer_indices = []
    model._sca_recalibration_mode = "local_mlp_geometry"
    model._sca_recalibration_trainable_modules = []
    model._sca_recalibration_hybrid_checkpoint_path = ""
    return model


def _build_forward_ready_decoder_mirror_model() -> NeuroplasticLlama:
    model = object.__new__(NeuroplasticLlama)
    nn.Module.__init__(model)
    model.model = _ToyCausalLM(hidden_size=8, vocab_size=8)
    model.config = types.SimpleNamespace(vocab_size=8)
    model.model_name = "toy-model"
    model.hidden_size = 8
    model.num_tasks = 4
    model.neuroplasticity_enabled = True
    model.buffer_layers = 0
    model.collect_bio_gate_telemetry = False
    model.bounded_context_config = types.SimpleNamespace(enabled=False)
    model._tier_policy = None
    model._tier_cache_compressor = None
    model._tier_mask_bundle = None
    model._latest_tier_metadata = None
    model._summary_manager = types.SimpleNamespace(snapshot=lambda: types.SimpleNamespace(summary_token_ids=None))
    model._summary_provider = None
    model._repo_index = None
    model.sparse_attention_config = types.SimpleNamespace(enabled=False)
    model._sparse_attention_runtime = None
    model._last_sparse_attention_step = {}
    model.sca_config = _sca_config(block_size=2, top_k=1)
    model.task_embedding = nn.Embedding(4, 8)
    model.spatial_proj = nn.Linear(8, 3, bias=True)
    model.adapter_scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
    model.score = nn.Linear(8, 1)
    model.block_centers = torch.zeros(4, 3)
    model.inhibition_matrix = torch.zeros(4, 4)
    model.refractory_until = torch.zeros((1, 4), dtype=torch.int32)
    model._routing_decode_mode = False
    model._routing_step = 0
    route_snapshot = {
        "active_idx": torch.zeros((2, 1), dtype=torch.long),
        "score_weights": torch.ones((2, 1), dtype=torch.float32),
        "rows": torch.tensor(2, dtype=torch.int32),
        "batch_size": torch.tensor(1, dtype=torch.int32),
        "seq_len": torch.tensor(2, dtype=torch.int32),
        "layer_idx": torch.tensor(0, dtype=torch.int32),
    }
    model.sca_sparse_mlps = [_DummySparseAlignmentWrapper(route_snapshot=route_snapshot)]
    model.attention_hybrid_enabled = False
    model.attention_hybrid_target_rank = 16
    model.attention_hybrid_variance_threshold = 0.9
    model.attention_hybrid_state_dim = None
    model.attention_hybrid_force_no_cache = True
    model.attention_gqa_mamba_enabled = False
    model._sparse_mlp_bank_manifest_path = None
    model._current_task_id = 0
    model._cached_task_emb = None
    model.disable_task_bias_injection = True
    model.strict_decode_upper_layer_cap_enabled = True
    model._sca_recalibration_layer_indices = []
    model._sca_recalibration_mode = "local_mlp_geometry"
    model._sca_recalibration_trainable_modules = []
    model._sca_recalibration_hybrid_checkpoint_path = ""
    model.decoder_mirror_config = None
    model.decoder_mirror = None
    model._decoder_mirror_enabled = False
    model._last_decoder_mirror_diagnostics = {}
    model._decoder_mirror_calibration_mode = "decoder_co_warp"
    model._decoder_mirror_trainable_modules = []
    model._decoder_mirror_hybrid_checkpoint_path = ""
    model._decoder_mirror_sca_checkpoint_path = ""
    model._decoder_mirror_source_layers = []
    model._decoder_mirror_route_prior_missing = True
    model._decoder_mirror_source_layers_used = []
    model._decoder_mirror_init_kwargs = {
        "top_k": 1,
        "rank": 2,
        "route_conditioned": True,
        "source_layers": [0],
        "route_prior_scale_init": 0.25,
        "residual_scale_init": 0.0,
    }
    return model


def test_prepare_sca_local_recalibration_freezes_output_stack():
    model = _build_fake_sca_model_for_recalibration()
    for p in model.parameters():
        p.requires_grad = True
    trainable = NeuroplasticLlama.prepare_sca_local_recalibration(model, include_spatial_proj=True, include_task_embedding=False)

    assert trainable
    assert all(p.requires_grad is False for p in model.lm_head.parameters())
    assert all(p.requires_grad is False for p in model.model.model.layers[0].self_attn.parameters())
    assert any(p.requires_grad for p in model.spatial_proj.parameters())


def test_prepare_sca_local_recalibration_freezes_task_embedding_by_default():
    model = _build_fake_sca_model_for_recalibration()
    NeuroplasticLlama.prepare_sca_local_recalibration(model, include_spatial_proj=True, include_task_embedding=False)
    assert all(p.requires_grad is False for p in model.task_embedding.parameters())


def test_compute_sca_local_recalibration_loss_decreases_when_sparse_matches_dense():
    model = _build_fake_sca_model_for_recalibration()
    dense = torch.randn(1, 2, 8)
    mismatch = dense + 0.5
    model.sca_sparse_mlps = [
        _DummySparseAlignmentWrapper(
            alignment={
                "dense_mlp_out": dense,
                "sparse_mlp_out": mismatch,
                "fallback_triggered": torch.tensor(1.0),
            }
        )
    ]
    NeuroplasticLlama.prepare_sca_local_recalibration(model, include_spatial_proj=True, include_task_embedding=False)
    loss_bad, metrics_bad = NeuroplasticLlama.compute_sca_local_recalibration_loss(model, loss_mode="mse_plus_norm")
    model.sca_sparse_mlps[0]._alignment["sparse_mlp_out"] = dense.clone()
    loss_good, _metrics_good = NeuroplasticLlama.compute_sca_local_recalibration_loss(model, loss_mode="mse_plus_norm")
    assert float(loss_good.item()) <= float(loss_bad.item())
    assert metrics_bad["fallback_rate"] >= 0.0


def test_export_sca_recalibration_state_round_trip(tmp_path):
    model_a = _build_fake_sca_model_for_recalibration()
    NeuroplasticLlama.prepare_sca_local_recalibration(
        model_a,
        include_spatial_proj=True,
        include_task_embedding=True,
        layer_indices=[0],
        hybrid_checkpoint_path="hybrid.pt",
    )
    with torch.no_grad():
        model_a.spatial_proj.weight.fill_(0.123)
        model_a.task_embedding.weight.fill_(0.321)
    payload = NeuroplasticLlama.export_sca_recalibration_state(model_a)
    ckpt = tmp_path / "sca_recalibrated_state.pt"
    torch.save(payload, ckpt)

    model_b = _build_fake_sca_model_for_recalibration()
    info = NeuroplasticLlama.load_sca_recalibration_state(model_b, str(ckpt), strict=True)
    assert info["loaded_items"] >= 1
    assert torch.allclose(model_b.spatial_proj.weight, model_a.spatial_proj.weight)
    assert torch.allclose(model_b.task_embedding.weight, model_a.task_embedding.weight)


def test_export_sca_recalibration_state_round_trip_with_learned_basis_wrapper(tmp_path):
    model_a = _build_fake_sca_model_for_recalibration()
    model_a.sca_sparse_mlps = [_DummyBasisWrapper(), _DummyBasisWrapper()]
    model_a.sca_config = _sca_config(
        sparse_placement="learned_basis",
        routing_mode="semantic_latent",
        basis_rank_by_layer={0: 3},
        basis_top_k_by_layer={0: 2},
        top_k_by_layer={0: 1},
    )
    NeuroplasticLlama.prepare_sca_local_recalibration(
        model_a,
        include_spatial_proj=True,
        include_task_embedding=False,
        layer_indices=[0],
        hybrid_checkpoint_path="hybrid.pt",
    )
    with torch.no_grad():
        model_a.sca_sparse_mlps[0].weight.fill_(2.5)
        model_a.sca_sparse_mlps[0].bias.fill_(1.5)
    payload = NeuroplasticLlama.export_sca_recalibration_state(model_a)
    ckpt = tmp_path / "sca_recalibrated_learned_basis_state.pt"
    torch.save(payload, ckpt)

    model_b = _build_fake_sca_model_for_recalibration()
    model_b.sca_sparse_mlps = [_DummyBasisWrapper(), _DummyBasisWrapper()]
    model_b.sca_config = _sca_config()
    info = NeuroplasticLlama.load_sca_recalibration_state(model_b, str(ckpt), strict=True)
    assert info["loaded_items"] >= 3
    assert model_b.sca_config.sparse_placement == "learned_basis"
    assert model_b.sca_config.routing_mode == "semantic_latent"
    assert model_b.sca_config.basis_rank_by_layer == {0: 3}
    assert model_b.sca_config.basis_top_k_by_layer == {0: 2}
    assert model_b.sca_config.top_k_by_layer == {0: 1}
    assert torch.allclose(model_b.sca_sparse_mlps[0].weight, model_a.sca_sparse_mlps[0].weight)
    assert torch.allclose(model_b.sca_sparse_mlps[0].bias, model_a.sca_sparse_mlps[0].bias)


def test_load_learned_basis_init_into_wrappers(tmp_path):
    model = _build_fake_sca_model_for_recalibration()
    model.sca_sparse_mlps = [_DummyBasisWrapper(), _DummyBasisWrapper()]
    model.sca_config = _sca_config(
        sparse_placement="learned_basis",
        routing_mode="semantic_latent",
        basis_rank=1,
        basis_top_k=1,
        top_k=1,
    )
    ckpt = tmp_path / "learned_basis_init.pt"
    payload = {
        "config": {
            "sparse_placement": "learned_basis",
            "routing_mode": "semantic_latent",
            "basis_rank": 1,
            "basis_rank_by_layer": {"0": 1},
            "basis_top_k_by_layer": {"0": 1},
            "top_k_by_layer": {"0": 1},
        },
        "layer_states": {
            "0": {"decoder_blocks": torch.tensor([3.0]), "decoder_bias": torch.tensor([4.0])},
            "1": {"decoder_blocks": torch.tensor([5.0]), "decoder_bias": torch.tensor([6.0])},
        },
    }
    torch.save(payload, ckpt)
    info = NeuroplasticLlama.load_learned_basis_init(model, str(ckpt), strict=True)
    assert info["loaded_items"] >= 4
    assert model.sca_config.routing_mode == "semantic_latent"
    assert model.sca_config.basis_rank_by_layer == {0: 1}
    assert model.sca_config.basis_top_k_by_layer == {0: 1}
    assert model.sca_config.top_k_by_layer == {0: 1}
    assert torch.allclose(model.sca_sparse_mlps[0].weight, torch.tensor([3.0]))
    assert torch.allclose(model.sca_sparse_mlps[0].bias, torch.tensor([4.0]))


def test_prepare_sca_local_recalibration_decode_manifold_alignment_freezes_spatial_proj():
    model = _build_fake_sca_model_for_recalibration()
    for p in model.parameters():
        p.requires_grad = True
    NeuroplasticLlama.prepare_sca_local_recalibration(
        model,
        include_spatial_proj=True,
        include_task_embedding=True,
        recalibration_mode="decode_manifold_alignment",
    )
    assert all(p.requires_grad is False for p in model.spatial_proj.parameters())
    assert all(p.requires_grad is False for p in model.task_embedding.parameters())


def test_prepare_sca_local_recalibration_can_run_lower_layers_sparse_but_frozen():
    model = _build_fake_sca_model_for_recalibration()
    NeuroplasticLlama.prepare_sca_local_recalibration(
        model,
        include_spatial_proj=False,
        include_task_embedding=False,
        layer_indices=[1],
        active_sparse_layer_indices=[0, 1],
        recalibration_mode="decode_manifold_alignment",
    )
    assert model._sca_recalibration_layer_indices == [1]
    assert model._sca_recalibration_active_sparse_layer_indices == [0, 1]
    assert model._sca_sparse_layer_override == {0, 1}


def test_export_sca_recalibration_state_keeps_all_active_sparse_layers(tmp_path):
    model_a = _build_fake_sca_model_for_recalibration()
    model_a.sca_sparse_mlps = [_DummyBasisWrapper(), _DummyBasisWrapper(), _DummyBasisWrapper()]
    model_a.sca_config = _sca_config(
        sparse_placement="learned_basis",
        routing_mode="semantic_latent",
        basis_rank_by_layer={0: 3, 1: 3, 2: 3},
        basis_top_k_by_layer={0: 2, 1: 2, 2: 2},
        top_k_by_layer={0: 1, 1: 1, 2: 1},
    )
    NeuroplasticLlama.prepare_sca_local_recalibration(
        model_a,
        include_spatial_proj=False,
        include_task_embedding=False,
        layer_indices=[2],
        active_sparse_layer_indices=[0, 1, 2],
        recalibration_mode="decode_manifold_alignment",
        hybrid_checkpoint_path="hybrid.pt",
    )
    with torch.no_grad():
        model_a.sca_sparse_mlps[0].weight.fill_(1.25)
        model_a.sca_sparse_mlps[1].weight.fill_(2.5)
        model_a.sca_sparse_mlps[2].weight.fill_(3.75)
    payload = NeuroplasticLlama.export_sca_recalibration_state(model_a)
    assert payload["layer_selection"] == [0, 1, 2]
    assert payload["active_sparse_layer_selection"] == [0, 1, 2]
    assert sorted(int(k) for k in payload["sparse_mlp_wrapper_state"].keys()) == [0, 1, 2]

    ckpt = tmp_path / "progressive_sparse_state.pt"
    torch.save(payload, ckpt)
    model_b = _build_fake_sca_model_for_recalibration()
    model_b.sca_sparse_mlps = [_DummyBasisWrapper(), _DummyBasisWrapper(), _DummyBasisWrapper()]
    model_b.sca_config = _sca_config()
    info = NeuroplasticLlama.load_sca_recalibration_state(model_b, str(ckpt), strict=True)
    assert info["loaded_items"] >= 3
    assert model_b._sca_recalibration_layer_indices == [0, 1, 2]
    assert model_b._sca_recalibration_active_sparse_layer_indices == [0, 1, 2]
    assert torch.allclose(model_b.sca_sparse_mlps[0].weight, model_a.sca_sparse_mlps[0].weight)
    assert torch.allclose(model_b.sca_sparse_mlps[1].weight, model_a.sca_sparse_mlps[1].weight)
    assert torch.allclose(model_b.sca_sparse_mlps[2].weight, model_a.sca_sparse_mlps[2].weight)


def test_latent_support_regularizer_penalizes_collapsed_routing_more_than_balanced():
    collapsed = _build_fake_sca_model_for_recalibration()
    collapsed.sca_sparse_mlps = [
        _DummyRegularizerWrapper(
            latent_importance=[1.0, 0.0, 0.0],
            latent_load=[1.0, 0.0, 0.0],
            block_importance=[1.0, 0.0],
            block_load=[1.0, 0.0],
        )
    ]
    balanced = _build_fake_sca_model_for_recalibration()
    balanced.sca_sparse_mlps = [
        _DummyRegularizerWrapper(
            latent_importance=[1 / 3, 1 / 3, 1 / 3],
            latent_load=[1 / 3, 1 / 3, 1 / 3],
            block_importance=[0.5, 0.5],
            block_load=[0.5, 0.5],
        )
    ]

    collapsed_latent, collapsed_block, collapsed_metrics = _compute_latent_support_regularizer(collapsed)
    balanced_latent, balanced_block, balanced_metrics = _compute_latent_support_regularizer(balanced)

    assert float(collapsed_latent.item()) > float(balanced_latent.item())
    assert float(collapsed_block.item()) > float(balanced_block.item())
    assert collapsed_metrics["max_latent_importance_fraction"] > balanced_metrics["max_latent_importance_fraction"]
    assert collapsed_metrics["max_block_importance_fraction"] > balanced_metrics["max_block_importance_fraction"]


def test_disable_task_bias_injection_recalibration_flag():
    model = _build_fake_sca_model_for_recalibration()
    assert model.disable_task_bias_injection is False
    model.disable_task_bias_injection = True
    assert model.disable_task_bias_injection is True


def test_same_model_logits_reference_uses_dense_disabled_pass_only():
    class _DualStreamProbe(NeuroplasticLlama):
        def __init__(self):
            nn.Module.__init__(self)
            self.neuroplasticity_enabled = True
            self.calls = []
            self.sca_config = types.SimpleNamespace(num_blocks=8, top_k=2)

        def forward(self, *args, **kwargs):
            del args, kwargs
            self.calls.append(bool(self.neuroplasticity_enabled))
            return types.SimpleNamespace(logits=torch.zeros(1, 2, 3))

    probe = _DualStreamProbe()
    _ = NeuroplasticLlama.forward_dual_stream(probe, input_ids=torch.ones(1, 2, dtype=torch.long))
    assert probe.calls == [False, True]
    assert probe.neuroplasticity_enabled is True


def test_recalibration_metrics_report_fallback_rate():
    model = _build_fake_sca_model_for_recalibration()
    dense = torch.randn(1, 2, 8)
    model.sca_sparse_mlps = [
        _DummySparseAlignmentWrapper(
            alignment={
                "dense_mlp_out": dense,
                "sparse_mlp_out": dense.clone(),
                "fallback_triggered": torch.tensor(1.0),
            }
        )
    ]
    NeuroplasticLlama.prepare_sca_local_recalibration(model, include_spatial_proj=True, include_task_embedding=False)
    _loss, metrics = NeuroplasticLlama.compute_sca_local_recalibration_loss(model, loss_mode="mse")
    assert "fallback_rate" in metrics
    assert "fallback_rate_by_layer" in metrics


def test_prepare_decoder_mirror_calibration_freezes_output_stack():
    model = _build_forward_ready_decoder_mirror_model()
    for p in model.parameters():
        p.requires_grad = True
    trainable = NeuroplasticLlama.prepare_decoder_mirror_calibration(
        model,
        source_layer_indices=[0],
        top_k=1,
        rank=2,
        route_conditioned=True,
    )
    assert trainable
    assert all(p.requires_grad is False for p in model.model.lm_head.parameters())
    assert all(p.requires_grad is False for p in model.model.model.parameters())
    assert any(p.requires_grad for p in model.decoder_mirror.parameters())


def test_forward_bypasses_decoder_mirror_when_neuroplasticity_disabled():
    class _ExplodingMirror(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = types.SimpleNamespace(enabled=True, source_layer_indices=[0])

        def forward(self, *args, **kwargs):
            raise RuntimeError("decoder mirror should be bypassed")

        def get_last_diagnostics(self):
            return {}

    model = _build_forward_ready_decoder_mirror_model()
    model.decoder_mirror = _ExplodingMirror()
    model.decoder_mirror_config = types.SimpleNamespace(source_layer_indices=[0], enabled=True)
    model._decoder_mirror_enabled = True
    model.neuroplasticity_enabled = False
    out = NeuroplasticLlama.forward(model, input_ids=torch.ones(1, 2, dtype=torch.long), return_dict=True, task_id=0)
    assert out.logits.shape[:2] == (1, 2)


def test_forward_uses_custom_pre_lm_head_path_when_decoder_mirror_enabled():
    class _AdditiveMirror(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.called = False
            self.config = types.SimpleNamespace(enabled=True, source_layer_indices=[0])
            self.route_prior_scale = nn.Parameter(torch.tensor(0.25))
            self.residual_scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, hidden_states, route_prior=None, source_layers_used=None):
            del route_prior, source_layers_used
            self.called = True
            return hidden_states + 1.0, {
                "enabled": True,
                "mean_active_blocks": 1.0,
                "delta_norm_ratio": 1.0,
                "route_prior_missing": False,
                "source_layers_used": [0],
            }

        def get_last_diagnostics(self):
            return {"enabled": True}

    model = _build_forward_ready_decoder_mirror_model()
    baseline = model.model(input_ids=torch.ones(1, 2, dtype=torch.long), return_dict=True).logits
    model.decoder_mirror = _AdditiveMirror()
    model.decoder_mirror_config = types.SimpleNamespace(source_layer_indices=[0], enabled=True)
    model._decoder_mirror_enabled = True
    model.set_decoder_mirror_route_capture(True, layer_indices=[0])
    out = NeuroplasticLlama.forward(model, input_ids=torch.ones(1, 2, dtype=torch.long), return_dict=True, task_id=0)
    assert model.decoder_mirror.called is True
    assert not torch.allclose(out.logits, baseline)


def test_decoder_mirror_diagnostics_present_after_forward():
    model = _build_forward_ready_decoder_mirror_model()
    NeuroplasticLlama.prepare_decoder_mirror_calibration(model, source_layer_indices=[0], top_k=1, rank=2)
    out = NeuroplasticLlama.forward(model, input_ids=torch.ones(1, 2, dtype=torch.long), return_dict=True, task_id=0)
    assert out.logits.shape[:2] == (1, 2)
    diagnostics = NeuroplasticLlama.get_decoder_mirror_diagnostics(model)
    assert "mean_active_blocks" in diagnostics
    assert "route_prior_missing" in diagnostics
    assert diagnostics["enabled"] is True


def test_decoder_mirror_export_load_round_trip(tmp_path):
    model_a = _build_forward_ready_decoder_mirror_model()
    NeuroplasticLlama.prepare_decoder_mirror_calibration(model_a, source_layer_indices=[0], top_k=1, rank=2)
    with torch.no_grad():
        model_a.decoder_mirror.route_prior_scale.fill_(1.25)
    payload = NeuroplasticLlama.export_decoder_mirror_state(model_a)
    ckpt = tmp_path / "decoder_mirror_state.pt"
    torch.save(payload, ckpt)

    model_b = _build_forward_ready_decoder_mirror_model()
    info = NeuroplasticLlama.load_decoder_mirror_state(model_b, str(ckpt), strict=True)
    assert info["loaded_items"] == 1
    assert torch.allclose(
        model_a.decoder_mirror.route_prior_scale.float(),
        model_b.decoder_mirror.route_prior_scale.float(),
    )


def test_inference_loader_accepts_decoder_mirror_checkpoint():
    parser = _build_arg_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "hybrid.pt",
            "--decoder-mirror-checkpoint",
            "decoder_mirror.pt",
        ]
    )
    assert args.decoder_mirror_checkpoint == "decoder_mirror.pt"
    assert args.sca_routing_mode == "semantic_latent"
    assert args.sca_bottom_buffer_layers == 2
    assert args.strict_decode_upper_layer_cap_enabled is False


def test_sparse_attention_mode_set_and_reset():
    model = object.__new__(NeuroplasticLlama)
    nn.Module.__init__(model)
    model.sparse_attention_config = SparseAttentionConfig(enabled=False)
    model._sparse_attention_runtime = SparseAttentionRuntime(
        config=model.sparse_attention_config,
        num_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=4,
    )
    model._last_sparse_attention_step = {}

    NeuroplasticLlama.set_sparse_attention_mode(
        model,
        enabled=True,
        local_window_tokens=32,
        page_size_tokens=8,
        retrieval_top_k_pages=2,
        strict_fully_sparse=True,
    )
    assert model.sparse_attention_config.enabled is True
    assert model.sparse_attention_config.local_window_tokens == 32
    assert model.sparse_attention_config.page_size_tokens == 8
    assert model.sparse_attention_config.retrieval_top_k_pages == 2
    assert model.sparse_attention_config.strict_fully_sparse is True

    NeuroplasticLlama.reset_sparse_attention_state(model)
    diag = NeuroplasticLlama.get_sparse_attention_diagnostics(model)
    assert diag["steps"] == 0
