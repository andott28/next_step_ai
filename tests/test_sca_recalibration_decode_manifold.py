import ast
import inspect
import os
import sys
import types
import textwrap

import pytest
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from run_sca_recalibration_from_hybrid_baseline import (  # noqa: E402
    SCALocalRecalibrationConfig,
    _build_stage_plan,
    _compute_decode_manifold_rollout_alignment,
    _layerwise_hidden_cosine_loss,
    main as recalibration_main,
)


class _ToyDecodeManifoldModel(nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.vocab_size = int(vocab_size)
        self.neuroplasticity_enabled = True

    @property
    def device(self) -> torch.device:
        return self.scale.device

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        use_cache: bool,
        task_id: int,
    ) -> torch.Tensor:
        del attention_mask, do_sample, use_cache, task_id
        steps = torch.arange(1, max_new_tokens + 1, device=input_ids.device, dtype=input_ids.dtype).view(1, -1)
        next_tokens = (input_ids[:, -1:] + steps) % self.vocab_size
        return torch.cat([input_ids, next_tokens], dim=-1)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        output_hidden_states: bool,
        task_id: int,
    ):
        del attention_mask, use_cache, task_id
        base = torch.nn.functional.one_hot(input_ids % self.vocab_size, num_classes=self.vocab_size).float()
        if self.neuroplasticity_enabled:
            logits = base + self.scale.view(1, 1, 1)
            hidden = base + self.scale.view(1, 1, 1)
        else:
            logits = base
            hidden = base
        out = types.SimpleNamespace(
            logits=logits,
            hidden_states=(hidden * 0.5, hidden) if output_hidden_states else None,
        )
        if return_dict:
            return out
        return logits, out.hidden_states


def test_decode_manifold_stage_plan_uses_progressive_depth_groups_by_default():
    cfg = SCALocalRecalibrationConfig(
        model_name="toy",
        hybrid_checkpoint="hybrid.pt",
        output_dir="out",
        recalibration_mode="decode_manifold_alignment",
        steps=10,
        progressive_depth_enabled=True,
        progressive_depth_group_size=2,
    )
    plan = _build_stage_plan(selected_layers=[1, 2, 3, 4], total_steps=10, cfg=cfg)
    assert [stage["name"] for stage in plan] == ["progressive_depth_1", "progressive_depth_2"]
    assert plan[0]["layers"] == [1, 2]
    assert plan[0]["active_sparse_layers"] == [1, 2]
    assert plan[0]["steps"] == 5
    assert plan[1]["layers"] == [3, 4]
    assert plan[1]["active_sparse_layers"] == [1, 2, 3, 4]
    assert plan[1]["steps"] == 5
    assert plan[1]["lr_scale"] == 0.5


def test_decode_manifold_stage_plan_can_disable_progressive_depth():
    cfg = SCALocalRecalibrationConfig(
        model_name="toy",
        hybrid_checkpoint="hybrid.pt",
        output_dir="out",
        recalibration_mode="decode_manifold_alignment",
        stage1_steps_ratio=0.5,
        steps=10,
        progressive_depth_enabled=False,
    )
    plan = _build_stage_plan(selected_layers=[1, 2, 3, 4], total_steps=10, cfg=cfg)
    assert [stage["name"] for stage in plan] == ["decode_manifold_stage1", "decode_manifold_stage2"]
    assert plan[0]["layers"] == [1, 2]
    assert plan[0]["active_sparse_layers"] == [1, 2]
    assert plan[0]["steps"] == 6
    assert plan[1]["layers"] == [1, 2, 3, 4]
    assert plan[1]["active_sparse_layers"] == [1, 2, 3, 4]
    assert plan[1]["steps"] == 4


def test_decode_manifold_rollout_alignment_returns_finite_grad_losses():
    model = _ToyDecodeManifoldModel()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    rollout_kl, rollout_cos, aux = _compute_decode_manifold_rollout_alignment(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_id=0,
        max_new_tokens=2,
    )
    loss = rollout_kl + rollout_cos
    assert torch.isfinite(rollout_kl)
    assert torch.isfinite(rollout_cos)
    assert loss.requires_grad
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)
    assert aux["rollout_sequence_length"] == 10.0
    assert aux["rollout_batch_size"] == 1.0


def test_layerwise_hidden_cosine_loss_uses_selected_sparse_layers():
    attention_mask = torch.ones((1, 3), dtype=torch.long)
    dense_hidden_states = (
        torch.zeros((1, 3, 2), dtype=torch.float32),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32),
    )
    sparse_hidden_states = (
        torch.zeros((1, 3, 2), dtype=torch.float32),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32),
    )
    loss, metrics = _layerwise_hidden_cosine_loss(
        dense_hidden_states=dense_hidden_states,
        sparse_hidden_states=sparse_hidden_states,
        attention_mask=attention_mask,
        layer_indices=[0, 1],
    )
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.5, rel=1e-6)
    assert metrics["hidden_alignment_layers_used"] == 2.0
    assert metrics["hidden_alignment_layer_cosine_mean"] == pytest.approx(0.5, rel=1e-6)
    assert metrics["hidden_alignment_final_cosine_mean"] == pytest.approx(1.0, rel=1e-6)


def test_decode_manifold_balance_regularizer_is_not_nested_under_rollout_gate():
    source = textwrap.dedent(inspect.getsource(recalibration_main))
    tree = ast.parse(source)
    regularizer_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_compute_latent_support_regularizer"
    ]
    assert regularizer_calls, "expected decode-manifold path to call _compute_latent_support_regularizer"

    gated_regularizer_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "stage_idx"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.GtE)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == 1
        ):
            continue
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_compute_latent_support_regularizer"
            ):
                gated_regularizer_calls.append(child)
    assert not gated_regularizer_calls, "_compute_latent_support_regularizer should run outside the rollout stage gate"
