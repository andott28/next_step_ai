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

from strict_decode_metrics import (  # noqa: E402
    accumulate_balance_from_model,
    dense_top1_ranks,
    evaluate_decode_prefixes,
    final_hidden_cosine,
    init_balance_accumulator,
    summarize_balance,
)


class _DummyWrapper:
    def __init__(self, reg):
        self._reg = reg

    def get_last_learned_basis_regularizer_tensors(self):
        return self._reg


class _ToyTokenizer:
    def decode(self, ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        return " ".join(str(int(v)) for v in ids.tolist())


class _ToyDecodeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._device_anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.neuroplasticity_enabled = True
        self.vocab_size = 5
        reg = {
            "_latent_importance_probs": torch.tensor([0.6, 0.4], dtype=torch.float32),
            "_latent_load_probs": torch.tensor([0.5, 0.5], dtype=torch.float32),
            "_block_importance_probs": torch.tensor([0.7, 0.3], dtype=torch.float32),
            "_block_load_probs": torch.tensor([0.4, 0.6], dtype=torch.float32),
        }
        self.sca_sparse_mlps = [_DummyWrapper(reg)]

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

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
        append = torch.full((input_ids.shape[0], max_new_tokens), 1, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, append], dim=-1)

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
        one_hot = torch.nn.functional.one_hot(input_ids % self.vocab_size, num_classes=self.vocab_size).float()
        logits = one_hot.clone()
        if self.neuroplasticity_enabled:
            logits[..., 0] += 0.5
            logits[..., 1] += 0.25
            hidden = one_hot + 0.1
        else:
            logits[..., 1] += 1.0
            hidden = one_hot
        hidden_states = (hidden * 0.5, hidden) if output_hidden_states else None
        if return_dict:
            return types.SimpleNamespace(logits=logits, hidden_states=hidden_states)
        return logits, hidden_states


def test_dense_top1_ranks_computes_teacher_rank_inside_sparse_logits():
    dense_logits = torch.tensor([[[0.1, 0.9, 0.0], [0.8, 0.1, 0.2]]], dtype=torch.float32)
    sparse_logits = torch.tensor([[[0.2, 0.7, 0.9], [0.9, 0.2, 0.1]]], dtype=torch.float32)
    ranks = dense_top1_ranks(sparse_logits, dense_logits)
    assert ranks.tolist() == [[2, 1]]


def test_final_hidden_cosine_uses_last_valid_token():
    dense_hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    sparse_hidden = torch.tensor(
        [[[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    cosine = final_hidden_cosine(dense_hidden, sparse_hidden, attention_mask=mask)
    assert torch.allclose(cosine, torch.tensor([1.0]))


def test_evaluate_decode_prefixes_reports_rollout_kl_hidden_cosine_and_balance():
    model = _ToyDecodeModel()
    tokenizer = _ToyTokenizer()
    prefixes = [
        {
            "input_ids": torch.tensor([[0, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }
    ]
    metrics = evaluate_decode_prefixes(
        model,
        tokenizer,
        prefixes,
        rollout_horizon=2,
        use_cache=True,
    )
    assert metrics["dense_top1_rank_median"] >= 1.0
    assert metrics["rollout_kl_mean"] > 0.0
    assert 0.0 <= metrics["final_hidden_cosine_mean"] <= 1.0
    assert metrics["balance"]["max_block_importance_fraction"] > 0.0


def test_balance_summary_tracks_largest_fraction():
    acc = init_balance_accumulator()
    model = types.SimpleNamespace(
        sca_sparse_mlps=[
            _DummyWrapper(
                {
                    "_latent_importance_probs": torch.tensor([0.9, 0.1], dtype=torch.float32),
                    "_latent_load_probs": torch.tensor([0.8, 0.2], dtype=torch.float32),
                    "_block_importance_probs": torch.tensor([0.6, 0.4], dtype=torch.float32),
                    "_block_load_probs": torch.tensor([0.7, 0.3], dtype=torch.float32),
                }
            )
        ]
    )
    accumulate_balance_from_model(model, acc)
    summary = summarize_balance(acc)
    assert summary["max_latent_importance_fraction"] == pytest.approx(0.9)
    assert summary["max_block_load_fraction"] == pytest.approx(0.7)
