import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from objective_scheduler import AdaptiveObjectiveConfig, AdaptiveObjectiveScheduler
from train_llama_sca_objective import (
    TrainObjectiveConfig,
    build_optimizer,
    save_objective_checkpoint,
    train_objective_loop,
)


class ToyLMDataset(Dataset):
    def __init__(self, num_samples: int = 24, seq_len: int = 12, vocab_size: int = 32):
        torch.manual_seed(0)
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class ToyDualStreamModel(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.adapter = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self._device = torch.device("cpu")
        self._step = 0

    @property
    def device(self):
        return self._device

    def forward_dual_stream(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.embedding(input_ids)
        base_hidden = hidden.detach()
        adapted_hidden = hidden + (self.scale * torch.tanh(self.adapter(hidden)))

        base_logits = self.head(base_hidden)
        adapted_logits = self.head(adapted_hidden)

        instability = float((self._step % 5) / 10.0)
        self._step += 1
        bio_stats = {
            "active_block_ratio": 3.0 / 128.0,
            "refractory_violation_rate": 0.0,
            "pattern_instability": instability,
            "biological_active_ratio": 0.5,
        }
        return {
            "adapted_logits": adapted_logits,
            "base_logits": base_logits.detach(),
            "bio_stats": bio_stats,
        }


def test_train_objective_smoke_and_checkpoint(tmp_path):
    model = ToyDualStreamModel()
    dataset = ToyLMDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    cfg = TrainObjectiveConfig(
        output_dir=str(tmp_path),
        max_steps=10,
        grad_accum=1,
        batch_size=2,
        log_every=5,
        save_every=1000,
        max_grad_norm=1.0,
        static_mode=False,
        objective=AdaptiveObjectiveConfig(
            warmup_steps=0,
            update_interval=1,
            static_mode=False,
        ),
    )
    objective_scheduler = AdaptiveObjectiveScheduler(cfg.objective)
    optimizer, lr_scheduler = build_optimizer(model, cfg)

    result = train_objective_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        objective_scheduler=objective_scheduler,
        train_config=cfg,
        output_dir=None,
    )

    assert result["steps_completed"] == 10
    assert len(result["history"]) == 10
    assert all(torch.isfinite(torch.tensor(row["loss_total"])) for row in result["history"])

    ckpt = save_objective_checkpoint(
        output_dir=str(tmp_path),
        step=result["steps_completed"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        objective_scheduler=objective_scheduler,
        train_config=cfg,
        history_tail=result["history"][-3:],
    )
    assert os.path.exists(ckpt)

    payload = torch.load(ckpt, map_location="cpu")
    assert "adaptive_scheduler_state_dict" in payload
    assert "metric_ema" in payload
    assert "train_config" in payload
