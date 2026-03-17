import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from objective_losses import (
    compute_biological_loss,
    compute_entropy_loss,
    compute_kl_loss,
    compute_lm_loss,
)


def test_kl_loss_non_negative():
    torch.manual_seed(0)
    adapted = torch.randn(2, 4, 13)
    base = adapted + 0.1 * torch.randn(2, 4, 13)
    kl = compute_kl_loss(adapted, base, temperature=1.0)
    assert torch.isfinite(kl)
    assert float(kl.item()) >= 0.0


def test_entropy_loss_sign_convention():
    logits = torch.zeros(2, 5, 7)  # uniform distribution
    entropy_loss = compute_entropy_loss(logits)
    assert torch.isfinite(entropy_loss)
    assert float(entropy_loss.item()) <= 0.0


def test_lm_loss_finite():
    torch.manual_seed(1)
    logits = torch.randn(2, 6, 11)
    labels = torch.randint(0, 11, (2, 6))
    loss = compute_lm_loss(logits, labels)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0


def test_biological_loss_monotonic_with_target_deviation():
    targets = {
        "active_block_ratio": 3.0 / 128.0,
        "refractory_violation_rate": 0.0,
        "pattern_instability": 0.10,
        "biological_active_ratio": 0.50,
        "weights": {
            "active_block_ratio": 1.0,
            "refractory_violation_rate": 1.0,
            "pattern_instability": 1.0,
            "biological_active_ratio": 0.5,
        },
    }

    near_stats = {
        "active_block_ratio": 3.0 / 128.0,
        "refractory_violation_rate": 0.0,
        "pattern_instability": 0.11,
        "biological_active_ratio": 0.48,
    }
    far_stats = {
        "active_block_ratio": 0.20,
        "refractory_violation_rate": 0.30,
        "pattern_instability": 0.90,
        "biological_active_ratio": 0.10,
    }

    near_loss = compute_biological_loss(near_stats, targets)
    far_loss = compute_biological_loss(far_stats, targets)

    assert torch.isfinite(near_loss)
    assert torch.isfinite(far_loss)
    assert float(far_loss.item()) > float(near_loss.item())
