import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from objective_scheduler import AdaptiveObjectiveConfig, AdaptiveObjectiveScheduler


def _config(**kwargs):
    base = dict(
        warmup_steps=0,
        update_interval=1,
        static_mode=False,
        alpha_init=0.80,
        beta_init=0.10,
        gamma_init=0.05,
        delta_init=0.05,
    )
    base.update(kwargs)
    return AdaptiveObjectiveConfig(**base)


def test_scheduler_respects_bounds_and_simplex():
    scheduler = AdaptiveObjectiveScheduler(_config())

    for _ in range(20):
        coeffs = scheduler.step(
            {
                "lm_loss": 2.0,
                "kl_loss": 0.05,
                "entropy_loss": -3.0,
                "bio_loss": 0.2,
                "ppl_drift": 0.01,
                "token_entropy": 3.0,
            }
        )

    cfg = scheduler.config
    assert cfg.alpha_min <= coeffs["alpha"] <= cfg.alpha_max
    assert cfg.beta_min <= coeffs["beta"] <= cfg.beta_max
    assert cfg.gamma_min <= coeffs["gamma"] <= cfg.gamma_max
    assert cfg.delta_min <= coeffs["delta"] <= cfg.delta_max
    assert abs(sum(coeffs.values()) - 1.0) < 1e-6


def test_beta_rises_when_stability_error_is_high():
    scheduler = AdaptiveObjectiveScheduler(_config())
    first = scheduler.step(
        {
            "lm_loss": 2.0,
            "kl_loss": 0.03,
            "entropy_loss": -3.0,
            "bio_loss": 0.1,
            "ppl_drift": 0.01,
            "token_entropy": 3.0,
        }
    )
    beta_start = first["beta"]

    for _ in range(15):
        coeffs = scheduler.step(
            {
                "lm_loss": 2.2,
                "kl_loss": 0.30,
                "entropy_loss": -2.8,
                "bio_loss": 0.2,
                "ppl_drift": 0.25,
                "token_entropy": 2.6,
            }
        )

    assert coeffs["beta"] > beta_start


def test_scheduler_fallback_on_nan_metrics():
    scheduler = AdaptiveObjectiveScheduler(_config())
    scheduler.step(
        {
            "lm_loss": 2.0,
            "kl_loss": 0.05,
            "entropy_loss": -3.0,
            "bio_loss": 0.1,
            "ppl_drift": 0.01,
        }
    )

    coeffs = scheduler.step({"lm_loss": float("nan")})
    diag = scheduler.diagnostics()

    assert torch.isfinite(torch.tensor(list(coeffs.values()))).all()
    assert abs(sum(coeffs.values()) - 1.0) < 1e-6
    assert diag.get("fallback", False) is True


def test_scheduler_handles_extreme_stability_metrics_without_overflow():
    scheduler = AdaptiveObjectiveScheduler(_config(warmup_steps=0, update_interval=1))

    for _ in range(50):
        coeffs = scheduler.step(
            {
                "lm_loss": 10.0,
                "kl_loss": 1e6,
                "entropy_loss": -10.0,
                "bio_loss": 1.0,
                "ppl_drift": 1e6,
                "token_entropy": 1.0,
            }
        )

    vals = torch.tensor(list(coeffs.values()), dtype=torch.float32)
    assert torch.isfinite(vals).all()
    assert abs(float(vals.sum().item()) - 1.0) < 1e-6
