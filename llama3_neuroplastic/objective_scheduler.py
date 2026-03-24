from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch


@dataclass
class AdaptiveObjectiveConfig:
    warmup_steps: int = 0
    update_interval: int = 1
    static_mode: bool = False
    alpha_init: float = 0.80
    beta_init: float = 0.10
    gamma_init: float = 0.05
    delta_init: float = 0.05
    alpha_min: float = 0.45
    alpha_max: float = 0.92
    beta_min: float = 0.03
    beta_max: float = 0.40
    gamma_min: float = 0.01
    gamma_max: float = 0.20
    delta_min: float = 0.01
    delta_max: float = 0.20
    ema_decay: float = 0.90
    target_kl: float = 0.05
    target_ppl_drift: float = 0.02
    target_token_entropy: float = 3.0
    target_bio_loss: float = 0.10
    pid_kp: float = 0.35
    pid_ki: float = 0.03
    pid_kd: float = 0.05
    max_pid_output: float = 0.25
    integral_clamp: float = 10.0
    stability_error_clamp: float = 5.0

    def __post_init__(self) -> None:
        if self.update_interval <= 0:
            raise ValueError("update_interval must be > 0")
        if self.ema_decay < 0.0 or self.ema_decay >= 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        mins = [self.alpha_min, self.beta_min, self.gamma_min, self.delta_min]
        maxs = [self.alpha_max, self.beta_max, self.gamma_max, self.delta_max]
        if sum(mins) > 1.0 + 1e-6:
            raise ValueError("coefficient minima must fit inside the simplex")
        if sum(maxs) < 1.0 - 1e-6:
            raise ValueError("coefficient maxima must cover the simplex")


def _project_simplex_with_bounds(values: torch.Tensor, mins: torch.Tensor, maxs: torch.Tensor) -> torch.Tensor:
    out = torch.clamp(values.float(), mins.float(), maxs.float())
    target_sum = 1.0
    for _ in range(32):
        total = float(out.sum().item())
        delta = target_sum - total
        if abs(delta) <= 1e-7:
            break
        if delta > 0.0:
            room = maxs - out
        else:
            room = out - mins
        active = room > 1e-8
        if not torch.any(active):
            break
        weights = room[active]
        weights = weights / weights.sum().clamp_min(1e-8)
        out[active] = out[active] + (weights * delta)
        out = torch.clamp(out, mins, maxs)
    residual = target_sum - float(out.sum().item())
    if abs(residual) > 1e-6:
        for idx in range(out.numel()):
            if residual > 0.0:
                room = float((maxs[idx] - out[idx]).item())
                if room <= 0.0:
                    continue
                step = min(residual, room)
            else:
                room = float((out[idx] - mins[idx]).item())
                if room <= 0.0:
                    continue
                step = max(residual, -room)
            out[idx] = out[idx] + step
            residual -= step
            if abs(residual) <= 1e-6:
                break
    return out


class AdaptiveObjectiveScheduler:
    def __init__(self, config: AdaptiveObjectiveConfig) -> None:
        self.config = config
        self.step_idx = 0
        self.metric_ema: Dict[str, float] = {}
        self._stability_integral = 0.0
        self._prev_stability_error = 0.0
        self._last_diag: Dict[str, Any] = {"fallback": False}
        self._coeffs = self._project(
            torch.tensor(
                [config.alpha_init, config.beta_init, config.gamma_init, config.delta_init],
                dtype=torch.float32,
            )
        )

    def _project(self, values: torch.Tensor) -> torch.Tensor:
        mins = torch.tensor(
            [self.config.alpha_min, self.config.beta_min, self.config.gamma_min, self.config.delta_min],
            dtype=torch.float32,
        )
        maxs = torch.tensor(
            [self.config.alpha_max, self.config.beta_max, self.config.gamma_max, self.config.delta_max],
            dtype=torch.float32,
        )
        return _project_simplex_with_bounds(values, mins, maxs)

    def _coeff_dict(self) -> Dict[str, float]:
        return {
            "alpha": float(self._coeffs[0].item()),
            "beta": float(self._coeffs[1].item()),
            "gamma": float(self._coeffs[2].item()),
            "delta": float(self._coeffs[3].item()),
        }

    def _sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float] | None:
        sanitized: Dict[str, float] = {}
        for key, value in dict(metrics).items():
            scalar = float(value)
            if not torch.isfinite(torch.tensor(scalar)):
                return None
            sanitized[str(key)] = scalar
        return sanitized

    def _update_ema(self, metrics: Dict[str, float]) -> None:
        decay = float(self.config.ema_decay)
        for key, value in metrics.items():
            prev = self.metric_ema.get(key)
            self.metric_ema[key] = value if prev is None else (decay * prev) + ((1.0 - decay) * value)

    def step(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        self.step_idx += 1
        sanitized = self._sanitize_metrics(metrics)
        if sanitized is None:
            self._last_diag = {"fallback": True, "reason": "non_finite_metrics", "step": int(self.step_idx)}
            return self._coeff_dict()

        self._update_ema(sanitized)
        self._last_diag = {"fallback": False, "step": int(self.step_idx), "metric_ema": dict(self.metric_ema)}

        if self.config.static_mode or self.step_idx <= int(self.config.warmup_steps):
            return self._coeff_dict()
        if (self.step_idx - int(self.config.warmup_steps)) % int(self.config.update_interval) != 0:
            return self._coeff_dict()

        kl_now = float(self.metric_ema.get("kl_loss", sanitized.get("kl_loss", 0.0)))
        ppl_drift = float(self.metric_ema.get("ppl_drift", sanitized.get("ppl_drift", 0.0)))
        token_entropy = float(self.metric_ema.get("token_entropy", sanitized.get("token_entropy", 0.0)))
        bio_loss = float(self.metric_ema.get("bio_loss", sanitized.get("bio_loss", 0.0)))

        stability_error = max(kl_now - float(self.config.target_kl), 0.0)
        stability_error = stability_error + max(ppl_drift - float(self.config.target_ppl_drift), 0.0)
        stability_error = min(stability_error, float(self.config.stability_error_clamp))
        self._stability_integral = max(
            min(self._stability_integral + stability_error, float(self.config.integral_clamp)),
            -float(self.config.integral_clamp),
        )
        stability_derivative = stability_error - self._prev_stability_error
        self._prev_stability_error = stability_error

        beta_delta = (
            (float(self.config.pid_kp) * stability_error)
            + (float(self.config.pid_ki) * self._stability_integral)
            + (float(self.config.pid_kd) * stability_derivative)
        )
        beta_delta = max(-float(self.config.max_pid_output), min(float(self.config.max_pid_output), beta_delta))

        entropy_error = max(float(self.config.target_token_entropy) - token_entropy, 0.0)
        bio_error = max(bio_loss - float(self.config.target_bio_loss), 0.0)
        target = torch.tensor(
            [
                float(self.config.alpha_init) - (0.60 * beta_delta) - (0.05 * entropy_error) - (0.05 * bio_error),
                float(self.config.beta_init) + beta_delta,
                float(self.config.gamma_init) + (0.05 * entropy_error),
                float(self.config.delta_init) + (0.05 * bio_error),
            ],
            dtype=torch.float32,
        )
        self._coeffs = self._project(target)
        self._last_diag.update(
            {
                "stability_error": float(stability_error),
                "stability_integral": float(self._stability_integral),
                "entropy_error": float(entropy_error),
                "bio_error": float(bio_error),
            }
        )
        return self._coeff_dict()

    def diagnostics(self) -> Dict[str, Any]:
        return {**self._last_diag, "coefficients": self._coeff_dict()}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "step_idx": int(self.step_idx),
            "coeffs": self._coeffs.detach().cpu(),
            "metric_ema": dict(self.metric_ema),
            "stability_integral": float(self._stability_integral),
            "prev_stability_error": float(self._prev_stability_error),
            "last_diag": dict(self._last_diag),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        coeffs = state.get("coeffs")
        if torch.is_tensor(coeffs):
            self._coeffs = self._project(coeffs.float())
        self.step_idx = int(state.get("step_idx", self.step_idx))
        self.metric_ema = {str(k): float(v) for k, v in dict(state.get("metric_ema", {})).items()}
        self._stability_integral = float(state.get("stability_integral", self._stability_integral))
        self._prev_stability_error = float(state.get("prev_stability_error", self._prev_stability_error))
        self._last_diag = dict(state.get("last_diag", self._last_diag))
