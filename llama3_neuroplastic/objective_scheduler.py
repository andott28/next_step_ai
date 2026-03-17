from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


COEFF_KEYS = ("alpha", "beta", "gamma", "delta")
LOSS_KEYS = ("lm_loss", "kl_loss", "entropy_loss", "bio_loss")


@dataclass
class AdaptiveObjectiveConfig:
    alpha_init: float = 0.80
    beta_init: float = 0.10
    gamma_init: float = 0.05
    delta_init: float = 0.05

    alpha_min: float = 0.55
    alpha_max: float = 0.95
    beta_min: float = 0.01
    beta_max: float = 0.35
    gamma_min: float = 0.01
    gamma_max: float = 0.20
    delta_min: float = 0.01
    delta_max: float = 0.20

    warmup_steps: int = 500
    ema_decay: float = 0.98
    update_interval: int = 20

    target_ppl_drift: float = 0.03
    target_kl: float = 0.08
    target_entropy: float = 0.0
    target_bio: float = 0.0

    pid_kp: float = 0.08
    pid_ki: float = 0.01
    pid_kd: float = 0.02
    stability_priority_lambda: float = 1.0
    static_mode: bool = False

    eps: float = 1e-8
    fallback_cooldown_steps: int = 100
    log_weight_min: float = -20.0
    log_weight_max: float = 20.0
    integral_error_clip: float = 1000.0
    max_pid_output: float = 5.0
    max_stability_error: float = 1000.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be > 0")
        if not (0.0 <= self.ema_decay < 1.0):
            raise ValueError("ema_decay must be in [0,1)")
        if self.pid_kp < 0 or self.pid_ki < 0 or self.pid_kd < 0:
            raise ValueError("PID gains must be >= 0")
        if self.log_weight_min >= self.log_weight_max:
            raise ValueError("log_weight_min must be < log_weight_max")
        if self.integral_error_clip <= 0:
            raise ValueError("integral_error_clip must be > 0")
        if self.max_pid_output <= 0:
            raise ValueError("max_pid_output must be > 0")
        if self.max_stability_error <= 0:
            raise ValueError("max_stability_error must be > 0")

        mins = self._mins()
        maxs = self._maxs()
        if sum(mins.values()) > 1.0 + 1e-9:
            raise ValueError("sum(min bounds) must be <= 1")
        if sum(maxs.values()) < 1.0 - 1e-9:
            raise ValueError("sum(max bounds) must be >= 1")

    def _mins(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha_min,
            "beta": self.beta_min,
            "gamma": self.gamma_min,
            "delta": self.delta_min,
        }

    def _maxs(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha_max,
            "beta": self.beta_max,
            "gamma": self.gamma_max,
            "delta": self.delta_max,
        }

    def init_weights(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha_init,
            "beta": self.beta_init,
            "gamma": self.gamma_init,
            "delta": self.delta_init,
        }

    def bounds(self) -> Dict[str, Tuple[float, float]]:
        mins = self._mins()
        maxs = self._maxs()
        return {k: (mins[k], maxs[k]) for k in COEFF_KEYS}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AdaptiveObjectiveConfig":
        return cls(**payload)


def _project_bounded_simplex(
    weights: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    target_sum: float = 1.0,
) -> Dict[str, float]:
    out = {
        k: float(min(max(weights.get(k, 0.0), bounds[k][0]), bounds[k][1]))
        for k in COEFF_KEYS
    }

    for _ in range(32):
        total = sum(out.values())
        delta = target_sum - total
        if abs(delta) < 1e-9:
            break

        if delta > 0:
            movable = [k for k in COEFF_KEYS if out[k] < bounds[k][1] - 1e-12]
            if not movable:
                break
            room = sum(bounds[k][1] - out[k] for k in movable)
            if room <= 1e-12:
                break
            for k in movable:
                portion = (bounds[k][1] - out[k]) / room
                out[k] += delta * portion
        else:
            movable = [k for k in COEFF_KEYS if out[k] > bounds[k][0] + 1e-12]
            if not movable:
                break
            room = sum(out[k] - bounds[k][0] for k in movable)
            if room <= 1e-12:
                break
            for k in movable:
                portion = (out[k] - bounds[k][0]) / room
                out[k] += delta * portion

        for k in COEFF_KEYS:
            out[k] = float(min(max(out[k], bounds[k][0]), bounds[k][1]))

    total = sum(out.values())
    if abs(total - target_sum) > 1e-9:
        anchor = "alpha"
        min_v, max_v = bounds[anchor]
        out[anchor] = float(min(max(out[anchor] + (target_sum - total), min_v), max_v))

        total = sum(out.values())
        residual = target_sum - total
        if abs(residual) > 1e-9:
            for k in ("beta", "gamma", "delta"):
                min_v, max_v = bounds[k]
                new_val = out[k] + residual
                clipped = float(min(max(new_val, min_v), max_v))
                residual -= clipped - out[k]
                out[k] = clipped
                if abs(residual) <= 1e-9:
                    break

    total = sum(out.values())
    if total <= 0:
        return {k: 1.0 / len(COEFF_KEYS) for k in COEFF_KEYS}
    return {k: float(out[k] / total) for k in COEFF_KEYS}


class AdaptiveObjectiveScheduler:
    def __init__(self, config: AdaptiveObjectiveConfig):
        self.config = config
        self.bounds = config.bounds()

        init_weights = _project_bounded_simplex(config.init_weights(), self.bounds)
        self.current = dict(init_weights)
        self.safe_static = dict(init_weights)
        self.log_weights = {k: float(math.log(max(v, config.eps))) for k, v in init_weights.items()}

        self.global_step = 0
        self.last_update_step = 0
        self.cooldown_until = 0

        self.integral_error = 0.0
        self.prev_error = 0.0

        self.ema = {k: 1.0 for k in LOSS_KEYS}
        self.last_norm = {k: 0.0 for k in LOSS_KEYS}
        self.last_diag: Dict[str, Any] = {}

    def _is_finite(self, value: float) -> bool:
        return math.isfinite(float(value))

    def _metrics_finite(self, metrics: Dict[str, float]) -> bool:
        for key in ("ppl_drift", "token_entropy", "bio_signal", *LOSS_KEYS):
            if key in metrics and not self._is_finite(metrics[key]):
                return False
        return True

    def _update_ema(self, metrics: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key in LOSS_KEYS:
            value = float(metrics.get(key, 0.0))
            scale_value = abs(value)
            prev = self.ema[key]
            new = (self.config.ema_decay * prev) + ((1.0 - self.config.ema_decay) * scale_value)
            self.ema[key] = max(new, self.config.eps)
            out[key] = value / (self.ema[key] + self.config.eps)
        self.last_norm = out
        return out

    def _pid_control(self, error_value: float) -> float:
        derivative = error_value - self.prev_error
        self.integral_error += error_value
        self.integral_error = float(
            min(max(self.integral_error, -self.config.integral_error_clip), self.config.integral_error_clip)
        )
        self.prev_error = error_value
        control = (
            (self.config.pid_kp * error_value)
            + (self.config.pid_ki * self.integral_error)
            + (self.config.pid_kd * derivative)
        )
        return float(min(max(control, -self.config.max_pid_output), self.config.max_pid_output))

    def _sanitize_log_weights(self) -> None:
        for k in COEFF_KEYS:
            value = float(self.log_weights.get(k, 0.0))
            if not self._is_finite(value):
                value = 0.0
            self.log_weights[k] = float(
                min(max(value, self.config.log_weight_min), self.config.log_weight_max)
            )

    def _apply_failsafe(self, reason: str) -> Dict[str, float]:
        self.current = dict(self.safe_static)
        self.log_weights = {
            k: float(math.log(max(v, self.config.eps))) for k, v in self.current.items()
        }
        self.cooldown_until = self.global_step + self.config.fallback_cooldown_steps
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.last_diag = {
            "step": int(self.global_step),
            "fallback": True,
            "fallback_reason": reason,
            **self.current,
        }
        return dict(self.current)

    def _renormalize(self) -> None:
        self._sanitize_log_weights()
        projected = _project_bounded_simplex(
            {
                k: float(math.exp(min(max(v, self.config.log_weight_min), self.config.log_weight_max)))
                for k, v in self.log_weights.items()
            },
            self.bounds,
        )
        self.current = projected
        self.log_weights = {
            k: float(math.log(max(self.current[k], self.config.eps)))
            for k in COEFF_KEYS
        }
        self._sanitize_log_weights()

    def step(self, metrics: Dict[str, float]) -> Dict[str, float]:
        self.global_step += 1

        metrics = {k: float(v) for k, v in metrics.items()}
        if not self._metrics_finite(metrics):
            return self._apply_failsafe("non_finite_metrics")

        normalized = self._update_ema(metrics)
        in_warmup = self.global_step <= self.config.warmup_steps
        in_cooldown = self.global_step < self.cooldown_until

        if self.config.static_mode or in_warmup or in_cooldown:
            self.last_diag = {
                "step": int(self.global_step),
                "updated": False,
                "in_warmup": bool(in_warmup),
                "in_cooldown": bool(in_cooldown),
                "normalized_losses": dict(normalized),
                **self.current,
            }
            return dict(self.current)

        should_update = (self.global_step - self.last_update_step) >= self.config.update_interval
        if not should_update:
            self.last_diag = {
                "step": int(self.global_step),
                "updated": False,
                "in_warmup": False,
                "in_cooldown": False,
                "normalized_losses": dict(normalized),
                **self.current,
            }
            return dict(self.current)

        self.last_update_step = self.global_step

        ppl_drift = max(0.0, float(metrics.get("ppl_drift", 0.0) - self.config.target_ppl_drift))
        kl_excess = max(0.0, float(metrics.get("kl_loss", 0.0) - self.config.target_kl))
        e_stab = (self.config.stability_priority_lambda * ppl_drift) + kl_excess
        e_stab = float(min(max(e_stab, 0.0), self.config.max_stability_error))

        control = self._pid_control(e_stab)
        self.log_weights["beta"] += control

        if e_stab > 0:
            damp = abs(control)
            self.log_weights["gamma"] -= 0.50 * damp
            self.log_weights["delta"] -= 0.50 * damp
        else:
            self.log_weights["gamma"] += 0.10 * (
                math.log(max(self.config.gamma_init, self.config.eps)) - self.log_weights["gamma"]
            )
            self.log_weights["delta"] += 0.10 * (
                math.log(max(self.config.delta_init, self.config.eps)) - self.log_weights["delta"]
            )

        token_entropy = float(metrics.get("token_entropy", 0.0))
        if self.config.target_entropy > 0:
            entropy_gap = self.config.target_entropy - token_entropy
            self.log_weights["gamma"] += 0.05 * entropy_gap

        bio_signal = float(metrics.get("bio_signal", metrics.get("bio_loss", 0.0)))
        if self.config.target_bio > 0:
            bio_gap = bio_signal - self.config.target_bio
            self.log_weights["delta"] += 0.05 * bio_gap

        self.log_weights["alpha"] += 0.10 * (
            math.log(max(self.config.alpha_init, self.config.eps)) - self.log_weights["alpha"]
        )

        self._renormalize()

        if not all(self._is_finite(v) for v in self.current.values()):
            return self._apply_failsafe("non_finite_coefficients")

        self.last_diag = {
            "step": int(self.global_step),
            "updated": True,
            "in_warmup": False,
            "in_cooldown": False,
            "normalized_losses": dict(normalized),
            "stability_error": float(e_stab),
            "pid_control": float(control),
            "ppl_drift_excess": float(ppl_drift),
            "kl_excess": float(kl_excess),
            **self.current,
        }
        return dict(self.current)

    def diagnostics(self) -> Dict[str, Any]:
        return dict(self.last_diag)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "current": dict(self.current),
            "safe_static": dict(self.safe_static),
            "log_weights": dict(self.log_weights),
            "global_step": int(self.global_step),
            "last_update_step": int(self.last_update_step),
            "cooldown_until": int(self.cooldown_until),
            "integral_error": float(self.integral_error),
            "prev_error": float(self.prev_error),
            "ema": dict(self.ema),
            "last_norm": dict(self.last_norm),
            "last_diag": dict(self.last_diag),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "current" in state:
            self.current = _project_bounded_simplex(dict(state["current"]), self.bounds)
        if "safe_static" in state:
            self.safe_static = _project_bounded_simplex(dict(state["safe_static"]), self.bounds)
        if "log_weights" in state:
            loaded_log = {k: float(state["log_weights"].get(k, self.log_weights[k])) for k in COEFF_KEYS}
            self.log_weights = dict(loaded_log)
            self._renormalize()
        else:
            self.log_weights = {
                k: float(math.log(max(self.current[k], self.config.eps)))
                for k in COEFF_KEYS
            }
        self.global_step = int(state.get("global_step", 0))
        self.last_update_step = int(state.get("last_update_step", 0))
        self.cooldown_until = int(state.get("cooldown_until", 0))
        self.integral_error = float(state.get("integral_error", 0.0))
        self.prev_error = float(state.get("prev_error", 0.0))

        loaded_ema = state.get("ema", {})
        self.ema = {
            k: float(max(loaded_ema.get(k, self.ema[k]), self.config.eps))
            for k in LOSS_KEYS
        }
        self.last_norm = {k: float(state.get("last_norm", {}).get(k, 0.0)) for k in LOSS_KEYS}
        self.last_diag = dict(state.get("last_diag", {}))
