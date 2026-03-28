from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR

try:
    from .objective_losses import (
        compute_biological_loss,
        compute_entropy_loss,
        compute_kl_loss,
        compute_lm_loss,
        compute_ppl_drift,
        compute_token_entropy,
    )
    from .objective_scheduler import AdaptiveObjectiveConfig, AdaptiveObjectiveScheduler
    from .performance_utils import (
        autocast_context,
        backward_step,
        build_grad_scaler,
        build_optimizer as build_runtime_optimizer,
        configure_runtime_environment,
        maybe_compile_module,
        maybe_enable_gradient_checkpointing,
        resolve_amp_dtype,
    )
except ImportError:  # pragma: no cover
    from objective_losses import (
        compute_biological_loss,
        compute_entropy_loss,
        compute_kl_loss,
        compute_lm_loss,
        compute_ppl_drift,
        compute_token_entropy,
    )
    from objective_scheduler import AdaptiveObjectiveConfig, AdaptiveObjectiveScheduler
    from performance_utils import (
        autocast_context,
        backward_step,
        build_grad_scaler,
        build_optimizer as build_runtime_optimizer,
        configure_runtime_environment,
        maybe_compile_module,
        maybe_enable_gradient_checkpointing,
        resolve_amp_dtype,
    )


@dataclass
class TrainObjectiveConfig:
    output_dir: str
    max_steps: int = 1000
    grad_accum: int = 1
    batch_size: int = 1
    log_every: int = 10
    save_every: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_warmup_steps: int = 0
    optimizer_name: str = "auto"
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    torch_compile_enabled: bool = False
    torch_compile_mode: str = "default"
    torch_compile_fullgraph: bool = False
    gradient_checkpointing: bool = False
    cudnn_benchmark: bool = True
    static_mode: bool = False
    sca_use_cuda: bool = False
    biological_targets: Dict[str, Any] = field(
        default_factory=lambda: {
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
    )
    objective: AdaptiveObjectiveConfig = field(default_factory=AdaptiveObjectiveConfig)

    def __post_init__(self) -> None:
        self.max_steps = int(self.max_steps)
        self.grad_accum = max(int(self.grad_accum), 1)
        self.batch_size = max(int(self.batch_size), 1)
        self.log_every = max(int(self.log_every), 1)
        self.save_every = max(int(self.save_every), 0)
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.max_grad_norm = float(self.max_grad_norm)
        self.lr_warmup_steps = max(int(self.lr_warmup_steps), 0)
        self.optimizer_name = str(self.optimizer_name)
        self.amp_enabled = bool(self.amp_enabled)
        self.amp_dtype = str(self.amp_dtype)
        self.torch_compile_enabled = bool(self.torch_compile_enabled)
        self.torch_compile_mode = str(self.torch_compile_mode)
        self.torch_compile_fullgraph = bool(self.torch_compile_fullgraph)
        self.gradient_checkpointing = bool(self.gradient_checkpointing)
        self.cudnn_benchmark = bool(self.cudnn_benchmark)
        self.sca_use_cuda = bool(self.sca_use_cuda)
        self.static_mode = bool(self.static_mode)
        self.objective.static_mode = bool(self.static_mode or self.objective.static_mode)


def _infer_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        device = getattr(model, "device")
        if isinstance(device, torch.device):
            return device
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def build_optimizer(model: torch.nn.Module, cfg: TrainObjectiveConfig) -> Tuple[torch.optim.Optimizer, LambdaLR]:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found for objective optimization")
    optimizer, metadata = build_runtime_optimizer(
        params,
        optimizer_name=str(cfg.optimizer_name),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    setattr(optimizer, "_performance_metadata", metadata)
    warmup = int(cfg.lr_warmup_steps)

    def _schedule(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(float(step + 1) / float(warmup), 1.0)

    lr_scheduler = LambdaLR(optimizer, lr_lambda=_schedule)
    return optimizer, lr_scheduler


def _next_token_views(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if logits.shape[1] < 2:
        raise RuntimeError("Need sequence length >= 2 for next-token objective")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:].contiguous()
    shifted_mask = None if attention_mask is None else attention_mask[:, 1:].contiguous()
    return shifted_logits, shifted_labels, shifted_mask


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {str(key): value.to(device=device) for key, value in batch.items()}


def train_objective_loop(
    *,
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    objective_scheduler: AdaptiveObjectiveScheduler,
    train_config: TrainObjectiveConfig,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    configure_runtime_environment(cudnn_benchmark=bool(train_config.cudnn_benchmark))
    device = _infer_device(model)
    maybe_enable_gradient_checkpointing(model, enabled=bool(train_config.gradient_checkpointing))
    model = maybe_compile_module(
        model,
        enabled=bool(train_config.torch_compile_enabled),
        mode=str(train_config.torch_compile_mode),
        fullgraph=bool(train_config.torch_compile_fullgraph),
    )
    model.train()
    amp_enabled, amp_dtype = resolve_amp_dtype(device=device, requested=str(train_config.amp_dtype))
    scaler = build_grad_scaler(enabled=bool(train_config.amp_enabled and amp_enabled))
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    history: List[Dict[str, Any]] = []
    data_iter = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    started_at = time.perf_counter()
    total_tokens = 0

    for step in range(int(train_config.max_steps)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = _move_batch_to_device(batch, device)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        batch_tokens = int(attention_mask.sum().item()) if attention_mask is not None else int(input_ids.numel())
        total_tokens += batch_tokens

        with autocast_context(
            device=device,
            enabled=bool(train_config.amp_enabled and amp_enabled),
            amp_dtype=amp_dtype,
        ):
            outputs = model.forward_dual_stream(input_ids=input_ids, attention_mask=attention_mask)
            adapted_logits = outputs["adapted_logits"]
            base_logits = outputs["base_logits"]
            bio_stats = dict(outputs.get("bio_stats", {}))

            adapted_shift, labels_shift, mask_shift = _next_token_views(adapted_logits, input_ids, attention_mask)
            base_shift, _, _ = _next_token_views(base_logits, input_ids, attention_mask)

            lm_loss = compute_lm_loss(adapted_shift, labels_shift)
            kl_loss = compute_kl_loss(adapted_shift, base_shift, attention_mask=mask_shift)
            entropy_loss = compute_entropy_loss(adapted_shift, attention_mask=mask_shift)
            bio_loss = compute_biological_loss(bio_stats, train_config.biological_targets).to(device=adapted_shift.device)
            ppl_drift = compute_ppl_drift(adapted_shift, base_shift, labels_shift)
            token_entropy = compute_token_entropy(adapted_shift, attention_mask=mask_shift)

        coeffs = objective_scheduler.step(
            {
                "lm_loss": float(lm_loss.detach().cpu().item()),
                "kl_loss": float(kl_loss.detach().cpu().item()),
                "entropy_loss": float(entropy_loss.detach().cpu().item()),
                "bio_loss": float(bio_loss.detach().cpu().item()),
                "ppl_drift": float(ppl_drift.detach().cpu().item()),
                "token_entropy": float(token_entropy.detach().cpu().item()),
            }
        )
        total_loss = (
            (float(coeffs["alpha"]) * lm_loss)
            + (float(coeffs["beta"]) * kl_loss)
            + (float(coeffs["gamma"]) * entropy_loss)
            + (float(coeffs["delta"]) * bio_loss)
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite total loss at step {step + 1}")

        backward_step(
            total_loss,
            optimizer=optimizer,
            scaler=scaler,
            params=trainable_params,
            max_grad_norm=float(train_config.max_grad_norm),
        )
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        elapsed = max(time.perf_counter() - started_at, 1e-6)
        tokens_per_second = float(total_tokens / elapsed)

        history.append(
            {
                "step": int(step + 1),
                "loss_total": float(total_loss.detach().cpu().item()),
                "lm_loss": float(lm_loss.detach().cpu().item()),
                "kl_loss": float(kl_loss.detach().cpu().item()),
                "entropy_loss": float(entropy_loss.detach().cpu().item()),
                "bio_loss": float(bio_loss.detach().cpu().item()),
                "ppl_drift": float(ppl_drift.detach().cpu().item()),
                "token_entropy": float(token_entropy.detach().cpu().item()),
                "tokens_per_second": tokens_per_second,
                **{k: float(v) for k, v in coeffs.items()},
            }
        )

        if output_dir and int(train_config.save_every) > 0 and (step + 1) % int(train_config.save_every) == 0:
            save_objective_checkpoint(
                output_dir=output_dir,
                step=step + 1,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                objective_scheduler=objective_scheduler,
                train_config=train_config,
                history_tail=history[-min(len(history), 16) :],
            )

    return {
        "steps_completed": int(train_config.max_steps),
        "history": history,
        "tokens_processed": int(total_tokens),
        "tokens_per_second": float(total_tokens / max(time.perf_counter() - started_at, 1e-6)),
        "optimizer": dict(getattr(optimizer, "_performance_metadata", {})),
        "amp_enabled": bool(getattr(scaler, "is_enabled", lambda: False)()),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
    }


def save_objective_checkpoint(
    *,
    output_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    objective_scheduler: AdaptiveObjectiveScheduler,
    train_config: TrainObjectiveConfig,
    history_tail: List[Dict[str, Any]],
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"objective_step_{int(step):07d}.pt"
    payload = {
        "step": int(step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "adaptive_scheduler_state_dict": objective_scheduler.state_dict(),
        "metric_ema": dict(objective_scheduler.metric_ema),
        "train_config": asdict(train_config),
        "history_tail": history_tail,
    }
    torch.save(payload, checkpoint_path)
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "step": int(step),
                "history_tail": history_tail,
                "metric_ema": dict(objective_scheduler.metric_ema),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(checkpoint_path)
