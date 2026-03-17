#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from .neuroplastic_llama import NeuroplasticLlama
    from .objective_losses import (
        compute_biological_loss,
        compute_entropy_loss,
        compute_kl_loss,
        compute_lm_loss,
        compute_ppl_drift,
        estimate_token_entropy,
    )
    from .objective_scheduler import AdaptiveObjectiveConfig, AdaptiveObjectiveScheduler
except ImportError:  # Script-mode fallback
    from neuroplastic_llama import NeuroplasticLlama
    from objective_losses import (
        compute_biological_loss,
        compute_entropy_loss,
        compute_kl_loss,
        compute_lm_loss,
        compute_ppl_drift,
        estimate_token_entropy,
    )
    from objective_scheduler import AdaptiveObjectiveConfig, AdaptiveObjectiveScheduler


LOGGER = logging.getLogger("train_llama_sca_objective")


@dataclass
class TrainObjectiveConfig:
    model_name: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    output_dir: str = "experiments/llama_sca_objective_v1"
    seed: int = 42
    task_id: int = 0

    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    text_column: str = "text"
    max_samples: int = 0

    max_seq_length: int = 128
    batch_size: int = 1
    grad_accum: int = 4
    max_steps: int = 1000
    log_every: int = 20
    save_every: int = 200

    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_lr_steps: int = 100
    max_grad_norm: float = 1.0

    kl_temperature: float = 1.0
    static_mode: bool = False

    sca_block_size: int = 32
    sca_block_rank: int = 4
    sca_top_k: int = 3
    sca_sigma: float = 1.0
    sca_refractory_steps: int = 100
    sca_inhibition_lambda: float = 0.0
    sca_use_cuda: bool = False
    sca_spmm_impl: str = "dense"
    sca_soft_mask: bool = True
    sca_grouped_row_gemm: bool = False
    sca_grouped_row_min_bucket: int = 2
    sca_grouped_row_allow_4bit_dequant: bool = False

    biological_targets: Dict[str, Any] = field(
        default_factory=lambda: {
            "active_block_ratio": 3.0 / 128.0,
            "refractory_violation_rate": 0.0,
            "pattern_instability": 0.0,
            "biological_active_ratio": 0.50,
            "weights": {
                "active_block_ratio": 1.0,
                "refractory_violation_rate": 1.0,
                "pattern_instability": 2.0,
                "biological_active_ratio": 0.0,
            },
        }
    )
    objective: AdaptiveObjectiveConfig = field(default_factory=AdaptiveObjectiveConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainObjectiveConfig":
        payload = dict(payload)
        objective_payload = payload.pop("objective", {})
        cfg = cls(**payload)
        if objective_payload:
            cfg.objective = AdaptiveObjectiveConfig.from_dict(objective_payload)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["objective"] = self.objective.to_dict()
        return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_config(path: str, force_static: bool = False) -> TrainObjectiveConfig:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            raw = json.load(f)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML is required for YAML config files")
            raw = yaml.safe_load(f)

    cfg = TrainObjectiveConfig.from_dict(raw)
    if force_static:
        cfg.static_mode = True
        cfg.objective.static_mode = True
    return cfg


def _linear_warmup_lambda(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def build_optimizer(
    model: torch.nn.Module,
    config: TrainObjectiveConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found")

    optimizer = AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _linear_warmup_lambda(step, config.warmup_lr_steps),
    )
    return optimizer, scheduler


def build_lm_dataloader(
    tokenizer: Any,
    config: TrainObjectiveConfig,
) -> DataLoader:
    if load_dataset is None:
        raise RuntimeError("datasets package is required to build dataloaders")

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
    )
    if config.max_samples > 0:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    text_col = config.text_column
    dataset = dataset.filter(lambda x: isinstance(x[text_col], str) and len(x[text_col].strip()) > 0)

    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=config.max_seq_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(
        tokenized,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _safe_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().item())
    return float(value)


def save_objective_checkpoint(
    output_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    objective_scheduler: AdaptiveObjectiveScheduler,
    train_config: TrainObjectiveConfig,
    history_tail: Optional[List[Dict[str, Any]]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"objective_step_{step:07d}.pt")

    payload = {
        "step": int(step),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "adaptive_scheduler_state_dict": objective_scheduler.state_dict(),
        "metric_ema": dict(objective_scheduler.ema),
        "train_config": train_config.to_dict(),
        "history_tail": history_tail or [],
    }

    if hasattr(model, "save_pretrained"):
        model_dir = os.path.join(output_dir, f"model_step_{step:07d}")
        model.save_pretrained(model_dir)
        payload["model_dir"] = model_dir
    else:
        payload["model_state_dict"] = model.state_dict()

    torch.save(payload, ckpt_path)
    return ckpt_path


def train_objective_loop(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    objective_scheduler: AdaptiveObjectiveScheduler,
    train_config: TrainObjectiveConfig,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    model.train()
    history: List[Dict[str, Any]] = []
    global_step = 0
    micro_step = 0
    data_iter = iter(dataloader)
    trainable_params = [
        param
        for group in optimizer.param_groups
        for param in group["params"]
        if param.requires_grad
    ]
    current_coeffs = dict(objective_scheduler.current)
    last_diag: Dict[str, Any] = objective_scheduler.diagnostics()
    metric_keys = ("lm_loss", "kl_loss", "entropy_loss", "bio_loss", "ppl_drift", "token_entropy", "bio_signal")
    metric_sums = {k: 0.0 for k in metric_keys}
    metric_count = 0
    tokens_in_pending_step = 0
    steps_since_log = 0
    tokens_since_log = 0
    step_timer_start = time.perf_counter()
    log_timer_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    while global_step < train_config.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        labels = input_ids.clone()
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)
        tokens_in_pending_step += int((labels != -100).sum().item())

        dual = model.forward_dual_stream(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            use_cache=False,
            output_hidden_states=False,
            task_id=train_config.task_id,
        )
        adapted_logits = dual["adapted_logits"]
        base_logits = dual["base_logits"]
        if base_logits.device != adapted_logits.device or base_logits.dtype != adapted_logits.dtype:
            base_logits = base_logits.to(device=adapted_logits.device, dtype=adapted_logits.dtype)
        bio_stats = dual.get("bio_stats", {})

        lm_loss = compute_lm_loss(adapted_logits, labels)
        with torch.no_grad():
            base_lm_loss = compute_lm_loss(base_logits, labels)
        kl_loss = compute_kl_loss(
            adapted_logits,
            base_logits,
            temperature=train_config.kl_temperature,
            labels=labels,
        )
        entropy_loss = compute_entropy_loss(adapted_logits)
        token_entropy = estimate_token_entropy(adapted_logits)
        bio_loss = compute_biological_loss(
            stats=bio_stats,
            targets=train_config.biological_targets,
            reference=adapted_logits,
        )
        ppl_drift = compute_ppl_drift(lm_loss.detach(), base_lm_loss.detach())

        scheduler_metrics = {
            "lm_loss": _safe_float(lm_loss.detach()),
            "kl_loss": _safe_float(kl_loss.detach()),
            "entropy_loss": _safe_float(entropy_loss.detach()),
            "bio_loss": _safe_float(bio_loss.detach()),
            "ppl_drift": _safe_float(ppl_drift.detach()),
            "token_entropy": _safe_float(token_entropy.detach()),
            "bio_signal": _safe_float(bio_stats.get("pattern_instability", 0.0)),
        }
        for k in metric_keys:
            metric_sums[k] += float(scheduler_metrics[k])
        metric_count += 1

        total_loss = (
            (current_coeffs["alpha"] * lm_loss)
            + (current_coeffs["beta"] * kl_loss)
            + (current_coeffs["gamma"] * entropy_loss)
            + (current_coeffs["delta"] * bio_loss)
        )
        scaled_loss = total_loss / float(train_config.grad_accum)

        if not torch.isfinite(scaled_loss):
            optimizer.zero_grad(set_to_none=True)
            current_coeffs = objective_scheduler.step({"lm_loss": float("nan")})
            last_diag = objective_scheduler.diagnostics()
            metric_sums = {k: 0.0 for k in metric_keys}
            metric_count = 0
            micro_step = 0
            tokens_in_pending_step = 0
            step_timer_start = time.perf_counter()
            continue

        scaled_loss.backward()
        micro_step += 1

        if micro_step % train_config.grad_accum != 0:
            continue

        if train_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, train_config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if lr_scheduler is not None:
            lr_scheduler.step()

        step_elapsed_sec = max(time.perf_counter() - step_timer_start, 1e-9)
        step_tokens = int(tokens_in_pending_step)
        step_tokens_per_sec = float(step_tokens) / step_elapsed_sec
        tokens_in_pending_step = 0
        step_timer_start = time.perf_counter()
        steps_since_log += 1
        tokens_since_log += step_tokens

        avg_metrics = {
            k: (metric_sums[k] / max(metric_count, 1))
            for k in metric_keys
        }
        current_coeffs = objective_scheduler.step(avg_metrics)
        last_diag = objective_scheduler.diagnostics()
        metric_sums = {k: 0.0 for k in metric_keys}
        metric_count = 0

        global_step += 1
        row = {
            "step": int(global_step),
            "loss_total": _safe_float(total_loss.detach()),
            "loss_lm": _safe_float(lm_loss.detach()),
            "loss_kl": _safe_float(kl_loss.detach()),
            "loss_entropy": _safe_float(entropy_loss.detach()),
            "loss_bio": _safe_float(bio_loss.detach()),
            "ppl_drift": _safe_float(ppl_drift.detach()),
            "token_entropy": _safe_float(token_entropy.detach()),
            "alpha": float(current_coeffs["alpha"]),
            "beta": float(current_coeffs["beta"]),
            "gamma": float(current_coeffs["gamma"]),
            "delta": float(current_coeffs["delta"]),
            "bio_active_ratio": float(bio_stats.get("active_block_ratio", 0.0)),
            "bio_instability": float(bio_stats.get("pattern_instability", 0.0)),
            "bio_active_score_mean": float(bio_stats.get("active_score_mean", 0.0)),
            "scheduler_updated": bool(last_diag.get("updated", False)),
            "scheduler_stability_error": float(last_diag.get("stability_error", 0.0)),
            "scheduler_pid_control": float(last_diag.get("pid_control", 0.0)),
            "step_time_ms": float(step_elapsed_sec * 1000.0),
            "step_tokens": int(step_tokens),
            "step_tokens_per_sec": float(step_tokens_per_sec),
        }
        history.append(row)

        if global_step == 1 or (global_step % train_config.log_every) == 0:
            log_elapsed_sec = max(time.perf_counter() - log_timer_start, 1e-9)
            interval_steps_per_sec = float(steps_since_log) / log_elapsed_sec
            interval_tokens_per_sec = float(tokens_since_log) / log_elapsed_sec
            interval_step_time_ms = (log_elapsed_sec / max(steps_since_log, 1)) * 1000.0
            LOGGER.info(
                "step=%d total=%.4f lm=%.4f kl=%.4f ent=%.4f bio=%.4f "
                "coeffs=(%.3f,%.3f,%.3f,%.3f) ppl_drift=%.4f "
                "sched(updated=%s err=%.4f pid=%.4f) "
                "throughput(step_ms=%.2f step_toks=%d step_tok_s=%.1f avg_step_ms=%.2f avg_step_s=%.3f avg_tok_s=%.1f)",
                global_step,
                row["loss_total"],
                row["loss_lm"],
                row["loss_kl"],
                row["loss_entropy"],
                row["loss_bio"],
                row["alpha"],
                row["beta"],
                row["gamma"],
                row["delta"],
                row["ppl_drift"],
                str(row["scheduler_updated"]),
                row["scheduler_stability_error"],
                row["scheduler_pid_control"],
                row["step_time_ms"],
                row["step_tokens"],
                row["step_tokens_per_sec"],
                interval_step_time_ms,
                interval_steps_per_sec,
                interval_tokens_per_sec,
            )
            log_timer_start = time.perf_counter()
            steps_since_log = 0
            tokens_since_log = 0

        if output_dir and (global_step % train_config.save_every) == 0:
            save_objective_checkpoint(
                output_dir=output_dir,
                step=global_step,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                objective_scheduler=objective_scheduler,
                train_config=train_config,
                history_tail=history[-10:],
            )

    return {
        "history": history,
        "final_scheduler": objective_scheduler.state_dict(),
        "steps_completed": int(global_step),
    }


def build_llama_model(config: TrainObjectiveConfig) -> NeuroplasticLlama:
    try:
        return NeuroplasticLlama(
            model_name=config.model_name,
            neuroplasticity_enabled=True,
            sca_block_size=config.sca_block_size,
            sca_block_rank=config.sca_block_rank,
            sca_top_k=config.sca_top_k,
            sca_sigma=config.sca_sigma,
            sca_refractory_steps=config.sca_refractory_steps,
            sca_inhibition_lambda=config.sca_inhibition_lambda,
            sca_use_cuda=config.sca_use_cuda,
            sca_spmm_impl=config.sca_spmm_impl,
            sca_soft_mask=config.sca_soft_mask,
            sca_grouped_row_gemm=config.sca_grouped_row_gemm,
            sca_grouped_row_min_bucket=config.sca_grouped_row_min_bucket,
            sca_grouped_row_allow_4bit_dequant=config.sca_grouped_row_allow_4bit_dequant,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if config.sca_use_cuda and "Failed to build/load SCA CUDA extension" in msg:
            LOGGER.warning(
                "SCA CUDA extension unavailable for training path; retrying with sca_use_cuda=False."
            )
            return NeuroplasticLlama(
                model_name=config.model_name,
                neuroplasticity_enabled=True,
                sca_block_size=config.sca_block_size,
                sca_block_rank=config.sca_block_rank,
                sca_top_k=config.sca_top_k,
                sca_sigma=config.sca_sigma,
                sca_refractory_steps=config.sca_refractory_steps,
                sca_inhibition_lambda=config.sca_inhibition_lambda,
                sca_use_cuda=False,
                sca_spmm_impl=config.sca_spmm_impl,
                sca_soft_mask=config.sca_soft_mask,
                sca_grouped_row_gemm=config.sca_grouped_row_gemm,
                sca_grouped_row_min_bucket=config.sca_grouped_row_min_bucket,
                sca_grouped_row_allow_4bit_dequant=config.sca_grouped_row_allow_4bit_dequant,
            )
        raise


def _promote_trainable_objective_modules_to_fp32(model: NeuroplasticLlama) -> None:
    # Keep trainable objective path in FP32 to avoid tiny FP16 updates being quantized away.
    model.spatial_proj.to(device=model.device, dtype=torch.float32)
    model.task_embedding.to(device=model.device, dtype=torch.float32)
    if hasattr(model, "sca_adapters") and isinstance(model.sca_adapters, torch.nn.Module):
        model.sca_adapters.to(device=model.device, dtype=torch.float32)
    if hasattr(model, "adapter_scale"):
        model.adapter_scale.data = model.adapter_scale.data.float()


def run_training_from_config(config: TrainObjectiveConfig) -> Dict[str, Any]:
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = build_lm_dataloader(tokenizer, config)
    model = build_llama_model(config)
    _promote_trainable_objective_modules_to_fp32(model)
    bio_active_weight = float(config.biological_targets.get("weights", {}).get("biological_active_ratio", 0.0))
    if bio_active_weight <= 0.0:
        model.collect_bio_gate_telemetry = False
        LOGGER.info(
            "Disabling biological coordinator gate telemetry in objective training (biological_active_ratio weight <= 0)."
        )
    model.model.gradient_checkpointing_enable()
    model.model.enable_input_require_grads()

    objective_cfg = config.objective
    objective_cfg.static_mode = bool(config.static_mode or objective_cfg.static_mode)
    objective_scheduler = AdaptiveObjectiveScheduler(objective_cfg)

    optimizer, lr_scheduler = build_optimizer(model, config)
    result = train_objective_loop(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        objective_scheduler=objective_scheduler,
        train_config=config,
        output_dir=config.output_dir,
    )

    final_ckpt = save_objective_checkpoint(
        output_dir=config.output_dir,
        step=result["steps_completed"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        objective_scheduler=objective_scheduler,
        train_config=config,
        history_tail=result["history"][-20:],
    )
    result["final_checkpoint"] = final_ckpt
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Llama SCA-v2 with adaptive multi-loss objective.")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    parser.add_argument("--static", action="store_true", help="Disable adaptive scheduler (ablation mode).")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional override for max training steps.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for output directory.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    cfg = load_train_config(args.config, force_static=args.static)

    if args.max_steps is not None:
        cfg.max_steps = int(args.max_steps)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    result = run_training_from_config(cfg)
    LOGGER.info(
        "Training complete: steps=%d final_checkpoint=%s",
        result["steps_completed"],
        result["final_checkpoint"],
    )


if __name__ == "__main__":
    main()
