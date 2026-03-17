#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
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
    from .objective_losses import compute_lm_loss
    from .train_llama_sca_objective import load_train_config
except ImportError:  # Script-mode fallback
    from neuroplastic_llama import NeuroplasticLlama
    from objective_losses import compute_lm_loss
    from train_llama_sca_objective import load_train_config


LOGGER = logging.getLogger("run_sca_posthoc_ablation")


@dataclass(frozen=True)
class EvalSetting:
    name: str
    neuroplasticity_enabled: bool
    spmm_impl: str
    top_k: int
    soft_mask: bool
    inhibition_lambda: float
    refractory_steps: int
    sigma: float


@dataclass
class AblationConfig:
    model_name: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    checkpoint_dir: Optional[str] = None
    task_id: int = 0
    seed: int = 42

    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "validation"
    text_column: str = "text"
    max_samples: int = 1024
    max_seq_length: int = 128
    batch_size: int = 1
    num_workers: int = 0
    max_eval_batches: int = 64
    warmup_batches: int = 4

    spmm_impls: List[str] = None
    top_k_values: List[int] = None
    soft_mask_values: List[bool] = None
    sigma_values: List[float] = None
    ablate_inhibition: bool = True
    ablate_refractory: bool = True
    include_frozen_baseline: bool = True

    output_json: Optional[str] = None

    def __post_init__(self) -> None:
        if self.spmm_impls is None:
            self.spmm_impls = ["dense", "cuda_spmm"]
        if self.top_k_values is None:
            self.top_k_values = []
        if self.soft_mask_values is None:
            self.soft_mask_values = []
        if self.sigma_values is None:
            self.sigma_values = []

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AblationConfig":
        return cls(**dict(payload))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_yaml_or_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return dict(json.load(f))
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML config files")
        return dict(yaml.safe_load(f))


def _parse_int_list(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def _parse_float_list(raw: str) -> List[float]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _parse_str_list(raw: str) -> List[str]:
    values = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(item)
    return values


def _parse_bool_list(raw: str) -> List[bool]:
    lut = {
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
    }
    values: List[bool] = []
    for part in raw.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item not in lut:
            raise ValueError(f"Unsupported bool token: {part}")
        values.append(lut[item])
    return values


def _resolve_checkpoint_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isfile(path):
        payload = torch.load(path, map_location="cpu")
        model_dir = payload.get("model_dir")
        if not model_dir:
            raise ValueError(f"Checkpoint file does not contain model_dir: {path}")
        return str(model_dir)
    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    sca_file = os.path.join(path, "neuroplastic_llama_sca_v2.bin")
    if os.path.isfile(sca_file):
        return path

    candidates: List[tuple[int, str]] = []
    for name in os.listdir(path):
        if not name.startswith("model_step_"):
            continue
        full = os.path.join(path, name)
        if not os.path.isdir(full):
            continue
        marker = os.path.join(full, "neuroplastic_llama_sca_v2.bin")
        if not os.path.isfile(marker):
            continue
        suffix = name.replace("model_step_", "").strip()
        try:
            step = int(suffix)
        except ValueError:
            continue
        candidates.append((step, full))

    if not candidates:
        objective_candidates: List[tuple[int, str]] = []
        for name in os.listdir(path):
            if not (name.startswith("objective_step_") and name.endswith(".pt")):
                continue
            full = os.path.join(path, name)
            if not os.path.isfile(full):
                continue
            stem = name.replace("objective_step_", "").replace(".pt", "").strip()
            try:
                step = int(stem)
            except ValueError:
                continue
            objective_candidates.append((step, full))
        objective_candidates.sort(key=lambda x: x[0])
        if objective_candidates:
            payload = torch.load(objective_candidates[-1][1], map_location="cpu")
            model_dir = payload.get("model_dir")
            if model_dir and os.path.isdir(model_dir):
                return str(model_dir)

        sibling_suggestions: List[str] = []
        parent = os.path.dirname(path) or "."
        for name in sorted(os.listdir(parent)):
            full = os.path.join(parent, name)
            if not os.path.isdir(full):
                continue
            has_model_step = any(
                child_name.startswith("model_step_") and os.path.isdir(os.path.join(full, child_name))
                for child_name in os.listdir(full)
            )
            if has_model_step:
                sibling_suggestions.append(full)
            if len(sibling_suggestions) >= 6:
                break

        hint = ""
        if sibling_suggestions:
            hint = f" Nearby runs with checkpoints: {sibling_suggestions}."
        raise FileNotFoundError(f"No model_step_* or objective_step_*.pt checkpoint found under: {path}.{hint}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _build_eval_dataloader(tokenizer: Any, cfg: AblationConfig) -> DataLoader:
    if load_dataset is None:
        raise RuntimeError("datasets package is required to run post-hoc ablation")
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
    if cfg.max_samples > 0:
        dataset = dataset.select(range(min(len(dataset), cfg.max_samples)))

    text_col = cfg.text_column
    dataset = dataset.filter(lambda x: isinstance(x[text_col], str) and len(x[text_col].strip()) > 0)

    def _tok(batch: Dict[str, Sequence[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_seq_length,
        )

    tokenized = dataset.map(_tok, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(
        tokenized,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _unique_keep_order(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for value in values:
        key = value if not isinstance(value, float) else round(float(value), 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _build_settings(model: NeuroplasticLlama, cfg: AblationConfig) -> List[EvalSetting]:
    base_cfg = model.sca_config
    top_k_values = [int(v) for v in cfg.top_k_values] if cfg.top_k_values else [int(base_cfg.top_k)]
    top_k_values = [max(1, min(int(v), int(base_cfg.num_blocks))) for v in top_k_values]
    soft_mask_values = list(cfg.soft_mask_values) if cfg.soft_mask_values else [bool(base_cfg.soft_mask)]
    sigma_values = [float(v) for v in cfg.sigma_values] if cfg.sigma_values else [float(base_cfg.sigma)]
    spmm_impls = _unique_keep_order([str(v) for v in cfg.spmm_impls])

    inhibition_values = [float(base_cfg.inhibition_lambda)]
    if cfg.ablate_inhibition and float(base_cfg.inhibition_lambda) != 0.0:
        inhibition_values.append(0.0)
    inhibition_values = _unique_keep_order(inhibition_values)

    refractory_values = [int(base_cfg.refractory_steps)]
    if cfg.ablate_refractory and int(base_cfg.refractory_steps) != 0:
        refractory_values.append(0)
    refractory_values = _unique_keep_order(refractory_values)

    settings: List[EvalSetting] = []
    for spmm_impl, top_k, soft_mask, inhibition_lambda, refractory_steps, sigma in itertools.product(
        spmm_impls,
        top_k_values,
        soft_mask_values,
        inhibition_values,
        refractory_values,
        sigma_values,
    ):
        name = (
            f"np_on|spmm={spmm_impl}|k={int(top_k)}|soft={int(bool(soft_mask))}"
            f"|inh={float(inhibition_lambda):.4f}|ref={int(refractory_steps)}|sig={float(sigma):.4f}"
        )
        settings.append(
            EvalSetting(
                name=name,
                neuroplasticity_enabled=True,
                spmm_impl=str(spmm_impl),
                top_k=int(top_k),
                soft_mask=bool(soft_mask),
                inhibition_lambda=float(inhibition_lambda),
                refractory_steps=max(0, int(refractory_steps)),
                sigma=max(1e-6, float(sigma)),
            )
        )

    if cfg.include_frozen_baseline:
        settings.append(
            EvalSetting(
                name="np_off|frozen_backbone",
                neuroplasticity_enabled=False,
                spmm_impl=str(base_cfg.spmm_impl),
                top_k=int(base_cfg.top_k),
                soft_mask=bool(base_cfg.soft_mask),
                inhibition_lambda=float(base_cfg.inhibition_lambda),
                refractory_steps=int(base_cfg.refractory_steps),
                sigma=float(base_cfg.sigma),
            )
        )
    return settings


def _apply_setting(model: NeuroplasticLlama, setting: EvalSetting) -> None:
    model.neuroplasticity_enabled = bool(setting.neuroplasticity_enabled)
    model.sca_config.spmm_impl = str(setting.spmm_impl)
    model.sca_config.top_k = int(max(1, min(setting.top_k, model.sca_config.num_blocks)))
    model.sca_config.soft_mask = bool(setting.soft_mask)
    model.sca_config.inhibition_lambda = float(setting.inhibition_lambda)
    model.sca_config.refractory_steps = int(max(0, setting.refractory_steps))
    model.sca_config.sigma = float(max(1e-6, setting.sigma))
    if hasattr(model, "refractory_until") and torch.is_tensor(model.refractory_until):
        model.refractory_until.zero_()


def _autocast_context(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _evaluate_setting(
    model: NeuroplasticLlama,
    dataloader: DataLoader,
    task_id: int,
    max_eval_batches: int,
    warmup_batches: int,
) -> Dict[str, Any]:
    model.eval()
    device = model.device
    is_cuda = device.type == "cuda"

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device=device)

    loss_weighted_sum = 0.0
    token_count_sum = 0
    timed_token_count = 0
    timed_sec = 0.0
    timed_steps = 0
    steps = 0

    with torch.inference_mode():
        for batch in dataloader:
            if steps >= max_eval_batches:
                break

            input_ids = batch["input_ids"].to(device=device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=device, non_blocking=True)

            labels = input_ids.clone()
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
            token_count = int((labels[..., 1:] != -100).sum().item())
            if token_count <= 0:
                continue

            if is_cuda:
                torch.cuda.synchronize(device=device)
            t0 = time.perf_counter()
            with _autocast_context(device):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                    task_id=task_id,
                )
                loss = compute_lm_loss(outputs.logits.float(), labels)
            if is_cuda:
                torch.cuda.synchronize(device=device)
            elapsed = time.perf_counter() - t0

            loss_weighted_sum += float(loss.item()) * float(token_count)
            token_count_sum += token_count

            if steps >= warmup_batches:
                timed_token_count += token_count
                timed_sec += elapsed
                timed_steps += 1

            steps += 1

    avg_loss = float(loss_weighted_sum / max(token_count_sum, 1))
    ppl = float(math.exp(min(max(avg_loss, -20.0), 20.0)))
    tok_per_sec = float(timed_token_count / max(timed_sec, 1e-9))
    avg_step_ms = float((timed_sec / max(timed_steps, 1)) * 1000.0)

    peak_vram_gb = 0.0
    if is_cuda:
        peak_vram_gb = float(torch.cuda.max_memory_allocated(device=device) / (1024**3))

    return {
        "avg_lm_loss": avg_loss,
        "perplexity": ppl,
        "tokens_per_second": tok_per_sec,
        "avg_step_ms": avg_step_ms,
        "peak_vram_gb": peak_vram_gb,
        "evaluated_batches": int(steps),
        "evaluated_tokens": int(token_count_sum),
        "timed_batches": int(timed_steps),
        "timed_tokens": int(timed_token_count),
        "timed_seconds": float(timed_sec),
    }


def _format_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        f"{'Setting':<72} {'Loss':>9} {'PPL':>10} {'Tok/s':>10} "
        f"{'Step ms':>10} {'VRAM GB':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['name']:<72} "
            f"{row['avg_lm_loss']:>9.4f} "
            f"{row['perplexity']:>10.2f} "
            f"{row['tokens_per_second']:>10.1f} "
            f"{row['avg_step_ms']:>10.2f} "
            f"{row['peak_vram_gb']:>9.2f}"
        )
    return "\n".join(lines)


def _load_effective_config(args: argparse.Namespace) -> AblationConfig:
    payload: Dict[str, Any] = {}
    if args.config is not None:
        payload = _load_yaml_or_json(args.config)
    cfg = AblationConfig.from_dict(payload)

    if args.train_config is not None:
        train_cfg = load_train_config(args.train_config, force_static=False)
        cfg.model_name = train_cfg.model_name
        cfg.task_id = int(train_cfg.task_id)
        cfg.dataset_name = train_cfg.dataset_name
        cfg.dataset_config = train_cfg.dataset_config
        cfg.dataset_split = "validation"
        cfg.text_column = train_cfg.text_column
        cfg.max_seq_length = int(train_cfg.max_seq_length)
        cfg.batch_size = int(train_cfg.batch_size)

    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.task_id is not None:
        cfg.task_id = int(args.task_id)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.dataset_name is not None:
        cfg.dataset_name = args.dataset_name
    if args.dataset_config is not None:
        cfg.dataset_config = args.dataset_config
    if args.dataset_split is not None:
        cfg.dataset_split = args.dataset_split
    if args.text_column is not None:
        cfg.text_column = args.text_column
    if args.max_samples is not None:
        cfg.max_samples = int(args.max_samples)
    if args.max_seq_length is not None:
        cfg.max_seq_length = int(args.max_seq_length)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.max_eval_batches is not None:
        cfg.max_eval_batches = int(args.max_eval_batches)
    if args.warmup_batches is not None:
        cfg.warmup_batches = int(args.warmup_batches)
    if args.output_json is not None:
        cfg.output_json = args.output_json

    if args.spmm_impls is not None:
        cfg.spmm_impls = _parse_str_list(args.spmm_impls)
    if args.top_k_values is not None:
        cfg.top_k_values = _parse_int_list(args.top_k_values)
    if args.soft_mask_values is not None:
        cfg.soft_mask_values = _parse_bool_list(args.soft_mask_values)
    if args.sigma_values is not None:
        cfg.sigma_values = _parse_float_list(args.sigma_values)

    if args.no_ablate_inhibition:
        cfg.ablate_inhibition = False
    if args.no_ablate_refractory:
        cfg.ablate_refractory = False
    if args.no_frozen_baseline:
        cfg.include_frozen_baseline = False
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast post-hoc SCA ablation with no additional training.")
    p.add_argument("--config", type=str, default=None, help="Optional ablation YAML/JSON config.")
    p.add_argument("--train-config", type=str, default=None, help="Optional training config to import defaults from.")
    p.add_argument("--checkpoint-dir", type=str, default=None, help="Model dir, run dir, or objective_step_*.pt path.")
    p.add_argument("--model-name", type=str, default=None, help="Base model name for cold load when no checkpoint is given.")
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--dataset-config", type=str, default=None)
    p.add_argument("--dataset-split", type=str, default=None)
    p.add_argument("--text-column", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--warmup-batches", type=int, default=None)
    p.add_argument("--spmm-impls", type=str, default=None, help="Comma list, e.g. dense,cuda_spmm")
    p.add_argument("--top-k-values", type=str, default=None, help="Comma list, e.g. 2,3,4")
    p.add_argument("--soft-mask-values", type=str, default=None, help="Comma list, e.g. true,false")
    p.add_argument("--sigma-values", type=str, default=None, help="Comma list, e.g. 1.0,1.5")
    p.add_argument("--no-ablate-inhibition", action="store_true")
    p.add_argument("--no-ablate-refractory", action="store_true")
    p.add_argument("--no-frozen-baseline", action="store_true")
    p.add_argument("--output-json", type=str, default=None)
    return p


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _build_parser().parse_args()
    cfg = _load_effective_config(args)
    _set_seed(cfg.seed)

    resolved_ckpt = _resolve_checkpoint_dir(cfg.checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = _build_eval_dataloader(tokenizer, cfg)
    if resolved_ckpt is not None:
        LOGGER.info("Loading checkpoint model from: %s", resolved_ckpt)
        model = NeuroplasticLlama.from_pretrained(
            resolved_ckpt,
            neuroplasticity_enabled=True,
        )
    else:
        LOGGER.info("No checkpoint provided; loading base model from: %s", cfg.model_name)
        model = NeuroplasticLlama(model_name=cfg.model_name, neuroplasticity_enabled=True)
    model.eval()
    model.collect_bio_gate_telemetry = False

    settings = _build_settings(model, cfg)
    LOGGER.info("Prepared %d settings", len(settings))

    rows: List[Dict[str, Any]] = []
    for idx, setting in enumerate(settings, start=1):
        _apply_setting(model, setting)
        LOGGER.info("(%d/%d) %s", idx, len(settings), setting.name)
        metrics = _evaluate_setting(
            model=model,
            dataloader=dataloader,
            task_id=cfg.task_id,
            max_eval_batches=cfg.max_eval_batches,
            warmup_batches=cfg.warmup_batches,
        )
        row = {
            "name": setting.name,
            "neuroplasticity_enabled": bool(setting.neuroplasticity_enabled),
            "spmm_impl": str(setting.spmm_impl),
            "top_k": int(setting.top_k),
            "soft_mask": bool(setting.soft_mask),
            "inhibition_lambda": float(setting.inhibition_lambda),
            "refractory_steps": int(setting.refractory_steps),
            "sigma": float(setting.sigma),
            **metrics,
        }
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (-float(r["tokens_per_second"]), float(r["avg_lm_loss"])))
    print()
    print(_format_table(rows_sorted))
    print()
    best_speed = rows_sorted[0]
    best_quality = min(rows_sorted, key=lambda r: float(r["avg_lm_loss"]))
    LOGGER.info(
        "Best speed: %s | tok/s=%.1f loss=%.4f",
        best_speed["name"],
        best_speed["tokens_per_second"],
        best_speed["avg_lm_loss"],
    )
    LOGGER.info(
        "Best quality: %s | loss=%.4f tok/s=%.1f",
        best_quality["name"],
        best_quality["avg_lm_loss"],
        best_quality["tokens_per_second"],
    )

    if cfg.output_json:
        os.makedirs(os.path.dirname(cfg.output_json) or ".", exist_ok=True)
        payload = {
            "model_name": cfg.model_name,
            "checkpoint_dir": resolved_ckpt,
            "task_id": int(cfg.task_id),
            "seed": int(cfg.seed),
            "dataset": {
                "name": cfg.dataset_name,
                "config": cfg.dataset_config,
                "split": cfg.dataset_split,
                "text_column": cfg.text_column,
                "max_samples": int(cfg.max_samples),
                "max_seq_length": int(cfg.max_seq_length),
                "batch_size": int(cfg.batch_size),
                "max_eval_batches": int(cfg.max_eval_batches),
                "warmup_batches": int(cfg.warmup_batches),
            },
            "rows_sorted": rows_sorted,
        }
        with open(cfg.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        LOGGER.info("Wrote results JSON: %s", cfg.output_json)


if __name__ == "__main__":
    run()
