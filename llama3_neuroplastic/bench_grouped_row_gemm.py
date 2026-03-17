#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from .neuroplastic_llama import NeuroplasticLlama
    from .objective_losses import compute_lm_loss
except ImportError:  # Script-mode fallback
    from neuroplastic_llama import NeuroplasticLlama
    from objective_losses import compute_lm_loss


LOGGER = logging.getLogger("bench_grouped_row_gemm")


@dataclass
class BenchConfig:
    run_name: str
    checkpoint_dir: Optional[str]
    model_name: str
    dataset_name: str
    dataset_config: str
    dataset_split: str
    text_column: str
    max_samples: int
    max_seq_length: int
    batch_size: int
    num_workers: int
    max_eval_batches: int
    warmup_batches: int
    top_k: int
    block_size: int
    grouped_min_bucket: int
    seed: int
    task_id: int
    output_json: Optional[str]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


def _build_eval_dataloader(tokenizer: Any, cfg: BenchConfig) -> DataLoader:
    if load_dataset is None:
        raise RuntimeError("datasets package is required to run grouped-row benchmark")

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


def _resolve_checkpoint_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    marker = os.path.join(path, "neuroplastic_llama_sca_v2.bin")
    if os.path.isfile(marker):
        return path
    candidates: List[tuple[int, str]] = []
    for name in os.listdir(path):
        if not name.startswith("model_step_"):
            continue
        full = os.path.join(path, name)
        if not os.path.isdir(full):
            continue
        child_marker = os.path.join(full, "neuroplastic_llama_sca_v2.bin")
        if not os.path.isfile(child_marker):
            continue
        suffix = name.replace("model_step_", "").strip()
        try:
            step = int(suffix)
        except ValueError:
            continue
        candidates.append((step, full))
    if not candidates:
        raise FileNotFoundError(f"No model_step_* checkpoint directory found under: {path}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _apply_mode(model: NeuroplasticLlama, mode: str, cfg: BenchConfig) -> None:
    model.neuroplasticity_enabled = True
    model.sca_config.top_k = int(max(1, min(cfg.top_k, model.sca_config.num_blocks)))
    model.sca_config.grouped_row_min_bucket = int(max(1, cfg.grouped_min_bucket))

    if mode == "baseline_cuda_spmm":
        model.sca_config.grouped_row_gemm = False
        model.sca_config.grouped_row_allow_4bit_dequant = False
        model.sca_config.spmm_impl = "cuda_spmm"
    elif mode == "grouped_row_gemm_dense":
        model.sca_config.grouped_row_gemm = True
        model.sca_config.grouped_row_allow_4bit_dequant = False
        model.sca_config.spmm_impl = "cuda_spmm"
    elif mode == "grouped_row_gemm_4bit_dequant":
        model.sca_config.grouped_row_gemm = True
        model.sca_config.grouped_row_allow_4bit_dequant = True
        model.sca_config.spmm_impl = "cuda_spmm"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if hasattr(model, "refractory_until") and torch.is_tensor(model.refractory_until):
        model.refractory_until.zero_()


def _evaluate_mode(
    model: NeuroplasticLlama,
    dataloader: DataLoader,
    cfg: BenchConfig,
    mode: str,
) -> Dict[str, Any]:
    _apply_mode(model, mode, cfg)
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
            if steps >= cfg.max_eval_batches:
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
                    task_id=cfg.task_id,
                )
                loss = compute_lm_loss(outputs.logits.float(), labels)
            if is_cuda:
                torch.cuda.synchronize(device=device)
            elapsed = time.perf_counter() - t0

            loss_weighted_sum += float(loss.item()) * float(token_count)
            token_count_sum += token_count

            if steps >= cfg.warmup_batches:
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
        "mode": mode,
        "tokens_per_second": tok_per_sec,
        "avg_step_ms": avg_step_ms,
        "avg_lm_loss": avg_loss,
        "perplexity": ppl,
        "peak_vram_gb": peak_vram_gb,
        "timed_batches": int(timed_steps),
        "timed_tokens": int(timed_token_count),
        "timed_seconds": float(timed_sec),
        "evaluated_batches": int(steps),
        "evaluated_tokens": int(token_count_sum),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark grouped-row GEMM against cuda_spmm baseline.")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--model-name", type=str, default="unsloth/Meta-Llama-3.1-8B-bnb-4bit")
    p.add_argument("--dataset-name", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", type=str, default="validation")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-eval-batches", type=int, default=16)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--grouped-min-bucket", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--output-json", type=str, default=None)
    return p


def _to_cfg(args: argparse.Namespace) -> BenchConfig:
    run_name = args.run_name
    if run_name is None:
        run_name = f"grouped_row_gemm_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return BenchConfig(
        run_name=str(run_name),
        checkpoint_dir=args.checkpoint_dir,
        model_name=str(args.model_name),
        dataset_name=str(args.dataset_name),
        dataset_config=str(args.dataset_config),
        dataset_split=str(args.dataset_split),
        text_column=str(args.text_column),
        max_samples=int(args.max_samples),
        max_seq_length=int(args.max_seq_length),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_eval_batches=int(args.max_eval_batches),
        warmup_batches=int(args.warmup_batches),
        top_k=int(args.top_k),
        block_size=int(args.block_size),
        grouped_min_bucket=int(args.grouped_min_bucket),
        seed=int(args.seed),
        task_id=int(args.task_id),
        output_json=args.output_json,
    )


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _build_parser().parse_args()
    cfg = _to_cfg(args)
    _set_seed(cfg.seed)

    resolved_ckpt = _resolve_checkpoint_dir(cfg.checkpoint_dir)
    if resolved_ckpt is not None:
        LOGGER.info("Loading checkpoint model from: %s", resolved_ckpt)
        model = NeuroplasticLlama.from_pretrained(
            resolved_ckpt,
            neuroplasticity_enabled=True,
        )
    else:
        LOGGER.info("Loading base model from: %s", cfg.model_name)
        model = NeuroplasticLlama(
            model_name=cfg.model_name,
            neuroplasticity_enabled=True,
            sca_spmm_impl="cuda_spmm",
            sca_top_k=cfg.top_k,
            sca_block_size=cfg.block_size,
            sca_grouped_row_gemm=False,
        )

    if int(model.sca_config.block_size) != int(cfg.block_size):
        raise ValueError(
            f"Requested block_size={cfg.block_size} but loaded model uses block_size={model.sca_config.block_size}. "
            "Changing block size post-load is unsupported because sparse wrappers are initialized with fixed block geometry."
        )

    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = _build_eval_dataloader(tokenizer, cfg)
    model.eval()
    model.collect_bio_gate_telemetry = False

    modes = [
        "baseline_cuda_spmm",
        "grouped_row_gemm_dense",
        "grouped_row_gemm_4bit_dequant",
    ]
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        LOGGER.info("Running mode: %s", mode)
        try:
            row = _evaluate_mode(model=model, dataloader=dataloader, cfg=cfg, mode=mode)
        except Exception as exc:
            LOGGER.exception("Mode failed: %s", mode)
            row = {
                "mode": mode,
                "tokens_per_second": 0.0,
                "avg_step_ms": 0.0,
                "avg_lm_loss": float("nan"),
                "perplexity": float("nan"),
                "peak_vram_gb": 0.0,
                "timed_batches": 0,
                "timed_tokens": 0,
                "timed_seconds": 0.0,
                "error": str(exc),
            }
        rows.append(row)

    payload = {
        "run_name": cfg.run_name,
        "checkpoint_dir": resolved_ckpt,
        "seed": int(cfg.seed),
        "shape": {
            "rows": int(cfg.batch_size * cfg.max_seq_length),
            "in_features": int(model.sca_config.hidden_size),
            "out_features": int(getattr(model.model.config, "intermediate_size", 0)),
            "top_k": int(cfg.top_k),
            "block_size": int(cfg.block_size),
        },
        "dataset": {
            "name": cfg.dataset_name,
            "config": cfg.dataset_config,
            "split": cfg.dataset_split,
            "max_samples": int(cfg.max_samples),
            "max_eval_batches": int(cfg.max_eval_batches),
            "warmup_batches": int(cfg.warmup_batches),
        },
        "results": rows,
    }

    print(json.dumps(payload, indent=2))
    if cfg.output_json:
        os.makedirs(os.path.dirname(cfg.output_json) or ".", exist_ok=True)
        with open(cfg.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        LOGGER.info("Wrote results JSON: %s", cfg.output_json)


if __name__ == "__main__":
    run()
